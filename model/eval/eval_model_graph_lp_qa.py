import argparse
import json
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN
from model.conversation import conv_templates
from model.model.builder import load_pretrained_model
from model.mm_utils import get_model_name_from_path, tokenizer_graph_image_token
from model.utils import disable_torch_init


DEFAULT_MODEL_BASE = "models/vicuna-7b-v1.5"
DEFAULT_DATA_ROOT = "/home/yak2025244233/LLAVA/amazon-cloth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-sample-size", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-negatives", type=int, default=-1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--scores-file", type=str, default=None)
    parser.add_argument("--print-every", type=int, default=2000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def resolve_device_arg(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        return "cpu"

    visible_count = torch.cuda.device_count()
    if device == "cuda":
        return "cuda:0"

    try:
        requested_idx = int(device.split(":", 1)[1])
    except (IndexError, ValueError):
        return "cuda:0"

    if requested_idx < visible_count:
        return device

    # CUDA_VISIBLE_DEVICES may remap a physical GPU id such as cuda:7 to cuda:0.
    return "cuda:0"


def infer_conv_mode(model_name: str, override: Optional[str] = None) -> str:
    if override is not None:
        return override
    lowered = model_name.lower()
    if "llama-2" in lowered:
        return "llava_llama_2"
    if "mistral" in lowered:
        return "mistral_instruct"
    if "v1.6-34b" in lowered:
        return "chatml_direct"
    if "v1" in lowered:
        return "llava_v1"
    if "mpt" in lowered:
        return "mpt"
    return "llava_v0"


def load_graph_caches(data_root: str, graph_hops: int, graph_sample_size: int) -> Tuple[torch.Tensor, np.memmap]:
    graph_ids_path = os.path.join(data_root, f"graph_ids_h{graph_hops}_s{graph_sample_size}.pt")
    graph_feat_path = os.path.join(data_root, f"graph_feat_h{graph_hops}_s{graph_sample_size}_fp16.npy")
    graph_ids = torch.load(graph_ids_path, map_location="cpu", weights_only=False).to(torch.long)
    graph_feats = np.load(graph_feat_path, mmap_mode="r")
    return graph_ids, graph_feats


def load_lp_split(data_root: str):
    split = torch.load(os.path.join(data_root, "lp-edge-split.pt"), map_location="cpu", weights_only=False)
    return split["valid"], split["test"]


def slice_split(split_dict: dict, max_samples: int = -1, max_negatives: int = -1):
    sliced = {}
    for key, value in split_dict.items():
        cur = value
        if max_samples > 0:
            cur = cur[:max_samples]
        if key == "target_node_neg" and max_negatives > 0:
            cur = cur[:, :max_negatives]
        sliced[key] = cur
    return sliced


def build_lp_question() -> str:
    return (
        f"{DEFAULT_GRAPH_TOKEN}\n"
        f"{DEFAULT_GRAPH_TOKEN}\n"
        "Given two node-centered subgraphs, predict whether the two center nodes should be connected. "
        "Answer with yes or no only."
    )


def build_conv_prompt(conv_mode: str, answer: Optional[str] = None) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], build_lp_question())
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


def strip_special_prefix(token_ids: Sequence[int], tokenizer) -> List[int]:
    ids = list(token_ids)
    while ids and ids[0] in {tokenizer.bos_token_id, tokenizer.pad_token_id}:
        ids = ids[1:]
    return ids


def continuation_token_ids(tokenizer, conv_mode: str, answer: str) -> List[int]:
    """Return the answer continuation tokens after the assistant prompt."""
    prompt = build_conv_prompt(conv_mode, answer=None)
    full_prompt = build_conv_prompt(conv_mode, answer=answer)
    prompt_ids = tokenizer_graph_image_token(prompt, tokenizer)
    full_ids = tokenizer_graph_image_token(full_prompt, tokenizer)

    if full_ids[: len(prompt_ids)] == prompt_ids:
        continuation = strip_special_prefix(full_ids[len(prompt_ids) :], tokenizer)
        if continuation:
            return [int(token_id) for token_id in continuation]

    # Fallback for tokenizers whose special-token handling prevents direct prefix matching.
    ids = strip_special_prefix(tokenizer(f" {answer}", add_special_tokens=False).input_ids, tokenizer)
    return [int(token_id) for token_id in ids]


def format_token_ids(tokenizer, token_ids: Iterable[int]) -> str:
    pieces = []
    for token_id in token_ids:
        token = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        pieces.append(f"{int(token_id)}:{token!r}")
    return ", ".join(pieces)


@torch.no_grad()
def score_candidate_batch(
    model,
    candidate_input_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    batch_graph_ids: torch.Tensor,
    batch_graph_emb: torch.Tensor,
    batch_graph_mask: torch.Tensor,
    batch_size_actual: int,
    device: torch.device,
) -> torch.Tensor:
    batch_input_ids = candidate_input_ids.expand(batch_size_actual, -1).contiguous()
    text_attention_mask = torch.ones_like(batch_input_ids, dtype=torch.bool, device=device)
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        batch_input_ids,
        None,
        text_attention_mask,
        None,
        None,
        None,
        image_sizes=None,
        graphs=batch_graph_ids,
        graph_emb=batch_graph_emb,
        graph_node_mask=batch_graph_mask,
    )
    outputs = model.get_model()(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    token_positions = torch.arange(attention_mask.shape[1], device=device).unsqueeze(0)
    last_token_idx = token_positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    candidate_len = candidate_token_ids.numel()
    first_candidate_idx = last_token_idx - candidate_len + 1
    pred_idx = first_candidate_idx.unsqueeze(1) - 1 + torch.arange(candidate_len, device=device).unsqueeze(0)
    if (pred_idx < 0).any():
        raise ValueError("Candidate answer starts before a valid prediction position.")

    row_idx = torch.arange(batch_size_actual, device=device).unsqueeze(1)
    selected_hidden = outputs.last_hidden_state[row_idx, pred_idx]
    selected_logits = model.lm_head(selected_hidden)
    selected_log_probs = F.log_softmax(selected_logits.float(), dim=-1)
    answer_log_probs = selected_log_probs.gather(
        dim=2,
        index=candidate_token_ids.view(1, -1, 1).expand(batch_size_actual, -1, 1),
    ).squeeze(-1)
    return answer_log_probs.sum(dim=1)


@torch.no_grad()
def score_pairs(
    model,
    yes_input_ids: torch.Tensor,
    no_input_ids: torch.Tensor,
    yes_token_ids: torch.Tensor,
    no_token_ids: torch.Tensor,
    graph_ids: torch.Tensor,
    graph_feats: np.memmap,
    source_node: torch.Tensor,
    target_node: torch.Tensor,
    batch_size: int,
    device: torch.device,
    print_every: int,
    label: str,
) -> torch.Tensor:
    model.eval()
    scores = []
    total = int(source_node.numel())

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_src = source_node[start:end].to(torch.long)
        batch_tgt = target_node[start:end].to(torch.long)
        batch_size_actual = end - start

        batch_graph_ids = torch.stack(
            [graph_ids[batch_src], graph_ids[batch_tgt]],
            dim=1,
        ).to(device)
        batch_graph_emb = torch.from_numpy(
            np.stack(
                [
                    np.array(graph_feats[batch_src.numpy()], copy=True),
                    np.array(graph_feats[batch_tgt.numpy()], copy=True),
                ],
                axis=1,
            )
        ).to(device=device, dtype=model.dtype)
        batch_graph_mask = batch_graph_ids.ne(DEFAULT_GRAPH_PAD_ID)

        yes_log_prob = score_candidate_batch(
            model=model,
            candidate_input_ids=yes_input_ids,
            candidate_token_ids=yes_token_ids,
            batch_graph_ids=batch_graph_ids,
            batch_graph_emb=batch_graph_emb,
            batch_graph_mask=batch_graph_mask,
            batch_size_actual=batch_size_actual,
            device=device,
        )
        no_log_prob = score_candidate_batch(
            model=model,
            candidate_input_ids=no_input_ids,
            candidate_token_ids=no_token_ids,
            batch_graph_ids=batch_graph_ids,
            batch_graph_emb=batch_graph_emb,
            batch_graph_mask=batch_graph_mask,
            batch_size_actual=batch_size_actual,
            device=device,
        )
        yes_prob = torch.exp(yes_log_prob - torch.logaddexp(yes_log_prob, no_log_prob))
        scores.append(yes_prob.detach().cpu())

        if print_every > 0 and (end == total or end % print_every == 0):
            print(f"{label}: scored {end}/{total} pairs", flush=True)

    return torch.cat(scores, dim=0)


def compute_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    num_samples, _ = neg_scores.shape

    ranks = 1 + (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1)
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    auc = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()

    return {
        "AUC": auc,
        "MRR": mrr,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10,
        "samples": int(num_samples),
        "negatives_per_sample": int(neg_scores.shape[1]),
    }


def evaluate_split(
    model,
    yes_input_ids: torch.Tensor,
    no_input_ids: torch.Tensor,
    yes_token_ids: torch.Tensor,
    no_token_ids: torch.Tensor,
    graph_ids: torch.Tensor,
    graph_feats: np.memmap,
    split_dict: dict,
    batch_size: int,
    device: torch.device,
    print_every: int,
    split_name: str,
):
    source_node = split_dict["source_node"].cpu()
    target_node = split_dict["target_node"].cpu()
    target_node_neg = split_dict["target_node_neg"].cpu()

    pos_scores = score_pairs(
        model=model,
        yes_input_ids=yes_input_ids,
        no_input_ids=no_input_ids,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        graph_ids=graph_ids,
        graph_feats=graph_feats,
        source_node=source_node,
        target_node=target_node,
        batch_size=batch_size,
        device=device,
        print_every=print_every,
        label=f"{split_name}/pos",
    )

    num_samples, num_neg = target_node_neg.shape
    src_expand = source_node.unsqueeze(1).expand(-1, num_neg).reshape(-1)
    neg_scores = score_pairs(
        model=model,
        yes_input_ids=yes_input_ids,
        no_input_ids=no_input_ids,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        graph_ids=graph_ids,
        graph_feats=graph_feats,
        source_node=src_expand,
        target_node=target_node_neg.reshape(-1),
        batch_size=batch_size,
        device=device,
        print_every=print_every,
        label=f"{split_name}/neg",
    ).reshape(num_samples, num_neg)

    return compute_metrics(pos_scores, neg_scores), pos_scores, neg_scores


def write_scores(path: str, split_name: str, split_dict: dict, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode) as f:
        source_node = split_dict["source_node"].cpu()
        target_node = split_dict["target_node"].cpu()
        target_node_neg = split_dict["target_node_neg"].cpu()
        for idx in range(source_node.numel()):
            f.write(
                json.dumps(
                    {
                        "split": split_name,
                        "source_node": int(source_node[idx].item()),
                        "target_node": int(target_node[idx].item()),
                        "pos_score": float(pos_scores[idx].item()),
                        "target_node_neg": [int(x) for x in target_node_neg[idx].tolist()],
                        "neg_scores": [float(x) for x in neg_scores[idx].tolist()],
                    }
                )
                + "\n"
            )


def print_metrics(title: str, metrics: dict):
    print(title)
    for key, value in metrics.items():
        if key in {"samples", "negatives_per_sample"}:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")


def main():
    args = parse_args()
    disable_torch_init()
    args.device = resolve_device_arg(args.device)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model_device = next(model.parameters()).device
    conv_mode = infer_conv_mode(model_name, args.conv_mode)

    graph_ids, graph_feats = load_graph_caches(args.data_root, args.graph_hops, args.graph_sample_size)
    valid_split, test_split = load_lp_split(args.data_root)
    valid_split = slice_split(valid_split, args.max_samples, args.max_negatives)
    test_split = slice_split(test_split, args.max_samples, args.max_negatives)

    prompt = build_conv_prompt(conv_mode, answer=None)
    prompt_ids = tokenizer_graph_image_token(prompt, tokenizer)
    yes_ids = continuation_token_ids(tokenizer, conv_mode, "yes")
    no_ids = continuation_token_ids(tokenizer, conv_mode, "no")
    if not yes_ids or not no_ids:
        raise ValueError(f"Invalid yes/no token ids: yes={yes_ids}, no={no_ids}")
    yes_input_ids = torch.tensor(prompt_ids + yes_ids, dtype=torch.long, device=model_device).unsqueeze(0)
    no_input_ids = torch.tensor(prompt_ids + no_ids, dtype=torch.long, device=model_device).unsqueeze(0)
    yes_token_ids = torch.tensor(yes_ids, dtype=torch.long, device=model_device)
    no_token_ids = torch.tensor(no_ids, dtype=torch.long, device=model_device)

    print(f"conv_mode: {conv_mode}")
    print(f"yes continuation ids: {format_token_ids(tokenizer, yes_ids)}")
    print(f"no continuation ids: {format_token_ids(tokenizer, no_ids)}")

    if args.scores_file is not None and os.path.exists(args.scores_file):
        os.remove(args.scores_file)

    valid_metrics, valid_pos_scores, valid_neg_scores = evaluate_split(
        model=model,
        yes_input_ids=yes_input_ids,
        no_input_ids=no_input_ids,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        graph_ids=graph_ids,
        graph_feats=graph_feats,
        split_dict=valid_split,
        batch_size=args.batch_size,
        device=model_device,
        print_every=args.print_every,
        split_name="valid",
    )
    test_metrics, test_pos_scores, test_neg_scores = evaluate_split(
        model=model,
        yes_input_ids=yes_input_ids,
        no_input_ids=no_input_ids,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        graph_ids=graph_ids,
        graph_feats=graph_feats,
        split_dict=test_split,
        batch_size=args.batch_size,
        device=model_device,
        print_every=args.print_every,
        split_name="test",
    )

    if args.scores_file is not None:
        write_scores(args.scores_file, "valid", valid_split, valid_pos_scores, valid_neg_scores)
        write_scores(args.scores_file, "test", test_split, test_pos_scores, test_neg_scores)

    print("=" * 60)
    print_metrics("Validation", valid_metrics)
    print("-" * 60)
    print_metrics("Test", test_metrics)
    print("=" * 60)


if __name__ == "__main__":
    main()
