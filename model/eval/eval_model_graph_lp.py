import argparse
import os
from typing import Tuple

import numpy as np
import torch

from model.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN
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

    # When CUDA_VISIBLE_DEVICES is set, users often pass the original physical GPU id
    # (e.g. cuda:7) even though the current process only sees a remapped cuda:0.
    return "cuda:0"


def load_graph_caches(data_root: str, graph_hops: int, graph_sample_size: int) -> Tuple[torch.Tensor, np.memmap]:
    graph_ids_path = os.path.join(data_root, f"graph_ids_h{graph_hops}_s{graph_sample_size}.pt")
    graph_feat_path = os.path.join(data_root, f"graph_feat_h{graph_hops}_s{graph_sample_size}_fp16.npy")
    graph_ids = torch.load(graph_ids_path, map_location="cpu", weights_only=False).to(torch.long)
    graph_feats = np.load(graph_feat_path, mmap_mode="r")
    return graph_ids, graph_feats


def maybe_build_prompt_input_ids(tokenizer, device):
    prompt = (
        f"{DEFAULT_GRAPH_TOKEN}\n"
        "Please briefly describe the center fashion product by using the BFS-serialized neighborhood graph."
    )
    input_ids = tokenizer_graph_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    return input_ids


@torch.no_grad()
def extract_graph_embeddings(model, tokenizer, graph_ids, graph_feats, batch_size: int, device: torch.device):
    model.eval()
    prompt_input_ids = maybe_build_prompt_input_ids(tokenizer, device)
    all_embeds = []

    for start in range(0, graph_ids.shape[0], batch_size):
        end = min(start + batch_size, graph_ids.shape[0])
        batch_graph_ids = graph_ids[start:end].to(device)
        batch_graph_emb = torch.from_numpy(np.array(graph_feats[start:end], copy=True)).to(device=device, dtype=model.dtype)
        batch_graph_mask = batch_graph_ids.ne(DEFAULT_GRAPH_PAD_ID)
        batch_input_ids = prompt_input_ids.expand(end - start, -1)

        (
            _,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            graph_meta,
        ) = model.prepare_inputs_labels_for_multimodal(
            batch_input_ids,
            None,
            None,
            None,
            None,
            None,
            image_sizes=None,
            graphs=batch_graph_ids,
            graph_emb=batch_graph_emb,
            graph_node_mask=batch_graph_mask,
            return_graph_meta=True,
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

        hidden_states = outputs.last_hidden_state
        batch_embeds = []
        for sample_hidden, sample_meta in zip(hidden_states, graph_meta):
            if sample_meta is None or len(sample_meta) == 0:
                batch_embeds.append(torch.zeros(hidden_states.shape[-1], device=hidden_states.device, dtype=torch.float32))
                continue
            # Single-node embedding extraction uses one <graph> span, so take the first span.
            first_meta = sample_meta[0]
            start_idx = first_meta["start"]
            length = first_meta["length"]
            graph_hidden = sample_hidden[start_idx:start_idx + length]
            batch_embeds.append(graph_hidden.float().mean(dim=0))
        all_embeds.append(torch.stack(batch_embeds, dim=0).cpu())

    return torch.cat(all_embeds, dim=0)


def load_lp_split(data_root: str):
    split = torch.load(os.path.join(data_root, "lp-edge-split.pt"), map_location="cpu", weights_only=False)
    return split["valid"], split["test"]


def slice_split(split_dict: dict, max_samples: int = -1):
    sliced = {}
    for key, value in split_dict.items():
        if max_samples > 0:
            sliced[key] = value[:max_samples]
        else:
            sliced[key] = value
    return sliced


def collect_eval_nodes(valid_split: dict, test_split: dict) -> torch.Tensor:
    valid_src, valid_tgt, valid_neg = (
        valid_split["source_node"],
        valid_split["target_node"],
        valid_split["target_node_neg"],
    )
    test_src, test_tgt, test_neg = (
        test_split["source_node"],
        test_split["target_node"],
        test_split["target_node_neg"],
    )
    nodes = torch.cat(
        [
            valid_src,
            valid_tgt,
            valid_neg.reshape(-1),
            test_src,
            test_tgt,
            test_neg.reshape(-1),
        ],
        dim=0,
    )
    return torch.unique(nodes)


def build_index_mapping(node_ids: torch.Tensor):
    return {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}


def score_edges(embeddings: torch.Tensor, source_node: torch.Tensor, target_node: torch.Tensor):
    src = embeddings[source_node]
    dst = embeddings[target_node]
    return (src * dst).sum(dim=-1)


def evaluate_split(embeddings: torch.Tensor, split_dict: dict, max_samples: int = -1):
    source_node = split_dict["source_node"]
    target_node = split_dict["target_node"]
    target_node_neg = split_dict["target_node_neg"]

    if max_samples > 0:
        source_node = source_node[:max_samples]
        target_node = target_node[:max_samples]
        target_node_neg = target_node_neg[:max_samples]

    pos_scores = score_edges(embeddings, source_node, target_node)

    num_samples, num_neg = target_node_neg.shape
    src_expand = source_node.unsqueeze(1).expand(-1, num_neg).reshape(-1)
    neg_scores = score_edges(
        embeddings,
        src_expand,
        target_node_neg.reshape(-1),
    ).reshape(num_samples, num_neg)

    # rank among one positive + many negatives
    ranks = 1 + (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1)
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()

    # pairwise AUC-style estimate
    auc = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()

    return {
        "AUC": auc,
        "MRR": mrr,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10,
        "samples": int(num_samples),
    }


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

    graph_ids, graph_feats = load_graph_caches(args.data_root, args.graph_hops, args.graph_sample_size)
    valid_split, test_split = load_lp_split(args.data_root)
    valid_split = slice_split(valid_split, args.max_samples)
    test_split = slice_split(test_split, args.max_samples)

    eval_node_ids = collect_eval_nodes(valid_split, test_split)
    sub_graph_ids = graph_ids[eval_node_ids]
    sub_graph_feats = graph_feats[eval_node_ids.numpy()]

    embeddings = extract_graph_embeddings(
        model=model,
        tokenizer=tokenizer,
        graph_ids=sub_graph_ids,
        graph_feats=sub_graph_feats,
        batch_size=args.batch_size,
        device=model_device,
    )
    embeddings = embeddings.to(torch.float32)
    node_id_to_local = build_index_mapping(eval_node_ids)

    def remap_split(split_dict):
        remapped = {}
        for key, value in split_dict.items():
            if value.dim() == 1:
                remapped[key] = torch.tensor(
                    [node_id_to_local[int(v)] for v in value.tolist()],
                    dtype=torch.long,
                )
            else:
                flat = value.reshape(-1).tolist()
                remapped_flat = [node_id_to_local[int(v)] for v in flat]
                remapped[key] = torch.tensor(remapped_flat, dtype=torch.long).reshape(value.shape)
        return remapped

    valid_split = remap_split(valid_split)
    test_split = remap_split(test_split)

    valid_metrics = evaluate_split(embeddings, valid_split, max_samples=-1)
    test_metrics = evaluate_split(embeddings, test_split, max_samples=-1)

    print("=" * 60)
    print("Validation")
    for k, v in valid_metrics.items():
        if k == "samples":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
    print("-" * 60)
    print("Test")
    for k, v in test_metrics.items():
        if k == "samples":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
