import argparse
import json
import os
import random
import re
from difflib import get_close_matches

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, IGNORE_INDEX
from model.conversation import conv_templates
from model.model.builder import load_pretrained_model
from model.mm_utils import get_model_name_from_path, tokenizer_graph_image_token
from model.utils import disable_torch_init


DEFAULT_MODEL_BASE = "models/vicuna-7b-v1.5"
DEFAULT_DATA_ROOT = "/home/yak2025244233/LLAVA/books-nc"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-config", type=str, default=None)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-sample-size", type=int, default=10)
    parser.add_argument("--graph-ids-path", type=str, default=None)
    parser.add_argument("--graph-edge-path", type=str, default=None)
    parser.add_argument("--graph-feat-path", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-candidates", action="store_true")
    parser.add_argument("--candidate-score-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--candidate-batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def resolve_path(data_root: str, explicit_path: str, filename: str) -> str:
    return explicit_path if explicit_path is not None else os.path.join(data_root, filename)


def load_label_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    valid_id2label = {int(k): str(v) for k, v in raw["valid_id2label"].items()}
    candidate_labels = [str(x) for x in raw.get("candidate_labels", [valid_id2label[idx] for idx in sorted(valid_id2label)])]
    return {
        "name": str(raw.get("name", "books-nc")),
        "subject_name": str(raw.get("subject_name", "book")),
        "prompt_template": str(
            raw.get(
                "prompt_template",
                "Classify the center {subject_name} into one of the following categories:\n{label_space}\nAnswer with the category name only.",
            )
        ),
        "candidate_labels": candidate_labels,
        "valid_id2label": valid_id2label,
    }


def build_prompt(label_spec):
    labels_text = "; ".join(label_spec["candidate_labels"])
    return (
        f"{DEFAULT_GRAPH_TOKEN}\n"
        + label_spec["prompt_template"].format(
            subject_name=label_spec["subject_name"],
            label_space=labels_text,
        )
    )


def build_conv_prompt(conv_mode: str, user_prompt: str, answer=None) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


def left_pad_input_ids_and_labels(input_ids_list, labels_list, pad_token_id: int):
    max_len = max(ids.shape[0] for ids in input_ids_list)
    batch_size = len(input_ids_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for idx, (ids, cur_labels) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = ids.shape[0]
        input_ids[idx, -seq_len:] = ids
        labels[idx, -seq_len:] = cur_labels
        attention_mask[idx, -seq_len:] = True
    return input_ids, labels, attention_mask


def label_token_ids_for_prompt(tokenizer, conv_mode: str, user_prompt: str, prompt_ids, label: str):
    full_prompt = build_conv_prompt(conv_mode, user_prompt, answer=label)
    full_ids = tokenizer_graph_image_token(full_prompt, tokenizer)
    prompt_ids = list(prompt_ids)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        label_ids = full_ids[len(prompt_ids) :]
    else:
        label_ids = tokenizer(f" {label}", add_special_tokens=False).input_ids

    if label_ids and label_ids[-1] == tokenizer.eos_token_id:
        label_ids = label_ids[:-1]
    if not label_ids:
        raise ValueError(f"Could not build label token ids for {label!r}")
    return [int(token_id) for token_id in label_ids]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def build_label_parser(label_spec):
    label2id = {normalize_text(v): k for k, v in label_spec["valid_id2label"].items()}
    candidate_labels = list(label_spec["candidate_labels"])

    def parse_prediction(output_text: str):
        normalized = normalize_text(output_text)
        if normalized in label2id:
            return label2id[normalized], normalized

        matches = [label for label in candidate_labels if normalize_text(label) in normalized]
        if len(matches) == 1:
            return label2id[normalize_text(matches[0])], matches[0]

        if len(matches) > 1:
            matches = sorted(matches, key=len, reverse=True)
            return label2id[normalize_text(matches[0])], matches[0]

        close = get_close_matches(normalized, list(label2id.keys()), n=1, cutoff=0.5)
        if close:
            return label2id[close[0]], close[0]

        return -1, output_text.strip()

    return parse_prediction


def infer_conv_mode(model_name, override=None):
    if override is not None:
        return override
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    if "mistral" in model_name.lower():
        return "mistral_instruct"
    if "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    if "v1" in model_name.lower():
        return "llava_v1"
    if "mpt" in model_name.lower():
        return "mpt"
    return "llava_v0"


def infer_model_name(model_path: str):
    model_name = get_model_name_from_path(model_path)
    config_path = os.path.join(model_path, "config.json")
    if "llava" not in model_name.lower() and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if str(config.get("model_type", "")).startswith("llava"):
            model_name = f"llava_{model_name}"
    return model_name


def load_node_mapping(path: str):
    node_mapping = torch.load(path, map_location="cpu", weights_only=False)
    return {int(v): str(k) for k, v in node_mapping.items()}


def load_edge_list(edge_path: str, num_nodes: int):
    edge_pairs = torch.load(edge_path, map_location="cpu", weights_only=False)
    if isinstance(edge_pairs, torch.Tensor):
        edge_pairs = edge_pairs.tolist()
    edge_list = [[] for _ in range(num_nodes)]
    for src, dst in edge_pairs:
        src = int(src)
        dst = int(dst)
        edge_list[src].append(dst)
        edge_list[dst].append(src)
    return edge_list


def bfs_serialize(edge_list, node_idx: int, k_hop: int, sample_size: int, seed: int, pad_val: int = DEFAULT_GRAPH_PAD_ID):
    rng = random.Random(seed + int(node_idx))
    neighbors = [[int(node_idx)]]
    for _ in range(k_hop):
        current_hop = []
        for parent in neighbors[-1]:
            if parent == pad_val:
                current_hop.extend([pad_val] * sample_size)
                continue
            node_neighbors = edge_list[parent]
            if len(node_neighbors) > sample_size:
                sampled_neighbors = rng.sample(node_neighbors, sample_size)
            else:
                sampled_neighbors = node_neighbors + [pad_val] * (sample_size - len(node_neighbors))
            current_hop.extend(sampled_neighbors)
        neighbors.append(current_hop)
    return [node for hop in neighbors for node in hop]


class GraphProvider:
    def __init__(self, args, num_nodes: int):
        self.graph_ids_path = resolve_path(args.data_root, args.graph_ids_path, f"graph_ids_h{args.graph_hops}_s{args.graph_sample_size}.pt")
        self.edge_path = resolve_path(args.data_root, args.graph_edge_path, "nc_edges-nodeid.pt")
        self.graph_ids = None
        self.edge_list = None
        self.args = args
        if os.path.exists(self.graph_ids_path):
            self.graph_ids = torch.load(self.graph_ids_path, map_location="cpu", weights_only=False).to(torch.long)
        else:
            if not os.path.exists(self.edge_path):
                raise FileNotFoundError(f"Neither graph ids cache nor edge file found: {self.graph_ids_path} / {self.edge_path}")
            self.edge_list = load_edge_list(self.edge_path, num_nodes)

    def get_batch(self, node_ids):
        if self.graph_ids is not None:
            return self.graph_ids[node_ids].to(torch.long)
        graphs = [
            bfs_serialize(
                self.edge_list,
                node_idx=node_id,
                k_hop=self.args.graph_hops,
                sample_size=self.args.graph_sample_size,
                seed=self.args.seed,
            )
            for node_id in node_ids
        ]
        return torch.tensor(graphs, dtype=torch.long)


def score_candidate_batch(
    model,
    input_ids,
    attention_mask,
    labels,
    graphs,
    graph_emb,
    graph_node_mask,
    reduction: str = "mean",
):
    (
        _,
        position_ids,
        mm_attention_mask,
        _,
        inputs_embeds,
        mm_labels,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        attention_mask,
        None,
        labels,
        None,
        graphs=graphs,
        graph_emb=graph_emb,
        graph_node_mask=graph_node_mask,
        return_graph_meta=True,
    )
    outputs = model.get_model()(
        input_ids=None,
        attention_mask=mm_attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    shift_logits = model.lm_head(outputs.last_hidden_state[:, :-1])
    shift_labels = mm_labels[:, 1:]
    label_mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~label_mask, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_scores = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_scores = token_scores.masked_fill(~label_mask, 0.0)
    scores = token_scores.sum(dim=1)
    if reduction == "mean":
        scores = scores / label_mask.sum(dim=1).clamp_min(1)
    return scores


def score_label_candidates(
    model,
    tokenizer,
    conv_mode,
    user_prompt,
    prompt_ids,
    pad_token_id,
    candidate_labels,
    graphs,
    graph_emb,
    graph_node_mask,
    reduction: str,
    candidate_batch_size: int = 0,
):
    model_device = graphs.device
    flat_inputs = []
    flat_labels = []
    for _ in range(graphs.shape[0]):
        for label in candidate_labels:
            label_ids = label_token_ids_for_prompt(tokenizer, conv_mode, user_prompt, prompt_ids.tolist(), label)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            full_ids = torch.cat([prompt_ids.cpu(), label_tensor], dim=0)
            full_labels = torch.full_like(full_ids, IGNORE_INDEX)
            full_labels[-label_tensor.shape[0] :] = label_tensor
            flat_inputs.append(full_ids)
            flat_labels.append(full_labels)

    chunk_size = candidate_batch_size if candidate_batch_size > 0 else len(flat_inputs)
    flat_scores = []
    num_labels = len(candidate_labels)
    for start in range(0, len(flat_inputs), chunk_size):
        end = min(start + chunk_size, len(flat_inputs))
        candidate_input_ids, candidate_label_ids, candidate_attention_mask = left_pad_input_ids_and_labels(
            flat_inputs[start:end],
            flat_labels[start:end],
            pad_token_id,
        )
        sample_indices = torch.arange(start, end, device=model_device, dtype=torch.long) // num_labels
        scores = score_candidate_batch(
            model=model,
            input_ids=candidate_input_ids.to(model_device),
            attention_mask=candidate_attention_mask.to(model_device),
            labels=candidate_label_ids.to(model_device),
            graphs=graphs.index_select(0, sample_indices),
            graph_emb=graph_emb.index_select(0, sample_indices),
            graph_node_mask=graph_node_mask.index_select(0, sample_indices),
            reduction=reduction,
        )
        flat_scores.append(scores)
    return torch.cat(flat_scores, dim=0).view(graphs.shape[0], num_labels)


def main(args):
    disable_torch_init()
    if args.model_base is not None and str(args.model_base).lower() in {"", "none", "null"}:
        args.model_base = None

    label_config = resolve_path(args.data_root, args.label_config, "graph_label_config.json")
    label_spec = load_label_config(label_config)
    parse_prediction = build_label_parser(label_spec)

    model_name = infer_model_name(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    graph_owner = model.get_model() if hasattr(model, "get_model") else model
    if hasattr(graph_owner, "graph_soft_token_encoder"):
        model.config.graph_use_soft_tokens = True
    model_device = next(model.parameters()).device
    conv_mode = infer_conv_mode(model_name, args.conv_mode)

    labels = torch.load(resolve_path(args.data_root, None, "labels.pt"), map_location="cpu", weights_only=False).to(torch.long)
    test_mask = torch.load(resolve_path(args.data_root, None, "test_mask.pt"), map_location="cpu", weights_only=False).bool()
    node_mapping = load_node_mapping(resolve_path(args.data_root, None, "node_mapping.pt"))
    clip_feat_path = resolve_path(args.data_root, args.graph_feat_path, "clip_feat.pt")
    clip_features = torch.load(clip_feat_path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(clip_features, dict):
        clip_features = clip_features["features"]
    clip_features = clip_features.to(torch.float32)

    graph_provider = GraphProvider(args, num_nodes=clip_features.shape[0])
    test_indices = torch.where(test_mask)[0].tolist()
    if args.max_samples > 0:
        test_indices = test_indices[:args.max_samples]

    prompt = build_prompt(label_spec)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    prompt_ids = tokenizer_graph_image_token(prompt_text, tokenizer, return_tensors="pt")
    prompt_input_ids = prompt_ids.unsqueeze(0).to(model_device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    y_true = []
    y_pred = []
    valid_label_ids = sorted(label_spec["valid_id2label"].keys())

    with open(args.answers_file, "w", encoding="utf-8") as ans_file:
        for start in range(0, len(test_indices), args.batch_size):
            node_batch = test_indices[start:start + args.batch_size]
            gt_labels = labels[node_batch].tolist()

            graph = graph_provider.get_batch(node_batch)
            graph_node_mask = graph.ne(DEFAULT_GRAPH_PAD_ID)
            graph_emb = torch.zeros(
                (graph.shape[0], graph.shape[1], clip_features.shape[1]),
                dtype=torch.float32,
            )
            graph_emb[graph_node_mask] = clip_features[graph[graph_node_mask]]

            graph = graph.to(model_device)
            graph_node_mask = graph_node_mask.to(model_device)
            graph_emb = graph_emb.to(model_device, dtype=model.dtype)
            input_ids = prompt_input_ids.expand(len(node_batch), -1)

            if args.score_candidates:
                with torch.inference_mode():
                    candidate_scores = score_label_candidates(
                        model=model,
                        tokenizer=tokenizer,
                        conv_mode=conv_mode,
                        user_prompt=prompt,
                        prompt_ids=prompt_ids,
                        pad_token_id=pad_token_id,
                        candidate_labels=label_spec["candidate_labels"],
                        graphs=graph,
                        graph_emb=graph_emb,
                        graph_node_mask=graph_node_mask,
                        reduction=args.candidate_score_reduction,
                        candidate_batch_size=args.candidate_batch_size,
                    )
                best_label_indices = candidate_scores.argmax(dim=1).tolist()
                outputs = [label_spec["candidate_labels"][idx] for idx in best_label_indices]
            else:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=None,
                        image_sizes=None,
                        graphs=graph,
                        graph_emb=graph_emb,
                        graph_node_mask=graph_node_mask,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 1e-6) if args.temperature > 0 else 1.0,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for node_id, gt_label_id, output_text in zip(node_batch, gt_labels, outputs):
                if args.score_candidates:
                    pred_label_text = output_text
                    pred_label_id = {v: k for k, v in label_spec["valid_id2label"].items()}[pred_label_text]
                else:
                    pred_label_id, pred_label_text = parse_prediction(output_text.strip())
                gt_label_text = label_spec["valid_id2label"][int(gt_label_id)]
                y_true.append(int(gt_label_id))
                y_pred.append(int(pred_label_id))
                ans_file.write(
                    json.dumps(
                        {
                            "node_id": int(node_id),
                            "asin": node_mapping.get(int(node_id), str(node_id)),
                            "gt_label_id": int(gt_label_id),
                            "gt_label_text": gt_label_text,
                            "pred_text": output_text.strip(),
                            "pred_label_id": int(pred_label_id),
                            "pred_label_text": pred_label_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            ans_file.flush()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=valid_label_ids,
        average="macro",
        zero_division=0,
    )

    print("=" * 60)
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
