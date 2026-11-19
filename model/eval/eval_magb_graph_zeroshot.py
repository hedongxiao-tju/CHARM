import argparse
import json
import os
import random
from dataclasses import replace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model import conversation as conversation_lib
from model.conversation import conv_templates
from model.eval.eval_model_graph_textcls import (
    build_graph_modality_masks,
    infer_conv_mode,
    infer_model_name,
    left_pad_input_ids_and_labels,
)
from model.model.builder import load_pretrained_model
from model.train.train import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    LazyGraphSupervisedDataset,
    _format_compact_label_space,
    _format_label_answer,
    _format_label_space,
    _format_label_space_promptv2,
    _format_option_label_space,
    _ordered_option_labels,
    _option_letter,
)
from model.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data/MAGB/Toys")
    parser.add_argument(
        "--synthetic-root",
        type=str,
        default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic_cliptext_bridge",
    )
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-mode", choices=["candidate_likelihood", "option_logits", "generate"], default="candidate_likelihood",
                        help="generate=generate a category name and match it to candidates (MLaGA-style evaluation)")
    parser.add_argument("--generate-max-new-tokens", type=int, default=16)
    parser.add_argument("--generate-batch-size", type=int, default=1)
    parser.add_argument(
        "--option-permutations",
        type=int,
        default=1,
        help="For option_logits, average LLM option scores over this many candidate label orderings.",
    )
    parser.add_argument(
        "--option-order-mode",
        choices=["fixed", "sample"],
        default="fixed",
        help="Use fixed/global option orders or the same per-sample shuffled option order used by training.",
    )
    parser.add_argument("--option-shuffle-seed", type=int, default=0)
    parser.add_argument("--graph-label-option-seed", type=int, default=0)
    parser.add_argument("--candidate-batch-size", type=int, default=0)
    parser.add_argument("--candidate-score-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--answer-with-description", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph-label-prompt-style", choices=["default", "promptv2"], default="default")
    parser.add_argument("--graph-context-prompt-style", choices=["full", "compact", "rich"], default="full")
    parser.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the identical serialized textual context but remove the <graph> placeholder and all continuous graph tokens.",
    )
    parser.add_argument("--calibrate-label-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-prior-weight", type=float, default=1.0)
    parser.add_argument("--graph-sample-size", type=int, default=8)
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-include-abstract-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-include-synthetic-soft-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-real-node-serializer", action=argparse.BooleanOptionalAction, default=False,
                        help="Use the real-node serializer (must match training).")
    parser.add_argument("--graph-real-neighbor-topk", type=int, default=8)
    parser.add_argument("--graph-real-cross-topk", type=int, default=4)
    parser.add_argument("--graph-include-cross-domain-soft-tokens", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--graph-ablate-hop", type=int, default=-1,
                        help="Ablate tokens of this hop type (-1=none, 2=layer1/abstract, 3=layer2/calibration, 4=global/transfer)")
    parser.add_argument("--graph-ablate-layer", type=int, default=-1,
                        help="Training-time layer ablation: 1=remove layer1, 2=remove layer2, 4=remove transfer")
    parser.add_argument("--graph-semantic-coord-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph-semantic-coord-topk", type=int, default=3)
    parser.add_argument("--graph-semantic-coord-graph-topk", type=int, default=4)
    parser.add_argument("--graph-semantic-coord-neighbor-topk", type=int, default=0)
    parser.add_argument("--graph-semantic-coord-transfer-topk", type=int, default=3)
    parser.add_argument("--graph-semantic-coord-use-global-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-semantic-coord-global-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-semantic-coord-hybrid-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph-semantic-coord-evidence-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-semantic-coord-center-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph-semantic-coord-use-abstract-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-semantic-coord-use-calibration-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-semantic-coord-use-transfer-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-max-consistent-neighbors", type=int, default=4)
    parser.add_argument("--graph-max-complement-neighbors", type=int, default=4)
    parser.add_argument("--graph-max-conflict-neighbors", type=int, default=2)
    parser.add_argument("--graph-min-abstract-similarity", type=float, default=-1.0)
    parser.add_argument("--graph-min-bridge-confidence", type=float, default=-1.0)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument(
        "--eval-sample-mode",
        choices=["first", "random", "stratified"],
        default="first",
        help="How to choose the max-samples subset after applying valid/test masks.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which MAGB split mask to evaluate. Use all to ignore train/val/test masks.",
    )
    parser.add_argument("--eval-sample-seed", type=int, default=0)
    parser.add_argument(
        "--graph-modality-mask-strategy",
        choices=["none", "center_one", "center_text", "center_image", "center_both"],
        default="none",
    )
    parser.add_argument("--graph-modality-mask-prob", type=float, default=1.0)
    parser.add_argument("--graph-disable-role-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-semantic-auxiliary-scale", type=float, default=None)
    parser.add_argument("--graph-semantic-neighbor-scale", type=float, default=None)
    parser.add_argument(
        "--center-score-blend-weight",
        type=float,
        default=0.0,
        help="When >0, score the same sample with center-only semantic tokens and blend scores; value is the enhanced-score weight.",
    )
    parser.add_argument(
        "--center-score-fusion-mode",
        choices=["blend", "margin_switch"],
        default="blend",
        help="How to combine center-only and enhanced semantic-token scores when center-score fusion is enabled.",
    )
    parser.add_argument(
        "--center-score-switch-center-margin",
        type=float,
        default=0.25,
        help="For margin_switch, only trust enhanced scores when the center-only top-1/top-2 margin is at most this value.",
    )
    parser.add_argument(
        "--center-score-switch-enhanced-margin",
        type=float,
        default=3.5,
        help="For margin_switch, only trust enhanced scores when the enhanced top-1/top-2 margin is at least this value.",
    )
    parser.add_argument(
        "--center-score-blend-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Row-normalize center/enhanced score vectors before center-score blending.",
    )
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def build_eval_data_args(args) -> DataArguments:
    return DataArguments(
        graph_data_root=args.data_root,
        graph_feat_path="clip_feat.pt",
        graph_label_config="graph_label_config.json",
        graph_synthetic_root=args.synthetic_root,
        graph_label_supervision=True,
        graph_only=True,
        graph_zero_shot_serializer=True,
        graph_include_label_descriptions=True,
        graph_answer_with_description=args.answer_with_description,
        graph_label_prompt_style=args.graph_label_prompt_style,
        graph_context_prompt_style=args.graph_context_prompt_style,
        graph_option_label_prompt=args.score_mode == "option_logits",
        graph_shuffle_label_options=args.option_order_mode == "sample",
        graph_label_option_seed=args.graph_label_option_seed,
        graph_hops=args.graph_hops,
        graph_sample_size=args.graph_sample_size,
        graph_include_abstract_nodes=args.graph_include_abstract_nodes,
        graph_include_synthetic_soft_tokens=args.graph_include_synthetic_soft_tokens,
        graph_real_node_serializer=args.graph_real_node_serializer,
        graph_real_neighbor_topk=args.graph_real_neighbor_topk,
        graph_real_cross_topk=args.graph_real_cross_topk,
        graph_include_cross_domain_soft_tokens=args.graph_include_cross_domain_soft_tokens,
        graph_semantic_coord_tokens=args.graph_semantic_coord_tokens,
        graph_semantic_coord_topk=args.graph_semantic_coord_topk,
        graph_semantic_coord_graph_topk=args.graph_semantic_coord_graph_topk,
        graph_semantic_coord_neighbor_topk=args.graph_semantic_coord_neighbor_topk,
        graph_semantic_coord_transfer_topk=args.graph_semantic_coord_transfer_topk,
        graph_semantic_coord_use_global_nodes=args.graph_semantic_coord_use_global_nodes,
        graph_semantic_coord_global_alignment=args.graph_semantic_coord_global_alignment,
        graph_semantic_coord_hybrid_prompt=args.graph_semantic_coord_hybrid_prompt,
        graph_semantic_coord_evidence_prompt=args.graph_semantic_coord_evidence_prompt,
        graph_semantic_coord_center_only=args.graph_semantic_coord_center_only,
        graph_semantic_coord_use_abstract_token=args.graph_semantic_coord_use_abstract_token,
        graph_semantic_coord_use_calibration_token=args.graph_semantic_coord_use_calibration_token,
        graph_semantic_coord_use_transfer_token=args.graph_semantic_coord_use_transfer_token,
        graph_max_consistent_neighbors=args.graph_max_consistent_neighbors,
        graph_max_complement_neighbors=args.graph_max_complement_neighbors,
        graph_max_conflict_neighbors=args.graph_max_conflict_neighbors,
        graph_min_abstract_similarity=args.graph_min_abstract_similarity,
        graph_min_bridge_confidence=args.graph_min_bridge_confidence,
        graph_cache_path="",
        graph_feat_cache_path="",
        max_train_samples=-1,
    )


def choose_eval_subset(dataset: LazyGraphSupervisedDataset, sample_indices, max_samples: int, mode: str, seed: int):
    if max_samples <= 0 or len(sample_indices) <= max_samples:
        return sample_indices
    if mode == "first":
        return sample_indices[:max_samples]

    rng = random.Random(seed)
    if mode == "random":
        selected = list(sample_indices)
        rng.shuffle(selected)
        return selected[:max_samples]

    if mode != "stratified":
        raise ValueError(f"Unknown eval sample mode: {mode}")

    label_ids = sorted(dataset.graph_label_spec["valid_id2label"].keys())
    buckets = {label_id: [] for label_id in label_ids}
    for idx in sample_indices:
        label_id = int(dataset.labels[idx].item())
        if label_id in buckets:
            buckets[label_id].append(idx)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected = []
    selected_set = set()
    # Round-robin gives each available label a chance before filling extra slots.
    while len(selected) < max_samples:
        progressed = False
        for label_id in label_ids:
            bucket = buckets[label_id]
            if bucket:
                idx = bucket.pop()
                selected.append(idx)
                selected_set.add(idx)
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
    if len(selected) < max_samples:
        remaining = [idx for idx in sample_indices if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    rng.shuffle(selected)
    return selected


def set_eval_indices(dataset: LazyGraphSupervisedDataset, max_samples: int, sample_mode: str, sample_seed: int, split: str = "test"):
    if dataset.labels is None:
        raise ValueError("labels.pt is required for MAGB zero-shot evaluation.")
    valid_label_ids = set(dataset.graph_label_spec["valid_id2label"].keys())
    valid_mask = torch.zeros(len(dataset.records), dtype=torch.bool)
    for label_id in valid_label_ids:
        valid_mask |= dataset.labels.eq(label_id)

    if split != "all":
        split_mask_path = os.path.join(dataset.data_args.graph_data_root, f"{split}_mask.pt")
        if not os.path.exists(split_mask_path):
            raise FileNotFoundError(f"Missing {split} split mask: {split_mask_path}")
        valid_mask &= torch.load(split_mask_path, map_location="cpu", weights_only=False).bool()

    sample_indices = torch.where(valid_mask)[0].tolist()
    sample_indices = choose_eval_subset(dataset, sample_indices, max_samples, sample_mode, sample_seed)
    dataset.sample_indices = sample_indices


def split_prompt_from_training_sample(sample):
    labels = sample["labels"]
    label_positions = torch.where(labels.ne(IGNORE_INDEX))[0]
    if label_positions.numel() == 0:
        raise ValueError("Sample has no supervised answer tokens.")
    prompt_end = int(label_positions[0].item())
    prompt_ids = sample["input_ids"][:prompt_end].clone()
    return prompt_ids


def remove_graph_placeholder(prompt_ids):
    """Preserve the serialized evidence while removing only continuous graph input."""
    return prompt_ids[prompt_ids.ne(GRAPH_TOKEN_INDEX)]


def build_candidate_sequence(prompt_ids, answer_ids, max_length):
    full_ids = torch.cat([prompt_ids.cpu(), answer_ids.cpu()], dim=0)
    full_labels = torch.full_like(full_ids, IGNORE_INDEX)
    full_labels[-answer_ids.numel() :] = answer_ids.cpu()
    if max_length > 0 and full_ids.numel() > max_length:
        special_positions = torch.where((prompt_ids.cpu() == GRAPH_TOKEN_INDEX) | (prompt_ids.cpu() == IMAGE_TOKEN_INDEX))[0]
        special_keep = int(special_positions.max().item()) + 1 if special_positions.numel() > 0 else 1
        answer_len = int(answer_ids.numel())
        if answer_len >= max_length:
            raise ValueError(
                f"Candidate sequence is too long to keep graph token and answer: "
                f"prompt={prompt_ids.numel()}, answer={answer_ids.numel()}, max_length={max_length}"
            )
        prefix_end = min(special_keep, int(prompt_ids.numel()))
        prompt_tail_budget = max_length - prefix_end - answer_len
        if prompt_tail_budget <= 0:
            keep_start = max(prefix_end, int(prompt_ids.numel()) - (max_length - prefix_end - answer_len))
            kept_prompt = torch.cat([prompt_ids[:prefix_end].cpu(), prompt_ids[keep_start:].cpu()], dim=0)
        else:
            tail_start = max(prefix_end, int(prompt_ids.numel()) - prompt_tail_budget)
            kept_prompt = torch.cat([prompt_ids[:prefix_end].cpu(), prompt_ids[tail_start:].cpu()], dim=0)
        if kept_prompt.numel() + answer_ids.numel() > max_length:
            keep_tail = max(0, max_length - answer_len - prefix_end)
            if keep_tail > 0:
                kept_prompt = torch.cat([kept_prompt[:prefix_end], kept_prompt[-keep_tail:]], dim=0)
            else:
                kept_prompt = kept_prompt[: max_length - answer_len]
        full_ids = torch.cat([kept_prompt, answer_ids.cpu()], dim=0)
        full_labels = torch.full_like(full_ids, IGNORE_INDEX)
        full_labels[-answer_ids.numel() :] = answer_ids.cpu()
    return full_ids, full_labels


def collate_candidate_batch(samples, tokenizer, model_device, model_dtype):
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        graph_modality_mask_strategy="none",
        graph_modality_mask_prob=0.0,
    )
    batch = collator(samples)
    graph_keys = [
        "graphs",
        "graph_emb",
        "graph_node_mask",
        "graph_hops",
        "graph_ppr_scores",
        "graph_role_ids",
        "graph_role_mask",
    ]
    out = {}
    for key in graph_keys:
        value = batch.get(key, None)
        if value is None:
            continue
        if key == "graph_emb":
            value = value.to(model_device, dtype=model_dtype)
        else:
            value = value.to(model_device)
        out[key] = value
    # Pass images for rich prompt style
    if "images" in batch:
        images = batch["images"]
        if isinstance(images, torch.Tensor):
            out["images"] = images.to(model_device, dtype=model_dtype)
        else:
            out["images"] = [img.to(model_device, dtype=model_dtype) for img in images]
    return out


@torch.no_grad()
@torch.inference_mode()
def score_candidate_batch(
    model,
    input_ids,
    attention_mask,
    labels,
    graph_batch,
    graph_text_mask,
    graph_image_mask,
    reduction,
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
        graph_batch.get("images", None),
        labels,
        None,
        graphs=graph_batch["graphs"],
        graph_emb=graph_batch["graph_emb"],
        graph_node_mask=graph_batch["graph_node_mask"],
        graph_hops=graph_batch["graph_hops"],
        graph_ppr_scores=graph_batch.get("graph_ppr_scores", None),
        graph_role_ids=graph_batch["graph_role_ids"],
        graph_role_mask=graph_batch["graph_role_mask"],
        graph_text_mask=graph_text_mask,
        graph_image_mask=graph_image_mask,
        return_graph_meta=True,
    )
    inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
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
    shift_hidden = torch.nan_to_num(outputs.last_hidden_state[:, :-1], nan=0.0, posinf=0.0, neginf=0.0).clamp(-16.0, 16.0)
    shift_labels = mm_labels[:, 1:]
    label_mask = shift_labels.ne(IGNORE_INDEX)
    scores = torch.zeros(shift_labels.shape[0], dtype=torch.float32, device=shift_hidden.device)
    if label_mask.any():
        target_hidden = shift_hidden[label_mask]
        target_labels = shift_labels[label_mask]
        target_logits = model.lm_head(target_hidden)
        target_logits = torch.nan_to_num(target_logits, nan=-1e4, posinf=1e4, neginf=-1e4).clamp(-100.0, 100.0)
        target_log_probs = F.log_softmax(target_logits.float(), dim=-1)
        target_scores = target_log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
        row_ids = torch.arange(shift_labels.shape[0], device=shift_labels.device).unsqueeze(1).expand_as(shift_labels)
        scores.index_add_(0, row_ids[label_mask], target_scores)
    if reduction == "mean":
        scores = scores / label_mask.sum(dim=1).clamp_min(1)
    return scores


@torch.inference_mode()
def score_text_candidate_batch(
    model,
    input_ids,
    attention_mask,
    labels,
    reduction,
):
    outputs = model.get_model()(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    shift_hidden = torch.nan_to_num(outputs.last_hidden_state[:, :-1], nan=0.0, posinf=0.0, neginf=0.0).clamp(-16.0, 16.0)
    shift_labels = labels[:, 1:]
    label_mask = shift_labels.ne(IGNORE_INDEX)
    scores = torch.zeros(shift_labels.shape[0], dtype=torch.float32, device=shift_hidden.device)
    if label_mask.any():
        target_hidden = shift_hidden[label_mask]
        target_labels = shift_labels[label_mask]
        target_logits = model.lm_head(target_hidden)
        target_logits = torch.nan_to_num(target_logits, nan=-1e4, posinf=1e4, neginf=-1e4).clamp(-100.0, 100.0)
        target_log_probs = F.log_softmax(target_logits.float(), dim=-1)
        target_scores = target_log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
        row_ids = torch.arange(shift_labels.shape[0], device=shift_labels.device).unsqueeze(1).expand_as(shift_labels)
        scores.index_add_(0, row_ids[label_mask], target_scores)
    if reduction == "mean":
        scores = scores / label_mask.sum(dim=1).clamp_min(1)
    return scores


def build_neutral_prior_prompt_ids(tokenizer, label_spec, conv_mode, prompt_style="default", context_prompt_style="full"):
    if context_prompt_style == "compact":
        label_space = _format_compact_label_space(label_spec)
        prompt = (
            f"Classify a {label_spec['subject_name']}.\n"
            f"Candidates: {label_space}\n"
            "No item evidence is provided. Answer with exactly one candidate category name."
        )
    elif prompt_style == "promptv2":
        label_space = _format_label_space_promptv2(label_spec, include_descriptions=True)
        prompt = (
            f"Classify a {label_spec['subject_name']} into one of the following categories.\n"
            f"{label_space}\n"
            "No specific item evidence is provided. Return the best category and a concise semantic description."
        )
    else:
        label_space = _format_label_space(label_spec, include_descriptions=True)
        prompt = (
            f"Classify a {label_spec['subject_name']} into one of the following categories.\n"
            f"{label_space}\n"
            "No specific item evidence is provided. Return the best category and a concise semantic description."
        )
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return tokenizer(conv.get_prompt(), add_special_tokens=False, return_tensors="pt").input_ids[0]


def score_label_priors(model, tokenizer, label_spec, candidate_answers, conv_mode, args):
    model_device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_ids = build_neutral_prior_prompt_ids(
        tokenizer,
        label_spec,
        conv_mode,
        args.graph_label_prompt_style,
        args.graph_context_prompt_style,
    )
    flat_inputs = []
    flat_labels = []
    for answer in candidate_answers:
        answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if answer_ids.numel() == 0:
            raise ValueError(f"Empty candidate answer after tokenization: {answer!r}")
        full_ids, full_labels = build_candidate_sequence(prompt_ids, answer_ids, args.model_max_length)
        flat_inputs.append(full_ids)
        flat_labels.append(full_labels)

    candidate_batch_size = args.candidate_batch_size if args.candidate_batch_size > 0 else len(flat_inputs)
    flat_scores = []
    for start in range(0, len(flat_inputs), candidate_batch_size):
        end = min(start + candidate_batch_size, len(flat_inputs))
        input_ids, labels, attention_mask = left_pad_input_ids_and_labels(
            flat_inputs[start:end],
            flat_labels[start:end],
            pad_token_id,
        )
        scores = score_text_candidate_batch(
            model=model,
            input_ids=input_ids.to(model_device),
            attention_mask=attention_mask.to(model_device),
            labels=labels.to(model_device),
            reduction=args.candidate_score_reduction,
        )
        flat_scores.append(scores.cpu())
    return torch.cat(flat_scores, dim=0)


def score_labels_for_samples(model, tokenizer, samples, prompt_ids_list, candidate_answers, args):
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.text_only:
        flat_inputs = []
        flat_labels = []
        for prompt_ids in prompt_ids_list:
            prompt_ids = remove_graph_placeholder(prompt_ids)
            for answer in candidate_answers:
                answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
                full_ids, full_labels = build_candidate_sequence(prompt_ids, answer_ids, args.model_max_length)
                flat_inputs.append(full_ids)
                flat_labels.append(full_labels)
        candidate_batch_size = args.candidate_batch_size if args.candidate_batch_size > 0 else len(flat_inputs)
        flat_scores = []
        for start in range(0, len(flat_inputs), candidate_batch_size):
            end = min(start + candidate_batch_size, len(flat_inputs))
            input_ids, labels, attention_mask = left_pad_input_ids_and_labels(
                flat_inputs[start:end], flat_labels[start:end], pad_token_id
            )
            flat_scores.append(
                score_text_candidate_batch(
                    model,
                    input_ids.to(model_device),
                    attention_mask.to(model_device),
                    labels.to(model_device),
                    args.candidate_score_reduction,
                ).cpu()
            )
        return torch.cat(flat_scores, dim=0).view(len(samples), len(candidate_answers))
    graph_batch = collate_candidate_batch(samples, tokenizer, model_device, model_dtype)
    graph_text_mask, graph_image_mask = build_graph_modality_masks(
        graph_batch["graph_node_mask"],
        args.graph_modality_mask_strategy,
        args.graph_modality_mask_prob,
    )

    flat_inputs = []
    flat_labels = []
    for sample_idx, prompt_ids in enumerate(prompt_ids_list):
        graph_placeholder_count = int((prompt_ids == GRAPH_TOKEN_INDEX).sum().item())
        graph_expand_extra = max(
            0,
            int(graph_batch["graph_node_mask"][sample_idx].sum().item()) - graph_placeholder_count,
        )
        candidate_text_max_length = args.model_max_length - graph_expand_extra
        for answer in candidate_answers:
            answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if answer_ids.numel() == 0:
                raise ValueError(f"Empty candidate answer after tokenization: {answer!r}")
            candidate_max_length = max(int(answer_ids.numel()) + 1, candidate_text_max_length)
            full_ids, full_labels = build_candidate_sequence(prompt_ids, answer_ids, candidate_max_length)
            flat_inputs.append(full_ids)
            flat_labels.append(full_labels)

    num_labels = len(candidate_answers)
    candidate_batch_size = args.candidate_batch_size if args.candidate_batch_size > 0 else len(flat_inputs)
    flat_scores = []
    for start in range(0, len(flat_inputs), candidate_batch_size):
        end = min(start + candidate_batch_size, len(flat_inputs))
        input_ids, labels, attention_mask = left_pad_input_ids_and_labels(
            flat_inputs[start:end],
            flat_labels[start:end],
            pad_token_id,
        )
        sample_indices = torch.arange(start, end, device=model_device, dtype=torch.long) // num_labels
        sub_graph_batch = {key: value.index_select(0, sample_indices) for key, value in graph_batch.items()}
        scores = score_candidate_batch(
            model=model,
            input_ids=input_ids.to(model_device),
            attention_mask=attention_mask.to(model_device),
            labels=labels.to(model_device),
            graph_batch=sub_graph_batch,
            graph_text_mask=graph_text_mask.index_select(0, sample_indices),
            graph_image_mask=graph_image_mask.index_select(0, sample_indices),
            reduction=args.candidate_score_reduction,
        )
        flat_scores.append(scores.cpu())
    return torch.cat(flat_scores, dim=0).view(len(samples), num_labels)


def _normalize_label_text(s: str) -> str:
    """Normalize label text by lowercasing and removing punctuation and spaces."""
    import re as _re
    return _re.sub(r"[^a-z0-9]", "", s.lower())


def match_generated_to_label(generated: str, candidate_labels):
    """Match generated text to candidate categories using normalized exact or substring matching; return -1 on failure.
    """
    gen_norm = _normalize_label_text(generated)
    norm_labels = [_normalize_label_text(c) for c in candidate_labels]
    # 1) Exact match
    for i, nl in enumerate(norm_labels):
        if gen_norm == nl:
            return i
    # 2) Substring matches in either direction
    best, best_len = -1, 0
    for i, nl in enumerate(norm_labels):
        if nl and nl in gen_norm and len(nl) > best_len:
            best, best_len = i, len(nl)
    if best >= 0:
        return best
    for i, nl in enumerate(norm_labels):
        if gen_norm and gen_norm in nl and len(gen_norm) > best_len:
            best, best_len = i, len(gen_norm)
    return best


@torch.no_grad()
@torch.inference_mode()
def generate_predictions_for_samples(model, tokenizer, samples, prompt_ids_list,
                                     candidate_labels, args):
    """Generate category names and match them to candidates.
    Return predicted candidate indices and generated text for each sample.
    """
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    pred_idx, gen_texts = [], []
    bs = max(1, int(getattr(args, "generate_batch_size", 1)))
    for start in range(0, len(samples), bs):
        end = min(start + bs, len(samples))
        sub_samples = samples[start:end]
        sub_prompts = prompt_ids_list[start:end]
        graph_batch = collate_candidate_batch(sub_samples, tokenizer, model_device, model_dtype)
        graph_text_mask, graph_image_mask = build_graph_modality_masks(
            graph_batch["graph_node_mask"], args.graph_modality_mask_strategy, args.graph_modality_mask_prob,
        )
        # Left-pad prompts for generation
        maxlen = max(p.numel() for p in sub_prompts)
        input_ids = torch.full((len(sub_prompts), maxlen), pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(sub_prompts), maxlen), dtype=torch.long)
        for i, p in enumerate(sub_prompts):
            input_ids[i, maxlen - p.numel():] = p
            attn[i, maxlen - p.numel():] = 1
        out = model.generate(
            inputs=input_ids.to(model_device),
            attention_mask=attn.to(model_device),
            graph_emb=graph_batch["graph_emb"],
            graph_node_mask=graph_batch["graph_node_mask"],
            graph_hops=graph_batch.get("graph_hops"),
            graph_ppr_scores=graph_batch.get("graph_ppr_scores"),
            graph_role_ids=graph_batch.get("graph_role_ids"),
            graph_role_mask=graph_batch.get("graph_role_mask"),
            graph_text_mask=graph_text_mask,
            graph_image_mask=graph_image_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(getattr(args, "generate_max_new_tokens", 16)),
            use_cache=True,
        )
        for i in range(out.shape[0]):
            text = tokenizer.decode(out[i], skip_special_tokens=True).strip()
            gen_texts.append(text)
            pred_idx.append(match_generated_to_label(text, candidate_labels))
    return pred_idx, gen_texts


def build_option_token_id_sets(tokenizer, num_options: int):
    option_token_id_sets = []
    for idx in range(num_options):
        letter = _option_letter(idx)
        variants = [
            letter,
            f" {letter}",
            f"{letter}.",
            f" {letter}.",
            f"{letter}\n",
            f" {letter}\n",
        ]
        token_ids = []
        for variant in variants:
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if ids:
                token_ids.append(int(ids[0]))
        unique_ids = sorted(set(token_ids))
        if not unique_ids:
            raise ValueError(f"Could not tokenize option letter {letter!r}")
        option_token_id_sets.append(unique_ids)
    return option_token_id_sets


@torch.inference_mode()
def score_option_logits_for_samples(model, tokenizer, samples, prompt_ids_list, option_token_id_sets, args):
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.text_only:
        prompt_ids_list = [remove_graph_placeholder(prompt_ids) for prompt_ids in prompt_ids_list]
        dummy_labels = [torch.full_like(prompt_ids.cpu(), IGNORE_INDEX) for prompt_ids in prompt_ids_list]
        input_ids, _, attention_mask = left_pad_input_ids_and_labels(
            [prompt_ids.cpu() for prompt_ids in prompt_ids_list], dummy_labels, pad_token_id
        )
        outputs = model.get_model()(
            input_ids=input_ids.to(model_device),
            attention_mask=attention_mask.to(model_device),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = torch.nan_to_num(outputs.last_hidden_state, nan=0.0, posinf=0.0, neginf=0.0).clamp(-16.0, 16.0)
        last_positions = attention_mask.to(model_device).long().sum(dim=1).sub(1)
        next_logits = model.lm_head(hidden[torch.arange(hidden.shape[0], device=model_device), last_positions])
        log_probs = F.log_softmax(torch.nan_to_num(next_logits, nan=-1e4, posinf=1e4, neginf=-1e4).float(), dim=-1)
        option_scores = []
        for token_ids in option_token_id_sets:
            ids = torch.tensor(token_ids, dtype=torch.long, device=model_device)
            option_scores.append(torch.logsumexp(log_probs.index_select(dim=-1, index=ids), dim=-1))
        return torch.stack(option_scores, dim=1).cpu()
    graph_batch = collate_candidate_batch(samples, tokenizer, model_device, model_dtype)
    graph_text_mask, graph_image_mask = build_graph_modality_masks(
        graph_batch["graph_node_mask"],
        args.graph_modality_mask_strategy,
        args.graph_modality_mask_prob,
    )
    dummy_labels = [torch.full_like(prompt_ids.cpu(), IGNORE_INDEX) for prompt_ids in prompt_ids_list]
    input_ids, _, attention_mask = left_pad_input_ids_and_labels(
        [prompt_ids.cpu() for prompt_ids in prompt_ids_list],
        dummy_labels,
        pad_token_id,
    )
    (
        _,
        position_ids,
        mm_attention_mask,
        _,
        inputs_embeds,
        _,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids.to(model_device),
        None,
        attention_mask.to(model_device),
        graph_batch.get("images", None),
        None,
        None,
        graphs=graph_batch["graphs"],
        graph_emb=graph_batch["graph_emb"],
        graph_node_mask=graph_batch["graph_node_mask"],
        graph_hops=graph_batch["graph_hops"],
        graph_ppr_scores=graph_batch.get("graph_ppr_scores", None),
        graph_role_ids=graph_batch["graph_role_ids"],
        graph_role_mask=graph_batch["graph_role_mask"],
        graph_text_mask=graph_text_mask,
        graph_image_mask=graph_image_mask,
        return_graph_meta=True,
    )
    inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
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
    hidden = torch.nan_to_num(outputs.last_hidden_state, nan=0.0, posinf=0.0, neginf=0.0).clamp(-16.0, 16.0)
    if mm_attention_mask is None:
        last_positions = torch.full((hidden.shape[0],), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device)
    else:
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        last_positions = (positions * mm_attention_mask.long()).argmax(dim=1)
    next_logits = model.lm_head(hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions])
    next_logits = torch.nan_to_num(next_logits, nan=-1e4, posinf=1e4, neginf=-1e4).clamp(-100.0, 100.0)
    log_probs = F.log_softmax(next_logits.float(), dim=-1)
    option_scores = []
    for token_ids in option_token_id_sets:
        ids = torch.tensor(token_ids, dtype=torch.long, device=log_probs.device)
        option_scores.append(torch.logsumexp(log_probs.index_select(dim=-1, index=ids), dim=-1))
    return torch.stack(option_scores, dim=1).cpu()


@torch.inference_mode()
def score_option_priors(model, tokenizer, label_spec, option_token_id_sets, conv_mode, args, option_labels=None):
    model_device = next(model.parameters()).device
    subject_name = label_spec["subject_name"]
    if args.graph_context_prompt_style == "compact":
        label_space = _format_compact_label_space(label_spec, option_labels=option_labels, option_prompt=True)
    else:
        label_space = _format_option_label_space(
            label_spec,
            include_descriptions=True,
            option_labels=option_labels,
            prompt_style=args.graph_label_prompt_style,
        )
    prompt = (
        f"Classify a {subject_name} into one of the following options.\n"
        f"{label_space}\n"
        "No specific item evidence is provided. Answer with only the single best option letter."
    )
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_ids = tokenizer(conv.get_prompt(), add_special_tokens=False, return_tensors="pt").input_ids.to(model_device)
    attention_mask = torch.ones_like(prompt_ids, device=model_device)
    outputs = model.get_model()(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    log_probs = F.log_softmax(model.lm_head(outputs.last_hidden_state[:, -1]).float(), dim=-1)
    scores = []
    for token_ids in option_token_id_sets:
        ids = torch.tensor(token_ids, dtype=torch.long, device=log_probs.device)
        scores.append(torch.logsumexp(log_probs.index_select(dim=-1, index=ids), dim=-1).squeeze(0).cpu())
    return torch.stack(scores, dim=0)


def build_option_label_orders(candidate_labels, num_permutations: int, seed: int):
    if num_permutations <= 0:
        raise ValueError(f"option_permutations must be positive, got {num_permutations}")
    canonical = _ordered_option_labels({"candidate_labels": candidate_labels, "name": "eval-options"})
    orders = [canonical]
    rng = random.Random(seed)
    for _ in range(1, num_permutations):
        shuffled = list(canonical)
        rng.shuffle(shuffled)
        orders.append(shuffled)
    return orders


def remap_option_scores_to_label_order(option_scores: torch.Tensor, option_labels, candidate_labels):
    option_labels = list(option_labels)
    candidate_labels = list(candidate_labels)
    if set(option_labels) != set(candidate_labels) or len(option_labels) != len(candidate_labels):
        raise ValueError("option_labels must be a permutation of candidate_labels during option-logit eval")
    label_to_offset = {label: idx for idx, label in enumerate(candidate_labels)}
    if option_scores.dim() == 1:
        out = torch.empty(len(candidate_labels), dtype=option_scores.dtype, device=option_scores.device)
        for option_idx, label in enumerate(option_labels):
            out[label_to_offset[label]] = option_scores[option_idx]
        return out
    out = torch.empty(
        option_scores.shape[0],
        len(candidate_labels),
        dtype=option_scores.dtype,
        device=option_scores.device,
    )
    for option_idx, label in enumerate(option_labels):
        out[:, label_to_offset[label]] = option_scores[:, option_idx]
    return out


def remap_sample_option_scores_to_label_order(option_scores: torch.Tensor, option_label_orders, candidate_labels):
    rows = [
        remap_option_scores_to_label_order(row_scores, option_labels, candidate_labels)
        for row_scores, option_labels in zip(option_scores, option_label_orders)
    ]
    return torch.stack(rows, dim=0)


def set_dataset_option_order(dataset: LazyGraphSupervisedDataset, local_indices, option_labels):
    option_labels = list(option_labels)
    for local_idx in local_indices:
        dataset_idx = int(dataset.sample_indices[local_idx])
        dataset.option_label_orders[dataset_idx] = option_labels


def get_sample_option_order(dataset: LazyGraphSupervisedDataset, dataset_idx: int, candidate_labels):
    option_labels = dataset._option_labels_for_sample(int(dataset_idx))
    if option_labels is None:
        option_labels = _ordered_option_labels(dataset.graph_label_spec)
    option_labels = [str(label) for label in option_labels]
    if set(option_labels) != set(candidate_labels) or len(option_labels) != len(candidate_labels):
        raise ValueError("Per-sample option order must be a permutation of candidate_labels")
    return option_labels


def build_center_only_samples(dataset: LazyGraphSupervisedDataset, local_indices):
    attrs = [
        "graph_semantic_coord_center_only",
        "graph_semantic_coord_neighbor_topk",
        "graph_semantic_coord_transfer_topk",
        "graph_semantic_coord_use_abstract_token",
        "graph_semantic_coord_use_calibration_token",
        "graph_semantic_coord_use_transfer_token",
        "graph_semantic_coord_use_global_nodes",
        "graph_semantic_coord_global_alignment",
        "graph_semantic_coord_evidence_prompt",
    ]
    old_values = {name: getattr(dataset.data_args, name) for name in attrs}
    try:
        dataset.data_args.graph_semantic_coord_center_only = True
        dataset.data_args.graph_semantic_coord_neighbor_topk = 0
        dataset.data_args.graph_semantic_coord_transfer_topk = 0
        dataset.data_args.graph_semantic_coord_use_abstract_token = False
        dataset.data_args.graph_semantic_coord_use_calibration_token = False
        dataset.data_args.graph_semantic_coord_use_transfer_token = False
        dataset.data_args.graph_semantic_coord_use_global_nodes = False
        dataset.data_args.graph_semantic_coord_global_alignment = False
        dataset.data_args.graph_semantic_coord_evidence_prompt = False
        return [dataset[i] for i in local_indices]
    finally:
        for name, value in old_values.items():
            setattr(dataset.data_args, name, value)


def blend_center_scores(center_scores: torch.Tensor, enhanced_scores: torch.Tensor, weight: float, normalize: bool = False):
    weight = max(0.0, min(1.0, float(weight)))
    if normalize:
        def row_zscore(scores):
            mean = scores.mean(dim=1, keepdim=True)
            std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (scores - mean) / std
        center_scores = row_zscore(center_scores)
        enhanced_scores = row_zscore(enhanced_scores)
    return (1.0 - weight) * center_scores + weight * enhanced_scores


def score_margins(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[1] < 2:
        return torch.zeros(scores.shape[0], device=scores.device, dtype=scores.dtype)
    top2 = torch.topk(scores, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def fuse_center_scores(
    center_scores: torch.Tensor,
    enhanced_scores: torch.Tensor,
    *,
    mode: str,
    weight: float,
    normalize: bool = False,
    center_margin_threshold: float = 0.25,
    enhanced_margin_threshold: float = 3.5,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if mode == "blend":
        return blend_center_scores(center_scores, enhanced_scores, weight, normalize=normalize), None
    if normalize:
        def row_zscore(scores):
            mean = scores.mean(dim=1, keepdim=True)
            std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (scores - mean) / std
        center_scores = row_zscore(center_scores)
        enhanced_scores = row_zscore(enhanced_scores)
    center_margin = score_margins(center_scores)
    enhanced_margin = score_margins(enhanced_scores)
    switch_mask = (center_margin <= float(center_margin_threshold)) & (
        enhanced_margin >= float(enhanced_margin_threshold)
    )
    fused_scores = torch.where(switch_mask.unsqueeze(-1), enhanced_scores, center_scores)
    return fused_scores, switch_mask


def main():
    args = parse_args()
    disable_torch_init()
    if args.model_base is not None and str(args.model_base).lower() in {"", "none", "null"}:
        args.model_base = None
    adapter_markers = [
        "mm_projector.bin",
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "non_lora_trainables.bin",
    ]
    has_adapter_weights = any(os.path.exists(os.path.join(args.model_path, name)) for name in adapter_markers)
    if args.model_base is not None and not has_adapter_weights:
        print(
            f"[eval] No adapter weights found in {args.model_path}; loading it as a full checkpoint.",
            flush=True,
        )
        args.model_base = None

    model_name = infer_model_name(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    tokenizer.model_max_length = args.model_max_length
    model.eval()
    conv_mode = infer_conv_mode(model_name, args.conv_mode)
    if conv_mode not in conv_templates:
        raise ValueError(f"Unknown conversation mode: {conv_mode}")
    conversation_lib.default_conversation = conv_templates[conv_mode]
    if args.graph_disable_role_tokens is not None:
        model.config.graph_disable_role_tokens = args.graph_disable_role_tokens
    model.config.graph_ablate_hop = getattr(args, 'graph_ablate_hop', -1)
    model.config.graph_ablate_layer = getattr(args, 'graph_ablate_layer', -1)
    if args.graph_semantic_auxiliary_scale is not None:
        model.config.graph_semantic_auxiliary_scale = args.graph_semantic_auxiliary_scale
        encoder = getattr(model.get_model(), "graph_soft_token_encoder", None)
        if encoder is not None and hasattr(encoder, "semantic_auxiliary_scale"):
            encoder.semantic_auxiliary_scale.fill_(float(args.graph_semantic_auxiliary_scale))
    if args.graph_semantic_neighbor_scale is not None:
        model.config.graph_semantic_neighbor_scale = args.graph_semantic_neighbor_scale
        encoder = getattr(model.get_model(), "graph_soft_token_encoder", None)
        if encoder is not None and hasattr(encoder, "semantic_neighbor_scale"):
            encoder.semantic_neighbor_scale.fill_(float(args.graph_semantic_neighbor_scale))

    data_args = build_eval_data_args(args)
    data_args = replace(data_args)
    data_args.image_processor = image_processor
    data_args.mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    dataset = LazyGraphSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    set_eval_indices(dataset, args.max_samples, args.eval_sample_mode, args.eval_sample_seed, args.eval_split)

    label_spec = dataset.graph_label_spec
    valid_id2label = label_spec["valid_id2label"]
    label_to_valid_id = {label: label_id for label_id, label in valid_id2label.items()}
    candidate_labels = [label for label in label_spec["candidate_labels"] if label in label_to_valid_id]
    if not candidate_labels:
        valid_label_ids = sorted(valid_id2label)
        candidate_labels = [valid_id2label[label_id] for label_id in valid_label_ids]
    else:
        valid_label_ids = [label_to_valid_id[label] for label in candidate_labels]
    option_orders = None
    option_prior_scores_by_order = None
    if args.score_mode == "option_logits":
        if args.option_order_mode == "sample" and args.option_permutations != 1:
            raise ValueError("--option-order-mode sample only supports --option-permutations 1")
        option_orders = (
            None
            if args.option_order_mode == "sample"
            else build_option_label_orders(
                candidate_labels,
                args.option_permutations,
                args.option_shuffle_seed,
            )
        )
        candidate_answers = [_option_letter(idx) for idx in range(len(candidate_labels))]
        option_token_id_sets = build_option_token_id_sets(tokenizer, len(candidate_labels))
    else:
        candidate_answers = [
            _format_label_answer(label, label_spec, with_description=args.answer_with_description)
            for label in candidate_labels
        ]
        option_token_id_sets = None
    label_prior_scores = None
    if args.calibrate_label_prior and args.score_mode != "generate":
        if args.score_mode == "option_logits":
            option_prior_scores_by_order = []
            canonical_prior_scores = []
            for option_labels in option_orders:
                prior_option_scores = score_option_priors(
                    model,
                    tokenizer,
                    label_spec,
                    option_token_id_sets,
                    conv_mode,
                    args,
                    option_labels=option_labels,
                )
                option_prior_scores_by_order.append(prior_option_scores)
                canonical_prior_scores.append(
                    remap_option_scores_to_label_order(prior_option_scores, option_labels, candidate_labels)
                )
            label_prior_scores = torch.stack(canonical_prior_scores, dim=0).mean(dim=0)
        else:
            label_prior_scores = score_label_priors(model, tokenizer, label_spec, candidate_answers, conv_mode, args)

    print(
        json.dumps(
            {
                "stage": "eval_magb_graph_zeroshot",
                "model_path": args.model_path,
                "model_base": args.model_base,
                "data_root": args.data_root,
                "synthetic_root": args.synthetic_root,
                "num_samples": len(dataset),
                "candidate_labels": dict(zip(valid_label_ids, candidate_labels)),
                "score_mode": args.score_mode,
                "answer_with_description": args.answer_with_description,
                "graph_label_prompt_style": args.graph_label_prompt_style,
                "graph_context_prompt_style": args.graph_context_prompt_style,
                "text_only": args.text_only,
                "calibrate_label_prior": args.calibrate_label_prior,
                "label_prior_weight": args.label_prior_weight,
                "option_permutations": args.option_permutations if args.score_mode == "option_logits" else 0,
                "option_order_mode": args.option_order_mode if args.score_mode == "option_logits" else None,
                "option_shuffle_seed": args.option_shuffle_seed if args.score_mode == "option_logits" else None,
                "graph_label_option_seed": args.graph_label_option_seed if args.score_mode == "option_logits" else None,
                "center_score_blend_weight": args.center_score_blend_weight,
                "center_score_fusion_mode": args.center_score_fusion_mode,
                "center_score_switch_center_margin": args.center_score_switch_center_margin,
                "center_score_switch_enhanced_margin": args.center_score_switch_enhanced_margin,
                "center_score_blend_normalize": args.center_score_blend_normalize,
                "eval_sample_mode": args.eval_sample_mode,
                "eval_sample_seed": args.eval_sample_seed,
                "eval_split": args.eval_split,
                "conv_mode": conv_mode,
                "context_len": context_len,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    y_true = []
    y_pred = []
    blend_weight = max(0.0, min(1.0, float(args.center_score_blend_weight)))
    use_center_score_fusion = args.center_score_fusion_mode == "margin_switch" or blend_weight > 0.0
    with open(args.answers_file, "w", encoding="utf-8") as ans_file:
        for start in range(0, len(dataset), args.batch_size):
            end = min(start + args.batch_size, len(dataset))
            if start == 0 or end % 50 == 0 or end == len(dataset):
                print(f"[eval] {end}/{len(dataset)}", flush=True)
            center_scores = None
            center_switch_mask = None
            scores = None
            raw_scores = None
            generated_texts = None
            if args.score_mode == "generate":
                # MLaGA-style autoregressive evaluation
                samples = [dataset[i] for i in range(start, end)]
                prompt_ids_list = [split_prompt_from_training_sample(s) for s in samples]
                gen_pred_idx, generated_texts = generate_predictions_for_samples(
                    model, tokenizer, samples, prompt_ids_list, candidate_labels, args
                )
                # Fall back to class 0 when matching fails (-1)
                pred_offsets = [p if p >= 0 else 0 for p in gen_pred_idx]
            elif args.score_mode == "option_logits":
                local_indices = list(range(start, end))
                if args.option_order_mode == "sample":
                    samples = [dataset[i] for i in local_indices]
                    prompt_ids_list = [split_prompt_from_training_sample(sample) for sample in samples]
                    option_raw_scores = score_option_logits_for_samples(
                        model,
                        tokenizer,
                        samples,
                        prompt_ids_list,
                        option_token_id_sets,
                        args,
                    )
                    option_label_orders = [
                        get_sample_option_order(dataset, dataset.sample_indices[i], candidate_labels)
                        for i in local_indices
                    ]
                    raw_scores = remap_sample_option_scores_to_label_order(
                        option_raw_scores,
                        option_label_orders,
                        candidate_labels,
                    )
                    scores = raw_scores
                    if use_center_score_fusion:
                        center_samples = build_center_only_samples(dataset, local_indices)
                        center_prompt_ids_list = [split_prompt_from_training_sample(sample) for sample in center_samples]
                        center_option_raw_scores = score_option_logits_for_samples(
                            model,
                            tokenizer,
                            center_samples,
                            center_prompt_ids_list,
                            option_token_id_sets,
                            args,
                        )
                        center_scores = remap_sample_option_scores_to_label_order(
                            center_option_raw_scores,
                            option_label_orders,
                            candidate_labels,
                        )
                        raw_scores, center_switch_mask = fuse_center_scores(
                            center_scores,
                            raw_scores,
                            mode=args.center_score_fusion_mode,
                            weight=blend_weight,
                            normalize=args.center_score_blend_normalize,
                            center_margin_threshold=args.center_score_switch_center_margin,
                            enhanced_margin_threshold=args.center_score_switch_enhanced_margin,
                        )
                        scores = raw_scores
                else:
                    raw_score_sum = None
                    score_sum = None
                    center_score_sum = None
                    for order_idx, option_labels in enumerate(option_orders):
                        set_dataset_option_order(dataset, local_indices, option_labels)
                        samples = [dataset[i] for i in local_indices]
                        prompt_ids_list = [split_prompt_from_training_sample(sample) for sample in samples]
                        option_raw_scores = score_option_logits_for_samples(
                            model,
                            tokenizer,
                            samples,
                            prompt_ids_list,
                            option_token_id_sets,
                            args,
                        )
                        canonical_raw_scores = remap_option_scores_to_label_order(
                            option_raw_scores,
                            option_labels,
                            candidate_labels,
                        )
                        if option_prior_scores_by_order is not None:
                            option_prior_scores = option_prior_scores_by_order[order_idx]
                            canonical_prior_scores = remap_option_scores_to_label_order(
                                option_prior_scores,
                                option_labels,
                                candidate_labels,
                            )
                            canonical_scores = canonical_raw_scores - args.label_prior_weight * canonical_prior_scores.unsqueeze(0)
                        else:
                            canonical_scores = canonical_raw_scores
                        raw_score_sum = canonical_raw_scores if raw_score_sum is None else raw_score_sum + canonical_raw_scores
                        score_sum = canonical_scores if score_sum is None else score_sum + canonical_scores
                        if use_center_score_fusion:
                            center_samples = build_center_only_samples(dataset, local_indices)
                            center_prompt_ids_list = [split_prompt_from_training_sample(sample) for sample in center_samples]
                            center_option_raw_scores = score_option_logits_for_samples(
                                model,
                                tokenizer,
                                center_samples,
                                center_prompt_ids_list,
                                option_token_id_sets,
                                args,
                            )
                            canonical_center_scores = remap_option_scores_to_label_order(
                                center_option_raw_scores,
                                option_labels,
                                candidate_labels,
                            )
                            center_score_sum = (
                                canonical_center_scores
                                if center_score_sum is None
                                else center_score_sum + canonical_center_scores
                            )
                    raw_scores = raw_score_sum / float(len(option_orders))
                    scores = score_sum / float(len(option_orders))
                    if use_center_score_fusion:
                        center_scores = center_score_sum / float(len(option_orders))
                        if label_prior_scores is not None:
                            center_scores_for_blend = center_scores - args.label_prior_weight * label_prior_scores.unsqueeze(0)
                        else:
                            center_scores_for_blend = center_scores
                        scores, center_switch_mask = fuse_center_scores(
                            center_scores_for_blend,
                            scores,
                            mode=args.center_score_fusion_mode,
                            weight=blend_weight,
                            normalize=args.center_score_blend_normalize,
                            center_margin_threshold=args.center_score_switch_center_margin,
                            enhanced_margin_threshold=args.center_score_switch_enhanced_margin,
                        )
                        raw_scores, _ = fuse_center_scores(
                            center_scores,
                            raw_scores,
                            mode=args.center_score_fusion_mode,
                            weight=blend_weight,
                            normalize=args.center_score_blend_normalize,
                            center_margin_threshold=args.center_score_switch_center_margin,
                            enhanced_margin_threshold=args.center_score_switch_enhanced_margin,
                        )
            else:
                samples = [dataset[i] for i in range(start, end)]
                prompt_ids_list = [split_prompt_from_training_sample(sample) for sample in samples]
                raw_scores = score_labels_for_samples(model, tokenizer, samples, prompt_ids_list, candidate_answers, args)
                if label_prior_scores is not None:
                    scores = raw_scores - args.label_prior_weight * label_prior_scores.unsqueeze(0)
                else:
                    scores = raw_scores
                if use_center_score_fusion:
                    local_indices = list(range(start, end))
                    center_samples = build_center_only_samples(dataset, local_indices)
                    center_prompt_ids_list = [split_prompt_from_training_sample(sample) for sample in center_samples]
                    center_scores = score_labels_for_samples(
                        model,
                        tokenizer,
                        center_samples,
                        center_prompt_ids_list,
                        candidate_answers,
                        args,
                    )
                    if label_prior_scores is not None:
                        center_scores_for_blend = center_scores - args.label_prior_weight * label_prior_scores.unsqueeze(0)
                    else:
                        center_scores_for_blend = center_scores
                    scores, center_switch_mask = fuse_center_scores(
                        center_scores_for_blend,
                        scores,
                        mode=args.center_score_fusion_mode,
                        weight=blend_weight,
                        normalize=args.center_score_blend_normalize,
                        center_margin_threshold=args.center_score_switch_center_margin,
                        enhanced_margin_threshold=args.center_score_switch_enhanced_margin,
                    )
                    raw_scores, _ = fuse_center_scores(
                        center_scores,
                        raw_scores,
                        mode=args.center_score_fusion_mode,
                        weight=blend_weight,
                        normalize=args.center_score_blend_normalize,
                        center_margin_threshold=args.center_score_switch_center_margin,
                        enhanced_margin_threshold=args.center_score_switch_enhanced_margin,
                    )
            if args.score_mode != "generate":
                pred_offsets = scores.argmax(dim=1).tolist()

            for local_idx, pred_offset in enumerate(pred_offsets):
                dataset_idx = dataset.sample_indices[start + local_idx]
                gt_label_id = int(dataset.labels[dataset_idx].item())
                pred_label_id = valid_label_ids[pred_offset]
                record = dataset.records[dataset_idx]
                y_true.append(gt_label_id)
                y_pred.append(pred_label_id)
                ans_file.write(
                    json.dumps(
                        {
                            "node_id": dataset_idx,
                            "asin": record.get("asin"),
                            "gt_label_id": gt_label_id,
                            "gt_label": valid_id2label.get(gt_label_id, str(gt_label_id)),
                            "pred_label_id": pred_label_id,
                            "pred_label": valid_id2label[pred_label_id],
                            "score_mode": args.score_mode,
                            "center_score_blend_weight": blend_weight,
                            "center_score_fusion_mode": args.center_score_fusion_mode,
                            "center_score_switch_center_margin": args.center_score_switch_center_margin,
                            "center_score_switch_enhanced_margin": args.center_score_switch_enhanced_margin,
                            "center_score_blend_normalize": args.center_score_blend_normalize,
                            "center_score_switched": bool(center_switch_mask[local_idx].item()) if center_switch_mask is not None else None,
                            "pred_answer": candidate_labels[pred_offset] if args.score_mode != "candidate_likelihood" else candidate_answers[pred_offset],
                            "generated_text": generated_texts[local_idx] if generated_texts is not None else None,
                            "option_permutations": len(option_orders) if option_orders is not None else 1,
                            "option_order_mode": args.option_order_mode if args.score_mode == "option_logits" else None,
                            "scores": {
                                valid_id2label[label_id]: float(scores[local_idx, offset].item())
                                for offset, label_id in enumerate(valid_label_ids)
                            } if scores is not None else None,
                            "raw_scores": {
                                valid_id2label[label_id]: float(raw_scores[local_idx, offset].item())
                                for offset, label_id in enumerate(valid_label_ids)
                            } if (label_prior_scores is not None and raw_scores is not None) else None,
                            "center_scores": {
                                valid_id2label[label_id]: float(center_scores[local_idx, offset].item())
                                for offset, label_id in enumerate(valid_label_ids)
                            } if center_scores is not None else None,
                            "label_prior_scores": {
                                valid_id2label[label_id]: float(label_prior_scores[offset].item())
                                for offset, label_id in enumerate(valid_label_ids)
                            } if label_prior_scores is not None else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=valid_label_ids,
        average="macro",
        zero_division=0,
    )
    print(f"Accuracy: {acc:.6f}", flush=True)
    print(f"Macro-Precision: {precision:.6f}", flush=True)
    print(f"Macro-Recall: {recall:.6f}", flush=True)
    print(f"Macro-F1: {macro_f1:.6f}", flush=True)
    print(f"Answers: {args.answers_file}", flush=True)


if __name__ == "__main__":
    main()
