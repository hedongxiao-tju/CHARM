import argparse
import json
import os
import re
import tarfile
import io
from difflib import get_close_matches

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, IGNORE_INDEX
from model.conversation import conv_templates
from model.model.builder import load_pretrained_model
from model.mm_utils import get_model_name_from_path, tokenizer_graph_image_token
from model.train.train import load_sample_graph
from model.utils import disable_torch_init


DEFAULT_MODEL_BASE = "models/vicuna-7b-v1.5"
DEFAULT_DATA_ROOT = "data/ele-fashion"

CANDIDATE_LABELS = [
    "Footwear",
    "Miscellaneous",
    "Jewelry",
    "Lingerie",
    "Noveltywear",
    "Accessories",
    "Uniforms",
    "T-shirts",
    "Shirts",
    "Dresses",
    "Sneakers",
]

ID2LABEL = {
    0: "Footwear",
    1: "Miscellaneous",
    2: "Jewelry",
    3: "Lingerie",
    4: "Noveltywear",
    5: "Unused",
    6: "Accessories",
    7: "Uniforms",
    8: "T-shirts",
    9: "Shirts",
    10: "Dresses",
    11: "Sneakers",
}

VALID_ID2LABEL = {
    0: "Footwear",
    1: "Miscellaneous",
    2: "Jewelry",
    3: "Lingerie",
    4: "Noveltywear",
    6: "Accessories",
    7: "Uniforms",
    8: "T-shirts",
    9: "Shirts",
    10: "Dresses",
    11: "Sneakers",
}

AVAILABLE_CLASS_ID2LABEL = {
    0: "SneakersandMen’sFormalShoes",
    1: "Lingerie,Costumes,andWomen’sFootwear",
    2: "JewelryandAccessories",
    3: "StockingsandWatches",
    4: "GraphicT-ShirtsandSweatshirts",
    6: "Scarves,Suspenders,andWallets",
    7: "UndergarmentsandSocks",
    8: "BasicT-Shirts",
    9: "Men’sCasualandFormalShirts",
    10: "Dresses",
    11: "ShoesandBoots",
}

LABEL2ID = {v: k for k, v in VALID_ID2LABEL.items()}
NORMALIZED_LABEL2ID = {normalize_label: label_id for normalize_label, label_id in ((re.sub(r"\s+", " ", label.lower().replace("-", " ")).strip(), label_id) for label, label_id in LABEL2ID.items())}


def use_label_text_mode(mode: str):
    global CANDIDATE_LABELS, VALID_ID2LABEL, LABEL2ID, NORMALIZED_LABEL2ID
    if mode == "short":
        VALID_ID2LABEL = {
            0: "Footwear",
            1: "Miscellaneous",
            2: "Jewelry",
            3: "Lingerie",
            4: "Noveltywear",
            6: "Accessories",
            7: "Uniforms",
            8: "T-shirts",
            9: "Shirts",
            10: "Dresses",
            11: "Sneakers",
        }
    elif mode == "available_classes":
        VALID_ID2LABEL = dict(AVAILABLE_CLASS_ID2LABEL)
    else:
        raise ValueError(f"Unknown label text mode: {mode}")
    CANDIDATE_LABELS = [VALID_ID2LABEL[label_id] for label_id in sorted(VALID_ID2LABEL.keys())]
    LABEL2ID = {v: k for k, v in VALID_ID2LABEL.items()}
    NORMALIZED_LABEL2ID = {
        normalize_label: label_id
        for normalize_label, label_id in (
            (re.sub(r"\s+", " ", label.lower().replace("-", " ")).strip(), label_id)
            for label, label_id in LABEL2ID.items()
        )
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--graph-feat-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--image-archive", type=str, default=None)
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
    parser.add_argument("--label-text-mode", choices=["short", "available_classes"], default="short")
    parser.add_argument(
        "--graph-modality-mask-strategy",
        choices=["none", "center_one", "center_text", "center_image", "center_both"],
        default="none",
    )
    parser.add_argument("--graph-modality-mask-prob", type=float, default=1.0)
    parser.add_argument("--graph-disable-role-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-disable-struct-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def left_pad_input_ids(input_ids_list, pad_token_id: int):
    max_len = max(ids.shape[0] for ids in input_ids_list)
    batch_size = len(input_ids_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for idx, ids in enumerate(input_ids_list):
        seq_len = ids.shape[0]
        input_ids[idx, -seq_len:] = ids
        attention_mask[idx, -seq_len:] = True
    return input_ids, attention_mask


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


def collate_graph_batch(graph_batch, tokenizer):
    batch_size = len(graph_batch)
    max_nodes = max(sample["graph"].shape[0] for sample in graph_batch)
    feat_dim = graph_batch[0]["graph_emb"].shape[-1]
    max_role_tokens = max(role.shape[0] for sample in graph_batch for role in sample["graph_role_ids"])

    graphs = torch.full((batch_size, max_nodes), DEFAULT_GRAPH_PAD_ID, dtype=torch.long)
    graph_emb = torch.zeros((batch_size, max_nodes, feat_dim), dtype=torch.float32)
    graph_node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    graph_hops = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    graph_degrees = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    graph_ppr_scores = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    role_ids = torch.full(
        (batch_size, max_nodes, max_role_tokens),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )

    for batch_idx, sample in enumerate(graph_batch):
        node_count = sample["graph"].shape[0]
        graphs[batch_idx, :node_count] = sample["graph"]
        graph_emb[batch_idx, :node_count] = sample["graph_emb"]
        graph_node_mask[batch_idx, :node_count] = sample["graph_node_mask"]
        graph_hops[batch_idx, :node_count] = sample["graph_hops"]
        graph_degrees[batch_idx, :node_count] = sample["graph_degrees"]
        graph_ppr_scores[batch_idx, :node_count] = sample["graph_ppr_scores"]
        for node_idx, role in enumerate(sample["graph_role_ids"]):
            role_ids[batch_idx, node_idx, : role.shape[0]] = role

    role_mask = role_ids.ne(tokenizer.pad_token_id)
    return {
        "graphs": graphs,
        "graph_emb": graph_emb,
        "graph_node_mask": graph_node_mask,
        "graph_hops": graph_hops,
        "graph_degrees": graph_degrees,
        "graph_ppr_scores": graph_ppr_scores,
        "graph_role_ids": role_ids,
        "graph_role_mask": role_mask,
    }


def build_graph_modality_masks(graph_node_mask, strategy: str, prob: float):
    text_mask = torch.zeros_like(graph_node_mask)
    image_mask = torch.zeros_like(graph_node_mask)
    if strategy == "none" or prob <= 0 or graph_node_mask.shape[1] == 0:
        return text_mask, image_mask

    center_valid = graph_node_mask[:, 0] & (torch.rand(graph_node_mask.shape[0], device=graph_node_mask.device) < prob)
    if strategy == "center_one":
        mask_text = torch.rand(graph_node_mask.shape[0], device=graph_node_mask.device) < 0.5
        text_mask[:, 0] = center_valid & mask_text
        image_mask[:, 0] = center_valid & ~mask_text
    elif strategy == "center_text":
        text_mask[:, 0] = center_valid
    elif strategy == "center_image":
        image_mask[:, 0] = center_valid
    elif strategy == "center_both":
        text_mask[:, 0] = center_valid
        image_mask[:, 0] = center_valid
    else:
        raise ValueError(f"Unknown graph_modality_mask_strategy: {strategy}")
    return text_mask, image_mask


def load_test_data(data_root, manifest_path=None):
    labels = torch.load(os.path.join(data_root, "labels.pt"), map_location="cpu", weights_only=False)
    test_mask = torch.load(os.path.join(data_root, "test_mask.pt"), map_location="cpu", weights_only=False).bool()
    node_mapping = torch.load(os.path.join(data_root, "node_mapping.pt"), map_location="cpu", weights_only=False)
    id2asin = {int(v): str(k) for k, v in node_mapping.items()}
    manifest = None
    if manifest_path is not None and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    return labels, test_mask, id2asin, manifest


def load_image_for_record(data_root, record, image_folder=None, image_archive=None):
    image_name = record.get("image")
    if image_name is None:
        raise ValueError("Manifest record is missing image field")
    if image_folder is not None:
        candidate = os.path.join(image_folder, image_name)
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB")
    archive_path = image_archive or os.path.join(data_root, "ele-fashion-images.tar")
    archive_prefix = os.path.splitext(os.path.basename(archive_path))[0]
    with tarfile.open(archive_path, "r") as tar:
        try:
            member = tar.getmember(image_name)
        except KeyError:
            member = tar.getmember(f"{archive_prefix}/{image_name}")
        with tar.extractfile(member) as fp:
            return Image.open(io.BytesIO(fp.read())).convert("RGB")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_prediction(output_text: str):
    normalized = normalize_text(output_text)
    if normalized in NORMALIZED_LABEL2ID:
        return NORMALIZED_LABEL2ID[normalized], normalized

    matches = [label for label in CANDIDATE_LABELS if normalize_text(label) in normalized]
    if len(matches) == 1:
        return LABEL2ID[matches[0]], matches[0]

    if len(matches) > 1:
        matches = sorted(matches, key=len, reverse=True)
        return LABEL2ID[matches[0]], matches[0]

    close = get_close_matches(normalized, list(NORMALIZED_LABEL2ID.keys()), n=1, cutoff=0.5)
    if close:
        matched_id = NORMALIZED_LABEL2ID[close[0]]
        return matched_id, VALID_ID2LABEL[matched_id]

    return -1, output_text.strip()


def build_conv_prompt(conv_mode: str, user_prompt: str, answer=None) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


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


@torch.no_grad()
def score_candidate_batch(
    model,
    input_ids,
    attention_mask,
    labels,
    graphs,
    graph_emb,
    graph_node_mask,
    graph_hops,
    graph_degrees,
    graph_ppr_scores,
    graph_role_ids,
    graph_role_mask,
    graph_text_mask,
    graph_image_mask,
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
        graph_hops=graph_hops,
        graph_degrees=graph_degrees,
        graph_ppr_scores=graph_ppr_scores,
        graph_role_ids=graph_role_ids,
        graph_role_mask=graph_role_mask,
        graph_text_mask=graph_text_mask,
        graph_image_mask=graph_image_mask,
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
    user_prompts,
    prompt_id_list,
    pad_token_id,
    graphs,
    graph_emb,
    graph_node_mask,
    graph_hops,
    graph_degrees,
    graph_ppr_scores,
    graph_role_ids,
    graph_role_mask,
    graph_text_mask,
    graph_image_mask,
    reduction: str,
    candidate_batch_size: int = 0,
):
    if candidate_batch_size > 0:
        num_labels = len(CANDIDATE_LABELS)
        flat_inputs = []
        flat_labels = []
        for user_prompt, prompt_ids in zip(user_prompts, prompt_id_list):
            for label in CANDIDATE_LABELS:
                label_ids = label_token_ids_for_prompt(tokenizer, conv_mode, user_prompt, prompt_ids.tolist(), label)
                label_tensor = torch.tensor(label_ids, dtype=torch.long)
                full_ids = torch.cat([prompt_ids.cpu(), label_tensor], dim=0)
                full_labels = torch.full_like(full_ids, IGNORE_INDEX)
                full_labels[-label_tensor.shape[0] :] = label_tensor
                flat_inputs.append(full_ids)
                flat_labels.append(full_labels)

        model_device = graphs.device
        flat_scores = []
        for start in range(0, len(flat_inputs), candidate_batch_size):
            end = min(start + candidate_batch_size, len(flat_inputs))
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
                graph_hops=graph_hops.index_select(0, sample_indices),
                graph_degrees=graph_degrees.index_select(0, sample_indices),
                graph_ppr_scores=graph_ppr_scores.index_select(0, sample_indices),
                graph_role_ids=graph_role_ids.index_select(0, sample_indices),
                graph_role_mask=graph_role_mask.index_select(0, sample_indices),
                graph_text_mask=graph_text_mask.index_select(0, sample_indices),
                graph_image_mask=graph_image_mask.index_select(0, sample_indices),
                reduction=reduction,
            )
            flat_scores.append(scores)
        return torch.cat(flat_scores, dim=0).view(len(user_prompts), num_labels)

    batch_scores = []
    model_device = graphs.device
    for label in CANDIDATE_LABELS:
        candidate_inputs = []
        candidate_labels = []
        for user_prompt, prompt_ids in zip(user_prompts, prompt_id_list):
            label_ids = label_token_ids_for_prompt(tokenizer, conv_mode, user_prompt, prompt_ids.tolist(), label)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            full_ids = torch.cat([prompt_ids.cpu(), label_tensor], dim=0)
            full_labels = torch.full_like(full_ids, IGNORE_INDEX)
            full_labels[-label_tensor.shape[0] :] = label_tensor
            candidate_inputs.append(full_ids)
            candidate_labels.append(full_labels)

        candidate_input_ids, candidate_label_ids, candidate_attention_mask = left_pad_input_ids_and_labels(
            candidate_inputs,
            candidate_labels,
            pad_token_id,
        )
        scores = score_candidate_batch(
            model=model,
            input_ids=candidate_input_ids.to(model_device),
            attention_mask=candidate_attention_mask.to(model_device),
            labels=candidate_label_ids.to(model_device),
            graphs=graphs,
            graph_emb=graph_emb,
            graph_node_mask=graph_node_mask,
            graph_hops=graph_hops,
            graph_degrees=graph_degrees,
            graph_ppr_scores=graph_ppr_scores,
            graph_role_ids=graph_role_ids,
            graph_role_mask=graph_role_mask,
            graph_text_mask=graph_text_mask,
            graph_image_mask=graph_image_mask,
            reduction=reduction,
        )
        batch_scores.append(scores)
    return torch.stack(batch_scores, dim=1)


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


def main(args):
    disable_torch_init()
    if args.model_base is not None and str(args.model_base).lower() in {"", "none", "null"}:
        args.model_base = None
    use_label_text_mode(args.label_text_mode)
    print(
        json.dumps(
            {
                "stage": "eval_graph_textcls",
                "model_path": args.model_path,
                "model_base": args.model_base,
                "data_root": args.data_root,
                "manifest_path": args.manifest_path,
                "answers_file": args.answers_file,
                "max_samples": args.max_samples,
                "label_text_mode": args.label_text_mode,
                "graph_modality_mask_strategy": args.graph_modality_mask_strategy,
                "graph_modality_mask_prob": args.graph_modality_mask_prob,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        "[eval] candidate_labels: "
        + json.dumps({label_id: label for label_id, label in VALID_ID2LABEL.items()}, ensure_ascii=False),
        flush=True,
    )
    if args.score_candidates:
        print(
            f"[eval] scoring fixed label candidates with {args.candidate_score_reduction} log-prob reduction",
            flush=True,
        )

    model_name = infer_model_name(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model_device = next(model.parameters()).device
    conv_mode = infer_conv_mode(model_name, args.conv_mode)
    if args.graph_disable_role_tokens is not None:
        model.config.graph_disable_role_tokens = args.graph_disable_role_tokens
    if args.graph_disable_struct_tokens is not None:
        model.config.graph_disable_struct_tokens = args.graph_disable_struct_tokens
    print(f"[eval] model_name: {model_name}", flush=True)
    print(f"[eval] conv_mode: {conv_mode}", flush=True)
    print(f"[eval] graph_disable_role_tokens: {model.config.graph_disable_role_tokens}", flush=True)
    print(f"[eval] graph_disable_struct_tokens: {model.config.graph_disable_struct_tokens}", flush=True)
    print(f"[eval] graph_modality_mask_strategy: {args.graph_modality_mask_strategy}", flush=True)
    print(f"[eval] graph_modality_mask_prob: {args.graph_modality_mask_prob}", flush=True)

    labels, test_mask, id2asin, manifest = load_test_data(args.data_root, args.manifest_path)
    graph_root = args.data_root
    if args.manifest_path is not None:
        graph_root = os.path.dirname(os.path.abspath(args.manifest_path))
    if manifest is not None:
        records = manifest[: args.max_samples] if args.max_samples > 0 else manifest
    else:
        test_indices = torch.where(test_mask)[0].tolist()
        if args.max_samples > 0:
            test_indices = test_indices[:args.max_samples]
        records = [{"id": node_id, "asin": id2asin.get(node_id, str(node_id)), "graph_feat_path": None} for node_id in test_indices]
    print(f"[eval] loaded {len(records)} records", flush=True)
    print(f"[eval] batch_size: {args.batch_size}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    y_true = []
    y_pred = []
    valid_label_ids = sorted(VALID_ID2LABEL.keys())

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        batch_end = batch_start + len(batch_records)
        if batch_start == 0 or batch_end % 50 == 0 or batch_end == len(records):
            print(f"[eval] {batch_end}/{len(records)}", flush=True)

        prompts = []
        user_prompts = []
        graph_batch = []
        batch_meta = []
        for record in batch_records:
            node_id = int(record["id"])
            gt_label_id = int(labels[node_id].item())
            gt_label_text = VALID_ID2LABEL[gt_label_id]
            if not record.get("graph_feat_path"):
                raise ValueError("manifest_path with graph_feat_path entries is required for PPR graph evaluation.")
            graph_data = load_sample_graph(
                os.path.join(graph_root, record["graph_feat_path"]),
                tokenizer,
                record.get("graph_key"),
                args.graph_feat_path,
            )
            graph_batch.append(graph_data)
            batch_meta.append((record, node_id, gt_label_id, gt_label_text))

            user_prompt = record["conversations"][0]["value"] if record.get("conversations") else (
                f"{DEFAULT_GRAPH_TOKEN}\nClassify the center fashion product into one of the following categories:\n"
                + "; ".join(CANDIDATE_LABELS)
                + "\nAnswer with the category name only."
            )
            user_prompts.append(user_prompt)
            prompts.append(build_conv_prompt(conv_mode, user_prompt, answer=None))

        input_id_list = [
            tokenizer_graph_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in prompts
        ]
        input_ids, attention_mask = left_pad_input_ids(input_id_list, pad_token_id)
        collated_graph = collate_graph_batch(graph_batch, tokenizer)

        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        graphs = collated_graph["graphs"].to(model_device)
        graph_emb = collated_graph["graph_emb"].to(model_device, dtype=model.dtype)
        graph_node_mask = collated_graph["graph_node_mask"].to(model_device)
        graph_hops = collated_graph["graph_hops"].to(model_device)
        graph_degrees = collated_graph["graph_degrees"].to(model_device, dtype=model.dtype)
        graph_ppr_scores = collated_graph["graph_ppr_scores"].to(model_device, dtype=model.dtype)
        role_ids = collated_graph["graph_role_ids"].to(model_device)
        role_mask = collated_graph["graph_role_mask"].to(model_device)
        graph_text_mask, graph_image_mask = build_graph_modality_masks(
            graph_node_mask,
            args.graph_modality_mask_strategy,
            args.graph_modality_mask_prob,
        )

        if args.score_candidates:
            with torch.inference_mode():
                candidate_scores = score_label_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    conv_mode=conv_mode,
                    user_prompts=user_prompts,
                    prompt_id_list=input_id_list,
                    pad_token_id=pad_token_id,
                    graphs=graphs,
                    graph_emb=graph_emb,
                    graph_node_mask=graph_node_mask,
                    graph_hops=graph_hops,
                    graph_degrees=graph_degrees,
                    graph_ppr_scores=graph_ppr_scores,
                    graph_role_ids=role_ids,
                    graph_role_mask=role_mask,
                    graph_text_mask=graph_text_mask,
                    graph_image_mask=graph_image_mask,
                    reduction=args.candidate_score_reduction,
                    candidate_batch_size=args.candidate_batch_size,
                )
            best_label_indices = candidate_scores.argmax(dim=1).tolist()
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    graphs=graphs,
                    graph_emb=graph_emb,
                    graph_node_mask=graph_node_mask,
                    graph_hops=graph_hops,
                    graph_degrees=graph_degrees,
                    graph_ppr_scores=graph_ppr_scores,
                    graph_role_ids=role_ids,
                    graph_role_mask=role_mask,
                    graph_text_mask=graph_text_mask,
                    graph_image_mask=graph_image_mask,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-6) if args.temperature > 0 else 1.0,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

        for sample_idx, (record, node_id, gt_label_id, gt_label_text) in enumerate(batch_meta):
            if args.score_candidates:
                output_text = CANDIDATE_LABELS[best_label_indices[sample_idx]]
                pred_label_id = LABEL2ID[output_text]
                pred_label_text = output_text
            else:
                sample_output_ids = output_ids[sample_idx]
                prompt_len = input_ids.shape[1]
                if sample_output_ids.shape[0] > prompt_len:
                    generated_ids = sample_output_ids[prompt_len:]
                else:
                    generated_ids = sample_output_ids
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                if not output_text and generated_ids.data_ptr() != sample_output_ids.data_ptr():
                    output_text = tokenizer.decode(sample_output_ids, skip_special_tokens=True).strip()
                pred_label_id, pred_label_text = parse_prediction(output_text)

            y_true.append(gt_label_id)
            y_pred.append(pred_label_id)

            ans_file.write(
                json.dumps(
                    {
                        "node_id": node_id,
                        "asin": id2asin.get(node_id, str(node_id)),
                        "gt_label_id": gt_label_id,
                        "gt_label_text": gt_label_text,
                        "pred_text": output_text,
                        "pred_label_id": pred_label_id,
                        "pred_label_text": pred_label_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        ans_file.flush()

    ans_file.close()

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
    print(f"Answers saved to: {args.answers_file}")
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
