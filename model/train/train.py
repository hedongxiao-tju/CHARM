# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import csv
from dataclasses import dataclass, field, replace
import json
import logging
import pathlib
import html
import io
import math
import random
import re
import tarfile
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import transformers
import tokenizers

from model.constants import (
    DEFAULT_GRAPH_PAD_ID,
    DEFAULT_GRAPH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    GRAPH_TOKEN_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from torch.utils.data import Dataset, ConcatDataset
from model.train.model_trainer import LLaVATrainer

from model import conversation as conversation_lib
from model.model import *
from model.mm_utils import tokenizer_graph_image_token, tokenizer_image_token
from model.graph_utils import build_undirected_edge_index, compute_node_degrees

from PIL import Image


local_rank = None
_GRAPH_BUNDLE_CACHE: Dict[str, object] = {}
_GRAPH_FEATURE_CACHE: Dict[str, torch.Tensor] = {}
_SYNTHETIC_GRAPH_CACHE: Dict[str, object] = {}


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


ELE_FASHION_ID2LABEL = {
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

ELE_FASHION_VALID_ID2LABEL = {
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

ELE_FASHION_CANDIDATE_LABELS = [ELE_FASHION_VALID_ID2LABEL[idx] for idx in ELE_FASHION_VALID_ID2LABEL]

ELE_FASHION_LABEL_DESCRIPTIONS = {
    "Footwear": "Products mainly designed to be worn on feet, including shoes, boots, sandals, heels, and other footwear.",
    "Sneakers": "Athletic or casual shoes designed for comfort, sports, walking, or everyday wear.",
    "Jewelry": "Accessories worn for decoration, including necklaces, rings, earrings, bracelets, brooches, and watches.",
    "Accessories": "Supplementary wearable or carried fashion items such as bags, belts, scarves, hats, wallets, and small personal accessories.",
    "Bags": "Items used to carry personal belongings, including handbags, backpacks, shoulder bags, clutches, and wallets.",
    "T-shirts": "Casual upper-body garments, usually short-sleeved knit tops.",
    "Shirts": "Upper-body garments such as button-down shirts, blouses, polos, and woven tops.",
    "Dresses": "One-piece garments that cover the body from shoulders or torso downward.",
    "Lingerie": "Intimate apparel and undergarments, including bras, panties, sleepwear, and shapewear.",
    "Noveltywear": "Fashion items with novelty, costume, themed, humorous, or special-event designs.",
    "Uniforms": "Standardized clothing designed for work, school, teams, or formal organizational roles.",
    "Miscellaneous": "Fashion-related products that do not cleanly fit a more specific clothing or accessory category.",
}

DEFAULT_GRAPH_LABEL_PROMPT_TEMPLATE = (
    "Classify the center {subject_name} into one of the following categories:\n"
    "{label_space}\n"
    "Answer with the category name only."
)


def resolve_graph_path(root: Optional[str], explicit_path: Optional[str], filename: str) -> Optional[str]:
    if explicit_path is not None:
        if not os.path.isabs(explicit_path) and root is not None:
            rooted = os.path.join(root, explicit_path)
            if os.path.exists(rooted):
                return rooted
        return explicit_path
    if root is None:
        return None
    return os.path.join(root, filename)


def load_graph_label_spec(config_path: Optional[str]) -> Dict[str, object]:
    if config_path is None:
        valid_id2label = dict(ELE_FASHION_VALID_ID2LABEL)
        candidate_labels = [valid_id2label[idx] for idx in valid_id2label]
        label_descriptions = {
            label: ELE_FASHION_LABEL_DESCRIPTIONS.get(label, f"Items whose main semantic category is {label}.")
            for label in candidate_labels
        }
        return {
            "subject_name": "fashion product",
            "prompt_template": DEFAULT_GRAPH_LABEL_PROMPT_TEMPLATE,
            "candidate_labels": candidate_labels,
            "label_descriptions": label_descriptions,
            "valid_id2label": valid_id2label,
            "name": "ele-fashion",
        }

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    valid_id2label_raw = raw.get("valid_id2label")
    if valid_id2label_raw is None:
        raise ValueError(f"Missing valid_id2label in graph label config: {config_path}")

    valid_id2label = {int(k): str(v) for k, v in valid_id2label_raw.items()}
    candidate_labels = raw.get("candidate_labels")
    if candidate_labels is None:
        candidate_labels = [valid_id2label[idx] for idx in sorted(valid_id2label)]
    candidate_labels = [str(label) for label in candidate_labels]

    raw_descriptions = raw.get("label_descriptions", raw.get("class_descriptions", {}))
    label_descriptions = {}
    if isinstance(raw_descriptions, dict):
        for key, value in raw_descriptions.items():
            key_str = str(key)
            if key_str.isdigit() and int(key_str) in valid_id2label:
                label_descriptions[valid_id2label[int(key_str)]] = str(value)
            else:
                label_descriptions[key_str] = str(value)
    for label in candidate_labels:
        label_descriptions.setdefault(
            label,
            ELE_FASHION_LABEL_DESCRIPTIONS.get(
                label,
                f"Semantic category for {label.lower()} examples.",
            ),
        )

    return {
        "subject_name": str(raw.get("subject_name", "item")),
        "prompt_template": str(raw.get("prompt_template", DEFAULT_GRAPH_LABEL_PROMPT_TEMPLATE)),
        "candidate_labels": candidate_labels,
        "label_descriptions": label_descriptions,
        "valid_id2label": valid_id2label,
        "name": str(raw.get("name", os.path.basename(config_path))),
    }


def select_label_covered_subset(
    candidate_indices: List[int],
    labels: torch.Tensor,
    valid_label_ids: List[int],
    max_samples: int,
    min_samples_per_label: int = 1,
    seed: int = 0,
) -> List[int]:
    if max_samples <= 0:
        return candidate_indices
    if min_samples_per_label <= 0:
        raise ValueError(f"min_samples_per_label must be positive, got {min_samples_per_label}")

    required_min = len(valid_label_ids) * min_samples_per_label
    if required_min > max_samples:
        raise ValueError(
            f"max_train_samples={max_samples} is too small to cover {len(valid_label_ids)} labels "
            f"with graph_min_samples_per_label={min_samples_per_label}. Need at least {required_min} samples."
        )

    rng = random.Random(seed)
    indices_by_label = {label_id: [] for label_id in valid_label_ids}
    for idx in candidate_indices:
        label_id = int(labels[idx].item())
        if label_id in indices_by_label:
            indices_by_label[label_id].append(idx)

    missing_labels = [label_id for label_id, bucket in indices_by_label.items() if len(bucket) < min_samples_per_label]
    if missing_labels:
        raise ValueError(
            f"Unable to build a label-covered subset because these labels do not have enough training samples: "
            f"{missing_labels}"
        )

    selected = []
    selected_set = set()
    for label_id in valid_label_ids:
        bucket = indices_by_label[label_id]
        rng.shuffle(bucket)
        for idx in bucket[:min_samples_per_label]:
            selected.append(idx)
            selected_set.add(idx)

    remaining = [idx for idx in candidate_indices if idx not in selected_set]
    rng.shuffle(remaining)
    remaining_needed = max_samples - len(selected)
    if remaining_needed > 0:
        selected.extend(remaining[:remaining_needed])
    rng.shuffle(selected)
    return selected


def build_edge_list_from_pairs(edge_pairs, num_nodes):
    edge_list = [[] for _ in range(num_nodes)]
    if isinstance(edge_pairs, torch.Tensor):
        edge_pairs = edge_pairs.tolist()
    for src, dst in edge_pairs:
        src = int(src)
        dst = int(dst)
        edge_list[src].append(dst)
        edge_list[dst].append(src)
    return edge_list


def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, pad_val=DEFAULT_GRAPH_PAD_ID):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[int(node_idx)]]
    for _ in range(k_hop):
        current_hop = []
        for parent in neighbors[-1]:
            if parent == pad_val:
                current_hop.extend([pad_val] * sample_size)
                continue
            node_neighbors = edge_list[parent]
            if len(node_neighbors) > sample_size:
                sampled_neighbors = random.sample(node_neighbors, sample_size)
            else:
                sampled_neighbors = node_neighbors + [pad_val] * (sample_size - len(node_neighbors))
            current_hop.extend(sampled_neighbors)
        neighbors.append(current_hop)
    return [node for hop in neighbors for node in hop]


def graph_role_description(hop: int) -> str:
    if hop <= 0:
        return "center node"
    if hop == 1:
        return "one-hop neighbor"
    if hop == 2:
        return "two-hop neighbor"
    return "higher-hop context node"


def resolve_sample_graph_path(data_args, data_path: Optional[str], graph_path: str) -> str:
    if os.path.isabs(graph_path):
        return graph_path
    roots = []
    if getattr(data_args, "graph_data_root", None) is not None:
        roots.append(data_args.graph_data_root)
    if data_path is not None:
        roots.append(os.path.dirname(os.path.abspath(data_path)))
    for root in roots:
        candidate = os.path.join(root, graph_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(graph_path)


def load_image_from_folder_or_archive(
    image_folder,
    image_file,
    image_processor,
    image_aspect_ratio,
    image_archive=None,
    image_tar=None,
    archive_prefix=None,
):
    image = None
    if image_folder is not None:
        folder_candidate = os.path.join(image_folder, image_file)
        if os.path.exists(folder_candidate):
            image = Image.open(folder_candidate).convert('RGB')
    if image is None and image_archive is not None:
        tar = image_tar if image_tar is not None else tarfile.open(image_archive, "r")
        member_name = image_file
        if archive_prefix is None:
            archive_prefix = os.path.splitext(os.path.basename(image_archive))[0]
        try:
            member = tar.getmember(member_name)
        except KeyError:
            member = tar.getmember(f"{archive_prefix}/{image_file}")
        with tar.extractfile(member) as fp:
            image = Image.open(io.BytesIO(fp.read())).convert('RGB')
        if image_tar is None:
            tar.close()
    if image is None:
        raise FileNotFoundError(f"Unable to locate image {image_file} from folder={image_folder} archive={image_archive}")

    if image_aspect_ratio == 'pad':
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    return image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


def load_sample_graph(
    graph_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    graph_key: Optional[int] = None,
    graph_feat_path: Optional[str] = None,
) -> Dict[str, object]:
    if graph_key is not None:
        if graph_path not in _GRAPH_BUNDLE_CACHE:
            _GRAPH_BUNDLE_CACHE[graph_path] = torch.load(graph_path, map_location="cpu", weights_only=False)
        graph = _GRAPH_BUNDLE_CACHE[graph_path]
    else:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    if graph_key is not None:
        if not isinstance(graph, dict):
            raise TypeError(f"Bundled graph file must be a dict when graph_key is provided: {graph_path}")
        key_candidates = [graph_key, str(graph_key)]
        matched = None
        for candidate in key_candidates:
            if candidate in graph:
                matched = graph[candidate]
                break
        if matched is None:
            raise KeyError(f"graph_key {graph_key} not found in bundled graph file: {graph_path}")
        graph = matched

    if torch.is_tensor(graph) or isinstance(graph, (list, tuple)):
        node_ids = torch.as_tensor(graph, dtype=torch.long)
        graph = {"node_ids": node_ids}
    elif isinstance(graph, dict) and "node_ids" in graph:
        node_ids = torch.as_tensor(graph["node_ids"], dtype=torch.long)
    else:
        raise ValueError(f"sample graph must store node indices only: {graph_path}")

    if graph_feat_path is None:
        raise ValueError(f"graph_feat_path is required when graph stores node indices only: {graph_path}")
    if graph_feat_path not in _GRAPH_FEATURE_CACHE:
        graph_features = torch.load(graph_feat_path, map_location="cpu", weights_only=False)
        if isinstance(graph_features, dict):
            if "features" in graph_features:
                graph_features = graph_features["features"]
            else:
                raise ValueError(f"Unsupported graph feature dict keys: {list(graph_features.keys())}")
        _GRAPH_FEATURE_CACHE[graph_feat_path] = torch.as_tensor(graph_features, dtype=torch.float32)
    graph_emb = _GRAPH_FEATURE_CACHE[graph_feat_path][node_ids].to(torch.float32)

    num_nodes = graph_emb.shape[0]
    node_ids = node_ids.to(torch.long)
    hops = torch.as_tensor(graph.get("hop_dist", graph.get("hops", torch.zeros(num_nodes))), dtype=torch.long)
    degrees = torch.as_tensor(graph.get("degree", graph.get("degrees", torch.zeros(num_nodes))), dtype=torch.float32)
    ppr_scores = torch.as_tensor(graph.get("ppr_score", graph.get("ppr_scores", torch.zeros(num_nodes))), dtype=torch.float32)
    local_edges = torch.as_tensor(graph.get("local_edges", torch.empty(0, 2)), dtype=torch.long).reshape(-1, 2)

    role_token_ids = []
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    for hop in hops.tolist():
        token_ids = tokenizer(graph_role_description(int(hop)), add_special_tokens=False).input_ids
        role_token_ids.append(torch.tensor(token_ids or [unk_id], dtype=torch.long))

    return {
        "graph": node_ids,
        "graph_emb": graph_emb,
        "graph_targets": graph_emb.clone(),
        "graph_node_mask": torch.ones(num_nodes, dtype=torch.bool),
        "graph_hops": hops,
        "graph_degrees": degrees,
        "graph_ppr_scores": ppr_scores,
        "graph_role_ids": role_token_ids,
        "graph_edges": local_edges,
    }


def build_graph_classification_sample(label_name: str, subject_name: str = "fashion product") -> List[Dict[str, str]]:
    prompt = (
        f"<image>\n<graph>\n"
        f"You are given the center {subject_name}, its image, and a PPR-ranked local graph context encoded as graph soft tokens. "
        f"Use the center node, one-hop neighbors, two-hop neighbors, and higher-hop context together. "
        f"Classify the center {subject_name} into exactly one category from: "
        f"{', '.join(ELE_FASHION_CANDIDATE_LABELS)}.\n"
        f"Answer with the category name only."
    )
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": label_name},
    ]


def _clean_graph_text(text: object, max_chars: int = 220) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text if x is not None)
    text = html.unescape(str(text)).replace("\n", " ").strip()
    return " ".join(text.split())[:max_chars]


def _load_graph_records_from_csv(csv_path: str) -> List[Dict[str, object]]:
    records = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            node_id = int(row.get("id", row_idx))
            text_parts = []
            # Use only instance-level content. Category/subreddit fields can be
            # identical to the node label in MAGB and would leak the answer.
            for key in ("title", "text", "caption", "description"):
                value = row.get(key)
                if value:
                    text_parts.append(value)
            records.append(
                {
                    "asin": str(node_id),
                    "node_id": node_id,
                    "raw_text": _clean_graph_text(" ".join(text_parts), max_chars=800),
                    "label": int(row["label"]) if row.get("label", "").strip().isdigit() else row.get("label"),
                }
            )
    records.sort(key=lambda item: int(item.get("node_id", 0)))
    return records


def _find_graph_csv(data_root: Optional[str]) -> Optional[str]:
    if data_root is None:
        return None
    root = pathlib.Path(data_root)
    candidates = sorted(root.glob("*.csv"))
    if len(candidates) == 1:
        return str(candidates[0])
    preferred = {
        "grocery": "Grocery.csv",
        "toys": "Toys.csv",
        "movies": "Movies.csv",
        "reddits": "RedditS.csv",
        "reddit-s": "RedditS.csv",
    }
    key = os.path.basename(os.path.abspath(data_root)).lower()
    filename = preferred.get(key)
    if filename and (root / filename).exists():
        return str(root / filename)
    return str(candidates[0]) if candidates else None


def _format_label_space(label_spec: Dict[str, object], include_descriptions: bool = True) -> str:
    descriptions = label_spec.get("label_descriptions", {}) if include_descriptions else {}
    lines = []
    for label in label_spec["candidate_labels"]:
        desc = descriptions.get(label)
        if desc:
            lines.append(f"- {label}: {_clean_graph_text(desc, max_chars=60)}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def _format_label_evidence(
    label_spec: Dict[str, object],
    label: str,
    include_descriptions: bool = True,
    max_desc_chars: int = 90,
) -> str:
    subject_name = str(label_spec.get("subject_name", "item"))
    desc = label_spec.get("label_descriptions", {}).get(label) if include_descriptions else None
    evidence = (
        f"{label}. Typical evidence: a {subject_name} photo, title, or description whose category is {label}; "
        f"look for visual appearance, title words, and neighbor items that match this category."
    )
    if desc:
        evidence += f" Semantic cue: {_clean_graph_text(desc, max_chars=max_desc_chars)}"
    return evidence


def _format_label_space_promptv2(label_spec: Dict[str, object], include_descriptions: bool = True) -> str:
    lines = []
    for label in label_spec["candidate_labels"]:
        lines.append(f"- {_format_label_evidence(label_spec, label, include_descriptions, max_desc_chars=70)}")
    return "\n".join(lines)


def _format_compact_label_space(
    label_spec: Dict[str, object],
    option_labels: Optional[Sequence[str]] = None,
    option_prompt: bool = False,
) -> str:
    labels = _ordered_option_labels(label_spec, option_labels)
    if option_prompt:
        return "\n".join(f"{_option_letter(idx)}. {label}" for idx, label in enumerate(labels))
    return ", ".join(labels)


def _format_label_answer(label: str, label_spec: Dict[str, object], with_description: bool = True) -> str:
    if not with_description:
        return label
    desc = label_spec.get("label_descriptions", {}).get(label)
    if not desc:
        return label
    return f"Category: {label}\nSemantic description: {desc}"


def _option_letter(index: int) -> str:
    if not 0 <= index < 1000:
        raise ValueError(f"Only up to 1000 option labels are supported, got index={index}")
    return chr(ord("A") + index)


def _ordered_option_labels(label_spec: Dict[str, object], option_labels: Optional[Sequence[str]] = None) -> List[str]:
    labels = [str(label) for label in (option_labels if option_labels is not None else label_spec["candidate_labels"])]
    if len(labels) > 1000:
        raise ValueError(f"Only up to 1000 option labels are supported, got {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate option labels are not supported: {labels}")
    known_labels = set(str(label) for label in label_spec["candidate_labels"])
    missing = [label for label in labels if label not in known_labels]
    if missing:
        raise ValueError(f"Option labels {missing} are not in candidate_labels for {label_spec.get('name')}")
    return labels


def _label_semantic_counter(label_spec: Dict[str, object], label: str) -> Dict[str, float]:
    desc = label_spec.get("label_descriptions", {}).get(label, "")
    words = re.findall(r"[a-z0-9]+", f"{label} {desc}".lower())
    counts: Dict[str, float] = {}

    def add(key: str, value: float = 1.0):
        counts[key] = counts.get(key, 0.0) + value

    for word in words:
        if len(word) >= 3:
            add(f"w:{word}", 2.0)
    for idx in range(max(0, len(words) - 1)):
        add(f"b:{words[idx]}_{words[idx + 1]}", 3.0)
    compact = "".join(words)
    for ngram_size in (3, 4, 5):
        for idx in range(max(0, len(compact) - ngram_size + 1)):
            add(f"c{ngram_size}:{compact[idx:idx + ngram_size]}", 0.25)
    return counts


def _counter_cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-12)


def _build_label_hard_negative_order(label_spec: Dict[str, object]) -> Dict[str, List[str]]:
    labels = _ordered_option_labels(label_spec)
    counters = {label: _label_semantic_counter(label_spec, label) for label in labels}
    hard_orders: Dict[str, List[str]] = {}
    for label in labels:
        scored = []
        for other in labels:
            if other == label:
                continue
            scored.append((_counter_cosine(counters[label], counters[other]), other))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hard_orders[label] = [other for _, other in scored]
    return hard_orders


def _format_option_label_space(
    label_spec: Dict[str, object],
    include_descriptions: bool = True,
    option_labels: Optional[Sequence[str]] = None,
    prompt_style: str = "default",
) -> str:
    descriptions = label_spec.get("label_descriptions", {}) if include_descriptions else {}
    lines = []
    for idx, label in enumerate(_ordered_option_labels(label_spec, option_labels)):
        desc = descriptions.get(label)
        prefix = _option_letter(idx)
        if prompt_style == "promptv2":
            lines.append(f"{prefix}. {_format_label_evidence(label_spec, label, include_descriptions)}")
        elif desc:
            lines.append(f"{prefix}. {label}: {_clean_graph_text(desc, max_chars=90)}")
        else:
            lines.append(f"{prefix}. {label}")
    return "\n".join(lines)


def _label_to_option_letter(
    label: str,
    label_spec: Dict[str, object],
    option_labels: Optional[Sequence[str]] = None,
) -> str:
    labels = _ordered_option_labels(label_spec, option_labels)
    if label not in labels:
        raise ValueError(f"Label {label!r} is not in candidate_labels for {label_spec.get('name')}")
    return _option_letter(labels.index(label))


def _normalize_dataset_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _infer_dataset_name(data_root: Optional[str], available: Sequence[str]) -> Optional[str]:
    if data_root is None:
        return None
    root_name = os.path.basename(os.path.abspath(data_root))
    normalized = _normalize_dataset_name(root_name)
    for candidate in available:
        if _normalize_dataset_name(candidate) == normalized:
            return candidate
    for candidate in available:
        if _normalize_dataset_name(candidate) in normalized or normalized in _normalize_dataset_name(candidate):
            return candidate
    return None


def _fixed_graph_hops(num_nodes: int, k_hop: int, sample_size: int) -> torch.Tensor:
    hops = []
    remaining = num_nodes
    for hop in range(k_hop + 1):
        count = sample_size ** hop
        take = min(count, remaining)
        hops.extend([hop] * take)
        remaining -= take
        if remaining <= 0:
            break
    if len(hops) < num_nodes:
        hops.extend([k_hop] * (num_nodes - len(hops)))
    return torch.tensor(hops[:num_nodes], dtype=torch.long)


def _modality_relation_from_sims(text_sim: float, image_sim: float, thresholds: Optional[Dict[str, float]]) -> Tuple[str, str]:
    if thresholds is None:
        thresholds = {"text_high": 0.55, "text_low": 0.35, "image_high": 0.55, "image_low": 0.35}
    text_high = text_sim >= float(thresholds.get("text_high", 0.55))
    text_low = text_sim <= float(thresholds.get("text_low", 0.35))
    image_high = image_sim >= float(thresholds.get("image_high", 0.55))
    image_low = image_sim <= float(thresholds.get("image_low", 0.35))
    if text_high and image_high:
        return "modality-consistent", "text-high, image-high; strong positive multimodal evidence."
    if text_high and image_low:
        return "modality-complement", "text-high, image-low; similar semantic text but different visual appearance."
    if image_high and text_low:
        return "modality-complement", "image-high, text-low; visual evidence may compensate for weak or noisy text."
    if text_low and image_low:
        return "modality-conflict", "text-low, image-low; weak or contrastive evidence."
    if image_high:
        return "visual-support", "image-high with partial textual agreement; moderate visual support."
    if text_high:
        return "text-support", "text-high with partial visual agreement; moderate semantic support."
    return "structural-context", "structurally close but only weakly aligned by modalities."


class SyntheticGraphContext:
    """Lightweight view over the MAGB synthetic graph for zero-shot serialization."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        if self.root in _SYNTHETIC_GRAPH_CACHE:
            cached = _SYNTHETIC_GRAPH_CACHE[self.root]
            self.__dict__.update(cached.__dict__)
            return

        self.offsets = json.load(open(os.path.join(self.root, "dataset_offsets.json"), "r", encoding="utf-8"))
        self.meta = {}
        meta_path = os.path.join(self.root, "kmeans_global_synthetic_graph_meta.json")
        if os.path.exists(meta_path):
            self.meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.layer1_nodes = torch.load(os.path.join(self.root, "layer1_abstract_nodes.pt"), map_location="cpu", weights_only=False)
        self.layer2_nodes = torch.load(os.path.join(self.root, "layer2_abstract_nodes.pt"), map_location="cpu", weights_only=False)
        self.global_nodes = torch.load(os.path.join(self.root, "global_semantic_prototype_nodes.pt"), map_location="cpu", weights_only=False)
        self.cluster_assignments = torch.load(os.path.join(self.root, "cluster_assignments.pt"), map_location="cpu", weights_only=False)
        self.layer2_assignments = torch.load(os.path.join(self.root, "layer2_cluster_assignments.pt"), map_location="cpu", weights_only=False)
        attachments_path = os.path.join(self.root, "target_anchor_attachments.pt")
        self.target_anchor_attachments = (
            torch.load(attachments_path, map_location="cpu", weights_only=False)
            if os.path.exists(attachments_path) else {}
        )

        self.layer1_by_dataset = {}
        for node in self.layer1_nodes:
            self.layer1_by_dataset.setdefault(node["dataset"], []).append(node)
        for nodes in self.layer1_by_dataset.values():
            nodes.sort(key=lambda item: int(item.get("cluster_id", 0)))

        self.layer2_by_dataset = {}
        for node in self.layer2_nodes:
            self.layer2_by_dataset.setdefault(node["dataset"], []).append(node)
        for nodes in self.layer2_by_dataset.values():
            nodes.sort(key=lambda item: int(item.get("topic_id", 0)))

        self.global_by_id = {int(node["global_node_id"]): node for node in self.global_nodes}
        self.layer1_by_global_id = {int(node["global_node_id"]): node for node in self.layer1_nodes}
        self.layer2_by_global_id = {int(node["global_node_id"]): node for node in self.layer2_nodes}
        self.layer2_to_global = {}
        self.global_to_layer2 = {}
        for proto in self.global_nodes:
            for layer2_id in proto.get("member_layer2_abstract_node_ids", []):
                self.layer2_to_global[int(layer2_id)] = int(proto["global_node_id"])
                self.global_to_layer2.setdefault(int(proto["global_node_id"]), []).append(int(layer2_id))

        self.abstract_coord_cache = {}
        for dataset in self.layer1_by_dataset.keys():
            nodes = list(self.layer1_by_dataset.get(dataset, [])) + list(self.layer2_by_dataset.get(dataset, []))
            text = []
            image = []
            for node in nodes:
                text.append(torch.as_tensor(node["text_embedding"], dtype=torch.float32))
                image.append(torch.as_tensor(node["image_embedding"], dtype=torch.float32))
            if text:
                self.abstract_coord_cache[dataset] = {
                    "nodes": nodes,
                    "text": F.normalize(torch.stack(text, dim=0).to(torch.float32), dim=-1),
                    "image": F.normalize(torch.stack(image, dim=0).to(torch.float32), dim=-1),
                }
        if self.target_anchor_attachments:
            source_nodes = list(self.layer1_nodes) + list(self.layer2_nodes)
            self.source_abstract_coord_candidates = {
                "nodes": source_nodes,
                "text": F.normalize(torch.stack([
                    torch.as_tensor(node["text_embedding"], dtype=torch.float32) for node in source_nodes
                ], dim=0), dim=-1),
                "image": F.normalize(torch.stack([
                    torch.as_tensor(node["image_embedding"], dtype=torch.float32) for node in source_nodes
                ], dim=0), dim=-1),
            }
        else:
            self.source_abstract_coord_candidates = None

        self.bridge_by_dataset = {}
        bridge_path = os.path.join(self.root, "modality_complement_bridge_edges.pt")
        if os.path.exists(bridge_path):
            bridge = torch.load(bridge_path, map_location="cpu", weights_only=False)
            edge_index = bridge["edge_index"]
            for idx in range(edge_index.shape[0]):
                dataset = bridge["dataset"][idx]
                offset = self.offsets[dataset]["start"]
                src = int(edge_index[idx, 0].item()) - int(offset)
                dst = int(edge_index[idx, 1].item()) - int(offset)
                record = {
                    "dst": dst,
                    "direction": bridge["direction"][idx],
                    "abstract_consistency_type": bridge["abstract_consistency_type"][idx],
                    "text_similarity": float(bridge["text_similarity"][idx].item()),
                    "image_similarity": float(bridge["image_similarity"][idx].item()),
                    "confidence": float(bridge["confidence"][idx].item()),
                    "structure_consistency_score": float(bridge["structure_consistency_score"][idx].item()),
                }
                self.bridge_by_dataset.setdefault(dataset, {}).setdefault(src, []).append(record)
            for adjacency in self.bridge_by_dataset.values():
                for records in adjacency.values():
                    records.sort(key=lambda item: item["confidence"], reverse=True)

        bridge_meta = self.meta.get("modality_complement_bridge", {})
        self.thresholds_by_dataset = bridge_meta.get("thresholds_by_dataset", {})
        _SYNTHETIC_GRAPH_CACHE[self.root] = self

    def membership(self, dataset: str, node_id: int) -> Dict[str, object]:
        attachment = self.target_anchor_attachments.get(dataset)
        if attachment is not None:
            layer1_id = int(attachment["layer1_global_node_ids"][node_id].item())
            layer2_id = int(attachment["layer2_global_node_ids"][node_id].item())
            global_id = int(attachment["global_node_ids"][node_id].item())
            return {
                "layer1": self.layer1_by_global_id[layer1_id],
                "layer2": self.layer2_by_global_id[layer2_id],
                "global": self.global_by_id[global_id],
            }
        cluster_id = int(self.cluster_assignments[dataset][node_id].item())
        layer1_nodes = self.layer1_by_dataset[dataset]
        cluster_to_pos = {int(node["cluster_id"]): pos for pos, node in enumerate(layer1_nodes)}
        layer1_pos = cluster_to_pos[cluster_id]
        layer1 = layer1_nodes[layer1_pos]
        layer2_topic = int(self.layer2_assignments[dataset][layer1_pos].item())
        layer2 = self.layer2_by_dataset[dataset][layer2_topic]
        global_node = self.global_by_id.get(self.layer2_to_global.get(int(layer2["global_node_id"]), -1))
        return {"layer1": layer1, "layer2": layer2, "global": global_node}

    @staticmethod
    def abstract_feature(node: Optional[Dict[str, object]], feat_dim: int) -> Optional[torch.Tensor]:
        if node is None:
            return None
        text = torch.as_tensor(node["text_embedding"], dtype=torch.float32)
        image = torch.as_tensor(node["image_embedding"], dtype=torch.float32)
        feat = torch.cat([text, image], dim=0)
        if feat.numel() == feat_dim:
            return feat
        if feat.numel() > feat_dim:
            return feat[:feat_dim]
        padded = torch.zeros(feat_dim, dtype=torch.float32)
        padded[: feat.numel()] = feat
        return padded

    def bridge_neighbors(self, dataset: str, node_id: int, limit: int) -> List[Dict[str, object]]:
        return list(self.bridge_by_dataset.get(dataset, {}).get(int(node_id), []))[:limit]

    def abstract_coord_candidates(self, dataset: str) -> Optional[Dict[str, object]]:
        if dataset in self.target_anchor_attachments:
            return self.source_abstract_coord_candidates
        return self.abstract_coord_cache.get(dataset)

    def layer2_source_nodes_for_global(self, global_node_id: int, target_dataset: str) -> List[Dict[str, object]]:
        out = []
        for layer2_id in self.global_to_layer2.get(int(global_node_id), []):
            node = self.layer2_by_global_id.get(int(layer2_id))
            if node is not None and node.get("dataset") != target_dataset:
                out.append(node)
        return out


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_graph_projector_only: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    graph_projector_type: Optional[str] = field(default="linear")
    graph_fusion_mode: Optional[str] = field(default="dimf", metadata={"help": "Cross-modal fusion mode: dimf (gated) or add (simple average)"})
    graph_init_seed: int = field(default=42, metadata={"help": "Random seed for model initialization (before Trainer.set_seed)"})
    graph_decoder_depth: int = field(default=2)
    graph_feat_loss_weight: float = field(default=1.0)
    graph_use_soft_tokens: bool = field(default=False)
    graph_use_strict_zeroshot: bool = field(default=False)
    graph_num_domains: int = field(default=0, metadata={"help": "GRL domain adversarial training: number of domains"})
    graph_grl_lambda: float = field(default=1.0, metadata={"help": "GRL strength"})
    sag_num_dataset_anchors: int = field(default=200, metadata={"help": "ZS: semantic anchors per dataset"})
    sag_num_global_anchors: int = field(default=50, metadata={"help": "ZS: number of global semantic anchors"})
    sag_anchor_topk: int = field(default=5, metadata={"help": "ZS: top-k anchor retrieval"})
    sag_global_anchor_topk: int = field(default=3, metadata={"help": "ZS: top-k global anchor retrieval"})
    sag_evidence_topk: int = field(default=10, metadata={"help": "ZS: top-k evidence aggregation"})
    zs_anchor_alignment_loss_weight: float = field(default=0.1, metadata={"help": "ZS: anchor alignment loss weight"})
    zs_calibration_consistency_loss_weight: float = field(default=0.1, metadata={"help": "ZS: calibration consistency loss weight"})
    zs_domain_adv_loss_weight: float = field(default=0.1, metadata={"help": "ZS: domain adversarial loss weight"})
    graph_aux_loss_weight: float = field(default=0.0)
    graph_text_rec_weight: float = field(default=1.0)
    graph_image_rec_weight: float = field(default=1.0)
    graph_edge_rec_weight: float = field(default=1.0)
    graph_modality_mask_prob: float = field(default=0.15)
    graph_modality_mask_strategy: str = field(default="random")
    graph_disable_role_tokens: bool = field(default=False)
    graph_disable_struct_tokens: bool = field(default=False)
    graph_soft_token_gnn_layers: int = field(default=0)
    graph_soft_token_encoder_type: str = field(default="charm", metadata={"help": "charm | plain_gcn"})
    graph_use_gcn: bool = field(default=True)
    graph_use_gcn_1hop: bool = field(default=True)
    graph_use_gcn_2hop: bool = field(default=True)
    graph_use_center_protect: bool = field(default=True)
    graph_use_ppr_prop: bool = field(default=True)
    graph_include_abstract_soft_tokens: bool = field(default=True)
    graph_include_global_soft_tokens: bool = field(default=True)
    graph_soft_token_mode: str = field(default="legacy")
    graph_semantic_evidence_scale: float = field(
        default=0.25,
        metadata={"help": "Initial multiplicative scale for non-center semantic-coordinate evidence tokens."},
    )
    graph_semantic_evidence_gate_init: float = field(
        default=-3.0,
        metadata={"help": "Initial bias for the semantic-coordinate evidence gate; more negative starts lighter."},
    )
    graph_hidden_size: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    graph_data_root: Optional[str] = field(default=None)
    graph_data_roots: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated graph_data_root values for joint graph fine-tuning."},
    )
    graph_text_path: Optional[str] = field(default=None)
    graph_text_paths: Optional[str] = field(default=None)
    graph_feat_path: Optional[str] = field(default=None)
    graph_feat_paths: Optional[str] = field(default=None)
    graph_edge_path: Optional[str] = field(default=None)
    graph_edge_paths: Optional[str] = field(default=None)
    graph_image_archive: Optional[str] = field(default=None)
    graph_image_archives: Optional[str] = field(default=None)
    graph_hops: int = field(default=2)
    graph_sample_size: int = field(default=10)
    graph_only: bool = field(default=False)
    graph_cache_path: Optional[str] = field(default=None)
    graph_cache_paths: Optional[str] = field(default=None)
    graph_feat_cache_path: Optional[str] = field(default=None)
    graph_feat_cache_paths: Optional[str] = field(default=None)
    graph_label_supervision: bool = field(default=False)
    graph_label_config: Optional[str] = field(default=None)
    graph_label_configs: Optional[str] = field(default=None)
    graph_synthetic_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root of the hierarchical synthetic graph with abstract nodes and modality bridge edges."},
    )
    graph_synthetic_roots: Optional[str] = field(default=None)
    graph_zero_shot_serializer: bool = field(default=True)
    graph_include_label_descriptions: bool = field(default=True)
    graph_answer_with_description: bool = field(default=True)
    graph_label_prompt_style: str = field(
        default="default",
        metadata={"help": "Label text style for zero-shot graph prompts: default or promptv2."},
    )
    graph_context_prompt_style: str = field(
        default="full",
        metadata={"help": "Graph evidence prompt style: full or compact."},
    )
    graph_option_label_prompt: bool = field(
        default=False,
        metadata={"help": "Format MAGB candidate labels as A/B/C options and use option letters as supervised answers."},
    )
    graph_shuffle_label_options: bool = field(
        default=False,
        metadata={"help": "Shuffle A/B/C candidate order per sample so option-letter supervision cannot learn fixed label positions."},
    )
    graph_label_option_seed: int = field(default=0)
    graph_candidate_likelihood_training: bool = field(
        default=False,
        metadata={"help": "Also train by ranking candidate label texts with frozen LLM likelihoods."},
    )
    graph_candidate_negatives: int = field(
        default=-1,
        metadata={"help": "Number of negative candidate labels per sample for likelihood ranking. Use -1 for all labels."},
    )
    graph_candidate_seed: int = field(default=0)
    graph_candidate_negative_strategy: str = field(
        default="random",
        metadata={"help": "Negative label sampling for candidate likelihood training: random, semantic, or semantic_mix."},
    )
    graph_max_consistent_neighbors: int = field(default=4)
    graph_max_complement_neighbors: int = field(default=4)
    graph_max_conflict_neighbors: int = field(default=2)
    graph_include_abstract_nodes: bool = field(default=True)
    graph_include_synthetic_soft_tokens: bool = field(
        default=True,
        metadata={"help": "Append selected synthetic abstract/bridge nodes as graph soft tokens in addition to prompt text."},
    )

    graph_semantic_coord_neighbor_topk: int = field(default=0)
    graph_semantic_coord_use_global_nodes: bool = field(default=True)
    graph_semantic_coord_global_alignment: bool = field(default=False)
    graph_semantic_coord_hybrid_prompt: bool = field(default=False)
    graph_semantic_coord_evidence_prompt: bool = field(default=True)
    graph_semantic_coord_center_only: bool = field(default=False)
    graph_semantic_coord_use_abstract_token: bool = field(default=True)
    graph_semantic_coord_use_calibration_token: bool = field(default=True)
    graph_semantic_coord_use_transfer_token: bool = field(default=True)

    graph_real_node_serializer: bool = field(default=False)
    graph_real_neighbor_topk: int = field(default=8)
    graph_real_cross_topk: int = field(default=4)
    graph_include_cross_domain_soft_tokens: bool = field(default=True)
    graph_ablate_hop: int = field(
        default=-1,
        metadata={"help": "Ablate tokens of this hop type (-1=none, 2=layer1/abstract, 3=layer2/calibration, 4=global/transfer)"},
    )
    graph_ablate_layer: int = field(
        default=-1,
        metadata={"help": "Training-time ablation: 1=remove layer1 token, 2=remove layer2 token, 4=remove transfer token"},
    )
    graph_semantic_coord_tokens: bool = field(
        default=False,
        metadata={"help": "Use five semantic-coordinate graph tokens: center text, center image, abstract, calibration, transfer."},
    )
    graph_semantic_coord_topk: int = field(default=3)
    graph_semantic_coord_graph_topk: int = field(default=4)
    graph_semantic_coord_transfer_topk: int = field(default=3)
    graph_residual_coord_tokens: bool = field(
        default=False,
        metadata={"help": "Use residual coordinate mode: graph evidence as bounded residuals to center tokens."},
    )
    graph_residual_max_ratio: float = field(
        default=0.15,
        metadata={"help": "Max ratio of graph delta norm to center token norm."},
    )
    graph_residual_cosine_gate: bool = field(
        default=True,
        metadata={"help": "Scale graph delta by cosine similarity with center."},
    )
    graph_min_abstract_similarity: float = field(
        default=-1.0,
        metadata={"help": "If >= -1, keep synthetic abstract anchors only when cosine(center, anchor) is at least this value."},
    )
    graph_min_bridge_confidence: float = field(
        default=-1.0,
        metadata={"help": "If >= -1, keep modality-complement bridge neighbors only when bridge confidence is at least this value."},
    )
    graph_stratified_label_subset: bool = field(default=False)
    graph_min_samples_per_label: int = field(default=1)
    graph_subset_seed: int = field(default=0)
    graph_train_sample_ratio: float = field(
        default=-1.0,
        metadata={"help": "If > 0, sample this fraction of supervised graph nodes after label/train-mask filtering."},
    )
    graph_task: str = field(default="nc")
    graph_neg_k: int = field(default=1)
    max_train_samples: int = field(default=-1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    graph_candidate_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Loss weight for graph candidate-text likelihood ranking."},
    )
    graph_lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Loss weight for the standard supervised LM objective."},
    )
    graph_candidate_score_reduction: str = field(
        default="mean",
        metadata={"help": "Reduce candidate answer token log-likelihoods by mean or sum."},
    )
    graph_candidate_batch_size: int = field(
        default=0,
        metadata={"help": "Flat candidate batch size for graph likelihood ranking. Use 0 for all candidates at once."},
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError:
        if hasattr(param, "detach"):
            return param.detach().cpu().clone()
        return param
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = [
        'mm_projector',
        'graph_projector',
        'graph_soft_token_encoder',
        'vision_tower',
        'vision_resampler',
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_graph_projector_only", False):
        keys_to_match = ['graph_soft_token_encoder', 'graph_projector']
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = [
            'mm_projector',
            'graph_projector',
            'graph_soft_token_encoder',
        ]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_graph: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    has_special_token = has_image or has_graph
    if has_special_token:
        input_ids = torch.stack([tokenizer_graph_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_special_token:
                round_len = len(tokenizer_graph_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_graph: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    has_special_token = has_image or has_graph
    if has_special_token:
        input_ids = torch.stack([tokenizer_graph_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_special_token:
                round_len = len(tokenizer_graph_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_graph: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    has_special_token = has_image or has_graph
    if has_special_token:
        input_ids = torch.stack([tokenizer_graph_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_special_token:
                round_len = len(tokenizer_graph_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    tokenizer_fn = tokenizer_graph_image_token if has_graph else tokenizer_image_token
    input_ids = [tokenizer_fn(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_fn(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_graph: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image, has_graph=has_graph)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, has_graph=has_graph)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image, has_graph=has_graph)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    tokenizer_fn = tokenizer_graph_image_token if (has_image or has_graph) else None

    def get_tokenize_len(prompts):
        return [len(tokenizer_fn(prompt, tokenizer)) for prompt in prompts]

    if has_image or has_graph:
        input_ids = [tokenizer_fn(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image or has_graph:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_path = data_path
        self.image_archive = resolve_graph_path(data_args.graph_data_root, data_args.graph_image_archive, "ele-fashion-images.tar")
        self.image_archive_prefix = os.path.splitext(os.path.basename(self.image_archive))[0] if self.image_archive else None
        self._image_tar = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            graph_tokens = sample.get("graph_num_nodes", 128 if "graph_feat_path" in sample else 0)
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens + graph_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample or 'graph_feat_path' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def _get_image_tar(self):
        if self.image_archive is None:
            return None
        if self._image_tar is None:
            self._image_tar = tarfile.open(self.image_archive, "r")
        return self._image_tar

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_tar = None
            if self.image_archive is not None:
                folder_candidate = None if image_folder is None else os.path.join(image_folder, image_file)
                if folder_candidate is None or not os.path.exists(folder_candidate):
                    image_tar = self._get_image_tar()
            image = load_image_from_folder_or_archive(
                image_folder=image_folder,
                image_file=image_file,
                image_processor=processor,
                image_aspect_ratio=self.data_args.image_aspect_ratio,
                image_archive=self.image_archive,
                image_tar=image_tar,
                archive_prefix=self.image_archive_prefix,
            )
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        has_graph = "graph_feat_path" in self.list_data_dict[i]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            has_graph=has_graph)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if has_graph:
            graph_record = self.list_data_dict[i]
            graph_path = resolve_sample_graph_path(
                self.data_args,
                self.data_path,
                graph_record["graph_feat_path"],
            )
            data_dict.update(load_sample_graph(graph_path, self.tokenizer, graph_record.get("graph_key"), self.data_args.graph_feat_path))

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


def _build_candidate_likelihood_sequence(
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    full_ids = torch.cat([prompt_ids.cpu(), answer_ids.cpu()], dim=0)
    full_labels = torch.full_like(full_ids, IGNORE_INDEX)
    full_labels[-answer_ids.numel():] = answer_ids.cpu()
    if max_length > 0 and full_ids.numel() > max_length:
        special_positions = torch.where((prompt_ids.cpu() == GRAPH_TOKEN_INDEX) | (prompt_ids.cpu() == IMAGE_TOKEN_INDEX))[0]
        prefix_end = int(special_positions.max().item()) + 1 if special_positions.numel() > 0 else 1
        prefix_end = min(prefix_end, int(prompt_ids.numel()))
        answer_len = int(answer_ids.numel())
        if answer_len >= max_length:
            full_ids = full_ids[-max_length:]
            full_labels = full_labels[-max_length:]
            return full_ids, full_labels
        prompt_tail_budget = max_length - prefix_end - answer_len
        tail_start = max(prefix_end, int(prompt_ids.numel()) - max(0, prompt_tail_budget))
        kept_prompt = torch.cat([prompt_ids[:prefix_end].cpu(), prompt_ids[tail_start:].cpu()], dim=0)
        if kept_prompt.numel() + answer_len > max_length:
            keep_tail = max(0, max_length - answer_len - prefix_end)
            kept_prompt = torch.cat([kept_prompt[:prefix_end], kept_prompt[-keep_tail:].cpu()], dim=0) if keep_tail > 0 else kept_prompt[: max_length - answer_len]
        full_ids = torch.cat([kept_prompt, answer_ids.cpu()], dim=0)
        full_labels = torch.full_like(full_ids, IGNORE_INDEX)
        full_labels[-answer_len:] = answer_ids.cpu()
    return full_ids, full_labels


class LazyGraphSupervisedDataset(Dataset):
    """Graph-enhanced supervised dataset following the RGLM/LLaGA BFS serialization style."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super().__init__()
        text_path = data_args.graph_text_path
        if text_path is None and data_args.graph_data_root is not None:
            text_candidates = sorted(pathlib.Path(data_args.graph_data_root).glob("*raw-text.jsonl"))
            if len(text_candidates) == 1:
                text_path = str(text_candidates[0])
            else:
                text_path = resolve_graph_path(data_args.graph_data_root, None, "ele-fashion-raw-text.jsonl")
        feat_path = resolve_graph_path(data_args.graph_data_root, data_args.graph_feat_path, "clip_feat.pt")
        edge_path = resolve_graph_path(data_args.graph_data_root, data_args.graph_edge_path, "nc_edges-nodeid.pt")
        ppr_path = resolve_graph_path(data_args.graph_data_root, None, "ppr.pt")
        image_archive = resolve_graph_path(data_args.graph_data_root, data_args.graph_image_archive, "ele-fashion-images.tar")
        node_mapping_path = resolve_graph_path(data_args.graph_data_root, None, "node_mapping.pt")

        csv_text_path = None
        if text_path is None or not os.path.exists(text_path):
            csv_text_path = _find_graph_csv(data_args.graph_data_root)
            if csv_text_path is None:
                raise FileNotFoundError(f"Graph text path not found and no CSV fallback found: {text_path}")
        if feat_path is None or not os.path.exists(feat_path):
            raise FileNotFoundError(f"Graph feature path not found: {feat_path}")
        if csv_text_path is not None:
            self.records = _load_graph_records_from_csv(csv_text_path)
        else:
            with open(text_path, "r") as f:
                self.records = [json.loads(line) for line in f]

        graph_features = torch.load(feat_path, map_location="cpu", weights_only=False)
        if isinstance(graph_features, dict):
            if "features" in graph_features:
                graph_features = graph_features["features"]
            else:
                raise ValueError(f"Unsupported graph feature dict keys: {list(graph_features.keys())}")
        if not isinstance(graph_features, torch.Tensor):
            raise TypeError(f"Unsupported graph feature type: {type(graph_features)}")

        self.graph_features = graph_features.to(torch.float32)
        label_config_path = resolve_graph_path(data_args.graph_data_root, data_args.graph_label_config, "graph_label_config.json")
        self.graph_label_spec = load_graph_label_spec(label_config_path)
        self.label_hard_negative_order = _build_label_hard_negative_order(self.graph_label_spec)
        self.synthetic_context = None
        self.synthetic_dataset_name = None
        if data_args.graph_synthetic_root:
            self.synthetic_context = SyntheticGraphContext(data_args.graph_synthetic_root)
            self.synthetic_dataset_name = _infer_dataset_name(
                data_args.graph_data_root,
                list(self.synthetic_context.offsets.keys()),
            )
            if self.synthetic_dataset_name is None:
                raise ValueError(
                    f"Unable to match graph_data_root={data_args.graph_data_root} to synthetic datasets "
                    f"{list(self.synthetic_context.offsets.keys())}"
                )
        self.labels = None
        labels_path = resolve_graph_path(data_args.graph_data_root, None, "labels.pt")
        if labels_path is not None and os.path.exists(labels_path):
            self.labels = torch.load(labels_path, map_location="cpu", weights_only=False).to(torch.long)
        self.train_mask = None
        train_mask_path = resolve_graph_path(data_args.graph_data_root, None, "train_mask.pt")
        if train_mask_path is not None and os.path.exists(train_mask_path):
            self.train_mask = torch.load(train_mask_path, map_location="cpu", weights_only=False).bool()
        self.graph_cache_path = data_args.graph_cache_path
        if self.graph_cache_path is not None and self.graph_cache_path.strip() == "":
            self.graph_cache_path = False
        if self.graph_cache_path is None and data_args.graph_data_root is not None:
            self.graph_cache_path = os.path.join(
                data_args.graph_data_root,
                f"graph_ids_h{data_args.graph_hops}_s{data_args.graph_sample_size}.pt",
            )
        graph_cache_exists = bool(self.graph_cache_path) and os.path.exists(self.graph_cache_path)
        self.edge_list = None
        self.ppr = None
        if ppr_path is not None and os.path.exists(ppr_path):
            self.ppr = torch.load(ppr_path, map_location="cpu", weights_only=False)
        if not graph_cache_exists:
            if (edge_path is None or not os.path.exists(edge_path)) and self.ppr is None:
                raise FileNotFoundError(f"Graph edge path not found: {edge_path}")
            if edge_path is not None and os.path.exists(edge_path):
                edge_pairs = torch.load(edge_path, map_location="cpu", weights_only=False)
                if not isinstance(edge_pairs, (list, tuple, torch.Tensor)):
                    raise TypeError(f"Unsupported edge file type: {type(edge_pairs)}")
                self.edge_list = build_edge_list_from_pairs(edge_pairs, self.graph_features.shape[0])

        self.graph_seq_len = sum(data_args.graph_sample_size ** hop for hop in range(data_args.graph_hops + 1))
        self.image_archive = image_archive
        self.image_member_prefix = os.path.splitext(os.path.basename(image_archive))[0] if image_archive else None
        self._image_tar = None
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.node_id_by_asin = {}

        if node_mapping_path is not None and os.path.exists(node_mapping_path):
            raw_mapping = torch.load(node_mapping_path, map_location="cpu", weights_only=False)
            if isinstance(raw_mapping, dict):
                for key, value in raw_mapping.items():
                    if isinstance(value, dict):
                        value = value.get("node_id", value.get("id", value.get("idx")))
                    if value is not None:
                        self.node_id_by_asin[str(key)] = int(value)
            elif isinstance(raw_mapping, (list, tuple)):
                for idx, item in enumerate(raw_mapping):
                    if isinstance(item, str):
                        self.node_id_by_asin[item] = idx
                    elif isinstance(item, dict):
                        asin = item.get("asin") or item.get("id")
                        node_id = item.get("node_id", item.get("idx", idx))
                        if asin is not None:
                            self.node_id_by_asin[str(asin)] = int(node_id)

        self.graph_ids_cache = None
        if self.graph_cache_path:
            self.graph_ids_cache = self._load_or_build_graph_cache()
            # BFS serialization is already cached, so the adjacency list is no longer needed.
            self.edge_list = None

        self.graph_feat_cache_path = data_args.graph_feat_cache_path
        if self.graph_feat_cache_path is not None and self.graph_feat_cache_path.strip() == "":
            self.graph_feat_cache_path = False
        if self.graph_feat_cache_path is None and data_args.graph_data_root is not None:
            self.graph_feat_cache_path = os.path.join(
                data_args.graph_data_root,
                f"graph_feat_h{data_args.graph_hops}_s{data_args.graph_sample_size}_fp16.npy",
            )
        self.graph_feat_cache = None
        if self.graph_feat_cache_path and self.synthetic_context is None:
            self.graph_feat_cache = self._load_or_build_graph_feat_cache()
            # Cached graph features make the full CLIP feature matrix unnecessary at runtime.
            self.graph_features = None

        self.sample_indices = list(range(len(self.records)))
        self.option_label_orders: Dict[int, List[str]] = {}
        if self.data_args.graph_label_supervision and self.labels is not None:
            valid_id2label = self.graph_label_spec["valid_id2label"]
            valid_mask = torch.zeros(len(self.records), dtype=torch.bool)
            for label_id in valid_id2label:
                valid_mask |= self.labels.eq(label_id)
            if self.train_mask is not None:
                valid_mask &= self.train_mask
            self.sample_indices = torch.where(valid_mask)[0].tolist()
        effective_max_train_samples = self.data_args.max_train_samples
        if self.data_args.graph_train_sample_ratio > 0:
            if self.data_args.graph_train_sample_ratio > 1:
                raise ValueError(
                    f"graph_train_sample_ratio must be in (0, 1], got {self.data_args.graph_train_sample_ratio}"
                )
            ratio_samples = max(1, math.ceil(len(self.sample_indices) * self.data_args.graph_train_sample_ratio))
            effective_max_train_samples = (
                ratio_samples
                if effective_max_train_samples <= 0
                else min(effective_max_train_samples, ratio_samples)
            )
        if effective_max_train_samples > 0:
            if self.data_args.graph_stratified_label_subset and self.data_args.graph_label_supervision and self.labels is not None:
                self.sample_indices = select_label_covered_subset(
                    candidate_indices=self.sample_indices,
                    labels=self.labels,
                    valid_label_ids=sorted(self.graph_label_spec["valid_id2label"].keys()),
                    max_samples=effective_max_train_samples,
                    min_samples_per_label=self.data_args.graph_min_samples_per_label,
                    seed=self.data_args.graph_subset_seed,
                )
            else:
                self.sample_indices = self.sample_indices[:effective_max_train_samples]

        rank0_print(f"Formatting graph-enhanced {self.graph_label_spec['name']} inputs...Skip in lazy mode")

    def _option_labels_for_sample(self, sample_idx: int) -> Optional[List[str]]:
        if not self.data_args.graph_option_label_prompt:
            return None
        if sample_idx in self.option_label_orders:
            return self.option_label_orders[sample_idx]
        if not self.data_args.graph_shuffle_label_options:
            return None
        labels = list(self.graph_label_spec["candidate_labels"])
        rng = random.Random(int(self.data_args.graph_label_option_seed) + int(sample_idx))
        rng.shuffle(labels)
        return labels

    def _candidate_negative_labels(
        self,
        gt_label: str,
        sample_idx: int,
        all_candidate_labels: Sequence[str],
    ) -> List[str]:
        neg_labels = [label for label in all_candidate_labels if label != gt_label]
        max_negatives = int(self.data_args.graph_candidate_negatives)
        if max_negatives < 0:
            return neg_labels
        if max_negatives == 0:
            return []

        rng = random.Random(int(self.data_args.graph_candidate_seed) + int(sample_idx))
        strategy = str(getattr(self.data_args, "graph_candidate_negative_strategy", "random")).lower()
        if strategy == "random":
            rng.shuffle(neg_labels)
            return neg_labels[:max_negatives]

        hard_order = [
            label
            for label in self.label_hard_negative_order.get(gt_label, [])
            if label in neg_labels
        ]
        if strategy in {"semantic", "hard", "semantic_hard"}:
            selected = hard_order[:max_negatives]
            if len(selected) < max_negatives:
                leftovers = [label for label in neg_labels if label not in selected]
                rng.shuffle(leftovers)
                selected.extend(leftovers[: max_negatives - len(selected)])
            return selected

        if strategy in {"semantic_mix", "mix"}:
            hard_count = min(len(hard_order), max(1, (max_negatives + 1) // 2))
            selected = hard_order[:hard_count]
            leftovers = [label for label in neg_labels if label not in selected]
            rng.shuffle(leftovers)
            selected.extend(leftovers[: max_negatives - len(selected)])
            return selected

        raise ValueError(
            f"Unknown graph_candidate_negative_strategy={self.data_args.graph_candidate_negative_strategy!r}; "
            "expected random, semantic, or semantic_mix."
        )

    def _build_graph_ids_cache(self):
        graph_ids_list = []
        for idx, record in enumerate(self.records):
            node_id = self._resolve_node_id(record["asin"], idx)
            graph_ids = self._serialize_node_context(node_id)
            graph_ids_list.append(graph_ids)
        return torch.tensor(graph_ids_list, dtype=torch.int32)

    def _serialize_node_context(self, node_id: int) -> List[int]:
        target_len = self.graph_seq_len
        if self.ppr is not None:
            neighbors = self.ppr.get(node_id, self.ppr.get(str(node_id), [])) if isinstance(self.ppr, dict) else []
            if torch.is_tensor(neighbors):
                neighbors = neighbors.tolist()
            sequence = [int(node_id)]
            for nbr in neighbors:
                nbr = int(nbr)
                if nbr != node_id:
                    sequence.append(nbr)
                if len(sequence) >= target_len:
                    break
            return sequence + [DEFAULT_GRAPH_PAD_ID] * max(0, target_len - len(sequence))
        return get_fix_shape_subgraph_sequence_fast(
            self.edge_list,
            node_id,
            self.data_args.graph_hops,
            self.data_args.graph_sample_size,
        )

    def _load_or_build_graph_cache(self):
        if os.path.exists(self.graph_cache_path):
            rank0_print(f"Loading graph serialization cache from {self.graph_cache_path}")
            return torch.load(self.graph_cache_path, map_location="cpu", weights_only=False)

        rank0_print(f"Building graph serialization cache at {self.graph_cache_path}")
        graph_ids_cache = self._build_graph_ids_cache()
        cache_dir = os.path.dirname(self.graph_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_cache_path = self.graph_cache_path + ".tmp"
        torch.save(graph_ids_cache, tmp_cache_path)
        os.replace(tmp_cache_path, self.graph_cache_path)
        return graph_ids_cache

    def _build_graph_feat_cache(self):
        if self.graph_ids_cache is None:
            raise ValueError("graph_ids_cache must be available before building graph feature cache.")
        feat_dim = self.graph_features.shape[1]
        num_samples = len(self.records)
        tmp_cache_path = self.graph_feat_cache_path + ".tmp.npy"
        graph_feat_cache = np.lib.format.open_memmap(
            tmp_cache_path,
            mode="w+",
            dtype=np.float16,
            shape=(num_samples, self.graph_seq_len, feat_dim),
        )

        chunk_size = 256
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            graph_ids = self.graph_ids_cache[start:end].to(torch.long)
            graph_node_mask = graph_ids.ne(DEFAULT_GRAPH_PAD_ID)
            graph_targets = torch.zeros(
                (end - start, self.graph_seq_len, feat_dim),
                dtype=self.graph_features.dtype,
            )
            graph_targets[graph_node_mask] = self.graph_features[graph_ids[graph_node_mask]]
            graph_feat_cache[start:end] = graph_targets.to(torch.float16).cpu().numpy()

        graph_feat_cache.flush()
        os.replace(tmp_cache_path, self.graph_feat_cache_path)
        return np.load(self.graph_feat_cache_path, mmap_mode="r")

    def _load_or_build_graph_feat_cache(self):
        if os.path.exists(self.graph_feat_cache_path):
            rank0_print(f"Loading graph feature cache from {self.graph_feat_cache_path}")
            return np.load(self.graph_feat_cache_path, mmap_mode="r")

        rank0_print(f"Building graph feature cache at {self.graph_feat_cache_path}")
        cache_dir = os.path.dirname(self.graph_feat_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        return self._build_graph_feat_cache()

    def __len__(self):
        return len(self.sample_indices)

    @property
    def lengths(self):
        length_list = []
        for sample_idx in self.sample_indices:
            record = self.records[sample_idx]
            text_len = len(self._get_record_text(record).split())
            image_token_budget = 0 if self.data_args.graph_only else 128
            length_list.append(text_len + image_token_budget + self.graph_seq_len)
        return length_list

    @property
    def modality_lengths(self):
        return self.lengths

    def _get_image_tar(self):
        if self.image_archive is None:
            return None
        if self._image_tar is None:
            self._image_tar = tarfile.open(self.image_archive, "r")
        return self._image_tar

    def _resolve_node_id(self, asin, fallback_idx):
        return self.node_id_by_asin.get(str(asin), fallback_idx)

    def _get_record_text(self, record):
        raw_text = record.get("raw_text", "")
        if isinstance(raw_text, list):
            raw_text = " ".join(str(x) for x in raw_text if x is not None)
        return html.unescape(str(raw_text)).strip()

    def _record_text_by_node(self, node_id: int, max_chars: int = 220) -> str:
        if node_id < 0 or node_id >= len(self.records):
            return ""
        return _clean_graph_text(self._get_record_text(self.records[node_id]), max_chars=max_chars)

    def _node_modality_relation(
        self,
        center_emb: torch.Tensor,
        node_emb: torch.Tensor,
        thresholds: Optional[Dict[str, float]],
    ) -> Tuple[str, str, float, float]:
        half = center_emb.numel() // 2
        if half <= 0:
            return "structural-context", "structurally close context node.", 0.0, 0.0
        c_text = F.normalize(center_emb[:half].float(), dim=0)
        c_image = F.normalize(center_emb[half:half * 2].float(), dim=0)
        n_text = F.normalize(node_emb[:half].float(), dim=0)
        n_image = F.normalize(node_emb[half:half * 2].float(), dim=0)
        text_sim = float((c_text * n_text).sum().item())
        image_sim = float((c_image * n_image).sum().item())
        relation, description = _modality_relation_from_sims(text_sim, image_sim, thresholds)
        return relation, description, text_sim, image_sim

    def _select_relation_neighbors(
        self,
        graph_ids: torch.Tensor,
        graph_targets: torch.Tensor,
        center_node_id: int,
    ) -> Dict[str, List[Dict[str, object]]]:
        thresholds = None
        if self.synthetic_context is not None and self.synthetic_dataset_name is not None:
            thresholds = self.synthetic_context.thresholds_by_dataset.get(self.synthetic_dataset_name)
        center_emb = graph_targets[0]
        buckets = {"modality-consistent": [], "modality-complement": [], "modality-conflict": [], "structural-context": []}
        seen = {int(center_node_id)}
        for pos in range(1, graph_ids.numel()):
            node_id = int(graph_ids[pos].item())
            if node_id == DEFAULT_GRAPH_PAD_ID or node_id in seen or node_id < 0 or node_id >= len(self.records):
                continue
            seen.add(node_id)
            relation, desc, text_sim, image_sim = self._node_modality_relation(center_emb, graph_targets[pos], thresholds)
            score = text_sim + image_sim
            if relation == "modality-complement":
                score = abs(text_sim - image_sim) + 0.5 * max(text_sim, image_sim)
            if relation not in buckets:
                relation = "structural-context"
            buckets[relation].append(
                {
                    "node_id": node_id,
                    "relation": relation,
                    "description": desc,
                    "text_similarity": text_sim,
                    "image_similarity": image_sim,
                    "score": score,
                    "text": self._record_text_by_node(node_id),
                }
            )
        for values in buckets.values():
            values.sort(key=lambda item: item["score"], reverse=True)
        return buckets

    def _weighted_average_features(
        self,
        weighted_features: Sequence[Tuple[torch.Tensor, float]],
        feat_dim: int,
    ) -> torch.Tensor:
        if not weighted_features:
            return torch.zeros(feat_dim, dtype=torch.float32)
        total = sum(max(float(weight), 0.0) for _, weight in weighted_features)
        if total <= 0:
            total = float(len(weighted_features))
            weighted_features = [(feat, 1.0) for feat, _ in weighted_features]
        out = torch.zeros(feat_dim, dtype=torch.float32)
        for feat, weight in weighted_features:
            out = out + feat.to(torch.float32) * (max(float(weight), 0.0) / total)
        return out

    @staticmethod
    def _split_modal_feature(feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        half = int(feat.numel() // 2)
        return feat[:half].to(torch.float32), feat[half: half * 2].to(torch.float32)

    @staticmethod
    def _node_level(node: Optional[Dict[str, object]]) -> int:
        if node is None:
            return 0
        if "layer_id" in node:
            try:
                return int(node.get("layer_id", 0))
            except Exception:
                return 0
        node_type = str(node.get("node_type", "")).lower()
        if "global" in node_type:
            return 3
        if "highlevel" in node_type or "layer2" in node_type:
            return 2
        if "abstract" in node_type:
            return 1
        return 0

    @staticmethod
    def _node_coord_key(node: Optional[Dict[str, object]]) -> Optional[Tuple[str, int]]:
        if node is None:
            return None
        if "global_node_id" not in node:
            return None
        return (str(node.get("dataset", "")), int(node.get("global_node_id")))

    @staticmethod
    def _node_global_alignment_key(
        synthetic: Optional["SyntheticGraphContext"],
        node: Optional[Dict[str, object]],
    ) -> Optional[int]:
        if synthetic is None or node is None:
            return None
        level = LazyGraphSupervisedDataset._node_level(node)
        node_id = int(node.get("global_node_id", -1))
        if level >= 3:
            return node_id
        if level == 2:
            return synthetic.layer2_to_global.get(node_id)
        return node_id

    @staticmethod
    def _node_label(node: Optional[Dict[str, object]], max_chars: int = 45) -> str:
        if node is None:
            return "none"
        label = node.get("semantic_label", node.get("summary_text", node.get("id", "")))
        return _clean_graph_text(label, max_chars=max_chars)

    @staticmethod
    def _coord_source_bonus(source: str) -> float:
        source = str(source).lower()
        if source == "membership":
            return 1.25
        if source == "ppr":
            return 1.10
        if source == "bridge":
            return 1.15
        if source in {"text", "image"}:
            return 1.0
        return 0.9

    def _ppr_neighbor_ids(self, center_node_id: int, limit: int) -> List[Tuple[int, float]]:
        if self.ppr is None or limit <= 0:
            return []
        neighbors = self.ppr.get(center_node_id, self.ppr.get(str(center_node_id), [])) if isinstance(self.ppr, dict) else []
        if torch.is_tensor(neighbors):
            neighbors = neighbors.tolist()
        out = []
        seen = {int(center_node_id)}
        for rank, item in enumerate(neighbors):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                nid = int(item[0])
                raw_score = float(item[1])
                score = raw_score if raw_score > 0 else 1.0 / float(rank + 2)
            else:
                nid = int(item)
                score = 1.0 / float(rank + 2)
            if nid in seen or nid < 0 or nid >= len(self.records):
                continue
            seen.add(nid)
            out.append((nid, score))
            if len(out) >= limit:
                break
        return out

    def _coord_membership_support(
        self,
        coords: Sequence[Dict[str, object]],
        synthetic: Optional["SyntheticGraphContext"],
    ) -> Dict[Tuple[str, int], float]:
        support: Dict[Tuple[str, int], float] = {}
        for rank, item in enumerate(coords):
            node = item.get("node")
            if node is None:
                continue
            keys = {self._node_coord_key(node)}
            global_key = self._node_global_alignment_key(synthetic, node)
            if global_key is not None:
                keys.add(("global", int(global_key)))
            weight = max(float(item.get("score", 0.0)), 0.0) / float(rank + 1)
            for key in keys:
                if key is None:
                    continue
                support[key] = max(support.get(key, 0.0), weight)
        return support

    def _top_abstract_coords(
        self,
        center_feat: torch.Tensor,
        dataset_name: str,
        modality: str,
        topk: int,
    ) -> List[Dict[str, object]]:
        synthetic = self.synthetic_context
        if synthetic is None or dataset_name is None:
            return []
        cache = synthetic.abstract_coord_candidates(dataset_name)
        if cache is None:
            return []
        text_feat, image_feat = self._split_modal_feature(center_feat)
        query = text_feat if modality == "text" else image_feat
        query = F.normalize(query.to(torch.float32), dim=0)
        bank = cache["text"] if modality == "text" else cache["image"]
        scores = torch.mv(bank, query)
        k = min(max(int(topk), 1), scores.numel())
        top_scores, top_idx = torch.topk(scores, k=k)
        out = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            node = cache["nodes"][int(idx)]
            out.append(
                {
                    "node": node,
                    "score": float(score),
                    "source": modality,
                    "level": str(node.get("node_type", node.get("layer_id", "abstract"))),
                }
            )
        return out

    def _graph_coord_candidates(
        self,
        center_node_id: int,
        center_feat: torch.Tensor,
        dataset_name: str,
        membership: Dict[str, object],
        graph_ids: torch.Tensor,
        graph_targets: torch.Tensor,
        topk: int,
    ) -> List[Dict[str, object]]:
        synthetic = self.synthetic_context
        if synthetic is None or dataset_name is None:
            return []
        feat_dim = int(center_feat.numel())
        coords = []
        for weight, key in [(1.0, "layer1"), (1.15, "layer2"), (1.3, "global")]:
            node = membership.get(key)
            feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
            if node is not None and feat is not None:
                sim = float(F.cosine_similarity(center_feat, feat.to(torch.float32), dim=0).item())
                coords.append(
                    {
                        "node": node,
                        "score": weight + 0.25 * sim,
                        "source": "membership",
                        "path": key,
                        "ppr_score": 1.0,
                        "level": self._node_level(node),
                    }
                )

        ppr_neighbors = self._ppr_neighbor_ids(center_node_id, max(8, int(topk) * 8))
        if not ppr_neighbors:
            ppr_neighbors = []
            seen = {int(center_node_id)}
            max_neighbors = min(int(graph_ids.numel()), max(2, int(topk) * 4 + 2))
            for rank in range(1, max_neighbors):
                nid = int(graph_ids[rank].item())
                if nid == DEFAULT_GRAPH_PAD_ID or nid in seen or nid < 0:
                    continue
                seen.add(nid)
                ppr_neighbors.append((nid, 1.0 / float(rank + 1)))

        for rank, (nid, ppr_score) in enumerate(ppr_neighbors):
            try:
                n_membership = synthetic.membership(dataset_name, nid)
            except Exception:
                continue
            node_candidates = [n_membership.get("layer2"), n_membership.get("layer1")]
            neighbor_feat = None
            if self.graph_features is not None:
                neighbor_feat = self.graph_features[nid].to(torch.float32)
            for node in node_candidates:
                feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
                if node is None or feat is None:
                    continue
                node_sim = float(F.cosine_similarity(center_feat, feat.to(torch.float32), dim=0).item())
                local_sim = (
                    float(F.cosine_similarity(center_feat, neighbor_feat, dim=0).item())
                    if neighbor_feat is not None
                    else node_sim
                )
                level_bonus = 1.2 if self._node_level(node) >= 2 else 1.0
                reachability = max(float(ppr_score), 0.0)
                coords.append(
                    {
                        "node": node,
                        "score": level_bonus * (reachability + 0.08 * max(node_sim, 0.0) + 0.05 * max(local_sim, 0.0)),
                        "source": "ppr",
                        "path": f"ppr_rank_{rank + 1}",
                        "ppr_score": reachability,
                        "neighbor_id": int(nid),
                        "level": self._node_level(node),
                    }
                )

        for bridge in synthetic.bridge_neighbors(dataset_name, center_node_id, max(0, int(topk))):
            if (
                self.data_args.graph_min_bridge_confidence >= -1.0
                and float(bridge.get("confidence", 0.0)) < self.data_args.graph_min_bridge_confidence
            ):
                continue
            dst = int(bridge["dst"])
            try:
                b_membership = synthetic.membership(dataset_name, dst)
            except Exception:
                continue
            node = b_membership.get("layer2") or b_membership.get("layer1")
            if node is not None:
                coords.append(
                    {
                        "node": node,
                        "score": float(bridge.get("confidence", 0.0)) + 0.25 * float(bridge.get("structure_consistency_score", 0.0)),
                        "source": "bridge",
                        "path": str(bridge.get("direction", "bridge")),
                        "ppr_score": 0.0,
                        "neighbor_id": int(dst),
                        "level": self._node_level(node),
                    }
                )

        merged: Dict[Tuple[str, int], Dict[str, object]] = {}
        for item in coords:
            key = self._node_coord_key(item.get("node"))
            if key is None:
                continue
            if key not in merged or float(item.get("score", 0.0)) > float(merged[key].get("score", 0.0)):
                merged[key] = item
        coords = list(merged.values())
        coords.sort(key=lambda item: float(item["score"]), reverse=True)
        return coords[: max(int(topk), 1)]

    def _coord_overlap_score(self, a: Sequence[Dict[str, object]], b: Sequence[Dict[str, object]]) -> float:
        if not a or not b:
            return 0.0
        ids_a = {int(item["node"].get("global_node_id", -1)) for item in a if item.get("node") is not None}
        ids_b = {int(item["node"].get("global_node_id", -2)) for item in b if item.get("node") is not None}
        if not ids_a or not ids_b:
            return 0.0
        return len(ids_a & ids_b) / float(min(len(ids_a), len(ids_b)))

    def _build_semantic_coordinate_graph(
        self,
        center_node_id: int,
        center_record: Dict[str, object],
        graph_ids: torch.Tensor,
        graph_targets: torch.Tensor,
        option_labels: Optional[Sequence[str]] = None,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        subject_name = self.graph_label_spec["subject_name"]
        center_text = _clean_graph_text(self._get_record_text(center_record), max_chars=150)
        compact_labels = _format_compact_label_space(
            self.graph_label_spec,
            option_labels=option_labels,
            option_prompt=self.data_args.graph_option_label_prompt,
        )
        synthetic = self.synthetic_context
        dataset_name = self.synthetic_dataset_name
        feat_dim = int(graph_targets.shape[-1])
        center_feat = graph_targets[0].to(torch.float32)
        text_feat, image_feat = self._split_modal_feature(center_feat)
        zero_half = torch.zeros_like(text_feat)
        topk = max(1, int(self.data_args.graph_semantic_coord_topk))
        graph_topk = max(1, int(self.data_args.graph_semantic_coord_graph_topk))
        transfer_topk = max(1, int(self.data_args.graph_semantic_coord_transfer_topk))

        membership = synthetic.membership(dataset_name, center_node_id) if synthetic is not None and dataset_name is not None else {}
        text_coords = self._top_abstract_coords(center_feat, dataset_name, "text", topk)
        image_coords = self._top_abstract_coords(center_feat, dataset_name, "image", topk)
        graph_coords = self._graph_coord_candidates(
            center_node_id=center_node_id,
            center_feat=center_feat,
            dataset_name=dataset_name,
            membership=membership,
            graph_ids=graph_ids,
            graph_targets=graph_targets,
            topk=graph_topk,
        )

        abstract_entries: Dict[Tuple[str, int], Dict[str, object]] = {}
        for view_name, coords, view_weight in [
            ("text", text_coords, 1.0),
            ("image", image_coords, 1.0),
            ("graph", graph_coords, 1.2),
        ]:
            for rank, item in enumerate(coords):
                node = item.get("node")
                level = self._node_level(node)
                if level not in {1, 2}:
                    continue
                feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
                key = self._node_coord_key(node)
                if node is None or feat is None or key is None:
                    continue
                raw_score = max(float(item.get("score", 0.0)), 0.0)
                source_bonus = self._coord_source_bonus(str(item.get("source", view_name)))
                weight = view_weight * source_bonus * raw_score / float(rank + 1)
                if key not in abstract_entries:
                    abstract_entries[key] = {
                        "node": node,
                        "feat": feat,
                        "views": set(),
                        "score": 0.0,
                        "ppr": 0.0,
                        "paths": [],
                    }
                entry = abstract_entries[key]
                entry["views"].add(view_name)
                entry["score"] = float(entry["score"]) + weight
                entry["ppr"] = max(float(entry["ppr"]), float(item.get("ppr_score", 0.0)))
                entry["paths"].append(f"{view_name}:{item.get('path', item.get('source', view_name))}")

        weighted_l1 = []
        weighted_l2 = []
        abstract_lines = []
        for entry in abstract_entries.values():
            node = entry["node"]
            views = set(entry["views"])
            level = self._node_level(node)
            agreement_bonus = 1.0 + 0.35 * max(0, len(views) - 1)
            if "graph" in views and ("text" in views or "image" in views):
                agreement_bonus += 0.30
            if level == 2:
                agreement_bonus += 0.20
            ppr_bonus = 1.0 + 0.25 * min(float(entry["ppr"]), 1.0)
            weight = float(entry["score"]) * agreement_bonus * ppr_bonus
            if level == 1:
                weighted_l1.append((entry["feat"], weight))
            elif level == 2:
                weighted_l2.append((entry["feat"], weight))
            abstract_lines.append(
                (
                    weight,
                    f"L{level}:{self._node_label(node, 48)} "
                    f"views={'+'.join(sorted(views))} w={weight:.2f}"
                )
            )
        if not weighted_l1 and not weighted_l2 and membership:
            for key, weight in [("layer2", 1.2), ("layer1", 1.0)]:
                node = membership.get(key)
                feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
                if node is not None and feat is not None:
                    level = self._node_level(node)
                    if level == 1:
                        weighted_l1.append((feat, weight))
                    else:
                        weighted_l2.append((feat, weight))
                    abstract_lines.append((weight, f"L{level}:{self._node_label(node, 48)} views=membership"))
        # graph_ablate_layer: 1=remove layer1 token, 2=remove layer2 token, 4=remove transfer token
        _ablate_layer = int(getattr(self.data_args, 'graph_ablate_layer', -1))
        layer1_token = self._weighted_average_features(weighted_l1 if _ablate_layer != 1 else [], feat_dim)
        layer2_token = self._weighted_average_features(weighted_l2 if _ablate_layer != 2 else [], feat_dim)
        abstract_token = self._weighted_average_features(weighted_l1 + weighted_l2, feat_dim)

        text_support = self._coord_membership_support(text_coords, synthetic)
        image_support = self._coord_membership_support(image_coords, synthetic)
        graph_support = self._coord_membership_support(graph_coords, synthetic)

        def support_agreement(a: Dict[Tuple[str, int], float], b: Dict[Tuple[str, int], float]) -> float:
            if not a or not b:
                return 0.0
            shared = set(a.keys()) & set(b.keys())
            if not shared:
                return 0.0
            numerator = sum(min(float(a[k]), float(b[k])) for k in shared)
            denominator = min(sum(max(float(v), 0.0) for v in a.values()), sum(max(float(v), 0.0) for v in b.values()))
            return max(0.0, min(1.0, numerator / max(denominator, 1e-6)))

        def support_for_node(node: Optional[Dict[str, object]], support: Dict[Tuple[str, int], float]) -> float:
            if node is None or not support:
                return 0.0
            keys = {self._node_coord_key(node)}
            global_key = self._node_global_alignment_key(synthetic, node)
            if global_key is not None:
                keys.add(("global", int(global_key)))
            return max([float(support.get(key, 0.0)) for key in keys if key is not None] or [0.0])

        text_graph_agree = support_agreement(text_support, graph_support)
        image_graph_agree = support_agreement(image_support, graph_support)
        text_image_agree = support_agreement(text_support, image_support)

        bridge_text = 0.0
        bridge_image = 0.0
        bridge_text_weighted = []
        bridge_image_weighted = []
        bridge_lines = []
        include_cross_domain = getattr(self.data_args, 'graph_include_cross_domain_soft_tokens', True)
        # ablate_layer=4 means remove transfer/global: also skip bridge (cross-domain) neighbors
        if _ablate_layer == 4:
            include_cross_domain = False
        if synthetic is not None and dataset_name is not None and include_cross_domain:
            bridge_limit = max(self.data_args.graph_max_complement_neighbors, graph_topk)
            for bridge in synthetic.bridge_neighbors(dataset_name, center_node_id, bridge_limit):
                confidence = float(bridge.get("confidence", 0.0))
                if (
                    self.data_args.graph_min_bridge_confidence >= -1.0
                    and confidence < self.data_args.graph_min_bridge_confidence
                ):
                    continue
                dst = int(bridge["dst"])
                try:
                    b_membership = synthetic.membership(dataset_name, dst)
                except Exception:
                    continue
                b_nodes = [b_membership.get("layer2"), b_membership.get("layer1")]
                text_align = max(support_for_node(node, text_support) for node in b_nodes)
                image_align = max(support_for_node(node, image_support) for node in b_nodes)
                structure_score = float(bridge.get("structure_consistency_score", 0.0))
                direction = str(bridge.get("direction", ""))
                text_like = direction.startswith("text") or float(bridge.get("text_similarity", 0.0)) >= float(bridge.get("image_similarity", 0.0))
                image_like = direction.startswith("image") or float(bridge.get("image_similarity", 0.0)) > float(bridge.get("text_similarity", 0.0))
                for node in b_nodes:
                    feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
                    if node is None or feat is None:
                        continue
                    if text_like:
                        bridge_text_weighted.append((feat, confidence * (1.0 + text_align + 0.25 * structure_score)))
                    if image_like:
                        bridge_image_weighted.append((feat, confidence * (1.0 + image_align + 0.25 * structure_score)))
                if text_like:
                    bridge_text = max(bridge_text, min(1.0, confidence * (1.0 + text_align + 0.25 * structure_score)))
                if image_like:
                    bridge_image = max(bridge_image, min(1.0, confidence * (1.0 + image_align + 0.25 * structure_score)))
                bridge_lines.append(
                    (
                        confidence,
                        f"{direction or 'bridge'} dst={dst} conf={confidence:.2f} "
                        f"t_align={text_align:.2f} i_align={image_align:.2f}"
                    )
                )

        graph_mass = min(1.0, sum(max(float(item.get("ppr_score", 0.0)), 0.0) for item in graph_coords))
        text_reliability = max(0.05, min(1.0, 0.25 + 0.38 * text_graph_agree + 0.17 * text_image_agree + 0.25 * bridge_text))
        image_reliability = max(0.05, min(1.0, 0.25 + 0.38 * image_graph_agree + 0.17 * text_image_agree + 0.25 * bridge_image))
        structure_reliability = max(
            0.05,
            min(1.0, 0.30 + 0.30 * max(text_graph_agree, image_graph_agree) + 0.20 * graph_mass + 0.20 * min(text_graph_agree, image_graph_agree)),
        )
        conflict = max(
            0.0,
            min(1.0, (1.0 - text_image_agree) * (1.0 - 0.5 * max(text_graph_agree, image_graph_agree))),
        )
        abstract_text, abstract_image = self._split_modal_feature(abstract_token)
        bridge_text_token = self._weighted_average_features(bridge_text_weighted, feat_dim)
        bridge_image_token = self._weighted_average_features(bridge_image_weighted, feat_dim)
        bridge_text_half, _ = self._split_modal_feature(bridge_text_token)
        _, bridge_image_half = self._split_modal_feature(bridge_image_token)
        cal_text_parts = [(text_feat, text_reliability), (abstract_text, structure_reliability)]
        cal_image_parts = [(image_feat, image_reliability), (abstract_image, structure_reliability)]
        if bridge_text > 0:
            cal_text_parts.append((bridge_text_half, bridge_text))
        if bridge_image > 0:
            cal_image_parts.append((bridge_image_half, bridge_image))
        cal_text_total = sum(weight for _, weight in cal_text_parts)
        cal_image_total = sum(weight for _, weight in cal_image_parts)
        calibration_text = sum(feat * (weight / max(cal_text_total, 1e-6)) for feat, weight in cal_text_parts)
        calibration_image = sum(feat * (weight / max(cal_image_total, 1e-6)) for feat, weight in cal_image_parts)
        calibration_token = torch.cat([calibration_text, calibration_image], dim=0)

        transfer_weighted = []
        transfer_lines = []
        if synthetic is not None and dataset_name is not None:
            layer2_entries: Dict[int, Dict[str, object]] = {}
            for item in graph_coords + text_coords + image_coords:
                node = item.get("node")
                if self._node_level(node) != 2:
                    continue
                node_id = int(node.get("global_node_id", -1))
                score = max(float(item.get("score", 0.0)), 0.0) * self._coord_source_bonus(str(item.get("source", "")))
                if node_id not in layer2_entries or score > float(layer2_entries[node_id].get("score", 0.0)):
                    layer2_entries[node_id] = {"node": node, "score": score, "path": item.get("path", item.get("source", ""))}
            if not layer2_entries and membership.get("layer2") is not None:
                node = membership["layer2"]
                layer2_entries[int(node.get("global_node_id", -1))] = {"node": node, "score": 1.0, "path": "membership"}

            transfer_candidates: Dict[int, Tuple[torch.Tensor, float, str]] = {}
            for entry in layer2_entries.values():
                target_node = entry["node"]
                target_feat = SyntheticGraphContext.abstract_feature(target_node, feat_dim)
                if target_feat is None:
                    continue
                global_id = synthetic.layer2_to_global.get(int(target_node.get("global_node_id", -1)))
                global_node = synthetic.global_by_id.get(int(global_id)) if global_id is not None else None
                global_feat = SyntheticGraphContext.abstract_feature(global_node, feat_dim)
                source_nodes = synthetic.layer2_source_nodes_for_global(int(global_id), dataset_name) if global_id is not None else []
                fallback_used = False
                if not source_nodes:
                    fallback_used = True
                    scored_sources = []
                    for source_dataset, nodes in synthetic.layer2_by_dataset.items():
                        if source_dataset == dataset_name:
                            continue
                        for source_node in nodes:
                            source_feat = SyntheticGraphContext.abstract_feature(source_node, feat_dim)
                            if source_feat is None:
                                continue
                            sim = float(F.cosine_similarity(target_feat.to(torch.float32), source_feat.to(torch.float32), dim=0).item())
                            scored_sources.append((sim, source_node))
                    scored_sources.sort(key=lambda item: item[0], reverse=True)
                    source_nodes = [node for _, node in scored_sources[: max(transfer_topk * 2, 6)]]
                target_global_strength = (
                    1.0 + max(float(F.cosine_similarity(target_feat.to(torch.float32), global_feat.to(torch.float32), dim=0).item()), 0.0)
                    if global_feat is not None
                    else 1.0
                )
                for source_node in source_nodes:
                    source_feat = SyntheticGraphContext.abstract_feature(source_node, feat_dim)
                    if source_feat is None:
                        continue
                    target_source_sim = float(F.cosine_similarity(target_feat.to(torch.float32), source_feat.to(torch.float32), dim=0).item())
                    abstract_source_sim = float(F.cosine_similarity(abstract_token.to(torch.float32), source_feat.to(torch.float32), dim=0).item())
                    source_global_strength = (
                        1.0 + max(float(F.cosine_similarity(source_feat.to(torch.float32), global_feat.to(torch.float32), dim=0).item()), 0.0)
                        if global_feat is not None
                        else 1.0
                    )
                    path_weight = (
                        max(float(entry.get("score", 0.0)), 0.0)
                        * target_global_strength
                        * source_global_strength
                        * (0.5 + 0.30 * max(target_source_sim, 0.0) + 0.20 * max(abstract_source_sim, 0.0))
                    )
                    if global_feat is not None:
                        path_feat = 0.45 * source_feat + 0.35 * global_feat + 0.20 * target_feat
                    else:
                        path_feat = 0.65 * source_feat + 0.35 * target_feat
                    source_id = int(source_node.get("global_node_id", -1))
                    path_desc = (
                        f"{self._node_label(target_node, 36)} -> "
                        f"{self._node_label(global_node, 36) if global_node else 'similar-L2'} -> "
                        f"{source_node.get('dataset')}:{self._node_label(source_node, 36)}"
                        f"{' fallback' if fallback_used else ''}"
                    )
                    if source_id not in transfer_candidates or path_weight > transfer_candidates[source_id][1]:
                        transfer_candidates[source_id] = (path_feat, path_weight, path_desc)

            sorted_transfer = sorted(transfer_candidates.values(), key=lambda item: float(item[1]), reverse=True)
            for path_feat, path_weight, path_desc in sorted_transfer[:transfer_topk]:
                transfer_weighted.append((path_feat, path_weight))
                transfer_lines.append((path_weight, f"{path_desc} w={path_weight:.2f}"))
        transfer_token = self._weighted_average_features(transfer_weighted[:transfer_topk], feat_dim)

        if self.data_args.graph_residual_coord_tokens:
            max_ratio = float(self.data_args.graph_residual_max_ratio)
            cosine_gate = bool(self.data_args.graph_residual_cosine_gate)
            center_concat = torch.cat([text_feat, image_feat], dim=0)

            def _bounded_residual(delta, base, max_r, use_cos_gate):
                delta_f = delta.float()
                base_f = base.float()
                if use_cos_gate:
                    cos_sim = F.cosine_similarity(delta_f.unsqueeze(0), base_f.unsqueeze(0)).item()
                    if cos_sim < 0:
                        delta_f = delta_f * 0.05
                    else:
                        delta_f = delta_f * (0.3 + 0.7 * cos_sim)
                delta_norm = delta_f.norm().clamp(min=1e-8)
                max_norm = max_r * base_f.norm().clamp(min=1e-8)
                if delta_norm > max_norm:
                    delta_f = delta_f * (max_norm / delta_norm)
                return delta_f.to(dtype=delta.dtype)

            l1_delta = _bounded_residual(layer1_token - center_concat, center_concat, max_ratio, cosine_gate)
            l2_delta = _bounded_residual(layer2_token - center_concat, center_concat, max_ratio, cosine_gate)
            cal_delta = _bounded_residual(cal_full - abstract_token, center_concat, max_ratio * 0.5, cosine_gate)
            graph_emb = torch.stack(
                [
                    enhanced_text,
                    enhanced_image,
                    l1_delta,
                    l2_delta,
                    cal_delta,
                    trans_delta,
                ],
                dim=0,
            ).to(torch.float32)
            graph_ids = torch.tensor([center_node_id, center_node_id, -2, -5, -3, -4], dtype=torch.long)
            graph_hops = torch.tensor([0, 1, 2, 5, 3, 4], dtype=torch.long)
        else:
            _ablate_transfer = (_ablate_layer == 4)
            graph_emb = torch.stack(
                [
                    torch.cat([text_feat, zero_half], dim=0),
                    torch.cat([zero_half, image_feat], dim=0),
                    layer1_token,
                    layer2_token,
                    calibration_token,
                    torch.zeros_like(transfer_token) if _ablate_transfer else transfer_token,
                ],
                dim=0,
            ).to(torch.float32)
            graph_ids = torch.tensor([center_node_id, center_node_id, -2, -5, -3, -4], dtype=torch.long)
            graph_hops = torch.tensor([0, 1, 2, 5, 3, 4], dtype=torch.long)

        def coord_line(prefix: str, coords: Sequence[Dict[str, object]], limit: int = 3) -> str:
            lines = []
            for item in list(coords)[:limit]:
                node = item.get("node") or {}
                level = self._node_level(node)
                source = item.get("source", prefix)
                path = item.get("path", "")
                ppr_suffix = f", ppr={float(item.get('ppr_score', 0.0)):.3f}" if float(item.get("ppr_score", 0.0)) > 0 else ""
                lines.append(
                    f"L{level}:{_clean_graph_text(node.get('semantic_label', ''), 45)}"
                    f"({node.get('dataset', dataset_name)}, {source}/{path}, s={float(item.get('score', 0.0)):.2f}{ppr_suffix})"
                )
            return f"{prefix}: " + ("; ".join(lines) if lines else "none")

        abstract_lines.sort(key=lambda item: item[0], reverse=True)
        bridge_lines.sort(key=lambda item: item[0], reverse=True)
        transfer_lines.sort(key=lambda item: item[0], reverse=True)
        if self.data_args.graph_residual_coord_tokens:
            l1_lines = [(w, l) for w, l in abstract_lines if ":L1:" in l or l.startswith("L1:")]
            l2_lines = [(w, l) for w, l in abstract_lines if ":L2:" in l or l.startswith("L2:")]
            role_texts = [
                f"GraphCalibratedText: {center_text}",
                "GraphCalibratedImage: calibrated center image evidence.",
                "Layer1Token: " + ("; ".join(line for _, line in l1_lines[:topk]) if l1_lines else "none"),
                "Layer2Token: " + ("; ".join(line for _, line in l2_lines[:topk]) if l2_lines else "none"),
                "CalibrationDelta: " + f"text_graph={text_graph_agree:.2f}, image_graph={image_graph_agree:.2f}, reliability(text={text_reliability:.2f}, image={image_reliability:.2f}, structure={structure_reliability:.2f})",
                "TransferDelta: " + ("; ".join(line for _, line in transfer_lines[:transfer_topk]) if transfer_lines else "none"),
            ]
        else:
            l1_lines = [(w, l) for w, l in abstract_lines if ":L1:" in l or l.startswith("L1:")]
            l2_lines = [(w, l) for w, l in abstract_lines if ":L2:" in l or l.startswith("L2:")]
            role_texts = [
                f"CenterTextToken: text-only center evidence. Text: {center_text}",
                "CenterImageToken: image-only center evidence from offline visual CLIP feature.",
                "Layer1Token: " + ("; ".join(line for _, line in l1_lines[:topk]) if l1_lines else "no layer1 abstract evidence"),
                "Layer2Token: " + ("; ".join(line for _, line in l2_lines[:topk]) if l2_lines else "no layer2 abstract evidence"),
                (
                    "CalibrationToken: "
                    f"text_graph={text_graph_agree:.2f}, image_graph={image_graph_agree:.2f}, "
                    f"text_image={text_image_agree:.2f}, bridge_text={bridge_text:.2f}, bridge_image={bridge_image:.2f}, "
                    f"reliability(text={text_reliability:.2f}, image={image_reliability:.2f}, "
                    f"structure={structure_reliability:.2f}), conflict={conflict:.2f}. "
                    + ("; ".join(line for _, line in bridge_lines[:2]) if bridge_lines else "no strong bridge evidence")
                ),
                "TransferToken: "
                + ("; ".join(line for _, line in transfer_lines[:transfer_topk]) if transfer_lines else "no cross-dataset global path found"),
            ]
        _residual_mode = bool(self.data_args.graph_residual_coord_tokens)
        prompt_parts = [
            DEFAULT_GRAPH_TOKEN,
            f"Classify the center {subject_name}." if _residual_mode else f"Classify the center {subject_name} with enhanced semantic-coordinate graph tokens.",
            "" if _residual_mode else "Graph token order: CenterTextToken, CenterImageToken, Layer1Token, Layer2Token, CalibrationToken, TransferToken.",
            f"Center: {center_text}",
            coord_line("TextCoord", text_coords),
            coord_line("ImageCoord", image_coords),
            coord_line("GraphCoord", graph_coords),
            "AbstractEvidence: " + ("; ".join(line for _, line in abstract_lines[:topk]) if abstract_lines else "none"),
            (
                "Calibration: "
                f"text={text_reliability:.2f}, image={image_reliability:.2f}, "
                f"structure={structure_reliability:.2f}, conflict={conflict:.2f}."
            ),
        ]
        if transfer_lines:
            prompt_parts.append(
                "TransferCoord: "
                + "; ".join(_clean_graph_text(line, 150) for _, line in transfer_lines[:transfer_topk])
            )
        prompt_parts.append(f"Candidates: {compact_labels}")
        prompt_parts.append(
            "Answer with only one option letter."
            if self.data_args.graph_option_label_prompt
            else "Answer with exactly one candidate category name."
        )
        role_token_ids = [
            torch.tensor(self.tokenizer(text, add_special_tokens=False).input_ids or [self.tokenizer.unk_token_id or 0], dtype=torch.long)
            for text in role_texts
        ]
        return "\n".join(prompt_parts), graph_ids, graph_emb, graph_hops, role_token_ids

    def _build_zero_shot_graph_prompt(
        self,
        center_node_id: int,
        center_record: Dict[str, object],
        graph_ids: torch.Tensor,
        graph_targets: torch.Tensor,
        option_labels: Optional[Sequence[str]] = None,
    ) -> Tuple[str, List[Tuple[int, torch.Tensor, str]], List[torch.Tensor]]:
        subject_name = self.graph_label_spec["subject_name"]
        if self.data_args.graph_option_label_prompt:
            label_space = _format_option_label_space(
                self.graph_label_spec,
                include_descriptions=self.data_args.graph_include_label_descriptions,
                option_labels=option_labels,
                prompt_style=self.data_args.graph_label_prompt_style,
            )
            label_space_title = "Candidate options with semantic descriptions"
            decision_rule = (
                "Decision rule: use the center first; use abstract summaries as semantic anchors; "
                "use modality-consistent neighbors as strong support; use complement bridge neighbors when structure agrees; "
                "treat conflicts as weak evidence. Answer with only the single best option letter."
            )
        else:
            if self.data_args.graph_label_prompt_style == "promptv2":
                label_space = _format_label_space_promptv2(
                    self.graph_label_spec,
                    include_descriptions=self.data_args.graph_include_label_descriptions,
                )
            else:
                label_space = _format_label_space(
                    self.graph_label_spec,
                    include_descriptions=self.data_args.graph_include_label_descriptions,
                )
            label_space_title = "Candidate labels with semantic descriptions"
            if self.data_args.graph_answer_with_description:
                decision_rule = (
                    "Decision rule: use the center first; use abstract summaries as semantic anchors; "
                    "use modality-consistent neighbors as strong support; use complement bridge neighbors when structure agrees; "
                    "treat conflicts as weak evidence. Return the best category and a concise semantic description."
                )
            else:
                decision_rule = (
                    "Decision rule: use the center first; use abstract summaries as semantic anchors; "
                    "use modality-consistent neighbors as strong support; use complement bridge neighbors when structure agrees; "
                    "treat conflicts as weak evidence. Answer with exactly one candidate category name and no extra text."
                )
        center_text = _clean_graph_text(self._get_record_text(center_record), max_chars=150)
        relation_neighbors = self._select_relation_neighbors(graph_ids, graph_targets, center_node_id)
        synthetic = self.synthetic_context
        dataset_name = self.synthetic_dataset_name
        membership = synthetic.membership(dataset_name, center_node_id) if synthetic is not None and dataset_name is not None else {}

        abstract_lines = []
        extra_nodes: List[Tuple[int, torch.Tensor, str]] = []
        role_texts = [
            "center node: target node to classify",
        ]
        feat_dim = graph_targets.shape[-1]
        center_feat = graph_targets[0].to(torch.float32)
        if self.data_args.graph_include_abstract_nodes and membership:
            _ablate_layer_zs = int(getattr(self.data_args, 'graph_ablate_layer', -1))
            for key, role_name in [
                ("layer1", "layer-1 abstract node"),
                ("layer2", "layer-2 high-level abstract node"),
                ("global", "global semantic prototype node"),
            ]:
                # training-time ablation: skip this layer
                if _ablate_layer_zs == 1 and key == "layer1":
                    continue
                if _ablate_layer_zs == 2 and key == "layer2":
                    continue
                if _ablate_layer_zs == 4 and key == "global":
                    continue
                node = membership.get(key)
                if node is None:
                    continue
                feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
                anchor_sim = None
                if feat is not None:
                    anchor_sim = float(F.cosine_similarity(center_feat, feat.to(torch.float32), dim=0).item())
                if (
                    self.data_args.graph_min_abstract_similarity >= -1.0
                    and anchor_sim is not None
                    and anchor_sim < self.data_args.graph_min_abstract_similarity
                ):
                    continue
                summary = _clean_graph_text(node.get("summary_text", ""), max_chars=110)
                label = _clean_graph_text(node.get("semantic_label", node.get("id", role_name)), max_chars=80)
                reliability = f" reliability={anchor_sim:.2f}." if anchor_sim is not None else ""
                abstract_lines.append(f"- {role_name}: {label}. {summary}{reliability}")
                if feat is not None and self.data_args.graph_include_synthetic_soft_tokens:
                    extra_nodes.append((int(node["global_node_id"]), feat, role_name))

        bridge_lines = []
        include_cross_domain = getattr(self.data_args, 'graph_include_cross_domain_soft_tokens', True)
        if synthetic is not None and dataset_name is not None and include_cross_domain:
            bridge_limit = self.data_args.graph_max_complement_neighbors if self.data_args.graph_real_cross_topk <= 0 else min(self.data_args.graph_max_complement_neighbors, self.data_args.graph_real_cross_topk)
            for bridge in synthetic.bridge_neighbors(dataset_name, center_node_id, bridge_limit):
                if (
                    self.data_args.graph_min_bridge_confidence >= -1.0
                    and float(bridge.get("confidence", 0.0)) < self.data_args.graph_min_bridge_confidence
                ):
                    continue
                dst = int(bridge["dst"])
                bridge_lines.append(
                    f"- bridge neighbor {dst}: {bridge['direction']}; "
                    f"{bridge['abstract_consistency_type']}; "
                    f"conf={bridge['confidence']:.3f}, "
                    f"text_sim={bridge['text_similarity']:.3f}, image_sim={bridge['image_similarity']:.3f}, "
                    f"structure={bridge['structure_consistency_score']:.3f}. "
                    f"Evidence text: {self._record_text_by_node(dst, 70)}"
                )
                if (
                    self.data_args.graph_include_synthetic_soft_tokens
                    and 0 <= dst < len(self.records)
                    and self.graph_features is not None
                ):
                    extra_nodes.append((dst, self.graph_features[dst].to(torch.float32), "modality-complement bridge neighbor"))

        def format_neighbors(title: str, values: List[Dict[str, object]], limit: int) -> str:
            lines = []
            for item in values[:limit]:
                lines.append(
                    f"- node {item['node_id']}: {item['description']} "
                    f"text_sim={item['text_similarity']:.3f}, image_sim={item['image_similarity']:.3f}. "
                    f"Evidence text: {_clean_graph_text(item['text'], max_chars=70)}"
                )
            if not lines:
                return f"{title}:\n- none selected"
            return f"{title}:\n" + "\n".join(lines)

        if str(self.data_args.graph_context_prompt_style).lower() == "compact":
            compact_labels = _format_compact_label_space(
                self.graph_label_spec,
                option_labels=option_labels,
                option_prompt=self.data_args.graph_option_label_prompt,
            )

            def compact_neighbors(title: str, values: List[Dict[str, object]], limit: int) -> Optional[str]:
                lines = []
                for item in values[:limit]:
                    lines.append(
                        f"- {item['node_id']}: t={item['text_similarity']:.2f}, "
                        f"i={item['image_similarity']:.2f}; "
                        f"{_clean_graph_text(item['text'], max_chars=55)}"
                    )
                return f"{title}:\n" + "\n".join(lines) if lines else None

            prompt_parts = [
                DEFAULT_GRAPH_TOKEN,
                f"Classify the center {subject_name}. Use graph tokens, center text, and short neighbor evidence.",
                f"Center: {center_text}",
            ]
            if abstract_lines:
                prompt_parts.append(
                    "Abstract anchors:\n"
                    + "\n".join(_clean_graph_text(line, max_chars=120) for line in abstract_lines[:3])
                )
            support = compact_neighbors(
                "Support",
                relation_neighbors["modality-consistent"],
                min(self.data_args.graph_max_consistent_neighbors, 2),
            )
            if support:
                prompt_parts.append(support)
            if bridge_lines:
                prompt_parts.append(
                    "Bridge:\n"
                    + "\n".join(_clean_graph_text(line, max_chars=105) for line in bridge_lines[:2])
                )
            else:
                bridge = compact_neighbors(
                    "Bridge",
                    relation_neighbors["modality-complement"],
                    min(self.data_args.graph_max_complement_neighbors, 2),
                )
                if bridge:
                    prompt_parts.append(bridge)
            contrast = compact_neighbors(
                "Weak/conflict",
                relation_neighbors["modality-conflict"],
                min(self.data_args.graph_max_conflict_neighbors, 1),
            )
            if contrast:
                prompt_parts.append(contrast)
            prompt_parts.append(f"Candidates: {compact_labels}")
            prompt_parts.append(
                "Answer with only one option letter."
                if self.data_args.graph_option_label_prompt
                else "Answer with exactly one candidate category name."
            )

            if self.ppr is not None:
                role_texts.extend(["neighbor" for _ in range(max(0, graph_ids.numel() - 1))])
            else:
                for hop in _fixed_graph_hops(graph_ids.numel(), self.data_args.graph_hops, self.data_args.graph_sample_size).tolist()[1:]:
                    role_texts.append("neighbor" if int(hop) > 0 else "center")
            return "\n".join(prompt_parts), extra_nodes, [
                torch.tensor(self.tokenizer(text, add_special_tokens=False).input_ids or [self.tokenizer.unk_token_id or 0], dtype=torch.long)
                for text in role_texts
            ]

        prompt_parts = [
            DEFAULT_GRAPH_TOKEN,
            "Classify the CENTER node in a multimodal graph. Graph soft tokens encode the center, selected PPR neighbors, abstract nodes, and modality bridge neighbors.",
        ]
        if abstract_lines:
            prompt_parts.append("ABSTRACT SEMANTIC CONTEXT:\n" + "\n".join(abstract_lines))
        prompt_parts.append(
            format_neighbors(
                "SUPPORTING MODALITY-CONSISTENT NEIGHBORS",
                relation_neighbors["modality-consistent"],
                self.data_args.graph_max_consistent_neighbors,
            )
        )
        if bridge_lines:
            prompt_parts.append("COMPLEMENTARY MULTIMODAL BRIDGE NEIGHBORS:\n" + "\n".join(bridge_lines))
        else:
            prompt_parts.append(
                format_neighbors(
                    "COMPLEMENTARY MULTIMODAL EVIDENCE",
                    relation_neighbors["modality-complement"],
                    self.data_args.graph_max_complement_neighbors,
                )
            )
        prompt_parts.append(
            format_neighbors(
                "CONTRASTIVE OR WEAK EVIDENCE",
                relation_neighbors["modality-conflict"],
                self.data_args.graph_max_conflict_neighbors,
            )
        )
        prompt_parts.append(f"CENTER NODE:\nRole: center node.\nText: {center_text}")
        prompt_parts.append(f"{label_space_title}:\n{label_space}")
        prompt_parts.append(decision_rule)
        if self.ppr is not None:
            role_texts.extend(["PPR-ranked structural neighbor" for _ in range(max(0, graph_ids.numel() - 1))])
        else:
            role_texts.extend([graph_role_description(int(h)) for h in _fixed_graph_hops(graph_ids.numel(), self.data_args.graph_hops, self.data_args.graph_sample_size).tolist()[1:]])
        return "\n\n".join(prompt_parts), extra_nodes, [
            torch.tensor(self.tokenizer(text, add_special_tokens=False).input_ids or [self.tokenizer.unk_token_id or 0], dtype=torch.long)
            for text in role_texts
        ]

    def _load_image(self, asin):
        processor = self.data_args.image_processor
        image = None
        if self.data_args.image_folder is not None:
            folder_candidate = os.path.join(self.data_args.image_folder, f"{asin}.jpg")
            if os.path.exists(folder_candidate):
                image = Image.open(folder_candidate).convert("RGB")
        if image is None and self.image_archive is not None:
            tar = self._get_image_tar()
            if self.image_member_prefix is None:
                member_name = f"{asin}.jpg"
            else:
                member_name = f"{self.image_member_prefix}/{asin}.jpg"
            try:
                member = tar.getmember(member_name)
                with tar.extractfile(member) as fp:
                    image = Image.open(io.BytesIO(fp.read())).convert("RGB")
            except KeyError as exc:
                raise FileNotFoundError(f"Image {member_name} not found in {self.image_archive}") from exc

        if image is None:
            raise FileNotFoundError(f"Unable to locate image for ASIN {asin}")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        return processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample_idx = self.sample_indices[i]
        record = self.records[sample_idx]
        asin = record["asin"]
        if self.graph_ids_cache is not None:
            graph_ids = self.graph_ids_cache[sample_idx].to(torch.long)
        else:
            node_id = self._resolve_node_id(asin, sample_idx)
            graph_ids = torch.tensor(self._serialize_node_context(node_id), dtype=torch.long)
        graph_node_mask = graph_ids.ne(DEFAULT_GRAPH_PAD_ID)
        if self.graph_feat_cache is not None:
            graph_targets = torch.from_numpy(np.array(self.graph_feat_cache[sample_idx], copy=True)).to(torch.float32)
        else:
            graph_targets = torch.zeros((graph_ids.shape[0], self.graph_features.shape[1]), dtype=self.graph_features.dtype)
            graph_targets[graph_node_mask] = self.graph_features[graph_ids[graph_node_mask]]

        # Apply graph_real_neighbor_topk truncation: keep only center + topk PPR neighbors
        neighbor_topk = getattr(self.data_args, 'graph_real_neighbor_topk', -1)
        if neighbor_topk >= 0 and graph_ids.numel() > 1 + neighbor_topk:
            keep = min(1 + neighbor_topk, graph_ids.numel())
            graph_ids = graph_ids[:keep]
            graph_targets = graph_targets[:keep]
            graph_node_mask = graph_node_mask[:keep]

        subject_name = self.graph_label_spec["subject_name"]
        extra_graph_nodes = []
        role_token_ids = None
        option_labels = self._option_labels_for_sample(sample_idx)
        semantic_coord_mode = bool(self.data_args.graph_semantic_coord_tokens)
        if (
            semantic_coord_mode
            and self.data_args.graph_zero_shot_serializer
            and self.data_args.graph_label_supervision
        ):
            node_id = self._resolve_node_id(asin, sample_idx)
            human_prompt, graph_ids, graph_targets, graph_hops, role_token_ids = self._build_semantic_coordinate_graph(
                node_id,
                record,
                graph_ids,
                graph_targets,
                option_labels=option_labels,
            )
            graph_node_mask = torch.ones(graph_ids.shape[0], dtype=torch.bool)
            extra_graph_nodes = []
            if not self.data_args.graph_only:
                human_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{human_prompt}"
        elif self.data_args.graph_zero_shot_serializer and self.data_args.graph_label_supervision:
            node_id = self._resolve_node_id(asin, sample_idx)
            human_prompt, extra_graph_nodes, role_token_ids = self._build_zero_shot_graph_prompt(
                node_id,
                record,
                graph_ids,
                graph_targets,
                option_labels=option_labels,
            )
            if not self.data_args.graph_only:
                human_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{human_prompt}"
        elif self.data_args.graph_only:
            if self.data_args.graph_label_supervision:
                label_space = ", ".join(self.graph_label_spec["candidate_labels"])
                human_prompt = f"{DEFAULT_GRAPH_TOKEN}\n" + self.graph_label_spec["prompt_template"].format(
                    subject_name=subject_name,
                    label_space=label_space,
                )
            else:
                human_prompt = (
                    f"{DEFAULT_GRAPH_TOKEN}\n"
                    f"Please briefly describe the center {subject_name} by using the BFS-serialized neighborhood graph."
                )
        else:
            human_prompt = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                f"{DEFAULT_GRAPH_TOKEN}\n"
                f"Please briefly describe the center {subject_name} by using the image and the BFS-serialized neighborhood graph."
            )
        if self.data_args.graph_label_supervision:
            if self.labels is None:
                raise ValueError("labels.pt is required when graph_label_supervision=True")
            label_id = int(self.labels[sample_idx].item())
            valid_id2label = self.graph_label_spec["valid_id2label"]
            if label_id not in valid_id2label:
                raise ValueError(f"Label id {label_id} is not in valid label mapping for {self.graph_label_spec['name']}")
            assistant_prompt = _format_label_answer(
                valid_id2label[label_id],
                self.graph_label_spec,
                with_description=self.data_args.graph_answer_with_description,
            )
            if self.data_args.graph_option_label_prompt:
                assistant_prompt = _label_to_option_letter(
                    valid_id2label[label_id],
                    self.graph_label_spec,
                    option_labels=option_labels,
                )
        else:
            assistant_prompt = self._get_record_text(record)

        if extra_graph_nodes:
            existing = set(int(x.item()) for x in graph_ids[graph_node_mask])
            extra_ids = []
            extra_embs = []
            extra_roles = []
            for node_id, feat, role in extra_graph_nodes:
                if int(node_id) in existing:
                    continue
                existing.add(int(node_id))
                extra_ids.append(int(node_id))
                extra_embs.append(feat.to(torch.float32))
                extra_roles.append(role)
            if extra_ids:
                graph_ids = torch.cat([graph_ids, torch.tensor(extra_ids, dtype=torch.long)], dim=0)
                graph_targets = torch.cat([graph_targets, torch.stack(extra_embs, dim=0)], dim=0)
                graph_node_mask = torch.cat([graph_node_mask, torch.ones(len(extra_ids), dtype=torch.bool)], dim=0)
                if role_token_ids is not None:
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    for role in extra_roles:
                        ids = self.tokenizer(role, add_special_tokens=False).input_ids
                        role_token_ids.append(torch.tensor(ids or [unk_id], dtype=torch.long))

        if role_token_ids is None:
            hops_for_roles = _fixed_graph_hops(graph_ids.numel(), self.data_args.graph_hops, self.data_args.graph_sample_size)
            unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
            role_token_ids = []
            for pos, hop in enumerate(hops_for_roles.tolist()):
                if self.ppr is not None and pos > 0:
                    role_text = "PPR-ranked structural neighbor"
                else:
                    role_text = graph_role_description(int(hop))
                ids = self.tokenizer(role_text, add_special_tokens=False).input_ids
                role_token_ids.append(torch.tensor(ids or [unk_id], dtype=torch.long))
        if not semantic_coord_mode:
            graph_hops = _fixed_graph_hops(graph_ids.numel(), self.data_args.graph_hops, self.data_args.graph_sample_size)
            if graph_hops.numel() < graph_ids.numel():
                graph_hops = torch.cat([graph_hops, torch.zeros(graph_ids.numel() - graph_hops.numel(), dtype=torch.long)], dim=0)
        sources = [[
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": assistant_prompt},
        ]]
        if not self.data_args.graph_only:
            sources = preprocess_multimodal(copy.deepcopy(sources), self.data_args)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=not self.data_args.graph_only,
            has_graph=True,
        )
        candidate_input_ids = None
        candidate_labels = None
        if (
            self.data_args.graph_candidate_likelihood_training
            and self.data_args.graph_label_supervision
            and self.labels is not None
        ):
            label_positions = torch.where(data_dict["labels"][0].ne(IGNORE_INDEX))[0]
            if label_positions.numel() == 0:
                raise ValueError("Unable to build candidate likelihood examples without supervised answer tokens.")
            prompt_ids = data_dict["input_ids"][0][: int(label_positions[0].item())].clone()
            graph_placeholder_count = int((prompt_ids == GRAPH_TOKEN_INDEX).sum().item())
            graph_expand_extra = max(0, int(graph_node_mask.sum().item()) - graph_placeholder_count)
            candidate_text_max_length = int(self.tokenizer.model_max_length) - graph_expand_extra
            gt_label = valid_id2label[label_id]
            all_candidate_labels = _ordered_option_labels(self.graph_label_spec)
            neg_labels = self._candidate_negative_labels(gt_label, sample_idx, all_candidate_labels)
            selected_labels = [gt_label] + neg_labels

            candidate_input_ids = []
            candidate_labels = []
            for candidate_label in selected_labels:
                if self.data_args.graph_option_label_prompt:
                    answer = _label_to_option_letter(
                        candidate_label,
                        self.graph_label_spec,
                        option_labels=option_labels,
                    )
                else:
                    answer = _format_label_answer(
                        candidate_label,
                        self.graph_label_spec,
                        with_description=self.data_args.graph_answer_with_description,
                    )
                answer_ids = torch.tensor(
                    self.tokenizer(answer, add_special_tokens=False).input_ids,
                    dtype=torch.long,
                )
                if answer_ids.numel() == 0:
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    answer_ids = torch.tensor([unk_id], dtype=torch.long)
                candidate_max_length = max(int(answer_ids.numel()) + 1, candidate_text_max_length)
                full_ids, full_labels = _build_candidate_likelihood_sequence(
                    prompt_ids,
                    answer_ids,
                    candidate_max_length,
                )
                candidate_input_ids.append(full_ids)
                candidate_labels.append(full_labels)
        sample = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            graph=graph_ids,
            graph_emb=graph_targets,
            graph_node_mask=graph_node_mask,
            graph_targets=graph_targets,
            graph_hops=graph_hops,
            graph_role_ids=role_token_ids,
        )
        if candidate_input_ids is not None:
            sample["candidate_input_ids"] = candidate_input_ids
            sample["candidate_labels"] = candidate_labels
        if not self.data_args.graph_only:
            sample["image"] = self._load_image(asin)
        return sample


class LazyGraphLinkPredictionDataset(Dataset):
    """Graph-only link prediction dataset with two serialized node-centered subgraphs."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super().__init__()
        data_root = data_args.graph_data_root
        if data_root is None:
            raise ValueError("graph_data_root is required for graph link prediction.")

        split_path = os.path.join(data_root, "lp-edge-split.pt")
        graph_ids_path = resolve_graph_path(data_root, data_args.graph_cache_path, f"graph_ids_h{data_args.graph_hops}_s{data_args.graph_sample_size}.pt")
        graph_feat_path = resolve_graph_path(data_root, data_args.graph_feat_cache_path, f"graph_feat_h{data_args.graph_hops}_s{data_args.graph_sample_size}_fp16.npy")

        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Link prediction split file not found: {split_path}")
        if graph_ids_path is None or not os.path.exists(graph_ids_path):
            raise FileNotFoundError(f"Graph cache not found: {graph_ids_path}")
        if graph_feat_path is None or not os.path.exists(graph_feat_path):
            raise FileNotFoundError(f"Graph feature cache not found: {graph_feat_path}")

        split = torch.load(split_path, map_location="cpu", weights_only=False)
        train_split = split["train"]
        neg_source = split["valid"]["source_node"]
        neg_target = split["valid"]["target_node_neg"]

        self.samples = []
        num_pos = train_split["source_node"].shape[0]
        if data_args.max_train_samples > 0:
            num_pos = min(num_pos, data_args.max_train_samples)
        neg_k = max(1, data_args.graph_neg_k)
        for idx in range(num_pos):
            src = int(train_split["source_node"][idx].item())
            tgt = int(train_split["target_node"][idx].item())
            self.samples.append((src, tgt, 1))

            if idx < neg_source.shape[0]:
                cur_src = int(neg_source[idx].item())
                for neg_idx in range(min(neg_k, neg_target.shape[1])):
                    neg_tgt = int(neg_target[idx, neg_idx].item())
                    self.samples.append((cur_src, neg_tgt, 0))

        self.graph_ids_cache = torch.load(graph_ids_path, map_location="cpu", weights_only=False).to(torch.long)
        self.graph_feat_cache = np.load(graph_feat_path, mmap_mode="r")
        self.tokenizer = tokenizer
        self.data_args = data_args

        rank0_print(f"Formatting graph LP inputs...Skip in lazy mode, size {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        graph_seq_len = self.graph_ids_cache.shape[1]
        prompt_len = 32
        return [prompt_len + graph_seq_len * 2 for _ in self.samples]

    @property
    def modality_lengths(self):
        return self.lengths

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        src, tgt, label = self.samples[i]
        graph_ids = torch.stack([self.graph_ids_cache[src], self.graph_ids_cache[tgt]], dim=0)
        graph_targets = torch.from_numpy(
            np.array(
                [self.graph_feat_cache[src], self.graph_feat_cache[tgt]],
                copy=True,
            )
        ).to(torch.float32)
        graph_node_mask = graph_ids.ne(DEFAULT_GRAPH_PAD_ID)

        human_prompt = (
            f"{DEFAULT_GRAPH_TOKEN}\n"
            f"{DEFAULT_GRAPH_TOKEN}\n"
            "Given two node-centered subgraphs, predict whether the two center nodes should be connected. "
            "Answer with yes or no only."
        )
        assistant_prompt = "yes" if label == 1 else "no"

        sources = [[
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": assistant_prompt},
        ]]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=False,
            has_graph=True,
        )
        return dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            graph=graph_ids,
            graph_emb=graph_targets,
            graph_node_mask=graph_node_mask,
            graph_targets=graph_targets,
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    graph_modality_mask_prob: float = 0.15
    graph_modality_mask_strategy: str = "random"

    def _truncate_multimodal_sequence(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        reserved_mm_extra_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = int(self.tokenizer.model_max_length) - max(0, int(reserved_mm_extra_tokens))
        max_length = max(1, max_length)
        if max_length <= 0 or input_ids.numel() <= max_length:
            return input_ids, labels

        label_positions = torch.where(labels.ne(IGNORE_INDEX))[0]
        graph_positions = torch.where(input_ids.eq(GRAPH_TOKEN_INDEX))[0]
        image_positions = torch.where(input_ids.eq(IMAGE_TOKEN_INDEX))[0]
        special_positions = torch.cat([graph_positions, image_positions], dim=0)
        if label_positions.numel() == 0 or special_positions.numel() == 0:
            return input_ids[:max_length], labels[:max_length]

        first_label = int(label_positions[0].item())
        last_label = int(label_positions[-1].item())
        last_special = int(special_positions.max().item())

        # Keep all multimodal placeholders, the prompt suffix closest to the
        # answer, and the supervised answer tail. The suffix contains the final
        # evidence/candidate-label instructions used by the frozen LLM.
        answer_tail_len = input_ids.numel() - first_label
        if answer_tail_len >= max_length:
            return input_ids[-max_length:], labels[-max_length:]

        prefix_end = min(last_special + 1, first_label)
        prompt_tail_budget = max_length - prefix_end - answer_tail_len
        if prompt_tail_budget <= 0:
            keep_start = max(0, input_ids.numel() - (max_length - prefix_end))
            kept_input_ids = torch.cat([input_ids[:prefix_end], input_ids[keep_start:]], dim=0)
            kept_labels = torch.cat([labels[:prefix_end], labels[keep_start:]], dim=0)
        else:
            tail_start = max(prefix_end, first_label - prompt_tail_budget)
            kept_input_ids = torch.cat([input_ids[:prefix_end], input_ids[tail_start:]], dim=0)
            kept_labels = torch.cat([labels[:prefix_end], labels[tail_start:]], dim=0)
        if kept_labels.ne(IGNORE_INDEX).sum() == 0 and label_positions.numel() > 0:
            answer_start = max(0, last_label - max_length + 1)
            kept_input_ids = input_ids[answer_start : answer_start + max_length]
            kept_labels = labels[answer_start : answer_start + max_length]
        if kept_input_ids.numel() > max_length:
            kept_input_ids = kept_input_ids[-max_length:]
            kept_labels = kept_labels[-max_length:]
        return kept_input_ids, kept_labels

    def _reserved_mm_extra_tokens(self, instance: Dict, input_ids: torch.Tensor) -> int:
        reserved = 0
        if "graph_node_mask" in instance:
            graph_placeholders = int(input_ids.eq(GRAPH_TOKEN_INDEX).sum().item())
            graph_tokens = int(instance["graph_node_mask"].sum().item())
            reserved += max(0, graph_tokens - graph_placeholders)
        return reserved

    def _collate_graphs(self, instances: Sequence[Dict]) -> Dict[str, object]:
        graph_instances = [instance for instance in instances if "graph_emb" in instance]
        if not graph_instances:
            return {}

        feat_dim = graph_instances[0]["graph_emb"].shape[-1]
        max_nodes = max(instance["graph_emb"].shape[0] if "graph_emb" in instance else 0 for instance in instances)
        max_role_len = max(
            (max((role_ids.numel() for role_ids in instance.get("graph_role_ids", [])), default=1)
             for instance in instances),
            default=1,
        )
        batch_size = len(instances)
        graphs = torch.full((batch_size, max_nodes), DEFAULT_GRAPH_PAD_ID, dtype=torch.long)
        graph_emb = torch.zeros((batch_size, max_nodes, feat_dim), dtype=graph_instances[0]["graph_emb"].dtype)
        graph_targets = torch.zeros_like(graph_emb)
        graph_node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        graph_hops = torch.zeros((batch_size, max_nodes), dtype=torch.long)
        graph_role_ids = torch.full((batch_size, max_nodes, max_role_len), self.tokenizer.pad_token_id, dtype=torch.long)
        graph_role_mask = torch.zeros((batch_size, max_nodes, max_role_len), dtype=torch.bool)

        for i, instance in enumerate(instances):
            if "graph_emb" not in instance:
                continue
            n = instance["graph_emb"].shape[0]
            graphs[i, :n] = instance["graph"]
            graph_emb[i, :n] = instance["graph_emb"]
            graph_targets[i, :n] = instance["graph_targets"]
            graph_node_mask[i, :n] = instance["graph_node_mask"]
            graph_hops[i, :n] = instance["graph_hops"]
            for j, role_ids in enumerate(instance["graph_role_ids"]):
                role_len = min(role_ids.numel(), max_role_len)
                graph_role_ids[i, j, :role_len] = role_ids[:role_len]
                graph_role_mask[i, j, :role_len] = True

        prob = self.graph_modality_mask_prob
        valid = graph_node_mask
        strategy = self.graph_modality_mask_strategy
        text_mask = torch.zeros_like(valid)
        image_mask = torch.zeros_like(valid)
        if prob > 0 and strategy != "none":
            if strategy == "random":
                text_mask = (torch.rand(valid.shape) < prob) & valid
                image_mask = (torch.rand(valid.shape) < prob) & valid
            else:
                if valid.shape[1] > 0:
                    center_valid = valid[:, 0] & (torch.rand(valid.shape[0]) < prob)
                    if strategy == "center_one":
                        mask_text = torch.rand(valid.shape[0]) < 0.5
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
        return {
            "graphs": graphs,
            "graph_emb": graph_emb,
            "graph_node_mask": graph_node_mask,
            "graph_targets": graph_targets,
            "graph_hops": graph_hops,
            "graph_role_ids": graph_role_ids,
            "graph_role_mask": graph_role_mask,
            "graph_text_mask": text_mask,
            "graph_image_mask": image_mask,
        }

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids, labels = zip(
            *[
                self._truncate_multimodal_sequence(
                    cur_input_ids,
                    cur_labels,
                    self._reserved_mm_extra_tokens(instance, cur_input_ids),
                )
                for instance, cur_input_ids, cur_labels in zip(instances, input_ids, labels)
            ]
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        batch.update(self._collate_graphs(instances))

        if all("candidate_input_ids" in instance for instance in instances):
            flat_candidate_input_ids = []
            flat_candidate_labels = []
            candidate_graph_indices = []
            candidate_group_sizes = []
            for sample_idx, instance in enumerate(instances):
                cur_input_ids = instance["candidate_input_ids"]
                cur_labels = instance["candidate_labels"]
                if len(cur_input_ids) != len(cur_labels):
                    raise ValueError("candidate_input_ids and candidate_labels must have the same length.")
                candidate_group_sizes.append(len(cur_input_ids))
                for ids, labels in zip(cur_input_ids, cur_labels):
                    flat_candidate_input_ids.append(ids)
                    flat_candidate_labels.append(labels)
                    candidate_graph_indices.append(sample_idx)
            if flat_candidate_input_ids:
                candidate_input_ids = torch.nn.utils.rnn.pad_sequence(
                    flat_candidate_input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                candidate_labels = torch.nn.utils.rnn.pad_sequence(
                    flat_candidate_labels,
                    batch_first=True,
                    padding_value=IGNORE_INDEX,
                )
                batch.update(
                    {
                        "candidate_input_ids": candidate_input_ids,
                        "candidate_labels": candidate_labels,
                        "candidate_attention_mask": candidate_input_ids.ne(self.tokenizer.pad_token_id),
                        "candidate_graph_indices": torch.tensor(candidate_graph_indices, dtype=torch.long),
                        "candidate_group_sizes": torch.tensor(candidate_group_sizes, dtype=torch.long),
                    }
                )

        return batch


class LengthAwareConcatDataset(ConcatDataset):
    """ConcatDataset that preserves the length metadata used by LLaVATrainer."""

    def __init__(self, datasets: Sequence[Dataset]):
        super().__init__(datasets)
        self._lengths = None
        self._modality_lengths = None

    @staticmethod
    def _merge_lengths(datasets: Sequence[Dataset], attr_name: str) -> List[int]:
        merged = []
        for dataset in datasets:
            values = getattr(dataset, attr_name, None)
            if values is None:
                raise AttributeError(f"{dataset.__class__.__name__} is missing {attr_name}")
            merged.extend(list(values))
        return merged

    @property
    def lengths(self):
        if self._lengths is None:
            self._lengths = self._merge_lengths(self.datasets, "lengths")
        return self._lengths

    @property
    def modality_lengths(self):
        if self._modality_lengths is None:
            self._modality_lengths = self._merge_lengths(self.datasets, "modality_lengths")
        return self._modality_lengths


def split_multi_graph_arg(value: Optional[str], keep_empty: bool = False) -> Optional[List[Optional[str]]]:
    if value is None:
        return None
    items = []
    for item in value.split(","):
        item = item.strip()
        if item == "":
            items.append("" if keep_empty else None)
        else:
            items.append(item)
    return items


def get_multi_graph_values(
    data_args: DataArguments,
    field_name: str,
    multi_field_name: str,
    num_datasets: int,
    keep_empty: bool = False,
) -> List[Optional[str]]:
    raw_values = split_multi_graph_arg(getattr(data_args, multi_field_name), keep_empty=keep_empty)
    if raw_values is None:
        return [getattr(data_args, field_name) for _ in range(num_datasets)]
    if len(raw_values) != num_datasets:
        raise ValueError(
            f"{multi_field_name} has {len(raw_values)} entries, but graph_data_roots has {num_datasets} entries."
        )
    return raw_values


def build_multi_graph_data_args(data_args: DataArguments) -> List[DataArguments]:
    roots = split_multi_graph_arg(data_args.graph_data_roots)
    if not roots:
        raise ValueError("graph_data_roots must contain at least one dataset root.")
    if any(root is None for root in roots):
        raise ValueError("graph_data_roots cannot contain empty entries.")

    num_datasets = len(roots)
    text_paths = get_multi_graph_values(data_args, "graph_text_path", "graph_text_paths", num_datasets)
    feat_paths = get_multi_graph_values(data_args, "graph_feat_path", "graph_feat_paths", num_datasets)
    edge_paths = get_multi_graph_values(data_args, "graph_edge_path", "graph_edge_paths", num_datasets)
    image_archives = get_multi_graph_values(
        data_args,
        "graph_image_archive",
        "graph_image_archives",
        num_datasets,
    )
    cache_paths = get_multi_graph_values(
        data_args,
        "graph_cache_path",
        "graph_cache_paths",
        num_datasets,
        keep_empty=True,
    )
    feat_cache_paths = get_multi_graph_values(
        data_args,
        "graph_feat_cache_path",
        "graph_feat_cache_paths",
        num_datasets,
        keep_empty=True,
    )
    label_configs = get_multi_graph_values(data_args, "graph_label_config", "graph_label_configs", num_datasets)
    synthetic_roots = get_multi_graph_values(data_args, "graph_synthetic_root", "graph_synthetic_roots", num_datasets)

    multi_fields = {
        "graph_data_roots": None,
        "graph_text_paths": None,
        "graph_feat_paths": None,
        "graph_edge_paths": None,
        "graph_image_archives": None,
        "graph_cache_paths": None,
        "graph_feat_cache_paths": None,
        "graph_label_configs": None,
        "graph_synthetic_roots": None,
    }
    graph_data_args_list = [
        replace(
            data_args,
            graph_data_root=roots[i],
            graph_text_path=text_paths[i],
            graph_feat_path=feat_paths[i],
            graph_edge_path=edge_paths[i],
            graph_image_archive=image_archives[i],
            graph_cache_path=cache_paths[i],
            graph_feat_cache_path=feat_cache_paths[i],
            graph_label_config=label_configs[i],
            graph_synthetic_root=synthetic_roots[i],
            **multi_fields,
        )
        for i in range(num_datasets)
    ]
    for graph_data_args in graph_data_args_list:
        for attr_name in ("image_processor", "mm_use_im_start_end"):
            if hasattr(data_args, attr_name):
                setattr(graph_data_args, attr_name, getattr(data_args, attr_name))
    return graph_data_args_list


def load_graph_hidden_size(data_args: DataArguments) -> int:
    graph_feat_path = resolve_graph_path(data_args.graph_data_root, data_args.graph_feat_path, "clip_feat.pt")
    if graph_feat_path is None or not os.path.exists(graph_feat_path):
        raise FileNotFoundError(f"Graph feature path not found: {graph_feat_path}")
    graph_feats = torch.load(graph_feat_path, map_location="cpu", weights_only=False)
    if isinstance(graph_feats, dict):
        if "features" not in graph_feats:
            raise ValueError(f"Unsupported graph feature dict keys: {list(graph_feats.keys())}")
        graph_feats = graph_feats["features"]
    if not isinstance(graph_feats, torch.Tensor):
        raise TypeError(f"Unsupported graph feature type: {type(graph_feats)}")
    hidden_size = int(graph_feats.shape[-1])
    del graph_feats
    return hidden_size


def resolve_graph_hidden_size(data_args: DataArguments) -> int:
    graph_data_args_list = (
        build_multi_graph_data_args(data_args)
        if data_args.graph_data_roots is not None
        else [data_args]
    )
    hidden_sizes = [load_graph_hidden_size(graph_data_args) for graph_data_args in graph_data_args_list]
    if len(set(hidden_sizes)) != 1:
        raise ValueError(f"All graph feature dimensions must match for joint training, got {hidden_sizes}.")
    return hidden_sizes[0]


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.data_path is not None and os.path.exists(data_args.data_path):
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
    elif data_args.graph_data_roots is not None:
        dataset_cls = LazyGraphLinkPredictionDataset if data_args.graph_task == "lp" else LazyGraphSupervisedDataset
        train_datasets = [
            dataset_cls(
                tokenizer=tokenizer,
                data_args=graph_data_args,
            )
            for graph_data_args in build_multi_graph_data_args(data_args)
        ]
        train_dataset = LengthAwareConcatDataset(train_datasets)
        rank0_print(f"Joint graph fine-tuning over {len(train_datasets)} datasets, total size {len(train_dataset)}")
    elif data_args.graph_data_root is not None or data_args.graph_text_path is not None:
        if data_args.graph_task == "lp":
            train_dataset = LazyGraphLinkPredictionDataset(
                tokenizer=tokenizer,
                data_args=data_args,
            )
        else:
            train_dataset = LazyGraphSupervisedDataset(
                tokenizer=tokenizer,
                data_args=data_args,
            )
    else:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        graph_modality_mask_prob=getattr(data_args, "graph_modality_mask_prob", 0.15),
        graph_modality_mask_strategy=getattr(data_args, "graph_modality_mask_strategy", "random"),
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set seed before model creation so GraphSoftTokenEncoder init is reproducible
    init_seed = getattr(model_args, "graph_init_seed", 42)
    import random, numpy as np
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    print(f"[init5] Set graph_init_seed={init_seed} before model creation")
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    use_graph_mode = (
        data_args.graph_data_roots is not None
        or data_args.graph_data_root is not None
        or data_args.graph_feat_path is not None
    )

    if use_graph_mode:
        model_args.graph_hidden_size = resolve_graph_hidden_size(data_args)
    data_args.graph_modality_mask_prob = model_args.graph_modality_mask_prob
    data_args.graph_modality_mask_strategy = model_args.graph_modality_mask_strategy

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None or use_graph_mode:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                **bnb_model_from_pretrained_args
            )
        else:
            model_args.graph_synthetic_root = getattr(data_args, "graph_synthetic_root", None)
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
        if model_args.graph_use_strict_zeroshot and hasattr(model.model, 'strict_zeroshot_encoder'):
            model.model.strict_zeroshot_encoder.requires_grad_(True)
        if model_args.graph_use_strict_zeroshot and hasattr(model.model, 'strict_zeroshot_encoder'):
            model.model.strict_zeroshot_encoder.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if hasattr(model.get_model(), "graph_projector"):
                for p in model.get_model().graph_projector.parameters():
                    p.requires_grad = True
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(model.get_model(), module_name):
                    for p in getattr(model.get_model(), module_name).parameters():
                        p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            if hasattr(model.get_model(), "graph_projector"):
                for p in model.get_model().graph_projector.parameters():
                    p.requires_grad = False
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(model.get_model(), module_name):
                    for p in getattr(model.get_model(), module_name).parameters():
                        p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if hasattr(model.get_model(), "graph_projector"):
                model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(model.get_model(), module_name):
                    getattr(model.get_model(), module_name).to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.graph_disable_role_tokens = model_args.graph_disable_role_tokens
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    if use_graph_mode:
        model.config.graph_soft_token_encoder_type = getattr(model_args, "graph_soft_token_encoder_type", "charm")
        model.get_model().initialize_graph_modules(model_args=model_args)
        if os.environ.get("CHARM_INITIALIZE_ADAPTER") == "1" and model_args.pretrain_mm_mlp_adapter:
            weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
            owner = model.get_model()
            for module_name in ("mm_projector", "graph_soft_token_encoder"):
                prefix = f"model.{module_name}."
                module_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
                module = getattr(owner, module_name, None)
                if not module_weights or module is None:
                    raise ValueError(f"Initialization adapter or model is missing {module_name}")
                module.load_state_dict(module_weights, strict=True)
                print(f"Loaded {len(module_weights)} {module_name} parameters from the initialization adapter")
            del weights, module_weights
        for module_name in ["graph_soft_token_encoder", "graph_projector"]:
            if hasattr(model.get_model(), module_name):
                getattr(model.get_model(), module_name).to(device=training_args.device)
        model.config.graph_disable_role_tokens = model_args.graph_disable_role_tokens
        model.config.graph_disable_struct_tokens = model_args.graph_disable_struct_tokens
        model.config.graph_soft_token_gnn_layers = model_args.graph_soft_token_gnn_layers
        model.config.graph_soft_token_mode = model_args.graph_soft_token_mode
        model.config.graph_real_neighbor_topk = data_args.graph_real_neighbor_topk
        model.config.graph_real_cross_topk = data_args.graph_real_cross_topk
        model.config.graph_include_cross_domain_soft_tokens = data_args.graph_include_cross_domain_soft_tokens
        model.config.graph_ablate_hop = getattr(data_args, 'graph_ablate_hop', -1)
        model.config.graph_ablate_layer = getattr(data_args, 'graph_ablate_layer', -1)
        model.config.graph_semantic_evidence_scale = model_args.graph_semantic_evidence_scale
        model.config.graph_semantic_evidence_gate_init = model_args.graph_semantic_evidence_gate_init
        model.config.graph_use_gcn = model_args.graph_use_gcn
        model.config.graph_use_gcn_1hop = model_args.graph_use_gcn_1hop
        model.config.graph_use_gcn_2hop = model_args.graph_use_gcn_2hop
        model.config.graph_use_center_protect = model_args.graph_use_center_protect
        model.config.graph_use_ppr_prop = model_args.graph_use_ppr_prop
        model.config.tune_graph_projector_only = training_args.tune_graph_projector_only = model_args.tune_graph_projector_only
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_graph_projector_only:
            model.requires_grad_(False)
            if hasattr(model.get_model(), "graph_soft_token_encoder"):
                for p in model.get_model().graph_soft_token_encoder.parameters():
                    p.requires_grad = True
            if hasattr(model.get_model(), "graph_projector"):
                for p in model.get_model().graph_projector.parameters():
                    p.requires_grad = True
        if model_args.tune_mm_mlp_adapter and model_args.vision_tower is None:
            model.requires_grad_(False)
            if hasattr(model.get_model(), "graph_projector"):
                for p in model.get_model().graph_projector.parameters():
                    p.requires_grad = True
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(model.get_model(), module_name):
                    for p in getattr(model.get_model(), module_name).parameters():
                        p.requires_grad = True
        if training_args.freeze_mm_mlp_adapter and model_args.vision_tower is None:
            if hasattr(model.get_model(), "graph_projector"):
                for p in model.get_model().graph_projector.parameters():
                    p.requires_grad = False
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(model.get_model(), module_name):
                    for p in getattr(model.get_model(), module_name).parameters():
                        p.requires_grad = False
        if training_args.bits in [4, 8]:
            if hasattr(model.get_model(), "graph_projector"):
                model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(model.get_model(), module_name):
                    getattr(model.get_model(), module_name).to(dtype=compute_dtype, device=training_args.device)
        if training_args.lora_enable:
            graph_owner = model.get_model()
            if hasattr(graph_owner, "graph_projector"):
                for p in graph_owner.graph_projector.parameters():
                    p.requires_grad = True
            for module_name in ["graph_soft_token_encoder"]:
                if hasattr(graph_owner, module_name):
                    for p in getattr(graph_owner, module_name).parameters():
                        p.requires_grad = True

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name and 'vision_tower' not in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
        if use_graph_mode and any(p.requires_grad for p in model.get_model().parameters()):
            # The LLM backbone is quantized and frozen, while graph adapters are normal trainable modules.
            # Transformers' Trainer otherwise treats this as forbidden "purely quantized" fine-tuning.
            model._hf_peft_config_loaded = True

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # Save initial weights before training for reproducibility
    if getattr(training_args, "tune_mm_mlp_adapter", False) or getattr(model_args, "tune_mm_mlp_adapter", False):
        init_dir = os.path.join(training_args.output_dir, "init_weights")
        os.makedirs(init_dir, exist_ok=True)
        keys_to_match = ["mm_projector", "graph_projector", "graph_soft_token_encoder"]
        init_weights = get_mm_adapter_state_maybe_zero_3(model.named_parameters(), keys_to_match)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            torch.save(init_weights, os.path.join(init_dir, "mm_projector.bin"))
            print(f"[init3] Saved initial weights to {init_dir}/mm_projector.bin ({len(init_weights)} params)")

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
