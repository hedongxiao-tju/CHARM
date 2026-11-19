import argparse
import csv
import html
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPModel


DEFAULT_GRAPH_PAD_ID = -200


def clean_text(text: object, max_chars: int = 240) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text if x is not None)
    text = html.unescape(str(text)).replace("\n", " ").strip()
    return " ".join(text.split())[:max_chars]


def load_label_spec(path: str) -> Dict[str, object]:
    raw = json.load(open(path, "r", encoding="utf-8"))
    valid_id2label = {int(k): str(v) for k, v in raw["valid_id2label"].items()}
    candidate_labels = [str(x) for x in raw.get("candidate_labels", [valid_id2label[i] for i in sorted(valid_id2label)])]
    descriptions = raw.get("label_descriptions", {})
    return {
        "name": raw.get("name", os.path.basename(os.path.dirname(path))),
        "subject_name": raw.get("subject_name", "node"),
        "valid_id2label": valid_id2label,
        "candidate_labels": candidate_labels,
        "label_descriptions": {str(k): str(v) for k, v in descriptions.items()},
    }


def label_text(label: str, label_spec: Dict[str, object], mode: str = "semantic") -> str:
    desc = label_spec.get("label_descriptions", {}).get(label, "")
    subject_name = str(label_spec.get("subject_name", "item"))
    if mode == "name":
        return label
    if mode == "category":
        return f"Category: {label}."
    if mode == "short":
        return f"{label}: {desc}" if desc else label
    if mode == "subject":
        return f"A {subject_name} category: {label}. {desc}" if desc else f"A {subject_name} category: {label}."
    if mode == "belongs":
        return f"This {subject_name} belongs to {label}. {desc}" if desc else f"This {subject_name} belongs to {label}."
    if mode == "semantic" and desc:
        return f"Category: {label}. Semantic description: {desc}"
    return f"Category: {label}."


def label_texts(label: str, label_spec: Dict[str, object], prompt_mode: str) -> List[str]:
    prompt_mode = resolve_label_prompt_mode(label_spec, prompt_mode)
    modes = {
        "name": ["name"],
        "category": ["category"],
        "short": ["short"],
        "semantic": ["semantic"],
        "subject": ["subject"],
        "belongs": ["belongs"],
        "short_belongs": ["short", "belongs"],
        "short_category": ["short", "category"],
        "all": ["name", "category", "short", "semantic", "subject", "belongs"],
    }.get(prompt_mode)
    if modes is None:
        raise ValueError(f"Unsupported label_prompt_mode: {prompt_mode}")
    return [label_text(label, label_spec, mode) for mode in modes]


def resolve_label_prompt_mode(label_spec: Dict[str, object], prompt_mode: str) -> str:
    if prompt_mode != "auto":
        return prompt_mode
    dataset_name = str(label_spec.get("name", "")).lower()
    subject_name = str(label_spec.get("subject_name", "")).lower()
    if "reddit" in dataset_name or "reddit" in subject_name:
        return "short_category"
    if "movie" in dataset_name or "movie" in subject_name:
        return "all"
    if "grocery" in dataset_name or "toy" in dataset_name or subject_name in {"product", "toy"}:
        return "short_belongs"
    return "short"


def find_csv(root: str) -> Optional[str]:
    candidates = [p for p in os.listdir(root) if p.lower().endswith(".csv")]
    if len(candidates) == 1:
        return os.path.join(root, candidates[0])
    exact = os.path.join(root, f"{os.path.basename(root)}.csv")
    return exact if os.path.exists(exact) else None


def load_records(root: str) -> List[Dict[str, object]]:
    csv_path = find_csv(root)
    if csv_path is None:
        return []
    records = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            node_id = int(row.get("id", row_idx))
            text_parts = []
            for key in ("title", "text", "caption", "description"):
                value = row.get(key)
                if value:
                    text_parts.append(value)
            records.append({"node_id": node_id, "raw_text": clean_text(" ".join(text_parts), 800)})
    records.sort(key=lambda item: int(item.get("node_id", 0)))
    return records


def load_feature_tensor(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(x, dict):
        if "features" in x:
            x = x["features"]
        else:
            raise ValueError(f"Unsupported feature dict keys in {path}: {list(x.keys())}")
    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported feature type in {path}: {type(x)}")
    return x.to(torch.float32)


class SyntheticGraphContext:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.offsets = json.load(open(os.path.join(self.root, "dataset_offsets.json"), "r", encoding="utf-8"))
        self.layer1_nodes = torch.load(os.path.join(self.root, "layer1_abstract_nodes.pt"), map_location="cpu", weights_only=False)
        self.layer2_nodes = torch.load(os.path.join(self.root, "layer2_abstract_nodes.pt"), map_location="cpu", weights_only=False)
        self.global_nodes = torch.load(os.path.join(self.root, "global_semantic_prototype_nodes.pt"), map_location="cpu", weights_only=False)
        self.cluster_assignments = torch.load(os.path.join(self.root, "cluster_assignments.pt"), map_location="cpu", weights_only=False)
        self.layer2_assignments = torch.load(os.path.join(self.root, "layer2_cluster_assignments.pt"), map_location="cpu", weights_only=False)

        self.layer1_by_dataset: Dict[str, List[Dict[str, object]]] = {}
        for node in self.layer1_nodes:
            self.layer1_by_dataset.setdefault(str(node["dataset"]), []).append(node)
        for nodes in self.layer1_by_dataset.values():
            nodes.sort(key=lambda item: int(item.get("cluster_id", 0)))
        self.layer1_pos_by_cluster: Dict[str, Dict[int, int]] = {
            dataset: {int(node["cluster_id"]): pos for pos, node in enumerate(nodes)}
            for dataset, nodes in self.layer1_by_dataset.items()
        }

        self.layer2_by_dataset: Dict[str, List[Dict[str, object]]] = {}
        for node in self.layer2_nodes:
            self.layer2_by_dataset.setdefault(str(node["dataset"]), []).append(node)
        for nodes in self.layer2_by_dataset.values():
            nodes.sort(key=lambda item: int(item.get("topic_id", 0)))

        self.global_by_id = {int(node["global_node_id"]): node for node in self.global_nodes}
        self.layer2_to_global = {}
        for proto in self.global_nodes:
            for layer2_id in proto.get("member_layer2_abstract_node_ids", []):
                self.layer2_to_global[int(layer2_id)] = int(proto["global_node_id"])

        self.bridge_by_dataset: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        bridge_path = os.path.join(self.root, "modality_complement_bridge_edges.pt")
        if os.path.exists(bridge_path):
            bridge = torch.load(bridge_path, map_location="cpu", weights_only=False)
            edge_index = bridge["edge_index"]
            for i in range(edge_index.shape[0]):
                dataset = str(bridge["dataset"][i])
                offset = int(self.offsets[dataset]["start"])
                src = int(edge_index[i, 0].item()) - offset
                dst = int(edge_index[i, 1].item()) - offset
                item = {
                    "dst": dst,
                    "confidence": float(bridge["confidence"][i].item()),
                    "text_similarity": float(bridge["text_similarity"][i].item()),
                    "image_similarity": float(bridge["image_similarity"][i].item()),
                    "structure_consistency_score": float(bridge["structure_consistency_score"][i].item()),
                    "direction": str(bridge["direction"][i]),
                }
                self.bridge_by_dataset.setdefault(dataset, {}).setdefault(src, []).append(item)
            for by_src in self.bridge_by_dataset.values():
                for records in by_src.values():
                    records.sort(key=lambda x: x["confidence"], reverse=True)

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
        out = torch.zeros(feat_dim, dtype=torch.float32)
        out[: feat.numel()] = feat
        return out

    def membership_features(self, dataset: str, node_id: int, feat_dim: int) -> List[torch.Tensor]:
        if dataset not in self.cluster_assignments:
            return []
        cluster_id = int(self.cluster_assignments[dataset][node_id].item())
        layer1_nodes = self.layer1_by_dataset[dataset]
        layer1_pos = self.layer1_pos_by_cluster[dataset][cluster_id]
        layer1 = layer1_nodes[layer1_pos]
        layer2_topic = int(self.layer2_assignments[dataset][layer1_pos].item())
        layer2 = self.layer2_by_dataset[dataset][layer2_topic]
        global_node = self.global_by_id.get(self.layer2_to_global.get(int(layer2["global_node_id"]), -1))
        out = []
        for node in (layer1, layer2, global_node):
            feat = self.abstract_feature(node, feat_dim)
            if feat is not None:
                out.append(feat)
        return out

    def bridge_neighbors(self, dataset: str, node_id: int, limit: int) -> List[Dict[str, object]]:
        return self.bridge_by_dataset.get(dataset, {}).get(int(node_id), [])[:limit]


@dataclass
class DataBundle:
    name: str
    root: str
    features: torch.Tensor
    labels: torch.Tensor
    ppr: Dict[object, torch.Tensor]
    label_spec: Dict[str, object]
    records: List[Dict[str, object]]
    masks: Dict[str, torch.Tensor]


def load_split_masks(root: str, num_nodes: int) -> Dict[str, torch.Tensor]:
    masks = {"all": torch.ones(num_nodes, dtype=torch.bool)}
    for split in ("train", "val", "test"):
        path = os.path.join(root, f"{split}_mask.pt")
        if os.path.exists(path):
            masks[split] = torch.load(path, map_location="cpu", weights_only=False).bool()
    return masks


def load_bundle(magb_root: str, dataset: str) -> DataBundle:
    root = os.path.join(magb_root, dataset)
    labels = torch.load(os.path.join(root, "labels.pt"), map_location="cpu", weights_only=False).to(torch.long)
    return DataBundle(
        name=dataset,
        root=root,
        features=load_feature_tensor(os.path.join(root, "clip_feat.pt")),
        labels=labels,
        ppr=torch.load(os.path.join(root, "ppr.pt"), map_location="cpu", weights_only=False),
        label_spec=load_label_spec(os.path.join(root, "graph_label_config.json")),
        records=load_records(root),
        masks=load_split_masks(root, int(labels.numel())),
    )


def get_ppr_neighbors(ppr: Dict[object, torch.Tensor], node_id: int, top_k: int) -> List[int]:
    values = ppr.get(node_id, ppr.get(str(node_id), []))
    if torch.is_tensor(values):
        values = values.tolist()
    out = []
    seen = {int(node_id)}
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= top_k:
            break
    return out


def weighted_feature_mean(features: torch.Tensor, node_ids: Sequence[int], weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, bool]:
    valid = [int(i) for i in node_ids if 0 <= int(i) < features.shape[0]]
    if not valid:
        return torch.zeros(features.shape[1], dtype=torch.float32), False
    vals = features[torch.as_tensor(valid, dtype=torch.long)]
    if weights is None:
        return vals.mean(dim=0), True
    weights = weights[: len(valid)].to(torch.float32)
    weights = weights / weights.sum().clamp_min(1e-6)
    return (vals * weights.unsqueeze(-1)).sum(dim=0), True


def select_indices(
    labels: torch.Tensor,
    valid_label_ids: Iterable[int],
    ratio: float,
    max_samples: int,
    seed: int,
    stratified: bool = True,
    split_mask: Optional[torch.Tensor] = None,
) -> List[int]:
    valid = set(int(x) for x in valid_label_ids)
    if split_mask is None:
        split_mask = torch.ones_like(labels, dtype=torch.bool)
    indices = [i for i, y in enumerate(labels.tolist()) if bool(split_mask[i]) and int(y) in valid]
    if ratio > 0:
        if ratio > 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        target = max(1, math.ceil(len(indices) * ratio))
    else:
        target = len(indices)
    if max_samples > 0:
        target = min(target, max_samples)
    if target >= len(indices):
        return indices
    rng = random.Random(seed)
    if not stratified:
        rng.shuffle(indices)
        return indices[:target]

    buckets: Dict[int, List[int]] = {label_id: [] for label_id in valid}
    for idx in indices:
        buckets[int(labels[idx].item())].append(idx)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < target:
        progressed = False
        for label_id in sorted(buckets):
            if buckets[label_id]:
                selected.append(buckets[label_id].pop())
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


class MAGBContrastiveDataset(Dataset):
    def __init__(
        self,
        bundles: Dict[str, DataBundle],
        synthetic: SyntheticGraphContext,
        samples: List[Tuple[str, int, int]],
        ppr_k: int,
        bridge_k: int,
        include_abstract: bool,
        component_dropout: float = 0.0,
        cache_components: bool = False,
    ):
        self.bundles = bundles
        self.synthetic = synthetic
        self.samples = samples
        self.ppr_k = ppr_k
        self.bridge_k = bridge_k
        self.include_abstract = include_abstract
        self.component_dropout = component_dropout
        self.cached_components = None
        if cache_components:
            components = []
            masks = []
            for dataset, node_id, _ in self.samples:
                bundle = self.bundles[dataset]
                component_tensor, component_mask = self._build_base_components(bundle, node_id)
                components.append(component_tensor)
                masks.append(component_mask)
            self.cached_components = (torch.stack(components, dim=0), torch.stack(masks, dim=0))

    def __len__(self) -> int:
        return len(self.samples)

    def _build_base_components(self, bundle: DataBundle, node_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_dim = bundle.features.shape[1]
        center = bundle.features[node_id]
        components = [center]
        mask = [True]

        ppr_neighbors = get_ppr_neighbors(bundle.ppr, node_id, self.ppr_k)
        ppr_weights = None
        if ppr_neighbors:
            ranks = torch.arange(len(ppr_neighbors), dtype=torch.float32)
            ppr_weights = 1.0 / torch.log2(ranks + 2.0)
        ppr_feat, has_ppr = weighted_feature_mean(bundle.features, ppr_neighbors, ppr_weights)
        components.append(ppr_feat)
        mask.append(has_ppr)

        abstract_feats = self.synthetic.membership_features(bundle.name, node_id, feat_dim) if self.include_abstract else []
        if abstract_feats:
            components.append(torch.stack(abstract_feats, dim=0).mean(dim=0))
            mask.append(True)
        else:
            components.append(torch.zeros(feat_dim, dtype=torch.float32))
            mask.append(False)

        bridges = self.synthetic.bridge_neighbors(bundle.name, node_id, self.bridge_k)
        bridge_ids = [int(x["dst"]) for x in bridges]
        bridge_weights = torch.tensor([float(x["confidence"]) for x in bridges], dtype=torch.float32) if bridges else None
        bridge_feat, has_bridge = weighted_feature_mean(bundle.features, bridge_ids, bridge_weights)
        components.append(bridge_feat)
        mask.append(has_bridge)

        component_tensor = torch.stack(components, dim=0)
        component_mask = torch.tensor(mask, dtype=torch.bool)
        return component_tensor, component_mask

    def _apply_component_dropout(self, component_tensor: torch.Tensor, component_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.component_dropout > 0 and self.training_like():
            component_tensor = component_tensor.clone()
            component_mask = component_mask.clone()
            drop = torch.rand(component_mask.shape) < self.component_dropout
            drop[0] = False
            component_mask = component_mask & ~drop
            component_tensor = component_tensor.masked_fill(~component_mask.unsqueeze(-1), 0.0)
        return component_tensor, component_mask

    def _build_components(self, bundle: DataBundle, node_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._apply_component_dropout(*self._build_base_components(bundle, node_id))

    def training_like(self) -> bool:
        return self.component_dropout > 0

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        dataset, node_id, target = self.samples[item]
        if self.cached_components is not None:
            components = self.cached_components[0][item]
            component_mask = self.cached_components[1][item]
            components, component_mask = self._apply_component_dropout(components, component_mask)
        else:
            bundle = self.bundles[dataset]
            components, component_mask = self._build_components(bundle, node_id)
        return {
            "components": components,
            "component_mask": component_mask,
            "target": torch.tensor(target, dtype=torch.long),
            "node_id": torch.tensor(node_id, dtype=torch.long),
            "dataset_id": torch.tensor(0, dtype=torch.long),
        }


class GraphLabelAligner(nn.Module):
    def __init__(self, input_dim: int = 1536, label_dim: int = 768, hidden_dim: int = 1024, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.component_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.component_score = nn.Linear(embed_dim, 1)
        self.graph_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.label_projector = nn.Sequential(
            nn.LayerNorm(label_dim),
            nn.Linear(label_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

    def encode_graph(self, components: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        z = self.component_encoder(components)
        scores = self.component_score(z).squeeze(-1)
        scores = scores.masked_fill(~component_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = (z * weights.unsqueeze(-1)).sum(dim=1)
        graph = pooled + self.graph_head(pooled)
        return F.normalize(graph, dim=-1)

    def encode_labels(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.label_projector(label_embeddings), dim=-1)

    def forward(self, components: torch.Tensor, component_mask: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        graph = self.encode_graph(components, component_mask)
        labels = self.encode_labels(label_embeddings)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * graph @ labels.t()


def parse_float_list(value: str, expected: int, name: str) -> List[float]:
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated values, got {value!r}")
    return parts


class ClipPrototypeGraphScorer(nn.Module):
    """Score graph components directly in frozen CLIP text space.

    This keeps unseen label descriptions in their original CLIP semantic space.
    Only small scalar evidence weights are optionally trainable.
    """

    def __init__(
        self,
        component_weights: Sequence[float] = (1.0, 0.2, 0.2, 0.0),
        modality_weights: Sequence[float] = (1.0, 1.0),
        learn_weights: bool = False,
        logit_scale: float = 20.0,
    ):
        super().__init__()
        component = torch.tensor(component_weights, dtype=torch.float32).clamp_min(1e-4)
        modality = torch.tensor(modality_weights, dtype=torch.float32).clamp_min(1e-4)
        self.component_logits = nn.Parameter(component.log(), requires_grad=learn_weights)
        self.modality_logits = nn.Parameter(modality.log(), requires_grad=learn_weights)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale), dtype=torch.float32), requires_grad=learn_weights)

    def encode_graph(self, components: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        text = F.normalize(components[..., :768], dim=-1)
        image = F.normalize(components[..., 768:], dim=-1)
        modality_weights = torch.softmax(self.modality_logits, dim=-1)
        component_features = F.normalize(modality_weights[0] * text + modality_weights[1] * image, dim=-1)

        component_logits = self.component_logits[: components.shape[1]].unsqueeze(0).expand(components.shape[0], -1)
        component_logits = component_logits.masked_fill(~component_mask, -1e4)
        component_weights = torch.softmax(component_logits, dim=-1)
        graph = (component_features * component_weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(graph, dim=-1)

    def encode_labels(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(label_embeddings, dim=-1)

    def forward(self, components: torch.Tensor, component_mask: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        graph = self.encode_graph(components, component_mask)
        labels = self.encode_labels(label_embeddings)
        return self.logit_scale.exp().clamp(max=100.0) * graph @ labels.t()

    def extra_state(self) -> Dict[str, object]:
        return {
            "component_weights": torch.softmax(self.component_logits.detach().cpu(), dim=-1).tolist(),
            "modality_weights": torch.softmax(self.modality_logits.detach().cpu(), dim=-1).tolist(),
            "logit_scale": float(self.logit_scale.detach().exp().cpu()),
        }


class LLMVerbalizerGraphScorer(nn.Module):
    """Map graph components into frozen LLM output-embedding verbalizer space."""

    def __init__(self, input_dim: int = 1536, llm_dim: int = 4096, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.llm_dim = llm_dim
        self.component_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
        )
        self.component_score = nn.Linear(llm_dim, 1)
        self.graph_head = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0), dtype=torch.float32))

    def encode_graph(self, components: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        z = self.component_encoder(components)
        scores = self.component_score(z).squeeze(-1)
        scores = scores.masked_fill(~component_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = (z * weights.unsqueeze(-1)).sum(dim=1)
        graph = pooled + self.graph_head(pooled)
        return F.normalize(graph, dim=-1)

    def encode_labels(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(label_embeddings, dim=-1)

    def forward(self, components: torch.Tensor, component_mask: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        graph = self.encode_graph(components, component_mask)
        labels = self.encode_labels(label_embeddings)
        return self.logit_scale.exp().clamp(max=100.0) * graph @ labels.t()

    def extra_state(self) -> Dict[str, object]:
        return {
            "logit_scale": float(self.logit_scale.detach().exp().cpu()),
        }


class HybridLLMClipGraphScorer(nn.Module):
    """Ensemble a trainable LLM verbalizer adapter with the frozen CLIP label prototype score."""

    def __init__(
        self,
        input_dim: int = 1536,
        llm_dim: int = 4096,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        component_weights: Sequence[float] = (1.0, 0.2, 0.2, 0.0),
        modality_weights: Sequence[float] = (1.0, 1.0),
        learn_weights: bool = True,
        logit_scale: float = 20.0,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.llm = LLMVerbalizerGraphScorer(input_dim=input_dim, llm_dim=llm_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.clip = ClipPrototypeGraphScorer(
            component_weights=component_weights,
            modality_weights=modality_weights,
            learn_weights=learn_weights,
            logit_scale=logit_scale,
        )
        self.score_mix_logits = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

    def forward(self, components: torch.Tensor, component_mask: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        llm_labels = label_embeddings[:, : self.llm_dim]
        clip_labels = label_embeddings[:, self.llm_dim :]
        llm_logits = self.llm(components, component_mask, llm_labels)
        clip_logits = self.clip(components, component_mask, clip_labels)
        mix = torch.softmax(self.score_mix_logits, dim=-1)
        return mix[0] * llm_logits + mix[1] * clip_logits

    def extra_state(self) -> Dict[str, object]:
        out = {
            "score_mix": torch.softmax(self.score_mix_logits.detach().cpu(), dim=-1).tolist(),
            "llm_logit_scale": float(self.llm.logit_scale.detach().exp().cpu()),
        }
        out.update({f"clip_{k}": v for k, v in self.clip.extra_state().items()})
        return out


@torch.inference_mode()
def encode_clip_texts(model_path: str, texts: Sequence[str], device: str, batch_size: int) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = CLIPModel.from_pretrained(model_path).to(device)
    model.eval()
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs)
        outputs.append(F.normalize(feats.float(), dim=-1).cpu())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def collect_label_prompt_groups(
    bundles: Dict[str, DataBundle],
    datasets: Sequence[str],
    prompt_mode: str,
) -> Tuple[List[Tuple[str, int, str]], List[str], List[int]]:
    entries = []
    texts = []
    group_sizes = []
    for dataset in datasets:
        spec = bundles[dataset].label_spec
        id2label = spec["valid_id2label"]
        for label_id in sorted(id2label):
            label = id2label[label_id]
            entries.append((dataset, label_id, label))
            label_prompts = label_texts(label, spec, prompt_mode)
            texts.extend(label_prompts)
            group_sizes.append(len(label_prompts))
    return entries, texts, group_sizes


def build_label_bank(
    bundles: Dict[str, DataBundle],
    datasets: Sequence[str],
    clip_model_path: str,
    device: str,
    batch_size: int,
    prompt_mode: str,
) -> Tuple[List[Tuple[str, int, str]], torch.Tensor]:
    entries, texts, group_sizes = collect_label_prompt_groups(bundles, datasets, prompt_mode)
    flat_embeddings = encode_clip_texts(clip_model_path, texts, device=device, batch_size=batch_size)
    embeddings = []
    offset = 0
    for size in group_sizes:
        embeddings.append(F.normalize(flat_embeddings[offset : offset + size].mean(dim=0), dim=-1))
        offset += size
    embeddings = torch.stack(embeddings, dim=0)
    return entries, embeddings


def load_llm_output_embeddings(model_path: str, weight_name: str = "lm_head.weight") -> torch.Tensor:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path, "r", encoding="utf-8"))["weight_map"]
        if weight_name not in weight_map:
            weight_name = "model.embed_tokens.weight"
        shard_path = os.path.join(model_path, weight_map[weight_name])
    else:
        shard_path = os.path.join(model_path, "model.safetensors")
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_name)
    return weight.float()


@torch.inference_mode()
def encode_llm_label_texts(model_path: str, texts: Sequence[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    output_embeddings = load_llm_output_embeddings(model_path)
    rows = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        token_ids = [int(idx) for idx in token_ids if 0 <= int(idx) < output_embeddings.shape[0]]
        if not token_ids:
            token_ids = [tokenizer.unk_token_id or 0]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        rows.append(F.normalize(output_embeddings.index_select(0, token_tensor).mean(dim=0), dim=0))
    del output_embeddings
    return torch.stack(rows, dim=0)


def build_llm_label_bank(
    bundles: Dict[str, DataBundle],
    datasets: Sequence[str],
    llm_model_path: str,
    prompt_mode: str,
) -> Tuple[List[Tuple[str, int, str]], torch.Tensor]:
    entries, texts, group_sizes = collect_label_prompt_groups(bundles, datasets, prompt_mode)
    flat_embeddings = encode_llm_label_texts(llm_model_path, texts)
    embeddings = []
    offset = 0
    for size in group_sizes:
        embeddings.append(F.normalize(flat_embeddings[offset : offset + size].mean(dim=0), dim=-1))
        offset += size
    return entries, torch.stack(embeddings, dim=0)


def build_label_bank_for_model(
    args: argparse.Namespace,
    bundles: Dict[str, DataBundle],
    datasets: Sequence[str],
) -> Tuple[List[Tuple[str, int, str]], torch.Tensor]:
    if args.model_type == "llm_verbalizer":
        return build_llm_label_bank(bundles, datasets, args.llm_model_path, args.label_prompt_mode)
    if args.model_type == "hybrid_llm_clip":
        llm_entries, llm_embeddings = build_llm_label_bank(bundles, datasets, args.llm_model_path, args.label_prompt_mode)
        clip_entries, clip_embeddings = build_label_bank(
            bundles, datasets, args.clip_model_path, args.device, args.label_encode_batch_size, args.label_prompt_mode
        )
        if llm_entries != clip_entries:
            raise ValueError("LLM and CLIP label banks produced different label orders")
        return llm_entries, torch.cat([llm_embeddings, clip_embeddings], dim=-1)
    return build_label_bank(
        bundles, datasets, args.clip_model_path, args.device, args.label_encode_batch_size, args.label_prompt_mode
    )


def build_samples(
    bundles: Dict[str, DataBundle],
    datasets: Sequence[str],
    bank_index: Dict[Tuple[str, int], int],
    ratio: float,
    max_samples: int,
    seed: int,
    sample_mode: str,
    split: str = "train",
) -> List[Tuple[str, int, int]]:
    samples = []
    for offset, dataset in enumerate(datasets):
        bundle = bundles[dataset]
        indices = select_indices(
            labels=bundle.labels,
            valid_label_ids=bundle.label_spec["valid_id2label"].keys(),
            ratio=ratio,
            max_samples=max_samples,
            seed=seed + offset,
            stratified=(sample_mode == "stratified"),
            split_mask=bundle.masks.get(split, bundle.masks["all"]),
        )
        if sample_mode == "first" and max_samples > 0:
            indices = indices[:max_samples]
        for idx in indices:
            label_id = int(bundle.labels[idx].item())
            if (dataset, label_id) in bank_index:
                samples.append((dataset, idx, bank_index[(dataset, label_id)]))
    random.Random(seed).shuffle(samples)
    return samples


def collate_batch(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "components": torch.stack([x["components"] for x in items], dim=0),
        "component_mask": torch.stack([x["component_mask"] for x in items], dim=0),
        "target": torch.stack([x["target"] for x in items], dim=0),
        "node_id": torch.stack([x["node_id"] for x in items], dim=0),
    }


def create_model(args: argparse.Namespace, input_dim: int, dropout: float) -> nn.Module:
    if args.model_type == "mlp":
        return GraphLabelAligner(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            dropout=dropout,
        )
    if args.model_type == "clip_prototype":
        if input_dim != 1536:
            raise ValueError(f"clip_prototype expects concatenated 768 text + 768 image features, got {input_dim}")
        return ClipPrototypeGraphScorer(
            component_weights=parse_float_list(args.component_weights, 4, "--component-weights"),
            modality_weights=parse_float_list(args.modality_weights, 2, "--modality-weights"),
            learn_weights=args.learn_graph_weights,
            logit_scale=args.prototype_logit_scale,
        )
    if args.model_type == "llm_verbalizer":
        return LLMVerbalizerGraphScorer(
            input_dim=input_dim,
            llm_dim=args.llm_hidden_dim,
            hidden_dim=args.hidden_dim,
            dropout=dropout,
        )
    if args.model_type == "hybrid_llm_clip":
        if input_dim != 1536:
            raise ValueError(f"hybrid_llm_clip expects concatenated 768 text + 768 image features, got {input_dim}")
        return HybridLLMClipGraphScorer(
            input_dim=input_dim,
            llm_dim=args.llm_hidden_dim,
            hidden_dim=args.hidden_dim,
            dropout=dropout,
            component_weights=parse_float_list(args.component_weights, 4, "--component-weights"),
            modality_weights=parse_float_list(args.modality_weights, 2, "--modality-weights"),
            learn_weights=True,
            logit_scale=args.prototype_logit_scale,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def train_one_run(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_datasets = [x.strip() for x in args.train_datasets.split(",") if x.strip()]
    eval_dataset = args.eval_dataset.strip()
    all_datasets = list(dict.fromkeys(train_datasets + [eval_dataset]))
    bundles = {name: load_bundle(args.magb_root, name) for name in all_datasets}
    synthetic = SyntheticGraphContext(args.synthetic_root)

    train_bank_entries, train_label_embeddings = build_label_bank_for_model(args, bundles, train_datasets)
    train_bank_index = {(dataset, label_id): i for i, (dataset, label_id, _) in enumerate(train_bank_entries)}
    train_samples = build_samples(
        bundles,
        train_datasets,
        train_bank_index,
        args.train_sample_ratio,
        args.max_train_samples,
        args.seed,
        args.train_sample_mode,
        split=args.train_split,
    )
    if not train_samples:
        raise ValueError("No training samples selected.")

    train_dataset = MAGBContrastiveDataset(
        bundles=bundles,
        synthetic=synthetic,
        samples=train_samples,
        ppr_k=args.ppr_k,
        bridge_k=args.bridge_k,
        include_abstract=not args.no_abstract,
        component_dropout=args.component_dropout,
        cache_components=args.cache_components,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=args.device.startswith("cuda"),
    )

    model = create_model(args, input_dim=bundles[train_datasets[0]].features.shape[1], dropout=args.dropout).to(args.device)
    label_embeddings = train_label_embeddings.to(args.device)
    params = trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay) if params else None

    class_counts = torch.zeros(len(train_bank_entries), dtype=torch.float32)
    for _, _, target in train_samples:
        class_counts[target] += 1
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).to(args.device)
    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)

    global_step = 0
    if opt is None or args.epochs <= 0:
        print(
            json.dumps(
                {
                    "stage": "skipped_train",
                    "reason": "no_trainable_parameters",
                    "model_type": args.model_type,
                    "train_samples": len(train_samples),
                }
            ),
            flush=True,
        )
    else:
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            correct = 0
            seen = 0
            for batch in loader:
                components = batch["components"].to(args.device, non_blocking=True)
                component_mask = batch["component_mask"].to(args.device, non_blocking=True)
                target = batch["target"].to(args.device, non_blocking=True)
                logits = model(components, component_mask, label_embeddings)
                loss = F.cross_entropy(logits, target, weight=class_weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                running += float(loss.item()) * target.numel()
                pred = logits.argmax(dim=-1)
                correct += int((pred == target).sum().item())
                seen += int(target.numel())
                global_step += 1
                if args.log_steps > 0 and global_step % args.log_steps == 0:
                    payload = {
                        "stage": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": running / max(seen, 1),
                        "acc": correct / max(seen, 1),
                        "samples": seen,
                    }
                    if hasattr(model, "extra_state"):
                        payload.update(model.extra_state())
                    print(json.dumps(payload), flush=True)

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "model_type": args.model_type,
        "model_extra": model.extra_state() if hasattr(model, "extra_state") else {},
        "train_bank_entries": train_bank_entries,
        "train_label_embeddings": train_label_embeddings,
    }
    ckpt_path = os.path.join(args.output_dir, "graph_label_aligner.pt")
    torch.save(ckpt, ckpt_path)
    print(json.dumps({"stage": "saved", "checkpoint": ckpt_path, "train_samples": len(train_samples)}), flush=True)
    return ckpt_path


@torch.inference_mode()
def evaluate(args: argparse.Namespace, ckpt_path: str):
    eval_dataset = args.eval_dataset.strip()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = state.get("args", {})
    if args.model_type == "auto":
        args.model_type = str(state.get("model_type", ckpt_args.get("model_type", "mlp")))
    bundles = {eval_dataset: load_bundle(args.magb_root, eval_dataset)}
    synthetic = SyntheticGraphContext(args.synthetic_root)
    eval_bank_entries, eval_label_embeddings = build_label_bank_for_model(args, bundles, [eval_dataset])
    eval_bank_index = {(dataset, label_id): i for i, (dataset, label_id, _) in enumerate(eval_bank_entries)}
    eval_samples = build_samples(
        bundles,
        [eval_dataset],
        eval_bank_index,
        ratio=0.0,
        max_samples=args.eval_max_samples,
        seed=args.seed,
        sample_mode=args.eval_sample_mode,
        split=args.eval_split,
    )
    eval_ds = MAGBContrastiveDataset(
        bundles=bundles,
        synthetic=synthetic,
        samples=eval_samples,
        ppr_k=args.ppr_k,
        bridge_k=args.bridge_k,
        include_abstract=not args.no_abstract,
        component_dropout=0.0,
        cache_components=False,
    )
    loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    input_dim = bundles[eval_dataset].features.shape[1]
    model = create_model(args, input_dim=input_dim, dropout=0.0).to(args.device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    label_embeddings = eval_label_embeddings.to(args.device)

    y_true = []
    y_pred = []
    answer_path = args.answers_file or os.path.join(args.output_dir, f"{eval_dataset.lower()}_contrastive_answers.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(answer_path)), exist_ok=True)
    id2label = bundles[eval_dataset].label_spec["valid_id2label"]
    with open(answer_path, "w", encoding="utf-8") as handle:
        sample_offset = 0
        for batch in loader:
            components = batch["components"].to(args.device)
            component_mask = batch["component_mask"].to(args.device)
            logits = model(components, component_mask, label_embeddings)
            pred_offsets = logits.argmax(dim=-1).cpu().tolist()
            for i, pred_offset in enumerate(pred_offsets):
                dataset, node_id, target_offset = eval_samples[sample_offset + i]
                gt_label_id = int(bundles[dataset].labels[node_id].item())
                pred_label_id = int(eval_bank_entries[pred_offset][1])
                y_true.append(gt_label_id)
                y_pred.append(pred_label_id)
                top = torch.topk(logits[i].detach().cpu(), k=min(5, logits.shape[1]))
                handle.write(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "node_id": node_id,
                            "gt_label_id": gt_label_id,
                            "gt_label": id2label.get(gt_label_id, str(gt_label_id)),
                            "pred_label_id": pred_label_id,
                            "pred_label": id2label.get(pred_label_id, str(pred_label_id)),
                            "top5": [
                                {
                                    "label_id": int(eval_bank_entries[j][1]),
                                    "label": id2label.get(int(eval_bank_entries[j][1]), str(eval_bank_entries[j][1])),
                                    "score": float(score),
                                }
                                for score, j in zip(top.values.tolist(), top.indices.tolist())
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            sample_offset += len(pred_offsets)

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    result = {
        "stage": "eval",
        "dataset": eval_dataset,
        "num_samples": len(y_true),
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "answers_file": answer_path,
        "checkpoint": ckpt_path,
        "model_type": args.model_type,
        "label_prompt_mode": args.label_prompt_mode,
    }
    if hasattr(model, "extra_state"):
        result.update(model.extra_state())
    print(json.dumps(result, ensure_ascii=False), flush=True)
    with open(os.path.join(args.output_dir, f"{eval_dataset.lower()}_contrastive_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--magb-root", default="data/MAGB")
    parser.add_argument(
        "--synthetic-root",
        default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic_cliptext_bridge",
    )
    parser.add_argument("--clip-model-path", default="models/clip-vit-large-patch14-336-hf")
    parser.add_argument("--llm-model-path", default="models/llava-v1.5-7b-merged-localclip")
    parser.add_argument("--train-datasets", default="Grocery,Movies,Reddit-S")
    parser.add_argument("--eval-dataset", default="Toys")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answers-file", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-type",
        choices=["clip_prototype", "mlp", "llm_verbalizer", "hybrid_llm_clip", "auto"],
        default="clip_prototype",
    )
    parser.add_argument(
        "--label-prompt-mode",
        choices=["auto", "name", "category", "short", "semantic", "subject", "belongs", "short_belongs", "short_category", "all"],
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--llm-hidden-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--component-dropout", type=float, default=0.1)
    parser.add_argument("--component-weights", default="1.0,0.2,0.2,0.0001")
    parser.add_argument("--modality-weights", default="1.0,1.0")
    parser.add_argument("--prototype-logit-scale", type=float, default=20.0)
    parser.add_argument("--learn-graph-weights", action="store_true")
    parser.add_argument("--train-sample-ratio", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--train-split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--train-sample-mode", choices=["first", "random", "stratified"], default="stratified")
    parser.add_argument("--eval-max-samples", type=int, default=1000)
    parser.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--eval-sample-mode", choices=["first", "random", "stratified"], default="stratified")
    parser.add_argument("--ppr-k", type=int, default=16)
    parser.add_argument("--bridge-k", type=int, default=4)
    parser.add_argument("--no-abstract", action="store_true")
    parser.add_argument("--cache-components", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--label-encode-batch-size", type=int, default=64)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.model_type == "auto" and not args.eval_only:
        args.model_type = "clip_prototype"
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --eval-only")
        evaluate(args, args.checkpoint)
        return
    ckpt_path = train_one_run(args)
    evaluate(args, ckpt_path)


if __name__ == "__main__":
    main()
