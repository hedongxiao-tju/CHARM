#!/usr/bin/env python3
"""MLaGA-style zero-shot link prediction.

Binary classification QA: determine whether two product nodes are connected
(purchased together, reviewed together, or associated by similar user behavior).

Three transfer settings:
  Grocery + Movies -> Toys
  Grocery + Toys   -> Movies
  Toys   + Movies  -> Grocery

Each source domain: 5000 pos + 5000 neg  (combined: 10000 pos + 10000 neg)
Target domain:      1600 pos + 1600 neg

Each sample contains a target pair (u, v) and an unlabeled reference pair
(ref_a, ref_b) from the 2-hop / PPR neighborhood.  The reference pair's
connection label is NOT provided in the prompt (zero-shot setting).

Training: supervised on the query pair's Yes/No label only.
Testing:  option-logit scoring on Yes / No.

Reports: Accuracy, F1, ROC-AUC, Average Precision.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from model.constants import (
    DEFAULT_GRAPH_PAD_ID,
    GRAPH_TOKEN_INDEX,
    IGNORE_INDEX,
    DEFAULT_GRAPH_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from model.conversation import conv_templates
import model.conversation as conversation_lib
from model.model.builder import load_pretrained_model
from model.mm_utils import get_model_name_from_path
from model.utils import disable_torch_init
from model.train.train import (
    DataArguments,
    LazyGraphSupervisedDataset,
    _clean_graph_text,
    _fixed_graph_hops,
    SyntheticGraphContext,
)
from model.eval.eval_model_graph_textcls import infer_conv_mode, infer_model_name


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--model-base",
        type=str,
        default="models/llava-v1.5-7b-merged-localclip",
    )
    p.add_argument("--source-domains", type=str, nargs="+", required=True)
    p.add_argument("--target-domain", type=str, required=True)
    p.add_argument("--magb-root", type=str, default="data/MAGB")
    p.add_argument(
        "--synthetic-root",
        type=str,
        default="data/MAGB/realnode_kmeans_product3_g24_topm3_semantic_cliptext_bridge",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--graph-hops", type=int, default=2)
    p.add_argument("--graph-sample-size", type=int, default=4)
    p.add_argument("--graph-real-neighbor-topk", type=int, default=8)
    p.add_argument("--graph-real-cross-topk", type=int, default=4)
    p.add_argument("--model-max-length", type=int, default=3072)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--bits", type=int, default=16)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor")
    p.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma, 0=disabled")
    p.add_argument("--trick-name", type=str, default="", help="Trick name for logging")
    p.add_argument("--contrastive-weight", type=float, default=0.0, help="Contrastive loss weight")
    p.add_argument("--hard-neg-weight", type=float, default=0.0, help="Hard negative upweight factor")
    p.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha, 0=disabled")
    p.add_argument("--eval-mode", type=str, default="logit", choices=["logit", "generate"],
                   help="logit: option-logit scoring; generate: free-generation + string match")
    p.add_argument("--optimize-metric", type=str, default="acc", choices=["acc", "f1"],
                   help="Metric to optimize threshold/calibration for: acc or f1")
    return p.parse_args()


# ============================================================================
# Domain name mapping
# ============================================================================

DOMAIN_NAMES = {
    "Grocery": "Grocery & Gourmet Food",
    "Toys": "Toys & Games",
    "Movies": "Movies & TV",
}


# ============================================================================
# Patch encode_graphs for multi-graph (G>1)
# ============================================================================

def patch_model_for_multigraph(model):
    encoder = getattr(model.model, "graph_soft_token_encoder", None)
    if encoder is None:
        raise RuntimeError("No graph_soft_token_encoder found; check the base model and checkpoint")

    original_forward = encoder.forward.__func__

    def patched_forward(
        self,
        graph_emb,
        embed_tokens,
        graph_node_mask=None,
        graph_hops=None,
        graph_degrees=None,
        graph_ppr_scores=None,
        graph_role_ids=None,
        graph_role_mask=None,
        graph_text_mask=None,
        graph_image_mask=None,
        graph_domain_id=None,
        graph_label_id=None,
    ):
        if graph_emb.dim() != 4:
            return original_forward(
                self,
                graph_emb,
                embed_tokens,
                graph_node_mask,
                graph_hops,
                graph_degrees,
                graph_ppr_scores,
                graph_role_ids,
                graph_role_mask,
                graph_text_mask,
                graph_image_mask,
                graph_domain_id,
                graph_label_id,
            )

        B, G, N, H = graph_emb.shape

        def flatten(t):
            if t is None:
                return None
            if t.dim() == 4:
                return t.reshape(B * G, *t.shape[2:])
            elif t.dim() == 3:
                return t.reshape(B * G, N)
            elif t.dim() == 2:
                return t.reshape(B * G)
            return t

        output = original_forward(
            self,
            flatten(graph_emb),
            embed_tokens,
            flatten(graph_node_mask),
            flatten(graph_hops),
            graph_degrees,
            flatten(graph_ppr_scores),
            flatten(graph_role_ids),
            flatten(graph_role_mask),
            flatten(graph_text_mask),
            flatten(graph_image_mask),
            flatten(graph_domain_id) if graph_domain_id is not None else None,
            graph_label_id,
        )

        if isinstance(output, torch.Tensor):
            output = output.reshape(B, G, *output.shape[1:])
        return output

    import types

    encoder.forward = types.MethodType(patched_forward, encoder)
    print("Patched GraphSoftTokenEncoder.forward for multi-graph (G>1)")


# ============================================================================
# Data args builder
# ============================================================================

def build_lp_data_args(data_root, synthetic_root, args):
    return DataArguments(
        graph_data_root=data_root,
        graph_feat_path="clip_feat.pt",
        graph_label_config="graph_label_config.json",
        graph_synthetic_root=synthetic_root,
        graph_label_supervision=True,
        graph_only=True,
        graph_zero_shot_serializer=True,
        graph_include_label_descriptions=True,
        graph_answer_with_description=False,
        graph_label_prompt_style="default",
        graph_context_prompt_style="full",
        graph_option_label_prompt=False,
        graph_shuffle_label_options=False,
        graph_label_option_seed=0,
        graph_hops=args.graph_hops,
        graph_sample_size=args.graph_sample_size,
        graph_include_abstract_nodes=True,
        graph_include_synthetic_soft_tokens=True,
        graph_real_node_serializer=True,
        graph_real_neighbor_topk=args.graph_real_neighbor_topk,
        graph_real_cross_topk=args.graph_real_cross_topk,
        graph_include_cross_domain_soft_tokens=True,
        graph_semantic_coord_tokens=False,
        graph_semantic_coord_topk=3,
        graph_semantic_coord_graph_topk=4,
        graph_semantic_coord_neighbor_topk=0,
        graph_semantic_coord_transfer_topk=3,
        graph_semantic_coord_use_global_nodes=True,
        graph_semantic_coord_global_alignment=True,
        graph_semantic_coord_hybrid_prompt=False,
        graph_semantic_coord_evidence_prompt=True,
        graph_semantic_coord_center_only=False,
        graph_semantic_coord_use_abstract_token=True,
        graph_semantic_coord_use_calibration_token=True,
        graph_semantic_coord_use_transfer_token=True,
        graph_max_consistent_neighbors=4,
        graph_max_complement_neighbors=4,
        graph_max_conflict_neighbors=2,
        graph_min_abstract_similarity=-1.0,
        graph_min_bridge_confidence=-1.0,
        graph_cache_path=None,
        graph_feat_cache_path=None,
        max_train_samples=-1,
    )


# ============================================================================
# Build node subgraph context (reused from su_zeroshot)
# ============================================================================

def build_node_subgraph(dataset, node_id, label_tag):
    record = dataset.records[node_id]
    asin = record["asin"]
    resolved_node_id = dataset._resolve_node_id(asin, node_id)

    if dataset.graph_ids_cache is not None:
        graph_ids = dataset.graph_ids_cache[node_id].to(torch.long)
    else:
        graph_ids = torch.tensor(
            dataset._serialize_node_context(resolved_node_id), dtype=torch.long
        )

    graph_node_mask = graph_ids.ne(DEFAULT_GRAPH_PAD_ID)
    feat_dim = dataset.graph_features.shape[1]
    graph_targets = torch.zeros(
        (graph_ids.shape[0], feat_dim), dtype=dataset.graph_features.dtype
    )
    graph_targets[graph_node_mask] = dataset.graph_features[
        graph_ids[graph_node_mask]
    ]

    neighbor_topk = dataset.data_args.graph_real_neighbor_topk
    if neighbor_topk >= 0 and graph_ids.numel() > 1 + neighbor_topk:
        keep = min(1 + neighbor_topk, graph_ids.numel())
        graph_ids = graph_ids[:keep]
        graph_targets = graph_targets[:keep]
        graph_node_mask = graph_node_mask[:keep]

    relation_neighbors = dataset._select_relation_neighbors(
        graph_ids, graph_targets, resolved_node_id
    )

    synthetic = dataset.synthetic_context
    ds_name = dataset.synthetic_dataset_name
    membership = (
        synthetic.membership(ds_name, resolved_node_id)
        if synthetic is not None and ds_name is not None
        else {}
    )

    abstract_lines = []
    extra_nodes = []
    center_feat = graph_targets[0].to(torch.float32)

    if dataset.data_args.graph_include_abstract_nodes and membership:
        for key, role_name in [
            ("layer1", "layer-1 abstract node"),
            ("layer2", "layer-2 high-level abstract node"),
            ("global", "global semantic prototype node"),
        ]:
            node = membership.get(key)
            if node is None:
                continue
            feat = SyntheticGraphContext.abstract_feature(node, feat_dim)
            anchor_sim = None
            if feat is not None:
                anchor_sim = float(
                    F.cosine_similarity(
                        center_feat, feat.to(torch.float32), dim=0
                    ).item()
                )
            summary = _clean_graph_text(
                node.get("summary_text", ""), max_chars=100
            )
            lbl = _clean_graph_text(
                node.get("semantic_label", node.get("id", role_name)),
                max_chars=70,
            )
            reliability = (
                f" reliability={anchor_sim:.2f}." if anchor_sim is not None else ""
            )
            abstract_lines.append(
                f"- {role_name}: {lbl}. {summary}{reliability}"
            )
            if feat is not None and dataset.data_args.graph_include_synthetic_soft_tokens:
                extra_nodes.append(
                    (int(node["global_node_id"]), feat, role_name)
                )

    bridge_lines = []
    if synthetic is not None and ds_name is not None:
        bridge_limit = dataset.data_args.graph_real_cross_topk
        for bridge in synthetic.bridge_neighbors(
            ds_name, resolved_node_id, bridge_limit
        ):
            dst = int(bridge["dst"])
            bridge_lines.append(
                f"- bridge neighbor: {bridge['direction']}; "
                f"conf={bridge['confidence']:.3f}, "
                f"text_sim={bridge['text_similarity']:.3f}, "
                f"image_sim={bridge['image_similarity']:.3f}. "
                f"Evidence: {dataset._record_text_by_node(dst, 60)}"
            )
            if (
                dataset.data_args.graph_include_synthetic_soft_tokens
                and 0 <= dst < len(dataset.records)
                and dataset.graph_features is not None
            ):
                extra_nodes.append(
                    (
                        dst,
                        dataset.graph_features[dst].to(torch.float32),
                        "bridge neighbor",
                    )
                )

    def fmt_neighbors(title, values, limit):
        lines = []
        for item in values[:limit]:
            lines.append(
                f"- node {item['node_id']}: {item['description']} "
                f"text_sim={item['text_similarity']:.3f}, "
                f"image_sim={item['image_similarity']:.3f}. "
                f"Evidence: {_clean_graph_text(item['text'], max_chars=60)}"
            )
        if not lines:
            return f"{title}:\n- none"
        return f"{title}:\n" + "\n".join(lines)

    center_text = _clean_graph_text(
        dataset._get_record_text(record), max_chars=80
    )

    prompt_parts = [f"SUBGRAPH {label_tag}:"]
    if abstract_lines:
        prompt_parts.append(
            "ABSTRACT SEMANTIC CONTEXT:\n" + "\n".join(abstract_lines[:3])
        )
    prompt_parts.append(
        fmt_neighbors(
            "SUPPORTING NEIGHBORS",
            relation_neighbors["modality-consistent"],
            dataset.data_args.graph_max_consistent_neighbors,
        )
    )
    if bridge_lines:
        prompt_parts.append(
            "BRIDGE NEIGHBORS:\n" + "\n".join(bridge_lines[:2])
        )
    else:
        prompt_parts.append(
            fmt_neighbors(
                "COMPLEMENTARY EVIDENCE",
                relation_neighbors["modality-complement"],
                min(dataset.data_args.graph_max_complement_neighbors, 2),
            )
        )
    prompt_parts.append(f"CENTER NODE {label_tag}:\nText: {center_text}")

    role_texts = ["center node"]
    if dataset.ppr is not None:
        role_texts.extend(
            ["PPR neighbor" for _ in range(max(0, graph_ids.numel() - 1))]
        )
    else:
        for hop in _fixed_graph_hops(
            graph_ids.numel(),
            dataset.data_args.graph_hops,
            dataset.data_args.graph_sample_size,
        ).tolist()[1:]:
            role_texts.append("neighbor" if int(hop) > 0 else "center")

    existing = set(int(x.item()) for x in graph_ids[graph_node_mask])
    for nid, feat, role in extra_nodes:
        if int(nid) not in existing:
            existing.add(int(nid))
            graph_ids = torch.cat(
                [graph_ids, torch.tensor([int(nid)], dtype=torch.long)]
            )
            graph_targets = torch.cat(
                [graph_targets, feat.unsqueeze(0).to(graph_targets.dtype)]
            )
            graph_node_mask = torch.cat(
                [graph_node_mask, torch.ones(1, dtype=torch.bool)]
            )
            role_texts.append(role)

    graph_hops = _fixed_graph_hops(
        graph_ids.numel(),
        dataset.data_args.graph_hops,
        dataset.data_args.graph_sample_size,
    )
    if graph_hops.numel() < graph_ids.numel():
        graph_hops = torch.cat(
            [
                graph_hops,
                torch.zeros(
                    graph_ids.numel() - graph_hops.numel(), dtype=torch.long
                ),
            ]
        )

    unk_id = (
        dataset.tokenizer.unk_token_id
        if dataset.tokenizer.unk_token_id is not None
        else 0
    )
    role_token_ids = [
        torch.tensor(
            dataset.tokenizer(text, add_special_tokens=False).input_ids
            or [unk_id],
            dtype=torch.long,
        )
        for text in role_texts
    ]

    return (
        "\n".join(prompt_parts),
        graph_ids,
        graph_targets,
        graph_node_mask,
        graph_hops,
        role_token_ids,
    )


# ============================================================================
# Compute pair-level graph evidence
# ============================================================================

def compute_pair_evidence(dataset, u, v):
    """Compute text-based pair-level structural evidence between u and v."""
    parts = []

    # Common neighbors
    u_nbrs = set()
    v_nbrs = set()
    if dataset.ppr is not None:
        u_ppr = dataset.ppr.get(u, dataset.ppr.get(str(u), []))
        v_ppr = dataset.ppr.get(v, dataset.ppr.get(str(v), []))
        if torch.is_tensor(u_ppr):
            u_ppr = u_ppr.tolist()
        if torch.is_tensor(v_ppr):
            v_ppr = v_ppr.tolist()
        for item in u_ppr:
            nid = int(item[0]) if isinstance(item, (list, tuple)) else int(item)
            if nid != u and nid != v:
                u_nbrs.add(nid)
        for item in v_ppr:
            nid = int(item[0]) if isinstance(item, (list, tuple)) else int(item)
            if nid != v and nid != u:
                v_nbrs.add(nid)
    if dataset.edge_list is not None:
        u_nbrs.update(dataset.edge_list.get(u, set()))
        v_nbrs.update(dataset.edge_list.get(v, set()))

    common = u_nbrs & v_nbrs
    if common:
        common_list = sorted(common)[:3]
        common_desc = []
        for cn in common_list:
            if 0 <= cn < len(dataset.records):
                txt = _clean_graph_text(
                    dataset._record_text_by_node(cn, max_chars=50)
                )
                common_desc.append(f"node {cn} ({txt})")
        parts.append(
            f"Common neighbors: {len(common)} found. "
            f"Top: {'; '.join(common_desc)}"
        )
    else:
        parts.append("Common neighbors: none found in PPR/edge neighborhood.")

    # PPR overlap (Jaccard)
    if u_nbrs or v_nbrs:
        union = u_nbrs | v_nbrs
        overlap = len(u_nbrs & v_nbrs) / max(1, len(union))
        parts.append(f"PPR/neighborhood overlap (Jaccard): {overlap:.3f}")

    # Abstract node sharing
    synthetic = dataset.synthetic_context
    ds_name = dataset.synthetic_dataset_name
    if synthetic is not None and ds_name is not None:
        mem_u = synthetic.membership(ds_name, u)
        mem_v = synthetic.membership(ds_name, v)
        shared_abstract = []
        for key, label in [
            ("layer1", "layer-1 abstract"),
            ("layer2", "layer-2 abstract"),
            ("global", "global prototype"),
        ]:
            nu = mem_u.get(key)
            nv = mem_v.get(key)
            if nu is not None and nv is not None:
                nu_id = nu.get("global_node_id")
                nv_id = nv.get("global_node_id")
                if nu_id == nv_id:
                    lbl = _clean_graph_text(
                        nu.get("semantic_label", label), max_chars=50
                    )
                    shared_abstract.append(f"{label}: {lbl}")
        if shared_abstract:
            parts.append(
                "Shared abstract nodes: " + "; ".join(shared_abstract)
            )
        else:
            parts.append("Shared abstract nodes: none.")

    return "\n".join(parts)


# ============================================================================
# Build LP sample
# ============================================================================

def build_lp_sample(dataset, u, v, ref_a, ref_b, label, tokenizer, args):
    """Build a complete link-prediction sample with 4 graph soft tokens."""
    domain_full = DOMAIN_NAMES.get(
        dataset.synthetic_dataset_name, "product"
    )

    # Build subgraph context for all 4 nodes
    (
        u_prompt, u_ids, u_targets, u_mask, u_hops, u_roles,
    ) = build_node_subgraph(dataset, u, "A")
    (
        v_prompt, v_ids, v_targets, v_mask, v_hops, v_roles,
    ) = build_node_subgraph(dataset, v, "B")
    (
        ra_prompt, ra_ids, ra_targets, ra_mask, ra_hops, ra_roles,
    ) = build_node_subgraph(dataset, ref_a, "Ref-A")
    (
        rb_prompt, rb_ids, rb_targets, rb_mask, rb_hops, rb_roles,
    ) = build_node_subgraph(dataset, ref_b, "Ref-B")

    # Pair-level evidence
    pair_evidence = compute_pair_evidence(dataset, u, v)

    # Text descriptions
    text_u = _clean_graph_text(dataset._record_text_by_node(u, max_chars=100))
    text_v = _clean_graph_text(dataset._record_text_by_node(v, max_chars=100))
    text_ra = _clean_graph_text(dataset._record_text_by_node(ref_a, max_chars=100))
    text_rb = _clean_graph_text(dataset._record_text_by_node(ref_b, max_chars=100))

    answer = "Yes" if label == 1 else "No"

    human_prompt = (
        f"{DEFAULT_GRAPH_TOKEN}\n"
        f"{DEFAULT_GRAPH_TOKEN}\n"
        f"{DEFAULT_GRAPH_TOKEN}\n"
        f"{DEFAULT_GRAPH_TOKEN}\n"
        f"Given two products from the {domain_full} product graph. "
        f"In this graph, a link indicates that two products are purchased "
        f"together, reviewed together, or associated by similar user behavior.\n\n"
        f"Target product pair:\n"
        f"Product A text description:\n{text_u}\n\n"
        f"Product B text description:\n{text_v}\n\n"
        f"Product A graph-aware evidence:\n{u_prompt}\n\n"
        f"Product B graph-aware evidence:\n{v_prompt}\n\n"
        f"Pair-level graph evidence:\n{pair_evidence}\n\n"
        f"Here is one reference product pair selected from the local two-hop "
        f"graph neighborhood or PPR-related neighborhood. "
        f"Its connection label is not provided.\n\n"
        f"Reference product pair:\n"
        f"Reference Product A text description:\n{text_ra}\n\n"
        f"Reference Product B text description:\n{text_rb}\n\n"
        f"Reference Product A graph-aware evidence:\n{ra_prompt}\n\n"
        f"Reference Product B graph-aware evidence:\n{rb_prompt}\n\n"
        f"Based on the target pair's text, image, graph-aware evidence, "
        f"pair-level graph evidence, and the unlabeled reference product pair, "
        f"determine whether the two target products are likely to be connected "
        f"in the product graph.\n\n"
        f"Answer only Yes or No."
    )

    assistant_prompt = answer

    sources = [[
        {"from": "human", "value": human_prompt},
        {"from": "gpt", "value": assistant_prompt},
    ]]

    from model.train.train import preprocess

    data_dict = preprocess(sources, tokenizer, has_image=False, has_graph=True)

    # Collect 4 subgraphs
    subgraph_data = [
        (u_ids, u_targets, u_mask, u_hops, u_roles),
        (v_ids, v_targets, v_mask, v_hops, v_roles),
        (ra_ids, ra_targets, ra_mask, ra_hops, ra_roles),
        (rb_ids, rb_targets, rb_mask, rb_hops, rb_roles),
    ]
    G = 4
    max_nodes = max(s[0].shape[0] for s in subgraph_data)
    feat_dim = subgraph_data[0][1].shape[-1]

    def pad_to(t, target_len, pad_val):
        if t.shape[0] >= target_len:
            return t[:target_len]
        pad_shape = list(t.shape)
        pad_shape[0] = target_len - t.shape[0]
        return torch.cat(
            [t, torch.full(pad_shape, pad_val, dtype=t.dtype)], dim=0
        )

    graph_ids = torch.stack(
        [pad_to(s[0], max_nodes, DEFAULT_GRAPH_PAD_ID) for s in subgraph_data],
        dim=0,
    )
    graph_emb = torch.stack(
        [pad_to(s[1], max_nodes, 0.0) for s in subgraph_data], dim=0
    )
    graph_node_mask = torch.stack(
        [pad_to(s[2], max_nodes, False) for s in subgraph_data], dim=0
    )
    graph_hops = torch.stack(
        [pad_to(s[3], max_nodes, 0) for s in subgraph_data], dim=0
    )
    graph_role_ids = [s[4] for s in subgraph_data]

    return dict(
        input_ids=data_dict["input_ids"][0],
        labels=data_dict["labels"][0],
        graph=graph_ids,
        graph_emb=graph_emb,
        graph_node_mask=graph_node_mask,
        graph_targets=graph_emb,
        graph_hops=graph_hops,
        graph_role_ids=graph_role_ids,
    )


# ============================================================================
# Collate (G=4)
# ============================================================================

def collate_lp_batch(samples, tokenizer, device, dtype):
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    input_ids = [s["input_ids"] for s in samples]
    labels = [s["labels"] for s in samples]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    attention_mask = input_ids.ne(pad_id)

    batch_size = len(samples)
    G = 4
    max_nodes = max(s["graph_emb"].shape[1] for s in samples)
    feat_dim = samples[0]["graph_emb"].shape[-1]

    graphs = torch.full(
        (batch_size, G, max_nodes), DEFAULT_GRAPH_PAD_ID, dtype=torch.long
    )
    graph_emb = torch.zeros(
        (batch_size, G, max_nodes, feat_dim), dtype=dtype
    )
    graph_node_mask = torch.zeros(
        (batch_size, G, max_nodes), dtype=torch.bool
    )
    graph_hops = torch.zeros(
        (batch_size, G, max_nodes), dtype=torch.long
    )

    max_role_len = 1
    for s in samples:
        for span_roles in s["graph_role_ids"]:
            for r in span_roles:
                max_role_len = max(max_role_len, r.numel())
    graph_role_ids = torch.full(
        (batch_size, G, max_nodes, max_role_len), pad_id, dtype=torch.long
    )
    graph_role_mask = torch.zeros(
        (batch_size, G, max_nodes, max_role_len), dtype=torch.bool
    )

    for i, s in enumerate(samples):
        for span_idx in range(G):
            n = s["graph_emb"].shape[1]
            graphs[i, span_idx, :n] = s["graph"][span_idx]
            graph_emb[i, span_idx, :n] = s["graph_emb"][span_idx].to(dtype)
            graph_node_mask[i, span_idx, :n] = s["graph_node_mask"][span_idx]
            graph_hops[i, span_idx, :n] = s["graph_hops"][span_idx]
            for j, role_ids in enumerate(s["graph_role_ids"][span_idx][:n]):
                rl = min(role_ids.numel(), max_role_len)
                graph_role_ids[i, span_idx, j, :rl] = role_ids[:rl]
                graph_role_mask[i, span_idx, j, :rl] = True

    graph_text_mask = torch.zeros_like(graph_node_mask)
    graph_image_mask = torch.zeros_like(graph_node_mask)

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "graphs": graphs.to(device),
        "graph_emb": graph_emb.to(device),
        "graph_node_mask": graph_node_mask.to(device),
        "graph_hops": graph_hops.to(device),
        "graph_role_ids": graph_role_ids.to(device),
        "graph_role_mask": graph_role_mask.to(device),
        "graph_text_mask": graph_text_mask.to(device),
        "graph_image_mask": graph_image_mask.to(device),
    }


# ============================================================================
# Answer token scoring
# ============================================================================

def build_answer_token_ids(tokenizer):
    """Get token IDs for Yes and No."""
    options = ["No", "Yes"]
    option_token_ids = []
    for opt in options:
        ids = set()
        for variant in [opt, opt.capitalize(), opt.upper(), opt.lower()]:
            tids = tokenizer(variant, add_special_tokens=False).input_ids
            if tids:
                if len(tids) == 1:
                    ids.add(int(tids[0]))
                else:
                    ids.update(int(t) for t in tids)
        option_token_ids.append(sorted(ids))
    print(f"Option tokens: No={option_token_ids[0]}, Yes={option_token_ids[1]}")
    return option_token_ids


@torch.no_grad()
@torch.inference_mode()
def score_answer_logits(model, batch, option_token_id_sets):
    """Score each answer option by log-prob of answer tokens."""
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    prompt_lengths = []
    for i in range(input_ids.shape[0]):
        pos = torch.where(labels[i].ne(IGNORE_INDEX))[0]
        if pos.numel() > 0:
            prompt_lengths.append(int(pos[0].item()))
        else:
            prompt_lengths.append(input_ids.shape[1] - 1)

    max_prompt = max(prompt_lengths)
    truncated_ids = input_ids[:, :max_prompt]
    truncated_mask = batch["attention_mask"][:, :max_prompt]

    (
        _, position_ids, mm_attention_mask, _, inputs_embeds, _, _,
    ) = model.prepare_inputs_labels_for_multimodal(
        truncated_ids,
        None,
        truncated_mask,
        None,
        None,
        None,
        graphs=batch["graphs"],
        graph_emb=batch["graph_emb"],
        graph_node_mask=batch["graph_node_mask"],
        graph_hops=batch["graph_hops"],
        graph_ppr_scores=None,
        graph_role_ids=batch["graph_role_ids"],
        graph_role_mask=batch["graph_role_mask"],
        graph_text_mask=batch["graph_text_mask"],
        graph_image_mask=batch["graph_image_mask"],
        return_graph_meta=True,
    )
    inputs_embeds = torch.nan_to_num(
        inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-8.0, 8.0)
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
    hidden = torch.nan_to_num(
        outputs.last_hidden_state, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-16.0, 16.0)
    if mm_attention_mask is None:
        last_positions = torch.full(
            (hidden.shape[0],),
            hidden.shape[1] - 1,
            dtype=torch.long,
            device=hidden.device,
        )
    else:
        positions = torch.arange(
            hidden.shape[1], device=hidden.device
        ).unsqueeze(0)
        last_positions = (positions * mm_attention_mask.long()).argmax(dim=1)
    next_logits = model.lm_head(
        hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            last_positions,
        ]
    )
    next_logits = torch.nan_to_num(
        next_logits, nan=-1e4, posinf=1e4, neginf=-1e4
    ).clamp(-100.0, 100.0)
    log_probs = F.log_softmax(next_logits.float(), dim=-1)

    option_scores = []
    for token_ids in option_token_id_sets:
        if not token_ids:
            option_scores.append(
                torch.full(
                    (log_probs.shape[0],), -1e4, device=log_probs.device
                )
            )
            continue
        ids = torch.tensor(
            token_ids, dtype=torch.long, device=log_probs.device
        )
        option_scores.append(
            torch.logsumexp(
                log_probs.index_select(dim=-1, index=ids), dim=-1
            )
        )
    return torch.stack(option_scores, dim=1).cpu()


# ============================================================================
# Load LP splits
# ============================================================================

def load_lp_zeroshot_split(data_root):
    path = os.path.join(data_root, "lp_zeroshot_split.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LP zeroshot split not found: {path}. "
            f"Run prepare_lp_splits.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================================
# Training
# ============================================================================

def train_model(model, tokenizer, datasets, lp_splits, args):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    all_train_pairs = []
    for ds, split in zip(datasets, lp_splits):
        for sample in split["train"]:
            all_train_pairs.append(
                (ds, sample["u"], sample["v"],
                 sample["ref_a"], sample["ref_b"], sample["label"])
            )
    rng = random.Random(42)
    rng.shuffle(all_train_pairs)

    print(f"  Total training pairs: {len(all_train_pairs)}")

    params = []
    for name, param in model.named_parameters():
        if "mm_projector" in name or "graph_soft_token_encoder" in name:
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        params, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    num_steps = max(
        1, len(all_train_pairs) // args.gradient_accumulation_steps
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps
    )

    print(
        f"  Training: {len(all_train_pairs)} pairs, {num_steps} steps, "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params"
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()
    step = 0
    accum_loss = 0.0
    for epoch in range(args.num_train_epochs):
        for i, (ds, u, v, ref_a, ref_b, label) in enumerate(all_train_pairs):
            try:
                sample = build_lp_sample(
                    ds, u, v, ref_a, ref_b, label, tokenizer, args
                )
            except Exception:
                continue

            if sample["input_ids"].numel() > args.model_max_length:
                continue

            batch = collate_lp_batch([sample], tokenizer, device, dtype)
            try:
                (
                    _, position_ids, mm_attention_mask, _,
                    inputs_embeds, mm_labels, _,
                ) = model.prepare_inputs_labels_for_multimodal(
                    batch["input_ids"],
                    None,
                    batch["attention_mask"],
                    None,
                    batch["labels"],
                    None,
                    graphs=batch["graphs"],
                    graph_emb=batch["graph_emb"],
                    graph_node_mask=batch["graph_node_mask"],
                    graph_hops=batch["graph_hops"],
                    graph_ppr_scores=None,
                    graph_role_ids=batch["graph_role_ids"],
                    graph_role_mask=batch["graph_role_mask"],
                    graph_text_mask=batch["graph_text_mask"],
                    graph_image_mask=batch["graph_image_mask"],
                    return_graph_meta=True,
                )
                inputs_embeds = torch.nan_to_num(
                    inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp(-8.0, 8.0)
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
                hidden = torch.nan_to_num(
                    outputs.last_hidden_state,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(-16.0, 16.0)
                logits = model.lm_head(hidden)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = mm_labels[..., 1:].contiguous()
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)

                if args.focal_gamma > 0:
                    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX, reduction='none')
                    pt = torch.exp(-ce)
                    mask = flat_labels != IGNORE_INDEX
                    loss = ((1 - pt) ** args.focal_gamma * ce)[mask].mean()
                elif args.label_smoothing > 0:
                    n_classes = flat_logits.size(-1)
                    log_probs = F.log_softmax(flat_logits, dim=-1)
                    mask = flat_labels != IGNORE_INDEX
                    safe_labels = flat_labels.clone()
                    safe_labels[~mask] = 0
                    safe_labels = safe_labels.clamp(0, n_classes - 1)
                    one_hot = F.one_hot(safe_labels, n_classes).float()
                    smooth = (1 - args.label_smoothing) * one_hot + args.label_smoothing / n_classes
                    loss = -(smooth * log_probs).sum(dim=-1)
                    loss = loss[mask].mean()
                else:
                    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX)

                # Contrastive auxiliary loss: maximize Yes logit for pos, No logit for neg
                if args.contrastive_weight > 0:
                    answer_pos = (mm_labels != IGNORE_INDEX).nonzero(as_tuple=True)
                    if len(answer_pos[0]) > 0:
                        ans_logits = logits[answer_pos]  # [num_answer_tokens, vocab]
                        ans_labels = mm_labels[answer_pos]  # token ids for Yes/No
                        # Contrastive: push Yes tokens up, No tokens down for pos samples and vice versa
                        yes_ids = torch.tensor([v for v in build_answer_token_ids(tokenizer)[1]], device=ans_logits.device)
                        no_ids = torch.tensor([v for v in build_answer_token_ids(tokenizer)[0]], device=ans_logits.device)
                        yes_logit = ans_logits[:, yes_ids].max(dim=-1).values
                        no_logit = ans_logits[:, no_ids].max(dim=-1).values
                        margin = 1.0
                        is_yes = torch.zeros(len(ans_labels), device=ans_logits.device)
                        for yid in yes_ids: is_yes = is_yes | (ans_labels == yid)
                        # pos samples (Yes): yes_logit > no_logit + margin
                        # neg samples (No): no_logit > yes_logit + margin
                        pos_mask = is_yes.bool()
                        neg_mask = ~pos_mask
                        cl_loss = F.relu(margin - (yes_logit[pos_mask] - no_logit[pos_mask])).mean() if pos_mask.any() else torch.tensor(0.0, device=device)
                        cl_loss += F.relu(margin - (no_logit[neg_mask] - yes_logit[neg_mask])).mean() if neg_mask.any() else torch.tensor(0.0, device=device)
                        loss = loss + args.contrastive_weight * cl_loss

                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [skip] forward error: {e}", flush=True)
                continue

            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0 or step == 1:
                    print(
                        f"  step {step}/{num_steps}, loss={accum_loss:.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                accum_loss = 0.0
                torch.cuda.empty_cache()

    print(f"  Training done. {step} steps completed.")
    return model


# ============================================================================
# Evaluation
# ============================================================================

def find_best_threshold(y_true, y_scores, metric='acc'):
    """Find the threshold that maximizes accuracy or F1 on validation set."""
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return 0.5
    best_val, best_thr = 0, 0.5
    for thr in sorted(set(y_scores)):
        preds = [1 if s > thr else 0 for s in y_scores]
        if metric == 'f1':
            val = f1_score(y_true, preds, average='binary', zero_division=0)
        else:
            val = accuracy_score(y_true, preds)
        if val > best_val:
            best_val = val
            best_thr = thr
    print(f"  Best threshold ({metric}): {best_thr:.4f} (val {metric}={best_val:.4f})")
    return best_thr


def temperature_scaling(val_scores, val_labels, test_scores, T_init=1.5, lr=0.01, n_iter=200):
    """Learn temperature T on val set, apply to test scores."""
    val_s = torch.tensor(val_scores, dtype=torch.float32, requires_grad=False)
    val_l = torch.tensor(val_labels, dtype=torch.float32)
    T = torch.tensor(T_init, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)
    def closure():
        optimizer.zero_grad()
        scaled = val_s / T
        loss = F.binary_cross_entropy_with_logits(scaled, val_l)
        loss.backward()
        return loss
    optimizer.step(closure)
    T_val = T.item()
    test_s = torch.tensor(test_scores, dtype=torch.float32) / T_val
    calibrated = torch.sigmoid(test_s).numpy()
    print(f"  Temperature: T={T_val:.4f}")
    return calibrated


def isotonic_calibration(val_scores, val_labels, test_scores):
    """Isotonic regression calibration: learn monotonic mapping from val set."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    # Convert scores to logits for better isotonic fitting
    val_logits = np.log(np.clip(val_scores, 1e-6, 1-1e-6) / (1 - np.clip(val_scores, 1e-6, 1-1e-6)))
    test_logits = np.log(np.clip(test_scores, 1e-6, 1-1e-6) / (1 - np.clip(test_scores, 1e-6, 1-1e-6)))
    iso.fit(val_logits, val_labels)
    calibrated = iso.predict(test_logits)
    print(f"  Isotonic calibration applied (val samples: {len(val_labels)})")
    return np.clip(calibrated, 0, 1)


def platt_scaling(val_scores, val_labels, test_scores):
    """Platt scaling: fit logistic regression on val scores."""
    from sklearn.linear_model import LogisticRegression
    val_logits = np.log(np.clip(val_scores, 1e-6, 1-1e-6) / (1 - np.clip(val_scores, 1e-6, 1-1e-6)))
    test_logits = np.log(np.clip(test_scores, 1e-6, 1-1e-6) / (1 - np.clip(test_scores, 1e-6, 1-1e-6)))
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(val_logits.reshape(-1, 1), val_labels)
    calibrated = lr.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    print(f"  Platt scaling applied (val samples: {len(val_labels)})")
    return calibrated


@torch.no_grad()
@torch.inference_mode()
def generate_answer(model, dataset, u, v, ref_a, ref_b, label, tokenizer, args):
    """Generate text answer by running model forward and checking Yes/No in response."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    sample = build_lp_sample(dataset, u, v, ref_a, ref_b, label, tokenizer, args)
    if sample["input_ids"].numel() > args.model_max_length:
        sample["input_ids"] = sample["input_ids"][:args.model_max_length]
        sample["labels"] = sample["labels"][:args.model_max_length]

    batch = collate_lp_batch([sample], tokenizer, device, dtype)

    # build_lp_sample includes the answer token (Yes/No + EOS) at the end of input_ids.
    # For generation we must strip those off, keeping only up to and including
    # the ASSISTANT: colon token (29901), so the model generates the answer itself.
    # The labels tensor marks answer tokens as non-IGNORE_INDEX; use that to find cutoff.
    raw_ids = batch["input_ids"][0]          # shape (seq_len,)
    raw_labels = sample["labels"]            # shape (seq_len,)  on CPU; IGNORE_INDEX=-100 for non-answer
    # answer positions = where labels != IGNORE_INDEX
    ans_positions = (raw_labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if len(ans_positions) > 0:
        cutoff = ans_positions[0].item()     # first answer token position
    else:
        cutoff = raw_ids.shape[0]            # no answer found, use full seq

    prompt_ids = batch["input_ids"][:, :cutoff]     # (1, prompt_len)
    prompt_mask = batch["attention_mask"][:, :cutoff]

    gen_kwargs = dict(max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    with torch.no_grad():
        try:
            output_ids = model.generate(
                prompt_ids,
                graphs=batch["graphs"],
                graph_emb=batch["graph_emb"],
                graph_node_mask=batch["graph_node_mask"],
                graph_hops=batch["graph_hops"],
                graph_role_ids=batch["graph_role_ids"],
                graph_role_mask=batch["graph_role_mask"],
                graph_text_mask=batch["graph_text_mask"],
                graph_image_mask=batch["graph_image_mask"],
                attention_mask=prompt_mask,
                **gen_kwargs,
            )
        except Exception as exc:
            raise RuntimeError("LP answer generation failed") from exc

    # output_ids contains only newly generated tokens (inputs_embeds path skips input)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated.strip()


@torch.no_grad()
@torch.inference_mode()
def evaluate_model_generate(model, dataset, lp_split, tokenizer, args):
    """Evaluate LP using free-generation + string matching (same protocol as MLaGA)."""
    test_samples = lp_split["test"]
    print(f"\n  Evaluating {len(test_samples)} test samples (free-generation)...")

    y_true, y_pred = [], []
    skip_count = 0

    for idx in range(len(test_samples)):
        s = test_samples[idx]
        u, v, ref_a, ref_b, label = s["u"], s["v"], s["ref_a"], s["ref_b"], s["label"]
        try:
            ans = generate_answer(model, dataset, u, v, ref_a, ref_b, label, tokenizer, args)
        except Exception as exc:
            raise RuntimeError(f"LP evaluation failed at test sample {idx}") from exc

        if ans is None:
            raise RuntimeError(f"LP generation returned no answer for test sample {idx}")

        y_true.append(label)
        if "Yes" in ans or "yes" in ans:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(test_samples):
            print(f"  eval {idx + 1}/{len(test_samples)}, skipped={skip_count}", flush=True)

    if skip_count > 0:
        print(f"  total skipped: {skip_count}/{len(test_samples)}", flush=True)

    if not y_true:
        print("No evaluation results!")
        return {}

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "num_samples": len(y_true),
        "eval_mode": "generate",
    }

    print(f"\n  Link Prediction Results (free-generation):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    return metrics


@torch.no_grad()
@torch.inference_mode()
def evaluate_model(model, dataset, lp_split, tokenizer, args):
    """Score test edges directly; classify with a fixed probability threshold."""
    samples = lp_split["test"]
    if not samples:
        raise ValueError("The LP test split is empty")
    option_token_id_sets = build_answer_token_ids(tokenizer)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    y_true, y_scores = [], []
    print(f"\n  Scoring test set ({len(samples)} samples)...")
    for idx, sample_info in enumerate(samples):
        try:
            u, v = sample_info["u"], sample_info["v"]
            ref_a, ref_b, label = sample_info["ref_a"], sample_info["ref_b"], sample_info["label"]
            sample = build_lp_sample(dataset, u, v, ref_a, ref_b, label, tokenizer, args)
            if sample["input_ids"].numel() > args.model_max_length:
                sample["input_ids"] = sample["input_ids"][:args.model_max_length]
                sample["labels"] = sample["labels"][:args.model_max_length]
            batch = collate_lp_batch([sample], tokenizer, device, dtype)
            scores = score_answer_logits(model, batch, option_token_id_sets)
            probability = float(torch.sigmoid(scores[0, 1] - scores[0, 0]).item())
        except Exception as exc:
            raise RuntimeError(f"LP evaluation failed at test sample {idx}") from exc
        y_true.append(label)
        y_scores.append(probability)
        if (idx + 1) % 200 == 0 or idx + 1 == len(samples):
            print(f"  eval {idx + 1}/{len(samples)}", flush=True)
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    y_pred = (y_scores > 0.5).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_scores)) if both_classes else None,
        "Average Precision": float(average_precision_score(y_true, y_scores)) if both_classes else None,
        "threshold": 0.5,
        "num_samples": len(y_true),
        "eval_mode": "logit",
    }
    print("\n  Link Prediction Results (fixed threshold 0.5):")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}")
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    disable_torch_init()
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = infer_model_name(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=(args.bits == 8),
        load_4bit=(args.bits == 4),
        device=args.device,
    )
    tokenizer.model_max_length = args.model_max_length
    model.eval()
    conv_mode = infer_conv_mode(model_name, "v1")
    conversation_lib.default_conversation = conv_templates[conv_mode]
    print(f"Model: {model_name}, conv_mode: {conv_mode}")

    patch_model_for_multigraph(model)

    # Load source domains
    source_datasets, source_splits = [], []
    for domain in args.source_domains:
        data_root = os.path.join(args.magb_root, domain)
        data_args = build_lp_data_args(data_root, args.synthetic_root, args)
        data_args = replace(data_args)
        data_args.image_processor = image_processor
        data_args.mm_use_im_start_end = getattr(
            model.config, "mm_use_im_start_end", False
        )
        ds = LazyGraphSupervisedDataset(
            tokenizer=tokenizer, data_args=data_args
        )
        ds.sample_indices = list(range(len(ds.records)))
        split = load_lp_zeroshot_split(data_root)
        source_datasets.append(ds)
        source_splits.append(split)
        print(f"Source {domain}: {len(ds.records)} nodes")
        n_train = len(split["train"])
        n_test = len(split["test"])
        n_train_pos = sum(1 for s in split["train"] if s["label"] == 1)
        print(f"  train={n_train} ({n_train_pos} pos)")

    # Load target domain
    target_root = os.path.join(args.magb_root, args.target_domain)
    target_args = build_lp_data_args(
        target_root, args.synthetic_root, args
    )
    target_args = replace(target_args)
    target_args.image_processor = image_processor
    target_args.mm_use_im_start_end = getattr(
        model.config, "mm_use_im_start_end", False
    )
    target_ds = LazyGraphSupervisedDataset(
        tokenizer=tokenizer, data_args=target_args
    )
    target_ds.sample_indices = list(range(len(target_ds.records)))
    target_split = load_lp_zeroshot_split(target_root)
    print(f"Target {args.target_domain}: {len(target_ds.records)} nodes")
    n_test = len(target_split["test"])
    n_test_pos = sum(1 for s in target_split["test"] if s["label"] == 1)
    print(f"  test={n_test} ({n_test_pos} pos)")

    if not args.eval_only:
        if args.load_checkpoint:
            ckpt_dir = args.load_checkpoint
            mp = os.path.join(ckpt_dir, "mm_projector.bin")
            gs = os.path.join(ckpt_dir, "graph_soft_token_encoder.bin")
            if os.path.exists(mp):
                sd = torch.load(mp)
                sd = {k.replace("model.mm_projector.", "").replace("model.graph_soft_token_encoder.", ""): v for k, v in sd.items()}
                model.model.mm_projector.load_state_dict({k: v for k, v in sd.items() if k in model.model.mm_projector.state_dict()}, strict=False)
                if hasattr(model.model, "graph_soft_token_encoder"):
                    model.model.graph_soft_token_encoder.load_state_dict({k: v for k, v in sd.items() if k in model.model.graph_soft_token_encoder.state_dict()}, strict=False)
                print(f"Loaded mm_projector from {mp}")
            if (
                os.path.exists(gs)
                and hasattr(model.model, "graph_soft_token_encoder")
                and model.model.graph_soft_token_encoder is not None
            ):
                gsd = torch.load(gs); gsd = {k.replace("model.graph_soft_token_encoder.", ""): v for k, v in gsd.items() if k.startswith("model.graph_soft_token_encoder.")}; model.model.graph_soft_token_encoder.load_state_dict(gsd, strict=False)
                print(f"Loaded graph_soft_token_encoder from {gs}")
        print("\n===== Training =====")
        model = train_model(
            model, tokenizer, source_datasets, source_splits, args
        )

        ckpt_dir = os.path.join(args.output_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            model.model.mm_projector.state_dict(),
            os.path.join(ckpt_dir, "mm_projector.bin"),
        )
        if (
            hasattr(model.model, "graph_soft_token_encoder")
            and model.model.graph_soft_token_encoder is not None
        ):
            torch.save(
                model.model.graph_soft_token_encoder.state_dict(),
                os.path.join(ckpt_dir, "graph_soft_token_encoder.bin"),
            )
        print(f"Checkpoint saved to {ckpt_dir}")
    elif args.load_checkpoint:
        ckpt_dir = args.load_checkpoint
        mp = os.path.join(ckpt_dir, "mm_projector.bin")
        gs = os.path.join(ckpt_dir, "graph_soft_token_encoder.bin")
        if os.path.exists(mp):
            sd = torch.load(mp); mm_sd = {k.replace("model.mm_projector.", ""): v for k, v in sd.items() if k.startswith("model.mm_projector.")}; model.model.mm_projector.load_state_dict(mm_sd, strict=False)
            print(f"Loaded mm_projector from {mp}")
        if (
            os.path.exists(gs)
            and hasattr(model.model, "graph_soft_token_encoder")
            and model.model.graph_soft_token_encoder is not None
        ):
            model.model.graph_soft_token_encoder.load_state_dict(
                torch.load(gs)
            )
            print(f"Loaded graph_soft_token_encoder from {gs}")

    print("\n===== Evaluation =====")
    if args.eval_mode == "generate":
        metrics = evaluate_model_generate(
            model, target_ds, target_split, tokenizer, args
        )
    else:
        metrics = evaluate_model(
            model, target_ds, target_split, tokenizer, args
        )

    print("\n" + "=" * 60)
    print(
        f"Transfer: {' + '.join(args.source_domains)} -> {args.target_domain}"
    )
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)

    result = {
        "source_domains": args.source_domains,
        "target_domain": args.target_domain,
        "metrics": metrics,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
