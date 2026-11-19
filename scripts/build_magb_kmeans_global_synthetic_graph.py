#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

# MM-SC: hierarchical clustering guided by structure and semantics
try:
    from mm_structure_cluster import mm_structure_clusters, attentive_aggregate_embeddings
except Exception:  # Allow execution from another working directory
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from mm_structure_cluster import mm_structure_clusters, attentive_aggregate_embeddings


DATASETS = {
    "Grocery": {"csv": "Grocery.csv", "text_col": "text", "edge_cols": ["also_buy", "also_view"]},
    "Toys": {"csv": "Toys.csv", "text_col": "text", "edge_cols": ["also_buy", "also_view"]},
    "Movies": {"csv": "Movies.csv", "text_col": "text", "edge_cols": ["also_buy", "also_view"]},
    "Reddit-S": {"csv": "RedditS.csv", "text_col": "caption", "edge_cols": ["also_posted"]},
    "Health": {"jsonl": "health-raw-text.jsonl", "text_col": "raw_text", "graph_pt": "HealthGraph.pt"},
    "Beauty": {"jsonl": "beauty-raw-text.jsonl", "text_col": "raw_text", "graph_pt": "BeautyGraph.pt"},
    "Arts": {"jsonl": "arts-raw-text.jsonl", "text_col": "raw_text", "graph_pt": "ArtsGraph.pt"},
    "CD": {"jsonl": "cd-raw-text.jsonl", "text_col": "raw_text", "graph_pt": "CDGraph.pt"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build MAGB global KMeans synthetic graph for zero-shot graph classification.")
    parser.add_argument("--data-root", default="data/MAGB")
    parser.add_argument("--output-root", default="data/MAGB/kmeans_global_synthetic_textfixed")
    parser.add_argument("--datasets", nargs="+", default=["Grocery", "Toys", "Movies", "Reddit-S"])
    parser.add_argument("--attach-targets", nargs="*", default=[],
                        help="Unlabeled targets attached to source-built anchors; never clustered into the hierarchy.")
    parser.add_argument("--target-cluster-size", type=int, default=64)
    parser.add_argument("--min-clusters", type=int, default=64)
    parser.add_argument("--max-clusters", type=int, default=384)
    parser.add_argument("--layer2-target-size", type=int, default=8, help="Target number of layer-1 abstract nodes per dataset-level layer-2 topic.")
    parser.add_argument("--layer2-min-clusters", type=int, default=8)
    parser.add_argument("--layer2-max-clusters", type=int, default=64)
    parser.add_argument("--global-prototypes", type=int, default=48)
    parser.add_argument("--global-top-m", type=int, default=3, help="Number of weak global semantic prototype links kept for each layer-2 node.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--random-state", type=int, default=42)
    # ---- MM-SC structural clustering; disable for the KMeans ablation----
    parser.add_argument("--use-mm-sc", action="store_true", default=True,
                        help="Build layer-1 abstractions with structure- and semantics-guided MM-SC.")
    parser.add_argument("--no-mm-sc", dest="use_mm_sc", action="store_false",
                        help="Ablation: use feature-only KMeans and mean pooling.")
    parser.add_argument("--mmsc-alpha", type=float, default=0.5,
                        help="MM-SC structural weight alpha in the structure/semantics mixture.")
    parser.add_argument("--mmsc-knn-k", type=int, default=10,
                        help="MM-SC semantic kNN neighborhood size.")
    parser.add_argument("--summary-examples", type=int, default=5)
    parser.add_argument("--memory-limit-gb", type=float, default=20.0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--estimate-only", action="store_true")
    return parser.parse_args()


def clean_text(value, max_chars=180) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())[:max_chars]


def parse_list_cell(value) -> List[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        raw = ast.literal_eval(str(value))
    except Exception:
        return []
    out = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def load_dataframe(root: Path, spec: dict) -> pd.DataFrame:
    if "jsonl" in spec:
        import json as _json
        records = []
        with open(root / spec["jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                rec = _json.loads(line.strip())
                records.append({
                    "id": rec.get("id", rec.get("node_id", 0)),
                    "label": rec.get("label", ""),
                    spec["text_col"]: rec.get("raw_text", rec.get("text", "")),
                })
        df = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
        # Load labels from labels.pt if available
        import os as _os
        labels_path = root / "labels.pt"
        if _os.path.exists(labels_path):
            import torch as _torch
            labels = _torch.load(labels_path, map_location="cpu", weights_only=False)
            df["label"] = labels.tolist()
        return df
    header = pd.read_csv(root / spec["csv"], nrows=0).columns.tolist()
    usecols = ["id", "label", spec["text_col"]] + spec["edge_cols"]
    for optional in ["title", "description"]:
        usecols.append(optional)
    usecols = [col for col in dict.fromkeys(usecols) if col in header]
    return pd.read_csv(root / spec["csv"], usecols=usecols).sort_values("id").reset_index(drop=True)


def node_snippet(row, text_col: str) -> str:
    title = clean_text(getattr(row, "title"), 100) if hasattr(row, "title") else ""
    description = clean_text(getattr(row, "description"), 180) if hasattr(row, "description") else ""
    text = clean_text(getattr(row, text_col), 180) if hasattr(row, text_col) else ""
    if text_col == "caption":
        return text or title or description
    return title or description or text


def dataset_offsets(data_root: Path, datasets: Sequence[str], feature_count_datasets=()) -> Dict[str, dict]:
    offsets = {}
    cursor = 0
    for dataset in datasets:
        if dataset in feature_count_datasets:
            count = int(torch.load(
                data_root / dataset / "features_clip_text.pt", map_location="cpu", weights_only=False
            ).shape[0])
        else:
            labels = torch.load(data_root / dataset / "labels.pt", map_location="cpu", weights_only=False)
            count = int(labels.numel())
        offsets[dataset] = {"dataset_id": len(offsets), "start": cursor, "end": cursor + count, "num_original_nodes": count}
        cursor += count
    return offsets


def choose_cluster_count(num_nodes: int, target_size: int, min_clusters: int, max_clusters: int) -> int:
    return max(min_clusters, min(max_clusters, int(math.ceil(num_nodes / max(target_size, 1)))))


def estimate_peak_gb(data_root: Path, datasets: Sequence[str], max_clusters: int) -> float:
    total_text = 0
    total_image = 0
    max_dataset = 0
    for dataset in datasets:
        root = data_root / dataset
        text = (root / "features_clip_text.pt").stat().st_size
        image = (root / "features_clip_image.pt").stat().st_size
        total_text += text
        total_image += image
        max_dataset = max(max_dataset, text + image)
    cluster_cache = max_clusters * 768 * 4 * 16
    return (max_dataset * 3.0 + total_text * 1.2 + total_image * 0.2 + cluster_cache + 1024**3) / (1024**3)


def run_kmeans(features: torch.Tensor, n_clusters: int, args) -> np.ndarray:
    x = F.normalize(features.float(), dim=-1).cpu().numpy().astype(np.float32, copy=False)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=args.batch_size,
        random_state=args.random_state,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )
    return model.fit_predict(x)


def summarize_cluster(
    dataset: str,
    cluster_id: int,
    node_ids: torch.Tensor,
    text_features: torch.Tensor,
    centroid: torch.Tensor,
    snippets: Sequence[str],
    examples: int,
) -> str:
    local_text = F.normalize(text_features[node_ids].float(), dim=-1)
    sims = local_text @ centroid.float()
    top = torch.topk(sims, k=min(examples, node_ids.numel())).indices.tolist()
    example_texts = [snippets[int(node_ids[i].item())] for i in top if snippets[int(node_ids[i].item())]]
    return (
        f"{dataset} abstract cluster {cluster_id}. "
        f"Representative descriptions: {'; '.join(example_texts)}."
    )


def softmax_image_embedding(centroid: torch.Tensor, child_text: torch.Tensor, child_image: torch.Tensor) -> torch.Tensor:
    sims = F.normalize(child_text.float(), dim=-1) @ F.normalize(centroid.float(), dim=0)
    weights = torch.softmax(sims / 0.07, dim=0)
    return F.normalize((weights[:, None] * child_image.float()).sum(dim=0), dim=0).cpu()


def build_original_edges(df: pd.DataFrame, spec: dict, global_start: int, data_root: Path = None, dataset: str = "") -> torch.Tensor:
    if "graph_pt" in spec and data_root is not None:
        g = torch.load(data_root / dataset / spec["graph_pt"], map_location="cpu", weights_only=False)
        ei = g["edge_index"] if isinstance(g, dict) else g.edge_index
        n = len(df)
        local_to_global = [global_start + i for i in range(n)]
        edges = []
        for j in range(ei.shape[1]):
            s, d = int(ei[0, j]), int(ei[1, j])
            if s < n and d < n and s != d:
                edges.append((local_to_global[s], local_to_global[d]))
        return torch.tensor(edges, dtype=torch.long) if edges else torch.empty(0, 2, dtype=torch.long)
    edges = []
    id_set = set(int(x) for x in df["id"].tolist())
    for row in df.itertuples(index=False):
        src = int(getattr(row, "id"))
        for col in spec["edge_cols"]:
            if not hasattr(row, col):
                continue
            for dst in parse_list_cell(getattr(row, col)):
                if dst in id_set and dst != src:
                    edges.append((global_start + src, global_start + dst))
    return torch.tensor(edges, dtype=torch.long) if edges else torch.empty(0, 2, dtype=torch.long)


def build_local_edges(df: pd.DataFrame, spec: dict, data_root: Path = None, dataset: str = "") -> torch.Tensor:
    """Build local edge indices (nodes 0..n-1) for MM-SC."""
    if "graph_pt" in spec and data_root is not None:
        g = torch.load(data_root / dataset / spec["graph_pt"], map_location="cpu", weights_only=False)
        ei = g["edge_index"] if isinstance(g, dict) else g.edge_index
        n = len(df)
        mask = (ei[0] < n) & (ei[1] < n) & (ei[0] != ei[1])
        return ei[:, mask].clone()
    id_to_local = {int(x): i for i, x in enumerate(df["id"].tolist())}
    edges = []
    for row in df.itertuples(index=False):
        src = id_to_local.get(int(getattr(row, "id")))
        if src is None:
            continue
        for col in spec["edge_cols"]:
            if not hasattr(row, col):
                continue
            for dst in parse_list_cell(getattr(row, col)):
                d = id_to_local.get(int(dst))
                if d is not None and d != src:
                    edges.append((src, d))
    return torch.tensor(edges, dtype=torch.long) if edges else torch.empty(0, 2, dtype=torch.long)


def main():
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_num_threads))

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    peak = estimate_peak_gb(data_root, args.datasets, args.max_clusters)
    print(json.dumps({"stage": "estimate", "estimated_peak_gb": round(float(peak), 3), "memory_limit_gb": args.memory_limit_gb}), flush=True)
    if peak > args.memory_limit_gb:
        raise RuntimeError(f"estimated peak {peak:.2f}GB exceeds limit {args.memory_limit_gb:.2f}GB")
    if args.estimate_only:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    all_datasets = list(dict.fromkeys(args.datasets + args.attach_targets))
    offsets = dataset_offsets(data_root, all_datasets, set(args.attach_targets))
    total_original = max(item["end"] for item in offsets.values())
    next_node_id = total_original

    layer1_nodes = []
    original_to_layer1_edges = []
    original_edges = []
    cluster_assignments = {}

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        root = data_root / dataset
        df = load_dataframe(root, spec)
        labels = torch.load(root / "labels.pt", map_location="cpu", weights_only=False).long()
        text_features = torch.load(root / "features_clip_text.pt", map_location="cpu", weights_only=False).float()
        image_features = torch.load(root / "features_clip_image.pt", map_location="cpu", weights_only=False).float()
        snippets = [node_snippet(row, spec["text_col"]) for row in df.itertuples(index=False)]
        n_clusters = choose_cluster_count(labels.numel(), args.target_cluster_size, args.min_clusters, args.max_clusters)
        original_edges.append(build_original_edges(df, spec, offsets[dataset]["start"], data_root, dataset))

        # ============ MM-SC clustering combines structure and semantics ============
        node_type_tag = "dataset-mmsc-abstract" if args.use_mm_sc else "dataset-kmeans-abstract"
        if args.use_mm_sc:
            print(f"[{dataset}] MM-SC nodes={labels.numel()} clusters={n_clusters} "
                  f"alpha={args.mmsc_alpha} knn={args.mmsc_knn_k}", flush=True)
            local_edges = build_local_edges(df, spec, data_root, dataset)
            assignment, mmsc_clusters = mm_structure_clusters(
                text_features, image_features, local_edges,
                n_clusters=n_clusters, alpha=args.mmsc_alpha,
                knn_k=args.mmsc_knn_k, seed=args.random_state,
            )
            cluster_assignments[dataset] = torch.tensor(assignment, dtype=torch.long)
            # Use MM-SC attention-aggregated embeddings and a real anchor for each cluster
            for cl in mmsc_clusters:
                cluster_id = cl["cluster_id"]
                node_ids = torch.tensor(cl["member_local_ids"], dtype=torch.long)
                text_embedding = cl["text_embedding"]                 # PPR attention weighting
                image_embedding = cl["image_embedding"]
                anchor_local = cl["anchor_node_local"]                # Real node with highest PPR in the cluster
                global_id = next_node_id
                next_node_id += 1
                summary = summarize_cluster(
                    dataset, cluster_id, node_ids, text_features,
                    text_embedding, snippets, args.summary_examples,
                )
                child_global = (node_ids + offsets[dataset]["start"]).to(torch.long)
                layer1_nodes.append({
                    "global_node_id": int(global_id),
                    "id": f"a1:{dataset}:{cluster_id}",
                    "node_type": node_type_tag,
                    "dataset": dataset,
                    "dataset_id": offsets[dataset]["dataset_id"],
                    "layer_id": 1,
                    "cluster_id": int(cluster_id),
                    "child_node_ids": node_ids.tolist(),
                    "child_global_node_ids": child_global.tolist(),
                    "anchor_global_node_id": int(anchor_local + offsets[dataset]["start"]),
                    "support_node_count": int(node_ids.numel()),
                    "summary_text": summary,
                    "text_embedding": text_embedding,
                    "image_embedding": image_embedding,
                })
                src = child_global
                dst = torch.full_like(src, int(global_id))
                original_to_layer1_edges.append(torch.stack([src, dst], dim=1))
                original_to_layer1_edges.append(torch.stack([dst, src], dim=1))
        else:
            # ---- Ablation baseline: feature-only KMeans and mean pooling----
            print(f"[{dataset}] kmeans nodes={labels.numel()} clusters={n_clusters}", flush=True)
            assignment = run_kmeans(text_features, n_clusters, args)
            cluster_assignments[dataset] = torch.tensor(assignment, dtype=torch.long)
            for cluster_id in range(n_clusters):
                node_ids = torch.where(cluster_assignments[dataset] == cluster_id)[0]
                if node_ids.numel() == 0:
                    continue
                child_text = F.normalize(text_features[node_ids].float(), dim=-1)
                child_image = F.normalize(image_features[node_ids].float(), dim=-1)
                text_embedding = F.normalize(child_text.mean(dim=0), dim=0).cpu()
                image_embedding = softmax_image_embedding(text_embedding, child_text, child_image)
                global_id = next_node_id
                next_node_id += 1
                summary = summarize_cluster(
                    dataset, cluster_id, node_ids, text_features,
                    text_embedding, snippets, args.summary_examples,
                )
                child_global = (node_ids + offsets[dataset]["start"]).to(torch.long)
                layer1_nodes.append({
                    "global_node_id": int(global_id),
                    "id": f"a1:{dataset}:{cluster_id}",
                    "node_type": node_type_tag,
                    "dataset": dataset,
                    "dataset_id": offsets[dataset]["dataset_id"],
                    "layer_id": 1,
                    "cluster_id": int(cluster_id),
                    "child_node_ids": node_ids.tolist(),
                    "child_global_node_ids": child_global.tolist(),
                    "support_node_count": int(node_ids.numel()),
                    "summary_text": summary,
                    "text_embedding": text_embedding,
                    "image_embedding": image_embedding,
                })
                src = child_global
                dst = torch.full_like(src, int(global_id))
                original_to_layer1_edges.append(torch.stack([src, dst], dim=1))
                original_to_layer1_edges.append(torch.stack([dst, src], dim=1))
        print(f"[{dataset}] layer1_done total_layer1={len(layer1_nodes)}", flush=True)

    layer2_nodes = []
    layer1_to_layer2_edges = []
    layer2_assignments = {}

    for dataset in args.datasets:
        dataset_layer1 = [node for node in layer1_nodes if node["dataset"] == dataset]
        if not dataset_layer1:
            continue
        layer1_embeddings = F.normalize(torch.stack([node["text_embedding"].float() for node in dataset_layer1]), dim=-1)
        n_layer2 = choose_cluster_count(
            len(dataset_layer1),
            args.layer2_target_size,
            args.layer2_min_clusters,
            args.layer2_max_clusters,
        )
        n_layer2 = min(n_layer2, len(dataset_layer1))
        print(f"[{dataset}] layer2 kmeans layer1_nodes={len(dataset_layer1)} clusters={n_layer2}", flush=True)
        assignment = run_kmeans(layer1_embeddings, n_layer2, args)
        layer2_assignments[dataset] = torch.tensor(assignment, dtype=torch.long)

        for topic_id in range(n_layer2):
            member_idx = torch.where(layer2_assignments[dataset] == topic_id)[0]
            if member_idx.numel() == 0:
                continue
            member_nodes = [dataset_layer1[int(i)] for i in member_idx.tolist()]
            member_text = F.normalize(torch.stack([m["text_embedding"].float() for m in member_nodes]), dim=-1)
            member_image = F.normalize(torch.stack([m["image_embedding"].float() for m in member_nodes]), dim=-1)
            if args.use_mm_sc:
                # Attention aggregation with a real layer-1 anchor node
                agg = attentive_aggregate_embeddings(member_text, member_image)
                text_embedding = agg["text_embedding"]
                image_embedding = agg["image_embedding"]
                anchor_layer1 = member_nodes[agg["anchor_member_pos"]]
            else:
                text_embedding = F.normalize(member_text.mean(dim=0), dim=0).cpu()
                image_embedding = softmax_image_embedding(text_embedding, member_text, member_image)
                anchor_layer1 = None
            sims = member_text @ text_embedding
            top = torch.topk(sims, k=min(args.summary_examples, member_idx.numel())).indices.tolist()
            examples = [member_nodes[i]["summary_text"][:180] for i in top]
            child_original = []
            for member in member_nodes:
                child_original.extend(member["child_global_node_ids"])
            global_id = next_node_id
            next_node_id += 1
            layer2_nodes.append(
                {
                    "global_node_id": int(global_id),
                    "id": f"a2:{dataset}:{topic_id}",
                    "node_type": "dataset-highlevel-kmeans-abstract",
                    "dataset": dataset,
                    "dataset_id": offsets[dataset]["dataset_id"],
                    "layer_id": 2,
                    "topic_id": int(topic_id),
                    "child_abstract_node_ids": [m["global_node_id"] for m in member_nodes],
                    "covered_original_global_node_ids": child_original,
                    "support_abstract_count": int(member_idx.numel()),
                    "support_original_node_count": int(sum(m["support_node_count"] for m in member_nodes)),
                    "anchor_layer1_global_node_id": (
                        int(anchor_layer1["global_node_id"]) if anchor_layer1 is not None else -1
                    ),
                    "summary_text": f"{dataset} high-level topic {topic_id}. It groups related layer-1 abstract clusters. Examples: {' | '.join(examples)}",
                    "text_embedding": text_embedding,
                    "image_embedding": image_embedding,
                }
            )
            src = torch.tensor([m["global_node_id"] for m in member_nodes], dtype=torch.long)
            dst = torch.full_like(src, int(global_id))
            layer1_to_layer2_edges.append(torch.stack([src, dst], dim=1))
            layer1_to_layer2_edges.append(torch.stack([dst, src], dim=1))

    layer2_embeddings = F.normalize(torch.stack([node["text_embedding"].float() for node in layer2_nodes]), dim=-1)
    global_k = min(args.global_prototypes, len(layer2_nodes))
    print(f"[global] kmeans layer2_nodes={len(layer2_nodes)} global_prototypes={global_k}", flush=True)
    global_assignment = run_kmeans(layer2_embeddings, global_k, args)

    global_nodes = []
    layer1_to_global_edges = []
    for proto_id in range(global_k):
        member_idx = torch.where(torch.tensor(global_assignment, dtype=torch.long) == proto_id)[0]
        if member_idx.numel() == 0:
            continue
        member_nodes = [layer2_nodes[int(i)] for i in member_idx.tolist()]
        member_text = F.normalize(torch.stack([m["text_embedding"].float() for m in member_nodes]), dim=-1)
        member_image = F.normalize(torch.stack([m["image_embedding"].float() for m in member_nodes]), dim=-1)
        datasets = sorted(set(m["dataset"] for m in member_nodes))
        if args.use_mm_sc:
            # Use inverse domain frequency weights when aggregating global prototypes,
            # so that a larger dataset does not dominate the prototypes.
            member_dsets = [m["dataset"] for m in member_nodes]
            from collections import Counter as _Counter
            dcount = _Counter(member_dsets)
            # Members of larger domains receive lower weights, balancing domain contributions
            balance_w = torch.tensor(
                [1.0 / dcount[d] for d in member_dsets], dtype=torch.float32
            )
            agg = attentive_aggregate_embeddings(member_text, member_image, weights=balance_w)
            text_embedding = agg["text_embedding"]
            image_embedding = agg["image_embedding"]
        else:
            text_embedding = F.normalize(member_text.mean(dim=0), dim=0).cpu()
            image_embedding = softmax_image_embedding(text_embedding, member_text, member_image)
        sims = member_text @ text_embedding
        top = torch.topk(sims, k=min(args.summary_examples, member_idx.numel())).indices.tolist()
        examples = [member_nodes[i]["summary_text"][:160] for i in top]
        global_id = next_node_id
        next_node_id += 1
        global_nodes.append(
            {
                "global_node_id": int(global_id),
                "id": f"gproto:{proto_id}",
                "node_type": "global-kmeans-semantic-prototype",
                "dataset": "global",
                "dataset_id": -1,
                "layer_id": 3,
                "prototype_id": int(proto_id),
                "member_layer2_abstract_node_ids": [m["global_node_id"] for m in member_nodes],
                "member_datasets": datasets,
                "member_domain_distribution": dict(__import__("collections").Counter(m["dataset"] for m in member_nodes)),
                "support_abstract_count": int(member_idx.numel()),
                "support_original_node_count": int(sum(m["support_original_node_count"] for m in member_nodes)),
                "summary_text": f"Global semantic prototype {proto_id} over datasets {', '.join(datasets)}. Examples: {' | '.join(examples)}",
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
            }
        )
        src = torch.tensor([m["global_node_id"] for m in member_nodes], dtype=torch.long)
        dst = torch.full_like(src, int(global_id))
        conf = (member_text @ text_embedding).cpu()
        layer1_to_global_edges.append({"edge_index": torch.stack([src, dst], dim=1), "confidence": conf})
        layer1_to_global_edges.append({"edge_index": torch.stack([dst, src], dim=1), "confidence": conf})

    global_topm_edges = []
    if global_nodes and args.global_top_m > 0:
        global_embeddings = F.normalize(torch.stack([node["text_embedding"].float() for node in global_nodes]), dim=-1)
        sims = layer2_embeddings @ global_embeddings.T
        top_k = min(args.global_top_m, global_embeddings.shape[0])
        top_conf, top_idx = torch.topk(sims, k=top_k, dim=1)
        src_nodes = []
        dst_nodes = []
        confs = []
        ranks = []
        for layer2_pos, layer2_node in enumerate(layer2_nodes):
            for rank in range(top_k):
                proto = global_nodes[int(top_idx[layer2_pos, rank].item())]
                src_nodes.append(int(layer2_node["global_node_id"]))
                dst_nodes.append(int(proto["global_node_id"]))
                confs.append(float(top_conf[layer2_pos, rank].item()))
                ranks.append(int(rank + 1))
        src = torch.tensor(src_nodes, dtype=torch.long)
        dst = torch.tensor(dst_nodes, dtype=torch.long)
        confidence = torch.tensor(confs, dtype=torch.float32)
        rank_tensor = torch.tensor(ranks, dtype=torch.long)
        global_topm_edges.append(
            {
                "edge_index": torch.stack([src, dst], dim=1),
                "confidence": confidence,
                "rank": rank_tensor,
                "edge_type": "layer2-to-global-semantic-topm",
            }
        )
        global_topm_edges.append(
            {
                "edge_index": torch.stack([dst, src], dim=1),
                "confidence": confidence,
                "rank": rank_tensor,
                "edge_type": "global-to-layer2-semantic-topm",
            }
        )

    target_anchor_attachments = {}
    if args.attach_targets:
        layer1_to_layer2 = {}
        for node in layer2_nodes:
            for layer1_id in node["child_abstract_node_ids"]:
                layer1_to_layer2[int(layer1_id)] = int(node["global_node_id"])
        layer2_to_global = {}
        for node in global_nodes:
            for layer2_id in node["member_layer2_abstract_node_ids"]:
                layer2_to_global[int(layer2_id)] = int(node["global_node_id"])

        source_text = F.normalize(torch.stack([node["text_embedding"].float() for node in layer1_nodes]), dim=-1)
        source_image = F.normalize(torch.stack([node["image_embedding"].float() for node in layer1_nodes]), dim=-1)
        source_ids = torch.tensor([int(node["global_node_id"]) for node in layer1_nodes], dtype=torch.long)
        for target in args.attach_targets:
            target_text = F.normalize(torch.load(
                data_root / target / "features_clip_text.pt", map_location="cpu", weights_only=False
            ).float(), dim=-1)
            target_image = F.normalize(torch.load(
                data_root / target / "features_clip_image.pt", map_location="cpu", weights_only=False
            ).float(), dim=-1)
            scores = (target_text @ source_text.T + target_image @ source_image.T) * 0.5
            best_score, best_pos = scores.max(dim=1)
            attached_layer1 = source_ids[best_pos]
            attached_layer2 = torch.tensor(
                [layer1_to_layer2[int(node_id)] for node_id in attached_layer1.tolist()], dtype=torch.long
            )
            attached_global = torch.tensor(
                [layer2_to_global[int(node_id)] for node_id in attached_layer2.tolist()], dtype=torch.long
            )
            target_anchor_attachments[target] = {
                "layer1_global_node_ids": attached_layer1,
                "layer2_global_node_ids": attached_layer2,
                "global_node_ids": attached_global,
                "similarity": best_score.cpu(),
            }
    torch.save(offsets, output_root / "dataset_offsets.pt")
    with (output_root / "dataset_offsets.json").open("w", encoding="utf-8") as handle:
        json.dump(offsets, handle, ensure_ascii=False, indent=2)
    torch.save(layer1_nodes, output_root / "layer1_abstract_nodes.pt")
    torch.save(layer2_nodes, output_root / "layer2_abstract_nodes.pt")
    torch.save(global_nodes, output_root / "global_semantic_prototype_nodes.pt")
    torch.save(cluster_assignments, output_root / "cluster_assignments.pt")
    torch.save(layer2_assignments, output_root / "layer2_cluster_assignments.pt")
    torch.save(torch.cat(original_to_layer1_edges, dim=0), output_root / "original_abstract_edges.pt")
    torch.save(torch.cat(original_edges, dim=0), output_root / "original_graph_edges.pt")
    torch.save(torch.cat(layer1_to_layer2_edges, dim=0), output_root / "layer1_layer2_edges.pt")
    torch.save(layer1_to_global_edges, output_root / "abstract_global_edges.pt")
    torch.save(global_topm_edges, output_root / "abstract_global_topm_edges.pt")
    if target_anchor_attachments:
        torch.save(target_anchor_attachments, output_root / "target_anchor_attachments.pt")

    meta = {
        "stage": "magb_kmeans_global_synthetic_graph",
        "datasets": args.datasets,
        "attached_targets": args.attach_targets,
        "design": "source-only hierarchical KMeans with frozen-feature target attachment" if args.attach_targets else "full-graph hierarchical KMeans: original nodes -> dataset layer-1 abstracts -> dataset layer-2 high-level abstracts -> cross-dataset global semantic prototypes",
        "num_original_nodes": int(total_original),
        "num_layer1_abstract_nodes": len(layer1_nodes),
        "num_layer2_abstract_nodes": len(layer2_nodes),
        "num_global_semantic_prototype_nodes": len(global_nodes),
        "num_total_nodes_with_synthetic": int(next_node_id),
        "target_cluster_size": args.target_cluster_size,
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "layer2_target_size": args.layer2_target_size,
        "layer2_min_clusters": args.layer2_min_clusters,
        "layer2_max_clusters": args.layer2_max_clusters,
        "global_prototypes": args.global_prototypes,
        "global_top_m": args.global_top_m,
        "schema": {
            "layer1_abstract_nodes.pt": "dataset-level KMeans abstract nodes covering all original nodes exactly once per dataset",
            "layer2_abstract_nodes.pt": "dataset-level high-level abstract nodes clustered from layer1 abstract embeddings",
            "global_semantic_prototype_nodes.pt": "cross-dataset semantic prototypes clustered from layer2 abstract embeddings",
            "original_abstract_edges.pt": "bidirectional original<->layer1 abstract edges",
            "layer1_layer2_edges.pt": "bidirectional layer1 abstract<->layer2 high-level abstract edges",
            "abstract_global_edges.pt": "bidirectional layer2 abstract<->global prototype edges with confidence",
            "abstract_global_topm_edges.pt": "bidirectional weak top-m semantic links between every layer2 abstract and global prototypes",
            "cluster_assignments.pt": "per-dataset original local node id -> layer1 cluster id",
            "layer2_cluster_assignments.pt": "per-dataset layer1 local index -> layer2 cluster id",
        },
    }
    with (output_root / "kmeans_global_synthetic_graph_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"stage": "done", "output_root": str(output_root), "layer1_nodes": len(layer1_nodes), "layer2_nodes": len(layer2_nodes), "global_nodes": len(global_nodes)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
