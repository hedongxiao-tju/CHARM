#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F


DATASETS = ["Grocery", "Toys", "Movies", "Reddit-S"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build modality-complement bridge edges for MAGB hierarchical synthetic graph.")
    parser.add_argument("--data-root", default="data/MAGB")
    parser.add_argument("--input-root", default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic_cliptext")
    parser.add_argument("--output-root", default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic_cliptext_bridge")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--ppr-topk", type=int, default=128)
    parser.add_argument("--sample-pairs", type=int, default=200000)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--low-quantile", type=float, default=0.35)
    parser.add_argument("--min-structure-score", type=float, default=0.45)
    parser.add_argument("--max-bridges-per-node", type=int, default=4)
    parser.add_argument("--rank-rule", choices=["complement", "text", "image", "mean"], default="complement",
                        help="Bridge selection rule. 'complement' is CHARM; the others are matched "
                             "standard-similarity controls that keep the same PPR candidate set, "
                             "structural filter and per-node budget.")
    parser.add_argument("--memory-limit-gb", type=float, default=20.0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--estimate-only", action="store_true")
    return parser.parse_args()


def estimate_peak_gb(data_root: Path, graph_root: Path, datasets):
    size = 0
    for dataset in datasets:
        root = data_root / dataset
        for name in ["features_clip_text.pt", "features_clip_image.pt", "ppr.pt"]:
            size += (root / name).stat().st_size
    for name in [
        "cluster_assignments.pt",
        "layer2_cluster_assignments.pt",
        "layer1_abstract_nodes.pt",
        "layer2_abstract_nodes.pt",
        "global_semantic_prototype_nodes.pt",
    ]:
        size += (graph_root / name).stat().st_size
    return (size * 2.2 + 1024**3) / (1024**3)


def copy_tree(input_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    for path in input_root.iterdir():
        if path.is_file():
            shutil.copy2(path, output_root / path.name)


def build_membership(graph_root: Path):
    offsets = json.load((graph_root / "dataset_offsets.json").open("r", encoding="utf-8"))
    layer1_assign = torch.load(graph_root / "cluster_assignments.pt", map_location="cpu", weights_only=False)
    layer2_assign = torch.load(graph_root / "layer2_cluster_assignments.pt", map_location="cpu", weights_only=False)
    layer1_nodes = torch.load(graph_root / "layer1_abstract_nodes.pt", map_location="cpu", weights_only=False)
    layer2_nodes = torch.load(graph_root / "layer2_abstract_nodes.pt", map_location="cpu", weights_only=False)
    global_nodes = torch.load(graph_root / "global_semantic_prototype_nodes.pt", map_location="cpu", weights_only=False)

    layer1_by_dataset = {}
    for node in layer1_nodes:
        layer1_by_dataset.setdefault(node["dataset"], []).append(node)
    for nodes in layer1_by_dataset.values():
        nodes.sort(key=lambda x: int(x["cluster_id"]))

    layer2_by_dataset = {}
    for node in layer2_nodes:
        layer2_by_dataset.setdefault(node["dataset"], []).append(node)
    for nodes in layer2_by_dataset.values():
        nodes.sort(key=lambda x: int(x["topic_id"]))

    layer2_to_global = {}
    for proto in global_nodes:
        for layer2_id in proto.get("member_layer2_abstract_node_ids", []):
            layer2_to_global[int(layer2_id)] = int(proto["global_node_id"])

    membership = {}
    for dataset, offset in offsets.items():
        n = int(offset["num_original_nodes"])
        l1_cluster = layer1_assign[dataset].long()
        l1_nodes = layer1_by_dataset[dataset]
        cluster_to_pos = {int(node["cluster_id"]): pos for pos, node in enumerate(l1_nodes)}
        l1_global = torch.empty(n, dtype=torch.long)
        l2_global = torch.empty(n, dtype=torch.long)
        g_global = torch.empty(n, dtype=torch.long)
        l2_assign_dataset = layer2_assign[dataset].long()
        l2_nodes = layer2_by_dataset[dataset]
        for node_id in range(n):
            cluster_id = int(l1_cluster[node_id].item())
            l1_pos = cluster_to_pos[cluster_id]
            layer1_id = int(l1_nodes[l1_pos]["global_node_id"])
            layer2_topic = int(l2_assign_dataset[l1_pos].item())
            layer2_id = int(l2_nodes[layer2_topic]["global_node_id"])
            l1_global[node_id] = layer1_id
            l2_global[node_id] = layer2_id
            g_global[node_id] = layer2_to_global.get(layer2_id, -1)
        membership[dataset] = {
            "start": int(offset["start"]),
            "num_nodes": n,
            "layer1_global": l1_global,
            "layer2_global": l2_global,
            "global_proto": g_global,
        }
    return membership


def iter_candidates(ppr, topk):
    for src, nbrs in ppr.items():
        src = int(src)
        nbrs = nbrs[:topk].tolist() if torch.is_tensor(nbrs) else list(nbrs)[:topk]
        for rank, dst in enumerate(nbrs):
            dst = int(dst)
            if dst == src:
                continue
            yield src, dst, rank


def estimate_thresholds(text_feat, image_feat, ppr, args):
    text_feat = F.normalize(text_feat.float(), dim=-1)
    image_feat = F.normalize(image_feat.float(), dim=-1)
    text_sims = []
    image_sims = []
    count = 0
    for src, dst, _ in iter_candidates(ppr, args.ppr_topk):
        text_sims.append(float((text_feat[src] * text_feat[dst]).sum().item()))
        image_sims.append(float((image_feat[src] * image_feat[dst]).sum().item()))
        count += 1
        if count >= args.sample_pairs:
            break
    text_tensor = torch.tensor(text_sims, dtype=torch.float32)
    image_tensor = torch.tensor(image_sims, dtype=torch.float32)
    return {
        "text_high": float(torch.quantile(text_tensor, args.high_quantile).item()),
        "text_low": float(torch.quantile(text_tensor, args.low_quantile).item()),
        "image_high": float(torch.quantile(image_tensor, args.high_quantile).item()),
        "image_low": float(torch.quantile(image_tensor, args.low_quantile).item()),
        "sample_pairs": int(count),
    }


def abstract_consistency(membership, src, dst):
    if int(membership["layer1_global"][src]) == int(membership["layer1_global"][dst]):
        return 1.0, "shared_layer1"
    if int(membership["layer2_global"][src]) == int(membership["layer2_global"][dst]):
        return 0.7, "shared_layer2"
    if int(membership["global_proto"][src]) >= 0 and int(membership["global_proto"][src]) == int(membership["global_proto"][dst]):
        return 0.45, "shared_global_prototype"
    return 0.0, "ppr_only"


def build_dataset_edges(dataset, data_root, membership, thresholds, args):
    root = data_root / dataset
    text_feat = F.normalize(torch.load(root / "features_clip_text.pt", map_location="cpu", weights_only=False).float(), dim=-1)
    image_feat = F.normalize(torch.load(root / "features_clip_image.pt", map_location="cpu", weights_only=False).float(), dim=-1)
    ppr = torch.load(root / "ppr.pt", map_location="cpu", weights_only=False)
    start = membership["start"]
    top_edges = {}

    for src, dst, rank in iter_candidates(ppr, args.ppr_topk):
        text_sim = float((text_feat[src] * text_feat[dst]).sum().item())
        image_sim = float((image_feat[src] * image_feat[dst]).sum().item())
        mean_sim = 0.5 * (text_sim + image_sim)
        direction = None
        complement_gap = 0.0
        if args.rank_rule == "complement":
            if text_sim >= thresholds["text_high"] and image_sim <= thresholds["image_low"]:
                direction = "text-strong-image-weak"
                complement_gap = (text_sim - thresholds["text_high"]) + (thresholds["image_low"] - image_sim)
            elif image_sim >= thresholds["image_high"] and text_sim <= thresholds["text_low"]:
                direction = "image-strong-text-weak"
                complement_gap = (image_sim - thresholds["image_high"]) + (thresholds["text_low"] - text_sim)
            if direction is None:
                continue

        ppr_rank_score = 1.0 - (rank / max(args.ppr_topk - 1, 1))
        abs_score, abs_type = abstract_consistency(membership, src, dst)
        structure_score = 0.55 * ppr_rank_score + 0.45 * abs_score
        if structure_score < args.min_structure_score:
            continue
        if args.rank_rule == "complement":
            confidence = 0.55 * max(complement_gap, 0.0) + 0.45 * structure_score
        else:
            # Matched standard-similarity controls: identical candidate set,
            # structural filter and budget; only the ranking score changes.
            primary = {"text": text_sim, "image": image_sim, "mean": mean_sim}[args.rank_rule]
            direction = f"{args.rank_rule}-similarity"
            confidence = primary
        record = {
            "src": start + src,
            "dst": start + dst,
            "text_similarity": text_sim,
            "image_similarity": image_sim,
            "ppr_rank": rank + 1,
            "ppr_rank_score": ppr_rank_score,
            "abstract_consistency_score": abs_score,
            "abstract_consistency_type": abs_type,
            "structure_consistency_score": structure_score,
            "confidence": confidence,
            "direction": direction,
            "dataset": dataset,
        }
        top_edges.setdefault(src, []).append(record)

    kept = []
    for src_edges in top_edges.values():
        src_edges.sort(key=lambda item: item["confidence"], reverse=True)
        kept.extend(src_edges[: args.max_bridges_per_node])
    return kept


def records_to_save(records):
    edge_index = torch.tensor([[r["src"], r["dst"]] for r in records], dtype=torch.long)
    return {
        "edge_index": edge_index,
        "text_similarity": torch.tensor([r["text_similarity"] for r in records], dtype=torch.float32),
        "image_similarity": torch.tensor([r["image_similarity"] for r in records], dtype=torch.float32),
        "ppr_rank": torch.tensor([r["ppr_rank"] for r in records], dtype=torch.long),
        "ppr_rank_score": torch.tensor([r["ppr_rank_score"] for r in records], dtype=torch.float32),
        "abstract_consistency_score": torch.tensor([r["abstract_consistency_score"] for r in records], dtype=torch.float32),
        "structure_consistency_score": torch.tensor([r["structure_consistency_score"] for r in records], dtype=torch.float32),
        "confidence": torch.tensor([r["confidence"] for r in records], dtype=torch.float32),
        "direction": [r["direction"] for r in records],
        "abstract_consistency_type": [r["abstract_consistency_type"] for r in records],
        "dataset": [r["dataset"] for r in records],
        "edge_type": "modality-complement-bridge",
    }


def main():
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    data_root = Path(args.data_root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    peak = estimate_peak_gb(data_root, input_root, args.datasets)
    print(json.dumps({"stage": "estimate", "estimated_peak_gb": round(float(peak), 3), "memory_limit_gb": args.memory_limit_gb}), flush=True)
    if peak > args.memory_limit_gb:
        raise RuntimeError(f"estimated peak {peak:.2f}GB exceeds limit {args.memory_limit_gb:.2f}GB")
    if args.estimate_only:
        return

    copy_tree(input_root, output_root)
    membership = build_membership(input_root)
    all_records = []
    thresholds_by_dataset = {}
    counts_by_dataset = {}

    for dataset in args.datasets:
        root = data_root / dataset
        text_feat = torch.load(root / "features_clip_text.pt", map_location="cpu", weights_only=False)
        image_feat = torch.load(root / "features_clip_image.pt", map_location="cpu", weights_only=False)
        ppr = torch.load(root / "ppr.pt", map_location="cpu", weights_only=False)
        thresholds = estimate_thresholds(text_feat, image_feat, ppr, args)
        thresholds_by_dataset[dataset] = thresholds
        del text_feat, image_feat, ppr
        print(f"[{dataset}] thresholds {thresholds}", flush=True)
        records = build_dataset_edges(dataset, data_root, membership[dataset], thresholds, args)
        counts_by_dataset[dataset] = len(records)
        all_records.extend(records)
        print(f"[{dataset}] bridge_edges={len(records)}", flush=True)

    bridge = records_to_save(all_records)
    torch.save(bridge, output_root / "modality_complement_bridge_edges.pt")
    meta_path = output_root / "kmeans_global_synthetic_graph_meta.json"
    meta = json.load(meta_path.open("r", encoding="utf-8"))
    meta["modality_complement_bridge"] = {
        "enabled": True,
        "candidate_source": f"PPR top-{args.ppr_topk}",
        "high_quantile": args.high_quantile,
        "low_quantile": args.low_quantile,
        "min_structure_score": args.min_structure_score,
        "max_bridges_per_node": args.max_bridges_per_node,
        "num_edges": len(all_records),
        "counts_by_dataset": counts_by_dataset,
        "thresholds_by_dataset": thresholds_by_dataset,
        "edge_file": "modality_complement_bridge_edges.pt",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"stage": "done", "output_root": str(output_root), "bridge_edges": len(all_records), "counts_by_dataset": counts_by_dataset}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
