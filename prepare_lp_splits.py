#!/usr/bin/env python3
"""Prepare link prediction splits for MLaGA-style zero-shot link prediction.

For each domain (Grocery, Toys, Movies):
  - Train: 5000 positive edges + 5000 negative non-edges
  - Val:   800 positive edges + 800 negative non-edges (for threshold calibration)
  - Test:  1600 positive edges + 1600 negative non-edges
  - Each sample includes a reference pair (ref_a, ref_b) selected from
    the PPR / 2-hop neighborhood of the target pair.
  - 50% negatives are hard negatives sampled from 2-hop neighborhood.

Three transfer settings:
  Grocery + Movies -> Toys
  Grocery + Toys   -> Movies
  Toys   + Movies  -> Grocery
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch


def build_adjacency(all_edges, start, end, num_nodes):
    adj = defaultdict(set)
    edge_set = set()
    mask = (
        (all_edges[:, 0] >= start) & (all_edges[:, 0] < end)
        & (all_edges[:, 1] >= start) & (all_edges[:, 1] < end)
    )
    ds_edges = all_edges[mask]
    for i in range(ds_edges.shape[0]):
        s = int(ds_edges[i, 0]) - start
        d = int(ds_edges[i, 1]) - start
        if s != d and 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].add(d)
            adj[d].add(s)
            edge_set.add((min(s, d), max(s, d)))
    return adj, edge_set


def load_ppr_neighbors(magb_root, ds_name, num_nodes):
    ppr_path = os.path.join(magb_root, ds_name, "ppr.pt")
    if not os.path.exists(ppr_path):
        return {}
    ppr = torch.load(ppr_path, map_location="cpu", weights_only=False)
    neighbors = {}
    for key, value in ppr.items():
        node_id = int(key)
        if isinstance(value, torch.Tensor):
            nbrs = value.tolist()
        elif isinstance(value, list):
            nbrs = value
        else:
            continue
        nbr_ids = []
        for item in nbrs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                nbr_ids.append(int(item[0]))
            else:
                nbr_ids.append(int(item))
        neighbors[node_id] = [
            n for n in nbr_ids if 0 <= n < num_nodes and n != node_id
        ]
    return neighbors


def sample_positive_edges(edge_list, n_samples, rng, existing=None):
    if existing is None:
        existing = set()
    indices = list(range(len(edge_list)))
    rng.shuffle(indices)
    sampled = []
    for idx in indices:
        if len(sampled) >= n_samples:
            break
        edge = edge_list[idx]
        if edge not in existing:
            sampled.append(edge)
            existing.add(edge)
    return sampled, existing


def sample_negative_edges(num_nodes, n_samples, rng, edge_set, existing=None):
    if existing is None:
        existing = set()
    sampled = []
    attempts = 0
    max_attempts = n_samples * 200
    while len(sampled) < n_samples and attempts < max_attempts:
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)
        if u == v:
            attempts += 1
            continue
        pair = (min(u, v), max(u, v))
        if pair in edge_set or pair in existing:
            attempts += 1
            continue
        sampled.append(pair)
        existing.add(pair)
        attempts += 1
    return sampled, existing


def sample_hard_negatives(num_nodes, n_samples, rng, edge_set, adj, ppr_neighbors, existing=None):
    """Sample hard negatives: pairs that share a 2-hop neighbor but are not directly connected."""
    if existing is None:
        existing = set()
    sampled = []
    all_nodes = list(range(num_nodes))
    rng.shuffle(all_nodes)
    attempts = 0
    max_attempts = n_samples * 500
    while len(sampled) < n_samples and attempts < max_attempts:
        u = rng.choice(all_nodes)
        u_nbrs = list(ppr_neighbors.get(u, [])) or list(adj.get(u, set()))
        if len(u_nbrs) < 2:
            attempts += 1
            continue
        for nbr in u_nbrs:
            nbrs_of_nbr = list(adj.get(nbr, set()))
            for nn in nbrs_of_nbr:
                if nn != u and nn not in adj.get(u, set()):
                    pair = (min(u, nn), max(u, nn))
                    if pair not in edge_set and pair not in existing:
                        sampled.append(pair)
                        existing.add(pair)
                        break
            if len(sampled) >= n_samples:
                break
        attempts += 1
    return sampled, existing


def find_reference_pair(u, v, ppr_neighbors, adj, rng, edge_set):
    u_nbrs = list(ppr_neighbors.get(u, []))
    if not u_nbrs:
        u_nbrs = list(adj.get(u, set()))
    v_nbrs = list(ppr_neighbors.get(v, []))
    if not v_nbrs:
        v_nbrs = list(adj.get(v, set()))
    u_nbrs = [n for n in u_nbrs if n != v]
    v_nbrs = [n for n in v_nbrs if n != u]
    if not u_nbrs or not v_nbrs:
        all_nodes = list(range(len(adj))) if adj else [0, 1]
        a = rng.choice(all_nodes) if all_nodes else 0
        b = rng.choice(all_nodes) if all_nodes else 0
        return (a, b)
    for _ in range(20):
        a = rng.choice(u_nbrs)
        b = rng.choice(v_nbrs)
        if a != b:
            pair = (min(a, b), max(a, b))
            if pair in edge_set:
                return (a, b)
    a = rng.choice(u_nbrs)
    b = rng.choice(v_nbrs)
    if a == b:
        b = rng.choice(v_nbrs) if len(v_nbrs) > 1 else a
    return (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--syn-root",
        type=str,
        default="data/MAGB/realnode_kmeans_product3_g24_topm3_semantic_cliptext_bridge",
    )
    parser.add_argument(
        "--magb-root", type=str, default="data/MAGB"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-pos", type=int, default=5000)
    parser.add_argument("--n-train-neg", type=int, default=5000)
    parser.add_argument("--n-val-pos", type=int, default=800)
    parser.add_argument("--n-val-neg", type=int, default=800)
    parser.add_argument("--n-test-pos", type=int, default=1600)
    parser.add_argument("--n-test-neg", type=int, default=1600)
    parser.add_argument("--hard-neg-ratio", type=float, default=0.5,
                        help="Fraction of negatives that are hard (2-hop).")
    parser.add_argument("--test-hard-neg-ratio", type=float, default=0.0,
                        help="Fraction of TEST negatives that are hard (2-hop non-edge).")
    parser.add_argument("--datasets", type=str, nargs="*", default=["Grocery", "Toys", "Movies"],
                        help="Dataset names to process.")
    args = parser.parse_args()

    with open(os.path.join(args.syn_root, "dataset_offsets.json")) as f:
        offsets = json.load(f)

    all_edges = torch.load(
        os.path.join(args.syn_root, "original_graph_edges.pt"),
        map_location="cpu",
        weights_only=False,
    )

    for ds_name in args.datasets:
        ds_info = offsets[ds_name]
        start = ds_info["start"]
        end = ds_info["end"]
        num_nodes = ds_info["num_original_nodes"]

        adj, edge_set = build_adjacency(all_edges, start, end, num_nodes)
        print(f"\n{ds_name}: {num_nodes} nodes, {len(edge_set)} edges")

        ppr_neighbors = load_ppr_neighbors(args.magb_root, ds_name, num_nodes)
        print(f"  PPR neighbors loaded for {len(ppr_neighbors)} nodes")

        edge_list = list(edge_set)
        rng = random.Random(args.seed)

        # Sample positive edges: train / val / test
        train_pos, used_pos = sample_positive_edges(
            edge_list, args.n_train_pos, rng
        )
        val_pos, used_pos = sample_positive_edges(
            edge_list, args.n_val_pos, rng, existing=used_pos
        )
        test_pos, used_pos = sample_positive_edges(
            edge_list, args.n_test_pos, rng, existing=used_pos
        )
        print(
            f"  Sampled {len(train_pos)} train pos, {len(val_pos)} val pos, {len(test_pos)} test pos"
        )

        # Sample negative edges with hard negative mining
        n_hard_train = int(args.n_train_neg * args.hard_neg_ratio)
        n_easy_train = args.n_train_neg - n_hard_train
        n_hard_val = int(args.n_val_neg * args.hard_neg_ratio)
        n_easy_val = args.n_val_neg - n_hard_val

        train_neg_hard, used_neg = sample_hard_negatives(
            num_nodes, n_hard_train, rng, edge_set, adj, ppr_neighbors
        )
        train_neg_easy, used_neg = sample_negative_edges(
            num_nodes, n_easy_train, rng, edge_set, existing=used_neg
        )
        train_neg = train_neg_hard + train_neg_easy
        rng.shuffle(train_neg)
        print(
            f"  Train neg: {len(train_neg_hard)} hard + {len(train_neg_easy)} easy = {len(train_neg)}"
        )

        val_neg_hard, used_neg = sample_hard_negatives(
            num_nodes, n_hard_val, rng, edge_set, adj, ppr_neighbors, existing=used_neg
        )
        val_neg_easy, used_neg = sample_negative_edges(
            num_nodes, n_easy_val, rng, edge_set, existing=used_neg
        )
        val_neg = val_neg_hard + val_neg_easy
        rng.shuffle(val_neg)

        n_hard_test = int(args.n_test_neg * args.test_hard_neg_ratio)
        n_easy_test = args.n_test_neg - n_hard_test
        if n_hard_test > 0:
            test_neg_hard, used_neg = sample_hard_negatives(
                num_nodes, n_hard_test, rng, edge_set, adj, ppr_neighbors, existing=used_neg
            )
        else:
            test_neg_hard = []
        test_neg_easy, used_neg = sample_negative_edges(
            num_nodes, n_easy_test, rng, edge_set, existing=used_neg
        )
        test_neg = test_neg_hard + test_neg_easy
        rng.shuffle(test_neg)
        print(
            f"  Sampled {len(val_neg)} val neg, {len(test_neg)} test neg"
        )

        def build_samples(pairs, label):
            samples = []
            for u, v in pairs:
                ref_a, ref_b = find_reference_pair(
                    u, v, ppr_neighbors, adj, rng, edge_set
                )
                samples.append(
                    {"u": u, "v": v, "label": label, "ref_a": ref_a, "ref_b": ref_b}
                )
            return samples

        train_samples = build_samples(train_pos, 1) + build_samples(train_neg, 0)
        val_samples = build_samples(val_pos, 1) + build_samples(val_neg, 0)
        test_samples = build_samples(test_pos, 1) + build_samples(test_neg, 0)

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        rng.shuffle(test_samples)

        split = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
            "num_nodes": num_nodes,
            "seed": args.seed,
        }

        save_path = os.path.join(
            args.magb_root, ds_name, "lp_zeroshot_split.pt"
        )
        torch.save(split, save_path)
        print(f"  Saved to {save_path}")
        n_train_pos = sum(1 for s in train_samples if s["label"] == 1)
        n_val_pos = sum(1 for s in val_samples if s["label"] == 1)
        n_test_pos = sum(1 for s in test_samples if s["label"] == 1)
        print(
            f"  Train: {len(train_samples)} ({n_train_pos} pos)"
        )
        print(
            f"  Val:   {len(val_samples)} ({n_val_pos} pos)"
        )
        print(
            f"  Test:  {len(test_samples)} ({n_test_pos} pos)"
        )


if __name__ == "__main__":
    main()
