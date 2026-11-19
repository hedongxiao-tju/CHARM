import os
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def build_undirected_edge_index(edge_pairs: Sequence[Sequence[int]], num_nodes: int) -> Tuple[List[List[int]], torch.Tensor]:
    adjacency = [[] for _ in range(num_nodes)]
    undirected_edges = set()
    for src, dst in edge_pairs:
        src = int(src)
        dst = int(dst)
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            continue
        adjacency[src].append(dst)
        adjacency[dst].append(src)
        if src == dst:
            undirected_edges.add((src, dst))
        else:
            undirected_edges.add((min(src, dst), max(src, dst)))
    edge_index = torch.tensor(sorted(undirected_edges), dtype=torch.long) if undirected_edges else torch.empty(0, 2, dtype=torch.long)
    return adjacency, edge_index


def compute_node_degrees(adjacency: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor([len(neighbors) for neighbors in adjacency], dtype=torch.float32)


def approximate_ppr_topk(
    adjacency: Sequence[Sequence[int]],
    center_node: int,
    top_k: int = 128,
    alpha: float = 0.15,
    epsilon: float = 1e-4,
    max_push_steps: int = 100000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = len(adjacency)
    if center_node < 0 or center_node >= num_nodes:
        raise IndexError(f"center_node {center_node} out of range for graph with {num_nodes} nodes")

    rank = torch.zeros(num_nodes, dtype=torch.float32)
    residue = torch.zeros(num_nodes, dtype=torch.float32)
    residue[center_node] = 1.0
    active = deque([center_node])
    in_queue = {center_node}
    push_steps = 0

    while active and push_steps < max_push_steps:
        node_id = active.popleft()
        in_queue.discard(node_id)
        neighbors = adjacency[node_id]
        degree = max(len(neighbors), 1)
        threshold = epsilon * degree
        if residue[node_id].item() < threshold:
            continue

        residual_value = residue[node_id].item()
        rank[node_id] += alpha * residual_value
        push_mass = (1.0 - alpha) * residual_value
        residue[node_id] = 0.0

        if neighbors:
            share = push_mass / (2.0 * degree)
            residue[node_id] += share * degree
            for neighbor in neighbors:
                residue[neighbor] += share
                neighbor_degree = max(len(adjacency[neighbor]), 1)
                if residue[neighbor].item() >= epsilon * neighbor_degree and neighbor not in in_queue:
                    active.append(neighbor)
                    in_queue.add(neighbor)
        else:
            residue[center_node] += push_mass
            if center_node not in in_queue:
                active.append(center_node)
                in_queue.add(center_node)
        push_steps += 1

    candidate_scores, candidate_nodes = torch.topk(rank, k=min(top_k, num_nodes), largest=True, sorted=True)
    candidate_nodes = candidate_nodes.tolist()
    candidate_scores = candidate_scores.tolist()

    ordered_nodes = [center_node]
    ordered_scores = [float(rank[center_node].item())]
    for node_id, score in zip(candidate_nodes, candidate_scores):
        if node_id == center_node:
            continue
        ordered_nodes.append(int(node_id))
        ordered_scores.append(float(score))
        if len(ordered_nodes) >= top_k:
            break

    if len(ordered_nodes) < top_k:
        remaining = [idx for idx in torch.argsort(rank, descending=True).tolist() if idx not in set(ordered_nodes)]
        for node_id in remaining:
            ordered_nodes.append(int(node_id))
            ordered_scores.append(float(rank[node_id].item()))
            if len(ordered_nodes) >= top_k:
                break

    return torch.tensor(ordered_nodes, dtype=torch.long), torch.tensor(ordered_scores, dtype=torch.float32)


def compute_hop_distances(adjacency: Sequence[Sequence[int]], center_node: int, selected_nodes: Iterable[int]) -> Dict[int, int]:
    selected_set = set(int(x) for x in selected_nodes)
    distances = {center_node: 0}
    queue = deque([center_node])
    remaining = set(selected_set)
    remaining.discard(center_node)
    while queue:
        node = queue.popleft()
        if not remaining:
            break
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                if neighbor in remaining:
                    remaining.discard(neighbor)
                queue.append(neighbor)
    return {node: distances.get(node, max(distances.values(), default=0) + 1) for node in selected_set}


def build_local_subgraph(
    center_node: int,
    selected_nodes: torch.Tensor,
    ppr_scores: torch.Tensor,
    adjacency: Sequence[Sequence[int]],
    degrees: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    selected_nodes = selected_nodes.to(torch.long)
    ppr_scores = ppr_scores.to(torch.float32)
    node_list = selected_nodes.tolist()
    local_index = {node_id: idx for idx, node_id in enumerate(node_list)}

    hop_map = compute_hop_distances(adjacency, center_node, node_list)
    hop_dist = torch.tensor([hop_map[node_id] for node_id in node_list], dtype=torch.long)
    if degrees is None:
        degree_vals = torch.tensor([len(adjacency[node_id]) for node_id in node_list], dtype=torch.float32)
    else:
        degree_vals = degrees[selected_nodes].to(torch.float32)

    local_edges = []
    for src_global in node_list:
        for dst_global in adjacency[src_global]:
            if dst_global not in local_index:
                continue
            src_local = local_index[src_global]
            dst_local = local_index[dst_global]
            if src_local < dst_local:
                local_edges.append([src_local, dst_local])
    local_edges = torch.tensor(local_edges, dtype=torch.long) if local_edges else torch.empty(0, 2, dtype=torch.long)

    return {
        "node_ids": selected_nodes,
        "hop_dist": hop_dist,
        "degree": degree_vals,
        "ppr_score": ppr_scores,
        "local_edges": local_edges,
    }


def ensure_dir(path: str) -> None:
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
