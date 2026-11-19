#!/usr/bin/env python3
"""Sparse multimodal structure clustering for hierarchical graph construction.

MM-SC combines structural and semantic adjacency, computes sparse PPR
embeddings for clustering, and uses attention aggregation with real anchors.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Sparse graph construction
# --------------------------------------------------------------------------- #
def _coo(indices_row, indices_col, values, n, device):
    idx = torch.stack([indices_row, indices_col], dim=0)
    # Disable sparse invariant checks for the valid indices constructed here.
    try:
        torch.sparse.check_sparse_tensor_invariants.disable()
    except Exception:
        pass
    return torch.sparse_coo_tensor(idx, values, (n, n), device=device).coalesce()


def build_structure_sparse(edge_index: torch.Tensor, n: int, device="cpu") -> torch.Tensor:
    """Build symmetric sparse adjacency (N,N) with self-loops from local edges (E,2)."""
    device = torch.device(device)
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.to(device)
        s = edge_index[:, 0].long()
        d = edge_index[:, 1].long()
        m = (s >= 0) & (s < n) & (d >= 0) & (d < n) & (s != d)
        s, d = s[m], d[m]
        rows = torch.cat([s, d])
        cols = torch.cat([d, s])
    else:
        rows = torch.empty(0, dtype=torch.long, device=device)
        cols = torch.empty(0, dtype=torch.long, device=device)
    # Self-loops
    self_loop = torch.arange(n, device=device)
    rows = torch.cat([rows, self_loop])
    cols = torch.cat([cols, self_loop])
    vals = torch.ones(rows.numel(), dtype=torch.float32, device=device)
    return _coo(rows, cols, vals, n, device)


def build_semantic_knn_sparse(feats: torch.Tensor, k: int, block: int = 2048) -> torch.Tensor:
    """Compute sparse kNN adjacency from features in blocks using O(N*k) memory."""
    n = feats.shape[0]
    device = feats.device
    x = torch.nan_to_num(feats.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = F.normalize(x, dim=-1, eps=1e-8)
    k = min(k, n - 1) if n > 1 else 0
    if k <= 0:
        return _coo(torch.arange(n, device=device), torch.arange(n, device=device),
                    torch.ones(n, device=device), n, device)
    rows, cols = [], []
    for start in range(0, n, block):
        end = min(start + block, n)
        sim = x[start:end] @ x.t()                      # (b, N) current block only
        # Exclude self-neighbors
        for i in range(end - start):
            sim[i, start + i] = -2.0
        _, idx = torch.topk(sim, k=k, dim=1)            # (b, k)
        r = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        c = idx.reshape(-1)
        rows.append(r)
        cols.append(c)
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    # Symmetrize by including both i->j and j->i
    all_r = torch.cat([rows, cols])
    all_c = torch.cat([cols, rows])
    vals = torch.ones(all_r.numel(), device=device)
    adj = _coo(all_r, all_c, vals, n, device)
    # Clamp values to 1 to remove duplicate counts
    adj = torch.sparse_coo_tensor(adj.indices(), torch.ones_like(adj.values()),
                                  (n, n), device=device).coalesce()
    return adj


def multimodal_enhanced_sparse(
    text_feat, image_feat, edge_index, alpha=0.5, knn_k=10
) -> torch.Tensor:
    """Combine structural and semantic adjacency in sparse COO format."""
    n = text_feat.shape[0]
    device = text_feat.device
    a_struct = build_structure_sparse(edge_index, n, device=device)
    a_text = build_semantic_knn_sparse(text_feat, knn_k)
    a_image = build_semantic_knn_sparse(image_feat, knn_k)
    # Semantics = text union image, with values set to 1
    a_sem = (a_text + a_image).coalesce()
    a_sem = torch.sparse_coo_tensor(a_sem.indices(), torch.ones_like(a_sem.values()),
                                    (n, n), device=device).coalesce()
    a_hat = (alpha * a_struct + (1.0 - alpha) * a_sem).coalesce()
    return a_hat


def _sparse_row_normalize(adj: torch.Tensor) -> torch.Tensor:
    n = adj.shape[0]
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-8)
    idx = adj.indices()
    vals = adj.values() / deg[idx[0]]
    return torch.sparse_coo_tensor(idx, vals, adj.shape, device=adj.device).coalesce()


# --------------------------------------------------------------------------- #
#  Sparse PPR spectral embeddings
# --------------------------------------------------------------------------- #
def sparse_ppr_embedding(
    a_hat: torch.Tensor, dim: int = 64, restart: float = 0.15, iters: int = 20, seed: int = 42
) -> torch.Tensor:
    """Compute PPR diffusion fingerprints (N, dim) using sparse power iteration.

    Project nodes into a lower-dimensional space and iterate
    X = restart*H + (1-restart)*P*X, in O(N*dim*iters).
    """
    n = a_hat.shape[0]
    device = a_hat.device
    p = _sparse_row_normalize(a_hat)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Random projection as the initial signal
    h = torch.randn(n, dim, generator=g).to(device)
    h = F.normalize(h, dim=0)
    x = h.clone()
    beta = 1.0 - restart
    for _ in range(iters):
        x = restart * h + beta * torch.sparse.mm(p, x)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def sparse_ppr_clusters(
    a_hat: torch.Tensor, n_clusters: int, embed_dim: int = 64,
    restart: float = 0.15, iters: int = 20, seed: int = 42,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Cluster PPR spectral embeddings with KMeans; return assignments and embeddings."""
    import warnings
    from sklearn.cluster import MiniBatchKMeans
    emb = sparse_ppr_embedding(a_hat, dim=embed_dim, restart=restart, iters=iters, seed=seed)
    fp = F.normalize(emb, dim=-1, eps=1e-8).cpu().numpy().astype(np.float32)
    k = min(n_clusters, fp.shape[0])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                         reassignment_ratio=0.01, batch_size=4096)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        assignment = km.fit_predict(fp)
    return assignment, emb


# --------------------------------------------------------------------------- #
#  Attention aggregation and real anchors
# --------------------------------------------------------------------------- #
def attentive_abstract(
    member_idx: torch.Tensor, member_weight: torch.Tensor,
    text_feat: torch.Tensor, image_feat: torch.Tensor,
) -> Dict[str, object]:
    """Aggregate features using PPR diffusion weights and select a real anchor node."""
    w = torch.softmax(member_weight / 0.1, dim=0)
    t = F.normalize(text_feat[member_idx].float(), dim=-1)
    im = F.normalize(image_feat[member_idx].float(), dim=-1)
    text_emb = F.normalize((w[:, None] * t).sum(0), dim=0)
    image_emb = F.normalize((w[:, None] * im).sum(0), dim=0)
    anchor_local = int(member_idx[int(torch.argmax(member_weight))].item())
    return {
        "text_embedding": text_emb.cpu(),
        "image_embedding": image_emb.cpu(),
        "anchor_node_local": anchor_local,
        "attn_weights": w.detach().cpu(),
    }


def mm_structure_clusters(
    text_feat: torch.Tensor, image_feat: torch.Tensor, edge_index: torch.Tensor,
    n_clusters: int, alpha: float = 0.5, knn_k: int = 10, restart: float = 0.15,
    embed_dim: int = 64, seed: int = 42, device: str = None,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Run sparse MM-SC and return node assignments and cluster abstractions.

    Node diffusion mass is the L2 norm of its PPR embedding.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    text_feat = text_feat.to(device)
    image_feat = image_feat.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)

    a_hat = multimodal_enhanced_sparse(text_feat, image_feat, edge_index, alpha=alpha, knn_k=knn_k)
    assignment, ppr_emb = sparse_ppr_clusters(
        a_hat, n_clusters, embed_dim=embed_dim, restart=restart, seed=seed
    )
    assignment_t = torch.tensor(assignment, dtype=torch.long)
    # Node diffusion mass as a sparse approximation of cluster centrality
    node_mass = ppr_emb.norm(dim=1).cpu()

    clusters: List[Dict[str, object]] = []
    for c in range(int(assignment.max()) + 1):
        member_idx = torch.where(assignment_t == c)[0]
        if member_idx.numel() == 0:
            continue
        member_weight = node_mass[member_idx]
        agg = attentive_abstract(
            member_idx, member_weight.to(device), text_feat, image_feat
        )
        agg["cluster_id"] = int(c)
        agg["member_local_ids"] = member_idx.tolist()
        agg["support_node_count"] = int(member_idx.numel())
        clusters.append(agg)
    return assignment, clusters


# --------------------------------------------------------------------------- #
#  Attention aggregation for existing layer-2 and global abstract nodes
# --------------------------------------------------------------------------- #
def attentive_aggregate_embeddings(
    text_emb: torch.Tensor, image_emb: torch.Tensor, weights: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """Apply attention aggregation to a group of abstract nodes.
    If weights are None, use similarity to the group centroid."""
    t = F.normalize(text_emb.float(), dim=-1)
    im = F.normalize(image_emb.float(), dim=-1)
    if weights is None:
        centroid = F.normalize(t.mean(0), dim=0)
        weights = (t @ centroid).clamp_min(0.0)        # Use similarity as weights
    w = torch.softmax(weights / 0.1, dim=0)
    out_text = F.normalize((w[:, None] * t).sum(0), dim=0)
    out_image = F.normalize((w[:, None] * im).sum(0), dim=0)
    anchor = int(torch.argmax(weights))
    return {"text_embedding": out_text.cpu(), "image_embedding": out_image.cpu(),
            "anchor_member_pos": anchor, "attn_weights": w.detach().cpu()}
