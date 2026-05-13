"""Graph builder — kNN(K=10) directed graph + edge features.

Edge features (3 dims, simplified from baseline 7 in 04_edge_features.md):
- d_ij (Euclidean distance, log-normalized)
- t0_ij (free-flow time, log-normalized)
- gaussian_decay (exp(-d^2 / sigma^2))

Subway-directly-connected flag and mode-availability mask are deferred to v2.1
(needs line topology join + bus stop presence join — non-trivial).
"""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.utils import add_self_loops

V2_ROOT = Path(__file__).resolve().parent.parent


def build_knn_graph(
    coords_bng_m: np.ndarray,  # (N, 2) BNG coords in metres
    K: int = 10,
    add_self_loop: bool = True,
):
    """Build kNN(K) directed graph + edge features.

    Returns:
      edge_index: (2, E) long tensor — directed edges (i→j), with self-loops
      edge_attr:  (E, 3) float tensor — [d_log, t0_log, gauss_decay]
    """
    N = coords_bng_m.shape[0]
    coords_t = torch.tensor(coords_bng_m, dtype=torch.float32)

    # Pairwise distances (N, N)
    diff = coords_t[:, None, :] - coords_t[None, :, :]
    d = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (N, N) in metres
    d_km = d / 1000.0

    # kNN: for each i, pick K smallest j (excluding self)
    d_for_knn = d.clone()
    d_for_knn.fill_diagonal_(float("inf"))
    _, knn_idx = torch.topk(d_for_knn, k=K, dim=1, largest=False)  # (N, K)

    src = torch.arange(N).repeat_interleave(K)  # (N*K,)
    dst = knn_idx.reshape(-1)                    # (N*K,)
    edge_index = torch.stack([src, dst], dim=0)  # (2, N*K)

    # Edge attributes for these edges
    src_np = src.numpy()
    dst_np = dst.numpy()
    d_km_edges = d_km[src_np, dst_np]  # (N*K,) — torch indexing

    # Free-flow time at avg_speed=20 km/h with circuity 1.3 → minutes
    t0 = d_km_edges * 1.3 / (20.0 / 60.0)  # minutes

    # Gaussian decay sigma = median edge distance (metres)
    sigma = float(d[src_np, dst_np].median())
    gauss = torch.exp(- (d[src_np, dst_np] ** 2) / (sigma ** 2 + 1e-8))

    # Stack: log normalize d and t0
    d_log = torch.log1p(d_km_edges).float()
    t0_log = torch.log1p(t0).float()
    gauss = gauss.float()

    # Z-score for d_log and t0_log (using their own mean/std on training edges)
    d_log = (d_log - d_log.mean()) / (d_log.std() + 1e-8)
    t0_log = (t0_log - t0_log.mean()) / (t0_log.std() + 1e-8)

    edge_attr = torch.stack([d_log, t0_log, gauss], dim=1)  # (N*K, 3)

    if add_self_loop:
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=0.0, num_nodes=N
        )

    return edge_index.long(), edge_attr.float()
