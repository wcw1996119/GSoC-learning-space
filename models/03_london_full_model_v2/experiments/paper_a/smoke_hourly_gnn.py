"""Smoke test for the T=24 hourly forward pass through StructuralGNN.

Verifies that:
  1. (24, N, 25) hourly tensor passes cleanly through the GNN+GRU stack.
  2. Output shape is (24, N).
  3. The GRU is actually doing temporal mixing (V_jt at different t differs
     by more than a trivial floor; otherwise the GRU collapsed to a static map).
  4. Output is zero-mean / unit-std per the StructuralGNN normalisation contract.

Run:
    python experiments/paper_a/smoke_hourly_gnn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import StructuralGNN


def build_edge_index(t_ij: torch.Tensor, k: int = 10) -> torch.Tensor:
    if t_ij.dim() == 3:
        t_ij = t_ij.mean(0)
    N = t_ij.shape[0]
    t_pen = t_ij.clone()
    t_pen.fill_diagonal_(float("inf"))
    nbrs = torch.topk(t_pen, k=k, largest=False).indices
    src = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    return torch.stack([src, dst], dim=0)


def main() -> None:
    torch.manual_seed(0)
    feats = np.load(ROOT / "data" / "processed" / "hourly_node_features.npz", allow_pickle=True)
    X_t = torch.tensor(feats["X_t"], dtype=torch.float32)        # (24, N, F)
    print(f"X_t: shape={tuple(X_t.shape)}, dtype={X_t.dtype}")
    T, N, F = X_t.shape
    assert T == 24 and F == 25

    t_ij = torch.tensor(
        np.load(ROOT / "data" / "processed" / "car_freeflow_t_ij.npy"),
        dtype=torch.float32,
    )
    print(f"t_ij: shape={tuple(t_ij.shape)}")
    edge_index = build_edge_index(t_ij, k=10)
    print(f"edge_index: shape={tuple(edge_index.shape)}")

    gnn = StructuralGNN(in_features=F, hidden=64, out=1, depth=2, gru_hidden=32)
    gnn.eval()
    with torch.no_grad():
        V_jt = gnn(X_t, edge_index=edge_index)
    print(f"V_jt: shape={tuple(V_jt.shape)}")
    assert V_jt.shape == (24, N), f"expected (24,{N}), got {tuple(V_jt.shape)}"

    # Normalisation contract: zero mean / unit std overall.
    mu = V_jt.mean().item()
    sigma = V_jt.std().item()
    print(f"V_jt mean={mu:+.4e}, std={sigma:.4f}")
    assert abs(mu) < 1e-3, f"V_jt mean not ≈ 0: {mu}"
    assert abs(sigma - 1.0) < 5e-2, f"V_jt std not ≈ 1: {sigma}"

    # GRU temporal mixing check: compare hour 8 (peak) vs hour 14 (off-peak).
    # If GRU collapsed to a static map, V_jt[8] == V_jt[14] (exactly).
    diff_8_14 = (V_jt[8] - V_jt[14]).abs().mean().item()
    diff_8_8 = (V_jt[8] - V_jt[8]).abs().mean().item()  # sanity = 0
    print(f"|V_8 - V_14|.mean()={diff_8_14:.4f}  |V_8 - V_8|={diff_8_8:.2e}")
    assert diff_8_14 > 0.05, f"GRU appears to have collapsed: hour 8 vs 14 differ by only {diff_8_14:.4f}"

    # Per-hour signal magnitude should also vary (otherwise hours all look same)
    per_hour_std = V_jt.std(dim=1)                                # (24,)
    print(f"per-hour std min/max: {per_hour_std.min():.4f} / {per_hour_std.max():.4f}")

    # Pairwise hour-distance matrix to confirm reasonable temporal structure
    hour_dist = (V_jt.unsqueeze(0) - V_jt.unsqueeze(1)).abs().mean(dim=-1)  # (24, 24)
    print(f"hour-pair mean |Δ|: median={hour_dist.median().item():.4f}, "
          f"max={hour_dist.max().item():.4f}")

    print("== ALL OK ==")


if __name__ == "__main__":
    main()
