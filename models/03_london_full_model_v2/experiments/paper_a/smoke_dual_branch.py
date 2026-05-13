"""Smoke test for DualBranchInverseTrainer.

Two-stage:
1. Synthetic-data smoke (50 nodes, 24 hours, random) — verify shapes flow,
   loss decreases, no NaN, β reaches negative sign within 20 epochs.
2. Real-data smoke (1725 grids, 5 epochs only, free-flow t_ij) — verify the
   trainer runs end-to-end on the full London dataset without OOM/NaN.

Run:
    python experiments/paper_a/smoke_dual_branch.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum.dual_branch_trainer import DualBranchInverseTrainer
from models_lib.inverse_rum.dual_branch_encoder import DualBranchEncoder


def synthetic_smoke():
    print("\n=== Stage 1: Synthetic-data smoke (50 nodes, 24h, 20 epochs) ===")
    torch.manual_seed(0)
    np.random.seed(0)

    N, T, F_s, F_d = 50, 24, 22, 5
    X_s = torch.randn(N, F_s)
    X_d = torch.randn(T, N, F_d)

    # Random sparse edge_index — each node connects to ~3 neighbours
    src = torch.randint(0, N, (3 * N,))
    dst = torch.randint(0, N, (3 * N,))
    edge_index = torch.stack([src, dst], dim=0)

    # Synthetic t_ij — distance proxy, free-flow minutes (5..50 range)
    coords = torch.randn(N, 2)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1) * 10.0 + 5.0
    t_ij = dist.clamp(min=5.0, max=60.0)
    log_d = torch.log(dist.clamp(min=0.1))

    # Synthetic observed OD: row-stochastic flows with peak near t=8 and t=17
    pi_t = torch.tensor([0.01, 0.005, 0.005, 0.01, 0.02, 0.04, 0.07,
                         0.12, 0.15, 0.07, 0.04, 0.03, 0.04, 0.04, 0.04, 0.06,
                         0.10, 0.12, 0.07, 0.04, 0.02, 0.01, 0.005, 0.005])
    daily_flow = torch.poisson(torch.full((N, N), 5.0))
    F_obs = daily_flow.unsqueeze(0) * pi_t.view(T, 1, 1) * 100.0

    t0 = time.time()
    trainer = DualBranchInverseTrainer(
        X_static=X_s,
        X_dynamic=X_d,
        edge_index=edge_index,
        observed_OD=F_obs,
        t_ij_t=t_ij,
        log_d_ij=log_d,
        hidden_dim=16,
        gru_hidden=12,
        epochs=20,
        patience=20,
        verbose=False,
    )
    encoder, alpha, beta_t, beta_c, gamma, log = trainer.fit()
    dt = time.time() - t0

    train_nll = log.train_nll
    print(f"  fit: {dt:.1f}s, {len(train_nll)} epochs")
    print(f"  train_nll: ep0={train_nll[0]:.4f} → ep{len(train_nll)-1}={train_nll[-1]:.4f}")
    print(f"  val_nll:   ep0={log.val_nll[0]:.4f} → ep{len(train_nll)-1}={log.val_nll[-1]:.4f}")
    print(f"  cpc_val:   ep0={log.cpc_val[0]:.3f} → ep{len(train_nll)-1}={log.cpc_val[-1]:.3f}")
    print(f"  beta_mean: {beta_t.mean().item():+.4f}  (should be NEGATIVE)")
    print(f"  gamma:     {gamma:+.4f}")
    print(f"  alpha:     {alpha:.4f} (should be 1.0 — fixed)")

    # Hard checks
    assert all(np.isfinite(x) for x in train_nll), "NaN in train_nll"
    assert train_nll[-1] < train_nll[0] + 1e-3, (
        f"loss did not decrease: ep0={train_nll[0]:.4f}, last={train_nll[-1]:.4f}"
    )
    assert beta_t.mean().item() < 0, f"beta should be negative; got {beta_t.mean().item()}"
    assert torch.isfinite(beta_t).all() and np.isfinite(gamma)
    print("  [PASS]")


def real_data_smoke():
    print("\n=== Stage 2: Real-data smoke (1725 grids, 5 epochs) ===")

    cache_path = ROOT / "data" / "processed" / "demo_cache.npz"
    od_path = ROOT / "data" / "processed" / "grid_hourly_od_2019.npz"
    dyn_path = ROOT / "data" / "processed" / "grid_dynamic_features.npz"
    ff_path = ROOT / "data" / "processed" / "car_freeflow_t_ij.npy"

    cache = np.load(cache_path, allow_pickle=True)
    od = np.load(od_path)
    dyn = np.load(dyn_path)
    ff_t_ij = np.load(ff_path).astype(np.float32)

    X_static = torch.from_numpy(cache["static_features"].astype(np.float32))
    X_dynamic = torch.from_numpy(dyn["X_dyn"])
    F_ij_t = torch.from_numpy(od["F_ij_t"])
    train_mask = torch.from_numpy(cache["train_mask"])
    val_mask = torch.from_numpy(cache["val_mask"])

    N = X_static.shape[0]
    T = X_dynamic.shape[0]

    # log_d_ij from coords_bng (BNG metres)
    coords = torch.from_numpy(cache["coords_bng"].astype(np.float32))
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_m = torch.linalg.norm(diff, dim=-1)
    dist_km = (dist_m / 1000.0).clamp(min=0.1)
    log_d = torch.log(dist_km)

    # Free-flow t_ij — repeat to (T, N, N) (training convention: free-flow only)
    t_ij_T = torch.from_numpy(ff_t_ij).unsqueeze(0).expand(T, N, N).contiguous()

    # Build edge_index: 1km grid 4-neighbour adjacency from coords (within ~1.5km Euclidean)
    edges = []
    for i in range(N):
        di_km = (dist_m[i] / 1000.0)
        nbrs = torch.where((di_km > 0.0) & (di_km < 1.5))[0]
        for j in nbrs.tolist():
            edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"  edge_index: {edge_index.shape}, {edge_index.shape[1]} edges (avg degree {edge_index.shape[1]/N:.1f})")

    # ---- Sanity check input ranges (per feedback_data_sanity_check.md) -------
    print(f"\n  Sanity check inputs:")
    print(f"    X_static: shape {tuple(X_static.shape)}, range [{X_static.min():.3f}, {X_static.max():.3f}], median {X_static.median():.3f}")
    print(f"    X_dynamic: shape {tuple(X_dynamic.shape)}, range [{X_dynamic.min():.3f}, {X_dynamic.max():.3f}]")
    print(f"    F_ij_t: shape {tuple(F_ij_t.shape)}, total {F_ij_t.sum():.0f}, max {F_ij_t.max():.0f}")
    print(f"    t_ij (free-flow minutes): range [{t_ij_T.min():.1f}, {t_ij_T.max():.1f}], median {t_ij_T.median():.1f}")
    print(f"    log_d_ij: range [{log_d.min():.2f}, {log_d.max():.2f}], median {log_d.median():.2f}")
    print(f"    train_mask: {train_mask.sum().item()} origins, val_mask: {val_mask.sum().item()} origins")
    assert 5 <= t_ij_T.median().item() <= 60, "t_ij_t doesn't look like minutes!"

    t0 = time.time()
    trainer = DualBranchInverseTrainer(
        X_static=X_static,
        X_dynamic=X_dynamic,
        edge_index=edge_index,
        observed_OD=F_ij_t,
        t_ij_t=t_ij_T,
        log_d_ij=log_d,
        train_mask=train_mask,
        val_mask=val_mask,
        hidden_dim=32,
        gru_hidden=32,
        epochs=5,
        patience=5,
        verbose=True,
    )
    encoder, alpha, beta_t, beta_c, gamma, log = trainer.fit()
    dt = time.time() - t0
    print(f"\n  fit: {dt:.1f}s")
    print(f"  train_nll: {log.train_nll[0]:.4f} → {log.train_nll[-1]:.4f}")
    print(f"  cpc_val:   {log.cpc_val[0]:.3f} → {log.cpc_val[-1]:.3f}")
    print(f"  beta_mean: {beta_t.mean().item():+.4f}")
    print(f"  gamma:     {gamma:+.4f}")
    assert log.train_nll[-1] < log.train_nll[0] + 1e-3
    assert beta_t.mean().item() < 0
    print("  [PASS]")


if __name__ == "__main__":
    synthetic_smoke()
    real_data_smoke()
    print("\n=== ALL SMOKE TESTS PASSED ===")
