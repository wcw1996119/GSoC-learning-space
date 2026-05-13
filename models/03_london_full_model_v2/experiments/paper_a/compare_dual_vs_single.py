"""Compare dual-branch encoder vs single-branch StructuralGNN on the same data.

Both trainers use the SAME:
- 1725 grids, 24 hours
- GEODS 2019 hourly OD (real per-hour OD, not pi_t × daily synthetic)
- Free-flow t_ij (minutes)
- Train/val masks from demo_cache.npz
- 50 epochs (more for full convergence than smoke)

The single-branch baseline gets the legacy hourly_node_features.npz X_t (24, N, 25)
where the 22 static features are broadcast across all 24 hours plus 1 hourly
congestion + sin/cos. This is the v2.0-v2.3 paper-A baseline encoder pattern.

The dual-branch model gets X_static (N, 22) + X_dynamic (24, N, 5) instead —
same information content, structurally separated.

Output: print final train_nll, val_nll, val_CPC, beta, gamma side-by-side.
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
from models_lib.inverse_rum.inverse_trainer import InverseRUMTrainer
from models_lib.inverse_rum.structural_gnn import StructuralGNN


def load_data():
    cache_path = ROOT / "data" / "processed" / "demo_cache.npz"
    od_path = ROOT / "data" / "processed" / "grid_hourly_od_2019.npz"
    dyn_path = ROOT / "data" / "processed" / "grid_dynamic_features.npz"
    hourly_path = ROOT / "data" / "processed" / "hourly_node_features.npz"
    ff_path = ROOT / "data" / "processed" / "car_freeflow_t_ij.npy"

    cache = np.load(cache_path, allow_pickle=True)
    od = np.load(od_path)
    dyn = np.load(dyn_path)
    hourly = np.load(hourly_path)
    ff_t_ij = np.load(ff_path).astype(np.float32)

    X_static = torch.from_numpy(cache["static_features"].astype(np.float32))
    X_dynamic = torch.from_numpy(dyn["X_dyn"])
    X_combined_legacy = torch.from_numpy(hourly["X_t"])         # (24, N, 25)
    F_ij_t = torch.from_numpy(od["F_ij_t"])
    train_mask = torch.from_numpy(cache["train_mask"])
    val_mask = torch.from_numpy(cache["val_mask"])
    coords = torch.from_numpy(cache["coords_bng"].astype(np.float32))

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_m = torch.linalg.norm(diff, dim=-1)
    dist_km = (dist_m / 1000.0).clamp(min=0.1)
    log_d = torch.log(dist_km)

    N = X_static.shape[0]
    T = X_dynamic.shape[0]
    t_ij_T = torch.from_numpy(ff_t_ij).unsqueeze(0).expand(T, N, N).contiguous()

    edges = []
    for i in range(N):
        di_km = (dist_m[i] / 1000.0)
        nbrs = torch.where((di_km > 0.0) & (di_km < 1.5))[0]
        for j in nbrs.tolist():
            edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return dict(
        X_static=X_static, X_dynamic=X_dynamic, X_combined_legacy=X_combined_legacy,
        F_ij_t=F_ij_t, t_ij_T=t_ij_T, log_d=log_d, edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, N=N, T=T,
    )


def run_dual_branch(d, epochs=50, seed=0):
    print(f"\n--- Dual-branch (seed={seed}) ---")
    t0 = time.time()
    trainer = DualBranchInverseTrainer(
        X_static=d["X_static"],
        X_dynamic=d["X_dynamic"],
        edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"],
        t_ij_t=d["t_ij_T"],
        log_d_ij=d["log_d"],
        train_mask=d["train_mask"],
        val_mask=d["val_mask"],
        hidden_dim=32, gru_hidden=32, n_sage_layers=2,
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    enc, alpha, beta, beta_c, gamma, log = trainer.fit()
    dt = time.time() - t0
    n_p = sum(p.numel() for p in enc.parameters())
    return dict(
        train_nll_final=log.train_nll[-1], val_nll_final=log.val_nll[-1],
        cpc_val_final=log.cpc_val[-1], cpc_val_max=max(log.cpc_val),
        beta_mean=beta.mean().item(), gamma=gamma, time_s=dt, n_params=n_p,
        n_epochs=len(log.train_nll),
    )


def run_single_branch(d, epochs=50, seed=0):
    print(f"\n--- Single-branch baseline (seed={seed}) ---")
    t0 = time.time()
    X_legacy = d["X_combined_legacy"]
    F_dim = X_legacy.shape[-1]
    gnn = StructuralGNN(node_dim=F_dim, hidden_dim=32, n_layers=2, gru_hidden=32)
    trainer = InverseRUMTrainer(
        grid_features=X_legacy,
        edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"],
        t_ij_t=d["t_ij_T"],
        log_d_ij=d["log_d"],
        utility_net=gnn,
        train_mask=d["train_mask"],
        val_mask=d["val_mask"],
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    gnn_h, alpha, beta, beta_c, gamma, log = trainer.fit()
    dt = time.time() - t0
    n_p = sum(p.numel() for p in gnn_h.parameters())
    return dict(
        train_nll_final=log.train_nll[-1], val_nll_final=log.val_nll[-1],
        cpc_val_final=log.cpc_val[-1], cpc_val_max=max(log.cpc_val),
        beta_mean=beta.mean().item(), gamma=gamma, time_s=dt, n_params=n_p,
        n_epochs=len(log.train_nll),
    )


def main():
    print("Loading data...")
    d = load_data()
    print(f"  N={d['N']}, T={d['T']}, edges={d['edge_index'].shape[1]}")
    print(f"  F_ij_t total flow: {d['F_ij_t'].sum():.0f}")
    print(f"  X_static: {tuple(d['X_static'].shape)}")
    print(f"  X_dynamic: {tuple(d['X_dynamic'].shape)}")
    print(f"  X_combined_legacy: {tuple(d['X_combined_legacy'].shape)}")

    EPOCHS = 50

    dual = run_dual_branch(d, epochs=EPOCHS, seed=0)
    single = run_single_branch(d, epochs=EPOCHS, seed=0)

    print("\n" + "=" * 60)
    print("COMPARISON  (50 epochs, seed=0, same data)")
    print("=" * 60)
    fmt = "  {:<22} {:>15} {:>15}"
    print(fmt.format("metric", "DUAL-BRANCH", "SINGLE-BASELINE"))
    print("  " + "-" * 56)
    for k, label in [
        ("train_nll_final", "train NLL final"),
        ("val_nll_final",   "val NLL final"),
        ("cpc_val_final",   "CPC val final"),
        ("cpc_val_max",     "CPC val MAX"),
        ("beta_mean",       "beta mean"),
        ("gamma",           "gamma"),
        ("n_params",        "# params"),
        ("n_epochs",        "# epochs"),
        ("time_s",          "fit time (s)"),
    ]:
        v_d, v_s = dual[k], single[k]
        if isinstance(v_d, float):
            print(fmt.format(label, f"{v_d:.4f}", f"{v_s:.4f}"))
        else:
            print(fmt.format(label, f"{v_d}", f"{v_s}"))

    # Verdict
    cpc_diff = dual["cpc_val_max"] - single["cpc_val_max"]
    print("  " + "-" * 56)
    if cpc_diff > 0.005:
        print(f"  [WIN] Dual-branch better by {cpc_diff:+.4f} CPC")
    elif cpc_diff < -0.005:
        print(f"  [LOSS] Single-branch better by {-cpc_diff:+.4f} CPC -- investigate")
    else:
        print(f"  [TIE] Within +/-0.005 CPC ({cpc_diff:+.4f})")

    import json
    out = ROOT / "evaluation_outputs" / "paper_a" / "dual_branch_smoke_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dual": dual, "single": single, "cpc_diff": cpc_diff,
                   "epochs": EPOCHS, "n_grids": d["N"], "n_hours": d["T"],
                   "edges": int(d["edge_index"].shape[1]),
                   "total_flow": float(d["F_ij_t"].sum())}, f, indent=2)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
