"""Train Phase B v6 with T=24 hourly inputs (D2/D4 enabling).

Inputs:
  - grid_features: (24, N, 25) hourly node features from
    data/processed/hourly_node_features.npz (built by D2 build_hourly_node_features.py)
  - observed_OD:   (24, N, N) hourly Census OD from demo_cache.npz "F_ij_t"
  - t_ij:          car_freeflow_t_ij (broadcast across 24 hours; inverse loop
                    deliberately uses free-flow per Phase B v6 doctrine)
  - log_d:         (N, N) static log distance

Same Variant 6 (sign-constrained) and Variant 4 (unconstrained) configs.

Saves: evaluation_outputs/paper_a/phase_b_v6_hourly_seed{seed}.pt
       (one ckpt per seed, includes config, metrics, scalars)

Note: training time scales ~24× the daily version. With 150 epochs × 5 seeds,
expect ~12-15 minutes per seed = ~1 hour total. Run on a quiet machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, cpc

OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_hourly():
    """Load T=24 hourly inputs."""
    cache = dict(np.load(ROOT / "data" / "processed" / "demo_cache.npz",
                          allow_pickle=True))
    F_ij_t = torch.tensor(cache["F_ij_t"], dtype=torch.float32)         # (24, N, N)
    print(f"  F_ij_t shape: {tuple(F_ij_t.shape)} sum: {F_ij_t.sum().item():,.0f}")

    # Free-flow t_ij broadcast to T=24 (Phase B v6 inverse uses free-flow)
    t_ij_static = torch.tensor(
        np.load(ROOT / "data" / "processed" / "car_freeflow_t_ij.npy").astype(np.float32),
        dtype=torch.float32,
    )                                                                    # (N, N)
    T = 24
    t_ij_t = t_ij_static.unsqueeze(0).expand(T, -1, -1).contiguous()    # (24, N, N)

    # Hourly node features (24, N, 25) from D2
    feats = np.load(ROOT / "data" / "processed" / "hourly_node_features.npz",
                    allow_pickle=True)
    static_t = torch.tensor(feats["X_t"], dtype=torch.float32)          # (24, N, 25)
    print(f"  static_t shape: {tuple(static_t.shape)}")

    coords = cache["coords_bng"]
    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d = torch.tensor(np.log1p(d_km), dtype=torch.float32)
    train_mask = torch.tensor(cache["train_mask"])
    val_mask = torch.tensor(cache["val_mask"])

    aux = np.load(ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    occ_match = torch.tensor(aux["occ_match"], dtype=torch.float32)
    income_score = torch.tensor(aux["income_score_per_origin"], dtype=torch.float32)
    wage_score = torch.tensor(aux["wage_score_per_dest"], dtype=torch.float32)

    return {
        "F_ij_t": F_ij_t, "t_ij_t": t_ij_t, "log_d": log_d, "static_t": static_t,
        "train_mask": train_mask, "val_mask": val_mask,
        "occ_match": occ_match, "income_score": income_score, "wage_score": wage_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce_main", action="store_true",
                        help="If set, sign-constrained Variant 6; else unconstrained Variant 4")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80,
                        help="fewer than daily because each epoch is 24x heavier")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--K", type=int, default=50)
    args = parser.parse_args()

    print(f"[hourly-ckpt] loading T=24 inputs ...")
    data = load_data_hourly()
    edge_index = build_edge_index(data["t_ij_t"], k=10)
    N = data["static_t"].shape[1]
    Fdim = data["static_t"].shape[-1]
    print(f"[hourly-ckpt] N={N} F={Fdim} T=24 edges={edge_index.shape[1]} "
          f"seed={args.seed} enforce_main={args.enforce_main}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    util_net = StructuralGNN(in_features=Fdim, hidden=args.hidden, out=1, depth=args.depth)
    trainer = InverseRUMTrainer(
        grid_features=data["static_t"],            # (24, N, F) — T=24 path
        edge_index=edge_index,
        observed_OD=data["F_ij_t"],                # (24, N, N)
        t_ij_t=data["t_ij_t"],                     # (24, N, N)
        log_d_ij=data["log_d"],
        K=args.K, utility_net=util_net,
        train_mask=data["train_mask"], val_mask=data["val_mask"],
        device="cpu", seed=args.seed,
        epochs=args.epochs, patience=args.patience, residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=True,
        enforce_mainstream_direction=args.enforce_main,
        verbose=True,
    )

    t0 = time.time()
    theta, alpha, beta_t, beta_c, gamma, log = trainer.fit()
    train_seconds = time.time() - t0

    pred = trainer.predict_OD().detach().cpu().numpy()                  # (24, N, N)
    F_obs_np = data["F_ij_t"].cpu().numpy()
    val_mask_np = data["val_mask"].cpu().numpy().astype(bool)
    train_mask_np = data["train_mask"].cpu().numpy().astype(bool)
    # Per-hour CPC + summed-day CPC
    daily_pred = pred.sum(0); daily_obs = F_obs_np.sum(0)
    cpc_daily_val = cpc(daily_obs, daily_pred, val_mask_np)
    cpc_daily_train = cpc(daily_obs, daily_pred, train_mask_np)
    cpc_per_hour = []
    for t in range(24):
        cpc_per_hour.append({
            "hour": t,
            "cpc_train": cpc(F_obs_np[t], pred[t], train_mask_np),
            "cpc_val": cpc(F_obs_np[t], pred[t], val_mask_np),
        })

    print(f"\n[hourly-ckpt] trained {train_seconds:.1f}s | "
          f"daily train_CPC={cpc_daily_train:.4f} val_CPC={cpc_daily_val:.4f}")
    print(f"[hourly-ckpt] hourly val CPC range: "
          f"[{min(r['cpc_val'] for r in cpc_per_hour):.4f}, "
          f"{max(r['cpc_val'] for r in cpc_per_hour):.4f}]")
    bt_arr = beta_t.detach().cpu().numpy()
    print(f"[hourly-ckpt] beta_t shape={bt_arr.shape} mean={bt_arr.mean():+.4f} "
          f"min={bt_arr.min():+.4f} max={bt_arr.max():+.4f}")

    suffix = "constrained" if args.enforce_main else "unconstrained"
    ckpt_path = OUT_DIR / f"phase_b_v6_hourly_{suffix}_seed{args.seed}.pt"
    torch.save({
        "gnn_state": trainer.gnn.state_dict(),
        "rum_state": trainer.rum.state_dict(),
        "config": {
            "in_features": Fdim, "hidden": args.hidden, "depth": args.depth,
            "out": 1, "K": args.K, "T": 24,
            "enforce_mainstream_direction": args.enforce_main,
            "enable_wage_attraction": True,
        },
        "metrics": {
            "cpc_daily_train": cpc_daily_train, "cpc_daily_val": cpc_daily_val,
            "cpc_per_hour": cpc_per_hour,
            "beta_base_mean": float(bt_arr.mean()),
            "beta_c": float(beta_c),
            "phi": float(trainer.rum.phi.item()),
            "psi": float(trainer.rum.psi.item()),
            "delta": float(trainer.rum.delta.item()),
            "gamma": float(gamma),
        },
        "meta": {
            "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
            "train_seconds": train_seconds, "T": 24, "framework": "phase_B_v6_hourly",
        },
    }, ckpt_path)
    print(f"[hourly-ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
