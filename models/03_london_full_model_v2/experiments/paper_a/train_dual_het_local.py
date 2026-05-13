"""Train DS-SIGNN (DUAL_HET) locally on CPU for §3 Results scenarios.

Reproduces the Colab T4 distance-aware DUAL_HET recipe (see
distance_aware_results.json) but for 1 seed only — enough to produce a ckpt
that scenario A/B can load. Multi-seed CPC numbers stay sourced from Colab.

Recipe (matches Colab):
  - DualBranchMixtureTrainer
  - distance-aware mode share (walk_threshold_km=5.0)
  - tier_specific_delta=True
  - 200 epochs, patience=30
  - hidden=32, gru_hidden=32, n_sage=2, tcn=(3,5,7)
  - lr_theta=1e-3, lr_rum=1e-2, weight_decay=1e-4

Run:
  python experiments/paper_a/train_dual_het_local.py --seed 0 --epochs 200

Output:
  evaluation_outputs/paper_a/dual_het_seed0.pt
  evaluation_outputs/paper_a/dual_het_seed0.log.json
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

from models_lib.inverse_rum.dual_branch_mixture_trainer import (
    DualBranchMixtureTrainer, make_distance_aware_mode_share,
)
from models_lib.inverse_rum.flow_metrics import all_metrics

from experiments.paper_a.compare_four_variants import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--walk_threshold_km", type=float, default=5.0)
    parser.add_argument("--out_dir", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dual_het_seed{args.seed}.pt"
    log_path = out_dir / f"dual_het_seed{args.seed}.log.json"

    print(f"[train_dual_het] seed={args.seed}  epochs={args.epochs}  "
          f"patience={args.patience}  walk_threshold={args.walk_threshold_km}km")
    print(f"[train_dual_het] loading data ...")
    d = load_data()
    N, T = d["N"], d["T"]
    print(f"  N={N}  T={T}  edges={d['edge_index'].shape[1]}  "
          f"F_ij_t total={d['F_ij_t'].sum():.0f}")

    # Build pair-level distance-aware mode share (walk only feasible <5km)
    coords = torch.from_numpy(
        np.load(ROOT / "data" / "processed" / "demo_cache.npz",
                allow_pickle=True)["coords_bng"].astype(np.float32))
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_km = (torch.linalg.norm(diff, dim=-1) / 1000.0).clamp(min=0.1)

    mode_names = ["car", "transit", "walk"]
    pair_mode_share = make_distance_aware_mode_share(
        d["mode_share"], dist_km, mode_names,
        walk_threshold_km=args.walk_threshold_km,
    )
    print(f"  pair_mode_share shape={tuple(pair_mode_share.shape)}  "
          f"(rows sum to 1: {pair_mode_share.sum(dim=-1).mean():.3f})")

    # Sanity check ranges before training (per feedback_data_sanity_check.md)
    for nm, t in [("car", d["t_car"]), ("transit", d["t_transit"]), ("walk", d["t_walk"])]:
        v = t[t > 0]
        print(f"  t_{nm}: median {v.median():.1f}min  max {t.max():.1f}min  "
              f"(zeros={int((t == 0).sum())} of {t.numel()})")

    print(f"[train_dual_het] building trainer ...")
    trainer = DualBranchMixtureTrainer(
        X_static=d["X_static"], X_dynamic=d["X_dynamic"],
        edge_index=d["edge_index"], observed_OD=d["F_ij_t"],
        t_ij_per_mode={"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]},
        mode_share_per_origin=pair_mode_share,                  # (N, N, 3) distance-aware
        income_tier_props=d["income_tier_props"],
        log_d_ij=d["log_d"],
        occ_match=d["occ_match"],
        train_mask=d["train_mask"], val_mask=d["val_mask"],
        device="cpu", seed=args.seed,
        hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
        lr_theta=1e-3, lr_rum=1e-2, weight_decay=1e-4,
        epochs=args.epochs, patience=args.patience,
        tier_specific_delta=True,
        verbose=True,
    )

    print(f"[train_dual_het] fit() ...")
    t0 = time.time()
    encoder, head, log = trainer.fit()
    fit_time = time.time() - t0
    print(f"[train_dual_het] fit done in {fit_time:.0f}s ({len(log.train_nll)} epochs)")

    print(f"[train_dual_het] eval ...")
    pred = trainer.predict_OD()
    metrics = all_metrics(pred, d["F_ij_t"], d["val_mask"], K_top=10)
    print(f"  CPC={metrics['cpc']:.4f}  RMSE={metrics['rmse']:.3f}  "
          f"Pearson r={metrics['pearson_r']:.3f}")

    bm = head.beta_t_per_mode.mean(dim=0).tolist()                 # (M,)
    kap = head.kappa.tolist()                                      # (K,)
    delta_per_tier = head.delta_per_tier().tolist()                # (K,)
    print(f"  β_car={bm[0]:+.4f}  β_transit={bm[1]:+.4f}  β_walk={bm[2]:+.4f}")
    print(f"  γ={head.gamma.item():+.4f}")
    print(f"  κ={[round(x, 3) for x in kap]}  δ_per_tier={[round(x, 3) for x in delta_per_tier]}")

    # ---- save ckpt -----------------------------------------------------------
    ckpt = {
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "config": {
            "hidden_dim": 32, "gru_hidden": 32, "n_sage_layers": 2,
            "tcn_kernels": (3, 5, 7),
            "n_modes": 3, "n_tiers": 3, "n_hours": T,
            "F_static": d["X_static"].shape[1],
            "F_dynamic": d["X_dynamic"].shape[2],
            "tier_specific_delta": True,
            "mode_names": mode_names,
            "walk_threshold_km": args.walk_threshold_km,
            "seed": args.seed, "epochs_target": args.epochs, "patience": args.patience,
            "epochs_actual": len(log.train_nll),
        },
        "metrics": {
            **{k: float(v) for k, v in metrics.items()},
            "beta_car": bm[0], "beta_transit": bm[1], "beta_walk": bm[2],
            "gamma": float(head.gamma.item()),
            "kappa": kap,
            "delta_per_tier": delta_per_tier,
            "fit_time_s": fit_time,
        },
    }
    torch.save(ckpt, ckpt_path)
    print(f"[train_dual_het] wrote ckpt: {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.2f} MB)")

    # save log JSON for plotting / debugging
    log_dump = {
        "epoch": log.epoch,
        "train_nll": log.train_nll,
        "val_nll": log.val_nll,
        "cpc_val": log.cpc_val,
        "alpha": log.alpha,
        "beta_mean": log.beta_mean,
        "gamma": log.gamma,
    }
    with open(log_path, "w") as f:
        json.dump(log_dump, f, indent=2)
    print(f"[train_dual_het] wrote log: {log_path}")


if __name__ == "__main__":
    main()
