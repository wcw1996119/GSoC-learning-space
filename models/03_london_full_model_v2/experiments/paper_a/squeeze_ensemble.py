"""Squeeze trick #1+#2: unconstrained variant + 5-seed ensemble.

Trains both constrained (Variant 6) and unconstrained (Variant 4) versions
with 5 seeds each on spatial holdout (4-borough leave-out). Reports:
  - per-seed val CPC for each condition
  - ensemble (mean prediction across seeds) val CPC for each condition

Output:
  evaluation_outputs/paper_a/squeeze_ensemble.csv  (10 rows)
  evaluation_outputs/paper_a/squeeze_ensemble.json (per-condition summary)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, cpc, load_data
from experiments.paper_a.spatial_holdout import HELDOUT_BOROUGHS, make_spatial_masks


def train_one(seed: int, enforce_main: bool, data: dict,
              edge_index: torch.Tensor, train_mask: torch.Tensor,
              val_mask: torch.Tensor) -> tuple[float, np.ndarray, dict]:
    """Train one (seed, condition). Returns (val_cpc, F_pred matrix, metrics)."""
    torch.manual_seed(seed); np.random.seed(seed)
    Fdim = data["static"].shape[-1]
    util_net = StructuralGNN(in_features=Fdim, hidden=128, out=1, depth=5)
    trainer = InverseRUMTrainer(
        grid_features=data["static"], edge_index=edge_index,
        observed_OD=data["F_ij"], t_ij_t=data["t_ij"], log_d_ij=data["log_d"],
        K=50, utility_net=util_net,
        train_mask=train_mask, val_mask=val_mask,
        device="cpu", seed=seed, epochs=150, patience=150, residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=True,
        enforce_mainstream_direction=enforce_main,
    )
    t0 = time.time()
    theta, alpha, beta_t, beta_c, gamma, log = trainer.fit()
    dt = time.time() - t0

    pred = trainer.predict_OD().detach().cpu().numpy()
    F_pred = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else (pred.sum(0) if pred.ndim == 3 else pred)
    val_mask_np = val_mask.cpu().numpy().astype(bool)
    val_cpc = cpc(data["F_ij"].cpu().numpy(), F_pred, val_mask_np)
    metrics = {
        "seed": seed, "constrained": enforce_main,
        "val_cpc_spatial": val_cpc,
        "beta_base": float(beta_t.mean().item()),
        "beta_c": float(beta_c),
        "phi": float(trainer.rum.phi.item()),
        "psi": float(trainer.rum.psi.item()),
        "delta": float(trainer.rum.delta.item()),
        "gamma": float(gamma),
        "train_seconds": dt,
    }
    return val_cpc, F_pred, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    print("[squeeze] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)
    F_obs_np = data["F_ij"].cpu().numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)
    print(f"[squeeze] holdout: {mask_info}")

    rows = []
    preds_constrained = []      # list of (N, N) F_pred
    preds_unconstrained = []
    for cond_label, enforce in [("constrained", True), ("unconstrained", False)]:
        print(f"\n[squeeze] === condition: {cond_label} (enforce_main={enforce}) ===")
        for seed in range(args.seeds):
            val_cpc, F_pred, m = train_one(seed, enforce, data, edge_index, train_mask, val_mask)
            print(f"  seed={seed}: val_CPC={val_cpc:.4f}  beta={m['beta_base']:+.4f}  "
                  f"phi={m['phi']:+.4f}  psi={m['psi']:+.4f}  t={m['train_seconds']:.1f}s")
            rows.append(m)
            if enforce:
                preds_constrained.append(F_pred)
            else:
                preds_unconstrained.append(F_pred)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "squeeze_ensemble.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[squeeze] wrote {out_csv}")

    # Ensemble: mean of predictions
    print(f"\n[squeeze] === ENSEMBLE (mean over {args.seeds} seeds) ===")
    ens_constrained = np.mean(np.stack(preds_constrained, axis=0), axis=0)
    ens_unconstrained = np.mean(np.stack(preds_unconstrained, axis=0), axis=0)
    cpc_ens_c = cpc(F_obs_np, ens_constrained, val_mask_np)
    cpc_ens_u = cpc(F_obs_np, ens_unconstrained, val_mask_np)
    cpc_seed_c = df[df["constrained"] == True]["val_cpc_spatial"].agg(["mean", "std"])
    cpc_seed_u = df[df["constrained"] == False]["val_cpc_spatial"].agg(["mean", "std"])

    print(f"  constrained:   per-seed mean={cpc_seed_c['mean']:.4f} +- {cpc_seed_c['std']:.4f}  "
          f"|  ENSEMBLE={cpc_ens_c:.4f}  (Δ={cpc_ens_c - cpc_seed_c['mean']:+.4f})")
    print(f"  unconstrained: per-seed mean={cpc_seed_u['mean']:.4f} +- {cpc_seed_u['std']:.4f}  "
          f"|  ENSEMBLE={cpc_ens_u:.4f}  (Δ={cpc_ens_u - cpc_seed_u['mean']:+.4f})")

    # Compare to old reference
    print(f"\n[squeeze] reference: Phase B v6 (constrained, single seed): 0.2892")
    print(f"  best in this run:")
    best_cond = "constrained ensemble" if cpc_ens_c >= cpc_ens_u else "unconstrained ensemble"
    best_cpc = max(cpc_ens_c, cpc_ens_u)
    print(f"    {best_cond}: CPC = {best_cpc:.4f}  (Δ vs ref = {best_cpc - 0.2892:+.4f})")

    summary = {
        "n_seeds": args.seeds,
        "heldout_boroughs": HELDOUT_BOROUGHS,
        "constrained": {
            "per_seed_mean": float(cpc_seed_c["mean"]),
            "per_seed_std": float(cpc_seed_c["std"]),
            "ensemble": float(cpc_ens_c),
        },
        "unconstrained": {
            "per_seed_mean": float(cpc_seed_u["mean"]),
            "per_seed_std": float(cpc_seed_u["std"]),
            "ensemble": float(cpc_ens_u),
        },
        "phase_b_v6_single_seed_baseline": 0.2892,
        "best_condition": best_cond,
        "best_cpc": float(best_cpc),
    }
    out_json = ROOT / "evaluation_outputs" / "paper_a" / "squeeze_ensemble.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[squeeze] wrote {out_json}")


if __name__ == "__main__":
    main()
