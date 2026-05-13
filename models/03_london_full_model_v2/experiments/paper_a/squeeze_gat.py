"""Squeeze trick #4: replace GraphSAGE with GAT (Graph Attention).

Same training pipeline as Phase B v6 but neighbour aggregation uses
multi-head attention rather than uniform mean. Tests both constrained
and unconstrained × 5 seeds.

Reports CPC alongside MAE/RMSE/Spearman/KL multi-metric for paper rigor.
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


def all_metrics(F_obs, F_pred, mask):
    """CPC + MAE + RMSE + Spearman + KL on flow vectors masked by mask."""
    obs = F_obs[mask]; pr = F_pred[mask]
    obs_f = obs.flatten(); pr_f = pr.flatten()
    cpc_val = float(2 * np.minimum(pr_f, obs_f).sum() / max(pr_f.sum() + obs_f.sum(), 1.0))
    mae = float(np.abs(pr_f - obs_f).mean())
    rmse = float(np.sqrt(((pr_f - obs_f) ** 2).mean()))
    from scipy.stats import spearmanr
    nonzero = (obs_f > 0) | (pr_f > 0)
    rho, _ = spearmanr(obs_f[nonzero], pr_f[nonzero])
    # KL(obs || pred), normalised distributions per row, averaged
    eps = 1e-9
    obs_p = obs / (obs.sum(axis=1, keepdims=True) + eps)
    pr_p = pr / (pr.sum(axis=1, keepdims=True) + eps)
    kl = float((obs_p * (np.log(obs_p + eps) - np.log(pr_p + eps))).sum(axis=1).mean())
    return {"cpc": cpc_val, "mae": mae, "rmse": rmse, "spearman": float(rho), "kl_per_origin": kl}


def train_one(seed, enforce_main, layer_type, data, edge_index, train_mask, val_mask,
              gat_heads=4, K=50):
    torch.manual_seed(seed); np.random.seed(seed)
    Fdim = data["static"].shape[-1]
    util_net = StructuralGNN(in_features=Fdim, hidden=128, out=1, depth=5,
                             layer_type=layer_type, gat_heads=gat_heads)
    trainer = InverseRUMTrainer(
        grid_features=data["static"], edge_index=edge_index,
        observed_OD=data["F_ij"], t_ij_t=data["t_ij"], log_d_ij=data["log_d"],
        K=K, utility_net=util_net,
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
    F_obs_np = data["F_ij"].cpu().numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)
    metrics = all_metrics(F_obs_np, F_pred, val_mask_np)
    metrics.update({
        "seed": seed, "constrained": enforce_main, "layer_type": layer_type,
        "K": K, "gat_heads": gat_heads,
        "beta_base": float(beta_t.mean().item()),
        "beta_c": float(beta_c),
        "phi": float(trainer.rum.phi.item()),
        "psi": float(trainer.rum.psi.item()),
        "delta": float(trainer.rum.delta.item()),
        "gamma": float(gamma),
        "train_seconds": dt,
    })
    return metrics["cpc"], F_pred, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--gat_heads", type=int, default=4)
    args = parser.parse_args()

    print("[squeeze-gat] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)
    F_obs_np = data["F_ij"].cpu().numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)

    rows = []
    preds_by_cond = {("gat", True): [], ("gat", False): []}
    for cond_label, enforce in [("constrained", True), ("unconstrained", False)]:
        print(f"\n[squeeze-gat] === GAT, {cond_label}, K={args.K}, heads={args.gat_heads} ===")
        for seed in range(args.seeds):
            cpc_v, F_pred, m = train_one(seed, enforce, "gat", data, edge_index,
                                          train_mask, val_mask,
                                          gat_heads=args.gat_heads, K=args.K)
            print(f"  s{seed}: CPC={m['cpc']:.4f} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
                  f"Spr={m['spearman']:.3f} KL={m['kl_per_origin']:.3f} "
                  f"beta={m['beta_base']:+.4f} t={m['train_seconds']:.1f}s")
            rows.append(m)
            preds_by_cond[("gat", enforce)].append(F_pred)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "squeeze_gat.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[squeeze-gat] wrote {out_csv}")

    print(f"\n[squeeze-gat] === ENSEMBLE (mean over {args.seeds} seeds) ===")
    summary = {"per_run": rows, "ensemble": {}}
    for cond_label, enforce in [("constrained", True), ("unconstrained", False)]:
        preds = preds_by_cond[("gat", enforce)]
        ens = np.mean(np.stack(preds, axis=0), axis=0)
        m_ens = all_metrics(F_obs_np, ens, val_mask_np)
        # per-seed mean / std
        df_c = df[df["constrained"] == enforce]
        cpc_seed = df_c["cpc"].agg(["mean", "std"])
        print(f"  {cond_label:<14} per-seed mean={cpc_seed['mean']:.4f} +- {cpc_seed['std']:.4f}  "
              f"|  ENSEMBLE CPC={m_ens['cpc']:.4f} MAE={m_ens['mae']:.3f} "
              f"Spr={m_ens['spearman']:.3f} KL={m_ens['kl_per_origin']:.3f}")
        summary["ensemble"][cond_label] = {
            "per_seed_cpc_mean": float(cpc_seed["mean"]),
            "per_seed_cpc_std": float(cpc_seed["std"]),
            **m_ens,
        }

    print(f"\n[squeeze-gat] === full leaderboard (best ensemble) ===")
    print(f"  Gravity (Wilson):                 CPC=0.170")
    print(f"  Radiation (Simini12):             CPC=0.174")
    print(f"  Deep Gravity (our impl):          CPC=0.016")
    print(f"  SAGE constrained ensemble:        CPC=0.290")
    print(f"  SAGE unconstrained ensemble:      CPC=0.344")
    for cond_label in ["constrained", "unconstrained"]:
        m = summary["ensemble"][cond_label]
        print(f"  GAT {cond_label:<14} ensemble:   CPC={m['cpc']:.4f}")

    out_json = ROOT / "evaluation_outputs" / "paper_a" / "squeeze_gat.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[squeeze-gat] wrote {out_json}")


if __name__ == "__main__":
    main()
