"""Final paper-grade evaluation: per-borough CPC + bootstrap CI + paired
significance test ours vs baselines.

Trains the chosen winning configuration N times (default 5 seeds),
captures all F_pred matrices, then computes:

  1. Multi-metric on val (CPC, MAE, RMSE, Spearman, KL)
  2. Per-borough CPC breakdown (which boroughs we win/lose on)
  3. Bootstrap 95% CI on ensemble CPC (resample origins with replacement)
  4. Paired bootstrap significance test: ours vs gravity, ours vs radiation
     (resample origin set, count how often ours > baseline)

Output: evaluation_outputs/paper_a/final_paper_eval_<config>.json
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
from experiments.paper_a.baselines_spatial_holdout import fit_gravity_softmax, fit_radiation
from experiments.paper_a.squeeze_gat import all_metrics, train_one as gat_train_one


def per_borough_cpc(F_obs, F_pred, gbi, boroughs, val_borough_idxs):
    rows = []
    for bi in val_borough_idxs:
        b_grids = np.where(gbi == bi)[0]
        if len(b_grids) < 5:
            continue
        F_obs_b = F_obs[b_grids].flatten()
        F_pred_b = F_pred[b_grids].flatten()
        if F_obs_b.sum() == 0 or F_pred_b.sum() == 0:
            continue
        cp = float(2 * np.minimum(F_obs_b, F_pred_b).sum() /
                   max(F_obs_b.sum() + F_pred_b.sum(), 1.0))
        rows.append({"borough": str(boroughs[bi]), "n_grids": int(len(b_grids)),
                     "cpc": cp})
    return rows


def bootstrap_origin_cpc(F_obs, F_pred, val_origins, n_boot=500, rng=None):
    """Resample val origins with replacement; report 95% CI on CPC."""
    rng = rng or np.random.default_rng(42)
    cpcs = []
    for _ in range(n_boot):
        sample = rng.choice(val_origins, size=len(val_origins), replace=True)
        F_obs_s = F_obs[sample].flatten()
        F_pred_s = F_pred[sample].flatten()
        cpcs.append(2 * np.minimum(F_obs_s, F_pred_s).sum() /
                    max(F_obs_s.sum() + F_pred_s.sum(), 1.0))
    cpcs = np.array(cpcs)
    return {
        "mean": float(cpcs.mean()),
        "ci95_lo": float(np.percentile(cpcs, 2.5)),
        "ci95_hi": float(np.percentile(cpcs, 97.5)),
        "std": float(cpcs.std()),
    }


def paired_bootstrap_diff(F_obs, F_pred_a, F_pred_b, val_origins,
                           n_boot=1000, rng=None):
    """Paired bootstrap test: resample val origins; for each, compute
    cpc_a - cpc_b. Report how often a > b (one-sided p-value = 1 - this)."""
    rng = rng or np.random.default_rng(123)
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(val_origins, size=len(val_origins), replace=True)
        obs_s = F_obs[sample].flatten()
        a_s = F_pred_a[sample].flatten()
        b_s = F_pred_b[sample].flatten()
        cpc_a = 2 * np.minimum(obs_s, a_s).sum() / max(obs_s.sum() + a_s.sum(), 1.0)
        cpc_b = 2 * np.minimum(obs_s, b_s).sum() / max(obs_s.sum() + b_s.sum(), 1.0)
        diffs.append(cpc_a - cpc_b)
    diffs = np.array(diffs)
    return {
        "diff_mean": float(diffs.mean()),
        "diff_ci95_lo": float(np.percentile(diffs, 2.5)),
        "diff_ci95_hi": float(np.percentile(diffs, 97.5)),
        "frac_a_better": float((diffs > 0).mean()),
        "p_one_sided_a_le_b": float((diffs <= 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--layer_type", type=str, default="gat", choices=["sage", "gat"])
    parser.add_argument("--enforce_main", action="store_true",
                        help="if set, constrain mainstream direction (Variant 6)")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    print("[final-eval] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)
    boroughs = cache["boroughs"]; gbi = cache["grid_borough_idx"]
    F_obs_np = data["F_ij"].cpu().numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)
    val_origins = np.where(val_mask_np)[0]
    val_borough_idxs = sorted(set(gbi[val_origins].tolist()))

    config_label = (f"{args.layer_type}_"
                    f"{'constrained' if args.enforce_main else 'unconstrained'}_"
                    f"K{args.K}_h{args.gat_heads}_seeds{args.seeds}")
    print(f"[final-eval] config: {config_label}")
    print(f"[final-eval] val origins: {len(val_origins)} across {len(val_borough_idxs)} boroughs")

    # Train ours N seeds
    preds_ours = []
    metrics_per_seed = []
    for seed in range(args.seeds):
        cpc_v, F_pred, m = gat_train_one(
            seed, args.enforce_main, args.layer_type, data, edge_index,
            train_mask, val_mask, gat_heads=args.gat_heads, K=args.K,
        )
        print(f"  s{seed}: CPC={m['cpc']:.4f} MAE={m['mae']:.3f} "
              f"Spr={m['spearman']:.3f} KL={m['kl_per_origin']:.3f}")
        preds_ours.append(F_pred)
        metrics_per_seed.append(m)

    # Ensemble
    ens_ours = np.mean(np.stack(preds_ours, axis=0), axis=0)
    metrics_ens = all_metrics(F_obs_np, ens_ours, val_mask_np)
    print(f"\n[final-eval] ensemble multi-metric: "
          f"CPC={metrics_ens['cpc']:.4f}  MAE={metrics_ens['mae']:.3f}  "
          f"Spr={metrics_ens['spearman']:.3f}  KL={metrics_ens['kl_per_origin']:.3f}")

    # Per-borough CPC
    print(f"\n[final-eval] per-borough CPC:")
    pb = per_borough_cpc(F_obs_np, ens_ours, gbi, boroughs, val_borough_idxs)
    for r in pb:
        print(f"  {r['borough']:<25} ({r['n_grids']:>3} grids)  CPC={r['cpc']:.4f}")

    # Bootstrap CI
    print(f"\n[final-eval] bootstrap 95% CI on ensemble CPC ({args.n_boot} resamples):")
    boot_ours = bootstrap_origin_cpc(F_obs_np, ens_ours, val_origins, n_boot=args.n_boot)
    print(f"  ours: CPC = {boot_ours['mean']:.4f}  "
          f"95% CI [{boot_ours['ci95_lo']:.4f}, {boot_ours['ci95_hi']:.4f}]  "
          f"std={boot_ours['std']:.4f}")

    # Baseline predictions (gravity + radiation, deterministic)
    print(f"\n[final-eval] computing baselines for paired test ...")
    df_grid = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    grid_ids = cache["grid_ids"].tolist()
    df_grid = df_grid.set_index("grid_id").reindex(grid_ids)
    D_j = torch.tensor(df_grid["total_employment"].fillna(0.0).to_numpy(), dtype=torch.float32)
    m_i = df_grid["population"].fillna(0.0).to_numpy().astype(np.float64)
    n_j = df_grid["total_employment"].fillna(0.0).to_numpy().astype(np.float64)
    coords = cache["coords_bng"].astype(np.float64)

    F_grav, info_grav = fit_gravity_softmax(data["F_ij"], D_j, data["t_ij"],
                                             data["log_d"], train_mask, epochs=400, lr=0.05)
    F_grav_np = F_grav.cpu().numpy()
    F_rad_np = fit_radiation(data["F_ij"], m_i, n_j, coords).cpu().numpy()

    # Paired bootstrap tests
    print(f"\n[final-eval] paired bootstrap significance ({args.n_boot} resamples):")
    paired_grav = paired_bootstrap_diff(F_obs_np, ens_ours, F_grav_np, val_origins,
                                         n_boot=args.n_boot)
    print(f"  ours vs gravity: ΔCPC = {paired_grav['diff_mean']:.4f}  "
          f"95% CI [{paired_grav['diff_ci95_lo']:.4f}, {paired_grav['diff_ci95_hi']:.4f}]  "
          f"P(ours > gravity) = {paired_grav['frac_a_better']*100:.1f}%  "
          f"p-value = {paired_grav['p_one_sided_a_le_b']:.4f}")
    paired_rad = paired_bootstrap_diff(F_obs_np, ens_ours, F_rad_np, val_origins,
                                        n_boot=args.n_boot)
    print(f"  ours vs radiation: ΔCPC = {paired_rad['diff_mean']:.4f}  "
          f"95% CI [{paired_rad['diff_ci95_lo']:.4f}, {paired_rad['diff_ci95_hi']:.4f}]  "
          f"P(ours > rad) = {paired_rad['frac_a_better']*100:.1f}%  "
          f"p-value = {paired_rad['p_one_sided_a_le_b']:.4f}")

    summary = {
        "config": config_label,
        "n_seeds": args.seeds,
        "n_boot": args.n_boot,
        "heldout_boroughs": HELDOUT_BOROUGHS,
        "metrics_per_seed": metrics_per_seed,
        "metrics_ensemble": metrics_ens,
        "bootstrap_ours": boot_ours,
        "per_borough_cpc": pb,
        "paired_vs_gravity": {
            "gravity_ensemble_cpc_overall": float(2 * np.minimum(F_obs_np[val_mask_np], F_grav_np[val_mask_np]).sum() /
                                                   max(F_obs_np[val_mask_np].sum() + F_grav_np[val_mask_np].sum(), 1.0)),
            **paired_grav,
        },
        "paired_vs_radiation": {
            "radiation_ensemble_cpc_overall": float(2 * np.minimum(F_obs_np[val_mask_np], F_rad_np[val_mask_np]).sum() /
                                                     max(F_obs_np[val_mask_np].sum() + F_rad_np[val_mask_np].sum(), 1.0)),
            **paired_rad,
        },
    }
    out = ROOT / "evaluation_outputs" / "paper_a" / f"final_paper_eval_{config_label}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[final-eval] wrote {out}")


if __name__ == "__main__":
    main()
