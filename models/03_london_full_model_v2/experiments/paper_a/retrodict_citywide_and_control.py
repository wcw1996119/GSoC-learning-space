"""D-RETRO 升级版：city-wide forward retrodict + Bromley negative control.

Goal: 测 model 真 generalization，不是单点 IIA-confounded intervention。

Three tests:

  1. CITY-WIDE FORWARD RETRODICT
     Use 2011-trained model on 2024 features (everything, not just Stratford),
     predict 2021 OD allocation. This tests "does the 2011-fitted model
     generalize to 2024-state inputs to recover 2021-observed flows?".
     Reports: per-borough CPC, Spearman over all OD pairs.

  2. STRATFORD INTERVENTION (already done by retrodict_stratford.py)
     Single-cluster do(X_Stratford = 2024) on 2011 baseline. Over-predicts
     5× due to MNL IIA — known limitation.

  3. BROMLEY NEGATIVE CONTROL
     Suburban borough with NO major intervention 2011-2021. Apply same
     do(X_Bromley = 2024) → predicted Δshare should be ≈ 0 (or much smaller
     than Stratford's). If model predicts large Δshare for Bromley too →
     model is unstable / sensitive to any feature perturbation.

Usage:
    python experiments/paper_a/retrodict_citywide_and_control.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index
from experiments.paper_a.train_phase_b_v6_2011 import load_data_2011
from experiments.paper_a.scenario_A_spatial import predict_OD_with_X


STRATFORD_GRIDS = [
    "L035_029", "L036_029", "L035_028", "L035_030", "L036_028",
    "L036_030", "L034_029", "L034_030", "L037_029",
]


def cpc(F_obs, F_pred, mask=None):
    if mask is not None:
        F_obs = F_obs[mask]; F_pred = F_pred[mask]
    return float(2.0 * np.minimum(F_obs, F_pred).sum() / max(F_obs.sum() + F_pred.sum(), 1.0))


def normalise_per_origin(F):
    rs = F.sum(axis=1, keepdims=True)
    return np.divide(F, rs, where=(rs > 0), out=np.zeros_like(F))


def load_2024_static() -> torch.Tensor:
    cache_2024 = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    return torch.tensor(cache_2024["static_features"], dtype=torch.float32)


def main():
    ckpt_path = ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_2011_seed0.pt"
    print("[retrodict-3] loading 2011 ckpt + data ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[retrodict-3] 2011 ckpt: train_CPC={ckpt['metrics']['train_cpc']:.4f} "
          f"val_CPC={ckpt['metrics']['val_cpc']:.4f}")

    data = load_data_2011()
    edge_index = build_edge_index(data["t_ij"], k=10)

    util_net = StructuralGNN(in_features=cfg["in_features"], hidden=cfg["hidden"],
                             out=cfg["out"], depth=cfg["depth"])
    trainer = InverseRUMTrainer(
        grid_features=data["static"], edge_index=edge_index,
        observed_OD=data["F_ij"], t_ij_t=data["t_ij"], log_d_ij=data["log_d"],
        K=cfg["K"], utility_net=util_net,
        train_mask=data["train_mask"], val_mask=data["val_mask"],
        device="cpu", seed=0, epochs=1, patience=1, residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=cfg["enable_wage_attraction"],
        enforce_mainstream_direction=cfg["enforce_mainstream_direction"],
    )
    trainer.gnn.load_state_dict(ckpt["gnn_state"])
    trainer.rum.load_state_dict(ckpt["rum_state"])
    trainer.gnn.eval(); trainer.rum.eval()

    cache_2011 = np.load(ROOT / "data" / "processed" / "demo_cache_2011.npz",
                          allow_pickle=True)
    grid_ids = cache_2011["grid_ids"].tolist()
    g2idx = {g: i for i, g in enumerate(grid_ids)}
    boroughs = cache_2011["boroughs"]; gbi = cache_2011["grid_borough_idx"]

    static_2024 = load_2024_static()

    # Frozen baseline norm stats from 2011 X
    norm_stats = trainer.gnn.compute_norm_stats(
        data["static"].unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )

    # Observed flows
    F_2011_obs = data["F_ij"].cpu().numpy()
    F_2021_obs = np.zeros_like(F_2011_obs)
    od_2021 = pd.read_csv(ROOT / "data" / "processed" / "grid_od_2021_full.csv")
    for _, r in od_2021.iterrows():
        i = g2idx.get(r["grid_home"]); j = g2idx.get(r["grid_work"])
        if i is None or j is None:
            continue
        F_2021_obs[i, j] += r["count21"]
    print(f"[retrodict-3] F_2011_obs sum {F_2011_obs.sum():,.0f} "
          f"F_2021_obs sum {F_2021_obs.sum():,.0f}")

    P_2011_obs = normalise_per_origin(F_2011_obs)
    P_2021_obs = normalise_per_origin(F_2021_obs)

    # ========================================================================
    # 1. CITY-WIDE FORWARD RETRODICT
    # ========================================================================
    print("\n[retrodict-3] === Test 1: city-wide forward retrodict ===")
    print("[retrodict-3] using 2011-trained model + 2024 features → predicting 2021 OD")
    F_pred_citywide = predict_OD_with_X(trainer, static_2024, norm_stats=norm_stats).sum(0).cpu().numpy()
    P_pred_citywide = normalise_per_origin(F_pred_citywide)

    # CPC on per-OD-pair flows: predicted (city-wide-2024) vs observed 2021
    # Need to scale predicted to 2021 totals first (different scale)
    O_2011 = F_2011_obs.sum(axis=1, keepdims=True)
    O_2021 = F_2021_obs.sum(axis=1, keepdims=True)
    pred_scaled = P_pred_citywide * O_2021                  # rescale row-mass to 2021
    cpc_citywide = cpc(F_2021_obs, pred_scaled)
    print(f"  city-wide CPC (predicted 2024-features → observed 2021): {cpc_citywide:.4f}")

    # Sanity: how well does the 2011 model predict 2011 itself?
    F_pred_2011 = predict_OD_with_X(trainer, data["static"], norm_stats=norm_stats).sum(0).cpu().numpy()
    cpc_self = cpc(F_2011_obs, F_pred_2011)
    print(f"  in-sample CPC (2011 model → 2011 obs): {cpc_self:.4f}")

    # Per-borough CPC
    print(f"  per-borough CPC (2024-features pred vs 2021 obs):")
    for b_idx, b_name in enumerate(boroughs):
        b_grids = np.where(gbi == b_idx)[0]
        if len(b_grids) < 5:
            continue
        F_obs_b = F_2021_obs[b_grids].flatten()
        F_pred_b = pred_scaled[b_grids].flatten()
        if F_obs_b.sum() == 0 or F_pred_b.sum() == 0:
            continue
        b_cpc = cpc(F_obs_b, F_pred_b)
        if b_cpc < 0.3 or b_cpc > 0.7 or b_name in ("Westminster", "Camden", "Hackney",
                                                     "Bromley", "Newham"):
            print(f"    {b_name:<25} ({len(b_grids):>3} grids)  CPC={b_cpc:.4f}")

    from scipy.stats import spearmanr
    flat_obs = F_2021_obs.flatten()
    flat_pred = pred_scaled.flatten()
    nonzero = (flat_obs > 0) | (flat_pred > 0)
    rho_all, _ = spearmanr(flat_obs[nonzero], flat_pred[nonzero])
    print(f"  Spearman ρ all OD pairs: {rho_all:.4f}")

    # ========================================================================
    # 2. STRATFORD INTERVENTION (re-run for direct comparison in same harness)
    # ========================================================================
    print("\n[retrodict-3] === Test 2: Stratford single-cluster intervention ===")
    strat_idx = [g2idx[g] for g in STRATFORD_GRIDS if g in g2idx]
    X_strat = data["static"].clone()
    for i in strat_idx:
        X_strat[i] = static_2024[i]
    F_pred_strat = predict_OD_with_X(trainer, X_strat, norm_stats=norm_stats).sum(0).cpu().numpy()
    P_pred_strat = normalise_per_origin(F_pred_strat)

    inflow_strat_pred_baseline = P_pred_2011 if False else normalise_per_origin(F_pred_2011)
    delta_pred_strat = P_pred_strat[:, strat_idx].sum(axis=1) - inflow_strat_pred_baseline[:, strat_idx].sum(axis=1)
    delta_obs_strat = P_2021_obs[:, strat_idx].sum(axis=1) - P_2011_obs[:, strat_idx].sum(axis=1)
    print(f"  predicted Δshare (mean over origins): {delta_pred_strat.mean()*100:+.3f} pp")
    print(f"  observed  Δshare (mean over origins): {delta_obs_strat.mean()*100:+.3f} pp")
    print(f"  ratio pred/obs: {delta_pred_strat.mean() / max(delta_obs_strat.mean(), 1e-9):+.2f}")

    # ========================================================================
    # 3. BROMLEY NEGATIVE CONTROL
    # ========================================================================
    print("\n[retrodict-3] === Test 3: Bromley NEGATIVE control ===")
    bromley_idx_borough = list(boroughs).index("Bromley")
    bromley_grids_idx = np.where(gbi == bromley_idx_borough)[0].tolist()
    print(f"  Bromley: {len(bromley_grids_idx)} grids (no major regeneration 2011-2021)")

    X_bromley = data["static"].clone()
    for i in bromley_grids_idx:
        X_bromley[i] = static_2024[i]
    F_pred_brom = predict_OD_with_X(trainer, X_bromley, norm_stats=norm_stats).sum(0).cpu().numpy()
    P_pred_brom = normalise_per_origin(F_pred_brom)

    # Bromley's 2011 vs 2024 employment, as sanity
    cache_2024 = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    feats_2024 = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    feats_2011 = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features_2011_proper.csv")
    bromley_emp_2024 = feats_2024.iloc[bromley_grids_idx]["total_employment"].sum()
    bromley_emp_2011 = feats_2011.iloc[bromley_grids_idx]["total_employment"].sum()
    print(f"  Bromley emp 2011: {bromley_emp_2011:,.0f}")
    print(f"  Bromley emp 2024: {bromley_emp_2024:,.0f}  (ratio {bromley_emp_2024/max(bromley_emp_2011,1):.2f}x)")

    delta_pred_brom = (P_pred_brom[:, bromley_grids_idx].sum(axis=1)
                       - inflow_strat_pred_baseline[:, bromley_grids_idx].sum(axis=1))
    delta_obs_brom = (P_2021_obs[:, bromley_grids_idx].sum(axis=1)
                      - P_2011_obs[:, bromley_grids_idx].sum(axis=1))
    print(f"  predicted Δshare Bromley (mean origins): {delta_pred_brom.mean()*100:+.3f} pp")
    print(f"  observed  Δshare Bromley (mean origins): {delta_obs_brom.mean()*100:+.3f} pp")
    print(f"  ratio pred/obs: {delta_pred_brom.mean() / max(delta_obs_brom.mean(), 1e-9):+.2f}")
    print(f"  pred Bromley / pred Stratford magnitude: "
          f"{abs(delta_pred_brom.mean()) / max(abs(delta_pred_strat.mean()), 1e-9):.3f}")
    print(f"  (lower is better for negative control: model should NOT predict big "
          f"Bromley change like it does for Stratford)")

    # ========================================================================
    # SAVE
    # ========================================================================
    summary = {
        "test_1_citywide_forward": {
            "in_sample_cpc_2011": float(cpc_self),
            "out_of_sample_cpc_2021": float(cpc_citywide),
            "spearman_all_OD_pairs": float(rho_all),
            "interpretation": "model trained on 2011 generalises to 2024-features → 2021 OD with this CPC",
        },
        "test_2_stratford_intervention": {
            "delta_share_predicted_pp": float(delta_pred_strat.mean() * 100),
            "delta_share_observed_pp": float(delta_obs_strat.mean() * 100),
            "ratio_pred_over_obs": float(delta_pred_strat.mean() / max(delta_obs_strat.mean(), 1e-9)),
            "interpretation": "single-cluster intervention; over-prediction expected (MNL IIA + isolated treatment)",
        },
        "test_3_bromley_negative_control": {
            "n_grids": len(bromley_grids_idx),
            "bromley_emp_2011": float(bromley_emp_2011),
            "bromley_emp_2024": float(bromley_emp_2024),
            "bromley_emp_ratio": float(bromley_emp_2024 / max(bromley_emp_2011, 1)),
            "delta_share_predicted_pp": float(delta_pred_brom.mean() * 100),
            "delta_share_observed_pp": float(delta_obs_brom.mean() * 100),
            "ratio_pred_over_obs": float(delta_pred_brom.mean() / max(delta_obs_brom.mean(), 1e-9)),
            "magnitude_relative_to_stratford": float(abs(delta_pred_brom.mean()) / max(abs(delta_pred_strat.mean()), 1e-9)),
            "interpretation": "should be small if model is specific; large => model says 'everything goes up' regardless",
        },
    }
    out = ROOT / "evaluation_outputs" / "paper_a" / "retrodict_3tests.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[retrodict-3] wrote {out}")


if __name__ == "__main__":
    main()
