"""D-RETRO: Stratford 2011 → 2021 retrodiction.

Headline test of the Phase B v6 method's ability to predict the spatial
allocation effect of a known major intervention (Stratford regeneration:
Westfield 2011, 2012 Olympics, Olympic Park redevelopment).

Pipeline:
  1. Load Phase B v6 trained on 2011 data.
  2. Forward F_pred_2011 (baseline, all features at 2011 levels).
  3. Apply intervention: replace Stratford grids' EMP_BLOCK 2011 features
     with 2024 levels (the post-intervention employment + sector mix).
  4. Forward F_pred_2011_with_2021_emp_at_Stratford → predicted ΔF.
  5. Observed ΔF = (F_2021 / scale_2021) - (F_2011 / scale_2011), where
     each scale is the row-total mass of that snapshot. Working in
     proportional terms eliminates the 2021/2011 sample-size mismatch
     (2021 = 1.1% Census sample, 2011 = 100% Census).
  6. Report 4 retrodict metrics from §4.4 of methodology/retrodiction_scope.md.

Usage:
    python experiments/paper_a/retrodict_stratford.py \
        --ckpt evaluation_outputs/paper_a/phase_b_v6_2011_seed0.pt
"""
from __future__ import annotations

import argparse
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


# Stratford treatment cluster (per scope §4.2 — grids around Stratford station)
# Center grid is L035_029 (E02006996, 2011-parent E02000726). We pick a 3x3
# block centred there + 2 high-employment Newham grids.
STRATFORD_GRIDS = [
    "L035_029",          # Stratford center (Olympic Park / Westfield)
    "L036_029",          # Stratford East (high emp)
    "L035_028",          # Stratford South
    "L035_030",          # Stratford North
    "L036_028",          # Newham
    "L036_030",          # Newham
    "L034_029",          # Tower Hamlets adjacent
    "L034_030",          # Hackney adjacent
    "L037_029",          # Newham E
]


def cpc_signed(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-aware CPC variant: 2 * sum(min(|a|, |b|)) / (sum(|a|) + sum(|b|))
    weighted by sign agreement."""
    if a.sum() + b.sum() == 0:
        return 0.0
    return float(2.0 * np.minimum(np.abs(a), np.abs(b)).sum() / (np.abs(a).sum() + np.abs(b).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "phase_b_v6_2011_seed0.pt"))
    args = parser.parse_args()

    print("[retrodict] loading 2011 ckpt + data ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]; metrics = ckpt["metrics"]
    print(f"[retrodict] 2011 ckpt: train_CPC={metrics['train_cpc']:.4f} "
          f"val_CPC={metrics['val_cpc']:.4f} beta_base={metrics['beta_base']:+.4f} "
          f"psi={metrics['psi']:+.4f}")

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
    strat_idx = [g2idx[g] for g in STRATFORD_GRIDS if g in g2idx]
    print(f"[retrodict] Stratford treatment cluster: {len(strat_idx)} grids")
    for g in STRATFORD_GRIDS:
        print(f"  {g}  (index {g2idx.get(g, '?')})")

    # 2024 features (post-intervention) for the Stratford grids
    cache_2024 = np.load(ROOT / "data" / "processed" / "demo_cache.npz",
                          allow_pickle=True)
    static_2024 = torch.tensor(cache_2024["static_features"], dtype=torch.float32)

    # Build intervened X: 2011 features everywhere, replaced at Stratford with 2024
    X_intervened = data["static"].clone()
    for i in strat_idx:
        X_intervened[i] = static_2024[i]    # full 22-dim swap

    # Frozen baseline norm stats so the intervention isn't washed by re-norming
    norm_stats = trainer.gnn.compute_norm_stats(
        data["static"].unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )

    F_pred_2011 = predict_OD_with_X(trainer, data["static"], norm_stats=norm_stats).sum(0).cpu().numpy()
    F_pred_under_int = predict_OD_with_X(trainer, X_intervened, norm_stats=norm_stats).sum(0).cpu().numpy()
    delta_F_pred = F_pred_under_int - F_pred_2011

    # Observed flows: 2011 + 2021 grid OD (BOTH from full Census now)
    F_2011_obs = data["F_ij"].cpu().numpy()
    F_2021_obs = np.zeros_like(F_2011_obs)
    use_full_2021 = (ROOT / "data" / "processed" / "grid_od_2021_full.csv").exists()
    od_2021_path = "grid_od_2021_full.csv" if use_full_2021 else "grid_od_2021.csv"
    od_2021 = pd.read_csv(ROOT / "data" / "processed" / od_2021_path)
    cnt_col = "count21" if "count21" in od_2021.columns else "count"
    for _, r in od_2021.iterrows():
        i = g2idx.get(r["grid_home"]); j = g2idx.get(r["grid_work"])
        if i is None or j is None:
            continue
        F_2021_obs[i, j] += r[cnt_col]
    print(f"[retrodict] F_2011_obs sum: {F_2011_obs.sum():,.0f} (full Census 2011)")
    print(f"[retrodict] F_2021_obs sum: {F_2021_obs.sum():,.0f} ({od_2021_path}; "
          f"{'full Census 2021' if use_full_2021 else '1.1% sample'})")

    # Normalise to proportions per origin for comparison
    def normalise_per_origin(F):
        rs = F.sum(axis=1, keepdims=True)
        return np.divide(F, rs, where=(rs > 0), out=np.zeros_like(F))
    P_2011_obs = normalise_per_origin(F_2011_obs)
    P_2021_obs = normalise_per_origin(F_2021_obs)
    P_pred_2011 = normalise_per_origin(F_pred_2011)
    P_pred_int = normalise_per_origin(F_pred_under_int)
    delta_P_obs = P_2021_obs - P_2011_obs
    delta_P_pred = P_pred_int - P_pred_2011

    # === Headline metrics =========================================
    # 1. Stratford inflow ratio (predicted vs observed)
    inflow_strat_pred_2011 = P_pred_2011[:, strat_idx].sum(axis=1)
    inflow_strat_pred_int = P_pred_int[:, strat_idx].sum(axis=1)
    inflow_strat_obs_2011 = P_2011_obs[:, strat_idx].sum(axis=1)
    inflow_strat_obs_2021 = P_2021_obs[:, strat_idx].sum(axis=1)
    # Aggregate over all origins (as overall share to Stratford)
    share_pred_2011 = inflow_strat_pred_2011.mean()
    share_pred_int = inflow_strat_pred_int.mean()
    share_obs_2011 = inflow_strat_obs_2011.mean()
    share_obs_2021 = inflow_strat_obs_2021.mean()
    delta_share_pred = share_pred_int - share_pred_2011
    delta_share_obs = share_obs_2021 - share_obs_2011
    inflow_ratio = delta_share_pred / max(delta_share_obs, 1e-9)
    print(f"\n[retrodict] === Stratford inflow share (mean over origins) ===")
    print(f"  2011 obs share : {share_obs_2011*100:>6.3f}%")
    print(f"  2021 obs share : {share_obs_2021*100:>6.3f}%   (Δ obs = {delta_share_obs*100:+.3f} pp)")
    print(f"  2011 pred share: {share_pred_2011*100:>6.3f}%")
    print(f"  pred under int : {share_pred_int*100:>6.3f}%   (Δ pred = {delta_share_pred*100:+.3f} pp)")
    print(f"  inflow_ratio (pred/obs) = {inflow_ratio:+.3f}  "
          f"(target [0.5, 2.0])")

    # 2. Sign agreement on top-K destinations by |ΔP_obs|
    dest_delta_obs = delta_P_obs.sum(axis=0)            # destination-side aggregate
    dest_delta_pred = delta_P_pred.sum(axis=0)
    K = 30
    top_obs = np.argsort(-np.abs(dest_delta_obs))[:K]
    sign_agree = np.sign(dest_delta_obs[top_obs]) == np.sign(dest_delta_pred[top_obs])
    sign_agree_pct = float(sign_agree.mean()) * 100
    print(f"\n[retrodict] === sign agreement on top-{K} dests by |ΔP_obs| ===")
    print(f"  agreement: {int(sign_agree.sum())}/{K} = {sign_agree_pct:.1f}%  "
          f"(target ≥ 60%)")

    # 3. Spearman ρ on per-grid Δinflow vector
    from scipy.stats import spearmanr
    rho_dest, p_dest = spearmanr(dest_delta_pred, dest_delta_obs)
    print(f"\n[retrodict] === Spearman ρ on per-dest Δinflow ===")
    print(f"  ρ = {rho_dest:+.4f}  (p = {p_dest:.2e})  (target ≥ 0.4)")

    # 4. Cluster-bound flow Spearman: among origins commuting to Stratford in 2021,
    # rank-correlate predicted vs observed Δshare-to-Stratford
    inflow_strat_origin_obs = inflow_strat_obs_2021 - inflow_strat_obs_2011
    inflow_strat_origin_pred = inflow_strat_pred_int - inflow_strat_pred_2011
    nonzero_orig = (inflow_strat_obs_2021 > 0) | (inflow_strat_obs_2011 > 0)
    rho_origin, p_origin = spearmanr(
        inflow_strat_origin_pred[nonzero_orig],
        inflow_strat_origin_obs[nonzero_orig],
    )
    print(f"\n[retrodict] === Spearman ρ on Δshare-to-Stratford by origin ===")
    print(f"  origins with non-zero Stratford traffic: {int(nonzero_orig.sum())}")
    print(f"  ρ = {rho_origin:+.4f}  (p = {p_origin:.2e})  (target ≥ 0.4)")

    # === Summary ===
    print(f"\n[retrodict] === HEADLINE summary ===")
    print(f"  inflow ratio (pred/obs, Stratford): {inflow_ratio:+.3f}  "
          f"{'PASS' if 0.5 <= inflow_ratio <= 2.0 else 'FAIL'} ([0.5, 2.0])")
    print(f"  sign agreement top-30 dests:        {sign_agree_pct:.1f}%   "
          f"{'PASS' if sign_agree_pct >= 60 else 'FAIL'} (≥60%)")
    print(f"  Spearman ρ per-dest:                 {rho_dest:+.3f}   "
          f"{'PASS' if rho_dest >= 0.4 else 'FAIL'} (≥0.4)")
    print(f"  Spearman ρ origin→Stratford:         {rho_origin:+.3f}   "
          f"{'PASS' if rho_origin >= 0.4 else 'FAIL'} (≥0.4)")

    summary = {
        "ckpt": str(args.ckpt),
        "stratford_grids": STRATFORD_GRIDS,
        "n_stratford_grids": len(strat_idx),
        "share_obs_2011": float(share_obs_2011),
        "share_obs_2021": float(share_obs_2021),
        "share_pred_2011": float(share_pred_2011),
        "share_pred_under_int": float(share_pred_int),
        "delta_share_obs": float(delta_share_obs),
        "delta_share_pred": float(delta_share_pred),
        "inflow_ratio_pred_over_obs": float(inflow_ratio),
        "sign_agree_top30_dests_pct": sign_agree_pct,
        "spearman_rho_per_dest": float(rho_dest),
        "spearman_p_per_dest": float(p_dest),
        "spearman_rho_origin_to_stratford": float(rho_origin),
        "spearman_p_origin_to_stratford": float(p_origin),
        "F_2011_obs_sum": float(F_2011_obs.sum()),
        "F_2021_obs_sum": float(F_2021_obs.sum()),
        "ckpt_metrics": ckpt["metrics"],
    }
    out_path = ROOT / "evaluation_outputs" / "paper_a" / "retrodict_stratford.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[retrodict] wrote {out_path}")


if __name__ == "__main__":
    main()
