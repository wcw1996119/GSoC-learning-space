"""D4 Scenario B: temporal intervention do(X_jt: peak ×0.5, off ×1.5).

"Flexible-work" scenario simulating commuters spreading away from the
morning peak. Operationalised on the trained T=24 hourly model
(phase_b_v6_hourly_*.pt) by intervening on the hourly congestion-ratio
feature: peak hours (defined 7-10 AM) get ×0.5 (less congested under
flex work); shoulder hours (10-15) get ×1.5 (more demand spread there).

Pipeline:
  1. Load T=24 ckpt + hourly features.
  2. Forward baseline → F_pred_t (24, N, N).
  3. Modify hourly features at peak/off-peak slots → X_intervened_t.
  4. Forward intervention → F_pred_t_scen.
  5. Per-hour delta + downstream metrics:
     - Peak-hour total flow change
     - Peak-hour congestion (BPR layer applied per-hour)
     - Total accessibility weighted by hour
     - Inequality (Gini over A_i hourly)

Output:
  evaluation_outputs/paper_a/scenario_B_temporal.json
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

from models_lib.inverse_rum import (
    InverseRUMTrainer, StructuralGNN,
    apply_bpr, hansen_accessibility, gini, palma_ratio, atkinson,
)
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index
from experiments.paper_a.train_phase_b_v6_hourly import load_data_hourly


PEAK_HOURS = (7, 8, 9)              # AM peak
SHOULDER_HOURS = (10, 11, 12, 13, 14, 15)   # off-peak day
PEAK_MULT = 0.5                     # flex work reduces peak congestion to 50%
SHOULDER_MULT = 1.5                 # extra demand spreads to off-peak


def build_intervened_features(X_t: torch.Tensor,
                              cong_col_idx: int,
                              peak_hours, shoulder_hours,
                              peak_mult, shoulder_mult) -> torch.Tensor:
    """Modify the congestion_ratio_z column at peak/off-peak hours.
    Note: column is z-scored, so multiplying = scaling deviation from mean."""
    X_int = X_t.clone()
    for h in peak_hours:
        X_int[h, :, cong_col_idx] = X_int[h, :, cong_col_idx] * peak_mult
    for h in shoulder_hours:
        X_int[h, :, cong_col_idx] = X_int[h, :, cong_col_idx] * shoulder_mult
    return X_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "phase_b_v6_hourly_constrained_seed0.pt"))
    parser.add_argument("--peak_mult", type=float, default=PEAK_MULT)
    parser.add_argument("--shoulder_mult", type=float, default=SHOULDER_MULT)
    args = parser.parse_args()

    print(f"[scenB] loading T=24 ckpt + data ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]; metrics = ckpt["metrics"]
    print(f"[scenB] ckpt: cpc_daily_val={metrics['cpc_daily_val']:.4f} "
          f"beta_mean={metrics['beta_base_mean']:+.4f}")

    data = load_data_hourly()
    edge_index = build_edge_index(data["t_ij_t"], k=10)

    util_net = StructuralGNN(in_features=cfg["in_features"], hidden=cfg["hidden"],
                             out=cfg["out"], depth=cfg["depth"])
    trainer = InverseRUMTrainer(
        grid_features=data["static_t"], edge_index=edge_index,
        observed_OD=data["F_ij_t"], t_ij_t=data["t_ij_t"], log_d_ij=data["log_d"],
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

    # Identify which feature column is congestion_ratio_z
    feat_names_path = ROOT / "data" / "processed" / "hourly_node_features.npz"
    fnpz = np.load(feat_names_path, allow_pickle=True)
    feat_names = list(fnpz["feat_names"])
    cong_col = feat_names.index("congestion_ratio_z")
    print(f"[scenB] congestion_ratio_z is column {cong_col} of {len(feat_names)} features")

    # Frozen baseline norm stats
    norm_stats = trainer.gnn.compute_norm_stats(data["static_t"], trainer.edge_index)

    # Baseline prediction (model P(j|i,t) × observed row sums per hour)
    print("[scenB] forward baseline ...")
    with torch.no_grad():
        F_pred_base = trainer.predict_OD(norm_stats=norm_stats).cpu().numpy()      # (24, N, N)

    # Two interventions chained:
    # (1) Feature-level: scale congestion_ratio at peak/shoulder → re-predict P(j|i,t)
    # (2) Demand-level: scale per-hour origin demand by same multiplier (with mass
    #     conservation per origin), since flex work spreads commuters across hours.
    X_int = build_intervened_features(
        data["static_t"], cong_col_idx=cong_col,
        peak_hours=PEAK_HOURS, shoulder_hours=SHOULDER_HOURS,
        peak_mult=args.peak_mult, shoulder_mult=args.shoulder_mult,
    )
    # Get P(j|i,t) under intervention features
    X_orig = trainer.X
    trainer.X = X_int.to(trainer.device)
    try:
        with torch.no_grad():
            # Manually compute P(j|i,t) from model (skip the row-sum scaling)
            V_jt = trainer._forward_V(trainer.X, norm_stats=norm_stats)
            t_use = trainer.t_ij_t
            log_p = trainer._per_hour_log_p_full(V_jt, t_use, trainer.log_d_ij)
            P_jt_scen = log_p.exp().cpu().numpy()                        # (T, N, N)
    finally:
        trainer.X = X_orig

    # Demand-level shift: scale per-hour-origin row sums by multiplier;
    # then renormalise per origin so daily totals conserved.
    row_sums_base = data["F_ij_t"].sum(dim=2).cpu().numpy()              # (T, N)
    multiplier_t = np.ones(24, dtype=np.float64)
    for h in PEAK_HOURS:
        multiplier_t[h] = args.peak_mult
    for h in SHOULDER_HOURS:
        multiplier_t[h] = args.shoulder_mult
    # Per-origin renormalisation: scenario_daily_sum = baseline_daily_sum
    row_sums_unnorm = row_sums_base * multiplier_t[:, None]              # (T, N)
    daily_base = row_sums_base.sum(axis=0, keepdims=True)                # (1, N)
    daily_unnorm = row_sums_unnorm.sum(axis=0, keepdims=True)
    norm_factor = np.divide(daily_base, daily_unnorm,
                              out=np.ones_like(daily_base),
                              where=(daily_unnorm > 0))
    row_sums_scen = row_sums_unnorm * norm_factor                        # (T, N)

    # Scenario flow: P_jt_scen(j|i,t) × scenario row sums
    F_pred_int = P_jt_scen * row_sums_scen[:, :, None]                  # (T, N, N)
    delta_F_t = F_pred_int - F_pred_base                                 # (24, N, N)

    # ===== Per-hour summaries =====
    print(f"\n[scenB] === per-hour total flow ===")
    rows_hour = []
    for h in range(24):
        flow_base = F_pred_base[h].sum()
        flow_scen = F_pred_int[h].sum()
        d = flow_scen - flow_base
        d_pct = 100 * d / max(flow_base, 1e-9)
        is_peak = h in PEAK_HOURS
        is_shoulder = h in SHOULDER_HOURS
        tag = "PEAK" if is_peak else "SHLD" if is_shoulder else "    "
        print(f"  h{h:02d} {tag}: base={flow_base:>6.0f}  scen={flow_scen:>6.0f}  "
              f"Δ={d:+8.1f} ({d_pct:+5.2f}%)")
        rows_hour.append({"hour": int(h), "is_peak": bool(is_peak), "is_shoulder": bool(is_shoulder),
                           "flow_base": float(flow_base), "flow_scen": float(flow_scen),
                           "delta": float(d), "delta_pct": float(d_pct)})

    peak_base = sum(F_pred_base[h].sum() for h in PEAK_HOURS)
    peak_scen = sum(F_pred_int[h].sum() for h in PEAK_HOURS)
    shlder_base = sum(F_pred_base[h].sum() for h in SHOULDER_HOURS)
    shlder_scen = sum(F_pred_int[h].sum() for h in SHOULDER_HOURS)
    peak_pct_base = 100 * peak_base / max(F_pred_base.sum(), 1)
    peak_pct_scen = 100 * peak_scen / max(F_pred_int.sum(), 1)
    print(f"\n[scenB] === peak/off-peak shift ===")
    print(f"  peak hours total: {peak_base:>7.0f}  -> {peak_scen:>7.0f}  "
          f"({peak_pct_base:.1f}% -> {peak_pct_scen:.1f}% of day)")
    print(f"  shoulder total:   {shlder_base:>7.0f}  -> {shlder_scen:>7.0f}")

    # ===== BPR per-hour congestion =====
    # Capacity = baseline PEAK-hour inflow (so peak V/C ≈ 1, scenario shifts visible).
    # Per-grid hourly inflow at the baseline busiest hour for each grid.
    print(f"\n[scenB] === BPR per-hour congestion (capacity = baseline peak-hour inflow) ===")
    rows_bpr = []
    t0_static = data["t_ij_t"][0]                                        # free-flow car
    inflow_per_hour_per_grid = F_pred_base.sum(axis=1)                   # (24, N) inflow at each grid each hour
    cap_per_grid = torch.tensor(inflow_per_hour_per_grid.max(axis=0) + 1.0,
                                 dtype=torch.float32)
    for h in [7, 8, 9, 10, 12, 14, 17, 18]:
        flow_base_h = torch.tensor(F_pred_base[h], dtype=torch.float32)
        flow_scen_h = torch.tensor(F_pred_int[h], dtype=torch.float32)
        t_base = apply_bpr(t0_static, flow_base_h, cap_per_grid, alpha=0.15, beta=4.0,
                            mult_cap=2.5)
        t_scen = apply_bpr(t0_static, flow_scen_h, cap_per_grid, alpha=0.15, beta=4.0,
                            mult_cap=2.5)
        mean_t_base = float(t_base.mean()); mean_t_scen = float(t_scen.mean())
        d = mean_t_scen - mean_t_base
        d_pct = 100 * d / mean_t_base
        print(f"  h{h:02d}: mean t_ij base={mean_t_base:.2f}min scen={mean_t_scen:.2f}min  "
              f"Δ={d:+.3f} ({d_pct:+.2f}%)")
        rows_bpr.append({"hour": int(h), "mean_t_base": mean_t_base,
                          "mean_t_scen": mean_t_scen, "delta_min": float(d),
                          "delta_pct": float(d_pct)})

    # ===== Hour-weighted accessibility =====
    print(f"\n[scenB] === peak-hour accessibility (h=8) ===")
    feats = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    grid_ids = cache["grid_ids"].tolist()
    feats = feats.set_index("grid_id").reindex(grid_ids)
    E_j = torch.tensor(feats["total_employment"].fillna(0.0).to_numpy(), dtype=torch.float32)
    pop_w = feats["population"].fillna(0.0).to_numpy().astype(np.float64)

    beta_t_recovered = float(metrics["beta_base_mean"])
    # Use BPR-equilibrium t at hour 8
    flow_base_8 = torch.tensor(F_pred_base[8], dtype=torch.float32)
    flow_scen_8 = torch.tensor(F_pred_int[8], dtype=torch.float32)
    t_base_8 = apply_bpr(t0_static, flow_base_8, cap_per_grid, mult_cap=2.5)
    t_scen_8 = apply_bpr(t0_static, flow_scen_8, cap_per_grid, mult_cap=2.5)
    A_base = hansen_accessibility(E_j, t_base_8, beta_t_recovered).cpu().numpy()
    A_scen = hansen_accessibility(E_j, t_scen_8, beta_t_recovered).cpu().numpy()
    print(f"  baseline A_8 mean: {A_base.mean():>10,.0f}  "
          f"Gini={gini(A_base, pop_w):.4f}  Palma={palma_ratio(A_base, pop_w):.4f}")
    print(f"  scenario A_8 mean: {A_scen.mean():>10,.0f}  "
          f"Gini={gini(A_scen, pop_w):.4f}  Palma={palma_ratio(A_scen, pop_w):.4f}")
    print(f"  Δ accessibility:   {(A_scen.mean() - A_base.mean()):>+10,.0f}  "
          f"({100 * (A_scen.mean() - A_base.mean()) / A_base.mean():+.2f}%)")
    print(f"  ΔGini: {(gini(A_scen, pop_w) - gini(A_base, pop_w)):+.4f}")

    summary = {
        "ckpt": str(args.ckpt),
        "intervention": {
            "peak_hours": list(PEAK_HOURS),
            "shoulder_hours": list(SHOULDER_HOURS),
            "peak_mult": args.peak_mult,
            "shoulder_mult": args.shoulder_mult,
            "intervened_feature": "congestion_ratio_z",
        },
        "per_hour_flow": rows_hour,
        "per_hour_bpr": rows_bpr,
        "peak_share_baseline_pct": float(peak_pct_base),
        "peak_share_scenario_pct": float(peak_pct_scen),
        "accessibility_h8": {
            "A_base_mean": float(A_base.mean()),
            "A_scen_mean": float(A_scen.mean()),
            "delta_abs": float(A_scen.mean() - A_base.mean()),
            "delta_pct": float(100 * (A_scen.mean() - A_base.mean()) / A_base.mean()),
            "gini_base": float(gini(A_base, pop_w)),
            "gini_scen": float(gini(A_scen, pop_w)),
            "palma_base": float(palma_ratio(A_base, pop_w)),
            "palma_scen": float(palma_ratio(A_scen, pop_w)),
        },
    }
    out = ROOT / "evaluation_outputs" / "paper_a" / "scenario_B_temporal.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[scenB] wrote {out}")


if __name__ == "__main__":
    main()
