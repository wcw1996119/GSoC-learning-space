"""Scenario B — flexible work (DUAL_HET / DS-SIGNN forward).

Two-step intervention:
  1. Feature-level: scale congestion_z column at peak/shoulder hours.
  2. Demand-level: scale per-hour origin row sums by same multiplier,
     re-normalised so daily total per origin is conserved.

Output:
  evaluation_outputs/paper_a/scenario_B_dual_het.json
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
    hansen_accessibility, gini, palma_ratio, bpr_multiplier,
)

from experiments.paper_a.dual_het_scenario_helpers import (
    load_dual_het_ckpt, build_scenario_inputs, forward_log_p,
)


PEAK_HOURS = (7, 8, 9)
SHOULDER_HOURS = (10, 11, 12, 13, 14, 15)
PEAK_MULT = 0.5
SHOULDER_MULT = 1.5
CONG_COL_NAME = "congestion_z"
INSPECT_HOURS = [7, 8, 9, 10, 12, 14, 17, 18]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "dual_het_seed0.pt"))
    parser.add_argument("--peak_mult", type=float, default=PEAK_MULT)
    parser.add_argument("--shoulder_mult", type=float, default=SHOULDER_MULT)
    parser.add_argument("--mult_cap", type=float, default=2.5)
    parser.add_argument("--capacity_scale", type=float, default=1.0)
    args = parser.parse_args()

    print("[scenB] loading ckpt + data ...")
    enc, head, ckpt = load_dual_het_ckpt(Path(args.ckpt))
    d = build_scenario_inputs(walk_threshold_km=ckpt["config"]["walk_threshold_km"])
    N, T = d["N"], d["T"]

    bm = head.beta_t_per_mode.mean(dim=0).tolist()
    beta_car, beta_transit, beta_walk = bm[0], bm[1], bm[2]
    gamma = float(head.gamma.item())
    print(f"[scenB] β_car={beta_car:+.4f}  β_transit={beta_transit:+.4f}  "
          f"β_walk={beta_walk:+.4f}  γ={gamma:+.4f}")

    # Identify congestion col index
    dyn_npz = np.load(ROOT / "data" / "processed" / "grid_dynamic_features.npz")
    feat_names = list(dyn_npz["feat_names"])
    cong_col = feat_names.index(CONG_COL_NAME)
    print(f"[scenB] {CONG_COL_NAME} = column {cong_col} of {feat_names}")

    t_per_mode = {"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]}

    # Frozen baseline V norm stats
    norm_stats = enc.compute_norm_stats(d["X_static"], d["X_dynamic"], d["edge_index"])

    # ---- Baseline forward → P(j|i,t) and F = P × row sums --------------------
    print("[scenB] forward baseline ...")
    log_p_base = forward_log_p(
        enc, head, d["X_static"], d["X_dynamic"], d["edge_index"],
        t_per_mode, d["log_d"], d["log_pi_mk_pair"],
        occ_match=d["occ_match"], norm_stats=norm_stats,
    )
    P_base = log_p_base.exp().numpy()                                    # (T, N, N)
    row_sums_base = d["F_ij_t"].sum(dim=2).numpy()                       # (T, N)
    F_pred_base = P_base * row_sums_base[:, :, None]                     # (T, N, N)

    # ---- Build intervened X_dynamic (scale congestion_z col) -----------------
    X_dyn_int = d["X_dynamic"].clone()
    for h in PEAK_HOURS:
        X_dyn_int[h, :, cong_col] *= args.peak_mult
    for h in SHOULDER_HOURS:
        X_dyn_int[h, :, cong_col] *= args.shoulder_mult

    # ---- Scenario forward (intervened X_dyn → P_jt_scen) ---------------------
    print("[scenB] forward scenario (intervened congestion z) ...")
    log_p_scen = forward_log_p(
        enc, head, d["X_static"], X_dyn_int, d["edge_index"],
        t_per_mode, d["log_d"], d["log_pi_mk_pair"],
        occ_match=d["occ_match"], norm_stats=norm_stats,
    )
    P_scen = log_p_scen.exp().numpy()                                    # (T, N, N)

    # Demand-level shift: scale per-hour origin row sums, re-normalise
    multiplier_t = np.ones(24, dtype=np.float64)
    for h in PEAK_HOURS: multiplier_t[h] = args.peak_mult
    for h in SHOULDER_HOURS: multiplier_t[h] = args.shoulder_mult
    row_sums_unnorm = row_sums_base * multiplier_t[:, None]              # (T, N)
    daily_base = row_sums_base.sum(axis=0, keepdims=True)                # (1, N)
    daily_unnorm = row_sums_unnorm.sum(axis=0, keepdims=True)
    norm_factor = np.divide(daily_base, daily_unnorm,
                              out=np.ones_like(daily_base),
                              where=(daily_unnorm > 0))
    row_sums_scen = row_sums_unnorm * norm_factor                        # (T, N)
    F_pred_scen = P_scen * row_sums_scen[:, :, None]                     # (T, N, N)

    # ---- Per-hour total flow summary ----------------------------------------
    rows_hour = []
    print(f"\n[scenB] per-hour total flow:")
    for h in range(24):
        flow_b = float(F_pred_base[h].sum())
        flow_s = float(F_pred_scen[h].sum())
        delta = flow_s - flow_b
        delta_pct = 100 * delta / max(flow_b, 1e-9)
        is_peak = h in PEAK_HOURS
        is_shoulder = h in SHOULDER_HOURS
        tag = "PEAK" if is_peak else "SHLD" if is_shoulder else "    "
        print(f"  h{h:02d} {tag}: base={flow_b:>7.0f}  scen={flow_s:>7.0f}  "
              f"Δ={delta:+8.1f} ({delta_pct:+6.2f}%)")
        rows_hour.append({"hour": h, "is_peak": bool(is_peak), "is_shoulder": bool(is_shoulder),
                           "flow_base": flow_b, "flow_scen": flow_s,
                           "delta": delta, "delta_pct": delta_pct})

    peak_share_base = sum(F_pred_base[h].sum() for h in PEAK_HOURS) / max(F_pred_base.sum(), 1) * 100
    peak_share_scen = sum(F_pred_scen[h].sum() for h in PEAK_HOURS) / max(F_pred_scen.sum(), 1) * 100
    print(f"\n[scenB] peak share: {peak_share_base:.1f}% → {peak_share_scen:.1f}%")

    # ---- Per-hour BPR (one-pass; no UE loop, just baseline-vs-scenario) ------
    # Capacity = baseline busiest-hour inflow per grid (so V/C ≈ 1 at peak)
    inflow_per_hr_per_grid = F_pred_base.sum(axis=1)                     # (T, N)
    cap_per_grid = torch.tensor(inflow_per_hr_per_grid.max(axis=0) + 1.0, dtype=torch.float32)

    print(f"\n[scenB] per-hour mean t_car under BPR (capacity = baseline peak inflow):")
    rows_bpr = []
    for h in INSPECT_HOURS:
        inflow_b = torch.tensor(F_pred_base[h].sum(axis=0), dtype=torch.float32)
        inflow_s = torch.tensor(F_pred_scen[h].sum(axis=0), dtype=torch.float32)
        mult_b = bpr_multiplier(inflow_b, cap_per_grid, alpha=0.15, beta=4.0,
                                  mult_cap=args.mult_cap)
        mult_s = bpr_multiplier(inflow_s, cap_per_grid, alpha=0.15, beta=4.0,
                                  mult_cap=args.mult_cap)
        t_b = (d["t_car"] * mult_b.view(1, -1)).mean().item()
        t_s = (d["t_car"] * mult_s.view(1, -1)).mean().item()
        delta = t_s - t_b; delta_pct = 100 * delta / max(t_b, 1e-9)
        print(f"  h{h:02d}: t_car_mean base={t_b:.2f}min  scen={t_s:.2f}min  "
              f"Δ={delta:+.3f} ({delta_pct:+.2f}%)")
        rows_bpr.append({"hour": h, "mean_t_base": t_b, "mean_t_scen": t_s,
                          "delta_min": delta, "delta_pct": delta_pct})

    # ---- Hansen accessibility @ h08 -----------------------------------------
    feats = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    feats = feats.set_index("grid_id").reindex(d["grid_ids"])
    E_j = torch.tensor(feats["total_employment"].fillna(0.0).to_numpy(), dtype=torch.float32)
    pop_w = d["pop_w"]

    inflow_b8 = torch.tensor(F_pred_base[8].sum(axis=0), dtype=torch.float32)
    inflow_s8 = torch.tensor(F_pred_scen[8].sum(axis=0), dtype=torch.float32)
    mult_b8 = bpr_multiplier(inflow_b8, cap_per_grid, alpha=0.15, beta=4.0,
                               mult_cap=args.mult_cap)
    mult_s8 = bpr_multiplier(inflow_s8, cap_per_grid, alpha=0.15, beta=4.0,
                               mult_cap=args.mult_cap)
    t_car_b8 = d["t_car"] * mult_b8.view(1, -1)
    t_car_s8 = d["t_car"] * mult_s8.view(1, -1)
    A_base = hansen_accessibility(E_j, t_car_b8, beta_car).numpy()
    A_scen = hansen_accessibility(E_j, t_car_s8, beta_car).numpy()
    delta_pct_A = 100 * (A_scen.mean() - A_base.mean()) / max(A_base.mean(), 1e-9)
    delta_gini = gini(A_scen, pop_w) - gini(A_base, pop_w)
    delta_palma = palma_ratio(A_scen, pop_w) - palma_ratio(A_base, pop_w)
    print(f"\n[scenB] h08 accessibility (β_car):")
    print(f"  baseline mean A_8: {A_base.mean():>12,.0f}  Gini={gini(A_base, pop_w):.4f}  "
          f"Palma={palma_ratio(A_base, pop_w):.4f}")
    print(f"  scenario mean A_8: {A_scen.mean():>12,.0f}  Gini={gini(A_scen, pop_w):.4f}  "
          f"Palma={palma_ratio(A_scen, pop_w):.4f}")
    print(f"  Δ mean A_8: {(A_scen.mean() - A_base.mean()):>+12,.0f}  ({delta_pct_A:+.2f}%)")
    print(f"  Δ Gini: {delta_gini:+.4f}  (negative = progressive)")
    print(f"  Δ Palma: {delta_palma:+.4f}")

    summary = {
        "ckpt": str(args.ckpt), "ckpt_metrics": ckpt["metrics"],
        "intervention": {
            "peak_hours": list(PEAK_HOURS), "shoulder_hours": list(SHOULDER_HOURS),
            "peak_mult": args.peak_mult, "shoulder_mult": args.shoulder_mult,
            "intervened_feature": CONG_COL_NAME, "mass_conserved_per_origin": True,
        },
        "recovered_params": {
            "beta_car": beta_car, "beta_transit": beta_transit, "beta_walk": beta_walk,
            "gamma": gamma,
            "kappa": ckpt["metrics"]["kappa"],
            "delta_per_tier": ckpt["metrics"]["delta_per_tier"],
        },
        "per_hour_flow": rows_hour,
        "per_hour_bpr": rows_bpr,
        "peak_share_baseline_pct": float(peak_share_base),
        "peak_share_scenario_pct": float(peak_share_scen),
        "accessibility_h8": {
            "A_base_mean": float(A_base.mean()), "A_scen_mean": float(A_scen.mean()),
            "delta_abs": float(A_scen.mean() - A_base.mean()),
            "delta_pct": float(delta_pct_A),
            "gini_base": float(gini(A_base, pop_w)),
            "gini_scen": float(gini(A_scen, pop_w)),
            "delta_gini": float(delta_gini),
            "palma_base": float(palma_ratio(A_base, pop_w)),
            "palma_scen": float(palma_ratio(A_scen, pop_w)),
            "delta_palma": float(delta_palma),
        },
    }
    out = ROOT / "evaluation_outputs" / "paper_a" / "scenario_B_dual_het.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[scenB] wrote {out}")


if __name__ == "__main__":
    main()
