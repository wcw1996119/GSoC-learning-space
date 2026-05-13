"""Scenario A — Old Oak Common +65k jobs (DUAL_HET / DS-SIGNN forward).

Replaces the older scenario_A_spatial.py + scenario_A_accessibility.py
pair with a DS-SIGNN-aware version. BPR is applied to the car mode only;
transit and walk travel times are not affected by congestion.

Output:
  evaluation_outputs/paper_a/scenario_A_dual_het.json
  evaluation_outputs/paper_a/scenario_A_dual_het.csv
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
    hansen_accessibility, all_indices, bpr_multiplier,
)

from experiments.paper_a.scenario_A_spatial import (
    apply_coherent_employment_intervention, EMP_BLOCK,
)
from experiments.paper_a.dual_het_scenario_helpers import (
    load_dual_het_ckpt, build_scenario_inputs, predict_flow,
)


# Old Oak Common 9-grid cluster (centred on grid 908, see paper_A_draft_v2 §7.1)
OOC_GRIDS = [907, 908, 909, 862, 863, 864, 951, 952, 953]
DELTA_PER_GRID = 7222.0                # +65k total / 9 grids
CAPACITY_SCALE = 2.0                   # baseline V/C ≈ 0.5
MULT_CAP = 2.5
BPR_MAX_ITER = 60
BPR_TOL = 5e-3


def get_raw_employment(static_z: torch.Tensor, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Recover raw total_employment per grid from z-scored log1p column (col 8)."""
    f_idx = 8                                                # total_employment
    z = static_z[:, f_idx].numpy()
    log_raw = z * sd[f_idx] + mu[f_idx]
    return np.expm1(log_raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "dual_het_seed0.pt"))
    parser.add_argument("--grids", type=int, nargs="+", default=OOC_GRIDS)
    parser.add_argument("--delta_per_grid", type=float, default=DELTA_PER_GRID)
    parser.add_argument("--capacity_scale", type=float, default=CAPACITY_SCALE)
    parser.add_argument("--mult_cap", type=float, default=MULT_CAP)
    args = parser.parse_args()

    print("[scenA] loading ckpt + data ...")
    enc, head, ckpt = load_dual_het_ckpt(Path(args.ckpt))
    d = build_scenario_inputs(walk_threshold_km=ckpt["config"]["walk_threshold_km"])
    mu, sd = d["cache_static_mu"], d["cache_static_sd"]
    grid_ids = d["grid_ids"]; pop_w = d["pop_w"]
    N, T = d["N"], d["T"]

    # Baseline V norm stats — frozen across baseline + scenario forwards
    print("[scenA] baseline V norm stats ...")
    norm_stats = enc.compute_norm_stats(d["X_static"], d["X_dynamic"], d["edge_index"])

    # Recovered behavioural params (for diagnostics + Hansen β_car)
    bm = head.beta_t_per_mode.mean(dim=0).tolist()
    beta_car, beta_transit, beta_walk = bm[0], bm[1], bm[2]
    gamma = float(head.gamma.item())
    print(f"[scenA] β_car={beta_car:+.4f}  β_transit={beta_transit:+.4f}  "
          f"β_walk={beta_walk:+.4f}  γ={gamma:+.4f}")

    t_per_mode = {"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]}

    # ---- Baseline forward ----------------------------------------------------
    print("[scenA] forward baseline ...")
    F_base = predict_flow(
        enc, head, d["X_static"], d["X_dynamic"], d["edge_index"],
        t_per_mode, d["log_d"], d["log_pi_mk_pair"], d["F_ij_t"],
        occ_match=d["occ_match"], norm_stats=norm_stats,
    ).sum(0).numpy()                                                     # (N, N) daily

    # Capacity proxy = baseline daily inflow per j × scale
    baseline_inflow = F_base.sum(axis=0) + 1.0                           # (N,)
    capacity_j = torch.tensor(baseline_inflow * args.capacity_scale, dtype=torch.float32)

    # ---- Apply OOC coherent intervention to X_static -------------------------
    print(f"[scenA] applying coherent OOC intervention: {len(args.grids)} grids "
          f"× +{args.delta_per_grid:.0f} jobs each (total +{len(args.grids)*args.delta_per_grid:.0f}) ...")
    X_int = apply_coherent_employment_intervention(
        d["X_static"], args.grids, args.delta_per_grid, mu, sd,
    )
    E_baseline = get_raw_employment(d["X_static"], mu, sd)
    E_scenario = get_raw_employment(X_int, mu, sd)
    E_added_total = float(E_scenario.sum() - E_baseline.sum())
    print(f"  Σ E_baseline={E_baseline.sum():,.0f}  Σ E_scenario={E_scenario.sum():,.0f}  "
          f"(+{E_added_total:,.0f})")

    # ---- Free-flow scenario forward -----------------------------------------
    print("[scenA] forward scenario (free-flow) ...")
    F_scen_ff = predict_flow(
        enc, head, X_int, d["X_dynamic"], d["edge_index"],
        t_per_mode, d["log_d"], d["log_pi_mk_pair"], d["F_ij_t"],
        occ_match=d["occ_match"], norm_stats=norm_stats,
    ).sum(0).numpy()
    delta_ff = F_scen_ff - F_base
    inflow_into_OOC_ff = float(delta_ff[:, args.grids].sum())
    print(f"  Δinflow into OOC (free-flow): {inflow_into_OOC_ff:+.1f}")

    # ---- BPR equilibrium under scenario X (car only) -------------------------
    print("[scenA] BPR UE under scenario X (car mode only) ...")
    flow_avg = torch.zeros((N, N))
    history = []
    for k in range(1, BPR_MAX_ITER + 1):
        # apply BPR mult to car based on current flow_avg
        inflow_j_t = flow_avg.sum(dim=0)                                  # (N,)
        mult_j = bpr_multiplier(inflow_j_t, capacity_j, alpha=0.15,
                                  beta=4.0, mult_cap=args.mult_cap)        # (N,)
        t_car_cong = d["t_car"] * mult_j.view(1, -1)                       # (N, N)
        t_per_mode_k = {"car": t_car_cong, "transit": d["t_transit"], "walk": d["t_walk"]}
        F_new = predict_flow(
            enc, head, X_int, d["X_dynamic"], d["edge_index"],
            t_per_mode_k, d["log_d"], d["log_pi_mk_pair"], d["F_ij_t"],
            occ_match=d["occ_match"], norm_stats=norm_stats,
        ).sum(0)                                                           # (N, N)
        weight = 1.0 / k
        flow_avg_new = (1.0 - weight) * flow_avg + weight * F_new
        gap = (flow_avg_new - flow_avg).abs().max().item() / max(flow_avg_new.abs().mean().item(), 1e-6)
        history.append(gap)
        flow_avg = flow_avg_new
        print(f"  iter {k:02d}  rel_gap={gap:.4e}  Δinflow→OOC={float((flow_avg - torch.from_numpy(F_base))[:, args.grids].sum()):+.1f}")
        if gap < BPR_TOL and k > 1:
            break
    F_scen_bpr = flow_avg.numpy()
    delta_bpr = F_scen_bpr - F_base
    inflow_into_OOC_bpr = float(delta_bpr[:, args.grids].sum())
    final_t_car_cong = d["t_car"] * mult_j.view(1, -1)                     # (N, N)
    print(f"  Δinflow into OOC (BPR eq): {inflow_into_OOC_bpr:+.1f} "
          f"(damping = {(1 - inflow_into_OOC_bpr/inflow_into_OOC_ff)*100:.1f}%)")

    # ---- Hansen accessibility (3 cases) --------------------------------------
    print("[scenA] Hansen accessibility (β_car, t_car) ...")
    E_b_t = torch.tensor(E_baseline, dtype=torch.float32)
    E_s_t = torch.tensor(E_scenario, dtype=torch.float32)
    A_baseline_freeflow = hansen_accessibility(E_b_t, d["t_car"], beta_car).numpy()
    A_scenario_freeflow = hansen_accessibility(E_s_t, d["t_car"], beta_car).numpy()
    A_scenario_bpr = hansen_accessibility(E_s_t, final_t_car_cong, beta_car).numpy()

    cases = [
        ("baseline (E_b, t0)", A_baseline_freeflow),
        ("scenario freeflow (E_s, t0)", A_scenario_freeflow),
        ("scenario BPR (E_s, t_eq)", A_scenario_bpr),
    ]
    rows = []
    for name, A in cases:
        idx_w = pop_w > 0
        u_idx = all_indices(A)
        p_idx = all_indices(A[idx_w], pop_w[idx_w])
        rows.append({
            "case": name,
            "A_mean": float(A.mean()),
            "A_p10": float(np.percentile(A, 10)),
            "A_p50": float(np.percentile(A, 50)),
            "A_p90": float(np.percentile(A, 90)),
            "unit_gini": u_idx["gini"], "unit_palma": u_idx["palma"],
            "unit_atkinson_e0_5": u_idx["atkinson"],
            "pop_mean_A": p_idx["mean"], "pop_gini": p_idx["gini"],
            "pop_palma": p_idx["palma"], "pop_atkinson_e0_5": p_idx["atkinson"],
        })
    df_out = pd.DataFrame(rows)
    csv_path = ROOT / "evaluation_outputs" / "paper_a" / "scenario_A_dual_het.csv"
    df_out.to_csv(csv_path, index=False)

    base_row = rows[0]
    for r in rows[1:]:
        d_meanA = (r["pop_mean_A"] - base_row["pop_mean_A"]) / base_row["pop_mean_A"] * 100
        d_gini = (r["pop_gini"] - base_row["pop_gini"]) / base_row["pop_gini"] * 100
        d_palma = (r["pop_palma"] - base_row["pop_palma"]) / base_row["pop_palma"] * 100
        d_atk = (r["pop_atkinson_e0_5"] - base_row["pop_atkinson_e0_5"]) / base_row["pop_atkinson_e0_5"] * 100
        print(f"  vs base | {r['case']:<30s} d_meanA={d_meanA:+.2f}% "
              f"d_Gini={d_gini:+.2f}% d_Palma={d_palma:+.2f}% d_Atk={d_atk:+.2f}%")

    # ---- Top destination winners/losers --------------------------------------
    dest_delta_ff = delta_ff.sum(axis=0)
    dest_delta_bpr = delta_bpr.sum(axis=0)
    top10_ff = np.argsort(-dest_delta_ff)[:10]
    bot10_ff = np.argsort(dest_delta_ff)[:10]
    top10_bpr = np.argsort(-dest_delta_bpr)[:10]
    bot10_bpr = np.argsort(dest_delta_bpr)[:10]

    summary = {
        "ckpt": str(args.ckpt),
        "ckpt_metrics": ckpt["metrics"],
        "intervention": {
            "name": "Old Oak Common cluster +65k jobs",
            "grids": list(args.grids),
            "delta_per_grid": float(args.delta_per_grid),
            "delta_total_jobs": float(len(args.grids) * args.delta_per_grid),
            "E_added_total": E_added_total,
            "coherent": True, "EMP_BLOCK": EMP_BLOCK,
        },
        "bpr": {
            "alpha": 0.15, "beta": 4.0,
            "mult_cap": args.mult_cap, "capacity_scale": args.capacity_scale,
            "max_iter": BPR_MAX_ITER, "tol": BPR_TOL,
            "n_iter": int(len(history)), "final_rel_gap": float(history[-1] if history else 0.0),
            "history": [float(g) for g in history],
        },
        "recovered_params": {
            "beta_car": beta_car, "beta_transit": beta_transit, "beta_walk": beta_walk,
            "gamma": gamma,
            "kappa": ckpt["metrics"]["kappa"],
            "delta_per_tier": ckpt["metrics"]["delta_per_tier"],
        },
        "delta_inflow_into_OOC_freeflow": inflow_into_OOC_ff,
        "delta_inflow_into_OOC_bpr": inflow_into_OOC_bpr,
        "bpr_damping_pct": float((1 - inflow_into_OOC_bpr / max(inflow_into_OOC_ff, 1e-6)) * 100),
        "accessibility": {
            "rows": rows,
            "delta_meanA_freeflow_pct": (rows[1]["pop_mean_A"] - rows[0]["pop_mean_A"]) / rows[0]["pop_mean_A"] * 100,
            "delta_meanA_bpr_pct": (rows[2]["pop_mean_A"] - rows[0]["pop_mean_A"]) / rows[0]["pop_mean_A"] * 100,
            "delta_gini_bpr_pct": (rows[2]["pop_gini"] - rows[0]["pop_gini"]) / rows[0]["pop_gini"] * 100,
            "delta_palma_bpr_pct": (rows[2]["pop_palma"] - rows[0]["pop_palma"]) / rows[0]["pop_palma"] * 100,
            "delta_atkinson_bpr_pct": (rows[2]["pop_atkinson_e0_5"] - rows[0]["pop_atkinson_e0_5"]) / rows[0]["pop_atkinson_e0_5"] * 100,
        },
        "top10_dest_gain_freeflow": [
            {"grid_idx": int(i), "delta_inflow": float(dest_delta_ff[i])} for i in top10_ff
        ],
        "top10_dest_loss_freeflow": [
            {"grid_idx": int(i), "delta_inflow": float(dest_delta_ff[i])} for i in bot10_ff
        ],
        "top10_dest_gain_bpr": [
            {"grid_idx": int(i), "delta_inflow": float(dest_delta_bpr[i])} for i in top10_bpr
        ],
        "top10_dest_loss_bpr": [
            {"grid_idx": int(i), "delta_inflow": float(dest_delta_bpr[i])} for i in bot10_bpr
        ],
    }
    json_path = ROOT / "evaluation_outputs" / "paper_a" / "scenario_A_dual_het.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[scenA] wrote {json_path}")
    print(f"[scenA] wrote {csv_path}")


if __name__ == "__main__":
    main()
