"""D5+D6: Job accessibility A_i and population-weighted inequality, baseline
vs scenario A (spatial intervention), with and without BPR equilibrium.

Reuses the scenario_A_spatial harness for the GNN forward + BPR fixed-point,
but here the **output of interest** is per-grid accessibility, not per-OD flow.

Pipeline (3 cases):
  (1) baseline               : E_baseline + t_freeflow
  (2) scenario freeflow      : E_scenario + t_freeflow
  (3) scenario BPR           : E_scenario + t_equilibrium (BPR UE)

For each, reports A_i summary (mean, p10, p50, p90) and 3 inequality indices
(Gini, Palma, Atkinson(0.5)) under both unit and population weights.

Run:
    python experiments/paper_a/scenario_A_accessibility.py \
        --grids 907 908 909 862 863 864 951 952 953 \
        --delta 7222 --coherent --capacity_scale 2.0
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
    apply_bpr, solve_user_equilibrium,
    hansen_accessibility, all_indices,
)
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, load_data
from experiments.paper_a.scenario_A_spatial import (
    apply_intervention_raw, apply_coherent_employment_intervention,
    build_trainer_from_ckpt, predict_OD_with_X, EMP_BLOCK,
)


def get_raw_employment(static_z: torch.Tensor, mu: np.ndarray, sd: np.ndarray,
                       feat_idx: int) -> np.ndarray:
    """Recover raw total_employment per grid from z-scored log1p column."""
    z = static_z[:, feat_idx].numpy()
    log_raw = z * sd[feat_idx] + mu[feat_idx]
    return np.expm1(log_raw)


def get_intervened_E_j(static_z: torch.Tensor, grid_indices: list, delta: float,
                       mu: np.ndarray, sd: np.ndarray, coherent: bool) -> np.ndarray:
    """Return raw total_employment after intervention."""
    if coherent:
        X_int = apply_coherent_employment_intervention(static_z, grid_indices, delta, mu, sd)
    else:
        X_int = apply_intervention_raw(static_z, grid_indices, "total_employment", delta, mu, sd)
    return get_raw_employment(X_int, mu, sd, feat_idx=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"))
    parser.add_argument("--grids", type=int, nargs="+", required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--capacity_scale", type=float, default=2.0)
    parser.add_argument("--bpr_alpha", type=float, default=0.15)
    parser.add_argument("--bpr_beta", type=float, default=4.0)
    parser.add_argument("--mult_cap", type=float, default=2.5)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    print("[D5+D6] loading data + ckpt ...")
    data = load_data()
    trainer, ckpt = build_trainer_from_ckpt(Path(args.ckpt), data)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    mu = np.asarray(cache["static_mean"])
    sd = np.asarray(cache["static_std"])
    grid_ids = cache["grid_ids"].tolist()

    # Raw baseline E_j (total_employment column index 8)
    E_baseline = get_raw_employment(data["static"], mu, sd, feat_idx=8)
    E_scenario = get_intervened_E_j(
        data["static"], args.grids, args.delta, mu, sd, args.coherent,
    )
    E_added = E_scenario.sum() - E_baseline.sum()
    print(f"[D5+D6] Σ E_baseline = {E_baseline.sum():,.0f} jobs")
    print(f"[D5+D6] Σ E_scenario = {E_scenario.sum():,.0f} jobs (+{E_added:,.0f})")

    # Population weights — use raw 'population' from grid_static_features.csv
    df = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    df = df.set_index("grid_id").reindex(grid_ids)
    pop_w = df["population"].fillna(0.0).to_numpy().astype(np.float64)
    print(f"[D5+D6] Σ population weights = {pop_w.sum():,.0f}")

    # Recovered β_t (signed, negative)
    beta_t = float(trainer.rum.beta_t.mean().item())
    print(f"[D5+D6] β_t (recovered, signed) = {beta_t:+.4f}")

    # Free-flow t_ij (same in both baseline and scenario freeflow case)
    t0 = data["t_ij"]                                              # (N, N) free-flow car

    E_b_t = torch.tensor(E_baseline, dtype=torch.float32)
    E_s_t = torch.tensor(E_scenario, dtype=torch.float32)

    # === Case 1: baseline ===
    A1 = hansen_accessibility(E_b_t, t0, beta_t).cpu().numpy()
    # === Case 2: scenario freeflow ===
    A2 = hansen_accessibility(E_s_t, t0, beta_t).cpu().numpy()
    # === Case 3: scenario + BPR equilibrium ===
    # Solve BPR UE under scenario X (so flow predict uses intervened E),
    # then read off t_eq.
    X_int = (apply_coherent_employment_intervention if args.coherent
             else apply_intervention_raw)(
        data["static"], args.grids,
        args.delta if not args.coherent else args.delta,
        mu, sd,
    ) if args.coherent else apply_intervention_raw(
        data["static"], args.grids, "total_employment", args.delta, mu, sd,
    )

    # Capacity proxy: baseline inflow * scale (need baseline OD first)
    F_baseline = predict_OD_with_X(trainer, data["static"]).sum(0).cpu().numpy()
    baseline_inflow = F_baseline.sum(axis=0) + 1.0
    capacity_j = torch.tensor(baseline_inflow * args.capacity_scale, dtype=torch.float32)

    # Freeze baseline norm stats for scenario forward
    norm_stats = trainer.gnn.compute_norm_stats(
        data["static"].unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )

    def predict_flow(t_ij: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            F_pred = predict_OD_with_X(trainer, X_int, t_ij_override=t_ij,
                                       norm_stats=norm_stats)
        return F_pred.sum(0)

    print("[D5+D6] solving BPR UE under scenario X ...")
    ue = solve_user_equilibrium(
        t0_ij=t0, capacity_j=capacity_j, predict_flow=predict_flow,
        alpha=args.bpr_alpha, beta=args.bpr_beta, mult_cap=args.mult_cap,
        max_iter=args.max_iter, tol=args.tol,
    )
    t_eq = ue.t_ij                                                # (N, N)
    print(f"[D5+D6] UE: converged={ue.converged} n_iter={ue.n_iter} "
          f"rel_gap={ue.final_gap:.2e}")
    A3 = hansen_accessibility(E_s_t, t_eq, beta_t).cpu().numpy()

    # === Inequality indices ===
    cases = [
        ("baseline (E_b, t0)", A1),
        ("scenario freeflow (E_s, t0)", A2),
        ("scenario BPR (E_s, t_eq)", A3),
    ]
    rows = []
    for name, A in cases:
        # Drop zero-population grids for population-weighted inequality
        # (otherwise they have no weight but still count toward unit Gini)
        idx_w = pop_w > 0
        u_idx = all_indices(A)                                        # unit weights
        p_idx = all_indices(A[idx_w], pop_w[idx_w])                   # population weights
        rows.append({
            "case": name,
            "A_mean": float(A.mean()),
            "A_p10": float(np.percentile(A, 10)),
            "A_p50": float(np.percentile(A, 50)),
            "A_p90": float(np.percentile(A, 90)),
            "unit_gini": u_idx["gini"],
            "unit_palma": u_idx["palma"],
            "unit_atkinson_e0_5": u_idx["atkinson"],
            "pop_mean_A": p_idx["mean"],
            "pop_gini": p_idx["gini"],
            "pop_palma": p_idx["palma"],
            "pop_atkinson_e0_5": p_idx["atkinson"],
        })
    df_out = pd.DataFrame(rows)
    out_path = ROOT / "evaluation_outputs" / "paper_a" / "scenario_A_accessibility.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[D5+D6] wrote {out_path}\n")
    # Pretty print
    pd.set_option("display.float_format", "{:,.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(df_out[["case", "A_mean", "pop_gini", "pop_palma", "pop_atkinson_e0_5"]].to_string(index=False))
    print()
    base_row = rows[0]
    for row in rows[1:]:
        d_mean = (row["pop_mean_A"] - base_row["pop_mean_A"]) / base_row["pop_mean_A"] * 100
        d_gini = (row["pop_gini"] - base_row["pop_gini"]) / base_row["pop_gini"] * 100
        d_palma = (row["pop_palma"] - base_row["pop_palma"]) / base_row["pop_palma"] * 100
        d_atk = (row["pop_atkinson_e0_5"] - base_row["pop_atkinson_e0_5"]) / base_row["pop_atkinson_e0_5"] * 100
        print(f"vs baseline | {row['case']:<35s} | "
              f"d_meanA={d_mean:+.2f}%  d_Gini={d_gini:+.2f}%  "
              f"d_Palma={d_palma:+.2f}%  d_Atk={d_atk:+.2f}%")

    # Dump JSON with side info
    summary = {
        "grids": args.grids, "delta": args.delta, "coherent": args.coherent,
        "capacity_scale": args.capacity_scale, "mult_cap": args.mult_cap,
        "beta_t_recovered": beta_t,
        "ue_converged": bool(ue.converged), "ue_n_iter": int(ue.n_iter),
        "ue_final_rel_gap": float(ue.final_gap),
        "E_added_total": float(E_added),
        "rows": rows,
    }
    json_path = ROOT / "evaluation_outputs" / "paper_a" / "scenario_A_accessibility.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[D5+D6] wrote {json_path}")


if __name__ == "__main__":
    main()
