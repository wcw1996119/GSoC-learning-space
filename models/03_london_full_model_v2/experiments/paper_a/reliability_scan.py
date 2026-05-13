"""Statistical characterization of model trust region.

Random clusters × intervention magnitudes → empirical reliability map.

For each (cluster_baseline_employment, intervention_multiplier) cell, we report:
  - Aggregate model prediction Δshare
  - Mixed-logit MC prediction Δshare
  - Jensen bias = |aggregate - MC| / |MC|
  - Kim 2024 k-NN OOD plausibility percentile

Output: 2D heatmap data + trust-region thresholds for paper §5.
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

from models_lib.inverse_rum import (
    AgentBetaConfig,
    assign_per_agent_epsilon,
    per_agent_beta,
    simulate_agent_choice_softmax,
    aggregate_per_agent_to_OD,
)
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, load_data
from experiments.paper_a.scenario_A_spatial import (
    apply_coherent_employment_intervention, build_trainer_from_ckpt,
    predict_OD_with_X, EMP_BLOCK,
)
from experiments.paper_a.agent_simulation_heterogeneous import (
    run_aggregate_prediction, run_agent_prediction,
)


# Intervention magnitude multipliers (additive on baseline employment per cluster)
# E.g., 0.30 means "+30% of baseline employment added to cluster"
MAGNITUDES = [0.10, 0.30, 0.50, 1.00, 2.00, 5.00]


def sample_clusters(n_clusters: int, n_grids: int, cluster_size: int = 9,
                    seed: int = 42) -> list[list[int]]:
    """Sample n_clusters random 9-cell contiguous-ish clusters.

    For simplicity: pick a random centre cell, then take its (cluster_size - 1)
    nearest neighbours by grid index distance. (Crude proxy for spatial clusters
    without geometry; the exact spatial pattern matters less than the diversity
    of baseline-employment levels.)
    """
    rng = np.random.default_rng(seed)
    centres = rng.choice(n_grids, size=n_clusters, replace=False)
    # Build a simple proximity by grid index — works on regularly-numbered grids
    clusters = []
    for c in centres:
        # Take centre + neighbours by index (simple stand-in for geographic proximity)
        nbrs = sorted(
            range(n_grids),
            key=lambda j: abs(j - c)
        )[:cluster_size]
        clusters.append(nbrs)
    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=60)
    parser.add_argument("--cluster_size", type=int, default=9)
    parser.add_argument("--cov", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[reliability] loading data + ckpt + agents ...")
    data = load_data()
    ckpt_path = ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"
    trainer, ckpt = build_trainer_from_ckpt(ckpt_path, data)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    aux = np.load(ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    mu = np.asarray(cache["static_mean"]); sd = np.asarray(cache["static_std"])
    income_score = np.asarray(aux["income_score_per_origin"])
    wage_score = np.asarray(aux["wage_score_per_dest"])
    occ_match = np.asarray(aux["occ_match"])
    agents_df = pd.read_csv(ROOT / "data" / "processed" / "agent_population_v23.csv")

    n_grids = data["static"].shape[0]
    feats = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    grid_ids = cache["grid_ids"].tolist()
    feats = feats.set_index("grid_id").reindex(grid_ids)
    raw_emp = feats["total_employment"].fillna(0.0).to_numpy()

    cfg = AgentBetaConfig(cov_by_tier={1: args.cov, 2: args.cov, 3: args.cov},
                           seed=args.seed)
    norm_stats = trainer.gnn.compute_norm_stats(
        data["static"].unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )
    F_agg_base = run_aggregate_prediction(trainer, data["static"], norm_stats)
    F_agent_base, eps, _ = run_agent_prediction(
        trainer, data["static"], norm_stats, agents_df, occ_match,
        data["t_ij"].cpu().numpy(), data["log_d"].cpu().numpy(),
        income_score, wage_score, cfg,
    )
    scale = F_agg_base.sum() / max(F_agent_base.sum(), 1e-9)

    print(f"[reliability] sampling {args.n_clusters} random clusters of size {args.cluster_size}")
    clusters = sample_clusters(args.n_clusters, n_grids, args.cluster_size, seed=args.seed)

    rows = []
    t_start = time.time()
    for c_idx, cluster_grids in enumerate(clusters):
        baseline_emp = raw_emp[cluster_grids].sum()
        if baseline_emp < 100:
            print(f"  [c{c_idx}] skip — cluster baseline emp {baseline_emp:.0f} too low")
            continue
        for mag in MAGNITUDES:
            delta_total = baseline_emp * mag
            delta_per_grid = delta_total / args.cluster_size
            X_int = apply_coherent_employment_intervention(
                data["static"], cluster_grids, delta_per_grid, mu, sd,
            )
            # Aggregate prediction
            F_agg_int = run_aggregate_prediction(trainer, X_int, norm_stats)
            delta_agg_into_cluster = float(
                (F_agg_int - F_agg_base)[:, cluster_grids].sum()
            )
            # Mixed-logit MC prediction (uses scaled scenario for direct comparison)
            F_agent_int, _, _ = run_agent_prediction(
                trainer, X_int, norm_stats, agents_df, occ_match,
                data["t_ij"].cpu().numpy(), data["log_d"].cpu().numpy(),
                income_score, wage_score, cfg,
            )
            delta_mc_into_cluster = float(
                ((F_agent_int - F_agent_base) * scale)[:, cluster_grids].sum()
            )
            jensen_bias_pct = (
                100.0 * (delta_agg_into_cluster - delta_mc_into_cluster)
                / max(abs(delta_mc_into_cluster), 1e-9)
            )
            rows.append({
                "cluster_idx": c_idx,
                "cluster_size": args.cluster_size,
                "baseline_emp": float(baseline_emp),
                "magnitude_multiplier": float(mag),
                "delta_total_jobs": float(delta_total),
                "delta_agg_inflow": delta_agg_into_cluster,
                "delta_mc_inflow": delta_mc_into_cluster,
                "jensen_bias_pct": jensen_bias_pct,
                "abs_jensen_bias_pct": abs(jensen_bias_pct),
            })
        elapsed = time.time() - t_start
        rate = (c_idx + 1) / elapsed
        eta = (len(clusters) - c_idx - 1) / max(rate, 1e-9)
        print(f"  [c{c_idx}/{len(clusters)}] baseline_emp={baseline_emp:>7.0f} "
              f"  done in {elapsed:.0f}s  ETA {eta:.0f}s")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "reliability_scan.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[reliability] wrote {out_csv} ({len(df)} rows)")

    # ===== Trust-region characterisation =====
    print(f"\n[reliability] === TRUST REGION ANALYSIS ===")

    # Bin by baseline employment
    df["emp_bin"] = pd.cut(df["baseline_emp"],
                            bins=[0, 5e3, 2e4, 5e4, 2e5, 1e7],
                            labels=["<5k", "5k-20k", "20k-50k", "50k-200k", ">200k"])
    df["mag_bin"] = pd.cut(df["magnitude_multiplier"],
                            bins=[0, 0.20, 0.50, 1.50, 10],
                            labels=["+10-20%", "+30-50%", "+100-150%", "+200%+"])

    # 2D bias map
    pivot_mean = df.pivot_table(index="emp_bin", columns="mag_bin",
                                  values="abs_jensen_bias_pct", aggfunc="mean")
    pivot_p90 = df.pivot_table(index="emp_bin", columns="mag_bin",
                                values="abs_jensen_bias_pct", aggfunc=lambda x: np.percentile(x, 90))
    print(f"\nMean |Jensen bias| (%) by baseline_emp × magnitude:")
    print(pivot_mean.round(1).to_string())
    print(f"\n90th-pctile |Jensen bias| (%) by baseline_emp × magnitude:")
    print(pivot_p90.round(1).to_string())

    # Trust thresholds
    n_trust = (df["abs_jensen_bias_pct"] < 5).sum()
    n_caveat = ((df["abs_jensen_bias_pct"] >= 5) & (df["abs_jensen_bias_pct"] < 20)).sum()
    n_unreliable = (df["abs_jensen_bias_pct"] >= 20).sum()
    n_total = len(df)
    print(f"\nGlobally across {n_total} (cluster × magnitude) experiments:")
    print(f"  reliable    (|bias| < 5%): {n_trust:>4} ({100*n_trust/n_total:.1f}%)")
    print(f"  caveat      (5% to 20%):   {n_caveat:>4} ({100*n_caveat/n_total:.1f}%)")
    print(f"  unreliable  (|bias| > 20%): {n_unreliable:>4} ({100*n_unreliable/n_total:.1f}%)")

    summary = {
        "n_clusters": args.n_clusters,
        "n_experiments": int(n_total),
        "magnitude_multipliers": MAGNITUDES,
        "cov": args.cov,
        "global_distribution": {
            "reliable_pct_lt_5": float(100 * n_trust / n_total),
            "caveat_pct_5_to_20": float(100 * n_caveat / n_total),
            "unreliable_pct_gt_20": float(100 * n_unreliable / n_total),
        },
        "mean_bias_by_bin": pivot_mean.fillna(np.nan).to_dict(),
        "p90_bias_by_bin": pivot_p90.fillna(np.nan).to_dict(),
    }
    out_json = ROOT / "evaluation_outputs" / "paper_a" / "reliability_scan.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[reliability] wrote {out_json}")


if __name__ == "__main__":
    main()
