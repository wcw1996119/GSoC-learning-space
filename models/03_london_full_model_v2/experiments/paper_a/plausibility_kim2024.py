"""D10: plausibility metric (Kim et al. 2024) — k-NN distance of the
intervened scenario inputs to the training feature distribution.

Recipe (Kim 2024 §4.2 plausibility loss, adapted to a post-hoc audit):

  For each intervened grid g, compute
    d_g = (1/k) Σ_{x ∈ kNN_train(X̃_g)} ||X̃_g − x||_2
  where X̃_g is the intervened static feature vector (z-scored). Larger d
  means the new X is farther from the training cloud → the GNN is being
  asked to extrapolate.

  Report d_g compared to the distribution of training in-sample distances
  d_train_i (same metric, X computed on baseline). The percentile of
  max(d_g) tells us how OOD the most extreme intervened grid is.

  Convention: percentile ≥ 95 → "outside training support" → flag the
  scenario in the paper.

Usage:
    python experiments/paper_a/plausibility_kim2024.py \
        --grids 907 908 909 862 863 864 951 952 953 \
        --delta 7222 --coherent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_a.train_phase_b_v6_ckpt import load_data
from experiments.paper_a.scenario_A_spatial import (
    apply_intervention_raw, apply_coherent_employment_intervention,
)


def knn_distance(X_query: np.ndarray, X_train: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean Euclidean distance from each query point to its k nearest neighbours
    in X_train. If a query is in X_train (e.g. baseline grids), self is included
    in kNN — be aware. For the OOD audit we want self-included to penalise
    points that are still close to themselves but far from anything else."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_train)
    dists, _ = nn.kneighbors(X_query)
    return dists.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", required=True)
    parser.add_argument("--feature", type=str, default="total_employment")
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("[D10] loading data ...")
    data = load_data()
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    mu = np.asarray(cache["static_mean"])
    sd = np.asarray(cache["static_std"])

    X_base = data["static"].numpy()                                  # (N, F) z-scored
    if args.coherent:
        X_int = apply_coherent_employment_intervention(
            data["static"], args.grids, args.delta, mu, sd,
        ).numpy()
    else:
        X_int = apply_intervention_raw(
            data["static"], args.grids, args.feature, args.delta, mu, sd,
        ).numpy()

    # Training distribution = baseline X over all grids
    print(f"[D10] training cloud: {X_base.shape}, k={args.k}")

    # In-sample distances (baseline → baseline)
    d_train = knn_distance(X_base, X_base, k=args.k)
    train_p50 = float(np.percentile(d_train, 50))
    train_p95 = float(np.percentile(d_train, 95))
    train_p99 = float(np.percentile(d_train, 99))
    train_max = float(d_train.max())
    print(f"[D10] training-cloud kNN distance percentiles: "
          f"p50={train_p50:.3f} p95={train_p95:.3f} p99={train_p99:.3f} max={train_max:.3f}")

    # Intervened query: only the grids that got bumped (their X_int row may differ)
    grid_indices = np.array(args.grids, dtype=int)
    d_intervened = knn_distance(X_int[grid_indices], X_base, k=args.k)
    print(f"\n[D10] intervened grids OOD scan ({len(grid_indices)} grids):")
    print(f"{'grid':>6} | {'baseline_d':>12} | {'scenario_d':>12} | {'ratio':>8} | {'pctile_in_train':>17}")
    rows = []
    for i, g in enumerate(grid_indices):
        d_base_g = float(d_train[g])
        d_scen_g = float(d_intervened[i])
        ratio = d_scen_g / max(d_base_g, 1e-9)
        # Where does d_scen sit in the training-distance distribution?
        pctile = float((d_train < d_scen_g).mean() * 100)
        flag = "  OOD!" if pctile >= 95 else ""
        print(f"{g:>6} | {d_base_g:>12.4f} | {d_scen_g:>12.4f} | {ratio:>8.2f} | {pctile:>16.2f}%{flag}")
        rows.append({"grid": int(g), "baseline_d": d_base_g, "scenario_d": d_scen_g,
                     "ratio": ratio, "pctile_in_train": pctile})

    max_pctile = max(r["pctile_in_train"] for r in rows)
    verdict = "OOD" if max_pctile >= 95 else "in-distribution"
    print(f"\n[D10] verdict: max intervened grid is at {max_pctile:.1f}-th percentile of "
          f"training kNN distance → {verdict}")
    if max_pctile >= 95:
        print(f"[D10] CAVEAT: scenario inputs reach into the tail of the training feature "
              f"distribution; GNN extrapolation may be unreliable.")
    else:
        print(f"[D10] OK: scenario is within training support — extrapolation risk LOW.")

    summary = {
        "k": args.k,
        "feature": args.feature, "delta": args.delta,
        "coherent": args.coherent,
        "n_intervened_grids": len(grid_indices),
        "training_distance_percentiles": {
            "p50": train_p50, "p95": train_p95, "p99": train_p99, "max": train_max,
        },
        "rows": rows,
        "max_pctile_in_train": max_pctile,
        "verdict": verdict,
    }
    out_path = ROOT / "evaluation_outputs" / "paper_a" / "plausibility_scenario_A.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D10] wrote {out_path}")


if __name__ == "__main__":
    main()
