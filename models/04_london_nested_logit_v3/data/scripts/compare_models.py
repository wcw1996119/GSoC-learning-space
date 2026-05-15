"""Compare v3 model variants across 3 seeds.

Reads JSON files for: v3b (additive), v3c (multiplicative Cervero), v3d (Wang residual)
and prints comparison tables of CPC + key parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_seeds(prefix: str, n: int = 3, root: Path = Path("evaluation_outputs/paper_a")) -> list[dict]:
    out = []
    for sd in range(n):
        path = root / f"{prefix}_seed{sd}.json"
        if path.exists():
            with open(path) as f:
                out.append(json.load(f))
        else:
            print(f"!! missing: {path}")
    return out


def stat(results, key, sub=None):
    vals = []
    for r in results:
        v = r["final_diagnostic"][key] if sub is None else r[key]
        if isinstance(v, list):
            v = float(np.mean(v))
        vals.append(float(v) if v is not None else float("nan"))
    arr = np.array(vals)
    return arr.mean(), arr.std(), vals


def per_mode_stat(results, key, mode_idx):
    vals = [r["final_diagnostic"][key][mode_idx] for r in results]
    arr = np.array(vals)
    return arr.mean(), arr.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", required=True,
                    help="result prefixes like cervero_shen_agent v3d_iter1_no_reg")
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()

    labels = args.labels or args.prefixes
    runs = [(label, load_seeds(p)) for label, p in zip(labels, args.prefixes)]

    print(f"\n=== CPC + key scalars across variants ===")
    print(f"{'metric':<22s}" + "".join(f"{l[:18]:<20s}" for l, _ in runs))
    metrics = [
        ("CPC", "final_cpc", True),  # top-level field
        ("alpha_wage", "alpha_wage", False),
        ("gamma_M", "gamma_M", False),
        ("nu_D", "nu_D", False),
        ("delta_match", "delta_match", False),
        ("beta_t_mean", "beta_t_mean", False),
        ("beta_t_slope_mean", "beta_t_slope_mean", False),
        ("lambda_b_mean", "lambda_b_mean", False),
        ("blend (convex)", "blend", False),
        ("gnn_residual_scale", "gnn_residual_scale", False),
        ("delta_measured (Wang)", "delta_measured", False),
        ("rum_rms", "rum_rms", False),
        ("nn_rms", "nn_rms", False),
    ]
    for name, key, top in metrics:
        cells = []
        for label, R in runs:
            if not R:
                cells.append("(no data)")
                continue
            try:
                if top:
                    vals = [r[key] for r in R]
                else:
                    vals = [r["final_diagnostic"].get(key) for r in R]
                vals = [v for v in vals if v is not None]
                if not vals:
                    cells.append("(n/a)")
                else:
                    arr = np.array(vals, dtype=float)
                    cells.append(f"{arr.mean():+.4f} ±{arr.std():.4f}")
            except Exception as e:
                cells.append(f"(err)")
        print(f"  {name:<20s}" + "".join(f"{c:<20s}" for c in cells))

    print(f"\n=== Per-mode θ (car / transit / walk, mean across seeds) ===")
    for theta_key in ("theta_inc_per_mode", "theta_kids_per_mode", "theta_cars_per_mode", "asc_per_mode"):
        print(f"\n  {theta_key}:")
        for label, R in runs:
            if not R:
                continue
            arr = np.array([r["final_diagnostic"][theta_key] for r in R])
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            print(f"    {label:<25s}  [{means[0]:+.4f}±{stds[0]:.4f}, {means[1]:+.4f}±{stds[1]:.4f}, {means[2]:+.4f}±{stds[2]:.4f}]")


if __name__ == "__main__":
    main()
