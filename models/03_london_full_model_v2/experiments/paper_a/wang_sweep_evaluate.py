"""Aggregate Wang TB-ResNet δ_max sweep results into multi-proxy trade-off table.

Reads ablation JSON outputs from a δ_max sweep:
  evaluation_outputs/paper_a/wang_sweep/blend_max_0p0/ablation_baseline_tb_resnet.json
  evaluation_outputs/paper_a/wang_sweep/blend_max_0p1/ablation_baseline_tb_resnet.json
  ...
  evaluation_outputs/paper_a/wang_sweep/blend_max_1p0/ablation_baseline_tb_resnet.json

Reports per blend_max:
  1. CPC mean ± std (3 seeds)        — fit quality
  2. Realised δ mean ± std            — where the hard cap actually bites
  3. β_car/transit/walk mean ± std    — recovered parameter stability
  4. γ mean ± std                     — distance decay stability
  5. δ_per_tier degeneracy            — income heterogeneity recovery

This is the "multi-proxy reliability" snapshot recommended after the
2026-05-14 EOD reframe — substitutes implementation-consistency reliability_scan
(which needs full MC simulation per config, ~30 min × 6) with parameter-side
stability proxies that fall straight out of the per-seed JSONs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = ROOT / "evaluation_outputs" / "paper_a" / "wang_sweep"


def load_config(blend_max_str: str) -> dict | None:
    sub = SWEEP_ROOT / f"blend_max_{blend_max_str}"
    js = sub / "ablation_baseline_tb_resnet.json"
    if not js.exists():
        # Pure-RUM case may use baseline_tb_resnet too; if file missing report None
        print(f"  WARN: {js} not found")
        return None
    return json.load(open(js))


def mean_std(values: List[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    blend_max_values = [
        ("0p0", 0.0),
        ("0p1", 0.1),
        ("0p2", 0.2),
        ("0p3", 0.3),
        ("0p5", 0.5),
        ("1p0", 1.0),
    ]

    rows = []
    for tag, blend_max in blend_max_values:
        cfg = load_config(tag)
        if cfg is None:
            continue
        results = cfg.get("results", [])
        if len(results) == 0:
            print(f"  WARN: blend_max={blend_max} has no seeds")
            continue
        cpcs = [r["cpc"] for r in results]
        deltas = [r.get("gnn_blend") for r in results]
        deltas = [d for d in deltas if d is not None]
        beta_cars = [r["beta_car"] for r in results]
        beta_transits = [r["beta_transit"] for r in results]
        beta_walks = [r["beta_walk"] for r in results]
        gammas = [r["gamma"] if not isinstance(r["gamma"], list) else float(np.mean(r["gamma"]))
                  for r in results]
        # delta_per_tier: list of (K,) per seed
        delta_tiers = [r.get("delta_per_tier") for r in results]
        delta_tiers = [d for d in delta_tiers if d is not None]
        if delta_tiers:
            delta_tier_arr = np.asarray(delta_tiers)              # (n_seeds, K)
            delta_span_per_seed = delta_tier_arr.max(axis=1) - delta_tier_arr.min(axis=1)
            tier_span_mean = float(delta_span_per_seed.mean())
        else:
            tier_span_mean = float("nan")

        rows.append({
            "blend_max": blend_max,
            "tag": tag,
            "n_seeds": len(results),
            "cpc_mean": mean_std(cpcs)[0],
            "cpc_std": mean_std(cpcs)[1],
            "delta_realized_mean": mean_std(deltas)[0] if deltas else float("nan"),
            "delta_realized_std": mean_std(deltas)[1] if deltas else float("nan"),
            "beta_car_mean": mean_std(beta_cars)[0],
            "beta_car_std": mean_std(beta_cars)[1],
            "beta_transit_mean": mean_std(beta_transits)[0],
            "beta_transit_std": mean_std(beta_transits)[1],
            "beta_walk_mean": mean_std(beta_walks)[0],
            "beta_walk_std": mean_std(beta_walks)[1],
            "gamma_mean": mean_std(gammas)[0],
            "gamma_std": mean_std(gammas)[1],
            "delta_tier_span_mean": tier_span_mean,
        })

    if not rows:
        print("No results found in", SWEEP_ROOT)
        sys.exit(1)

    print()
    print(f"{'blend_max':>10}  {'CPC':>16}  {'realized δ':>18}  "
          f"{'β_car':>17}  {'β_transit':>17}  {'γ':>17}  {'δ_tier span':>12}")
    print("-" * 130)
    for r in rows:
        cpc_str = f"{r['cpc_mean']:.4f}±{r['cpc_std']:.4f}"
        delta_str = (f"{r['delta_realized_mean']:.3f}±{r['delta_realized_std']:.3f}"
                     if not np.isnan(r['delta_realized_mean']) else "      n/a")
        bc = f"{r['beta_car_mean']:+.4f}±{r['beta_car_std']:.4f}"
        bt = f"{r['beta_transit_mean']:+.4f}±{r['beta_transit_std']:.4f}"
        gm = f"{r['gamma_mean']:+.4f}±{r['gamma_std']:.4f}"
        ts = f"{r['delta_tier_span_mean']:.3f}" if not np.isnan(r['delta_tier_span_mean']) else "  n/a"
        print(f"{r['blend_max']:>10.2f}  {cpc_str:>16}  {delta_str:>18}  "
              f"{bc:>17}  {bt:>17}  {gm:>17}  {ts:>12}")

    # Save aggregated JSON for paper Table generation
    out_json = SWEEP_ROOT / "wang_sweep_summary.json"
    json.dump(
        {"rows": rows,
         "notes": "Wang TB-ResNet δ_max sweep, multi-proxy stability table."},
        open(out_json, "w"),
        indent=2,
    )
    print(f"\nwrote {out_json}")

    # Identify best trade-off
    # Score: high CPC + low β/γ std + reasonable δ recovery
    print("\n=== trade-off ranking ===")
    print("Lower std on (β_car, γ) and high CPC ⇒ better stability + fit. "
          "Low realised δ ⇒ more theory-dominant.")


if __name__ == "__main__":
    main()
