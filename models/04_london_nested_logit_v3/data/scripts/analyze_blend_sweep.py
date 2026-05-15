"""Analyze blend_max sweep results — replicate Wang TB-ResNet Figure 5 for London."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    sweep_root = Path("evaluation_outputs/paper_a")
    blend_vals = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

    rows = []
    for b in blend_vals:
        tag = f"{int(b*100):03d}"
        path = sweep_root / f"blend_sweep_{tag}.json"
        if not path.exists():
            print(f"!! missing: {path}")
            continue
        with open(path) as f:
            r = json.load(f)
        d = r["final_diagnostic"]
        rows.append({
            "blend_max": b,
            "actual_blend": d["blend"],
            "CPC": r["final_cpc"],
            "alpha_wage": d["alpha_wage"],
            "gamma_M": d["gamma_M"],
            "nu_D": d["nu_D"],
            "delta_match": d["delta_match"],
            "beta_t_mean": d["beta_t_mean"],
            "epochs": r["epochs_actual"],
        })

    if not rows:
        print("No data loaded.")
        return

    print(f"\n=== Wang Fig 5 replication: blend_max sweep on London 2019 OD ===\n")
    print(f"{'blend_max':>10s} {'actual_δ':>10s} {'CPC':>8s} {'γ_M':>8s} {'δ_match':>10s} {'α_wage':>8s} {'ν_D':>8s} {'β_t,0':>8s} {'epochs':>7s}")
    print(f"{'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for r in rows:
        print(f"{r['blend_max']:>10.2f} {r['actual_blend']:>10.4f} {r['CPC']:>8.4f} {r['gamma_M']:>+8.4f} "
              f"{r['delta_match']:>+10.4f} {r['alpha_wage']:>+8.4f} {r['nu_D']:>+8.4f} "
              f"{r['beta_t_mean']:>+8.4f} {r['epochs']:>7d}")

    # Find optimum δ (max CPC)
    best = max(rows, key=lambda r: r["CPC"])
    print(f"\n=== Optimum ===")
    print(f"  Best blend_max:  {best['blend_max']:.2f}  →  actual δ converged to {best['actual_blend']:.4f}")
    print(f"  Best CPC:        {best['CPC']:.4f}")
    print(f"  Wang's Singapore MNL-ResNet optimum: δ ≈ 0.008")
    print(f"  Our London optimum: δ ≈ {best['actual_blend']:.3f} → {best['actual_blend']/0.008:.0f}× larger")
    print(f"  Interpretation: London commute data needs more DNN help than Wang's MNL on Singapore travel")


if __name__ == "__main__":
    main()
