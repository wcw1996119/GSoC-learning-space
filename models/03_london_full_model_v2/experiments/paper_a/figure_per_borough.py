"""F4 Figure: per-borough CPC bar chart for the headline ensemble.

Reads from final_paper_eval_sage_unconstrained_K50_h4_seeds5.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def main():
    p10 = OUT_DIR / "final_paper_eval_sage_unconstrained_K50_h4_seeds10.json"
    p5 = OUT_DIR / "final_paper_eval_sage_unconstrained_K50_h4_seeds5.json"
    data = json.load(open(p10 if p10.exists() else p5))
    pb = data["per_borough_cpc"]
    overall_cpc = data["bootstrap_ours"]["mean"]
    overall_lo = data["bootstrap_ours"]["ci95_lo"]
    overall_hi = data["bootstrap_ours"]["ci95_hi"]
    grav_cpc = data["paired_vs_gravity"]["gravity_ensemble_cpc_overall"]
    rad_cpc = data["paired_vs_radiation"]["radiation_ensemble_cpc_overall"]

    boroughs = [r["borough"] for r in pb]
    n_grids = [r["n_grids"] for r in pb]
    cpcs = [r["cpc"] for r in pb]

    # Sort by CPC desc
    order = sorted(range(len(pb)), key=lambda i: -cpcs[i])
    boroughs = [boroughs[i] for i in order]
    n_grids = [n_grids[i] for i in order]
    cpcs = [cpcs[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(pb))
    bars = ax.bar(xs, cpcs, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"][:len(pb)])
    for b, c, ng in zip(bars, cpcs, n_grids):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.005,
                f"{c:.3f}\nn={ng}", ha="center", va="bottom", fontsize=9)
    ax.axhline(overall_cpc, color="#1f77b4", linestyle="-", lw=1.5,
                label=f"Ours overall: {overall_cpc:.3f} (95% CI [{overall_lo:.3f}, {overall_hi:.3f}])")
    ax.axhspan(overall_lo, overall_hi, alpha=0.15, color="#1f77b4")
    ax.axhline(grav_cpc, color="grey", linestyle=":", lw=1, label=f"Gravity: {grav_cpc:.3f}")
    ax.axhline(rad_cpc, color="grey", linestyle="--", lw=1, label=f"Radiation: {rad_cpc:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(boroughs, fontsize=10)
    ax.set_ylabel("CPC", fontsize=11)
    ax.set_title("Per-borough CPC, headline 5-seed ensemble (4-borough spatial holdout)",
                  fontsize=12)
    ax.set_ylim(0, max(max(cpcs), overall_hi) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "F4_per_borough.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
