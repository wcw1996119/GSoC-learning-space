"""F9 Figure: Jensen bias by intervention type + CoV sensitivity."""
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
    cov_sens = json.load(open(OUT_DIR / "het_cov_sensitivity.json"))
    rows = cov_sens["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: CoV sensitivity scan
    ax = axes[0]
    covs = [r["cov"] for r in rows]
    bias_pct = [r["jensen_bias_pct"] for r in rows]
    ax.plot(covs, bias_pct, "o-", color="#1f77b4", lw=2, markersize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0.40, color="grey", linestyle="--", lw=1, label="WebTAG/Wardman σ=0.40")
    ax.fill_between([0, 1.05], -4, 4, alpha=0.1, color="green",
                     label="|bias| < 4%")
    ax.set_xlabel("σ / mean (Coefficient of Variation on β per income tier)", fontsize=11)
    ax.set_ylabel("Jensen bias (% of agent-level prediction)", fontsize=11)
    ax.set_title("(A) CoV sensitivity at Stratford 9-cluster intervention\n"
                 "|bias| stays < 4% across full WebTAG σ range",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(linestyle=":", alpha=0.3)

    # Panel B: bias by intervention type
    ax = axes[1]
    interventions = ["Stratford\n(mature CBD)", "Random outer\n(greenfield)"]
    bias_vals = [2.1, -16.7]   # from agent_simulation_heterogeneous outputs
    colors = ["#2ca02c" if abs(b) < 5 else "#d62728" for b in bias_vals]
    bars = ax.bar(interventions, bias_vals, color=colors)
    for b, v in zip(bars, bias_vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (1 if v > 0 else -1.5),
                f"{v:+.1f}%", ha="center",
                fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Jensen bias = aggregate − agent (% of agent prediction)", fontsize=10)
    ax.set_title("(B) Bias direction depends on intervention context\n"
                 "Mature: aggregate slightly OVER; greenfield: aggregate UNDER-predicts",
                 fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_ylim(-22, 8)

    fig.suptitle("Agent-level heterogeneity reveals context-dependent Jensen bias",
                  fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "F9_jensen_bias.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
