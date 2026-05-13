"""Agent heterogeneity dispatch icon for Figure 1 v6.

Shows that 'agent heterogeneity' isn't just an aggregate parameter table —
it's per-agent dispatch where 3 example agent types use DIFFERENT
parameters to compute DIFFERENT utilities and make DIFFERENT choices.

  Agent 1 (low income, walks)  → β_walk, κ_low  → softmax over close j
  Agent 2 (mid income, transit) → β_transit, κ_mid → broader softmax
  Agent 3 (high income, car)   → β_car, κ_high  → wide reach softmax

Visually: 3 stick-figure agents (left), each with a parameter tag,
arrows to a 'utility computation' middle column, then to softmax bars
on the right showing different choice distributions.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "methodology" / "figures" / "arch"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


def main():
    fig, ax = plt.subplots(figsize=(4.5, 2.6))

    # 3 example agents — each with mode + tier + colour
    agents = [
        {"name": "Agent₁",  "mode": "walk",    "tier": "low",
         "beta": -0.05, "kappa": 1.00,
         "y": 2.4, "color": "#10b981"},     # green
        {"name": "Agent₂",  "mode": "transit", "tier": "mid",
         "beta": -0.10, "kappa": 1.97,
         "y": 1.4, "color": "#f59e0b"},     # amber
        {"name": "Agent₃",  "mode": "car",     "tier": "high",
         "beta": -0.19, "kappa": 1.97,
         "y": 0.4, "color": "#3b82f6"},     # blue
    ]

    # ─── left column: 3 agents (stick figure + tag) ───────────────────────
    for a in agents:
        ax.add_patch(Circle((0.30, a["y"] + 0.18), 0.10, color=a["color"], zorder=3))
        ax.plot([0.30, 0.30], [a["y"] + 0.08, a["y"] - 0.20], color=a["color"], lw=1.6)
        ax.plot([0.18, 0.42], [a["y"] - 0.05, a["y"] - 0.05], color=a["color"], lw=1.4)
        ax.plot([0.30, 0.20], [a["y"] - 0.20, a["y"] - 0.40], color=a["color"], lw=1.4)
        ax.plot([0.30, 0.40], [a["y"] - 0.20, a["y"] - 0.40], color=a["color"], lw=1.4)
        # text labels (tighter to stick fig, no overlap with params)
        ax.text(0.55, a["y"] + 0.20, a["name"], color=a["color"], fontweight="bold", fontsize=8.5, va="center")
        ax.text(0.55, a["y"] - 0.02, f"{a['mode']} · {a['tier']}", fontsize=7, va="center", color="#475569")

    # ─── middle column: personalised parameters (pills, no overlap) ──────
    ax.text(2.6, 3.10, "personalised\nparameters",
             ha="center", fontsize=7.5, fontweight="bold", color="#334155", style="italic")
    for a in agents:
        ax.add_patch(FancyBboxPatch((2.05, a["y"] - 0.22), 1.10, 0.50,
                                       boxstyle="round,pad=0.01,rounding_size=0.05",
                                       facecolor="white", edgecolor=a["color"], lw=0.9))
        ax.text(2.60, a["y"] + 0.10,
                 fr"$\beta_{{{a['mode'][:2]}}} = {a['beta']:+.2f}$",
                 fontsize=7.5, ha="center", color=a["color"])
        ax.text(2.60, a["y"] - 0.12,
                 fr"$\kappa_{{{a['tier'][:1]}}} = {a['kappa']:.2f}$",
                 fontsize=7.5, ha="center", color=a["color"])
        # arrow agent → params (start AFTER text)
        arr = FancyArrowPatch((1.55, a["y"]), (2.05, a["y"]),
                                arrowstyle="-|>", mutation_scale=7,
                                color="#94a3b8", lw=0.8)
        ax.add_patch(arr)

    # ─── right column: each agent's softmax over destinations ─────────────
    ax.text(5.4, 3.10, "softmax over destinations\n(Gumbel-max → choice $j^*$)",
             ha="center", fontsize=7.5, fontweight="bold", color="#334155", style="italic")
    np.random.seed(7)
    n_dests = 10
    for ai, a in enumerate(agents):
        if a["mode"] == "walk":
            logits = -0.4 * np.arange(n_dests)
        elif a["mode"] == "transit":
            logits = -0.15 * np.arange(n_dests) + 0.3 * np.random.rand(n_dests)
        else:
            logits = -0.05 * np.arange(n_dests) + 0.4 * np.random.rand(n_dests)
        p = np.exp(logits) / np.exp(logits).sum()
        chosen = int(np.argmax(logits + np.random.gumbel(0, 1, n_dests)))
        for k in range(n_dests):
            x0 = 4.4 + k * 0.18
            h = p[k] * 1.4
            ax.add_patch(plt.Rectangle((x0, a["y"] - 0.30), 0.15, h,
                                          facecolor=a["color"] if k == chosen else "#cbd5e1",
                                          edgecolor="none"))
        x_star = 4.4 + chosen * 0.18 + 0.075
        h_star = p[chosen] * 1.4 + (a["y"] - 0.30) + 0.04
        ax.text(x_star, h_star, "★", ha="center", fontsize=10, color=a["color"])
        arr = FancyArrowPatch((3.20, a["y"]), (4.35, a["y"]),
                                arrowstyle="-|>", mutation_scale=7,
                                color="#94a3b8", lw=0.8)
        ax.add_patch(arr)

    # ─── aggregation: thick arrow from RIGHT side spanning all 3 agents ──
    # vertical brace
    ax.plot([6.40, 6.40], [agents[0]["y"]-0.20, agents[2]["y"]-0.20],
             color="#E07A5F", lw=2.2, solid_capstyle="round")
    # dashes from each agent's bar end to the brace
    for a in agents:
        ax.plot([6.20, 6.40], [a["y"], a["y"]], color="#E07A5F", lw=1.0, alpha=0.6)
    # arrow from brace center to "aggregate flow" output
    arr = FancyArrowPatch((6.40, agents[1]["y"]), (7.25, agents[1]["y"]),
                            arrowstyle="-|>", mutation_scale=11,
                            color="#E07A5F", lw=2.4)
    ax.add_patch(arr)
    ax.text(7.30, agents[1]["y"] + 0.30, r"$\Sigma$ aggregate",
             ha="left", fontsize=8.5, fontweight="bold", color="#E07A5F")
    ax.text(7.30, agents[1]["y"] + 0.05,
             r"$\to \widehat{F}_{ij,t}$",
             ha="left", fontsize=8.5, color="#E07A5F")

    ax.set_xlim(-0.3, 9.0)
    ax.set_ylim(-0.6, 3.4)
    ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    fig.savefig(OUT / "agent_heterogeneity.png")
    plt.close(fig)
    print(f"wrote {OUT / 'agent_heterogeneity.png'}")


if __name__ == "__main__":
    main()
