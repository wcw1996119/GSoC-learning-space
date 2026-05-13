"""F1 Intuitive conceptual framework for JTG paper.

5-panel pedagogical diagram showing the actual model mechanics in geographic
language (not ML jargon):

  A. Input — London 1km grid with feature icons per cell
  B. GNN message passing — one destination receiving neighbor information
  C. Agent random utility decision — stick figure choosing among destinations
  D. Aggregate flows + BPR congestion feedback loop
  E. Counterfactual intervention — same pipeline under do(X)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                 Circle, Polygon, RegularPolygon)
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def add_arrow(ax, x1, y1, x2, y2, *, lw=1.2, color="#222", style="-|>",
              connectionstyle="arc3,rad=0", alpha=1.0, mut=10):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=mut,
                           linewidth=lw, color=color,
                           connectionstyle=connectionstyle, alpha=alpha)
    ax.add_patch(arr)


def stick_figure(ax, cx, cy, scale=0.5, color="#444", lw=1.5):
    """Draw a stick figure centred at (cx, cy)."""
    # Head
    head = Circle((cx, cy + 0.6 * scale), 0.18 * scale, ec=color, fc="white", lw=lw)
    ax.add_patch(head)
    # Body
    ax.plot([cx, cx], [cy + 0.42 * scale, cy - 0.2 * scale], color=color, lw=lw)
    # Arms
    ax.plot([cx - 0.3 * scale, cx + 0.3 * scale], [cy + 0.1 * scale, cy + 0.1 * scale],
             color=color, lw=lw)
    # Legs
    ax.plot([cx, cx - 0.25 * scale], [cy - 0.2 * scale, cy - 0.65 * scale],
             color=color, lw=lw)
    ax.plot([cx, cx + 0.25 * scale], [cy - 0.2 * scale, cy - 0.65 * scale],
             color=color, lw=lw)


def building_icon(ax, cx, cy, scale=0.5, color="#264", windows=2):
    """Tiny building / office icon."""
    w = 0.5 * scale; h = 0.7 * scale
    rect = Rectangle((cx - w / 2, cy - h / 2), w, h, fc=color, ec="black", lw=0.6)
    ax.add_patch(rect)
    # Windows
    win_size = 0.08 * scale
    for r in range(2):
        for c in range(windows):
            wx = cx - w / 2 + (c + 1) * w / (windows + 1) - win_size / 2
            wy = cy - h / 2 + (r + 1) * h / 3 - win_size / 2
            ax.add_patch(Rectangle((wx, wy), win_size, win_size, fc="white", ec="none"))


def grid_cell(ax, cx, cy, size=0.7, fc="#f6f6f6", ec="#888", highlight=False, lw=0.5):
    if highlight:
        fc = "#ffe9b3"; ec = "#c87f00"; lw = 1.5
    rect = Rectangle((cx - size / 2, cy - size / 2), size, size, fc=fc, ec=ec, lw=lw)
    ax.add_patch(rect)


def main():
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.2)

    # ===================================================================
    # Panel A — Input: London 1km grid with node features
    # ===================================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(a) Input: 1 km grid of London cells\nwith per-cell features",
                  fontsize=11, fontweight="bold", loc="left")

    # 4x4 mini London grid
    cell_centres = [(1 + c * 0.85, 1 + r * 0.85) for r in range(4) for c in range(4)]
    target_idx = 5  # one cell highlighted = "j"
    for k, (cx, cy) in enumerate(cell_centres):
        grid_cell(ax, cx, cy, size=0.78, highlight=(k == target_idx))
    cx_t, cy_t = cell_centres[target_idx]
    ax.text(cx_t, cy_t + 0.25, "j", fontsize=11, fontweight="bold", ha="center",
            color="#7b3f00")

    # Feature panel right of grid
    ax.text(5.0, 5.4, "Features per cell j:", fontsize=9.5, fontweight="bold")
    feature_rows = [
        ("[B]", "$E_j$  workplace employment"),
        ("[P]", "$P_j$  resident population"),
        ("[S]", "POI counts (8 categories)"),
        ("[T]", "subway station count"),
        ("[$]", "$z_w$ destination wage"),
    ]
    for i, (icon, label) in enumerate(feature_rows):
        ax.text(4.85, 4.95 - i * 0.45, icon, fontsize=8.5, ha="center",
                family="monospace", fontweight="bold")
        ax.text(5.10, 4.95 - i * 0.45, label, fontsize=8.5, va="center")

    # ===================================================================
    # Panel B — GNN message passing
    # ===================================================================
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(b) GNN demand prior: each cell j combines\nits own + 10 nearest neighbours' features",
                  fontsize=11, fontweight="bold", loc="left")

    # Centre node + 6 neighbours arranged radially (simplified for visual)
    cj = (3.0, 3.0)
    Circle_target = Circle(cj, 0.30, fc="#ffd573", ec="#c87f00", lw=2)
    ax.add_patch(Circle_target)
    ax.text(cj[0], cj[1], "j", fontsize=12, fontweight="bold", ha="center", va="center")

    n_neigh = 6
    radii = 1.6
    for i in range(n_neigh):
        ang = 2 * np.pi * i / n_neigh + np.pi / 6
        nx = cj[0] + radii * np.cos(ang)
        ny = cj[1] + radii * np.sin(ang)
        ax.add_patch(Circle((nx, ny), 0.22, fc="#cde3f3", ec="#446", lw=1))
        ax.text(nx, ny, f"$u_{i+1}$", fontsize=8, ha="center", va="center")
        # Edge from neighbour to j with width labelled by t (simplified)
        add_arrow(ax, nx, ny, cj[0], cj[1], lw=1.3, color="#446",
                  style="-|>", mut=8)
        # Edge label = travel time
        mx = (nx + cj[0]) / 2 + 0.1
        my = (ny + cj[1]) / 2 + 0.05
        ax.text(mx, my, f"$t_{{u_{i+1} \\to j}}$", fontsize=6.5, color="#446")

    ax.text(0.2, 1.0,
            r"$V_j = \mathrm{GraphSAGE}_\theta\left(X_j, \{X_u : u \in \mathcal{N}_k(j)\}\right)$",
            fontsize=10, color="#a04",
            bbox=dict(facecolor="#fff", edgecolor="#a04", boxstyle="round,pad=0.3"))
    ax.text(0.2, 0.45,
            "= 'attractiveness score' for cell j\n(learned from observed OD)",
            fontsize=8.5, fontstyle="italic", color="#444")

    # ===================================================================
    # Panel C — Agent decision (random utility)
    # ===================================================================
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(c) Agent decision: which workplace?",
                  fontsize=11, fontweight="bold", loc="left")

    # Stick figure (agent at home) on left
    cx_home = 0.9
    cy_home = 3.0
    stick_figure(ax, cx_home, cy_home, scale=0.9, color="#222")
    ax.text(cx_home, cy_home - 1.0, "Agent n at home grid $i$", fontsize=8.5,
            ha="center", fontweight="bold")
    ax.text(cx_home, cy_home - 1.4, f"income tier, occupation",
            fontsize=7.5, ha="center", color="#444")

    # 5 destination buildings of varying attractiveness (and distance)
    dest_pos = [(3.4, 5.0), (4.5, 4.0), (5.0, 2.7), (4.0, 1.6), (2.8, 0.9)]
    dest_V = [0.9, 0.6, 0.3, 0.4, 0.2]  # attractiveness illustrative
    dest_t = [38, 22, 15, 28, 18]      # travel time (min)
    for k, (dx, dy) in enumerate(dest_pos):
        # Building icon
        building_icon(ax, dx, dy, scale=0.9, color="#3a5", windows=2)
        ax.text(dx + 0.6, dy + 0.05, f"$j_{k+1}$", fontsize=8, fontweight="bold")
        ax.text(dx + 0.6, dy - 0.25, f"$V_{{j_{k+1}}}$={dest_V[k]:.1f}", fontsize=6.5)
        ax.text(dx + 0.6, dy - 0.45, f"$t={dest_t[k]}$ min", fontsize=6.5)

    # Compute illustrative softmax probabilities (for line widths)
    beta_t = -0.07; alpha = 1.0
    util = [alpha * v + beta_t * t for v, t in zip(dest_V, dest_t)]
    util_max = max(util)
    expu = [np.exp(u - util_max) for u in util]
    Z = sum(expu)
    probs = [e / Z for e in expu]

    for (dx, dy), p in zip(dest_pos, probs):
        lw = 0.5 + p * 5
        alpha_a = 0.4 + p * 0.6
        add_arrow(ax, cx_home + 0.4, cy_home, dx - 0.3, dy, lw=lw,
                  color="#a04", alpha=alpha_a, style="-|>", mut=10)

    # Decision rule callout
    ax.text(0.2, 0.45,
            r"$U_{ij} = \alpha\,V_j + \beta_t\,t_{ij} + \gamma \log d_{ij} + \delta\,\mathrm{OccMatch}_{ij}$"
            "\n"
            r"$P_n(j) = \frac{\exp U_{ij}}{\sum_{j'} \exp U_{ij'}}$",
            fontsize=9, color="#a04",
            bbox=dict(facecolor="#fff", edgecolor="#a04", boxstyle="round,pad=0.25"))

    # ===================================================================
    # Panel D — Aggregate + BPR equilibrium loop
    # ===================================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(d) Aggregate flow → BPR congestion → re-decide",
                  fontsize=11, fontweight="bold", loc="left")

    # 4 agents on left
    for k in range(4):
        stick_figure(ax, 0.7, 4.5 - k * 1.0, scale=0.55, color="#444", lw=1)
    ax.text(0.7, 0.4, "10 000 agents", fontsize=8.5, fontweight="bold", ha="center")

    # Destination cluster with employment cap
    cx_d = 4.5; cy_d = 4.5
    building_icon(ax, cx_d, cy_d, scale=1.5, color="#3a5", windows=3)
    ax.text(cx_d + 0.95, cy_d + 0.5, "destination $j$\nwith capacity $C_j$",
            fontsize=8.5, fontweight="bold")
    ax.text(cx_d + 0.95, cy_d - 0.25, f"inflow $V_j = \\sum_i F_{{ij}}$",
            fontsize=8)

    # Many arrows in
    for k in range(4):
        add_arrow(ax, 1.0, 4.5 - k * 1.0, cx_d - 0.4, cy_d - 0.2,
                   lw=1.1, color="#666", alpha=0.55, style="-|>", mut=7,
                   connectionstyle="arc3,rad=0.05")

    # BPR formula
    ax.text(2.7, 1.7,
            r"$t_{ij}^{\rm new} = t_{ij}^{0} \cdot \left(1 + 0.15\,(V_j/C_j)^4\right)$",
            fontsize=9.5, color="#a04",
            bbox=dict(facecolor="#fff", edgecolor="#a04", boxstyle="round,pad=0.25"))
    ax.text(2.7, 0.95,
            r"More inflow $\Rightarrow$ longer travel time $\Rightarrow$ agents re-decide",
            fontsize=8.5, fontstyle="italic", color="#222")

    # Loop arrow
    add_arrow(ax, 4.5, 1.5, 1.6, 3.0, color="#a04", lw=1.7,
              connectionstyle="arc3,rad=0.4", style="-|>", mut=12)
    ax.text(2.6, 2.6, "feedback\n(MSA fixed-point)", fontsize=8, color="#a04",
            ha="center", fontstyle="italic")

    # ===================================================================
    # Panel E — Counterfactual intervention
    # ===================================================================
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(e) Counterfactual: change $X_S$, re-run pipeline",
                  fontsize=11, fontweight="bold", loc="left")

    # Spatial intervention: OOC cluster, baseline → +65k jobs
    ax.text(0.2, 5.6, "Spatial:  do$(X_{OOC,emp} \\,{+=}\\, 65 000)$",
            fontsize=10, fontweight="bold")
    # Baseline
    ax.text(0.2, 5.05, "baseline $E_{OOC}$", fontsize=8)
    building_icon(ax, 1.4, 4.7, scale=0.7, color="#3a5", windows=2)
    # arrow
    add_arrow(ax, 2.0, 4.7, 2.6, 4.7, lw=1.5, color="#a04", style="-|>", mut=10)
    # Scenario
    ax.text(2.6, 5.05, "scenario $E_{OOC} + 65\\rm k$", fontsize=8)
    building_icon(ax, 4.0, 4.7, scale=1.05, color="#3a5", windows=3)

    # Temporal intervention: peak * 0.5, off-peak * 1.5
    ax.text(0.2, 3.5, "Temporal:  do$(\\pi_t)$, peak ↓ × 0.5, off-peak ↑ × 1.5",
            fontsize=10, fontweight="bold")
    # mini hour profile
    hours = np.arange(24)
    base_demand = np.array([0.05, 0.05, 0.1, 0.25, 1.0, 3.0, 6.75, 7.5, 3.0,
                            1.25, 1.0, 1.25, 1.5, 1.5, 2.75, 5.25, 6.5, 3.75,
                            1.5, 0.75, 0.5, 0.4, 0.3, 0.2]) / 8
    scen_demand = base_demand.copy()
    for h in (7, 8, 9):
        scen_demand[h] *= 0.5
    for h in (10, 11, 12, 13, 14, 15):
        scen_demand[h] *= 1.5

    bar_y0 = 1.5
    bar_h_max = 1.4
    bar_w = 0.18
    for h in range(24):
        ax.bar(0.4 + h * 0.21, base_demand[h] * 1.6, bottom=bar_y0,
               width=bar_w, color="#888", alpha=0.4)
        ax.bar(0.4 + h * 0.21, scen_demand[h] * 1.6, bottom=bar_y0,
               width=bar_w * 0.7, color="#3a5", alpha=0.85)
    ax.text(0.4, bar_y0 - 0.25, "h0", fontsize=6)
    ax.text(0.4 + 12 * 0.21, bar_y0 - 0.25, "h12", fontsize=6)
    ax.text(0.4 + 23 * 0.21, bar_y0 - 0.25, "h23", fontsize=6)
    ax.text(0.4 + 23 * 0.21 + 0.4, bar_y0 + 0.3,
            "grey = baseline\ngreen = scenario", fontsize=7)

    # Pipeline output arrow
    add_arrow(ax, 5.5, 5.2, 5.5, 0.9, lw=2, color="#a04", style="-|>", mut=12)
    ax.text(5.7, 3.5, "(d)\nrun\nagain", fontsize=8.5, color="#a04",
            fontweight="bold")

    # Outputs at bottom
    ax.text(0.2, 0.5,
            "Output: $\\Delta F_{ij}$, $\\Delta A_i$ accessibility, "
            "$\\Delta\\mathrm{Gini}$ on $A_i$",
            fontsize=9, fontweight="bold", color="#143",
            bbox=dict(facecolor="#d6e9c6", edgecolor="#143",
                       boxstyle="round,pad=0.25"))

    # ===================================================================
    # Panel F — Mixed-logit Monte Carlo comparison
    # ===================================================================
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(f) Mixed-logit Monte Carlo reveals Jensen bias",
                  fontsize=11, fontweight="bold", loc="left")

    # Deterministic forward: one β
    ax.text(0.2, 5.4, "Deterministic forward: one $\\hat{\\beta}_t$ for all",
            fontsize=9, fontweight="bold")
    stick_figure(ax, 1.4, 4.5, scale=0.55, color="#888")
    ax.text(2.0, 4.55, r"all use $\hat{\beta}_t = -0.058$", fontsize=8)
    add_arrow(ax, 1.7, 4.5, 4.5, 4.5, color="#888", lw=1.5)
    ax.text(4.7, 4.55, "softmax", fontsize=8, color="#888")

    # Mixed-logit MC: per-agent β_n
    ax.text(0.2, 3.5, "Mixed-logit MC: per-agent $\\beta_n$ from WebTAG",
            fontsize=9, fontweight="bold")
    for k, (col, beta_label) in enumerate([("#a04", "$\\beta_n=-0.04$ (low VOT)"),
                                            ("#048", "$\\beta_n=-0.06$ (mid)"),
                                            ("#080", "$\\beta_n=-0.10$ (high VOT)")]):
        stick_figure(ax, 0.7 + k * 0.4, 2.8 - k * 0.6, scale=0.5, color=col)
        ax.text(2.4, 2.95 - k * 0.6, beta_label, fontsize=7.5, color=col)

    add_arrow(ax, 2.0, 2.5, 4.5, 1.5, color="#048", lw=1.4)
    ax.text(2.7, 1.8, "per-agent softmax\n→ aggregate", fontsize=7.5,
            color="#048", fontstyle="italic")

    # Bias outputs
    ax.text(0.2, 0.85,
            "Jensen bias = deterministic $-$ MC\n"
            "  CBD intervention: $+2.1\\%$ (small)\n"
            "  Greenfield intervention: $-16.7\\%$ (deterministic UNDER-predicts)",
            fontsize=8, color="#143",
            bbox=dict(facecolor="#fff2cc", edgecolor="#a07c00",
                       boxstyle="round,pad=0.3"))

    # ===================================================================
    # Title bar
    # ===================================================================
    fig.suptitle("Conceptual framework: GraphSAGE-parameterised deep choice model "
                  "with mixed-logit Monte Carlo policy simulation",
                  fontsize=13, fontweight="bold", y=0.99)

    # Save
    out_png = OUT_DIR / "F0_intuitive_framework.png"
    out_svg = OUT_DIR / "F0_intuitive_framework.svg"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
