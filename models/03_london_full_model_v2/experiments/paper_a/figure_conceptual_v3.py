"""F1 v3 conceptual framework — concept-level (not technical).

4-step horizontal flow with plain-English captions, no formulas.
Designed for a transport-geography reader to grasp in 5 seconds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                 Circle, Polygon, Wedge)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def big_arrow(ax, x, y, dx=1.0, color="#a04", lw=3):
    arr = FancyArrowPatch((x, y), (x + dx, y),
                           arrowstyle="-|>", mutation_scale=25,
                           linewidth=lw, color=color)
    ax.add_patch(arr)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5),
                              gridspec_kw={"wspace": 0.35})

    # Common style: each panel has a circled step number, an icon area, and a caption
    step_titles = [
        "Observe",
        "Learn",
        "Simulate",
        "Evaluate",
    ]
    step_captions = [
        "London's 2021 daily\ncommuting flows\n(1.21M workers, 1km grid)",
        "What makes a place attractive?\nHow do commuters trade off\ntime, distance, wages?",
        "Change the city — add jobs\nat one place, or spread peak\ndemand to off-peak hours.",
        "Who reaches more jobs now?\nWho reaches fewer?\nIs the policy fair?",
    ]
    panel_colors = ["#cde3f3", "#fde6e2", "#e8d8f0", "#d6e9c6"]

    # Panel 1: Observe — London grid with commute arrows
    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 6, 6, fc=panel_colors[0], ec="none"))
    # Step circle
    ax.add_patch(Circle((0.5, 5.5), 0.35, fc="white", ec="#222", lw=1.5))
    ax.text(0.5, 5.5, "1", fontsize=14, fontweight="bold", ha="center", va="center")
    ax.text(1.1, 5.5, step_titles[0], fontsize=14, fontweight="bold", va="center")
    # London grid (5x5 mini map with home cells outer, work cells in centre)
    grid_cells = [(1.2 + c * 0.7, 1.7 + r * 0.7) for r in range(5) for c in range(5)]
    centre_idxs = [12, 11, 13, 7, 17]   # 5 inner cells
    for k, (cx, cy) in enumerate(grid_cells):
        if k in centre_idxs:
            ax.add_patch(Rectangle((cx - 0.28, cy - 0.28), 0.56, 0.56,
                                    fc="#3a5", ec="black", lw=0.6))
        else:
            ax.add_patch(Rectangle((cx - 0.28, cy - 0.28), 0.56, 0.56,
                                    fc="#aaa", ec="black", lw=0.4))
    # Arrows from outer to inner (commute flows)
    flow_pairs = [(0, 12), (4, 12), (20, 12), (24, 12), (2, 11), (22, 13)]
    for o, d in flow_pairs:
        ox, oy = grid_cells[o]
        dx_, dy = grid_cells[d]
        arr = FancyArrowPatch((ox, oy), (dx_, dy),
                               arrowstyle="-|>", mutation_scale=8,
                               linewidth=0.8, color="#a04", alpha=0.6)
        ax.add_patch(arr)
    ax.text(3, 1.2, step_captions[0], fontsize=10, ha="center", va="center")

    # Panel 2: Learn
    ax = axes[1]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 6, 6, fc=panel_colors[1], ec="none"))
    ax.add_patch(Circle((0.5, 5.5), 0.35, fc="white", ec="#222", lw=1.5))
    ax.text(0.5, 5.5, "2", fontsize=14, fontweight="bold", ha="center", va="center")
    ax.text(1.1, 5.5, step_titles[1], fontsize=14, fontweight="bold", va="center")
    # Two visuals stacked: "attractiveness map" + "trade-off lever"
    # Top: attractiveness map (gradient)
    cmap = plt.cm.YlOrRd
    for r in range(4):
        for c in range(4):
            cx = 0.9 + c * 0.45
            cy = 4.4 - r * 0.4
            attractiveness = (3 - abs(r - 1.5)) * (3 - abs(c - 1.5)) / 9
            ax.add_patch(Rectangle((cx - 0.2, cy - 0.18), 0.4, 0.36,
                                    fc=cmap(attractiveness), ec="none"))
    ax.text(2.95, 4.65, "→ attractiveness\n     of each place",
            fontsize=9, va="center", color="#222")
    # Bottom: trade-off scale visual
    bx, by = 1.7, 2.0
    ax.plot([bx - 1.0, bx + 1.0], [by, by], "k-", lw=2)
    ax.plot([bx, bx], [by, by + 0.6], "k-", lw=2)
    ax.plot([bx - 1.0, bx - 1.0], [by, by - 0.5], "k-", lw=2)
    ax.plot([bx + 1.0, bx + 1.0], [by, by - 0.5], "k-", lw=2)
    ax.text(bx - 1.0, by - 0.75, "TIME", fontsize=8, ha="center")
    ax.text(bx + 1.0, by - 0.75, "WAGE", fontsize=8, ha="center")
    ax.text(bx, by + 0.85, "trade-off\nlearned", fontsize=8.5, ha="center",
            fontstyle="italic", color="#222")
    ax.text(3, 0.65, step_captions[1], fontsize=10, ha="center", va="center")

    # Panel 3: Simulate
    ax = axes[2]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 6, 6, fc=panel_colors[2], ec="none"))
    ax.add_patch(Circle((0.5, 5.5), 0.35, fc="white", ec="#222", lw=1.5))
    ax.text(0.5, 5.5, "3", fontsize=14, fontweight="bold", ha="center", va="center")
    ax.text(1.1, 5.5, step_titles[2], fontsize=14, fontweight="bold", va="center")
    # Two intervention examples side by side
    # A: spatial — add jobs at one cluster
    ax.text(0.4, 4.5, "Spatial:", fontsize=10, fontweight="bold", color="#222")
    for r in range(3):
        for c in range(3):
            cx = 0.5 + c * 0.32
            cy = 4.0 - r * 0.32
            if r == 1 and c == 1:
                ax.add_patch(Rectangle((cx - 0.13, cy - 0.13), 0.26, 0.26,
                                        fc="#3a5", ec="black", lw=1))
            else:
                ax.add_patch(Rectangle((cx - 0.13, cy - 0.13), 0.26, 0.26,
                                        fc="#aaa", ec="black", lw=0.3))
    big_arrow(ax, 1.5, 3.65, dx=0.5, color="#a04", lw=2)
    for r in range(3):
        for c in range(3):
            cx = 2.2 + c * 0.32
            cy = 4.0 - r * 0.32
            if r == 1 and c == 1:
                ax.add_patch(Rectangle((cx - 0.18, cy - 0.18), 0.36, 0.36,
                                        fc="#3a5", ec="black", lw=1.5))
            else:
                ax.add_patch(Rectangle((cx - 0.13, cy - 0.13), 0.26, 0.26,
                                        fc="#aaa", ec="black", lw=0.3))
    ax.text(3.3, 3.65, "+jobs", fontsize=9, color="#a04", fontweight="bold")
    # B: temporal — shift peak hours
    ax.text(0.4, 2.6, "Temporal:", fontsize=10, fontweight="bold", color="#222")
    hours = np.arange(6, 22)
    base = np.array([0.3, 1.0, 3.5, 6.5, 7.5, 3.0, 1.0, 1.0, 1.2, 1.4, 1.5, 2.5,
                     5.5, 6.0, 3.0, 1.0]) / 8
    scen = base.copy()
    scen[1:4] *= 0.5
    scen[4:10] *= 1.5
    bar_y0 = 0.7
    bar_w = 0.2
    for h in range(16):
        ax.bar(0.5 + h * 0.32, base[h] * 1.6, bottom=bar_y0,
                width=bar_w, color="#888", alpha=0.4)
        ax.bar(0.5 + h * 0.32, scen[h] * 1.6, bottom=bar_y0,
                width=bar_w * 0.7, color="#3a5", alpha=0.85)
    ax.text(5.6, 1.5, "shift\npeak\n→ off-peak",
            fontsize=8, color="#a04", ha="center", fontweight="bold")
    ax.text(3, 0.25, step_captions[2], fontsize=10, ha="center", va="center")

    # Panel 4: Evaluate
    ax = axes[3]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 6, 6, fc=panel_colors[3], ec="none"))
    ax.add_patch(Circle((0.5, 5.5), 0.35, fc="white", ec="#222", lw=1.5))
    ax.text(0.5, 5.5, "4", fontsize=14, fontweight="bold", ha="center", va="center")
    ax.text(1.1, 5.5, step_titles[3], fontsize=14, fontweight="bold", va="center")
    # Visualise winners/losers as arrows up (green) and down (red) on London map mini
    # Plus a simple Gini bar
    grid_cells2 = [(1.0 + c * 0.7, 2.5 + r * 0.7) for r in range(3) for c in range(3)]
    deltas = [+0.3, -0.1, -0.05, -0.2, +0.5, +0.4, -0.15, +0.05, -0.3]
    for k, (cx, cy) in enumerate(grid_cells2):
        d = deltas[k]
        if d > 0:
            ax.add_patch(Rectangle((cx - 0.28, cy - 0.28), 0.56, 0.56,
                                    fc="#3a5", ec="black", lw=0.6, alpha=0.4 + d))
            arr = FancyArrowPatch((cx, cy - 0.05), (cx, cy + 0.25),
                                   arrowstyle="-|>", mutation_scale=10,
                                   linewidth=1.5, color="#080")
            ax.add_patch(arr)
        else:
            ax.add_patch(Rectangle((cx - 0.28, cy - 0.28), 0.56, 0.56,
                                    fc="#a04", ec="black", lw=0.6, alpha=0.4 - d))
            arr = FancyArrowPatch((cx, cy + 0.05), (cx, cy - 0.25),
                                   arrowstyle="-|>", mutation_scale=10,
                                   linewidth=1.5, color="#a04")
            ax.add_patch(arr)
    ax.text(4.7, 4.5, "↑ gains\n   access", fontsize=9, color="#080",
            fontweight="bold", ha="center")
    ax.text(4.7, 3.2, "↓ loses\n   access", fontsize=9, color="#a04",
            fontweight="bold", ha="center")
    # Inequality measure
    ax.text(0.5, 1.5, "→ measures",
            fontsize=9, color="#222")
    ax.text(0.5, 1.1, "Gini, Palma,\nAtkinson",
            fontsize=8.5, color="#222", fontweight="bold")
    # Spatial vs temporal comparison
    ax.bar([3.5, 4.5], [+1.26, -1.10], width=0.6,
            color=["#a04", "#080"], alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.text(3.5, +1.5, "spatial:\nGini ↑\n(unfair)",
            fontsize=8, ha="center", color="#a04")
    ax.text(4.5, -1.5, "temporal:\nGini ↓\n(fairer)",
            fontsize=8, ha="center", color="#080", va="top")
    ax.text(3, 0.0, step_captions[3], fontsize=10, ha="center", va="center")

    # Big arrows between panels
    fig.canvas.draw()
    for k in range(3):
        # Arrow between axes — using figure coords
        x_start = (k + 1) * 0.245 + 0.005
        x_end = x_start + 0.012
        y_mid = 0.5
        fig.add_artist(FancyArrowPatch(
            (x_start, y_mid), (x_end + 0.018, y_mid),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=18,
            linewidth=2, color="#444",
        ))

    fig.suptitle("Conceptual framework: from observed flows to policy evaluation",
                  fontsize=13, fontweight="bold", y=1.02)

    out_png = OUT_DIR / "F1_conceptual.png"
    out_svg = OUT_DIR / "F1_conceptual.svg"
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
