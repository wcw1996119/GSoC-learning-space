"""F1 v2 conceptual framework — A4-friendly portrait layout (max ~6.5 inches wide).

Single column, vertical flow: data → calibration → counterfactual → mixed-logit MC.
Designed to fit A4 page width without horizontal cropping.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                 Circle, RegularPolygon)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def add_arrow(ax, x1, y1, x2, y2, *, lw=1.2, color="#222", style="-|>",
              connectionstyle="arc3,rad=0", alpha=1.0, mut=10):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=mut,
                           linewidth=lw, color=color,
                           connectionstyle=connectionstyle, alpha=alpha)
    ax.add_patch(arr)


def stick(ax, cx, cy, scale=0.5, color="#444", lw=1.5):
    head = Circle((cx, cy + 0.6 * scale), 0.18 * scale, ec=color, fc="white", lw=lw)
    ax.add_patch(head)
    ax.plot([cx, cx], [cy + 0.42 * scale, cy - 0.2 * scale], color=color, lw=lw)
    ax.plot([cx - 0.3 * scale, cx + 0.3 * scale], [cy + 0.1 * scale, cy + 0.1 * scale],
             color=color, lw=lw)
    ax.plot([cx, cx - 0.25 * scale], [cy - 0.2 * scale, cy - 0.65 * scale], color=color, lw=lw)
    ax.plot([cx, cx + 0.25 * scale], [cy - 0.2 * scale, cy - 0.65 * scale], color=color, lw=lw)


def building(ax, cx, cy, scale=0.5, color="#264", windows=2):
    w = 0.5 * scale; h = 0.7 * scale
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=color, ec="black", lw=0.6))
    for r in range(2):
        for c in range(windows):
            wx = cx - w / 2 + (c + 1) * w / (windows + 1) - 0.04 * scale
            wy = cy - h / 2 + (r + 1) * h / 3 - 0.04 * scale
            ax.add_patch(Rectangle((wx, wy), 0.08 * scale, 0.08 * scale,
                                    fc="white", ec="none"))


def main():
    # 6.5 inches wide × 8 inches tall — fits inside A4 page margins
    fig = plt.figure(figsize=(6.5, 8.5))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.25)

    # ─────────────────────────────────────────────
    # Panel A — Input grid + features
    # ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(a) 1 km London grid",
                  fontsize=9.5, fontweight="bold", loc="left")
    cell_centres = [(0.7 + c * 0.7, 1.2 + r * 0.7) for r in range(4) for c in range(4)]
    target_idx = 5
    for k, (cx, cy) in enumerate(cell_centres):
        if k == target_idx:
            ax.add_patch(Rectangle((cx - 0.32, cy - 0.32), 0.64, 0.64,
                                    fc="#ffe9b3", ec="#c87f00", lw=1.5))
        else:
            ax.add_patch(Rectangle((cx - 0.32, cy - 0.32), 0.64, 0.64,
                                    fc="#f6f6f6", ec="#888", lw=0.5))
    cx_t, cy_t = cell_centres[target_idx]
    ax.text(cx_t, cy_t, "j", fontsize=10, fontweight="bold", ha="center",
            va="center", color="#7b3f00")
    ax.text(0.2, 0.7, "Cell features:", fontsize=8, fontweight="bold")
    ax.text(0.2, 0.3,
            "$E_j$ jobs, $P_j$ pop., POIs,\nstations, IMD, wage",
            fontsize=7.5)

    # ─────────────────────────────────────────────
    # Panel B — GNN aggregation
    # ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(b) GraphSAGE: $V_j$ from cell + 10 neighbours",
                  fontsize=9.5, fontweight="bold", loc="left")
    cj = (2.5, 2.7)
    ax.add_patch(Circle(cj, 0.30, fc="#ffd573", ec="#c87f00", lw=1.5))
    ax.text(cj[0], cj[1], "j", fontsize=10, fontweight="bold", ha="center", va="center")
    n_neigh = 6
    radii = 1.3
    for i in range(n_neigh):
        ang = 2 * np.pi * i / n_neigh + np.pi / 6
        nx = cj[0] + radii * np.cos(ang)
        ny = cj[1] + radii * np.sin(ang)
        ax.add_patch(Circle((nx, ny), 0.18, fc="#cde3f3", ec="#446", lw=1))
        ax.text(nx, ny, f"$u_{i+1}$", fontsize=6.5, ha="center", va="center")
        add_arrow(ax, nx, ny, cj[0], cj[1], lw=0.9, color="#446", mut=6)
    ax.text(0.2, 0.7,
            r"$V_j = \mathrm{GNN}_\theta(X_j, \{X_{u}\})$",
            fontsize=8.5, color="#a04",
            bbox=dict(facecolor="#fff", edgecolor="#a04",
                       boxstyle="round,pad=0.2"))
    ax.text(0.2, 0.25, "= attractiveness score",
            fontsize=7, fontstyle="italic", color="#444")

    # ─────────────────────────────────────────────
    # Panel C — Random utility decision
    # ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(c) Random-utility choice (deterministic)",
                  fontsize=9.5, fontweight="bold", loc="left")
    stick(ax, 0.7, 2.7, scale=0.7, color="#222")
    ax.text(0.7, 1.7, "agent", fontsize=7.5, ha="center", fontweight="bold")
    ax.text(0.7, 1.4, "(home $i$)", fontsize=6.5, ha="center")
    dest_pos = [(2.6, 4.2), (3.6, 3.3), (4.0, 2.3), (3.4, 1.4), (2.4, 0.7)]
    dest_V = [0.9, 0.6, 0.3, 0.4, 0.2]
    dest_t = [38, 22, 15, 28, 18]
    util = [v + (-0.07) * t for v, t in zip(dest_V, dest_t)]
    um = max(util)
    expu = [np.exp(u - um) for u in util]
    Z = sum(expu)
    probs = [e / Z for e in expu]
    for k, ((dx, dy), p) in enumerate(zip(dest_pos, probs)):
        building(ax, dx, dy, scale=0.7, color="#3a5", windows=2)
        ax.text(dx + 0.3, dy + 0.05, f"$j_{k+1}$", fontsize=7, fontweight="bold")
        lw = 0.5 + p * 4
        a = 0.4 + p * 0.6
        add_arrow(ax, 1.0, 2.7, dx - 0.2, dy, lw=lw, color="#a04",
                   alpha=a, mut=8)
    ax.text(0.2, 0.0,
            r"$U_{ij} = \alpha V_j + \beta_t t_{ij} + \gamma \log d_{ij} + \delta \mathrm{OM}_{ij}$",
            fontsize=7.5, color="#a04",
            bbox=dict(facecolor="#fff", edgecolor="#a04",
                       boxstyle="round,pad=0.2"))

    # ─────────────────────────────────────────────
    # Panel D — BPR equilibrium
    # ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(d) Aggregate flow + BPR feedback",
                  fontsize=9.5, fontweight="bold", loc="left")
    for k in range(4):
        stick(ax, 0.6, 4.2 - k * 0.85, scale=0.4, color="#444", lw=1)
    cx_d, cy_d = 3.5, 2.7
    building(ax, cx_d, cy_d, scale=1.4, color="#3a5", windows=3)
    ax.text(cx_d, cy_d - 1.0, "destination $j$", fontsize=8, ha="center",
            fontweight="bold")
    ax.text(cx_d, cy_d - 1.3, "capacity $C_j$", fontsize=7, ha="center")
    for k in range(4):
        add_arrow(ax, 0.9, 4.2 - k * 0.85, cx_d - 0.35, cy_d - 0.2,
                   lw=0.9, color="#666", alpha=0.55, mut=6,
                   connectionstyle="arc3,rad=0.05")
    add_arrow(ax, 4.0, 1.5, 1.4, 2.5, color="#a04", lw=1.4,
               connectionstyle="arc3,rad=0.4", mut=10)
    ax.text(2.7, 1.85, "feedback", fontsize=7, color="#a04",
            ha="center", fontstyle="italic")
    ax.text(0.2, 0.05,
            r"$t_{ij}^{\rm new} = t_{ij}^0 (1 + 0.15 (V_j/C_j)^4)$",
            fontsize=7.5, color="#a04",
            bbox=dict(facecolor="#fff", edgecolor="#a04",
                       boxstyle="round,pad=0.2"))

    # ─────────────────────────────────────────────
    # Panel E — Counterfactual: spatial + temporal
    # ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(e) Counterfactual: do$(X_S = X_S^*)$",
                  fontsize=9.5, fontweight="bold", loc="left")
    # Spatial
    ax.text(0.2, 4.5, "Spatial: +65k jobs at OOC", fontsize=8.5, fontweight="bold")
    building(ax, 1.2, 3.7, scale=0.55, color="#3a5", windows=2)
    add_arrow(ax, 1.7, 3.7, 2.4, 3.7, lw=1.2, color="#a04", mut=8)
    building(ax, 3.3, 3.7, scale=0.95, color="#3a5", windows=3)
    # Temporal
    ax.text(0.2, 2.5, "Temporal: peak ×0.5, off-peak ×1.5",
            fontsize=8.5, fontweight="bold")
    base = np.array([0.3, 1.0, 3.0, 6.5, 7.5, 3.5, 1.3, 1.0, 1.3, 1.6, 1.5, 2.7,
                     5.5, 6.5, 3.5, 1.4]) / 10
    scen = base.copy()
    scen[3:6] *= 0.5
    scen[6:12] *= 1.5
    bar_y0 = 0.4; bar_w = 0.18
    for h in range(16):
        ax.bar(0.5 + h * 0.27, base[h] * 1.6, bottom=bar_y0,
                width=bar_w, color="#888", alpha=0.4)
        ax.bar(0.5 + h * 0.27, scen[h] * 1.6, bottom=bar_y0,
                width=bar_w * 0.7, color="#3a5", alpha=0.85)
    ax.text(0.5, bar_y0 - 0.2, "h6", fontsize=6)
    ax.text(0.5 + 15 * 0.27, bar_y0 - 0.2, "h21", fontsize=6)

    # ─────────────────────────────────────────────
    # Panel F — Mixed-logit Monte Carlo
    # ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(f) Mixed-logit MC (heterogeneous $\\beta_n$)",
                  fontsize=9.5, fontweight="bold", loc="left")
    ax.text(0.2, 4.2, "10K agents,\neach $\\beta_n$ from WebTAG dist.",
            fontsize=8)
    for k, c in enumerate(["#a04", "#048", "#080"]):
        stick(ax, 0.7 + k * 0.5, 3.0 - k * 0.3, scale=0.4, color=c)
    ax.text(2.8, 3.05, r"$\beta_n=\hat{\beta}_t(1+\varepsilon_n)$",
            fontsize=8.5, color="#222")
    ax.text(2.8, 2.65, r"$\varepsilon_n \sim \mathcal{N}(0,\sigma^2)$",
            fontsize=8, color="#222")
    ax.text(0.2, 1.3,
            "Bias = deterministic - MC:\n"
            "CBD intervention: $+2.1\\%$\n"
            "Greenfield: $-16.7\\%$",
            fontsize=7.5, color="#143",
            bbox=dict(facecolor="#fff2cc", edgecolor="#a07c00",
                       boxstyle="round,pad=0.25"))

    # Title
    fig.suptitle("Conceptual framework",
                  fontsize=12, fontweight="bold", y=0.985)

    out_png = OUT_DIR / "F1_framework.png"
    out_svg = OUT_DIR / "F1_framework.svg"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png} ({out_png.stat().st_size/1024:.0f} KB)")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
