"""Generate mechanism mini-viz PNGs for the architecture figure.

Each mini-viz is a small (~3x2 inch) PNG focused on ONE mechanism inside
the architecture. Embedded as <img> in figure1_architecture.html.

Outputs to: methodology/figures/arch/{name}.png   (200 dpi)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "methodology" / "figures" / "arch"
OUT.mkdir(parents=True, exist_ok=True)

# ---- shared style --------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.titlepad": 4,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

GREEN = "#22c55e"
GREEN_DK = "#15803d"
AMBER = "#f59e0b"
AMBER_DK = "#b45309"
TEAL = "#0d9488"
PURPLE = "#7c3aed"
RED = "#dc2626"
ORANGE = "#ea580c"
SLATE = "#334155"
GRAY = "#94a3b8"


def save(fig, name: str):
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.name}  ({p.stat().st_size/1024:.1f} KB)")


# ============================================================================
# 1. GraphSAGE neighbour aggregation schematic
# ============================================================================
def fig_sage():
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    # Centre node
    ax.add_patch(Circle((0, 0), 0.18, color=GREEN_DK, zorder=3))
    ax.text(0, 0, "i", color="white", ha="center", va="center", fontsize=10, fontweight="bold")
    # Neighbours
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    for a in angles:
        x, y = 0.7 * np.cos(a), 0.7 * np.sin(a)
        ax.add_patch(Circle((x, y), 0.13, color=GREEN, alpha=0.85, zorder=2))
        # Edge from neighbour to centre
        arr = FancyArrowPatch((x*0.8, y*0.8), (0.18*np.cos(a), 0.18*np.sin(a)),
                                arrowstyle="-|>", mutation_scale=8, color=GRAY, lw=1.0, zorder=1)
        ax.add_patch(arr)
    # Equation underneath
    ax.text(0, -1.05, r"$h_i^{(\ell+1)} = \sigma\!\left(W^{(\ell)} \cdot \mathrm{mean}_{j \in \mathcal{N}(i)} h_j^{(\ell)}\right)$",
             ha="center", va="top", fontsize=8.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.4, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "sage_aggregation")


# ============================================================================
# 2. GRU cell schematic
# ============================================================================
def fig_gru():
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    # Boxes for h_{t-1} and h_t
    for x, label in [(0.05, r"$h_{t-1}$"), (0.85, r"$h_t$")]:
        ax.add_patch(FancyBboxPatch((x, 0.45), 0.12, 0.18,
                                       boxstyle="round,pad=0.01,rounding_size=0.02",
                                       facecolor=AMBER, edgecolor=AMBER_DK, lw=0.8))
        ax.text(x+0.06, 0.54, label, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    # Centre cell
    ax.add_patch(FancyBboxPatch((0.30, 0.30), 0.45, 0.5,
                                   boxstyle="round,pad=0.01,rounding_size=0.04",
                                   facecolor="white", edgecolor=AMBER_DK, lw=1.0))
    ax.text(0.525, 0.71, "GRU cell", ha="center", fontsize=8.5, color=AMBER_DK, fontweight="bold")
    # Internal gates
    for i, g in enumerate(["z", "r", "ñ"]):
        cx = 0.36 + i*0.15
        ax.add_patch(Circle((cx, 0.50), 0.04, color=AMBER_DK, alpha=0.75))
        ax.text(cx, 0.50, g, color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(cx, 0.40, ["update", "reset", "cand"][i], ha="center", fontsize=6.5, color=SLATE)
    # x_t input
    ax.text(0.525, 0.20, r"$+\;x_t$", ha="center", fontsize=9, color=SLATE)
    # Arrows in/out
    for (sx, sy, ex, ey) in [(0.17, 0.54, 0.30, 0.55), (0.75, 0.55, 0.85, 0.54)]:
        ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>",
                                        mutation_scale=8, color=SLATE, lw=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")
    save(fig, "gru_cell")


# ============================================================================
# 3. Multi-scale TCN kernel widths over 24-hour strip
# ============================================================================
def fig_tcn():
    fig, ax = plt.subplots(figsize=(2.8, 1.6))
    # 24-hour strip
    for h in range(24):
        ax.add_patch(FancyBboxPatch((h, 0), 0.95, 0.5,
                                       boxstyle="round,pad=0.01,rounding_size=0.02",
                                       facecolor=AMBER, alpha=0.18 + 0.05*np.cos(h*np.pi/12),
                                       edgecolor="none"))
        if h % 6 == 0:
            ax.text(h+0.5, -0.18, f"{h}", ha="center", fontsize=7, color=SLATE)
    ax.text(12, -0.45, "hour of day (T = 24)", ha="center", fontsize=8, color=SLATE)

    # Three kernel windows (3, 5, 7) centred on h=8
    centre = 8
    kernels = [(3, 0.85, "#fbbf24"), (5, 1.10, "#f97316"), (7, 1.40, "#dc2626")]
    for k, y, c in kernels:
        x0 = centre - k/2 + 0.5
        ax.add_patch(FancyBboxPatch((x0, y - 0.16), k, 0.32,
                                       boxstyle="round,pad=0.01,rounding_size=0.06",
                                       facecolor=c, alpha=0.65, edgecolor=c, lw=1.0))
        ax.text(centre + 0.5, y, f"k={k}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    ax.text(20, 1.10, "3 parallel\ncausal convs\n(circular)",
             ha="left", va="center", fontsize=7, color=SLATE)
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-0.7, 1.7)
    ax.axis("off")
    save(fig, "tcn_kernels")


# ============================================================================
# 4. Gated fusion — sigmoid + bar plot of gate(t, j)
# ============================================================================
def fig_gated_fusion():
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.6), gridspec_kw={"width_ratios": [1.2, 1.8]})

    # (a) sigmoid
    ax = axes[0]
    x = np.linspace(-5, 5, 200)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color=TEAL, lw=1.6)
    ax.set_xlim(-5, 5); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.axhline(0.5, color=GRAY, lw=0.5, ls="--")
    ax.set_title(r"$\sigma$ gate", fontsize=8.5, color=TEAL)
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")

    # (b) gate(t, j) example: random per-hour gate
    ax = axes[1]
    np.random.seed(0)
    g = 0.3 + 0.4 * np.random.rand(24)
    ax.bar(range(24), g, color=TEAL, alpha=0.85, width=0.85)
    ax.bar(range(24), 1 - g, bottom=g, color=AMBER, alpha=0.55, width=0.85)
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(["0", "6", "12", "18", "23"], fontsize=7)
    ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=7)
    ax.set_xlim(-0.6, 23.6); ax.set_ylim(0, 1.05)
    ax.set_title(r"gate$(t, j)$ per hour", fontsize=8.5)
    ax.set_xlabel("hour", fontsize=7, color=SLATE)
    legend = [mpatches.Patch(color=TEAL, label="static"),
              mpatches.Patch(color=AMBER, label="dynamic")]
    ax.legend(handles=legend, fontsize=6, loc="upper right", frameon=False, ncol=1)
    plt.tight_layout()
    save(fig, "gated_fusion")


# ============================================================================
# 5. Mode × Tier β matrix heatmap (with recovered values)
# ============================================================================
def fig_mode_tier_beta():
    fig, ax = plt.subplots(figsize=(2.7, 1.8))
    # Recovered (mean over hours): β_car = -0.187, β_transit = -0.105, β_walk = -0.049
    # κ = [1.0, 1.97, 1.97]
    beta_per_mode = np.array([-0.187, -0.105, -0.049])
    kappa = np.array([1.0, 1.97, 1.97])
    # β_eff[m, k] = β_per_mode[m] * κ[k]
    beta_eff = beta_per_mode[:, None] * kappa[None, :]
    im = ax.imshow(beta_eff, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for m in range(3):
        for k in range(3):
            ax.text(k, m, f"{beta_eff[m,k]:+.2f}", ha="center", va="center",
                     fontsize=7.5, color="white" if abs(beta_eff[m,k]) > 0.18 else SLATE)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low\n(κ=1.0)", "mid\n(κ=1.97)", "high\n(κ=1.97)"], fontsize=7)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["car", "transit", "walk"], fontsize=8)
    ax.set_title(r"$\beta_{m,k} = \beta_m \cdot \kappa_k$ (recovered, mean over T)", fontsize=8)
    ax.set_xlabel("income tier k", fontsize=7.5, color=SLATE)
    ax.set_ylabel("mode m", fontsize=7.5, color=SLATE)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "mode_tier_beta")


# ============================================================================
# 6. Training curve — NLL + CPC over 200 epochs (real data)
# ============================================================================
def fig_training_curve():
    log_path = ROOT / "evaluation_outputs" / "paper_a" / "dual_het_seed0.log.json"
    if not log_path.exists():
        # fallback: no real log; placeholder
        fig, ax = plt.subplots(figsize=(2.8, 1.7))
        ax.text(0.5, 0.5, "training log\nnot found", ha="center", va="center", color=GRAY)
        ax.axis("off"); save(fig, "training_curve"); return
    with open(log_path) as f:
        log = json.load(f)
    epoch = log["epoch"]
    train_nll = log["train_nll"]
    val_nll = log["val_nll"]
    cpc = log["cpc_val"]

    fig, ax1 = plt.subplots(figsize=(2.8, 1.7))
    ax1.plot(epoch, train_nll, color=PURPLE, lw=1.0, label="train NLL", alpha=0.85)
    ax1.plot(epoch, val_nll, color=PURPLE, lw=1.5, label="val NLL", linestyle="--")
    ax1.set_xlabel("epoch", fontsize=7.5, color=SLATE)
    ax1.set_ylabel("NLL", fontsize=7.5, color=PURPLE)
    ax1.tick_params(labelsize=7, axis="y", colors=PURPLE)
    ax1.tick_params(labelsize=7, axis="x")
    ax1.spines["left"].set_color(PURPLE)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(epoch, cpc, color=GREEN_DK, lw=1.5)
    ax2.set_ylabel("CPC", fontsize=7.5, color=GREEN_DK)
    ax2.tick_params(labelsize=7, axis="y", colors=GREEN_DK)
    ax2.spines["right"].set_color(GREEN_DK)
    ax2.set_ylim(0.25, 0.50)
    # final values
    ax1.text(epoch[-1], val_nll[-1], f"  val NLL\n  {val_nll[-1]:.1f}", fontsize=6.5, color=PURPLE, va="center")
    ax2.text(epoch[-1], cpc[-1], f"  CPC\n  {cpc[-1]:.3f}", fontsize=6.5, color=GREEN_DK, va="center")
    save(fig, "training_curve")


# ============================================================================
# 7. BPR V/C curve
# ============================================================================
def fig_bpr_curve():
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    vc = np.linspace(0, 1.6, 200)
    mult = 1 + 0.15 * vc**4
    mult_capped = np.minimum(mult, 2.5)
    ax.plot(vc, mult, color=ORANGE, lw=1.0, ls="--", alpha=0.7, label="uncapped")
    ax.plot(vc, mult_capped, color=ORANGE, lw=1.6, label="capped at 2.5")
    ax.axhline(1.0, color=GRAY, lw=0.5)
    ax.axvline(1.0, color=GRAY, lw=0.5, ls=":")
    ax.text(1.02, 1.05, "V/C = 1\n(at capacity)", fontsize=6.5, color=SLATE)
    ax.set_xlim(0, 1.6); ax.set_ylim(0.95, 2.7)
    ax.set_xlabel("V/C ratio (inflow / capacity)", fontsize=7.5, color=SLATE)
    ax.set_ylabel(r"$t \,/\, t_0$", fontsize=8, color=SLATE)
    ax.tick_params(labelsize=7)
    ax.set_title(r"BPR: $t = t_0 (1 + 0.15 \cdot (V/C)^4)$", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    save(fig, "bpr_curve")


# ============================================================================
# 8. Synthetic agent grid (for Stage 2 ABM)
# ============================================================================
def fig_agent_grid():
    fig, ax = plt.subplots(figsize=(2.4, 1.7))
    np.random.seed(1)
    # 50 agents scattered
    n = 50
    x = np.random.rand(n)
    y = np.random.rand(n) * 0.7
    # Mode: 60% car, 30% transit, 10% walk
    modes = np.random.choice(["car", "transit", "walk"], n, p=[0.6, 0.3, 0.1])
    cmap_mode = {"car": "#3b82f6", "transit": "#f59e0b", "walk": "#10b981"}
    for xi, yi, m in zip(x, y, modes):
        ax.scatter(xi, yi, color=cmap_mode[m], s=22, alpha=0.85,
                    edgecolors="white", linewidths=0.5)
    # Legend
    for i, m in enumerate(["car", "transit", "walk"]):
        ax.scatter(0.05 + i*0.3, 0.95, color=cmap_mode[m], s=24)
        ax.text(0.10 + i*0.3, 0.95, m, fontsize=7, va="center")
    ax.text(0.5, 0.82, r"$N_{\mathrm{agents}}$ samples from $\pi_m(i)$, $\pi_k(i)$",
             ha="center", fontsize=7.5, color=SLATE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.axis("off")
    save(fig, "synthetic_agents")


# ============================================================================
# 9. Softmax + Gumbel choice
# ============================================================================
def fig_softmax_choice():
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    np.random.seed(3)
    n = 12
    util = -1.5 + 3.0 * np.random.rand(n)
    p = np.exp(util) / np.exp(util).sum()
    chosen = np.argmax(util + np.random.gumbel(0, 1, n))
    cols = [PURPLE if i == chosen else GRAY for i in range(n)]
    ax.bar(range(n), p, color=cols, width=0.7, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([0, max(p)])
    ax.set_yticklabels(["0", f"{max(p):.2f}"], fontsize=6.5)
    ax.text(chosen, p[chosen] + 0.01, "★", ha="center", fontsize=10, color=PURPLE)
    ax.set_xlabel(r"candidate workplaces $j$", fontsize=7, color=SLATE)
    ax.set_title(r"$P(j \mid i, t)$ + Gumbel $\to$ $\arg\max$", fontsize=7.5)
    save(fig, "softmax_gumbel")


# ============================================================================
if __name__ == "__main__":
    print("[arch_miniviz] generating mechanism mini-viz to figures/arch/ ...")
    fig_sage()
    fig_gru()
    fig_tcn()
    fig_gated_fusion()
    fig_mode_tier_beta()
    fig_training_curve()
    fig_bpr_curve()
    fig_agent_grid()
    fig_softmax_choice()
    print("[arch_miniviz] done.")
