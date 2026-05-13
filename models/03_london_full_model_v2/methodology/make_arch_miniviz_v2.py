"""V2 mini-viz: smaller, tighter aspect ratios for horizontal layout.

Some of the v1 mini-viz are too tall for the new horizontal layout.
This script regenerates the ones that need re-aspecting; the rest stay.
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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

GREEN, GREEN_DK = "#22c55e", "#15803d"
AMBER, AMBER_DK = "#f59e0b", "#b45309"
TEAL = "#0d9488"
PURPLE, PURPLE_DK = "#a78bfa", "#6d28d9"
RED = "#dc2626"
ORANGE, ORANGE_DK = "#fb923c", "#c2410c"
SLATE = "#334155"
GRAY = "#94a3b8"
BLUE = "#3b82f6"


def save(fig, name: str):
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.name}  ({p.stat().st_size/1024:.1f} KB)")


# --------- 1. SAGE small (tighter, square)
def fig_sage_v2():
    fig, ax = plt.subplots(figsize=(2.0, 1.6))
    ax.add_patch(Circle((0, 0), 0.15, color=GREEN_DK, zorder=3))
    ax.text(0, 0, "i", color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    for a in angles:
        x, y = 0.6 * np.cos(a), 0.6 * np.sin(a)
        ax.add_patch(Circle((x, y), 0.10, color=GREEN, alpha=0.85, zorder=2))
        arr = FancyArrowPatch((x*0.78, y*0.78), (0.15*np.cos(a), 0.15*np.sin(a)),
                                arrowstyle="-|>", mutation_scale=7, color=GRAY, lw=0.9, zorder=1)
        ax.add_patch(arr)
    ax.text(0, -1.0, r"$h_i^{(\ell+1)}=\sigma(W^{(\ell)} \cdot \mathrm{mean}\,h_{N(i)}^{(\ell)})$",
             ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-1.25, 0.95)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "sage_v2")


# --------- 2. GRU + TCN combined small horizontal strip
def fig_dyn_branch_v2():
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.4), gridspec_kw={"width_ratios": [1.0, 1.7]})
    # GRU cell
    ax = axes[0]
    ax.add_patch(FancyBboxPatch((0.2, 0.25), 0.6, 0.55,
                                   boxstyle="round,pad=0.02,rounding_size=0.05",
                                   facecolor="white", edgecolor=AMBER_DK, lw=0.9))
    ax.text(0.5, 0.74, "GRU", ha="center", fontsize=8, color=AMBER_DK, fontweight="bold")
    for i, g in enumerate(["z", "r", "ñ"]):
        cx = 0.27 + i*0.20
        ax.add_patch(Circle((cx, 0.45), 0.045, color=AMBER_DK, alpha=0.8))
        ax.text(cx, 0.45, g, color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(0.05, 0.50, r"$h_{t-1}$", fontsize=8, va="center")
    ax.text(0.95, 0.50, r"$h_t$", fontsize=8, va="center")
    ax.text(0.5, 0.10, "sequential\nmemory", ha="center", fontsize=7, color=SLATE, style="italic")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1)
    ax.axis("off")
    # TCN
    ax = axes[1]
    for h in range(24):
        ax.add_patch(FancyBboxPatch((h, 0), 0.92, 0.40,
                                       boxstyle="round,pad=0.005",
                                       facecolor=AMBER, alpha=0.18 + 0.04*np.cos(h*np.pi/12),
                                       edgecolor="none"))
    ax.text(12, -0.20, "T = 24 hours", ha="center", fontsize=7, color=SLATE)
    centre = 8.5
    kernels = [(3, 0.65, "#fbbf24"), (5, 0.85, "#f97316"), (7, 1.10, "#dc2626")]
    for k, y, c in kernels:
        x0 = centre - k/2
        ax.add_patch(FancyBboxPatch((x0, y - 0.11), k, 0.22,
                                       boxstyle="round,pad=0.005,rounding_size=0.04",
                                       facecolor=c, alpha=0.65, edgecolor=c, lw=0.9))
        ax.text(centre, y, f"k={k}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(0.5, 1.30, "multi-scale TCN (kernels 3, 5, 7)", ha="left", fontsize=7.5, color=SLATE)
    ax.set_xlim(-0.5, 25); ax.set_ylim(-0.45, 1.45)
    ax.axis("off")
    plt.tight_layout(pad=0.3, w_pad=0.5)
    save(fig, "dyn_branch_v2")


# --------- 3. Gated fusion small (just the equation visualised)
def fig_fusion_v2():
    fig, ax = plt.subplots(figsize=(2.4, 1.4))
    x = np.linspace(-4, 4, 120)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color=TEAL, lw=1.4)
    ax.axhline(0.5, color=GRAY, lw=0.4, ls="--")
    ax.axvline(0, color=GRAY, lw=0.4, ls="--")
    ax.set_xlim(-4, 4); ax.set_ylim(-0.05, 1.10)
    ax.set_xticks([]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6.5)
    ax.text(0.0, 1.05, r"$\sigma$ gate(t,j)", ha="center", fontsize=7.5, color=TEAL, fontweight="bold")
    ax.text(-3.7, 0.10, "→ static\ndominates", fontsize=6.5, color=SLATE, va="bottom")
    ax.text(2.0, 0.85, "→ dynamic\ndominates", fontsize=6.5, color=SLATE)
    save(fig, "fusion_v2")


# --------- 4. Mode×tier β heatmap (compact)
def fig_mode_tier_v2():
    fig, ax = plt.subplots(figsize=(2.4, 1.6))
    beta_per_mode = np.array([-0.187, -0.105, -0.049])
    kappa = np.array([1.0, 1.97, 1.97])
    beta_eff = beta_per_mode[:, None] * kappa[None, :]
    im = ax.imshow(beta_eff, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for m in range(3):
        for k in range(3):
            ax.text(k, m, f"{beta_eff[m,k]:+.2f}", ha="center", va="center",
                     fontsize=7, color="white" if abs(beta_eff[m,k]) > 0.18 else SLATE)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "mid", "high"], fontsize=7)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["car", "tr", "wk"], fontsize=7)
    ax.set_title(r"recovered $\beta_{m,k}$ (mean over T)", fontsize=7.5)
    ax.set_xlabel("tier k", fontsize=7); ax.set_ylabel("mode m", fontsize=7)
    save(fig, "mode_tier_v2")


# --------- 5. Training curve (already wide; just regen at smaller size)
def fig_train_curve_v2():
    log_path = ROOT / "evaluation_outputs" / "paper_a" / "dual_het_seed0.log.json"
    with open(log_path) as f:
        log = json.load(f)
    epoch = log["epoch"]
    train_nll = log["train_nll"]; val_nll = log["val_nll"]; cpc = log["cpc_val"]
    fig, ax1 = plt.subplots(figsize=(2.6, 1.5))
    ax1.plot(epoch, train_nll, color=PURPLE_DK, lw=0.8, alpha=0.7)
    ax1.plot(epoch, val_nll, color=PURPLE_DK, lw=1.3, linestyle="--")
    ax1.set_xlabel("epoch", fontsize=7, color=SLATE)
    ax1.set_ylabel("NLL", fontsize=7, color=PURPLE_DK)
    ax1.tick_params(labelsize=6.5, axis="y", colors=PURPLE_DK)
    ax1.tick_params(labelsize=6.5, axis="x")
    ax1.spines["left"].set_color(PURPLE_DK)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(epoch, cpc, color=GREEN_DK, lw=1.3)
    ax2.set_ylabel("CPC", fontsize=7, color=GREEN_DK)
    ax2.tick_params(labelsize=6.5, axis="y", colors=GREEN_DK)
    ax2.spines["right"].set_color(GREEN_DK)
    ax2.set_ylim(0.25, 0.50)
    ax1.text(0.04, 0.92, f"final CPC = {cpc[-1]:.3f}", transform=ax1.transAxes,
              fontsize=7, color=GREEN_DK, fontweight="bold")
    save(fig, "train_curve_v2")


# --------- 6. Synthetic agents + softmax compact
def fig_agent_choice_v2():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.4), gridspec_kw={"width_ratios": [1.2, 1.8]})
    # left: agent dots
    ax = axes[0]
    np.random.seed(2)
    n = 30
    x = np.random.rand(n); y = np.random.rand(n) * 0.85
    modes = np.random.choice(["car", "transit", "walk"], n, p=[0.6, 0.3, 0.1])
    cmap_mode = {"car": "#3b82f6", "transit": "#f59e0b", "walk": "#10b981"}
    for xi, yi, m in zip(x, y, modes):
        ax.scatter(xi, yi, color=cmap_mode[m], s=18, alpha=0.85,
                    edgecolors="white", linewidths=0.4)
    ax.text(0.5, 0.96, r"$N$ agents $\sim (\pi_m, \pi_k)$",
             ha="center", fontsize=6.5, color=SLATE, transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.axis("off")
    # right: softmax+Gumbel
    ax = axes[1]
    np.random.seed(7)
    n = 14
    util = -1.5 + 3.0 * np.random.rand(n)
    p = np.exp(util) / np.exp(util).sum()
    chosen = int(np.argmax(util + np.random.gumbel(0, 1, n)))
    cols = [PURPLE_DK if i == chosen else GRAY for i in range(n)]
    ax.bar(range(n), p, color=cols, width=0.7, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([0, max(p)])
    ax.set_yticklabels(["0", f"{max(p):.2f}"], fontsize=6)
    ax.text(chosen, p[chosen] + 0.015, "★", ha="center", fontsize=10, color=PURPLE_DK)
    ax.set_xlabel(r"candidate $j$", fontsize=6.5, color=SLATE)
    ax.set_title(r"$P(j|i,t) +$ Gumbel $\to \arg\max$", fontsize=7)
    plt.tight_layout(pad=0.3, w_pad=0.5)
    save(fig, "agent_choice_v2")


# --------- 7. BPR V/C curve compact
def fig_bpr_v2():
    fig, ax = plt.subplots(figsize=(2.4, 1.4))
    vc = np.linspace(0, 1.5, 100)
    mult = 1 + 0.15 * vc**4
    mult_capped = np.minimum(mult, 2.5)
    ax.plot(vc, mult_capped, color=ORANGE_DK, lw=1.6)
    ax.axhline(1.0, color=GRAY, lw=0.4)
    ax.axvline(1.0, color=GRAY, lw=0.4, ls=":")
    ax.set_xlim(0, 1.5); ax.set_ylim(0.95, 2.7)
    ax.set_xticks([0, 0.5, 1, 1.5]); ax.set_yticks([1, 2, 2.5])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("V/C ratio", fontsize=7, color=SLATE)
    ax.set_ylabel(r"$t \,/\, t_0$", fontsize=8, color=SLATE)
    ax.set_title(r"BPR multiplier (cap = 2.5)", fontsize=7.5)
    save(fig, "bpr_v2")


if __name__ == "__main__":
    print("[arch_miniviz_v2] regenerating compact mini-viz ...")
    fig_sage_v2()
    fig_dyn_branch_v2()
    fig_fusion_v2()
    fig_mode_tier_v2()
    fig_train_curve_v2()
    fig_agent_choice_v2()
    fig_bpr_v2()
    print("[arch_miniviz_v2] done.")
