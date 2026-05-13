"""Generate publication-quality architecture figure (HLGX-density, two-row).

Layout:
  Top row    : STAGE 1 — ML TRAINING (encoder → fusion → choice → loss)
  Middle     : TRAINED PARAMETERS bus (shared)
  Bottom row : STAGE 2 — ABM DEPLOYMENT (agents → utility → BPR loop → counterfactual)
  Left col   : INPUT (spans both rows)
  Right col  : OUTPUT (spans both rows)

Each module box contains 2-4 sub-visualizations (graph, hour strip, GRU,
TCN kernels, sigmoid curve, softmax bars, BPR curve, London grids, agent
icons, distribution plots) plus a numbered step badge and inline formula.

Output:
  paper_methods_architecture.png  (300 dpi)
  paper_methods_architecture.pdf  (vector)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Rectangle, FancyArrowPatch,
                                 Circle, Wedge)
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent

PAL = {
    "input":   {"fill": "#D6E4F0", "banner": "#2F5F8C", "border": "#5A8FB8"},
    "encoder": {"fill": "#E2EFD0", "banner": "#446B2A", "border": "#7EA655"},
    "choice":  {"fill": "#D9E8C9", "banner": "#3F7F2F", "border": "#5C9A4F"},
    "loss":    {"fill": "#F8EBC4", "banner": "#A87B1E", "border": "#D4AC4A"},
    "abm":     {"fill": "#F8D5BB", "banner": "#A85928", "border": "#D8814D"},
    "bpr":     {"fill": "#F2C2B5", "banner": "#963F2D", "border": "#C4624C"},
    "param":   {"fill": "#D9D5C0", "banner": "#5F5536", "border": "#8C8254"},
    "output":  {"fill": "#EAE3CC", "banner": "#6B5A2F", "border": "#A89878"},
}


def stage_box(ax, x, y, w, h, banner_text, palette_key, banner_h=0.30):
    p = PAL[palette_key]
    outer = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.5, edgecolor=p["border"], facecolor=p["fill"], zorder=1)
    ax.add_patch(outer)
    banner = FancyBboxPatch((x, y + h - banner_h), w, banner_h,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=0, edgecolor="none", facecolor=p["banner"], zorder=2)
    ax.add_patch(banner)
    cover = Rectangle((x + 0.01, y + h - banner_h), w - 0.02, banner_h * 0.45,
        linewidth=0, facecolor=p["banner"], zorder=2)
    ax.add_patch(cover)
    ax.text(x + w / 2, y + h - banner_h / 2, banner_text,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=3)


def module_box(ax, x, y, w, h, palette_key="input", step=None):
    p = PAL[palette_key]
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.9, edgecolor=p["border"], facecolor="white", zorder=4)
    ax.add_patch(box)
    if step is not None:
        # numbered badge in top-left corner
        ax.add_patch(Circle((x + 0.16, y + h - 0.16), 0.13,
                             facecolor=p["banner"], edgecolor="none", zorder=6))
        ax.text(x + 0.16, y + h - 0.16, str(step),
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", zorder=7)


def big_arrow(ax, x1, y1, x2, y2, color="#444", width=2.5, head=22,
              text=None, fontsize=8, label_offset=(0, 0.08)):
    a = FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=head, linewidth=width,
        color=color, zorder=2)
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1], text,
                ha="center", va="center", fontsize=fontsize, color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=color, linewidth=0.5), zorder=4)


def small_arrow(ax, x1, y1, x2, y2, color="#444", width=1.2, dashed=False,
                text=None, fontsize=6.5, head=10, label_offset=(0, 0.05)):
    a = FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=head, linewidth=width,
        color=color, zorder=2,
        linestyle="--" if dashed else "-")
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1], text, ha="center",
                va="center", fontsize=fontsize, color=color,
                bbox=dict(boxstyle="round,pad=0.13", facecolor="white",
                          edgecolor="none", alpha=0.95), zorder=4)


# ==================================================================
# Module-internal visualizations
# ==================================================================

def draw_grid_tensor(ax, x, y, w, h, n_rows, n_cols, cmap_name="viridis",
                     seed=0):
    rng = np.random.default_rng(seed)
    vals = rng.random((n_rows, n_cols))
    cmap = plt.get_cmap(cmap_name)
    cell_w = w / n_cols; cell_h = h / n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(Rectangle(
                (x + c * cell_w, y + (n_rows - 1 - r) * cell_h),
                cell_w * 0.95, cell_h * 0.95,
                facecolor=cmap(0.2 + 0.6 * vals[r, c]),
                edgecolor="#fff", linewidth=0.3, zorder=5))


def draw_static_graph(ax, cx, cy, scale=1.0, color="#446B2A"):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    nbrs = [(cx + 0.45 * scale * np.cos(a),
             cy + 0.45 * scale * np.sin(a)) for a in angles]
    for nb in nbrs:
        ax.plot([cx, nb[0]], [cy, nb[1]], color="#7EA655",
                linewidth=0.8, zorder=5)
    for nb in nbrs:
        ax.add_patch(Circle(nb, 0.06 * scale, facecolor="#A7C957",
                            edgecolor=color, linewidth=0.6, zorder=6))
    ax.add_patch(Circle((cx, cy), 0.10 * scale, facecolor=color,
                        edgecolor="#1f3614", linewidth=0.7, zorder=7))


def draw_hour_strip(ax, x, y, w, h, seed=42, cmap_name="YlOrRd",
                    show_ticks=False):
    rng = np.random.default_rng(seed)
    vals = 0.3 + 0.7 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 24) + rng.random()))
    cmap = plt.get_cmap(cmap_name)
    cell_w = w / 24
    for t in range(24):
        ax.add_patch(Rectangle((x + t * cell_w, y),
                                cell_w * 0.95, h,
                                facecolor=cmap(0.2 + 0.6 * vals[t]),
                                edgecolor="#fff", linewidth=0.2, zorder=5))
    if show_ticks:
        for t in [0, 8, 12, 18, 23]:
            ax.text(x + t * cell_w + cell_w / 2, y - 0.04,
                    f"{t}", ha="center", va="top",
                    fontsize=5.0, color="#666")


def draw_gru_cell(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#E2EFD0",
                            edgecolor="#446B2A", linewidth=0.7, zorder=5))
    # input arrow
    ax.add_patch(FancyArrowPatch((x - 0.1, y + h * 0.35),
                                  (x + 0.05, y + h * 0.35),
                                  arrowstyle="-|>", mutation_scale=7,
                                  linewidth=0.6, color="#444", zorder=6))
    ax.text(x - 0.13, y + h * 0.35, r"$x_t$", ha="right", va="center",
            fontsize=5.5, color="#333")
    ax.add_patch(FancyArrowPatch((x - 0.1, y + h * 0.75),
                                  (x + 0.05, y + h * 0.75),
                                  arrowstyle="-|>", mutation_scale=7,
                                  linewidth=0.6, color="#444", zorder=6))
    ax.text(x - 0.13, y + h * 0.75, r"$h_{t-1}$", ha="right", va="center",
            fontsize=5.5, color="#333")
    ax.add_patch(FancyArrowPatch((x + w - 0.05, y + h * 0.5),
                                  (x + w + 0.10, y + h * 0.5),
                                  arrowstyle="-|>", mutation_scale=7,
                                  linewidth=0.6, color="#444", zorder=6))
    ax.text(x + w + 0.13, y + h * 0.5, r"$h_t$", ha="left", va="center",
            fontsize=5.5, color="#333")
    ax.text(x + w / 2, y + h / 2, "GRU",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#446B2A", zorder=7)


def draw_tcn_kernels(ax, x, y, w, h):
    cell_w = w / 9
    for i, (k, color) in enumerate([(3, "#A7C957"), (5, "#7EA655"),
                                     (7, "#446B2A")]):
        row_y = y + (2 - i) * (h / 3 + 0.015)
        for t in range(9):
            ax.add_patch(Rectangle((x + t * cell_w, row_y),
                                   cell_w * 0.92, h / 3 - 0.025,
                                   facecolor="#f5f5f5", edgecolor="#aaa",
                                   linewidth=0.4, zorder=5))
        ks = (9 - k) // 2
        for t in range(ks, ks + k):
            ax.add_patch(Rectangle((x + t * cell_w, row_y),
                                   cell_w * 0.92, h / 3 - 0.025,
                                   facecolor=color, edgecolor=color,
                                   linewidth=0.4, zorder=6))
        ax.text(x + w + 0.04, row_y + (h / 3 - 0.025) / 2,
                f"k={k}", ha="left", va="center",
                fontsize=5.5, color=color, fontweight="bold")


def draw_sigmoid_gate(ax, cx, cy, w, h):
    xs = np.linspace(-3, 3, 50)
    ys = 1 / (1 + np.exp(-xs))
    pxs = cx - w / 2 + (xs + 3) / 6 * w
    pys = cy - h / 2 + ys * h
    ax.plot(pxs, pys, color="#3F7F2F", linewidth=1.2, zorder=5)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                            edgecolor="#aaa", linewidth=0.4, zorder=5))


def draw_softmax_bars(ax, x, y, w, h, n=8, seed=1, color="#3F7F2F"):
    rng = np.random.default_rng(seed)
    raw = rng.random(n)
    raw[2] = 1.5; raw[5] = 1.2
    p = np.exp(raw) / np.exp(raw).sum()
    bar_w = w / n
    for i in range(n):
        ax.add_patch(Rectangle((x + i * bar_w, y),
                                bar_w * 0.85, p[i] * h * 4,
                                facecolor=color, edgecolor="none",
                                zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                            edgecolor="#aaa", linewidth=0.4, zorder=4))


def draw_bpr_curve(ax, x, y, w, h):
    vc = np.linspace(0, 1.5, 50)
    mult = 1 + 0.15 * vc ** 4
    pxs = x + vc / 1.5 * w
    pys = y + (mult - 1) / 2.5 * h
    pys = np.clip(pys, y, y + h)
    ax.plot(pxs, pys, color="#963F2D", linewidth=1.5, zorder=5)
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                            edgecolor="#aaa", linewidth=0.4, zorder=4))


def draw_london_grid(ax, x, y, w, h, highlight_idx=None,
                     cmap_name="YlGn", seed=7):
    rng = np.random.default_rng(seed)
    n = 8
    vals = rng.random((n, n))
    cell_w = w / n; cell_h = h / n
    cmap = plt.get_cmap(cmap_name)
    for r in range(n):
        for c in range(n):
            d = np.hypot(r - n / 2 + 0.5, c - n / 2 + 0.5)
            if d > n / 2 - 0.1:
                continue
            color = cmap(0.2 + 0.6 * vals[r, c])
            if highlight_idx == (r, c):
                color = "#D8814D"
            ax.add_patch(Rectangle(
                (x + c * cell_w, y + (n - 1 - r) * cell_h),
                cell_w * 0.95, cell_h * 0.95,
                facecolor=color, edgecolor="#fff",
                linewidth=0.3, zorder=5))
    if highlight_idx:
        r, c = highlight_idx
        ax.add_patch(Rectangle(
            (x + c * cell_w - 0.01, y + (n - 1 - r) * cell_h - 0.01),
            cell_w + 0.02, cell_h + 0.02, fill=False,
            edgecolor="#963F2D", linewidth=1.0, zorder=6))


def draw_agent_icons(ax, x, y, w, h, n=4):
    """Row of agent icons with attribute labels."""
    spacing = w / n
    for i in range(n):
        cx = x + spacing * (i + 0.5)
        cy = y + h * 0.7
        ax.add_patch(Circle((cx, cy), 0.07, facecolor="#D8814D",
                             edgecolor="#A85928", linewidth=0.6, zorder=6))
        ax.text(cx, cy - 0.13, f"$agent_{i+1}$",
                ha="center", va="top", fontsize=5.5, color="#A85928")
        ax.text(cx, cy - 0.23, f"$(i_{i+1}, t, m, k)$",
                ha="center", va="top", fontsize=4.8, color="#666",
                style="italic")


def draw_gumbel_dist(ax, x, y, w, h):
    """Skewed bell curve representing Gumbel distribution."""
    xs = np.linspace(-2, 4, 50)
    pdf = np.exp(-(xs + np.exp(-xs)))
    pxs = x + (xs + 2) / 6 * w
    pys = y + pdf / pdf.max() * h * 0.95
    ax.fill_between(pxs, y, pys, color="#A85928", alpha=0.3,
                     zorder=5, edgecolor="none")
    ax.plot(pxs, pys, color="#A85928", linewidth=1.0, zorder=6)
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                            edgecolor="#aaa", linewidth=0.4, zorder=4))


def draw_pred_vs_obs(ax, x, y, w, h, n=10):
    """Two paired bar groups: predicted P (filled) vs observed F/O (outline)."""
    rng = np.random.default_rng(2)
    pred = rng.random(n); pred[3] = 1.5; pred[6] = 1.2
    pred = np.exp(pred) / np.exp(pred).sum()
    obs = pred + 0.3 * (rng.random(n) - 0.5) * pred  # noisy version
    obs = np.clip(obs, 0, None); obs = obs / obs.sum()
    bar_w = w / n
    for i in range(n):
        # predicted (filled green)
        ax.add_patch(Rectangle((x + i * bar_w, y),
                                bar_w * 0.40, pred[i] * h * 4,
                                facecolor="#3F7F2F", edgecolor="none",
                                zorder=5))
        # observed (outlined orange)
        ax.add_patch(Rectangle((x + i * bar_w + bar_w * 0.45, y),
                                bar_w * 0.40, obs[i] * h * 4,
                                facecolor="none", edgecolor="#A87B1E",
                                linewidth=0.8, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                            edgecolor="#aaa", linewidth=0.4, zorder=4))


# ==================================================================
# MAIN
# ==================================================================
def main():
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_aspect("equal")

    # Title
    fig.suptitle(
        "Figure 1 — ML-ABM Hybrid Framework for Counterfactual Commuting Analysis",
        fontsize=13, fontweight="bold", y=0.985, color="#1a1a1a")

    # ============== INPUT (left, full height) ==============
    stage_box(ax, 0.20, 0.55, 1.95, 9.65, "INPUT", "input")

    inputs = [
        (r"$X_{static}$", "(N, 22)\nPOI / employ /\npopulation", 9.05),
        (r"$X_{dynamic}$", "(T, N, 5)\ncongestion +\nhourly inflow", 8.20),
        (r"$\mathcal{E}$", "(2, E)\n1km grid\nadjacency", 7.30),
        (r"$t_{ij}^{(m)}$", "car / transit /\nwalk minutes", 6.40),
        (r"$\log d_{ij}$", "(N, N)\ndistance", 5.50),
        (r"$\pi_m(i,j)$", "pair-level\nmode share\n(distance-aware)", 4.55),
        (r"$\pi_k(i)$", "(N, K)\nIMD income\ntier weights", 3.55),
        (r"OccMatch$_{ij}$", "(N, N)\nSOC9 cosine\nsimilarity", 2.55),
        (r"$F_{ij,t}$  (label)", "(T, N, N)\nobserved\nhourly OD", 1.40),
    ]
    for name, body, y in inputs:
        ax.text(1.175, y + 0.06, name, ha="center", va="center",
                fontsize=8.2, fontweight="bold",
                color="#1a3d6e" if "F_{ij,t}" not in name else "#A87B1E",
                zorder=5)
        ax.text(1.175, y - 0.28, body, ha="center", va="center",
                fontsize=6.0, color="#333", zorder=5, linespacing=1.15)
        ax.add_line(Line2D([0.40, 1.95], [y - 0.55, y - 0.55],
                            color="#aacde2", linewidth=0.4, zorder=4))

    # Bus arrow INPUT → STAGE 1
    big_arrow(ax, 2.20, 7.85, 2.55, 7.85, color="#5A8FB8", width=2.8, head=22)

    # ============== STAGE 1 — TOP ROW ==============
    stage_box(ax, 2.55, 5.45, 15.10, 4.75,
              "STAGE 1 — ML TRAINING (supervised: backprop on cross-entropy ≡ multinomial NLL)",
              "encoder")

    # ===== Module 1: Static branch =====
    module_box(ax, 2.80, 5.95, 2.40, 3.40, "encoder", step=1)
    ax.text(4.00, 9.05, "Static branch", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#446B2A", zorder=7)
    ax.text(4.00, 8.85, "$h^{stat}_j = $ GraphSAGE × 2",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Sub-element a: 1 km grid map
    draw_london_grid(ax, 2.95, 7.95, 0.85, 0.80, cmap_name="Greens",
                     seed=4)
    ax.text(3.375, 7.85, "$X_{static}$ on grid",
            ha="center", va="top", fontsize=5.5, color="#333", style="italic")
    # Sub-element b: graph
    draw_static_graph(ax, 4.65, 8.30, scale=0.55)
    ax.text(4.65, 7.85, "neighbour aggreg.",
            ha="center", va="top", fontsize=5.5, color="#333", style="italic")
    # Formula
    ax.text(4.00, 7.45,
            r"$h^{(\ell+1)}_i = \sigma(W_s h^{(\ell)}_i + W_n \frac{1}{|\mathcal{N}|}\sum_{j} h^{(\ell)}_j)$",
            ha="center", va="center", fontsize=5.7, color="#222")
    ax.text(4.00, 7.10, "$\\ell = 0, 1$",
            ha="center", va="center", fontsize=5.5, color="#666",
            style="italic")
    # Output tensor
    draw_grid_tensor(ax, 3.10, 6.40, 1.80, 0.55, 4, 16, "Greens", seed=10)
    ax.text(4.00, 6.32, r"$h^{stat}$ (N, 32)", ha="center", va="top",
            fontsize=6, color="#446B2A", fontweight="bold")
    ax.text(5.16, 5.97, "Eq. (2)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Module 2: Dynamic branch =====
    module_box(ax, 5.40, 5.95, 3.40, 3.40, "encoder", step=2)
    ax.text(7.10, 9.05, "Dynamic branch", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#446B2A", zorder=7)
    ax.text(7.10, 8.85, "per-hour SAGE → GRU + multi-scale TCN",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Hour strip
    draw_hour_strip(ax, 5.55, 8.40, 3.10, 0.20, seed=12, show_ticks=False)
    ax.text(7.10, 8.32, "$\\tilde{h}_t$ : 24 hourly states",
            ha="center", va="top", fontsize=5.5, color="#333", style="italic")
    # GRU cell
    draw_gru_cell(ax, 5.85, 7.65, 1.20, 0.50)
    ax.text(6.45, 7.55, "sequential memory",
            ha="center", va="top", fontsize=5.0, color="#666", style="italic")
    # TCN kernels
    draw_tcn_kernels(ax, 7.40, 7.45, 1.05, 0.70)
    ax.text(7.92, 7.40, "multi-scale (3,5,7)",
            ha="center", va="top", fontsize=5.0, color="#666", style="italic")
    # Output tensor (T, N, d) - shown as 3D-like
    for i in range(3):
        ax.add_patch(Rectangle((5.85 + i * 0.04, 6.60 + i * 0.06),
                               2.30, 0.40, facecolor="#E2EFD0",
                               edgecolor="#446B2A", linewidth=0.4, zorder=5))
    draw_grid_tensor(ax, 5.93, 6.55, 2.20, 0.30, 2, 24, "YlGn", seed=20)
    ax.text(7.05, 6.42, r"$h^{dyn}$ (T, N, 32)", ha="center", va="top",
            fontsize=6, color="#446B2A", fontweight="bold")
    ax.text(8.76, 5.97, "Eq. (2)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Module 3: Gated fusion =====
    module_box(ax, 8.95, 5.95, 2.20, 3.40, "encoder", step=3)
    ax.text(10.05, 9.05, "Gated fusion", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#446B2A", zorder=7)
    ax.text(10.05, 8.85, "weighted blend + z-norm",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Sigmoid gate
    draw_sigmoid_gate(ax, 10.05, 8.30, 0.65, 0.40)
    ax.text(10.55, 8.30, r"$g_{j,t}$", ha="left", va="center",
            fontsize=7, color="#3F7F2F", fontweight="bold")
    ax.text(10.05, 7.90, "gate per (t, j)",
            ha="center", va="top", fontsize=5.5, color="#666", style="italic")
    # Combination formula
    ax.text(10.05, 7.55,
            r"$h_{j,t} = g \cdot h^{stat}_j + (1{-}g) \cdot h^{dyn}_{j,t}$",
            ha="center", va="center", fontsize=5.6, color="#222")
    ax.text(10.05, 7.30, "linear → scalar → z-norm",
            ha="center", va="center", fontsize=5.3, color="#666",
            style="italic")
    # V_jt heat strip output
    draw_grid_tensor(ax, 9.20, 6.55, 1.70, 0.30, 2, 24, "viridis", seed=99)
    ax.text(10.05, 6.42, r"$V_{j,t}$ (T, N)", ha="center", va="top",
            fontsize=6.5, color="#3F7F2F", fontweight="bold")
    ax.text(10.05, 6.20,
            "destination attractiveness", ha="center", va="top",
            fontsize=5.3, color="#666", style="italic")
    ax.text(11.11, 5.97, "Eq. (3, 4)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Module 4: Behavioural choice layer =====
    module_box(ax, 11.30, 5.95, 3.20, 3.40, "choice", step=4)
    ax.text(12.90, 9.05, "Behavioural choice layer",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#3F7F2F", zorder=7)
    ax.text(12.90, 8.85, "mode × tier mixture multinomial logit",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Utility formula
    ax.text(12.90, 8.55,
            r"$U^{(m,k)}_{ij,t} = \alpha V_{j,t} + \beta_{t,m,k}\, t^{(m)}_{ij}"
            r" + \gamma \log d_{ij} + \delta_k\, \mathrm{OM}_{ij}$",
            ha="center", va="center", fontsize=5.5, color="#222")
    ax.text(12.90, 8.30, r"$\beta_{t,m,k} = \beta_{t,m} \cdot \kappa_k$",
            ha="center", va="center", fontsize=5.5, color="#3F7F2F",
            style="italic")
    # Mixture weight visualization
    rng = np.random.default_rng(7)
    pi_m = np.array([0.44, 0.45, 0.11])  # car/transit/walk
    pi_k = np.array([0.27, 0.33, 0.40])
    bar_w = 0.20
    for i, (v, lbl, col) in enumerate(zip(pi_m,
                                ["car", "tran.", "walk"],
                                ["#A85928", "#3F7F2F", "#7EA655"])):
        ax.add_patch(Rectangle((11.65 + i * 0.30, 7.35),
                                bar_w, v * 0.7, facecolor=col,
                                edgecolor="none", zorder=5))
        ax.text(11.65 + i * 0.30 + bar_w / 2, 7.30, lbl,
                ha="center", va="top", fontsize=5, color="#666")
    ax.text(11.95, 8.0, r"$\pi_m(i,j)$", ha="center", va="bottom",
            fontsize=5.5, color="#666", style="italic")
    for i, (v, lbl, col) in enumerate(zip(pi_k,
                                ["T1", "T2", "T3"],
                                ["#85B96A", "#7EA655", "#446B2A"])):
        ax.add_patch(Rectangle((12.75 + i * 0.30, 7.35),
                                bar_w, v * 0.7, facecolor=col,
                                edgecolor="none", zorder=5))
        ax.text(12.75 + i * 0.30 + bar_w / 2, 7.30, lbl,
                ha="center", va="top", fontsize=5, color="#666")
    ax.text(13.05, 8.0, r"$\pi_k(i)$", ha="center", va="bottom",
            fontsize=5.5, color="#666", style="italic")
    # Softmax formula
    ax.text(13.85, 7.55,
            r"$P(j|i,t) = \!\!\sum_{m,k}\!\! \pi_m \pi_k$",
            ha="center", va="center", fontsize=5.3, color="#222")
    ax.text(13.85, 7.30,
            r"$\cdot \mathrm{softmax}_j(U^{(m,k)})$",
            ha="center", va="center", fontsize=5.3, color="#3F7F2F",
            fontweight="bold")
    # Output P(j|i,t) bars
    draw_softmax_bars(ax, 11.55, 6.55, 2.85, 0.45, n=15, seed=8)
    ax.text(12.975, 6.45, r"$P(j|i,t)$ over $j = 1..N$",
            ha="center", va="top", fontsize=5.8, color="#3F7F2F",
            fontweight="bold")
    ax.text(14.46, 5.97, "Eq. (5–8)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Module 5: Cross-entropy loss =====
    module_box(ax, 14.70, 5.95, 2.85, 3.40, "loss", step=5)
    ax.text(16.125, 9.05, "Cross-entropy loss",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#A87B1E", zorder=7)
    ax.text(16.125, 8.85, "= multinomial NLL (Bentz \\& Merunka 2000)",
            ha="center", va="center", fontsize=6.0, color="#222",
            style="italic")
    # Predicted vs observed bars
    draw_pred_vs_obs(ax, 14.85, 7.55, 2.55, 0.85, n=12)
    ax.text(16.125, 8.50, "predicted P  vs  observed F/O",
            ha="center", va="top", fontsize=5.8, color="#333",
            style="italic")
    # Mini legend below bars
    ax.add_patch(Rectangle((14.95, 7.40), 0.15, 0.10,
                            facecolor="#3F7F2F", edgecolor="none"))
    ax.text(15.13, 7.45, "predicted P",
            ha="left", va="center", fontsize=5.5, color="#333")
    ax.add_patch(Rectangle((15.95, 7.40), 0.15, 0.10,
                            facecolor="none", edgecolor="#A87B1E",
                            linewidth=0.8))
    ax.text(16.13, 7.45, "observed F/O",
            ha="left", va="center", fontsize=5.5, color="#333")
    # Formula
    ax.text(16.125, 7.15,
            r"$\mathcal{L}(\theta) = -\!\!\sum_{i,j,t}\! F_{ij,t} \log P(j|i,t;\theta)$",
            ha="center", va="center", fontsize=6.5, color="#222")
    ax.text(16.125, 6.85, r"$\equiv -\log \mathrm{Multinomial}(F | O, P)$",
            ha="center", va="center", fontsize=6.0, color="#666",
            style="italic")
    # Optimiser
    ax.text(16.125, 6.55, "AdamW backprop",
            ha="center", va="center", fontsize=6.5, color="#A87B1E",
            fontweight="bold")
    ax.text(16.125, 6.30,
            r"updates $\Theta_{enc}, \beta, \gamma, \delta, \kappa$",
            ha="center", va="center", fontsize=5.8, color="#A87B1E")
    ax.text(17.51, 5.97, "Eq. (1, 11)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Stage 1 horizontal flow arrows (clean, single between modules) =====
    small_arrow(ax, 5.20, 7.65, 5.40, 7.65, color="#446B2A", width=1.5,
                head=11)
    small_arrow(ax, 8.80, 7.65, 8.95, 7.65, color="#446B2A", width=1.5,
                head=11, text=r"$h^{stat},h^{dyn}$", fontsize=5.5,
                label_offset=(0, 0.18))
    small_arrow(ax, 11.15, 7.65, 11.30, 7.65, color="#446B2A", width=1.5,
                head=11, text=r"$V_{j,t}$", fontsize=6, label_offset=(0, 0.12))
    small_arrow(ax, 14.50, 7.65, 14.70, 7.65, color="#3F7F2F", width=1.5,
                head=11, text=r"$P(j|i,t)$", fontsize=6, label_offset=(0, 0.12))

    # Loss → params bus (down)
    small_arrow(ax, 16.125, 5.95, 16.125, 5.45, color="#A87B1E", width=1.5,
                head=11)

    # ============== TRAINED PARAMETERS BUS ==============
    stage_box(ax, 2.55, 4.55, 15.10, 0.85,
              "TRAINED PARAMETERS  (shared between Stage 1 and Stage 2 — no re-fit at deployment)",
              "param", banner_h=0.28)
    ax.text(10.10, 4.85,
            r"$\Theta_{enc}$ ($\approx$33,000 weights)   |   $\beta_{t,m}$   |   $\gamma$   |   $\delta_k$   |   $\kappa_k$",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1a1a1a", zorder=5)

    # ============== STAGE 2 — BOTTOM ROW ==============
    stage_box(ax, 2.55, 0.55, 15.10, 3.85,
              "STAGE 2 — ABM DEPLOYMENT (deterministic deployment with BPR congestion feedback loop)",
              "abm")

    # ===== Module 6: Agent population =====
    module_box(ax, 2.80, 1.10, 2.30, 2.85, "abm", step=6)
    ax.text(3.95, 3.65, "Synthetic agents",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#A85928", zorder=7)
    ax.text(3.95, 3.45, "sample from $(\\pi_m, \\pi_k)$",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    draw_agent_icons(ax, 2.95, 2.40, 2.05, 0.85, n=4)
    ax.text(3.95, 2.30, "each agent: $(i_n, t_n, m_n, k_n)$",
            ha="center", va="top", fontsize=5.7, color="#333",
            style="italic")
    # Per-agent attribute distribution
    ax.text(3.95, 2.05, "$N_\\mathrm{agents}$ commuters", ha="center",
            va="top", fontsize=6.0, color="#A85928", fontweight="bold")
    # Attribute hint
    ax.text(3.95, 1.75,
            "$i_n$: home grid\n$t_n$: hour\n$m_n$: mode (car/tran./walk)\n$k_n$: income tier",
            ha="center", va="top", fontsize=5.3, color="#666",
            linespacing=1.3)

    # ===== Module 7: Utility eval (free-flow) =====
    module_box(ax, 5.30, 1.10, 3.10, 2.85, "abm", step=7)
    ax.text(6.85, 3.65, "Utility eval (free-flow)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#A85928", zorder=7)
    ax.text(6.85, 3.45, "evaluate $U_{nj}$ for each candidate $j$",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Formula
    ax.text(6.85, 3.15,
            r"$U_{nj} = V_{j,t_n} + \beta\, t^{free}_{i_n j}$",
            ha="center", va="center", fontsize=6.5, color="#222")
    ax.text(6.85, 2.90,
            r"$+ \gamma \log d_{i_n j} + \delta\, \mathrm{OM}_{i_n j} + \varepsilon_{nj}$",
            ha="center", va="center", fontsize=6.5, color="#222")
    # Gumbel noise distribution
    draw_gumbel_dist(ax, 5.50, 1.95, 1.10, 0.55)
    ax.text(6.05, 1.88, r"$\varepsilon \sim \mathrm{Gumbel}(0,1)$",
            ha="center", va="top", fontsize=5.5, color="#A85928",
            style="italic")
    # Argmax visualization with mini bars
    draw_softmax_bars(ax, 6.85, 2.05, 1.40, 0.50, n=8, seed=15,
                      color="#D8814D")
    ax.text(7.55, 1.95, r"$j^*_n = \arg\max_j U_{nj}$",
            ha="center", va="top", fontsize=5.5, color="#A85928",
            fontweight="bold")
    # Gumbel-max trick note
    ax.text(6.85, 1.45,
            "Gumbel-max trick: aggregating $j^*_n$",
            ha="center", va="center", fontsize=5.3, color="#666",
            style="italic")
    ax.text(6.85, 1.30,
            "reproduces softmax (7) in expectation",
            ha="center", va="center", fontsize=5.3, color="#666",
            style="italic")
    ax.text(8.36, 1.12, "Eq. (9)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Module 8: BPR feedback loop =====
    module_box(ax, 8.60, 1.10, 3.40, 2.85, "bpr", step=8)
    ax.text(10.30, 3.65, "BPR congestion feedback",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#963F2D", zorder=7)
    ax.text(10.30, 3.45, "iterate aggregate → BPR → re-evaluate",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Aggregate inflow (sub-element a)
    ax.text(9.30, 3.15, "(a) aggregate", ha="center", va="center",
            fontsize=6.0, color="#963F2D", fontweight="bold")
    draw_softmax_bars(ax, 8.75, 2.65, 1.10, 0.40, n=8, seed=42,
                      color="#D8814D")
    ax.text(9.30, 2.55, r"$\widehat{F}_{j,t}$",
            ha="center", va="top", fontsize=6, color="#963F2D",
            style="italic")
    # BPR curve (sub-element b)
    ax.text(11.10, 3.15, "(b) BPR multiplier", ha="center", va="center",
            fontsize=6.0, color="#963F2D", fontweight="bold")
    draw_bpr_curve(ax, 10.55, 2.40, 1.10, 0.65)
    ax.text(11.10, 2.30, "$V/C$ ratio",
            ha="center", va="top", fontsize=5.0, color="#666")
    ax.text(10.50, 2.70, "mult.", ha="center", va="center",
            fontsize=5.0, color="#666", rotation=90)
    # Formula
    ax.text(10.30, 2.05,
            r"$t^{cong}_{ij,t} = t^{free}_{ij} \cdot (1 + 0.15(\widehat{F}/C)^4)$",
            ha="center", va="center", fontsize=6.0, color="#222")
    # Iteration arrow visual
    ax.text(10.30, 1.70, "(c) re-evaluate $U$ with $t^{cong}$",
            ha="center", va="center", fontsize=5.8, color="#963F2D")
    ax.text(10.30, 1.50, "(d) iterate until $\\widehat{F}$ converges",
            ha="center", va="center", fontsize=5.8, color="#963F2D")
    ax.text(10.30, 1.30,
            "(typically 5–10 iterations)",
            ha="center", va="center", fontsize=5.3, color="#666",
            style="italic")
    # Iteration loop indicator (curved arrow)
    iter_loop = FancyArrowPatch((11.85, 1.35), (11.85, 2.35),
                                  arrowstyle="-|>", mutation_scale=10,
                                  linewidth=1.4, color="#963F2D",
                                  zorder=6, connectionstyle="arc3,rad=0.5",
                                  linestyle="--")
    ax.add_patch(iter_loop)
    ax.text(12.05, 1.85, "loop", ha="left", va="center",
            fontsize=5.5, color="#963F2D", fontweight="bold",
            style="italic")
    ax.text(11.96, 1.12, "Eq. (10)", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Module 9: Counterfactual =====
    module_box(ax, 12.20, 1.10, 5.30, 2.85, "abm", step=9)
    ax.text(14.85, 3.65, "Counterfactual scenarios",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#A85928", zorder=7)
    ax.text(14.85, 3.45, "intervene on inputs, re-encode, re-deploy",
            ha="center", va="center", fontsize=6.2, color="#222",
            style="italic")
    # Scenario A label + grids
    ax.text(13.00, 3.15, "Scenario A (spatial)",
            ha="center", va="center", fontsize=6.0, color="#A85928",
            fontweight="bold")
    draw_london_grid(ax, 12.40, 2.20, 0.55, 0.85, highlight_idx=(2, 4),
                     cmap_name="YlGn", seed=33)
    ax.text(12.675, 2.15, "before", ha="center", va="top",
            fontsize=5.3, color="#444")
    draw_london_grid(ax, 13.05, 2.20, 0.55, 0.85, highlight_idx=(2, 4),
                     cmap_name="YlOrRd", seed=33)
    ax.text(13.325, 2.15, "after OOC", ha="center", va="top",
            fontsize=5.3, color="#963F2D")
    ax.text(13.00, 1.90,
            r"$\mathrm{do}(X_{static}) = +65{,}000$ jobs",
            ha="center", va="top", fontsize=5.6, color="#A85928")
    # Scenario B label + visual
    ax.text(15.40, 3.15, "Scenario B (temporal)",
            ha="center", va="center", fontsize=6.0, color="#A85928",
            fontweight="bold")
    # Show two hour strips: before / after peak shift
    draw_hour_strip(ax, 14.55, 2.85, 1.65, 0.15, seed=12,
                    cmap_name="YlOrRd")
    ax.text(15.375, 2.78, "baseline (peak-heavy)",
            ha="center", va="top", fontsize=5.0, color="#444",
            style="italic")
    draw_hour_strip(ax, 14.55, 2.40, 1.65, 0.15, seed=22,
                    cmap_name="Greens")
    ax.text(15.375, 2.33, "after flexible-hours",
            ha="center", va="top", fontsize=5.0, color="#3F7F2F",
            style="italic")
    ax.text(15.375, 2.05,
            r"$\mathrm{do}(X_{dyn})$: peak ×0.5 / shoulder ×1.5",
            ha="center", va="center", fontsize=5.6, color="#A85928")
    # Pipeline note
    ax.text(14.85, 1.60,
            "Both scenarios: re-encode V_jt → re-run agents (steps 6-8) → BPR converge",
            ha="center", va="center", fontsize=5.4, color="#666",
            style="italic")
    ax.text(14.85, 1.35,
            r"Reports $\Delta \widehat{F}_{ij,t}$, $\Delta A_i$, $\Delta$Gini, $\Delta$Palma",
            ha="center", va="center", fontsize=5.8, color="#A85928",
            fontweight="bold")
    ax.text(17.46, 1.12, "§ 2.9", ha="right", va="bottom",
            fontsize=5.8, color="#666", style="italic")

    # ===== Stage 2 horizontal flow arrows =====
    small_arrow(ax, 5.10, 2.50, 5.30, 2.50, color="#A85928", width=1.5,
                head=11)
    small_arrow(ax, 8.40, 2.50, 8.60, 2.50, color="#A85928", width=1.5,
                head=11, text=r"$j^*$", fontsize=6.5,
                label_offset=(0, 0.12))
    small_arrow(ax, 12.00, 2.50, 12.20, 2.50, color="#963F2D", width=1.5,
                head=11, text=r"$\widehat{F}^*, t^{cong,*}$", fontsize=6,
                label_offset=(0, 0.12))

    # Trained params → Stage 2 connector
    big_arrow(ax, 10.10, 4.55, 10.10, 3.95, color="#5F5536",
              width=2.5, head=20, text=r"trained Θ, β, γ, δ, κ",
              fontsize=8, label_offset=(0.1, 0))

    # ============== OUTPUT (right, full height) ==============
    stage_box(ax, 17.95, 0.55, 1.90, 9.65, "OUTPUT", "output")

    outputs = [
        (r"$\widehat{F}_{ij,t}$", "predicted hourly\nOD flow tensor", 9.05),
        (r"$A_i$", "Hansen accessibility\nEq. (12)", 8.20),
        ("Gini, Palma", "by income tier\n(inequality)", 7.30),
        (r"$\Delta \widehat{F}, \Delta A_i$",
         "counterfactual\nscenario effects", 6.30),
        (r"$\beta_{t,m}$",
         "mode-specific\ntime sensitivity\n(VOT per mode)", 5.10),
        (r"$\delta_k, \kappa_k$",
         "tier-specific\nlabour-match +\ntier scaling", 3.85),
        (r"$\gamma$",
         "distance-decay\ncoefficient", 2.65),
        (r"$\Theta_{enc}$",
         "encoder weights\n(transferable\nto Beijing)", 1.50),
    ]
    for name, body, y in outputs:
        ax.text(18.90, y + 0.06, name, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color="#6B5A2F", zorder=5)
        ax.text(18.90, y - 0.32, body, ha="center", va="center",
                fontsize=5.8, color="#333", zorder=5, linespacing=1.15)
        ax.add_line(Line2D([18.10, 19.70], [y - 0.65, y - 0.65],
                            color="#c8b88f", linewidth=0.4, zorder=4))

    # Bus arrow Stage 2 → OUTPUT
    big_arrow(ax, 17.50, 2.50, 17.95, 2.50, color="#A89878",
              width=2.8, head=22)

    # ============== LEGEND (bottom) ==============
    items = [
        ("Input / Output", PAL["input"]["fill"], PAL["input"]["border"]),
        ("ML encoder", PAL["encoder"]["fill"], PAL["encoder"]["border"]),
        ("Behavioural choice", PAL["choice"]["fill"], PAL["choice"]["border"]),
        ("Loss / training", PAL["loss"]["fill"], PAL["loss"]["border"]),
        ("ABM agents", PAL["abm"]["fill"], PAL["abm"]["border"]),
        ("BPR feedback", PAL["bpr"]["fill"], PAL["bpr"]["border"]),
        ("Trained params", PAL["param"]["fill"], PAL["param"]["border"]),
    ]
    legend_y = 0.10
    x_off = 0.50
    for txt, fc, ec in items:
        ax.add_patch(Rectangle((x_off, legend_y), 0.30, 0.16,
                                facecolor=fc, edgecolor=ec, linewidth=0.7))
        ax.text(x_off + 0.36, legend_y + 0.08, txt, va="center",
                fontsize=7, color="#333")
        x_off += 2.65

    plt.tight_layout()
    out_png = ROOT / "paper_methods_architecture.png"
    out_pdf = ROOT / "paper_methods_architecture.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png.name}  ({out_png.stat().st_size // 1024} KB)")
    print(f"saved {out_pdf.name}  ({out_pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
