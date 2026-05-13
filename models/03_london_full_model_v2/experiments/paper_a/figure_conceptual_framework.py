"""F0 Conceptual framework figure for path A reframe.

Shows the end-to-end pipeline:
  Inputs (Census + BRES + OSMnx + WebTAG)
    -> Calibration (inverse training: GNN + RUM head)
    -> Counterfactual forward (do(X) intervention + BPR equilibrium)
    -> Agent-level heterogeneity layer (per-agent beta_n)
    -> Outputs (per-scenario flow / accessibility / inequality)

Designed as a single landscape PNG suitable for a paper figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def add_box(ax, x, y, w, h, text, *, fc="#ffffff", ec="#222", lw=1.0, fontsize=8.5,
            bold=False, italic=False, ha="center", va="center"):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    style = "normal"
    if bold and italic:
        style = "italic"
    elif italic:
        style = "italic"
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fontsize,
            fontstyle=style, fontweight=weight, wrap=True)


def add_arrow(ax, x1, y1, x2, y2, *, lw=1.2, color="#222", style="-|>",
              connectionstyle="arc3,rad=0", label=None, label_offset=(0, 0)):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=14,
                           linewidth=lw, color=color,
                           connectionstyle=connectionstyle)
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, fontsize=7.5, ha="center", color=color)


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # === LANE 1: INPUT DATA (left) ===
    ax.text(0.4, 9.5, "INPUT DATA", fontsize=10, fontweight="bold", color="#445")
    add_box(ax, 0.3, 7.7, 2.6, 1.5,
            "London Census 2021\nfull OD table\n(1.21M commuters)\nNOMIS WU03EW 2011\n(2.14M commuters)",
            fc="#e8f1f9", fontsize=8)
    add_box(ax, 0.3, 5.9, 2.6, 1.5,
            "Node features\n22 dim per 1km grid:\nBRES 2024 sectors,\nKS101 population, OSM POI,\nTfL stations, IMD",
            fc="#e8f1f9", fontsize=8)
    add_box(ax, 0.3, 4.1, 2.6, 1.5,
            "OSMnx free-flow\ntravel time\n+ haversine distance",
            fc="#e8f1f9", fontsize=8)
    add_box(ax, 0.3, 2.3, 2.6, 1.5,
            "WebTAG TAG A1.3\nVOT central + spread\n(Wardman 2014\nmeta-analysis)",
            fc="#fff2cc", fontsize=8)

    # === LANE 2: CALIBRATION (middle-left) ===
    ax.text(3.7, 9.5, "CALIBRATION (inverse training)",
            fontsize=10, fontweight="bold", color="#445")
    # GNN box
    add_box(ax, 3.6, 6.6, 3.2, 2.6,
            "GraphSAGE (5 layers, hidden 128)\n+ GRU temporal head\n+ output normalisation\n\n"
            r"$V_j = \mathrm{GNN}_\theta(X_j)$",
            fc="#fde6e2", fontsize=8.5)
    # RUM head box
    add_box(ax, 3.6, 4.4, 3.2, 1.9,
            r"$U_{ij} = \alpha V_j + \beta_t(i,j) t_{ij} + \gamma \log d_{ij} + \delta\, \mathrm{OccMatch}_{ij}$" + "\n\n"
            r"$\beta_t(i,j) = \beta_t \cdot (1+\phi z_o(i)+\psi z_w(j))$",
            fc="#fde6e2", fontsize=8)
    add_box(ax, 3.6, 2.6, 3.2, 1.5,
            "Maximum-likelihood inverse training\n"
            "on aggregate Census 2021 OD\n"
            "Top-K=50 sampling-of-alternatives\n(McFadden 1978)",
            fc="#fde6e2", fontsize=8)
    add_box(ax, 3.6, 1.0, 3.2, 1.3,
            "Recovered parameters:\n"
            r"$\hat\theta, \hat\beta_t, \hat\psi, \hat\delta, \hat\phi, \hat\gamma$"
            "\n→ behaviourally interpretable",
            fc="#d6e9c6", fontsize=8.5, bold=True)

    # === LANE 3: COUNTERFACTUAL FORWARD (middle-right) ===
    ax.text(7.6, 9.5, "COUNTERFACTUAL FORWARD",
            fontsize=10, fontweight="bold", color="#445")
    add_box(ax, 7.5, 7.6, 3.2, 1.6,
            r"do(X_S = X_S^*)" "\n(spatial intervention or\ntemporal demand shift)\n+ frozen baseline norm",
            fc="#e8d8f0", fontsize=8)
    add_box(ax, 7.5, 5.6, 3.2, 1.7,
            r"$P_t(j \mid i) = \mathrm{softmax}_j(U_{ij}^t)$" "\n"
            r"with $\hat{\beta}, \hat{\psi}, \hat{\theta}$ frozen",
            fc="#e8d8f0", fontsize=8.5)
    add_box(ax, 7.5, 3.6, 3.2, 1.7,
            "BPR user equilibrium\n(Sheffi 1985)\n"
            r"$t_{ij}^{eq} = t^0_{ij}(1+\alpha(V/C)^\beta)$" "\n"
            "MSA fixed-point, mult-cap",
            fc="#e8d8f0", fontsize=8)

    # === LANE 4: AGENT-LEVEL HETEROGENEITY (parallel right) ===
    ax.text(11.4, 9.5, "AGENT LAYER (ABM)",
            fontsize=10, fontweight="bold", color="#445")
    add_box(ax, 11.3, 7.6, 3.2, 1.6,
            "10K synthetic London agents\nincome × home_grid × occ\n"
            "(matched to ASHE/KS101 marginals)",
            fc="#fff2cc", fontsize=8)
    add_box(ax, 11.3, 5.6, 3.2, 1.7,
            "per-agent " + r"$\beta_n = \hat\beta_t(i_n,j) (1+\varepsilon_n)$" + "\n"
            r"$\varepsilon_n \sim \mathcal{N}(0, \sigma^2(\mathrm{tier}_n))$" + "\n"
            "σ from WebTAG / Wardman",
            fc="#fff2cc", fontsize=8)
    add_box(ax, 11.3, 3.6, 3.2, 1.7,
            "Per-agent softmax over\n"
            "destinations\n"
            "→ aggregated OD prediction\n"
            "(no Jensen bias)",
            fc="#fff2cc", fontsize=8)

    # === BOTTOM: OUTPUTS ===
    ax.text(7.6, 0.5, "OUTPUTS / POLICY DELIVERABLES",
            fontsize=10, fontweight="bold", color="#445")
    add_box(ax, 7.5, -1.6, 3.2, 1.7,
            "Aggregate ΔF\nPer-grid ΔAccessibility\nGini / Palma / Atkinson\nBPR ΔTravel time",
            fc="#d6e9c6", fontsize=8.5, bold=True)
    add_box(ax, 11.3, -1.6, 3.2, 1.7,
            "Agent-level ΔF\nJensen bias (vs aggregate)\nPer-agent welfare effects\n→ ABM policy fidelity",
            fc="#d6e9c6", fontsize=8.5, bold=True)

    # === ARROWS ===
    # Inputs → Calibration
    add_arrow(ax, 2.9, 8.4, 3.6, 8.0)        # OD → GNN
    add_arrow(ax, 2.9, 6.6, 3.6, 6.6)        # features → GNN
    add_arrow(ax, 2.9, 4.8, 3.6, 5.4)        # t/d → RUM head
    # Calibration internal
    add_arrow(ax, 5.2, 6.6, 5.2, 6.3)        # V_j → RUM head
    add_arrow(ax, 5.2, 4.4, 5.2, 4.1)        # RUM head → MLE
    add_arrow(ax, 5.2, 2.6, 5.2, 2.3)        # MLE → recovered

    # Calibration → Forward
    add_arrow(ax, 6.8, 1.6, 9.1, 4.5,
              connectionstyle="arc3,rad=-0.25",
              label=r"$\hat\beta, \hat\psi, \hat\theta$",
              label_offset=(-0.4, 0.2))

    # Forward internal
    add_arrow(ax, 9.1, 7.6, 9.1, 7.3)        # do(X) → softmax
    add_arrow(ax, 9.1, 5.6, 9.1, 5.3)        # softmax → BPR
    add_arrow(ax, 9.1, 3.6, 9.1, 0.1)        # BPR → outputs (down)

    # Inputs → Agent
    add_arrow(ax, 1.6, 2.3, 11.3, 6.4,
              connectionstyle="arc3,rad=-0.55",
              color="#a07c00",
              label="WebTAG σ",
              label_offset=(-1.5, 1.2))
    # Calibration → Agent
    add_arrow(ax, 6.8, 1.6, 12.9, 6.3,
              connectionstyle="arc3,rad=-0.45",
              color="#a07c00")
    # Agent internal
    add_arrow(ax, 12.9, 7.6, 12.9, 7.3)
    add_arrow(ax, 12.9, 5.6, 12.9, 5.3)
    add_arrow(ax, 12.9, 3.6, 12.9, 0.1)

    # Forward ↔ Agent comparison
    add_arrow(ax, 10.7, 4.5, 11.3, 4.5,
              style="<|-|>", color="#666",
              label="Jensen bias\nquantification",
              label_offset=(0, 0.5))

    # === Title ===
    ax.text(7.5, 9.85,
            "Conceptual framework: London commuting policy simulator",
            fontsize=14, fontweight="bold", ha="center")

    # === Legend ===
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="#e8f1f9", ec="#222", label="Open data inputs"),
        Rectangle((0, 0), 1, 1, fc="#fde6e2", ec="#222", label="GNN + inverse training (calibration)"),
        Rectangle((0, 0), 1, 1, fc="#e8d8f0", ec="#222", label="Counterfactual forward + BPR"),
        Rectangle((0, 0), 1, 1, fc="#fff2cc", ec="#222", label="Agent-level (ABM layer)"),
        Rectangle((0, 0), 1, 1, fc="#d6e9c6", ec="#222", label="Output / parameter products"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8.5,
              frameon=True, bbox_to_anchor=(0.0, -0.2))

    plt.tight_layout()
    out = OUT_DIR / "F0_conceptual_framework.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    out_svg = OUT_DIR / "F0_conceptual_framework.svg"
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
