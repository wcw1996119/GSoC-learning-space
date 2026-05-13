"""Counterfactual reliability boundary figure.

Shows where (in the input perturbation space) the aggregate ML-ABM
framework can be trusted to make counterfactual predictions, vs. where
the aggregate prediction diverges from the per-agent Monte-Carlo ground
truth (the "Jensen bias").

Inputs:
  evaluation_outputs/paper_a/reliability_scan.csv  — 360 synthetic
    experiments (60 random 9-grid clusters × 6 magnitudes).
    Columns include cluster_size, baseline_emp, magnitude_multiplier,
    delta_total_jobs, jensen_bias_pct.

Output:
  paper_methods_reliability.png  — 2-panel publication figure:
    (a) 2D scatter of (baseline cluster size, intervention magnitude)
        coloured by |Jensen bias|, with reliability zones shaded and
        Scenario A (OOC), Scenario B (flexible hours) markers.
    (b) CDF of |Jensen bias| across all 360 experiments, with the
        12% / 31% / 56% zone fractions reported.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "evaluation_outputs" / "paper_a" / "reliability_scan.csv"
OUT_PNG = ROOT / "paper_methods_reliability.png"
OUT_PDF = ROOT / "paper_methods_reliability.pdf"


def main():
    df = pd.read_csv(DATA)
    print(f"loaded {len(df)} reliability-scan experiments")

    df["baseline_emp_k"] = df["baseline_emp"] / 1000.0
    df["abs_bias"] = df["abs_jensen_bias_pct"]
    df["abs_bias_clip"] = df["abs_bias"].clip(upper=200)
    df["delta_jobs_k"] = df["delta_total_jobs"] / 1000.0

    # --- Reliability classification ---
    bias_reliable = 5.0
    bias_caveat = 20.0
    df["zone"] = "unreliable"
    df.loc[df["abs_bias"] < bias_caveat, "zone"] = "caveat"
    df.loc[df["abs_bias"] < bias_reliable, "zone"] = "reliable"

    n_total = len(df)
    n_reliable = (df["zone"] == "reliable").sum()
    n_caveat = (df["zone"] == "caveat").sum()
    n_unreliable = (df["zone"] == "unreliable").sum()
    print(f"reliable:   {n_reliable}/{n_total}  ({100 * n_reliable / n_total:.1f}%)")
    print(f"caveat:     {n_caveat}/{n_total}  ({100 * n_caveat / n_total:.1f}%)")
    print(f"unreliable: {n_unreliable}/{n_total}  ({100 * n_unreliable / n_total:.1f}%)")

    # ============================================================
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.30,
                           left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ============================================================
    # PANEL A — 2D reliability map
    # ============================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xscale("log")

    # Background reliability zones (vertical-ish bands by magnitude)
    # since x-axis = baseline cluster size, y-axis = intervention magnitude

    # Use scatter coloured by |bias|
    scat = ax.scatter(df["baseline_emp_k"], df["magnitude_multiplier"],
                      c=df["abs_bias_clip"], s=68, cmap="RdYlGn_r",
                      vmin=0, vmax=80, edgecolors="white", linewidths=0.4,
                      zorder=4, alpha=0.92)
    cbar = plt.colorbar(scat, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("|Jensen bias|  (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Zone shading (light overlay to indicate reliability)
    # We approximate zones as: reliable region is small/moderate
    # (low bias for moderate magnitude on adequately-sized clusters);
    # caveat is the middle band; unreliable is extremes.

    # Compute a rough reliable region via density of low-bias points
    reliable_x = df.loc[df["zone"] == "reliable", "baseline_emp_k"]
    reliable_y = df.loc[df["zone"] == "reliable", "magnitude_multiplier"]
    if len(reliable_x) > 0:
        # 1-σ ellipse in log(x), y space
        lx = np.log10(reliable_x.values + 1e-3)
        ly = reliable_y.values
        ax.add_patch(Rectangle((10 ** (lx.mean() - lx.std()),
                                 max(ly.mean() - ly.std(), 0.01)),
                                10 ** (2 * lx.std()), 2 * ly.std(),
                                facecolor="#a0c878", edgecolor="#3e7c30",
                                linewidth=1.2, alpha=0.18, zorder=2))
        ax.text(10 ** lx.mean(), ly.mean() + 0.05, "RELIABLE\n(|bias| < 5%)",
                ha="center", va="center", fontsize=8.5, color="#3e7c30",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#3e7c30", linewidth=0.5))

    # Mark Scenario A (OOC) and Scenario B (flexible hours)
    # Scenario A: 9-grid cluster with baseline ~150,000 jobs, +65,000 → mag = 0.43
    ooc_baseline = 150.0  # in thousands
    ooc_mag = 0.43
    ax.scatter([ooc_baseline], [ooc_mag], s=350, marker="*",
               edgecolors="#963F2D", facecolors="#FFD17A",
               linewidths=2.0, zorder=10)
    ax.annotate("Scenario A\n(OOC +65k jobs)",
                xy=(ooc_baseline, ooc_mag),
                xytext=(ooc_baseline * 0.20, ooc_mag + 0.55),
                fontsize=9, color="#963F2D", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#963F2D",
                                 linewidth=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#963F2D", linewidth=0.7),
                zorder=10)

    # Scenario B (flexible hours) — temporal intervention. We represent
    # it as a small magnitude perturbation on the median-size cluster.
    flexb_baseline = df["baseline_emp_k"].median()
    flexb_mag = 0.5  # 50% peak shift
    ax.scatter([flexb_baseline], [flexb_mag], s=350, marker="^",
               edgecolors="#3F7F2F", facecolors="#A8D586",
               linewidths=2.0, zorder=10)
    ax.annotate("Scenario B\n(flex-hours peak ×0.5)",
                xy=(flexb_baseline, flexb_mag),
                xytext=(flexb_baseline * 4.5, flexb_mag + 0.50),
                fontsize=9, color="#3F7F2F", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#3F7F2F",
                                 linewidth=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#3F7F2F", linewidth=0.7),
                zorder=10)

    ax.set_xlabel("Cluster baseline employment  (thousand jobs, log scale)",
                  fontsize=10)
    ax.set_ylabel("Intervention magnitude  (Δ employment / baseline)",
                  fontsize=10)
    ax.set_title("(a)  Counterfactual reliability map  "
                 "(360 synthetic 9-grid cluster × magnitude experiments)",
                 fontsize=11, fontweight="bold", color="#222")
    ax.grid(True, alpha=0.25, linewidth=0.5, zorder=1)
    ax.set_ylim(0.05, 6)
    ax.set_yticks([0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    ax.set_yticklabels(["10%", "30%", "50%", "100%", "200%", "500%"])

    # ============================================================
    # PANEL B — CDF of |bias|
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_bias = np.sort(df["abs_bias"].values)
    cdf = np.arange(1, len(sorted_bias) + 1) / len(sorted_bias)
    ax2.plot(sorted_bias, cdf, color="#2F5F8C", linewidth=2.0, zorder=4)
    ax2.fill_between(sorted_bias, 0, cdf, alpha=0.10, color="#5A8FB8",
                      zorder=3)

    # Vertical thresholds
    ax2.axvline(5.0, color="#3e7c30", linestyle="--", linewidth=1.5,
                zorder=3)
    ax2.axvline(20.0, color="#A87B1E", linestyle="--", linewidth=1.5,
                zorder=3)

    ax2.text(5.0, 0.97, "5%", ha="center", va="top", fontsize=8,
             color="#3e7c30", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="#3e7c30"), zorder=5)
    ax2.text(20.0, 0.97, "20%", ha="center", va="top", fontsize=8,
             color="#A87B1E", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="#A87B1E"), zorder=5)

    # Zone shading on x-axis
    ax2.axvspan(0, 5.0, ymin=0, ymax=1, alpha=0.10, color="#a0c878",
                 zorder=1)
    ax2.axvspan(5.0, 20.0, ymin=0, ymax=1, alpha=0.10, color="#f1d97a",
                 zorder=1)
    ax2.axvspan(20.0, 200, ymin=0, ymax=1, alpha=0.10, color="#dc8060",
                 zorder=1)

    # Zone fraction annotations
    ax2.text(2.5, 0.50, f"reliable\n{100*n_reliable/n_total:.0f}%\n"
                         f"({n_reliable}/{n_total})",
             ha="center", va="center", fontsize=9, color="#3e7c30",
             fontweight="bold", zorder=5)
    ax2.text(11.0, 0.30, f"caveat\n{100*n_caveat/n_total:.0f}%\n"
                          f"({n_caveat}/{n_total})",
             ha="center", va="center", fontsize=9, color="#A87B1E",
             fontweight="bold", zorder=5)
    ax2.text(60.0, 0.15, f"unreliable\n{100*n_unreliable/n_total:.0f}%\n"
                          f"({n_unreliable}/{n_total})",
             ha="center", va="center", fontsize=9, color="#963F2D",
             fontweight="bold", zorder=5)

    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 200)
    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel("|Jensen bias|  (%)  (log scale)", fontsize=10)
    ax2.set_ylabel("Cumulative fraction of experiments", fontsize=10)
    ax2.set_title("(b)  Distribution of aggregate-prediction bias  "
                  "across 360 experiments",
                  fontsize=11, fontweight="bold", color="#222")
    ax2.grid(True, alpha=0.25, linewidth=0.5, zorder=2)

    # Annotation: where do scenarios sit?
    ooc_bias = 14.0
    flexb_bias = 4.0
    ax2.scatter([ooc_bias], [(df["abs_bias"] < ooc_bias).mean()],
                marker="*", s=350, edgecolors="#963F2D",
                facecolors="#FFD17A", linewidths=2.0, zorder=6)
    ax2.scatter([flexb_bias],
                [(df["abs_bias"] < flexb_bias).mean()],
                marker="^", s=350, edgecolors="#3F7F2F",
                facecolors="#A8D586", linewidths=2.0, zorder=6)

    # ============================================================
    # Title and overall caption
    # ============================================================
    fig.suptitle(
        "Figure 2 — Counterfactual reliability boundary  "
        "(where the ML-ABM aggregate prediction can be trusted)",
        fontsize=13, fontweight="bold", y=0.98, color="#1a1a1a")

    # Bottom caption / legend
    legend_elems = [
        Line2D([0], [0], marker="*", markersize=14, linestyle="none",
                markerfacecolor="#FFD17A", markeredgecolor="#963F2D",
                markeredgewidth=1.5,
                label="Scenario A (OOC +65k jobs):  caveat zone, |bias|≈14%"),
        Line2D([0], [0], marker="^", markersize=12, linestyle="none",
                markerfacecolor="#A8D586", markeredgecolor="#3F7F2F",
                markeredgewidth=1.5,
                label="Scenario B (flex-hours):  reliable zone, |bias|≈4%"),
        Patch(facecolor="#a0c878", alpha=0.18, edgecolor="#3e7c30",
               label="Reliable  (|bias| < 5%)"),
        Patch(facecolor="#f1d97a", alpha=0.18, edgecolor="#A87B1E",
               label="Caveat  (5% ≤ |bias| < 20%)"),
        Patch(facecolor="#dc8060", alpha=0.18, edgecolor="#963F2D",
               label="Unreliable  (|bias| ≥ 20%)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.01))

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PNG.name}  ({OUT_PNG.stat().st_size // 1024} KB)")
    print(f"saved {OUT_PDF.name}  ({OUT_PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
