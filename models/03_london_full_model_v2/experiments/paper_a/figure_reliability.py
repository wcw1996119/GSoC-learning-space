"""F8 Figure: model reliability heatmap from reliability_scan.csv.

Two panels:
  Left:  scatter of |Jensen bias| (%) vs intervention magnitude × baseline emp.
         Colour = bias magnitude. Highlights the trust region.
  Right: cumulative-distribution of |Jensen bias| over all (cluster, magnitude)
         experiments with thresholds at 5% (reliable) and 20% (caveat).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def main():
    df = pd.read_csv(OUT_DIR / "reliability_scan.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Jensen bias scatter ---
    ax = axes[0]
    sc = ax.scatter(df["magnitude_multiplier"], df["baseline_emp"],
                     c=df["abs_jensen_bias_pct"].clip(upper=40),
                     cmap="RdYlGn_r", vmin=0, vmax=40,
                     s=20, edgecolor="grey", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intervention magnitude (Δemp / baseline cluster emp)", fontsize=10)
    ax.set_ylabel("Baseline cluster employment (jobs)", fontsize=10)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("|Jensen bias| (%)", fontsize=9)

    # Trust-region annotation: shade where bias < 5% empirically
    rel_mask = df["abs_jensen_bias_pct"] < 5
    if rel_mask.sum() > 0:
        rel = df[rel_mask]
        # Outline the convex region of reliable points
        from scipy.spatial import ConvexHull
        try:
            pts = rel[["magnitude_multiplier", "baseline_emp"]].values
            log_pts = np.log10(pts.clip(min=1e-3))
            hull = ConvexHull(log_pts)
            poly = pts[hull.vertices]
            poly = np.vstack([poly, poly[0]])
            ax.plot(poly[:, 0], poly[:, 1], "g-", lw=2, alpha=0.6,
                     label="reliable region (|bias| < 5%)")
        except Exception:
            pass
    ax.set_title("(a) Trust map: |Jensen bias| by intervention type",
                  fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    # --- Panel B: CDF of Jensen bias ---
    ax = axes[1]
    sorted_bias = np.sort(df["abs_jensen_bias_pct"].values)
    cdf = np.arange(1, len(sorted_bias) + 1) / len(sorted_bias)
    ax.plot(sorted_bias, cdf * 100, "b-", lw=2)
    ax.axvline(5, color="green", linestyle="--", lw=1, label="reliable (5%)")
    ax.axvline(20, color="red", linestyle="--", lw=1, label="unreliable (20%)")
    pct_below_5 = (df["abs_jensen_bias_pct"] < 5).mean() * 100
    pct_below_20 = (df["abs_jensen_bias_pct"] < 20).mean() * 100
    ax.text(5.5, 70, f"{pct_below_5:.0f}% reliable", fontsize=9,
            color="green", fontweight="bold")
    ax.text(21, 50, f"{pct_below_20:.0f}% within caveat", fontsize=9,
            color="orange", fontweight="bold")
    ax.set_xlabel("|Jensen bias| (%)", fontsize=10)
    ax.set_ylabel("Cumulative % of experiments", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0, 100)
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_title("(b) Reliability CDF across all (cluster × magnitude) cells",
                  fontsize=11)
    ax.legend(fontsize=9, loc="lower right")

    plt.suptitle(f"Empirical reliability of counterfactual prediction "
                  f"(n = {len(df)} synthetic interventions, "
                  f"{df['cluster_idx'].nunique()} clusters)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "F8_reliability.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
