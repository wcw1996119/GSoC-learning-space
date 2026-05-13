"""F3 Figure: multiverse CPC distribution histogram.

Reads multiverse_d9.csv. Run after D9 finishes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def main():
    df = pd.read_csv(OUT_DIR / "multiverse_d9.csv")
    df = df[df["val_cpc"].notna()]
    n = len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    df_un = df[df["enforce_main"] == False]
    df_co = df[df["enforce_main"] == True]
    bins = np.linspace(0.15, 0.40, 26)
    ax.hist(df_un["val_cpc"], bins=bins, alpha=0.65, label=f"Unconstrained (n={len(df_un)})",
             color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.hist(df_co["val_cpc"], bins=bins, alpha=0.65, label=f"Constrained (n={len(df_co)})",
             color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax.axvline(0.170, color="grey", linestyle=":", lw=1.2, label="Gravity baseline 0.170")
    ax.axvline(0.343, color="#1f77b4", linestyle="-", lw=1.5,
               label="Headline ensemble 0.343")
    ax.set_xlabel("Spatial-holdout CPC", fontsize=11)
    ax.set_ylabel("Count of configurations", fontsize=11)
    ax.set_title(f"Multiverse robustness: distribution of CPC over {n} hyperparameter configurations\n"
                 f"100% of configs exceed gravity baseline 0.17",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Annotate stats
    pct_above_gravity = (df["val_cpc"] > 0.17).mean() * 100
    pct_above_25 = (df["val_cpc"] > 0.25).mean() * 100
    pct_above_30 = (df["val_cpc"] > 0.30).mean() * 100
    txt = (f"All-config stats:\n"
           f"  median {df['val_cpc'].median():.3f}\n"
           f"  IQR [{df['val_cpc'].quantile(0.25):.3f}, {df['val_cpc'].quantile(0.75):.3f}]\n"
           f"  range [{df['val_cpc'].min():.3f}, {df['val_cpc'].max():.3f}]\n"
           f"  > 0.170 (gravity): {pct_above_gravity:.0f}%\n"
           f"  > 0.250: {pct_above_25:.0f}%\n"
           f"  > 0.300: {pct_above_30:.0f}%")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes,
             ha="right", va="top", fontsize=8,
             bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9))

    plt.tight_layout()
    out = OUT_DIR / "F3_multiverse.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
