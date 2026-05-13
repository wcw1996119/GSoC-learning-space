"""Compute real Hansen accessibility A_i for London using DUAL_HET ckpt + real data.

A_i = sum_j E_j * exp(beta_car * t_car_ij)

Uses:
  - Real London grid coords (1725 grids)
  - Real t_car matrix (free-flow, OSMnx-derived)
  - Real total_employment per grid (BRES)
  - Recovered beta_car = -0.187 from DUAL_HET training
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "methodology" / "figures" / "arch"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load real data
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    coords = cache["coords_bng"].astype(np.float32)         # (N, 2)
    grid_ids = cache["grid_ids"].tolist()

    # t_car free-flow
    t_car = np.load(ROOT / "data" / "processed" / "car_freeflow_t_ij.npy").astype(np.float32)

    # employment per grid
    feats = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    feats = feats.set_index("grid_id").reindex(grid_ids)
    E = feats["total_employment"].fillna(0).to_numpy().astype(np.float32)
    print(f"Σ E = {E.sum():,.0f} jobs total across {len(E)} grids")

    # Recovered β_car = -0.187 (from §3.2 / dual_het_seed0.pt)
    beta_car = -0.187

    # Compute Hansen A_i
    decay = np.exp(beta_car * t_car)                          # (N, N)
    np.fill_diagonal(decay, 0.0)
    A = decay @ E                                              # (N,)
    print(f"A_i: mean={A.mean():,.0f}  p10={np.percentile(A, 10):,.0f}  "
          f"p50={np.percentile(A, 50):,.0f}  p90={np.percentile(A, 90):,.0f}  "
          f"max={A.max():,.0f}")

    # Heatmap with proper transparency for London shape
    fig, ax = plt.subplots(figsize=(2.8, 2.2))
    sc = ax.scatter(coords[:, 0]/1e3, coords[:, 1]/1e3,
                     c=A, s=2.0, cmap="YlOrRd",
                     alpha=0.95, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, aspect=22)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label(r"$A_i$ (Hansen, jobs reachable)", fontsize=7)
    cb.formatter.set_powerlimits((6, 6))
    cb.update_ticks()
    ax.set_title("London job accessibility (real)", fontsize=8, color="#334155")
    fig.savefig(OUT / "grid_accessibility.png", dpi=220, bbox_inches="tight",
                  pad_inches=0.04)
    plt.close(fig)
    print(f"wrote grid_accessibility.png")

    # Also a smaller version for embedding
    fig, ax = plt.subplots(figsize=(2.2, 1.7))
    sc = ax.scatter(coords[:, 0]/1e3, coords[:, 1]/1e3,
                     c=A, s=1.5, cmap="YlOrRd",
                     alpha=0.95, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, aspect=18)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label(r"$A_i$", fontsize=6.5)
    fig.savefig(OUT / "grid_accessibility_small.png", dpi=220, bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
