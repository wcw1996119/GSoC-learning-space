"""Compute Moran's I + LISA for v2.2 grid features and predicted V_j(t).

Outputs
-------
- evaluation_outputs/v22_morans_results.npz
- evaluation_outputs/v22_morans_map.png

Variables tested
----------------
- total_employment, population (static features)
- V_j(hour=8), V_j(hour=17) (STGNN-predicted attractiveness)

Usage
-----
    python compute_v22_morans.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from metrics.spatial_autocorr import (
    build_kNN_weight,
    morans_I,
    lisa,
    lisa_classify,
    lisa_with_pvalues,
)

PROC = V2_ROOT / "data" / "processed"
OUT = V2_ROOT / "evaluation_outputs"
OUT.mkdir(exist_ok=True)


def main(k: int = 8, n_perm: int = 999, hours=(8, 17)):
    print(f"=== Moran's I / LISA for v2.2 (k={k}, n_perm={n_perm}) ===")

    # ----- Load grid geometry, project to BNG -----
    grid = gpd.read_file(PROC / "london_1km_grid.geojson")
    if grid.crs is None or grid.crs.to_epsg() != 27700:
        grid = grid.to_crs(epsg=27700)
    grid["x_bng"] = grid.geometry.centroid.x
    grid["y_bng"] = grid.geometry.centroid.y
    grid = grid.sort_values("grid_id").reset_index(drop=True)
    coords = grid[["x_bng", "y_bng"]].values.astype(np.float64)
    grid_ids_geo = grid["grid_id"].tolist()
    N = len(grid)
    print(f"Loaded {N} grids in EPSG:27700")

    # ----- Load static features, align by grid_id -----
    feat = pd.read_csv(PROC / "grid_static_features.csv")
    feat = feat.set_index("grid_id").reindex(grid_ids_geo).reset_index()
    if feat["grid_id"].isnull().any():
        raise RuntimeError("Static features missing for some grids")
    total_employment = feat["total_employment"].values.astype(np.float64)
    population = feat["population"].values.astype(np.float64)

    # ----- Load STGNN baseline V_jt -----
    cache = dict(np.load(PROC / "demo_cache.npz", allow_pickle=True))
    V_jt = cache["V_jt_baseline"]  # (T, N)
    cache_grid_ids = cache["grid_ids"].tolist() if "grid_ids" in cache else None
    if cache_grid_ids is not None and cache_grid_ids != grid_ids_geo:
        # Reorder V_jt to match geojson ordering
        idx = {g: i for i, g in enumerate(cache_grid_ids)}
        order = np.array([idx[g] for g in grid_ids_geo], dtype=np.int64)
        V_jt = V_jt[:, order]
        print("Reordered V_jt to match geojson grid_ids")
    if V_jt.shape[1] != N:
        raise RuntimeError(
            f"V_jt grids ({V_jt.shape[1]}) != geojson grids ({N})"
        )
    V_h_dict = {int(h): V_jt[h].astype(np.float64) for h in hours}

    # ----- Build spatial weight matrix -----
    print(f"Building kNN(k={k}) row-standardised weight matrix...")
    W = build_kNN_weight(coords, k=k)

    # ----- Compute Moran's I for each variable -----
    variables = {
        "total_employment": total_employment,
        "population": population,
        **{f"V_j_h{h:02d}": V_h_dict[h] for h in hours},
    }

    results = {}
    print("\nGlobal Moran's I:")
    print(f"  {'variable':<22s} {'I':>8s} {'p_value':>10s}")
    for name, vals in variables.items():
        I, p = morans_I(vals, W, n_permutations=n_perm, seed=42)
        results[name] = {"I": I, "p_value": p}
        print(f"  {name:<22s} {I:>8.4f} {p:>10.4f}")

    # ----- LISA for V_j(hour=8) -----
    focus_hour = hours[0]
    focus_name = f"V_j_h{focus_hour:02d}"
    print(f"\nLISA classification for {focus_name} "
          f"(conditional permutation, p<0.05)...")
    I_local, p_local = lisa_with_pvalues(
        V_h_dict[focus_hour], W, n_permutations=n_perm, seed=42
    )
    labels = lisa_classify(V_h_dict[focus_hour], W, p_local=p_local,
                           p_threshold=0.05)
    label_names = {0: "ns", 1: "HH", 2: "LL", 3: "HL", 4: "LH"}
    counts = {label_names[k]: int((labels == k).sum())
              for k in label_names}
    print(f"  LISA counts: {counts}")
    n_sig = int((labels > 0).sum())
    n_hot = int((labels == 1).sum())
    n_cold = int((labels == 2).sum())
    n_anom = int(((labels == 3) | (labels == 4)).sum())
    print(f"  significant: {n_sig}/{N} ({100*n_sig/N:.1f}%)  "
          f"HH={n_hot} LL={n_cold} anomalies={n_anom}")

    # ----- Save results -----
    out_npz = OUT / "v22_morans_results.npz"
    np.savez_compressed(
        out_npz,
        variable_names=np.array(list(variables.keys())),
        morans_I=np.array([results[k]["I"] for k in variables]),
        p_values=np.array([results[k]["p_value"] for k in variables]),
        focus_hour=np.int64(focus_hour),
        lisa_local=I_local.astype(np.float64),
        lisa_pvalues=p_local.astype(np.float64),
        lisa_labels=labels.astype(np.int64),
        grid_ids=np.array(grid_ids_geo),
        coords_bng=coords.astype(np.float64),
        k=np.int64(k),
        n_permutations=np.int64(n_perm),
    )
    print(f"\nSaved results -> {out_npz}")

    # ----- LISA map -----
    fig, ax = plt.subplots(figsize=(11, 11))
    palette = {
        0: "#dddddd",  # ns
        1: "#d62728",  # HH (red)
        2: "#1f77b4",  # LL (blue)
        3: "#fdae61",  # HL (light orange)
        4: "#abd9e9",  # LH (light blue)
    }
    grid_plot = grid.copy()
    grid_plot["lisa"] = labels
    for code, colour in palette.items():
        sub = grid_plot[grid_plot["lisa"] == code]
        if not sub.empty:
            sub.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.1)

    # Borough overlay (lightweight: just outline grid boundary)
    grid_plot.boundary.plot(ax=ax, color="black", linewidth=0.05, alpha=0.3)

    ax.set_title(
        f"LISA hot/cold spots — V_j(hour={focus_hour}) "
        f"[k={k} kNN, {n_perm} permutations, p<0.05]",
        fontsize=13,
    )
    ax.set_axis_off()
    legend = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=palette[1],
               markersize=12, label=f"High-High hot spot (n={n_hot})",
               markeredgecolor="white"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=palette[2],
               markersize=12, label=f"Low-Low cold spot (n={n_cold})",
               markeredgecolor="white"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=palette[3],
               markersize=12, label=f"High-Low anomaly (n={int((labels==3).sum())})",
               markeredgecolor="white"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=palette[4],
               markersize=12, label=f"Low-High anomaly (n={int((labels==4).sum())})",
               markeredgecolor="white"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=palette[0],
               markersize=12, label=f"Not significant (n={int((labels==0).sum())})",
               markeredgecolor="white"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10, frameon=True)

    out_png = OUT / "v22_morans_map.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LISA map -> {out_png}")

    # ----- Summary table to stdout -----
    print("\n=== Summary ===")
    print(f"  {'variable':<22s} {'I':>8s} {'p_value':>10s}")
    for name in variables:
        r = results[name]
        sig = "***" if r["p_value"] < 0.001 else (
            "**" if r["p_value"] < 0.01 else (
                "*" if r["p_value"] < 0.05 else ""))
        print(f"  {name:<22s} {r['I']:>8.4f} {r['p_value']:>10.4f} {sig}")
    print(f"\nLISA on {focus_name}:")
    print(f"  hot spots (HH):   {n_hot} ({100*n_hot/N:.1f}%)")
    print(f"  cold spots (LL):  {n_cold} ({100*n_cold/N:.1f}%)")
    print(f"  anomalies:        {n_anom} ({100*n_anom/N:.1f}%)")
    print(f"  total significant: {n_sig} ({100*n_sig/N:.1f}%)")

    return results, labels


if __name__ == "__main__":
    main()
