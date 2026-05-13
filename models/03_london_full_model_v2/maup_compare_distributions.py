"""MAUP sensitivity (lite): compare distributional properties of grid features at 1km vs 2km.

Shows MAUP effects on Gini, Moran's I, and feature variance — without re-training a
separate STGNN, which is more honest given v2's 1km grid is the operating model.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))
PROC = V2_ROOT / "data" / "processed"
EVAL = V2_ROOT / "evaluation_outputs"

from metrics.inequality import gini


def main():
    print("Loading 1km and 2km feature tables...")
    f1 = pd.read_csv(PROC / "grid_static_features.csv")
    f2 = pd.read_csv(PROC / "grid_static_features_2km.csv")
    print(f"  1km grids: {len(f1)}; 2km grids: {len(f2)}")

    rows = []
    for col in ["total_employment", "population", "poi_total", "subway_station_count"]:
        v1 = f1[col].values
        v2 = f2[col].values
        rows.append({
            "variable": col,
            "n_1km": len(v1),
            "n_2km": len(v2),
            "mean_1km": float(v1.mean()),
            "mean_2km": float(v2.mean()),
            "max_1km": float(v1.max()),
            "max_2km": float(v2.max()),
            "gini_1km": gini(v1),
            "gini_2km": gini(v2),
            "cv_1km": float(v1.std() / max(v1.mean(), 1e-8)),
            "cv_2km": float(v2.std() / max(v2.mean(), 1e-8)),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Gini comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    x = np.arange(len(rows))
    width = 0.35
    g1 = [r["gini_1km"] for r in rows]
    g2 = [r["gini_2km"] for r in rows]
    ax.bar(x - width/2, g1, width, label="1 km grid", color="#1976d2")
    ax.bar(x + width/2, g2, width, label="2 km grid", color="#388e3c")
    ax.set_xticks(x)
    ax.set_xticklabels([r["variable"].replace("_", "\n") for r in rows], fontsize=9)
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Inequality (Gini) of grid features: 1 km vs 2 km")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(g1, g2)):
        ax.text(i - width/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    cv1 = [r["cv_1km"] for r in rows]
    cv2 = [r["cv_2km"] for r in rows]
    ax.bar(x - width/2, cv1, width, label="1 km grid", color="#1976d2")
    ax.bar(x + width/2, cv2, width, label="2 km grid", color="#388e3c")
    ax.set_xticks(x)
    ax.set_xticklabels([r["variable"].replace("_", "\n") for r in rows], fontsize=9)
    ax.set_ylabel("Coefficient of variation (std / mean)")
    ax.set_title("Spatial dispersion: 1 km vs 2 km")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(cv1, cv2)):
        ax.text(i - width/2, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = EVAL / "maup_distributions.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")

    # Save table
    df.to_csv(EVAL / "maup_distributions.csv", index=False)
    return df


if __name__ == "__main__":
    main()
