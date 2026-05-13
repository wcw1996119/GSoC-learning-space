"""Quick MAUP comparison: aggregate 1km features → 2km cells (in BNG), compute Gini/CV at both scales."""
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
sys.path.insert(0, str(V2_ROOT))
from metrics.inequality import gini


def main():
    # Load 1km grid + features (in WGS84)
    grid_1km = gpd.read_file(PROC / "london_1km_grid.geojson").sort_values("grid_id").reset_index(drop=True)
    feats_1km = pd.read_csv(PROC / "grid_static_features.csv").set_index("grid_id").reindex(grid_1km["grid_id"]).reset_index()
    grid_1km = grid_1km.merge(feats_1km.drop(columns=["centroid_lat","centroid_lon","area_in_london_km2","area_km2"], errors="ignore"),
                                on="grid_id", how="left")
    print(f"1km grids: {len(grid_1km)}")

    # Load 2km grid (in WGS84) and convert to BNG for spatial join
    grid_2km = gpd.read_file(PROC / "london_2km_grid.geojson")
    print(f"2km grids: {len(grid_2km)}")

    # Project both to BNG for spatial join
    grid_1km_bng = grid_1km.to_crs(epsg=27700)
    grid_2km_bng = grid_2km.to_crs(epsg=27700)

    # Compute 1km centroids in BNG
    cents_1km = grid_1km_bng.copy()
    cents_1km["geometry"] = cents_1km.geometry.centroid

    # Spatial join: each 1km centroid lands in exactly one 2km cell
    joined = gpd.sjoin(cents_1km, grid_2km_bng[["grid_id","geometry"]].rename(columns={"grid_id":"grid_id_2km"}),
                        how="inner", predicate="within")
    print(f"Joined 1km centroids → 2km: {len(joined)}")

    # Aggregate (sum) per 2km
    SUM_COLS = [
        "sec1_primary","sec2_manufacturing","sec3_construction","sec4_retail",
        "sec5_fnb","sec6_info_finance","sec7_public","sec8_other",
        "total_employment","population","poi_total",
        "poi_commercial","poi_education","poi_fnb","poi_healthcare",
        "poi_office","poi_public","poi_retail","poi_transport",
        "subway_station_count","bus_stop_count",
    ]
    SUM_COLS = [c for c in SUM_COLS if c in joined.columns]
    feats_2km = joined.groupby("grid_id_2km")[SUM_COLS].sum().reset_index()
    feats_2km = feats_2km.rename(columns={"grid_id_2km":"grid_id"})
    feats_2km.to_csv(PROC / "grid_static_features_2km.csv", index=False)
    print(f"Saved 2km features ({len(feats_2km)} rows)")

    # Compute Gini and CV at each scale
    rows = []
    for col in ["total_employment","population","poi_total","subway_station_count"]:
        v1 = feats_1km[col].values.astype(float)
        v2 = feats_2km[col].values.astype(float)
        rows.append({
            "variable": col,
            "n_1km": len(v1),
            "mean_1km": v1.mean(),
            "max_1km": v1.max(),
            "gini_1km": gini(v1),
            "cv_1km": v1.std() / max(v1.mean(), 1e-8),
            "n_2km": len(v2),
            "mean_2km": v2.mean(),
            "max_2km": v2.max(),
            "gini_2km": gini(v2),
            "cv_2km": v2.std() / max(v2.mean(), 1e-8),
        })
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv(EVAL / "maup_distributions.csv", index=False)

    # Plot Gini + CV comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(rows))
    width = 0.35

    g1 = df["gini_1km"].values; g2 = df["gini_2km"].values
    axes[0].bar(x - width/2, g1, width, label="1 km", color="#1976d2")
    axes[0].bar(x + width/2, g2, width, label="2 km", color="#388e3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["variable"].str.replace("_","\n"), fontsize=9)
    axes[0].set_ylabel("Gini coefficient")
    axes[0].set_title("Inequality (Gini): 1 km vs 2 km grid")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(g1, g2)):
        axes[0].text(i - width/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        axes[0].text(i + width/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)

    cv1 = df["cv_1km"].values; cv2 = df["cv_2km"].values
    axes[1].bar(x - width/2, cv1, width, label="1 km", color="#1976d2")
    axes[1].bar(x + width/2, cv2, width, label="2 km", color="#388e3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["variable"].str.replace("_","\n"), fontsize=9)
    axes[1].set_ylabel("Coefficient of variation")
    axes[1].set_title("Spatial dispersion (CV): 1 km vs 2 km grid")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(cv1, cv2)):
        axes[1].text(i - width/2, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
        axes[1].text(i + width/2, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = EVAL / "maup_distributions.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
