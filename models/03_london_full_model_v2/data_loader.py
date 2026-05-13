"""Data loader — loads all v2 data, builds tensors for training.

Input files (processed):
- grid_static_features.csv
- grid_hourly_congestion.csv
- grid_od_2021.csv
- grid_borough_mapping.csv
- london_1km_grid.geojson  (for centroids in BNG)
- nts_commute_departure_time.csv  (for synthesizing F_ij^t)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import torch

V2_ROOT = Path(__file__).resolve().parent
PROC = V2_ROOT / "data" / "processed"


def load_v2_data():
    """Load all data, return a dict of tensors / arrays / dataframes ready for use."""
    # 1. Grid (BNG centroids)
    grid_geo = gpd.read_file(PROC / "london_1km_grid.geojson").to_crs(epsg=27700)
    grid_geo["x_bng"] = grid_geo.geometry.centroid.x
    grid_geo["y_bng"] = grid_geo.geometry.centroid.y
    grid_geo = grid_geo.sort_values("grid_id").reset_index(drop=True)
    coords_bng = grid_geo[["x_bng", "y_bng"]].values.astype(np.float32)

    # 2. Static features + temporal
    from providers.features import LondonFeatureProvider
    feat_provider = LondonFeatureProvider()
    # Verify alignment
    assert feat_provider.grid_ids() == grid_geo["grid_id"].tolist(), \
        "Grid order mismatch between geo and static features"

    # 3. Borough mapping (for travel time provider)
    borough_map = pd.read_csv(PROC / "grid_borough_mapping.csv")
    borough_map = borough_map.set_index("grid_id").reindex(grid_geo["grid_id"])
    boroughs = sorted(borough_map["borough"].unique())
    borough_to_idx = {b: i for i, b in enumerate(boroughs)}
    grid_borough_idx = borough_map["borough"].map(borough_to_idx).values.astype(np.int64)

    # 4. Hourly congestion (per borough)
    cong = pd.read_csv(PROC / "grid_hourly_congestion.csv")
    cong = cong.merge(borough_map.reset_index(), on="grid_id", how="left")
    borough_hourly = cong.groupby(["borough", "hour"])["congestion_ratio"].mean().reset_index()
    bh_pivot = borough_hourly.pivot(index="borough", columns="hour", values="congestion_ratio")
    bh_pivot = bh_pivot.reindex(boroughs)
    borough_hourly_arr = bh_pivot.fillna(1.2).values.astype(np.float32)  # (n_boroughs, 24)

    # 5. OD ground truth
    od = pd.read_csv(PROC / "grid_od_2021.csv")
    grid_to_idx = {g: i for i, g in enumerate(grid_geo["grid_id"])}
    od["i"] = od["grid_home"].map(grid_to_idx)
    od["j"] = od["grid_work"].map(grid_to_idx)
    od = od.dropna(subset=["i", "j"])
    od["i"] = od["i"].astype(int)
    od["j"] = od["j"].astype(int)
    print(f"OD pairs: {len(od)}, total flow: {od['count'].sum():,.0f}")

    # 6. NTS departure-time → π(t) per mode
    nts = pd.read_csv(PROC / "nts_commute_departure_time.csv", comment="#")
    # Use 'all' mode for now (since we don't have mode-specific OD)
    pi_t = nts[nts["mode"] == "all"].sort_values("hour")["share"].values  # (24,)
    # Normalize
    pi_t = pi_t / pi_t.sum()
    print(f"Departure time profile (all mode), peak hour: {np.argmax(pi_t)} ({pi_t.max():.3f})")

    # 7. Synthesize F_ij^t = F_ij × π(t)
    N = len(grid_geo)
    T = 24
    F_ij = np.zeros((N, N), dtype=np.float32)
    for _, row in od.iterrows():
        F_ij[row["i"], row["j"]] += row["count"]
    F_ij_t = F_ij[None, :, :] * pi_t[:, None, None]  # (T, N, N)
    print(f"F_ij_t shape: {F_ij_t.shape}, sum: {F_ij_t.sum():,.0f}")

    return {
        "grid_ids": grid_geo["grid_id"].tolist(),
        "coords_bng": coords_bng,
        "feat_provider": feat_provider,
        "boroughs": boroughs,
        "grid_borough_idx": grid_borough_idx,
        "borough_hourly_congestion": borough_hourly_arr,
        "F_ij_t": F_ij_t.astype(np.float32),
        "pi_t": pi_t.astype(np.float32),
        "N": N, "T": T,
    }


def build_features_tensor(feat_provider, T=24):
    """Stack static + temporal into (T, N, F) tensor."""
    static = feat_provider.get_static()  # (N, F_static)
    N, F_static = static.shape
    seq = np.zeros((T, N, F_static + 4), dtype=np.float32)
    for t in range(T):
        temporal = feat_provider.get_temporal(t)  # (N, 4)
        seq[t, :, :F_static] = static
        seq[t, :, F_static:] = temporal
    return torch.tensor(seq)


def spatial_holdout(N: int, train_frac=0.75, val_frac=0.10, seed=42):
    """Random spatial holdout: train / val / test masks."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_mask = np.zeros(N, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(N, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(N, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
