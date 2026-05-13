"""Diagnose whether the OSMnx car matrix gives realistic urban speeds.

Two checks:
  1. Effective speed distribution: t_ij / euclidean_distance over all OD pairs.
     Urban London with peak congestion should sit around 18-25 km/h average.
  2. Spot-check known OD pairs against published / known typical drive times.

Outputs to stdout only (no plot / no saved files).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

V2_ROOT = Path(__file__).resolve().parent
PROC = V2_ROOT / "data" / "processed"

WELL_KNOWN_PAIRS = [
    # name, lat_o, lon_o, lat_d, lon_d, expected_drive_min_peak
    ("Heathrow → Westminster",            51.4700, -0.4543, 51.4974, -0.1352, 50),  # ~22km, 50min
    ("Stratford → Canary Wharf",          51.5416, -0.0033, 51.5054, -0.0235, 18),  # ~5km
    ("Wimbledon → Hyde Park",             51.4214, -0.2065, 51.5074, -0.1657, 35),  # ~12km
    ("Camden → London Bridge",            51.5390, -0.1426, 51.5048, -0.0865, 22),  # ~6km
    ("Croydon → Kings Cross",             51.3762, -0.0982, 51.5308, -0.1238, 60),  # ~18km
    ("Richmond → Stratford",              51.4613, -0.3037, 51.5416, -0.0033, 65),  # ~22km
    ("Greenwich → Notting Hill",          51.4825, -0.0076, 51.5085, -0.1953, 55),  # ~15km
]


def closest_grid_idx(grid_df, lat, lon):
    d2 = (grid_df["centroid_lat"] - lat) ** 2 + (grid_df["centroid_lon"] - lon) ** 2
    return int(d2.idxmin())


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    grid_df = pd.read_csv(PROC / "grid_static_features.csv")[["grid_id", "centroid_lat", "centroid_lon"]].copy().reset_index(drop=True)
    t0 = np.load(PROC / "car_freeflow_t_ij.npy")  # free-flow minutes
    cong_long = pd.read_csv(PROC / "grid_hourly_congestion.csv")
    cong_wide = cong_long.pivot(index="grid_id", columns="hour", values="congestion_ratio").reindex(grid_df["grid_id"].tolist()).fillna(1.20)
    cong_jt = cong_wide.to_numpy().astype(np.float32)
    t_peak = t0 * cong_jt[None, :, 8]  # destination-side congestion at 8am
    N = len(grid_df)

    # ---- 1. Effective speed distribution (peak hour) ----
    print("=== 1. Effective speed distribution at peak (8am) ===")
    lats = grid_df["centroid_lat"].to_numpy()
    lons = grid_df["centroid_lon"].to_numpy()
    # Pairwise haversine (vectorised)
    lat1 = np.radians(lats[:, None]); lat2 = np.radians(lats[None, :])
    dlat = lat2 - lat1
    dlon = np.radians(lons[None, :] - lons[:, None])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))  # (N, N)
    mask = (~np.eye(N, dtype=bool)) & (t_peak > 0.5) & (d_km > 1.0)  # filter near-zero
    t_minutes = t_peak[mask]
    d_km_arr = d_km[mask]
    speed_kmh = d_km_arr / (t_minutes / 60.0)
    print(f"  n filtered OD pairs: {mask.sum()} / {N*(N-1)}")
    print(f"  Effective speed:  mean={speed_kmh.mean():.1f}, median={np.median(speed_kmh):.1f}, "
          f"p10={np.percentile(speed_kmh,10):.1f}, p90={np.percentile(speed_kmh,90):.1f} km/h")
    print(f"  Reference: London peak driving ~18-25 km/h; off-peak ~30-40 km/h")
    print()

    # Time per km
    print("  Time per km (peak):")
    print(f"    mean   = {(t_minutes/d_km_arr).mean():.2f} min/km")
    print(f"    median = {np.median(t_minutes/d_km_arr):.2f} min/km  ← target: 2-3 min/km")
    print()

    # ---- 2. Spot-check known OD pairs ----
    print("=== 2. Spot-check known OD pairs (peak 8am) ===")
    print(f"  {'route':<40s} {'eucl_km':>8s} {'model_min':>10s} {'expected':>10s} {'effective_kmh':>14s}")
    for name, lat_o, lon_o, lat_d, lon_d, expected_min in WELL_KNOWN_PAIRS:
        i = closest_grid_idx(grid_df, lat_o, lon_o)
        j = closest_grid_idx(grid_df, lat_d, lon_d)
        d_km_pair = haversine_km(lat_o, lon_o, lat_d, lon_d)
        t_model = float(t_peak[i, j])
        eff_speed = d_km_pair / (t_model / 60.0) if t_model > 0 else 0
        print(f"  {name:<40s} {d_km_pair:8.1f} {t_model:10.1f} {expected_min:10d} {eff_speed:14.1f}")

    print()
    print("Target: model_min within 30% of expected; effective_kmh between 18-30.")


if __name__ == "__main__":
    main()
