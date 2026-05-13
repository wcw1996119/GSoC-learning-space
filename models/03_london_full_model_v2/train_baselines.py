"""Train and evaluate the 4 OD-flow baselines on a spatial+temporal holdout.

Outputs:
    evaluation_outputs/baselines_holdout_metrics.csv

Usage:
    python train_baselines.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models_lib.baselines.spatial_holdout import make_holdout, grid_index_map
from models_lib.baselines.metrics import all_metrics
from models_lib.baselines.gravity import GravityModel
from models_lib.baselines.radiation import RadiationModel
from models_lib.baselines.mlp_baseline import MLPBaseline
from models_lib.baselines.deep_gravity import DeepGravity


PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "evaluation_outputs"
OUT.mkdir(exist_ok=True)

EVAL_HOUR = 8           # main hour-block for headline metrics
HOLDOUT_HOURS = (5, 14, 22)
N_HOLDOUT_BOROUGHS = 4
SEED = 42


# ---------- data loading helpers ---------------------------------------- #

def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_data() -> Dict[str, object]:
    grid = pd.read_csv(DATA / "grid_static_features.csv")
    od = pd.read_csv(DATA / "grid_od_2021.csv")
    borough = pd.read_csv(DATA / "grid_borough_mapping.csv")

    cache = np.load(DATA / "demo_cache.npz", allow_pickle=True)
    if "t_ij_t" in cache.files:
        t_ij_t = cache["t_ij_t"]   # (T, N, N)
    else:
        # Fallback: free-flow car matrix tiled across 24 hours.
        car = np.load(DATA / "car_freeflow_t_ij.npy")
        t_ij_t = np.broadcast_to(car, (24, *car.shape)).copy()

    # Stable index in the same order as grid_static_features rows.
    g_idx = grid_index_map(grid["grid_id"].tolist())

    # Pairwise haversine distance for all grids (km).
    lat = grid["centroid_lat"].to_numpy()
    lon = grid["centroid_lon"].to_numpy()
    d_full = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])

    return {
        "grid": grid,
        "od": od,
        "borough": borough,
        "t_ij_t": t_ij_t,
        "g_idx": g_idx,
        "d_full": d_full,
    }


def od_with_features(
    od: pd.DataFrame,
    grid: pd.DataFrame,
    g_idx: Dict[str, int],
    t_mat: np.ndarray,
    d_full: np.ndarray,
) -> pd.DataFrame:
    """Attach origin/destination indices, t_ij, d_ij, O_i, D_j to each OD row."""
    pop = grid.set_index("grid_id")["population"].to_dict()
    emp = grid.set_index("grid_id")["total_employment"].to_dict()

    df = od.copy()
    df = df[df["grid_home"].isin(g_idx) & df["grid_work"].isin(g_idx)].reset_index(drop=True)
    df["oi"] = df["grid_home"].map(g_idx).astype(int)
    df["dj"] = df["grid_work"].map(g_idx).astype(int)
    df["t_ij"] = t_mat[df["oi"].to_numpy(), df["dj"].to_numpy()]
    df["d_ij"] = d_full[df["oi"].to_numpy(), df["dj"].to_numpy()]
    df["O_i"] = df["grid_home"].map(pop).astype(float).fillna(0.0)
    df["D_j"] = df["grid_work"].map(emp).astype(float).fillna(0.0)
    return df


def build_node_features(grid: pd.DataFrame) -> np.ndarray:
    cols = [
        "total_employment", "population",
        "sec1_primary", "sec2_manufacturing", "sec3_construction",
        "sec4_retail", "sec5_fnb", "sec6_info_finance",
        "sec7_public", "sec8_other",
    ]
    cols = [c for c in cols if c in grid.columns]
    X = grid[cols].to_numpy(dtype=np.float32)
    # log1p + standardise (column-wise) to keep features comparable.
    X = np.log1p(np.maximum(X, 0.0))
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sd


# ---------- main routine ------------------------------------------------ #

def main() -> None:
    print("[load] reading data...")
    d = load_data()
    grid: pd.DataFrame = d["grid"]
    od: pd.DataFrame = d["od"]
    borough: pd.DataFrame = d["borough"]
    t_ij_t: np.ndarray = d["t_ij_t"]
    g_idx: Dict[str, int] = d["g_idx"]
    d_full: np.ndarray = d["d_full"]

    holdout = make_holdout(borough, od, n_holdout_boroughs=N_HOLDOUT_BOROUGHS,
                            holdout_hours=HOLDOUT_HOURS, seed=SEED)
    print(f"[holdout] boroughs held out: {holdout['holdout_boroughs']}")
    print(f"[holdout] hours held out:    {holdout['holdout_hours']}")

    node_feat = build_node_features(grid)

    rows: List[Dict[str, object]] = []

    for hour in (EVAL_HOUR, *HOLDOUT_HOURS):
        t_mat = t_ij_t[hour]
        df = od_with_features(od, grid, g_idx, t_mat, d_full)

        # Restrict to OD rows that appear in the original od_df ordering so masks align.
        train_mask = holdout["train_mask"][: len(df)]
        test_mask = holdout["test_mask"][: len(df)]

        train = df.loc[train_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        if len(train) == 0 or len(test) == 0:
            print(f"[hour={hour}] empty split, skipping")
            continue

        block = "holdout_hour" if hour in HOLDOUT_HOURS else "eval_hour"
        borough_block = "+".join(holdout["holdout_boroughs"])

        # ---- 1. Gravity (Wilson, Poisson) ----
        print(f"[hour={hour}] fitting Gravity...")
        gm = GravityModel().fit(
            train["O_i"].to_numpy(), train["D_j"].to_numpy(),
            train["t_ij"].to_numpy(), train["count"].to_numpy(),
        )
        pred = gm.predict(
            test["O_i"].to_numpy(), test["D_j"].to_numpy(),
            test["t_ij"].to_numpy(),
            origins=test["grid_home"].to_numpy(),
            dests=test["grid_work"].to_numpy(),
        )
        m = all_metrics(test["grid_home"].to_numpy(), test["count"].to_numpy(), pred)
        rows.append({"model": "gravity", "hour_block": block, "hour": hour,
                     "borough_block": borough_block, **m})

        # ---- 2. Radiation (Simini 2012) ----
        print(f"[hour={hour}] fitting Radiation...")
        rm = RadiationModel().fit(grid)
        # Use observed origin totals from the *training* set as T_i; for
        # test origins not seen in train, fall back to grid population.
        T_train = train.groupby("grid_home")["count"].sum().to_dict()
        T_test = test["grid_home"].map(T_train).fillna(
            test["grid_home"].map(grid.set_index("grid_id")["population"].to_dict())
        ).to_numpy(dtype=float)
        pred = rm.predict(test["grid_home"].to_numpy(),
                          test["grid_work"].to_numpy(), T_test)
        m = all_metrics(test["grid_home"].to_numpy(), test["count"].to_numpy(), pred)
        rows.append({"model": "radiation", "hour_block": block, "hour": hour,
                     "borough_block": borough_block, **m})

        # ---- 3. MLP baseline ----
        print(f"[hour={hour}] fitting MLP...")
        mlp = MLPBaseline(epochs=200, seed=SEED).fit(
            train["oi"].to_numpy(), train["dj"].to_numpy(), node_feat,
            train["t_ij"].to_numpy(), train["d_ij"].to_numpy(),
            train["count"].to_numpy(),
        )
        pred = mlp.predict(test["oi"].to_numpy(), test["dj"].to_numpy(),
                           node_feat, test["t_ij"].to_numpy(),
                           test["d_ij"].to_numpy())
        m = all_metrics(test["grid_home"].to_numpy(), test["count"].to_numpy(), pred)
        rows.append({"model": "mlp", "hour_block": block, "hour": hour,
                     "borough_block": borough_block, **m})

        # ---- 4. Deep Gravity (Simini 2021) ----
        print(f"[hour={hour}] fitting Deep Gravity...")
        dg = DeepGravity(epochs=100, seed=SEED).fit(
            train["oi"].to_numpy(), train["dj"].to_numpy(), node_feat,
            train["t_ij"].to_numpy(), train["d_ij"].to_numpy(),
            train["count"].to_numpy(),
        )
        pred = dg.predict(test["oi"].to_numpy(), test["dj"].to_numpy(),
                          node_feat, test["t_ij"].to_numpy(),
                          test["d_ij"].to_numpy(), T_test)
        m = all_metrics(test["grid_home"].to_numpy(), test["count"].to_numpy(), pred)
        rows.append({"model": "deep_gravity", "hour_block": block, "hour": hour,
                     "borough_block": borough_block, **m})

    out_df = pd.DataFrame(rows)
    out_path = OUT / "baselines_holdout_metrics.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")

    # Pretty print headline table at EVAL_HOUR.
    head = out_df[out_df["hour"] == EVAL_HOUR][
        ["model", "CPC", "MAE_log1p", "Spearman", "KL_per_origin"]
    ].copy()
    head.columns = ["model", "CPC", "MAE", "Spearman", "KL"]
    print("\n=== Headline (hour=8) ===")
    print(head.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
