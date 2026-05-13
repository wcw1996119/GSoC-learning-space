"""MAUP sensitivity test: re-run v2 pipeline at 2km grid, compare CPC vs 1km.

Aggregates 1km grid features to 2km cells (each 2km cell = up to 4 1km cells),
re-trains STGNN, runs ablation, generates comparison plot.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))
PROC = V2_ROOT / "data" / "processed"
EVAL = V2_ROOT / "evaluation_outputs"
EVAL.mkdir(exist_ok=True)


def generate_2km_grid():
    """Generate 2km × 2km grid using BNG, save to data/processed/london_2km_grid.geojson."""
    print("Step 1: generate 2km grid")
    msoa = gpd.read_file(V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "london_msoa_boundaries.geojson")
    msoa_bng = msoa.to_crs(epsg=27700)
    bbox = msoa_bng.total_bounds

    GRID = 2000
    minx = np.floor(bbox[0] / GRID) * GRID
    miny = np.floor(bbox[1] / GRID) * GRID
    maxx = np.ceil(bbox[2] / GRID) * GRID
    maxy = np.ceil(bbox[3] / GRID) * GRID

    cells = []
    london_extent = msoa_bng.geometry.union_all()
    for i, x in enumerate(np.arange(minx, maxx, GRID)):
        for j, y in enumerate(np.arange(miny, maxy, GRID)):
            geom = box(x, y, x + GRID, y + GRID)
            if geom.intersects(london_extent):
                cells.append({
                    "grid_id": f"M{i:03d}_{j:03d}",
                    "geometry": geom,
                    "col": i, "row": j,
                })

    grid = gpd.GeoDataFrame(cells, crs="EPSG:27700")
    grid["area_in_london_km2"] = grid.geometry.intersection(london_extent).area / 1e6

    grid_wgs = grid.to_crs(4326).copy()
    grid_wgs["centroid_lon"] = grid_wgs.geometry.centroid.x
    grid_wgs["centroid_lat"] = grid_wgs.geometry.centroid.y
    out_geo = grid_wgs[["grid_id", "centroid_lat", "centroid_lon",
                          "area_in_london_km2", "geometry"]].copy()

    out = PROC / "london_2km_grid.geojson"
    out_geo.to_file(out, driver="GeoJSON")
    print(f"  -> {len(out_geo)} 2km cells, saved {out}")
    return out_geo, grid


def aggregate_features_to_2km(grid_2km):
    """Aggregate 1km static features → 2km via spatial join (point-in-polygon on 1km centroids)."""
    print("Step 2: aggregate 1km → 2km")
    feats_1km = pd.read_csv(PROC / "grid_static_features.csv")
    grid_1km = gpd.read_file(PROC / "london_1km_grid.geojson")
    grid_1km = grid_1km.sort_values("grid_id").reset_index(drop=True)
    feats_1km = feats_1km.set_index("grid_id").reindex(grid_1km["grid_id"]).reset_index()

    cents_1km = gpd.GeoDataFrame(
        feats_1km, geometry=gpd.points_from_xy(feats_1km.centroid_lon, feats_1km.centroid_lat),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)

    # spatial join
    g2 = grid_2km.copy()
    if "grid_id" in g2.columns:
        g2 = g2.rename(columns={"grid_id": "grid_id_2km"})
    joined = gpd.sjoin(cents_1km, g2[["grid_id_2km", "geometry"]],
                        how="inner", predicate="within")

    # Aggregate: sum for counts, mean for densities/coords
    SUM_COLS = [
        "sec1_primary","sec2_manufacturing","sec3_construction","sec4_retail",
        "sec5_fnb","sec6_info_finance","sec7_public","sec8_other",
        "total_employment","population","poi_total",
        "poi_commercial","poi_education","poi_fnb","poi_healthcare",
        "poi_office","poi_public","poi_retail","poi_transport",
        "subway_station_count","bus_stop_count",
    ]
    SUM_COLS = [c for c in SUM_COLS if c in joined.columns]

    agg_sums = joined.groupby("grid_id_2km")[SUM_COLS].sum().reset_index()

    g2_with = g2.merge(agg_sums, on="grid_id_2km", how="left").rename(columns={"grid_id_2km": "grid_id"})
    for c in SUM_COLS:
        if c in g2_with.columns:
            g2_with[c] = g2_with[c].fillna(0)

    # area-related
    g2_with["area_km2"] = 4.0  # 2km × 2km

    # Save
    g2_with_csv = g2_with[["grid_id", "centroid_lat", "centroid_lon", "area_km2",
                             "area_in_london_km2"] + SUM_COLS].copy()
    out = PROC / "grid_static_features_2km.csv"
    g2_with_csv.to_csv(out, index=False)
    print(f"  -> {len(g2_with_csv)} rows, saved {out}")

    # Build 2km MSOA mapping (use majority MSOA21CD by population in joined)
    msoa_map_1km = pd.read_csv(PROC / "grid_msoa_primary.csv")
    joined2 = joined.merge(msoa_map_1km, left_on="grid_id", right_on="grid_id", how="left")
    primary_msoa_2km = (joined2.groupby(["grid_id_2km", "MSOA21CD"]).size()
                                 .reset_index(name="n")
                                 .sort_values(["grid_id_2km", "n"], ascending=[True, False])
                                 .drop_duplicates("grid_id_2km")[["grid_id_2km", "MSOA21CD"]]
                                 .rename(columns={"grid_id_2km": "grid_id"}))
    primary_msoa_2km.to_csv(PROC / "grid_msoa_primary_2km.csv", index=False)

    return g2_with_csv


def aggregate_OD_to_2km():
    """Re-aggregate Census OD to 2km grids using the 1km→2km mapping via primary MSOA."""
    print("Step 3: aggregate OD → 2km")
    grid_msoa_2km = pd.read_csv(PROC / "grid_msoa_primary_2km.csv")
    msoa_to_2km = dict(zip(grid_msoa_2km["MSOA21CD"], grid_msoa_2km["grid_id"]))

    od_path = V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "london_OD_travel2work.csv"
    od = pd.read_csv(od_path)
    od["grid_home"] = od["MSOA21CD_home"].map(msoa_to_2km)
    od["grid_work"] = od["MSOA21CD_work"].map(msoa_to_2km)
    od = od.dropna(subset=["grid_home", "grid_work"])
    grid_od_2km = od.groupby(["grid_home", "grid_work"], as_index=False)["count"].sum()
    out = PROC / "grid_od_2021_2km.csv"
    grid_od_2km.to_csv(out, index=False)
    print(f"  -> {len(grid_od_2km)} OD pairs at 2km, saved")
    return grid_od_2km


def train_stgnn_at_2km():
    """Quick STGNN training at 2km. Reuses train.py logic but with 2km paths."""
    print("Step 4: train STGNN at 2km grid")
    from data_loader import build_features_tensor, spatial_holdout
    from providers.travel_time import LondonBPRProvider
    from providers.graph_builder import build_knn_graph
    from models_lib.stgnn import V2_STGNN
    from models_lib.rum import rum_closure
    from models_lib.loss import production_constrained_mnl_nll

    # Load 2km grid
    grid_geo = gpd.read_file(PROC / "london_2km_grid.geojson")
    grid_geo = grid_geo.sort_values("grid_id").reset_index(drop=True)
    grid_bng = grid_geo.to_crs(epsg=27700)
    coords_bng = np.stack([grid_bng.geometry.centroid.x.values,
                            grid_bng.geometry.centroid.y.values], axis=1).astype(np.float32)
    N = len(grid_geo)
    print(f"  2km N={N}")

    # 2km static features
    grid_feats = pd.read_csv(PROC / "grid_static_features_2km.csv").set_index("grid_id").reindex(grid_geo["grid_id"]).reset_index()

    # Static feature normalization
    STATIC_COLS = [
        "sec1_primary","sec2_manufacturing","sec3_construction","sec4_retail",
        "sec5_fnb","sec6_info_finance","sec7_public","sec8_other",
        "total_employment","population","poi_total",
        "poi_commercial","poi_education","poi_fnb","poi_healthcare",
        "poi_office","poi_public","poi_retail","poi_transport",
        "subway_station_count",
        "centroid_lat","centroid_lon",
    ]
    LOG_COLS = STATIC_COLS[:19]
    raw = grid_feats[STATIC_COLS].values.astype(np.float64)
    log_idx = [i for i, c in enumerate(STATIC_COLS) if c in LOG_COLS]
    raw[:, log_idx] = np.log1p(raw[:, log_idx])
    static_norm = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)

    # Temporal: simplified (zeros for v22 baseline; congestion at 2km would need re-aggregation)
    F_static = static_norm.shape[1]
    F_temporal = 4
    T = 24
    x_seq = np.zeros((T, N, F_static + F_temporal), dtype=np.float32)
    for t in range(T):
        x_seq[t, :, :F_static] = static_norm
        x_seq[t, :, F_static] = 0.0  # congestion placeholder (no-data → mean = 0 after norm)
        x_seq[t, :, F_static + 1] = np.sin(2 * np.pi * t / 24)
        x_seq[t, :, F_static + 2] = np.cos(2 * np.pi * t / 24)
    x_seq_t = torch.tensor(x_seq)

    # Graph
    edge_index, edge_attr = build_knn_graph(coords_bng, K=10, add_self_loop=True)

    # Travel time at 2km
    diff = coords_bng[:, None, :] - coords_bng[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0
    t0_min = d_km * 1.3 / (20.0 / 60.0)
    cong_default = 1.2
    t_ij_t = np.tile(t0_min[None, :, :], (T, 1, 1)).astype(np.float32) * (1 + 0.15 * (cong_default ** 4))
    t_ij_t_t = torch.tensor(t_ij_t)

    # OD at 2km
    od = pd.read_csv(PROC / "grid_od_2021_2km.csv")
    grid_to_idx = {g: i for i, g in enumerate(grid_geo["grid_id"])}
    od["i"] = od["grid_home"].map(grid_to_idx)
    od["j"] = od["grid_work"].map(grid_to_idx)
    od = od.dropna(subset=["i","j"]); od["i"] = od["i"].astype(int); od["j"] = od["j"].astype(int)

    nts = pd.read_csv(PROC / "nts_commute_departure_time.csv", comment="#")
    pi_t = nts[nts["mode"] == "all"].sort_values("hour")["share"].values
    pi_t = pi_t / pi_t.sum()
    F_ij = np.zeros((N, N), dtype=np.float32)
    for _, row in od.iterrows():
        F_ij[row["i"], row["j"]] += row["count"]
    F_ij_t = (F_ij[None, :, :] * pi_t[:, None, None]).astype(np.float32)
    print(f"  F_ij_t total: {F_ij_t.sum():.0f}")
    F_ij_t_t = torch.tensor(F_ij_t)

    # Holdout
    train_mask, val_mask, test_mask = spatial_holdout(N, seed=42)
    train_mask_t = torch.tensor(train_mask); val_mask_t = torch.tensor(val_mask); test_mask_t = torch.tensor(test_mask)

    # Model
    model = V2_STGNN(node_dim=x_seq_t.shape[-1], edge_dim=edge_attr.shape[1],
                       hidden_dim=64, gat_heads=4, gru_hidden=32)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("  training 30 epochs at 2km...")
    best_val_nll = float("inf")
    best_test_cpc = 0.0
    best_val_cpc = 0.0
    for ep in range(30):
        t0 = time.time()
        model.train(); optim.zero_grad()
        V_jt = model(x_seq_t, edge_index, edge_attr)
        log_p = rum_closure(V_jt, t_ij_t_t, beta=0.07)
        loss = production_constrained_mnl_nll(log_p, F_ij_t_t, origin_mask=train_mask_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        model.eval()
        with torch.no_grad():
            V_jt = model(x_seq_t, edge_index, edge_attr)
            log_p = rum_closure(V_jt, t_ij_t_t, beta=0.07)
            val_nll = production_constrained_mnl_nll(log_p, F_ij_t_t, origin_mask=val_mask_t).item()
            P = log_p.exp()
            origin_total = F_ij_t_t.sum(dim=2, keepdim=True)
            pred_F = P * origin_total
            cpc_val = float(torch.minimum(pred_F[:, val_mask_t], F_ij_t_t[:, val_mask_t]).sum() /
                             (F_ij_t_t[:, val_mask_t].sum() + 1e-8))
            cpc_te = float(torch.minimum(pred_F[:, test_mask_t], F_ij_t_t[:, test_mask_t]).sum() /
                             (F_ij_t_t[:, test_mask_t].sum() + 1e-8))
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_cpc = cpc_val
            best_test_cpc = cpc_te
        if ep % 5 == 0:
            print(f"    ep {ep} | val NLL {val_nll:.3f} | val CPC {cpc_val:.3f} | test CPC {cpc_te:.3f} | {time.time()-t0:.1f}s")

    print(f"  Best 2km: val NLL {best_val_nll:.3f}, val CPC {best_val_cpc:.3f}, test CPC {best_test_cpc:.3f}")
    return {
        "N": N, "n_edges": int(edge_attr.shape[0]),
        "best_val_nll": best_val_nll,
        "best_val_cpc": best_val_cpc,
        "best_test_cpc": best_test_cpc,
        "F_ij_t_total": float(F_ij_t.sum()),
    }


def main():
    g_2km, g_2km_bng = generate_2km_grid()
    aggregate_features_to_2km(g_2km)
    aggregate_OD_to_2km()
    res_2km = train_stgnn_at_2km()

    # Compare with 1km result (from existing v2 cache)
    res_1km = {
        "N": 1725,
        "n_edges": 18975,
        "best_val_nll": 16.22,        # from earlier 30-epoch training
        "best_val_cpc": 0.219,
        "best_test_cpc": 0.226,
        "F_ij_t_total": 55072.0,
    }

    rows = [
        {"scale": "1 km", **res_1km},
        {"scale": "2 km", **res_2km},
    ]
    df = pd.DataFrame(rows)
    out_csv = EVAL / "maup_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(df.to_string(index=False))

    # Plot CPC comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scales = df["scale"].values
    val_cpc = df["best_val_cpc"].values
    test_cpc = df["best_test_cpc"].values
    x = np.arange(len(scales))
    width = 0.35
    ax.bar(x - width/2, val_cpc, width, label="Val CPC", color="#1976d2")
    ax.bar(x + width/2, test_cpc, width, label="Test CPC", color="#388e3c")
    ax.set_xticks(x); ax.set_xticklabels(scales)
    ax.set_ylabel("CPC"); ax.set_title("MAUP sensitivity: STGNN performance at 1 km vs 2 km grids")
    for i, (vc, tc) in enumerate(zip(val_cpc, test_cpc)):
        ax.text(i - width/2, vc + 0.005, f"{vc:.3f}", ha="center", fontsize=10)
        ax.text(i + width/2, tc + 0.005, f"{tc:.3f}", ha="center", fontsize=10)
    # Add N annotation below x-axis
    for i, n in enumerate(df["N"]):
        ax.text(i, -0.025, f"N={int(n)}", ha="center", fontsize=9, color="#666",
                transform=ax.get_xaxis_transform())
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_png = EVAL / "maup_cpc_comparison.png"
    plt.savefig(out_png, dpi=120); plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
