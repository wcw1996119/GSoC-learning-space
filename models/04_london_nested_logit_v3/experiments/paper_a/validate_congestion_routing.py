"""Route-assignment congestion validation.

Implements the proper validation pipeline that converts model-predicted OD
flow into per-grid through-traffic via shortest-path routing on the OSM drive
network, then correlates against TomTom congestion.

Pipeline:
  1. Load trained model, forward → F_pred[t, i, j] (predicted OD flow)
  2. Load OSM drive graph + grid-to-OSM-node centroid mapping (from v2)
  3. KDTree-snap every OSM node to its nearest grid → node_to_grid[node_idx]
  4. Multi-source Dijkstra from each origin grid's centroid node
  5. For each (i, j) pair, reconstruct shortest path → list of grids on path
  6. Accumulate: through_flow[g, t] += F_pred[t, i, j] for each g on path(i, j)
  7. Correlate through_flow with TomTom congestion_ratio, restricted to
     instrumented boroughs (where TomTom has real data, not 1.2 fallback)

Two reference correlations reported:
  - observed-OD through-flow vs congestion  (sanity baseline)
  - model-predicted through-flow vs congestion  (the validation target)

If the through-flow approach is correct, both should correlate substantially
better than the naive inflow-based validation (which got r ≈ 0.21 in v2).

Usage:
    python experiments/paper_a/validate_congestion_routing.py \\
        --ckpt evaluation_outputs/paper_a/v3l_matchcut050_s0.pt \\
        --out evaluation_outputs/paper_a/v3l_route_val_s0.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"

if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "v3_train_cervero_shen",
    V3_ROOT / "experiments" / "paper_a" / "train_cervero_shen.py",
)
_trainer = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_trainer)
forward_cs = _trainer.forward_cs

_spec_sa = _ilu.spec_from_file_location(
    "v3l_scenario_A_mod",
    V3_ROOT / "experiments" / "paper_a" / "scenario_v3l_A.py",
)
_sa = _ilu.module_from_spec(_spec_sa)
_spec_sa.loader.exec_module(_sa)
reconstruct_model = _sa.reconstruct_model
load_scenario_inputs = _sa.load_scenario_inputs

_spec_cv = _ilu.spec_from_file_location(
    "validate_congestion_v2_mod",
    V3_ROOT / "experiments" / "paper_a" / "validate_congestion_v2.py",
)
_cv = _ilu.module_from_spec(_spec_cv)
_spec_cv.loader.exec_module(_cv)
load_grid_meta = _cv.load_grid_meta
load_grid_congestion = _cv.load_grid_congestion
identify_real_data_boroughs = _cv.identify_real_data_boroughs
commute_share_per_hour = _cv.commute_share_per_hour
correlation_stats = _cv.correlation_stats


def load_drive_graph_csr():
    """Load networkx drive graph and convert to scipy CSR for dijkstra.

    Returns:
      csr: scipy.sparse.csr_matrix (N_nodes, N_nodes) with travel_time weights
      node_ids: list of OSM node ids in CSR index order
      node_to_idx: dict osm_id → csr index
      node_lonlat: (N_nodes, 2) — lon, lat for each node
    """
    from scipy.sparse import csr_matrix

    path = V2_ROOT / "data" / "raw" / "london_drive_graph.pkl"
    print(f"[routing] loading drive graph from {path.name} ...")
    t0 = time.time()
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"          {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
          f"in {time.time()-t0:.1f}s")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # Extract lat/lon (osmnx uses x=lon, y=lat)
    node_lonlat = np.zeros((N, 2), dtype=np.float64)
    for i, n in enumerate(nodes):
        data = G.nodes[n]
        node_lonlat[i] = (data.get("x", 0.0), data.get("y", 0.0))

    # Build edge list. Use travel_time if available, else length/speed_default.
    print("[routing] building CSR ...")
    t0 = time.time()
    row, col, data = [], [], []
    fallback_used = 0
    for u, v, eattr in G.edges(data=True):
        tt = eattr.get("travel_time")
        if tt is None or tt <= 0:
            length = eattr.get("length", 1.0)
            tt = length / 13.4   # 30 mph ≈ 13.4 m/s fallback
            fallback_used += 1
        row.append(node_to_idx[u])
        col.append(node_to_idx[v])
        data.append(float(tt))
    csr = csr_matrix((data, (row, col)), shape=(N, N))
    print(f"          built {csr.nnz} edges in {time.time()-t0:.1f}s  "
          f"(fallback travel_time on {fallback_used} edges)")
    return csr, nodes, node_to_idx, node_lonlat


def load_grid_centroids_lonlat(grid_id_to_idx: dict) -> np.ndarray:
    """Read grid centroids (lat, lon) from grid_static_features.csv. Return (N, 2) lonlat."""
    N = len(grid_id_to_idx)
    out = np.zeros((N, 2), dtype=np.float64)
    static_path = V2_ROOT / "data" / "processed" / "grid_static_features.csv"
    with open(static_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            gid = row["grid_id"]
            if gid in grid_id_to_idx:
                idx = grid_id_to_idx[gid]
                out[idx] = (float(row["centroid_lon"]), float(row["centroid_lat"]))
    return out


def load_grid_to_osm_node(grid_id_to_idx: dict) -> dict:
    """Read grid_to_osm_node.csv → dict grid_idx → osm_node_id."""
    out = {}
    path = V2_ROOT / "data" / "processed" / "grid_to_osm_node.csv"
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            gid = row["grid_id"]
            if gid in grid_id_to_idx:
                out[grid_id_to_idx[gid]] = int(row["osm_node_id"])
    return out


def build_node_to_grid(node_lonlat: np.ndarray, grid_lonlat: np.ndarray) -> np.ndarray:
    """For each OSM node, find the nearest grid (by lon-lat KDTree).
    Returns (N_nodes,) int array of grid indices.
    """
    from scipy.spatial import cKDTree
    print("[routing] building node→grid KDTree (lon, lat naive)...")
    t0 = time.time()
    tree = cKDTree(grid_lonlat)
    _, idx = tree.query(node_lonlat, k=1)
    print(f"          mapped {len(node_lonlat)} nodes to {len(grid_lonlat)} grids in "
          f"{time.time()-t0:.1f}s")
    return idx.astype(np.int32)


def run_multi_source_dijkstra(csr, source_node_indices):
    """Run Dijkstra from each source. Return (dist_mat, pred_mat) both (n_sources, N_nodes)."""
    from scipy.sparse.csgraph import dijkstra
    print(f"[routing] dijkstra from {len(source_node_indices)} sources ...")
    t0 = time.time()
    dist_mat, pred_mat = dijkstra(
        csr, indices=source_node_indices, return_predecessors=True
    )
    print(f"          done in {time.time()-t0:.1f}s  "
          f"dist shape={dist_mat.shape}  reachable={np.isfinite(dist_mat).mean()*100:.1f}%")
    return dist_mat, pred_mat


def accumulate_through_flow(pred_mat, grid_to_node_idx, node_to_grid, F_pred):
    """For each (i, j), reconstruct path and accumulate F[t, i, j] onto every grid on path.

    Args:
      pred_mat:         (N_grids, N_nodes) predecessor matrix from multi-source dijkstra
      grid_to_node_idx: dict grid_idx → osm node csr index
      node_to_grid:     (N_nodes,) grid index per OSM node
      F_pred:           (T, N_grids, N_grids) flow prediction

    Returns:
      through_flow: (N_grids, T) accumulated through-traffic per grid per hour
      path_len_dist: (n_pairs,) — number of grids traversed per (i, j) path (diagnostic)
    """
    T, N, _ = F_pred.shape
    through_flow = np.zeros((N, T), dtype=np.float64)
    path_lens = []

    NULL_PRED = -9999  # scipy uses this for unreachable / source

    grid_node = np.array([grid_to_node_idx[i] for i in range(N)], dtype=np.int64)

    t0 = time.time()
    skipped = 0
    for i in range(N):
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"    origin {i}/{N}  ({elapsed:.0f}s elapsed)")
        pred_i = pred_mat[i]
        src_node = grid_node[i]
        for j in range(N):
            if i == j:
                continue
            F_ij = F_pred[:, i, j]                              # (T,)
            if F_ij.sum() < 1e-3:
                continue                                        # negligible flow, skip
            # Walk back from j's centroid node to source
            curr = grid_node[j]
            if not np.isfinite(pred_mat[i, curr]) and pred_i[curr] == NULL_PRED:
                # Unreachable (no path) — skip
                skipped += 1
                continue
            path_grids = set()
            steps = 0
            while curr != NULL_PRED and curr != src_node and steps < 5000:
                g = int(node_to_grid[curr])
                path_grids.add(g)
                curr = int(pred_i[curr])
                steps += 1
            if curr == src_node:
                path_grids.add(int(node_to_grid[src_node]))
            # path_grids now contains all grids on shortest path i→j
            path_lens.append(len(path_grids))
            # Accumulate flow onto each grid in path
            for g in path_grids:
                through_flow[g] += F_ij
    print(f"    total accumulation time: {time.time()-t0:.0f}s  skipped {skipped} pairs")
    return through_flow, np.array(path_lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[route-val] device: {device}")
    print(f"[route-val] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder, rum, train_args = reconstruct_model(ckpt, device)
    train_args.aux_path = "data/processed/paperA_v3_aux_cervero.npz"
    data = load_scenario_inputs(train_args, device)
    N_grids, T = data["N"], data["T"]
    print(f"[route-val] N_grids={N_grids} T={T}")

    # --- Model forward → F_pred ---
    print("[route-val] forward to get F_pred ...")
    t0 = time.time()
    with torch.no_grad():
        out = forward_cs(
            encoder, rum,
            data["X_static"], data["X_dynamic"], data["edge_index"],
            data["t_per_mode"], data["mode_names"],
            data["log_d_ij"], data["match_prob"],
            data["log_M_z"], data["log_W_z"], data["log_D_z"],
            data["income_score"], data["pct_kids"], data["mean_cars"],
            data["income_tier_props"],
            data["pi_m_pair"], data["grid_borough_idx"],
            data["observed_OD"], data["val_mask"],
        )
        P = out["log_P_D"].exp()
        row_total = data["observed_OD"].sum(dim=2, keepdim=True)
        F_pred = (P * row_total).cpu().numpy().astype(np.float32)
    print(f"        forward {time.time()-t0:.1f}s  F_pred shape={F_pred.shape}  "
          f"total flow={F_pred.sum():.0f}")

    F_obs = data["observed_OD"].cpu().numpy().astype(np.float32)

    # --- Graph + grid metadata ---
    grid_id_to_idx, borough_idx, borough_names = load_grid_meta()
    grid_lonlat = load_grid_centroids_lonlat(grid_id_to_idx)
    grid_to_osm_node = load_grid_to_osm_node(grid_id_to_idx)
    print(f"[route-val] grid_to_osm_node covers {len(grid_to_osm_node)}/{N_grids} grids")

    csr, nodes, node_to_idx_csr, node_lonlat = load_drive_graph_csr()
    node_to_grid_arr = build_node_to_grid(node_lonlat, grid_lonlat)
    print(f"[route-val] node_to_grid built")

    # Map grid_to_osm_node values (raw OSM ids) → csr indices
    grid_to_node_idx = {}
    missing = 0
    for g_idx, osm_id in grid_to_osm_node.items():
        if osm_id in node_to_idx_csr:
            grid_to_node_idx[g_idx] = node_to_idx_csr[osm_id]
        else:
            missing += 1
    if missing > 0:
        print(f"    WARNING: {missing} grids have OSM node not in CSR — these will be skipped")

    # --- Dijkstra ---
    source_idxs = np.array([grid_to_node_idx[i] for i in sorted(grid_to_node_idx)],
                           dtype=np.int64)
    dist_mat, pred_mat = run_multi_source_dijkstra(csr, source_idxs)

    # --- Accumulate through-flow for model + observed ---
    print("\n[route-val] accumulating MODEL through-flow ...")
    through_mod, path_lens = accumulate_through_flow(
        pred_mat, grid_to_node_idx, node_to_grid_arr, F_pred
    )
    print(f"    path-length distribution: mean={path_lens.mean():.1f}  "
          f"median={np.median(path_lens):.1f}  p95={np.percentile(path_lens, 95):.1f}")

    print("\n[route-val] accumulating OBSERVED through-flow (sanity baseline) ...")
    through_obs, _ = accumulate_through_flow(
        pred_mat, grid_to_node_idx, node_to_grid_arr, F_obs
    )

    # --- Load TomTom congestion ---
    cong = load_grid_congestion(grid_id_to_idx)
    real_b_set, b_diag = identify_real_data_boroughs(cong, borough_idx, borough_names)
    real_grid_mask = np.array([b in real_b_set for b in borough_idx])
    print(f"[route-val] kept {int(real_grid_mask.sum())}/{N_grids} grids in "
          f"{len(real_b_set)} instrumented boroughs")

    # NTS commute-attribution
    cs = commute_share_per_hour()
    excess = cong - 1.0
    cong_attr = 1.0 + excess * cs[np.newaxis, :]                         # (N, T)

    # --- Correlation cells ---
    results = {
        "config": {
            "ckpt": args.ckpt,
            "ckpt_cpc": float(ckpt["final_cpc"]),
            "N_grids": int(N_grids),
            "N_instrumented_grids": int(real_grid_mask.sum()),
            "instrumented_boroughs": sorted(borough_names[b] for b in real_b_set),
        },
        "path_stats": {
            "mean": float(path_lens.mean()),
            "median": float(np.median(path_lens)),
            "p95": float(np.percentile(path_lens, 95)),
            "max": int(path_lens.max()),
        },
        "correlations": {},
    }

    def add_corr(key, x, y, label):
        results["correlations"][key] = correlation_stats(x, y, label)

    # Through-flow vs RAW congestion (all grids)
    add_corr("through_mod_vs_raw_all", through_mod.ravel(), cong.ravel(),
             "model through-flow vs raw congestion (all grids)")
    add_corr("through_obs_vs_raw_all", through_obs.ravel(), cong.ravel(),
             "obs OD through-flow vs raw congestion (all grids)")
    # Through-flow vs commute-attributed (all grids)
    add_corr("through_mod_vs_attr_all", through_mod.ravel(), cong_attr.ravel(),
             "model through-flow vs commute-attr congestion (all grids)")
    add_corr("through_obs_vs_attr_all", through_obs.ravel(), cong_attr.ravel(),
             "obs OD through-flow vs commute-attr congestion (all grids)")
    # Same, instrumented only
    add_corr("through_mod_vs_raw_inst", through_mod[real_grid_mask].ravel(),
             cong[real_grid_mask].ravel(),
             "model through-flow vs raw congestion (instrumented boroughs)")
    add_corr("through_obs_vs_raw_inst", through_obs[real_grid_mask].ravel(),
             cong[real_grid_mask].ravel(),
             "obs OD through-flow vs raw congestion (instrumented boroughs)")
    add_corr("through_mod_vs_attr_inst", through_mod[real_grid_mask].ravel(),
             cong_attr[real_grid_mask].ravel(),
             "model through-flow vs commute-attr congestion (instrumented) ← MAIN")
    add_corr("through_obs_vs_attr_inst", through_obs[real_grid_mask].ravel(),
             cong_attr[real_grid_mask].ravel(),
             "obs OD through-flow vs commute-attr congestion (instrumented)")

    # Per-hour, instrumented + attributed
    per_hour = []
    for h in range(T):
        x_mod = through_mod[real_grid_mask, h]
        x_obs = through_obs[real_grid_mask, h]
        y = cong_attr[real_grid_mask, h]
        r_mod = correlation_stats(x_mod, y, f"hour {h} model")
        r_obs = correlation_stats(x_obs, y, f"hour {h} obs")
        per_hour.append({
            "hour": h,
            "obs_throughflow_vs_attr_pearson": r_obs["pearson"],
            "mod_throughflow_vs_attr_pearson": r_mod["pearson"],
            "commute_share": float(cs[h]),
            "n": r_mod["n"],
        })
    results["per_hour_inst_attributed"] = per_hour

    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path.with_suffix(".npz"),
        through_mod=through_mod.astype(np.float32),
        through_obs=through_obs.astype(np.float32),
        cong=cong.astype(np.float32),
        cong_attr=cong_attr.astype(np.float32),
        real_grid_mask=real_grid_mask,
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[route-val] wrote {out_path} (+ .npz)")

    print("\n=== Correlation Cells ===")
    for key, r in results["correlations"].items():
        print(f"  {r['label']:65s}  Pearson {r['pearson']:+.4f}  Spearman {r['spearman']:+.4f}  (n={r['n']})")

    print("\n=== Per-hour (instrumented + commute-attributed target) ===")
    print(f"  {'hour':>4s}  {'commute share':>13s}  {'obs corr':>8s}  {'model corr':>10s}")
    for ph in per_hour:
        h = ph["hour"]
        peak = "  *PEAK*" if h in (7, 8, 9, 17, 18) else ""
        print(f"  {h:>4d}  {ph['commute_share']:>13.3f}  "
              f"{ph['obs_throughflow_vs_attr_pearson']:>+7.4f}  "
              f"{ph['mod_throughflow_vs_attr_pearson']:>+9.4f}{peak}")


if __name__ == "__main__":
    main()
