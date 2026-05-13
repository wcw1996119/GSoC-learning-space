"""Two policy scenarios with v2.1 ABM:

A) Old Oak Common new employment center: +65k jobs at OOC grid (~lat 51.529, lon -0.252)
B) Flexible work hours: shift 50% agents from 9am-peak departure to 8-11am flat distribution

Compare baseline vs counterfactual ABM run results.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.linear_utility import LinearUtility
from models_lib.stgnn import V2_STGNN
from model import LondonV2Model
from run_v21_demo import assign_departure_hours
from data_loader import build_features_tensor
from providers.features import LondonFeatureProvider, STATIC_COLS, LOG_TRANSFORM_COLS
from providers.graph_builder import build_knn_graph

OUT = V2_ROOT / "evaluation_outputs"
OUT.mkdir(exist_ok=True)


def recompute_V_jt(grid_feats_modified, cache):
    """Re-run STGNN forward with modified grid features → new V_jt baseline."""
    # Re-normalize using cached training stats (DON'T re-fit)
    raw = grid_feats_modified[STATIC_COLS].values.astype(np.float64)
    log_idx = [i for i, c in enumerate(STATIC_COLS) if c in LOG_TRANSFORM_COLS]
    raw[:, log_idx] = np.log1p(raw[:, log_idx])
    static_norm = (raw - cache["static_mean"]) / cache["static_std"]

    # Build (T, N, F) feature tensor: static + temporal (use baseline temporal)
    feat_provider = LondonFeatureProvider()
    x_seq_orig = build_features_tensor(feat_provider, T=24)
    F_static = static_norm.shape[1]
    x_seq_new = x_seq_orig.clone()
    for t in range(24):
        x_seq_new[t, :, :F_static] = torch.tensor(static_norm, dtype=torch.float32)

    # Build graph
    edge_index, edge_attr = build_knn_graph(cache["coords_bng"], K=10, add_self_loop=True)

    # Load STGNN
    stgnn = V2_STGNN(
        node_dim=x_seq_new.shape[-1], edge_dim=edge_attr.shape[1],
        hidden_dim=64, gat_heads=4, gru_hidden=32,
    )
    stgnn.load_state_dict(torch.load(V2_ROOT / "best_model.pt", map_location="cpu"))
    stgnn.eval()
    with torch.no_grad():
        V_jt_new = stgnn(x_seq_new, edge_index, edge_attr).numpy()
    return V_jt_new


def load_inputs():
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    aux = dict(np.load(V2_ROOT / "data" / "processed" / "training_aux.npz", allow_pickle=True))
    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    nts_path = V2_ROOT / "data" / "processed" / "nts_commute_departure_time.csv"
    agents_df = assign_departure_hours(agents_df, nts_path)

    util = LinearUtility()
    util.load_state_dict(torch.load(V2_ROOT / "linear_utility_best.pt", map_location="cpu"))
    util.eval()

    norm_stats = {k: aux[k] for k in
        ("V_jt_mean","V_jt_std","t_log_mean","t_log_std","d_log_mean","d_log_std","om_mean","om_std")}

    return cache, aux, grid_feats, agents_df, util, norm_stats


def find_grid_near(grid_feats: pd.DataFrame, lat: float, lon: float):
    """Find the grid index whose centroid is closest to (lat, lon)."""
    d = (grid_feats["centroid_lat"] - lat) ** 2 + (grid_feats["centroid_lon"] - lon) ** 2
    idx = int(d.idxmin())
    return idx, grid_feats.iloc[idx]["grid_id"], float(grid_feats.iloc[idx]["total_employment"])


def run_scenario(cache, aux, grid_feats, agents_df, util, norm_stats,
                 modify_fn=None, label="baseline", n_iters=2,
                 recompute_stgnn=False):
    """Run ABM with optional modification function applied to grid_features or agents.

    If `recompute_stgnn=True`, re-runs STGNN forward with modified grid_features
    to get an updated V_jt baseline (necessary for spatial scenarios like Scenario A).
    """
    gf = grid_feats.copy()
    ag = agents_df.copy()
    cc = {k: cache[k].copy() if hasattr(cache[k], "copy") else cache[k] for k in cache}

    if modify_fn is not None:
        gf, ag, cc = modify_fn(gf, ag, cc)

    if recompute_stgnn:
        print(f"  Recomputing V_jt via STGNN (with modified features)...")
        V_jt_new = recompute_V_jt(gf, cc)
        cc["V_jt_baseline"] = V_jt_new

    print(f"\n=== Running scenario: {label} ===")
    model = LondonV2Model(
        agents_df=ag,
        grid_features_df=gf,
        V_jt_baseline=cc["V_jt_baseline"],
        t_ij_t_initial=cc["t_ij_t"],
        log_d_ij=aux["log_d_ij"],
        linear_utility_model=util,
        norm_stats=norm_stats,
    )
    for it in range(n_iters):
        t0 = time.time()
        model.step(recompute_congestion=(it < n_iters - 1))
        info = model.last_step_info
        m = model.collect_metrics()
        elapsed = time.time() - t0
        print(
            f"  Iter {info['iter']} | changes={info['agent_changes']} | "
            f"mean_commute={m['mean_commute_time']:.1f} min | peak h={m['peak_hour']} flow={m['peak_hour_flow']:.0f} | {elapsed:.1f}s"
        )

    return {
        "label": label,
        "metrics": m,
        "agent_assignments": np.array(
            [(a.home_grid_idx, a.work_grid_idx, a.departure_hour, a.commute_time) for a in model.agents],
            dtype=np.float32,
        ),
        "flow_per_grid_hour": model.flow_per_grid_hour,
    }


# =========================================================
# SCENARIO A: Old Oak Common +65k jobs
# =========================================================

def make_scenario_A_modify_fn(grid_feats, target_jobs_delta=65_000):
    """Add jobs to Old Oak Common (lat 51.529, lon -0.252).
    Adds the delta primarily to Information & Finance + Public services sectors
    (representative for OOC: tech / public service offices planned).
    """
    target_idx, target_id, current_emp = find_grid_near(grid_feats, 51.529, -0.252)
    print(f"  Scenario A target: idx={target_idx}, id={target_id}, current jobs={current_emp:.0f}")

    def modify(gf, ag, cc):
        # Allocate delta: 50% info_finance, 30% public, 20% other
        gf.loc[target_idx, "sec6_info_finance"] += target_jobs_delta * 0.5
        gf.loc[target_idx, "sec7_public"] += target_jobs_delta * 0.3
        gf.loc[target_idx, "sec8_other"] += target_jobs_delta * 0.2
        gf.loc[target_idx, "total_employment"] = (
            gf.loc[target_idx, "total_employment"] + target_jobs_delta
        )
        return gf, ag, cc

    return modify, target_idx, target_id


def scenario_A():
    cache, aux, gf, ag, util, norm_stats = load_inputs()
    base = run_scenario(cache, aux, gf, ag, util, norm_stats, label="Baseline")

    modify, target_idx, target_id = make_scenario_A_modify_fn(gf)
    cf = run_scenario(cache, aux, gf, ag, util, norm_stats,
                       modify_fn=modify, label="Scenario A (+65k OOC)",
                       recompute_stgnn=True)

    # Compare
    baseline_inflow = base["flow_per_grid_hour"].sum(axis=(0, 1))   # (N,)
    cf_inflow = cf["flow_per_grid_hour"].sum(axis=(0, 1))
    delta_inflow = cf_inflow - baseline_inflow

    print(f"\nScenario A summary:")
    print(f"  Baseline mean commute:    {base['metrics']['mean_commute_time']:.2f} min")
    print(f"  Counterfactual:           {cf['metrics']['mean_commute_time']:.2f} min")
    print(f"  Δ inflow at target ({target_id}): {delta_inflow[target_idx]:+.0f} agents")
    top_winners = np.argsort(-delta_inflow)[:5]
    print(f"  Top inflow gainers:")
    for j in top_winners:
        print(f"    {gf.iloc[j]['grid_id']}: {delta_inflow[j]:+.0f}")

    # Plot inflow change map
    import geopandas as gpd
    grid_geo = gpd.read_file(V2_ROOT / "data" / "processed" / "london_1km_grid.geojson").sort_values("grid_id").reset_index(drop=True)
    grid_geo["delta"] = delta_inflow
    fig, ax = plt.subplots(figsize=(10, 8))
    cap = max(abs(delta_inflow).max(), 1)
    grid_geo.plot(column="delta", ax=ax, cmap="RdBu_r", vmin=-cap, vmax=cap,
                  legend=True, legend_kwds={"shrink": 0.5, "label": "Delta inflow"},
                  edgecolor="none")
    grid_geo.iloc[[target_idx]].boundary.plot(ax=ax, color="black", linewidth=2)
    ax.set_axis_off()
    ax.set_title(f"Scenario A: Old Oak Common +65k jobs — inflow change")
    plt.tight_layout()
    plt.savefig(OUT / "scenario_A_inflow_change.png", dpi=120)
    plt.close()
    print(f"  Saved: {OUT / 'scenario_A_inflow_change.png'}")

    # Save data
    np.savez_compressed(OUT / "scenario_A_results.npz",
                       baseline_inflow=baseline_inflow,
                       cf_inflow=cf_inflow,
                       delta_inflow=delta_inflow,
                       baseline_mean_commute=base["metrics"]["mean_commute_time"],
                       cf_mean_commute=cf["metrics"]["mean_commute_time"],
                       target_idx=target_idx)
    return base, cf


# =========================================================
# SCENARIO B: Flexible work hours
# =========================================================

def make_scenario_B_modify_fn(adoption_rate=0.5):
    """50% of agents shift their departure hour from peak (7-9am) to flat 8-11am.

    For agents with current departure_hour in 7,8,9 — half (random) get reassigned
    uniformly to 8,9,10,11.
    """
    rng = np.random.default_rng(42)
    def modify(gf, ag, cc):
        peak_mask = ag["departure_hour"].isin([7, 8, 9]).values
        peak_idx = np.where(peak_mask)[0]
        n_shift = int(adoption_rate * len(peak_idx))
        shift_idx = rng.choice(peak_idx, n_shift, replace=False)
        new_hours = rng.choice([8, 9, 10, 11], size=n_shift)
        ag.loc[shift_idx, "departure_hour"] = new_hours
        return gf, ag, cc
    return modify


def scenario_B():
    cache, aux, gf, ag, util, norm_stats = load_inputs()
    base = run_scenario(cache, aux, gf, ag, util, norm_stats, label="Baseline")

    modify = make_scenario_B_modify_fn(adoption_rate=0.5)
    cf = run_scenario(cache, aux, gf, ag, util, norm_stats,
                       modify_fn=modify, label="Scenario B (50% flex)")

    # Compare hourly flow profile
    base_flow_per_hr = base["flow_per_grid_hour"].sum(axis=(1, 2))   # (T,)
    cf_flow_per_hr = cf["flow_per_grid_hour"].sum(axis=(1, 2))

    print(f"\nScenario B summary:")
    print(f"  Baseline mean commute:  {base['metrics']['mean_commute_time']:.2f} min")
    print(f"  Counterfactual:         {cf['metrics']['mean_commute_time']:.2f} min")
    print(f"  Baseline peak hour:     {base['metrics']['peak_hour']} (flow {base['metrics']['peak_hour_flow']:.0f})")
    print(f"  Counterfactual peak:    {cf['metrics']['peak_hour']} (flow {cf['metrics']['peak_hour_flow']:.0f})")

    # Plot flow profile comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    hours = np.arange(24)
    ax.bar(hours - 0.2, base_flow_per_hr, 0.4, label="Baseline", color="#888")
    ax.bar(hours + 0.2, cf_flow_per_hr, 0.4, label="Scenario B (50% flex)", color="#2ca02c")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Total agent flow")
    ax.set_title("Scenario B: Flexible work hours — departure profile shift")
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "scenario_B_flow_profile.png", dpi=120)
    plt.close()
    print(f"  Saved: {OUT / 'scenario_B_flow_profile.png'}")

    np.savez_compressed(OUT / "scenario_B_results.npz",
                        baseline_flow_per_hr=base_flow_per_hr,
                        cf_flow_per_hr=cf_flow_per_hr,
                        baseline_mean_commute=base["metrics"]["mean_commute_time"],
                        cf_mean_commute=cf["metrics"]["mean_commute_time"])
    return base, cf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    if args.scenario in ("A", "both"):
        print("=" * 60)
        print("SCENARIO A: Old Oak Common new employment center")
        print("=" * 60)
        scenario_A()
    if args.scenario in ("B", "both"):
        print("\n" + "=" * 60)
        print("SCENARIO B: Flexible work hours")
        print("=" * 60)
        scenario_B()
    print("\n=== ALL DONE ===")
