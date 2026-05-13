"""Run v2.2 scenarios with heterogeneous utility.

Same flow as scenarios.py but loads HeterogeneousUtility + training_aux_v22.npz.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.heterogeneous_utility import HeterogeneousUtility
from scenarios import (
    make_scenario_A_modify_fn, make_scenario_B_modify_fn,
    run_scenario, find_grid_near, recompute_V_jt, OUT,
)
from run_v21_demo import assign_departure_hours


def load_inputs_v22():
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    aux = dict(np.load(V2_ROOT / "data" / "processed" / "training_aux_v22.npz", allow_pickle=True))
    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    nts_path = V2_ROOT / "data" / "processed" / "nts_commute_departure_time.csv"
    agents_df = assign_departure_hours(agents_df, nts_path)

    util = HeterogeneousUtility()
    util.load_state_dict(torch.load(V2_ROOT / "heterogeneous_utility_best.pt", map_location="cpu"))
    util.eval()

    norm_stats = {k: aux[k] for k in
        ("V_jt_mean","V_jt_std","t_log_mean","t_log_std","d_log_mean","d_log_std","om_mean","om_std")}
    return cache, aux, grid_feats, agents_df, util, norm_stats


def per_tier_stats(model):
    """Compute mean commute by income tier."""
    out = {}
    for tier in (1, 2, 3):
        cts = [a.commute_time for a in model.agents
               if a.income_tier == tier and a.commute_time is not None]
        out[tier] = (len(cts), float(np.mean(cts)) if cts else 0.0)
    return out


def scenario_A_v22():
    cache, aux, gf, ag, util, norm_stats = load_inputs_v22()
    base = run_scenario(cache, aux, gf, ag, util, norm_stats, label="Baseline (v2.2)")

    modify, target_idx, target_id = make_scenario_A_modify_fn(gf)
    cf = run_scenario(cache, aux, gf, ag, util, norm_stats,
                       modify_fn=modify, label="Scenario A v2.2 (+65k OOC)",
                       recompute_stgnn=True)

    base_inflow = base["flow_per_grid_hour"].sum(axis=(0, 1))
    cf_inflow = cf["flow_per_grid_hour"].sum(axis=(0, 1))
    delta = cf_inflow - base_inflow

    print(f"\n=== Scenario A v2.2 results ===")
    print(f"  Baseline mean commute:  {base['metrics']['mean_commute_time']:.2f} min")
    print(f"  Counterfactual:         {cf['metrics']['mean_commute_time']:.2f} min")
    print(f"  Δ inflow at OOC ({target_id}): {delta[target_idx]:+.0f} agents")
    print(f"  Top 5 inflow gainers:")
    for j in np.argsort(-delta)[:5]:
        print(f"    {gf.iloc[j]['grid_id']}: {delta[j]:+.0f}")

    # Heatmap
    import geopandas as gpd
    grid_geo = gpd.read_file(V2_ROOT / "data" / "processed" / "london_1km_grid.geojson").sort_values("grid_id").reset_index(drop=True)
    grid_geo["delta"] = delta
    fig, ax = plt.subplots(figsize=(10, 8))
    cap = max(abs(delta).max(), 1)
    grid_geo.plot(column="delta", ax=ax, cmap="RdBu_r", vmin=-cap, vmax=cap,
                  legend=True, legend_kwds={"shrink": 0.5, "label": "Delta inflow"},
                  edgecolor="none")
    grid_geo.iloc[[target_idx]].boundary.plot(ax=ax, color="black", linewidth=2)
    ax.set_axis_off()
    ax.set_title(f"v2.2 Scenario A: OOC +65k jobs — inflow change (heterogeneous agents)")
    plt.tight_layout()
    plt.savefig(OUT / "scenario_A_v22.png", dpi=120)
    plt.close()
    print(f"  Saved: {OUT / 'scenario_A_v22.png'}")

    np.savez_compressed(OUT / "scenario_A_v22_results.npz",
                        baseline_inflow=base_inflow, cf_inflow=cf_inflow, delta_inflow=delta,
                        baseline_mean_commute=base['metrics']['mean_commute_time'],
                        cf_mean_commute=cf['metrics']['mean_commute_time'],
                        target_idx=target_idx)
    return base, cf


def scenario_B_v22():
    cache, aux, gf, ag, util, norm_stats = load_inputs_v22()
    base = run_scenario(cache, aux, gf, ag, util, norm_stats, label="Baseline (v2.2)")
    modify = make_scenario_B_modify_fn(adoption_rate=0.5)
    cf = run_scenario(cache, aux, gf, ag, util, norm_stats,
                       modify_fn=modify, label="Scenario B v2.2 (50% flex)")

    base_flow = base["flow_per_grid_hour"].sum(axis=(1, 2))
    cf_flow = cf["flow_per_grid_hour"].sum(axis=(1, 2))

    print(f"\n=== Scenario B v2.2 results ===")
    print(f"  Baseline mean commute:  {base['metrics']['mean_commute_time']:.2f} min")
    print(f"  Counterfactual:         {cf['metrics']['mean_commute_time']:.2f} min")
    print(f"  Baseline peak hour: {base['metrics']['peak_hour']} flow={base['metrics']['peak_hour_flow']:.0f}")
    print(f"  CF peak hour:       {cf['metrics']['peak_hour']} flow={cf['metrics']['peak_hour_flow']:.0f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    hours = np.arange(24)
    ax.bar(hours - 0.2, base_flow, 0.4, label="Baseline", color="#888")
    ax.bar(hours + 0.2, cf_flow, 0.4, label="Scenario B v2.2 (50% flex)", color="#2ca02c")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Total agent flow")
    ax.set_title("v2.2 Scenario B: Flexible work — heterogeneous agents")
    ax.set_xticks(hours); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "scenario_B_v22.png", dpi=120)
    plt.close()

    np.savez_compressed(OUT / "scenario_B_v22_results.npz",
                        baseline_flow_per_hr=base_flow, cf_flow_per_hr=cf_flow,
                        baseline_mean_commute=base['metrics']['mean_commute_time'],
                        cf_mean_commute=cf['metrics']['mean_commute_time'])
    return base, cf


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    args = p.parse_args()
    if args.scenario in ("A", "both"):
        print("=" * 60); print("v2.2 SCENARIO A: OOC +65k jobs"); print("=" * 60)
        scenario_A_v22()
    if args.scenario in ("B", "both"):
        print("\n" + "=" * 60); print("v2.2 SCENARIO B: Flexible work"); print("=" * 60)
        scenario_B_v22()
    print("\n=== DONE ===")
