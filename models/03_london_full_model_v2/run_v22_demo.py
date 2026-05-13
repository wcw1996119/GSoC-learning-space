"""Run v2.2 ABM demo with heterogeneous utility.

Same as run_v21_demo.py but loads HeterogeneousUtility instead of LinearUtility.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.heterogeneous_utility import HeterogeneousUtility
from model import LondonV2Model
from run_v21_demo import assign_departure_hours


def main(n_agents_subsample: int = 10000, n_iters: int = 2):
    print("Loading inputs...")
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    aux = dict(np.load(V2_ROOT / "data" / "processed" / "training_aux_v22.npz", allow_pickle=True))
    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    if n_agents_subsample and n_agents_subsample < len(agents_df):
        agents_df = agents_df.sample(n_agents_subsample, random_state=42).reset_index(drop=True)

    nts_path = V2_ROOT / "data" / "processed" / "nts_commute_departure_time.csv"
    agents_df = assign_departure_hours(agents_df, nts_path)

    util = HeterogeneousUtility()
    util.load_state_dict(torch.load(V2_ROOT / "heterogeneous_utility_best.pt", map_location="cpu"))
    util.eval()
    print(f"\nLoaded v2.2 heterogeneous utility:")
    print(f"  alpha_V    = {util.alpha_V.item():.3f}")
    print(f"  alpha_occ  = {util.alpha_occ.item():.3f}")
    print(f"  gamma      = {util.gamma.item():.3f}")
    print(f"  bias       = {util.bias.item():.3f}")
    print(f"  beta_income = {util.beta_income.tolist()}")
    print(f"  beta_mode   = {util.beta_mode.tolist()}")

    norm_stats = {k: aux[k] for k in
        ("V_jt_mean","V_jt_std","t_log_mean","t_log_std","d_log_mean","d_log_std","om_mean","om_std")}

    print("\nBuilding LondonV2Model...")
    model = LondonV2Model(
        agents_df=agents_df,
        grid_features_df=grid_feats,
        V_jt_baseline=cache["V_jt_baseline"],
        t_ij_t_initial=cache["t_ij_t"],
        log_d_ij=aux["log_d_ij"],
        linear_utility_model=util,        # passed as 'utility' inside model
        norm_stats=norm_stats,
    )

    print("\n=== Running equilibrium iterations ===")
    for it in range(n_iters):
        import time
        t0 = time.time()
        model.step(recompute_congestion=(it < n_iters - 1))
        info = model.last_step_info
        m = model.collect_metrics()
        elapsed = time.time() - t0
        print(
            f"Iter {info['iter']:2d} | n={info['n_agents']} | "
            f"changes={info['agent_changes']} ({100*info['convergence_rate']:.1f}%) | "
            f"mean_commute={m['mean_commute_time']:.1f} min | "
            f"peak hour={m['peak_hour']} (flow={m['peak_hour_flow']:.0f}) | "
            f"{elapsed:.1f}s"
        )

    # Per-income-tier mean commute analysis
    print("\n--- Heterogeneity check: mean commute by income tier ---")
    for tier in (1, 2, 3):
        commute_times = [a.commute_time for a in model.agents
                         if a.income_tier == tier and a.commute_time is not None]
        if commute_times:
            label = ["", "low", "mid", "high"][tier]
            print(f"  Tier {tier} ({label}): n={len(commute_times)}, "
                  f"mean={np.mean(commute_times):.1f} min, median={np.median(commute_times):.1f}")

    print("\n--- Heterogeneity check: mean commute by mode ---")
    for mode in ("car", "pt", "active"):
        commute_times = [a.commute_time for a in model.agents
                         if a.mode == mode and a.commute_time is not None]
        if commute_times:
            print(f"  Mode {mode}: n={len(commute_times)}, "
                  f"mean={np.mean(commute_times):.1f} min")

    out = V2_ROOT / "evaluation_outputs" / "v22_baseline_results.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        flow_per_grid_hour=model.flow_per_grid_hour,
        agent_assignments=np.array(
            [(a.home_grid_idx, a.work_grid_idx, a.departure_hour, a.commute_time,
              a.income_tier, ["car","pt","active"].index(a.mode))
             for a in model.agents], dtype=np.float32,
        ),
    )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main(n_agents_subsample=10000, n_iters=2)
