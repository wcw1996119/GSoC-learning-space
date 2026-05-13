"""Run a v2.1 ABM demo: build LondonV2Model, run 1-2 equilibrium iterations, save results.

Usage:
  python run_v21_demo.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.linear_utility import LinearUtility
from model import LondonV2Model

DEVICE = "cpu"


def assign_departure_hours(agents_df: pd.DataFrame, nts_path: Path, seed: int = 42):
    """Sample departure hour per agent from NTS commute departure-time distribution."""
    nts = pd.read_csv(nts_path, comment="#")
    pi_t = nts[nts["mode"] == "all"].sort_values("hour")["share"].values
    pi_t = pi_t / pi_t.sum()
    rng = np.random.default_rng(seed)
    hours = rng.choice(24, size=len(agents_df), p=pi_t)
    agents_df = agents_df.copy()
    agents_df["departure_hour"] = hours
    return agents_df


def main(n_agents_subsample: int = 10000, n_iters: int = 2):
    print("Loading inputs...")
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    aux = dict(np.load(V2_ROOT / "data" / "processed" / "training_aux.npz", allow_pickle=True))

    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    if n_agents_subsample and n_agents_subsample < len(agents_df):
        agents_df = agents_df.sample(n_agents_subsample, random_state=42).reset_index(drop=True)

    nts_path = V2_ROOT / "data" / "processed" / "nts_commute_departure_time.csv"
    agents_df = assign_departure_hours(agents_df, nts_path)
    print(f"Agents: {len(agents_df)}, departure hour distribution:")
    print(agents_df["departure_hour"].value_counts().sort_index().to_string())

    util = LinearUtility()
    util.load_state_dict(torch.load(V2_ROOT / "linear_utility_best.pt", map_location=DEVICE))
    util.eval()
    print(f"\nLoaded utility: {dict((n, p.item()) for n, p in util.named_parameters())}")

    norm_stats = {
        "V_jt_mean": aux["V_jt_mean"], "V_jt_std": aux["V_jt_std"],
        "t_log_mean": aux["t_log_mean"], "t_log_std": aux["t_log_std"],
        "d_log_mean": aux["d_log_mean"], "d_log_std": aux["d_log_std"],
        "om_mean": aux["om_mean"], "om_std": aux["om_std"],
    }

    print("\nBuilding LondonV2Model...")
    model = LondonV2Model(
        agents_df=agents_df,
        grid_features_df=grid_feats,
        V_jt_baseline=cache["V_jt_baseline"],
        t_ij_t_initial=cache["t_ij_t"],
        log_d_ij=aux["log_d_ij"],
        linear_utility_model=util,
        norm_stats=norm_stats,
    )

    print("\n=== Running equilibrium iterations ===")
    for it in range(n_iters):
        import time
        t0 = time.time()
        model.step(recompute_congestion=(it < n_iters - 1))
        info = model.last_step_info
        elapsed = time.time() - t0
        m = model.collect_metrics()
        print(
            f"Iter {info['iter']:2d} | n={info['n_agents']} | "
            f"changes={info['agent_changes']} ({100*info['convergence_rate']:.1f}%) | "
            f"mean_commute={m['mean_commute_time']:.1f} min | "
            f"peak hour={m['peak_hour']} (flow={m['peak_hour_flow']:.0f}) | "
            f"{elapsed:.1f}s"
        )

    print("\nSaving results...")
    out = V2_ROOT / "evaluation_outputs" / "v21_baseline_results.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        flow_per_grid_hour=model.flow_per_grid_hour,
        t_ij_t_final=model.t_ij_t_raw.numpy(),
        agent_assignments=np.array([(a.home_grid_idx, a.work_grid_idx, a.departure_hour, a.commute_time)
                                     for a in model.agents], dtype=np.float32),
    )
    print(f"Saved -> {out}")
    print(f"Final mean commute time: {m['mean_commute_time']:.2f} min")


if __name__ == "__main__":
    main(n_agents_subsample=10000, n_iters=2)
