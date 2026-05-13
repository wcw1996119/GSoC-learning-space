"""Run v2.3 ABM (with WageUtility + per-agent real income).

Self-contained — does not depend on LondonV2Model (v2.2 specific) to avoid coupling.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.wage_utility import WageUtility
from models_lib.occupation_match import occupation_match_per_soc, grid_industry_vector
from models_lib.wage_priors import LONDON_MEDIAN_WAGE
from run_v21_demo import assign_departure_hours

PROC = V2_ROOT / "data" / "processed"
EVAL = V2_ROOT / "evaluation_outputs"


def _load_inputs():
    cache = dict(np.load(PROC / "demo_cache.npz", allow_pickle=True))
    aux = dict(np.load(PROC / "training_aux_v23.npz", allow_pickle=True))
    grid_feats = pd.read_csv(PROC / "grid_static_features_v23.csv")
    agents_df = pd.read_csv(PROC / "agent_population_v23.csv")
    nts_path = PROC / "nts_commute_departure_time.csv"
    agents_df = assign_departure_hours(agents_df, nts_path)

    util = WageUtility()
    util.load_state_dict(torch.load(V2_ROOT / "wage_utility_best.pt", map_location="cpu"))
    util.eval()
    return cache, aux, grid_feats, agents_df, util


def _normalise_features(cache, grid_feats, aux):
    """Return all normalised tensors aligned to grid order."""
    V_jt = torch.tensor(cache["V_jt_baseline"], dtype=torch.float32)
    t_ij_t = torch.tensor(cache["t_ij_t"], dtype=torch.float32)
    log_d_ij = torch.tensor(aux["log_d_ij"], dtype=torch.float32)

    V_norm = (V_jt - aux["V_jt_mean"]) / (aux["V_jt_std"] + 1e-8)
    t_log = torch.log1p(t_ij_t)
    t_norm = (t_log - aux["t_log_mean"]) / (aux["t_log_std"] + 1e-8)
    d_norm = (log_d_ij - aux["d_log_mean"]) / (aux["d_log_std"] + 1e-8)

    log_wage_norm = (
        np.log(grid_feats["wage_j_weekly"].values) - np.log(LONDON_MEDIAN_WAGE)
    ).astype(np.float32)
    log_wage_norm_t = torch.tensor(log_wage_norm)

    # Per-SOC OccMatch — normalised
    gx = grid_industry_vector(grid_feats)
    om_per_soc = occupation_match_per_soc(gx)  # (9, N)
    om_norm_per_soc = (om_per_soc - aux["om_mean"]) / (aux["om_std"] + 1e-8)
    om_norm_per_soc_t = torch.tensor(om_norm_per_soc, dtype=torch.float32)

    return V_norm, t_norm, d_norm, log_wage_norm_t, om_norm_per_soc_t, t_ij_t, float(aux["t_log_std"])


def _compute_log_p(util, V_norm_h, om_norm_personal, log_wage_norm_t,
                    t_norm_origin, d_norm_origin, beta_personal_scaled):
    """Per-agent log P(j | origin, hour, soc, income)."""
    with torch.no_grad():
        V_ij = util.utility_personal(
            V_j_t=V_norm_h,
            occ_match_personal=om_norm_personal,
            log_wage_norm=log_wage_norm_t,
            t_ij_t=t_norm_origin,
            log_d_ij=d_norm_origin,
            beta_personal=beta_personal_scaled,
        )
        return torch.log_softmax(V_ij, dim=0).numpy()


def run_abm(cache, aux, grid_feats, agents_df, util, label="v2.3"):
    print(f"\n=== Running ABM ({label}) ===")
    V_norm, t_norm, d_norm, log_wage_norm_t, om_norm_per_soc, t_ij_t_raw, t_log_std = \
        _normalise_features(cache, grid_feats, aux)
    N = V_norm.shape[1]

    soc_to_idx = {f"soc{i}": i - 1 for i in range(1, 10)}
    rng = np.random.default_rng(42)

    flow = np.zeros((24, N, N), dtype=np.float32)
    commute_times = []
    income_per_agent = []
    work_idx_per_agent = []
    home_idx_per_agent = []
    soc_per_agent = []

    t0 = time.time()
    for _, row in agents_df.iterrows():
        i = int(row["home_grid_idx"])
        h = int(row["departure_hour"])
        soc_idx = soc_to_idx.get(str(row["soc"]), 0)
        income = float(row["income_real"])
        beta_p_minutes = float(row["beta_personal"])
        beta_p_scaled = beta_p_minutes * t_log_std

        Vj = V_norm[h]
        om = om_norm_per_soc[soc_idx]
        tt = t_norm[h, i]
        d = d_norm[i]

        log_p = _compute_log_p(util, Vj, om, log_wage_norm_t, tt, d,
                                 torch.tensor(beta_p_scaled, dtype=torch.float32))
        # Sample
        p = np.exp(log_p - log_p.max()); p = p / p.sum()
        j = int(rng.choice(N, p=p))
        flow[h, i, j] += 1.0
        commute_times.append(float(t_ij_t_raw[h, i, j]))
        income_per_agent.append(income)
        work_idx_per_agent.append(j)
        home_idx_per_agent.append(i)
        soc_per_agent.append(soc_idx + 1)
    print(f"  agents sampled: {len(commute_times)} in {time.time() - t0:.1f}s")

    return {
        "flow": flow,
        "commute_times": np.array(commute_times),
        "incomes": np.array(income_per_agent),
        "work_idx": np.array(work_idx_per_agent, dtype=int),
        "home_idx": np.array(home_idx_per_agent, dtype=int),
        "soc": np.array(soc_per_agent, dtype=int),
    }


def income_tier_analysis(result):
    """Mean commute by income tier — direct comparison with v2.2."""
    inc = result["incomes"]
    ct = result["commute_times"]
    # Tier by income quintile (data-driven, not SOC-derived)
    q33 = np.percentile(inc, 33)
    q67 = np.percentile(inc, 67)
    tier_idx = np.where(inc <= q33, 1, np.where(inc <= q67, 2, 3))
    rows = []
    for tier in (1, 2, 3):
        mask = tier_idx == tier
        rows.append({
            "tier": tier,
            "label": ["", "low", "mid", "high"][tier],
            "n": int(mask.sum()),
            "mean_commute": float(ct[mask].mean()),
            "median_commute": float(np.median(ct[mask])),
            "mean_income": float(inc[mask].mean()),
        })
    print("\n--- Income tier analysis (v2.3, by real income terciles) ---")
    for r in rows:
        print(f"  Tier {r['tier']} ({r['label']:>4}): n={r['n']}, "
              f"mean_commute={r['mean_commute']:.1f} min, "
              f"mean_income=£{r['mean_income']:.0f}")
    return rows


def main():
    cache, aux, grid_feats, agents_df, util = _load_inputs()
    print(f"v2.3 utility: αV={util.alpha_V.item():.3f}, αocc={util.alpha_occ.item():.3f}, "
          f"αwage={util.alpha_wage.item():.3f}, γ={util.gamma.item():.3f}, b={util.bias.item():.3f}")

    res = run_abm(cache, aux, grid_feats, agents_df, util, label="v2.3 baseline")
    rows = income_tier_analysis(res)

    # Save
    out = EVAL / "v23_baseline_results.npz"
    np.savez_compressed(
        out,
        flow=res["flow"],
        commute_times=res["commute_times"],
        incomes=res["incomes"],
        work_idx=res["work_idx"],
        home_idx=res["home_idx"],
        soc=res["soc"],
        tier_means=np.array([r["mean_commute"] for r in rows]),
        tier_incomes=np.array([r["mean_income"] for r in rows]),
    )
    print(f"\nSaved {out}")

    # v2.2 comparison
    print("\n--- v2.2 reference (for comparison) ---")
    print("  Tier 1 ( low): mean_commute=37.3 min")
    print("  Tier 2 ( mid): mean_commute=40.5 min")
    print("  Tier 3 (high): mean_commute=42.4 min")
    print(f"  Δ (high − low) = +5.1 min  (v2.2)")
    delta_v23 = rows[2]["mean_commute"] - rows[0]["mean_commute"]
    print(f"  Δ (high − low) = {delta_v23:+.1f} min  (v2.3)")


if __name__ == "__main__":
    main()
