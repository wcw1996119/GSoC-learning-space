"""Compile all paper tables (T1-T8) from existing JSON outputs.

Outputs:
  evaluation_outputs/paper_a/T1_leaderboard.csv
  evaluation_outputs/paper_a/T2_recovered_params.csv
  evaluation_outputs/paper_a/T3_per_borough.csv
  evaluation_outputs/paper_a/T5_retrodict_3tests.csv
  evaluation_outputs/paper_a/T6_scenario_A_inequality.csv
  evaluation_outputs/paper_a/T7_scenario_B_per_hour.csv
  evaluation_outputs/paper_a/T8_jensen_bias.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def load(name):
    try:
        return json.load(open(OUT_DIR / name))
    except FileNotFoundError:
        return None


def main():
    # ===== T1 leaderboard =====
    baselines = load("baselines_spatial_holdout.json")
    sq_ens = load("squeeze_ensemble.json")
    sq_gat = load("squeeze_gat.json")
    # Prefer 10-seed eval if available; fall back to 5-seed
    final_ours = (load("final_paper_eval_sage_unconstrained_K50_h4_seeds10.json")
                   or load("final_paper_eval_sage_unconstrained_K50_h4_seeds5.json"))

    rows = []
    rows.append({"model": "Gravity (Wilson 1971)", "cpc": baselines["leaderboard"]["gravity"],
                 "ci95_lo": "", "ci95_hi": "", "n_seeds": "n/a", "p_vs_gravity": ""})
    rows.append({"model": "Radiation (Simini 2012)", "cpc": baselines["leaderboard"]["radiation"],
                 "ci95_lo": "", "ci95_hi": "", "n_seeds": "n/a", "p_vs_gravity": ""})
    rows.append({"model": "Deep Gravity (min MLP)", "cpc": baselines["leaderboard"]["deep_gravity_mean"],
                 "ci95_lo": baselines["leaderboard"]["deep_gravity_mean"] - baselines["leaderboard"]["deep_gravity_std"],
                 "ci95_hi": baselines["leaderboard"]["deep_gravity_mean"] + baselines["leaderboard"]["deep_gravity_std"],
                 "n_seeds": 3, "p_vs_gravity": ""})
    if sq_gat:
        gat = sq_gat["ensemble"]["unconstrained"]
        rows.append({"model": "GAT (4-head, ours impl)", "cpc": gat["cpc"],
                     "ci95_lo": gat["per_seed_cpc_mean"] - gat["per_seed_cpc_std"],
                     "ci95_hi": gat["per_seed_cpc_mean"] + gat["per_seed_cpc_std"],
                     "n_seeds": len(sq_gat["per_run"]) // 2,  # 2 conditions × n_seeds
                     "p_vs_gravity": ""})
    if sq_ens:
        c = sq_ens["constrained"]
        rows.append({"model": "Phase B v6 constrained (ours)", "cpc": c["ensemble"],
                     "ci95_lo": c["per_seed_mean"] - c["per_seed_std"],
                     "ci95_hi": c["per_seed_mean"] + c["per_seed_std"],
                     "n_seeds": sq_ens["n_seeds"], "p_vs_gravity": ""})
    if final_ours:
        b = final_ours["bootstrap_ours"]
        p = final_ours["paired_vs_gravity"]["p_one_sided_a_le_b"]
        rows.append({"model": "**Phase B v6 unconstrained (ours)**", "cpc": b["mean"],
                     "ci95_lo": b["ci95_lo"], "ci95_hi": b["ci95_hi"],
                     "n_seeds": final_ours["n_seeds"],
                     "p_vs_gravity": f"<{max(p, 1e-4):.4f}" if p < 0.001 else f"{p:.4f}"})
    pd.DataFrame(rows).to_csv(OUT_DIR / "T1_leaderboard.csv", index=False)

    # ===== T2 recovered parameters =====
    params_rows = []
    if final_ours:
        for r in final_ours.get("metrics_per_seed", []):
            params_rows.append({
                "seed": r["seed"], "constrained": r["constrained"],
                "beta_base": r.get("beta_base"), "beta_c": r.get("beta_c"),
                "phi": r.get("phi"), "psi": r.get("psi"),
                "delta": r.get("delta"), "gamma": r.get("gamma"),
            })
    pd.DataFrame(params_rows).to_csv(OUT_DIR / "T2_recovered_params.csv", index=False)

    # ===== T3 per-borough =====
    if final_ours:
        pb = final_ours["per_borough_cpc"]
        pd.DataFrame(pb).to_csv(OUT_DIR / "T3_per_borough.csv", index=False)

    # ===== T5 retrodict three tests =====
    retro = load("retrodict_3tests.json")
    if retro:
        retro_rows = [
            {"test": "Test 1: city-wide forward retrodict",
             "metric": "in_sample_cpc_2011",
             "value": retro["test_1_citywide_forward"]["in_sample_cpc_2011"]},
            {"test": "Test 1: city-wide forward retrodict",
             "metric": "out_of_sample_cpc_2021",
             "value": retro["test_1_citywide_forward"]["out_of_sample_cpc_2021"]},
            {"test": "Test 1: city-wide forward retrodict",
             "metric": "spearman_all_OD_pairs",
             "value": retro["test_1_citywide_forward"]["spearman_all_OD_pairs"]},
            {"test": "Test 2: Stratford intervention",
             "metric": "delta_share_predicted_pp",
             "value": retro["test_2_stratford_intervention"]["delta_share_predicted_pp"]},
            {"test": "Test 2: Stratford intervention",
             "metric": "delta_share_observed_pp",
             "value": retro["test_2_stratford_intervention"]["delta_share_observed_pp"]},
            {"test": "Test 2: Stratford intervention",
             "metric": "ratio_pred_over_obs",
             "value": retro["test_2_stratford_intervention"]["ratio_pred_over_obs"]},
            {"test": "Test 3: Bromley negative control",
             "metric": "delta_share_predicted_pp",
             "value": retro["test_3_bromley_negative_control"]["delta_share_predicted_pp"]},
            {"test": "Test 3: Bromley negative control",
             "metric": "delta_share_observed_pp",
             "value": retro["test_3_bromley_negative_control"]["delta_share_observed_pp"]},
            {"test": "Test 3: Bromley negative control",
             "metric": "ratio_pred_over_obs",
             "value": retro["test_3_bromley_negative_control"]["ratio_pred_over_obs"]},
        ]
        pd.DataFrame(retro_rows).to_csv(OUT_DIR / "T5_retrodict_3tests.csv", index=False)

    # ===== T6 scenario A inequality =====
    sa = load("scenario_A_accessibility.json")
    if sa:
        pd.DataFrame(sa["rows"]).to_csv(OUT_DIR / "T6_scenario_A_inequality.csv", index=False)

    # ===== T7 scenario B per-hour =====
    sb = load("scenario_B_temporal.json")
    if sb:
        pd.DataFrame(sb["per_hour_flow"]).to_csv(OUT_DIR / "T7_scenario_B_per_hour.csv", index=False)

    # ===== T8 jensen bias =====
    cov = load("het_cov_sensitivity.json")
    if cov:
        pd.DataFrame(cov["rows"]).to_csv(OUT_DIR / "T8_jensen_bias_cov.csv", index=False)

    print(f"compiled tables T1-T8 to {OUT_DIR}/T*.csv")
    for f in sorted(OUT_DIR.glob("T*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
