"""F1 Figure: model leaderboard bar chart with 95% CIs.

Reads results from prior runs:
  evaluation_outputs/paper_a/baselines_spatial_holdout.json
  evaluation_outputs/paper_a/squeeze_ensemble.json
  evaluation_outputs/paper_a/squeeze_gat.json
  evaluation_outputs/paper_a/final_paper_eval_sage_unconstrained_K50_h4_seeds5.json

Outputs:
  evaluation_outputs/paper_a/F1_leaderboard.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def safe_load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def main():
    baselines = safe_load(OUT_DIR / "baselines_spatial_holdout.json")
    squeeze_ens = safe_load(OUT_DIR / "squeeze_ensemble.json")
    squeeze_gat = safe_load(OUT_DIR / "squeeze_gat.json")
    final_ours = (safe_load(OUT_DIR / "final_paper_eval_sage_unconstrained_K50_h4_seeds10.json")
                   or safe_load(OUT_DIR / "final_paper_eval_sage_unconstrained_K50_h4_seeds5.json"))

    rows = []
    if baselines:
        rows.append({"model": "Gravity\n(Wilson)", "cpc": baselines["leaderboard"]["gravity"], "ci_lo": None, "ci_hi": None, "color": "#999999"})
        rows.append({"model": "Radiation\n(Simini12)", "cpc": baselines["leaderboard"]["radiation"], "ci_lo": None, "ci_hi": None, "color": "#999999"})
        rows.append({"model": "Deep Gravity\n(min MLP)", "cpc": baselines["leaderboard"]["deep_gravity_mean"], "ci_lo": baselines["leaderboard"]["deep_gravity_mean"] - baselines["leaderboard"]["deep_gravity_std"], "ci_hi": baselines["leaderboard"]["deep_gravity_mean"] + baselines["leaderboard"]["deep_gravity_std"], "color": "#cccccc"})
    if squeeze_gat:
        gat_ens = squeeze_gat["ensemble"]["unconstrained"]
        rows.append({"model": "GAT\n(4-head)", "cpc": gat_ens["cpc"], "ci_lo": gat_ens["per_seed_cpc_mean"] - gat_ens["per_seed_cpc_std"], "ci_hi": gat_ens["per_seed_cpc_mean"] + gat_ens["per_seed_cpc_std"], "color": "#bbbbbb"})
    if squeeze_ens:
        rows.append({"model": "Ours SAGE\nconstrained", "cpc": squeeze_ens["constrained"]["ensemble"], "ci_lo": squeeze_ens["constrained"]["per_seed_mean"] - squeeze_ens["constrained"]["per_seed_std"], "ci_hi": squeeze_ens["constrained"]["per_seed_mean"] + squeeze_ens["constrained"]["per_seed_std"], "color": "#5da7d8"})
    if final_ours:
        b = final_ours["bootstrap_ours"]
        rows.append({"model": "Ours SAGE\nunconstrained ⭐", "cpc": b["mean"], "ci_lo": b["ci95_lo"], "ci_hi": b["ci95_hi"], "color": "#1f77b4"})
    elif squeeze_ens:
        u = squeeze_ens["unconstrained"]
        rows.append({"model": "Ours SAGE\nunconstrained ⭐", "cpc": u["ensemble"], "ci_lo": u["per_seed_mean"] - u["per_seed_std"], "ci_hi": u["per_seed_mean"] + u["per_seed_std"], "color": "#1f77b4"})

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(rows))
    cpcs = [r["cpc"] for r in rows]
    yerr_lo = [r["cpc"] - r["ci_lo"] if r["ci_lo"] is not None else 0 for r in rows]
    yerr_hi = [r["ci_hi"] - r["cpc"] if r["ci_hi"] is not None else 0 for r in rows]
    colors = [r["color"] for r in rows]
    bars = ax.bar(xs, cpcs, color=colors, yerr=[yerr_lo, yerr_hi],
                   error_kw=dict(ecolor="black", capsize=4, lw=1.2))
    for bar, val in zip(bars, cpcs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(y=baselines["leaderboard"]["gravity"] if baselines else 0.17,
                color="red", linestyle=":", alpha=0.5, label="Gravity baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["model"] for r in rows], fontsize=10)
    ax.set_ylabel("CPC (4-borough spatial holdout)", fontsize=11)
    ax.set_title("Model leaderboard, 4-borough spatial holdout (95% CI)\nLondon Census 2021, full Census table",
                  fontsize=12)
    ax.set_ylim(0, max(cpcs) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "F1_leaderboard.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
