"""Compute accessibility-inequality metrics on the v2.2 baseline ABM output.

Pipeline
--------
1. Load grid_static_features.csv  -> employment per grid E_j  (and centroids).
2. Load training_aux_v22.npz       -> log_d_ij  + normalisation stats.
3. Load v22_baseline_results.npz  -> per-agent assignments (income tier, commute time).
4. Build a per-grid travel-time matrix  t_ij  (in minutes):
        we use straight-line distance in km divided by an effective speed,
        falling back to log_d_ij which is already aligned with grid index.
   (We do not need the model's full t_ij_t tensor for the gravity computation:
    log_d_ij + a constant speed is sufficient and matches the WebTAG specification
    that uses generalised travel time in minutes.)
5. Compute  A_i = Σ_j  E_j · exp(-β · t_ij)  with β = 0.07 (WebTAG mid).
6. Combine with agent_population.csv to get tier-share per home grid, and compute:
       - tier-weighted accessibility per agent (inheriting their home grid's A_i)
       - Gini, Palma per income tier
       - Overall Gini, Palma over the agent-level distribution
7. Save  evaluation_outputs/v22_inequality_results.npz
   Plot  evaluation_outputs/v22_inequality.png
         (Lorenz curve overall + per-tier;  bar chart Gini & Palma per tier)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from metrics.inequality import gini, palma, lorenz_curve  # noqa: E402

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BETA_WEBTAG_MID = 0.07          # 1 / minutes  (WebTAG generalised time decay)
TIER_LABELS = {1: "Low", 2: "Mid", 3: "High"}
TIER_COLOURS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}

DATA_DIR = V2_ROOT / "data" / "processed"
OUT_DIR = V2_ROOT / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Loading inputs...")
    grid_feats = pd.read_csv(DATA_DIR / "grid_static_features.csv")
    aux = dict(np.load(DATA_DIR / "training_aux_v22.npz", allow_pickle=True))
    baseline = dict(np.load(OUT_DIR / "v22_baseline_results.npz", allow_pickle=True))
    agents_df = pd.read_csv(DATA_DIR / "agent_population.csv")

    N = len(grid_feats)
    print(f"  N = {N} grids")

    # ------------------------------------------------------------------
    # 1. Employment per grid (column 'total_employment' in static features)
    # ------------------------------------------------------------------
    if "total_employment" not in grid_feats.columns:
        raise KeyError("grid_static_features.csv missing 'total_employment'")
    E_j = grid_feats["total_employment"].to_numpy(dtype=float)  # (N,)

    # ------------------------------------------------------------------
    # 2. Travel time matrix t_ij in minutes
    #    log_d_ij is log(km) — invert to km, convert to minutes via avg speed.
    #    25 km/h is a London-wide effective door-to-door speed (used by the v1 model).
    # ------------------------------------------------------------------
    log_d_ij = np.asarray(aux["log_d_ij"], dtype=float)
    if log_d_ij.shape != (N, N):
        raise ValueError(f"log_d_ij shape {log_d_ij.shape} != ({N}, {N})")

    avg_speed_kmh = 25.0
    d_ij_km = np.expm1(log_d_ij)            # invert log1p (training pipeline uses log1p)
    d_ij_km = np.where(d_ij_km < 0, 0.0, d_ij_km)
    # Some pipelines use log() rather than log1p() — guard with a sanity fallback
    if np.nanmedian(d_ij_km) > 200:
        # Looks like log_d_ij was natural log, not log1p; redo with exp
        d_ij_km = np.exp(log_d_ij)
    t_ij_min = (d_ij_km / avg_speed_kmh) * 60.0    # minutes
    np.fill_diagonal(t_ij_min, 0.5)                # local trip ~30s

    print(f"  t_ij stats:  min={t_ij_min.min():.2f}  median={np.median(t_ij_min):.2f}"
          f"  max={t_ij_min.max():.2f} min")

    # ------------------------------------------------------------------
    # 3. Gravity accessibility per grid:  A_i = Σ_j E_j · exp(-β t_ij)
    # ------------------------------------------------------------------
    decay = np.exp(-BETA_WEBTAG_MID * t_ij_min)    # (N, N)
    A_i = decay @ E_j                              # (N,)
    print(f"  A_i stats:  min={A_i.min():.1f}  mean={A_i.mean():.1f}  max={A_i.max():.1f}")

    # ------------------------------------------------------------------
    # 4. Per-agent accessibility = home-grid accessibility
    # ------------------------------------------------------------------
    # baseline 'agent_assignments' columns: home_grid, work_grid, dep_hour, t, tier, mode_idx
    assigned = baseline["agent_assignments"]
    if assigned.size > 0:
        home_idx = assigned[:, 0].astype(int)
        tiers = assigned[:, 4].astype(int)
    else:
        # Fall back to the full population (baseline ran without saving assignments)
        home_idx = agents_df["home_grid_idx"].to_numpy(dtype=int)
        tiers = agents_df["income_tier"].to_numpy(dtype=int)

    # Clip to valid grid range
    home_idx = np.clip(home_idx, 0, N - 1)
    A_per_agent = A_i[home_idx]                    # (n_agents,)

    # ------------------------------------------------------------------
    # 5. Inequality metrics — overall + per income tier
    # ------------------------------------------------------------------
    overall_gini = gini(A_per_agent)
    overall_palma = palma(A_per_agent)
    pop_x, pop_y = lorenz_curve(A_per_agent)
    print(f"\nOverall   : Gini={overall_gini:.4f}   Palma={overall_palma:.3f}"
          f"   n_agents={A_per_agent.size}")

    tier_results: dict[int, dict] = {}
    for t in (1, 2, 3):
        mask = tiers == t
        a_t = A_per_agent[mask]
        if a_t.size == 0:
            continue
        g = gini(a_t)
        p = palma(a_t)
        x, y = lorenz_curve(a_t)
        tier_results[t] = {
            "n": int(a_t.size),
            "gini": g,
            "palma": p,
            "mean_acc": float(a_t.mean()),
            "lorenz_x": x,
            "lorenz_y": y,
        }
        print(f"Tier {t} ({TIER_LABELS[t]:>4}): Gini={g:.4f}   Palma={p:.3f}"
              f"   n={a_t.size}   mean_A={a_t.mean():.0f}")

    # ------------------------------------------------------------------
    # 6. Save NPZ
    # ------------------------------------------------------------------
    out_npz = OUT_DIR / "v22_inequality_results.npz"
    save_dict = {
        "beta": BETA_WEBTAG_MID,
        "A_per_grid": A_i,
        "A_per_agent": A_per_agent,
        "agent_tier": tiers,
        "overall_gini": overall_gini,
        "overall_palma": overall_palma,
        "lorenz_overall_x": pop_x,
        "lorenz_overall_y": pop_y,
    }
    for t, res in tier_results.items():
        save_dict[f"tier{t}_gini"] = res["gini"]
        save_dict[f"tier{t}_palma"] = res["palma"]
        save_dict[f"tier{t}_mean"] = res["mean_acc"]
        save_dict[f"tier{t}_lorenz_x"] = res["lorenz_x"]
        save_dict[f"tier{t}_lorenz_y"] = res["lorenz_y"]
    np.savez_compressed(out_npz, **save_dict)
    print(f"\nSaved -> {out_npz}")

    # ------------------------------------------------------------------
    # 7. Figure: Lorenz curves (left) + Gini/Palma bars (right)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ----- Left: Lorenz curves -----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0,
            label="Perfect equality")
    ax.plot(pop_x, pop_y, color="#444444", linewidth=2.4,
            label=f"Overall (Gini={overall_gini:.3f})")
    for t, res in tier_results.items():
        ax.plot(res["lorenz_x"], res["lorenz_y"],
                color=TIER_COLOURS[t], linewidth=1.6,
                label=f"Tier {t} {TIER_LABELS[t]} (Gini={res['gini']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative accessibility share")
    ax.set_title("Lorenz curve of agent accessibility — overall + by income tier")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # ----- Right: Gini and Palma bars per tier -----
    ax = axes[1]
    tiers_sorted = sorted(tier_results.keys())
    labels = [f"Tier {t}\n{TIER_LABELS[t]}" for t in tiers_sorted] + ["Overall"]
    gini_vals = [tier_results[t]["gini"] for t in tiers_sorted] + [overall_gini]
    palma_vals = [tier_results[t]["palma"] for t in tiers_sorted] + [overall_palma]

    x = np.arange(len(labels))
    w = 0.36
    bars_g = ax.bar(x - w / 2, gini_vals, w, color="#1f77b4", label="Gini")
    ax2 = ax.twinx()
    bars_p = ax2.bar(x + w / 2, palma_vals, w, color="#ff7f0e", alpha=0.85,
                     label="Palma")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gini coefficient", color="#1f77b4")
    ax2.set_ylabel("Palma ratio", color="#ff7f0e")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#ff7f0e")
    ax.set_ylim(0, max(0.3, max(gini_vals) * 1.25))
    ax2.set_ylim(0, max(2.5, max(palma_vals) * 1.25))
    ax.set_title("Inequality metrics by income tier")

    for bar, v in zip(bars_g, gini_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#1f77b4")
    for bar, v in zip(bars_p, palma_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#ff7f0e")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("v2.2 baseline — accessibility inequality "
                 f"(β = {BETA_WEBTAG_MID}, gravity model)", fontsize=12)
    plt.tight_layout()

    out_png = OUT_DIR / "v22_inequality.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")

    # Print a small summary the parent agent can echo upstream
    print("\n=== SUMMARY (for caller) ===")
    print(f"Overall:  Gini={overall_gini:.4f}   Palma={overall_palma:.3f}")
    for t in tiers_sorted:
        r = tier_results[t]
        print(f"Tier {t} {TIER_LABELS[t]:>4}:  Gini={r['gini']:.4f}   "
              f"Palma={r['palma']:.3f}   n={r['n']}")


if __name__ == "__main__":
    main()
