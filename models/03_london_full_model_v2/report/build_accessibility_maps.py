"""Compute job accessibility metrics + generate maps for the report.

Audience: urban researchers + general public.
Anchor: job accessibility A_i = sum_j E_j * exp(-beta * t_ij)
        + socio-spatial inequality A_i broken out by income tier.

Outputs (PNG into report/figures/):
  M1_employment.png        Where the jobs are
  M2_population.png        Where people live
  M3_accessibility.png     Hansen A_i baseline
  M4_access_by_income.png  3-panel A_i for tier 1/2/3 residents (gap)
  M5_scenarioA_delta.png   Old Oak Common: dA_i (free-flow)
  M6_scenarioB_share.png   Share of peak-hour commuters per home cell
  M7_inequality_summary.png  Lorenz / decile bars

Also writes report/data/accessibility_per_grid.csv summarising metrics per cell
so the report text can quote numbers consistently.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT_FIG = ROOT / "report" / "figures"
OUT_DATA = ROOT / "report" / "data"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

# Travel-time decay (per minute). Recovered from trained Phase B+ model:
# evaluation_outputs/paper_a/scenario_A_accessibility.json -> beta_t_recovered = -0.058
BETA_T = 0.058

# Hours over which to average t_ij when defining "typical" travel time
PEAK_HOURS = [7, 8, 9]  # AM peak, when most commute trips occur

# Old Oak Common cluster (from scenario_A_accessibility.json -> "grids")
OOC_GRIDS = [907, 908, 909, 862, 863, 864, 951, 952, 953]
OOC_E_ADDED_TOTAL = 65000.0  # jobs added across the cluster


def load_inputs():
    print("[load] geojson + features + cache + agents")
    grid = gpd.read_file(DATA / "london_1km_grid.geojson").set_index("grid_id")
    feats = pd.read_csv(DATA / "grid_static_features.csv").set_index("grid_id")
    boroughs = pd.read_csv(DATA / "grid_borough_mapping.csv").set_index("grid_id")
    agents = pd.read_csv(DATA / "agent_population_v23.csv")
    cache = np.load(DATA / "demo_cache.npz", allow_pickle=True)
    grid_ids = cache["grid_ids"].tolist()  # canonical order matching t_ij_t axes
    t_ij_t = np.asarray(cache["t_ij_t"])   # (24, N, N)
    feats = feats.reindex(grid_ids)
    boroughs = boroughs.reindex(grid_ids)
    grid = grid.reindex(grid_ids)
    print(f"  N grids = {len(grid_ids)}, agents = {len(agents)}, t_ij_t = {t_ij_t.shape}")
    return dict(
        grid=grid, feats=feats, boroughs=boroughs, agents=agents,
        grid_ids=grid_ids, t_ij_t=t_ij_t,
    )


def hansen_accessibility(t_ij: np.ndarray, employment: np.ndarray,
                          beta: float = BETA_T) -> np.ndarray:
    """A_i = sum_j E_j * exp(-beta * t_ij). Returns shape (N,)."""
    decay = np.exp(-beta * t_ij)  # (N, N)
    A = decay @ employment        # (N,)
    return A


def compute_baseline_accessibility(t_ij_t, feats):
    print("[A] baseline Hansen accessibility (peak-hour averaged t_ij)")
    t_peak = t_ij_t[PEAK_HOURS].mean(axis=0)              # (N, N)
    employment = feats["total_employment"].fillna(0).to_numpy().astype(np.float32)
    A = hansen_accessibility(t_peak, employment, BETA_T)
    print(f"  A: mean={A.mean():.0f}  p10={np.percentile(A,10):.0f}  "
          f"p50={np.percentile(A,50):.0f}  p90={np.percentile(A,90):.0f}")
    return A, t_peak, employment


def compute_scenarioA_delta(t_peak, employment, beta=BETA_T):
    """Free-flow approximation: only count NEW jobs at OOC.

    dA_i = sum_{j in OOC} dE_j * exp(-beta * t_ij)
    Note: this ignores choice substitution + congestion equilibrium. The
    full equilibrium delta is reported as a single headline number from the
    saved scenario_A_accessibility.json; this map shows the *opportunity
    field* of the intervention, which is what residents would experience
    before behavioural & congestion responses dampen it.
    """
    print("[A] scenario A (Old Oak Common) free-flow dA per cell")
    delta_E_per_cell = OOC_E_ADDED_TOTAL / len(OOC_GRIDS)
    decay_to_OOC = np.exp(-beta * t_peak[:, OOC_GRIDS])  # (N, k)
    dA = decay_to_OOC.sum(axis=1) * delta_E_per_cell
    print(f"  dA: mean={dA.mean():.0f}  max={dA.max():.0f}")
    return dA


def compute_per_income_accessibility(A, agents, grid_ids):
    """Mean A_i across the home cells of agents in each income tier.

    Result is a (3,) vector: low/mid/high tier mean access.
    Also returns per-grid 'resident income tier mode' for the income map.
    """
    print("[A] accessibility by resident income tier")
    grid_idx_of = {g: i for i, g in enumerate(grid_ids)}
    home_idx = agents["home_grid_id"].map(grid_idx_of).dropna().astype(int).to_numpy()
    valid = agents.dropna(subset=["home_grid_id"]).copy()
    valid["A_home"] = A[home_idx]
    tier_means = valid.groupby("income_tier")["A_home"].mean()
    print("  mean A by tier:")
    for t, v in tier_means.items():
        print(f"    tier {t}: {v:>10.0f}")

    # Per-grid: dominant income tier of residents (for income-cluster map)
    per_grid = (valid.groupby(["home_grid_id", "income_tier"]).size()
                      .unstack(fill_value=0))
    per_grid.columns = [f"n_t{int(c)}" for c in per_grid.columns]
    per_grid["n_total"] = per_grid.sum(axis=1)
    per_grid["share_t1"] = per_grid["n_t1"] / per_grid["n_total"]
    per_grid["share_t3"] = per_grid["n_t3"] / per_grid["n_total"]
    return tier_means, per_grid


def compute_scenarioB_peak_share(F_ij_t):
    """Per home cell: fraction of trips that depart in 7-9am peak."""
    print("[A] scenario B context: per-cell peak-hour commute share")
    flows_per_home_t = F_ij_t.sum(axis=2)             # (24, N) outflow
    flows_per_home = flows_per_home_t.sum(axis=0)     # (N,)
    peak_flows = flows_per_home_t[PEAK_HOURS].sum(axis=0)
    share = np.where(flows_per_home > 0, peak_flows / np.maximum(flows_per_home, 1e-6), 0.0)
    return share


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.all(x == 0):
        return float("nan")
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


# ---------- plotting helpers ----------

def _setup_ax(ax, title, grid):
    minx, miny, maxx, maxy = grid.total_bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888"); spine.set_linewidth(0.4)
    ax.set_title(title, fontsize=11, loc="left", pad=6, fontweight="bold")


def _borough_outlines(grid, boroughs):
    """Dissolve grid by borough to get a borough overlay."""
    g = grid.copy()
    g["borough"] = boroughs["borough"].values
    return g.dissolve(by="borough", as_index=False)


def make_map_choropleth(grid, values, title, cmap, label,
                         outpath, vmin=None, vmax=None, log=False,
                         center=None, borough_overlay=None,
                         annotate_grids: dict | None = None):
    fig, ax = plt.subplots(figsize=(8, 7))
    plotting = grid.copy()
    plotting["v"] = values
    if log:
        # log-scale: shift to avoid zeros
        v_pos = plotting["v"][plotting["v"] > 0]
        vmin_eff = vmin or np.percentile(v_pos, 1)
        vmax_eff = vmax or np.percentile(v_pos, 99)
        norm = mpl.colors.LogNorm(vmin=max(vmin_eff, 1e-6), vmax=vmax_eff)
    elif center is not None:
        vmax_eff = vmax or np.nanpercentile(np.abs(plotting["v"]), 99)
        norm = TwoSlopeNorm(vcenter=center, vmin=-vmax_eff, vmax=vmax_eff)
    else:
        vmin_eff = vmin if vmin is not None else np.nanpercentile(plotting["v"], 2)
        vmax_eff = vmax if vmax is not None else np.nanpercentile(plotting["v"], 98)
        norm = mpl.colors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
    plotting.plot(column="v", cmap=cmap, norm=norm, ax=ax,
                  linewidth=0.0, edgecolor="none")
    if borough_overlay is not None:
        borough_overlay.boundary.plot(ax=ax, linewidth=0.4, color="#333", alpha=0.6)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if annotate_grids:
        for label_text, idx_list in annotate_grids.items():
            sub = grid.iloc[idx_list]
            cx, cy = sub.geometry.unary_union.centroid.coords[0]
            ax.scatter(cx, cy, s=80, marker="*", color="black",
                        edgecolor="white", linewidth=0.7, zorder=5)
            ax.annotate(label_text, (cx, cy),
                         xytext=(8, 8), textcoords="offset points",
                         fontsize=8.5, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
    _setup_ax(ax, title, grid)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def make_three_panel_income(grid, A, per_grid, tier_means,
                              borough_overlay, outpath):
    """One A_i map per dominant-income-tier subset."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    cmap = "viridis"
    vmin = np.nanpercentile(A, 5)
    vmax = np.nanpercentile(A, 95)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    grid_resid = grid.copy()
    grid_resid["A"] = A
    grid_resid = grid_resid.join(per_grid, how="left")
    grid_resid["share_t1"] = grid_resid["share_t1"].fillna(0)
    grid_resid["share_t3"] = grid_resid["share_t3"].fillna(0)
    # mid-tier share = 1 - t1 - t3
    grid_resid["share_t2"] = (1 - grid_resid["share_t1"] - grid_resid["share_t3"]).clip(lower=0)

    titles = [
        f"(a) Tier 1 (lower-income) residents\nmean A = {tier_means.get(1,0):,.0f}",
        f"(b) Tier 2 (middle-income) residents\nmean A = {tier_means.get(2,0):,.0f}",
        f"(c) Tier 3 (higher-income) residents\nmean A = {tier_means.get(3,0):,.0f}",
    ]
    cols = ["share_t1", "share_t2", "share_t3"]
    for ax, col, title in zip(axes, cols, titles):
        sub = grid_resid.copy()
        # weight cell colour by A but alpha = tier share
        sub["alpha"] = (sub[col].clip(0, 1)) ** 0.6
        sub.plot(column="A", cmap=cmap, norm=norm, ax=ax,
                  linewidth=0.0, edgecolor="none")
        # overlay translucent grey where this tier is rare
        mask = sub.copy()
        mask["v"] = 1 - mask["alpha"]
        mask.plot(column="v", ax=ax, cmap="Greys", alpha=0.5,
                   linewidth=0.0, edgecolor="none", legend=False,
                   vmin=0, vmax=1)
        if borough_overlay is not None:
            borough_overlay.boundary.plot(ax=ax, linewidth=0.4,
                                           color="#333", alpha=0.6)
        _setup_ax(ax, title, grid)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.01)
    cb.set_label("Hansen accessibility A_i (jobs equivalent)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        "(a–c) Job accessibility seen by residents of each income tier "
        "— darker grey = fewer residents of that tier live there",
        fontsize=11, fontweight="bold")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def make_inequality_summary(A, agents, grid_ids, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Lorenz curve
    ax = axes[0]
    A_sorted = np.sort(A[A > 0])
    cum_pop = np.arange(1, len(A_sorted) + 1) / len(A_sorted) * 100
    cum_A = np.cumsum(A_sorted) / A_sorted.sum() * 100
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=0.8, label="perfect equality")
    ax.plot(cum_pop, cum_A, color="#1f77b4", lw=2,
            label=f"actual (Gini = {gini(A):.2f})")
    ax.set_xlabel("Cumulative % of grid cells (sorted by A)", fontsize=10)
    ax.set_ylabel("Cumulative % of total accessibility", fontsize=10)
    ax.set_title("(a) Lorenz curve of cell-level accessibility",
                  fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle=":")

    # Decile bar chart per income tier
    ax = axes[1]
    grid_idx_of = {g: i for i, g in enumerate(grid_ids)}
    valid = agents.dropna(subset=["home_grid_id"]).copy()
    valid["home_idx"] = valid["home_grid_id"].map(grid_idx_of)
    valid = valid.dropna(subset=["home_idx"])
    valid["home_idx"] = valid["home_idx"].astype(int)
    valid["A_home"] = A[valid["home_idx"].values]

    tier_means = valid.groupby("income_tier")["A_home"].agg(["mean", "median"])
    x = np.arange(len(tier_means))
    width = 0.36
    ax.bar(x - width/2, tier_means["mean"], width, color="#1f77b4", label="mean A")
    ax.bar(x + width/2, tier_means["median"], width, color="#ff7f0e", label="median A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {int(t)}" for t in tier_means.index])
    ax.set_ylabel("Accessibility A_home (jobs equiv.)", fontsize=10)
    ax.set_title("(b) Accessibility by resident income tier",
                  fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    for i, (m, md) in enumerate(zip(tier_means["mean"], tier_means["median"])):
        ax.text(i - width/2, m * 1.02, f"{m/1000:.0f}K", ha="center", fontsize=8)
        ax.text(i + width/2, md * 1.02, f"{md/1000:.0f}K", ha="center", fontsize=8)
    ax.grid(alpha=0.3, linestyle=":", axis="y")

    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def write_summary_csv(grid_ids, feats, boroughs, A, dA_OOC, peak_share):
    df = pd.DataFrame({
        "grid_id": grid_ids,
        "borough": boroughs["borough"].values,
        "employment": feats["total_employment"].fillna(0).values,
        "population": feats["population"].fillna(0).values,
        "accessibility_A": A,
        "dA_scenarioA_freeflow": dA_OOC,
        "dA_scenarioA_pct": np.where(A > 0, 100 * dA_OOC / A, 0),
        "peak_share": peak_share,
    })
    p = OUT_DATA / "accessibility_per_grid.csv"
    df.to_csv(p, index=False)
    print(f"[summary] wrote {p}")
    return df


def main():
    inputs = load_inputs()
    grid, feats, boroughs, agents = (
        inputs["grid"], inputs["feats"], inputs["boroughs"], inputs["agents"]
    )
    grid_ids, t_ij_t = inputs["grid_ids"], inputs["t_ij_t"]

    A, t_peak, employment = compute_baseline_accessibility(t_ij_t, feats)
    dA_OOC = compute_scenarioA_delta(t_peak, employment)
    tier_means, per_grid = compute_per_income_accessibility(A, agents, grid_ids)

    cache = np.load(DATA / "demo_cache.npz", allow_pickle=True)
    F_ij_t = np.asarray(cache["F_ij_t"])
    peak_share = compute_scenarioB_peak_share(F_ij_t)

    boroughs_overlay = _borough_outlines(grid, boroughs)

    # M1 employment
    make_map_choropleth(
        grid, employment,
        "M1. Where the jobs are — total employment per 1×1 km cell (log scale)",
        "YlOrRd", "Jobs per cell (log)",
        OUT_FIG / "M1_employment.png", log=True,
        borough_overlay=boroughs_overlay,
        annotate_grids={"Old Oak Common": OOC_GRIDS},
    )

    # M2 population
    make_map_choropleth(
        grid, feats["population"].fillna(0),
        "M2. Where people live — residential population per 1×1 km cell (log)",
        "BuPu", "Residents per cell (log)",
        OUT_FIG / "M2_population.png", log=True,
        borough_overlay=boroughs_overlay,
    )

    # M3 baseline accessibility
    make_map_choropleth(
        grid, A,
        f"M3. Baseline job accessibility A_i (Hansen, β = {BETA_T:.3f}/min, "
        f"AM peak)\n     Gini across cells = {gini(A):.2f}",
        "viridis", "A_i (jobs-equivalent)",
        OUT_FIG / "M3_accessibility.png",
        borough_overlay=boroughs_overlay,
        annotate_grids={"Old Oak Common": OOC_GRIDS},
    )

    # M4 access by income tier
    make_three_panel_income(grid, A, per_grid, tier_means,
                              boroughs_overlay, OUT_FIG / "M4_access_by_income.png")

    # M5 scenario A delta
    make_map_choropleth(
        grid, dA_OOC,
        "M5. Scenario A (Old Oak Common, +65,000 jobs) — opportunity field "
        f"ΔA_i\n     ΔA total per worker reachability footprint ≈ {dA_OOC.sum():,.0f}",
        "YlGn", "ΔA_i (jobs-equivalent gained)",
        OUT_FIG / "M5_scenarioA_delta.png",
        borough_overlay=boroughs_overlay,
        annotate_grids={"OOC cluster": OOC_GRIDS},
    )

    # M6 scenario B context
    make_map_choropleth(
        grid, peak_share,
        "M6. Scenario B context — share of commute trips that depart in "
        "7–9am AM peak\n     (highest-share areas benefit most from off-peak shift)",
        "Oranges", "Peak-hour commute share",
        OUT_FIG / "M6_scenarioB_share.png",
        vmin=0, vmax=0.7,
        borough_overlay=boroughs_overlay,
    )

    # M7 inequality summary
    make_inequality_summary(A, agents, grid_ids, OUT_FIG / "M7_inequality_summary.png")

    df = write_summary_csv(grid_ids, feats, boroughs, A, dA_OOC, peak_share)

    # Print key numbers for the report text
    print("\n=== HEADLINE NUMBERS for report text ===")
    print(f"Cell-level Gini of A:               {gini(A):.3f}")
    print(f"Top decile / bottom decile ratio:   "
          f"{np.percentile(A, 90) / max(np.percentile(A, 10), 1):.1f}x")
    print(f"Mean A by income tier 1/2/3:        "
          f"{tier_means.get(1,0):,.0f} / {tier_means.get(2,0):,.0f} / {tier_means.get(3,0):,.0f}")
    print(f"Tier3 / Tier1 access ratio:         "
          f"{tier_means.get(3,0)/max(tier_means.get(1,0),1):.2f}x")
    print(f"Mean ΔA from OOC (free-flow):       {dA_OOC.mean():,.0f} "
          f"(= {100*dA_OOC.mean()/A.mean():.1f}% of mean A)")
    boroughs_summary = (df.assign(_w=df["population"])
                        .groupby("borough")
                        .apply(lambda x: (x["accessibility_A"] * x["_w"]).sum()
                                          / max(x["_w"].sum(), 1))
                        .sort_values())
    print(f"\nBottom 5 boroughs by pop-weighted A:")
    print(boroughs_summary.head().round(0))
    print(f"\nTop 5 boroughs by pop-weighted A:")
    print(boroughs_summary.tail().round(0))


if __name__ == "__main__":
    main()
