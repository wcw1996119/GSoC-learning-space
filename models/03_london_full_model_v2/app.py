"""v2 interactive demo (Solara) — English UI.

4 tabs:
  1. Map view — choropleth of employment / population / POI / V_j(t)
  2. Counterfactual — pick a grid, slide employment delta, see predicted inflow change
  3. Validation — training curves + 3-model cross-check + plausibility framework
  4. About — plain-language summary

Launch:
  solara run app.py
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import solara

import torch

V2_ROOT = Path(__file__).resolve().parent
PROC = V2_ROOT / "data" / "processed"
EVAL = V2_ROOT / "evaluation_outputs"

# =========================================================
# DATA LOADING (cached)
# =========================================================

@functools.lru_cache(maxsize=1)
def load_cache():
    return dict(np.load(PROC / "demo_cache.npz", allow_pickle=True))

@functools.lru_cache(maxsize=1)
def load_grid_geo():
    grid = gpd.read_file(PROC / "london_1km_grid.geojson")
    grid = grid.sort_values("grid_id").reset_index(drop=True)
    return grid

@functools.lru_cache(maxsize=1)
def load_static_features():
    return pd.read_csv(PROC / "grid_static_features.csv")

@functools.lru_cache(maxsize=1)
def load_methodology_summary():
    p = V2_ROOT / "views" / "methodology_summary_en.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "*(summary not generated yet)*"


@functools.lru_cache(maxsize=1)
def load_model():
    """Load trained STGNN. Used only for counterfactual tab."""
    import sys
    sys.path.insert(0, str(V2_ROOT))
    from models_lib.stgnn import V2_STGNN
    cache = load_cache()
    model = V2_STGNN(
        node_dim=cache["static_features"].shape[1] + 4,
        edge_dim=3,
        hidden_dim=64, gat_heads=4, gru_hidden=32,
    )
    model.load_state_dict(torch.load(V2_ROOT / "best_model.pt", map_location="cpu"))
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def load_graph_and_features():
    """Build edge_index, edge_attr, x_seq once."""
    import sys
    sys.path.insert(0, str(V2_ROOT))
    from data_loader import build_features_tensor
    from providers.features import LondonFeatureProvider
    from providers.graph_builder import build_knn_graph
    cache = load_cache()
    fp = LondonFeatureProvider()
    x_seq = build_features_tensor(fp, T=24)
    edge_index, edge_attr = build_knn_graph(cache["coords_bng"], K=10, add_self_loop=True)
    return x_seq, edge_index, edge_attr, fp


# =========================================================
# REACTIVE STATE
# =========================================================
choropleth_var = solara.reactive("total_employment")
choropleth_hour = solara.reactive(8)
selected_grid_id = solara.reactive("L029_026")  # default: City of London area
employment_delta = solara.reactive(0)
last_counterfactual_result = solara.reactive(None)

# Heterogeneity tab state — two agent profiles to compare
hetero_profile_a = solara.reactive("low / pt")     # high cost sensitivity
hetero_profile_b = solara.reactive("high / car")   # low cost sensitivity
hetero_home_grid = solara.reactive("L029_026")     # shared origin for fair comparison
hetero_hour = solara.reactive(8)


@functools.lru_cache(maxsize=1)
def load_agent_population():
    return pd.read_csv(PROC / "agent_population.csv")


# =========================================================
# UTIL: matplotlib choropleth on 1km grid
# =========================================================

def plot_choropleth(values: np.ndarray, title: str, cmap="viridis", log_scale=False, ax=None):
    grid = load_grid_geo().copy()
    grid["_value"] = values
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure
    if log_scale:
        norm = LogNorm(vmin=max(values[values > 0].min(), 1e-3), vmax=values.max())
    else:
        norm = None
    grid.plot(
        column="_value", ax=ax, cmap=cmap, norm=norm,
        legend=True, legend_kwds={"shrink": 0.6, "label": title},
        edgecolor="none", linewidth=0,
    )
    ax.set_axis_off()
    ax.set_title(title)
    return fig


# =========================================================
# TAB 1: MAP
# =========================================================

CHOROPLETH_OPTIONS = {
    "total_employment": ("Total employment (jobs)", "Reds", True),
    "population": ("Total population", "Blues", True),
    "poi_total": ("POI count", "Greens", True),
    "subway_station_count": ("Subway stations", "Purples", False),
    "V_baseline": ("Model attractiveness V_j (selected hour)", "plasma", False),
}

@solara.component
def MapTab():
    feat = load_static_features()
    cache = load_cache()
    var = choropleth_var.value
    title, cmap, log_scale = CHOROPLETH_OPTIONS[var]

    if var == "V_baseline":
        values = cache["V_jt_baseline"][choropleth_hour.value]
    else:
        values = feat[var].values

    with solara.Card("Map controls"):
        solara.Select(
            label="Variable to display",
            value=choropleth_var,
            values=list(CHOROPLETH_OPTIONS.keys()),
        )
        if var == "V_baseline":
            solara.SliderInt(
                label=f"Hour of day (h={choropleth_hour.value})",
                value=choropleth_hour, min=0, max=23,
            )

    fig = plot_choropleth(values, title, cmap=cmap, log_scale=log_scale)
    solara.FigureMatplotlib(fig, format="png")
    plt.close(fig)


# =========================================================
# TAB 2: COUNTERFACTUAL
# =========================================================

def get_top_employment_grids(n=30):
    feat = load_static_features()
    return feat.nlargest(n, "total_employment")[["grid_id", "total_employment", "centroid_lat", "centroid_lon"]]


def run_counterfactual(grid_id: str, delta: int):
    """Run STGNN with perturbed grid_id + delta employment. Returns dict with results."""
    import sys
    sys.path.insert(0, str(V2_ROOT))
    from models_lib.rum import rum_closure
    from providers.features import STATIC_COLS, LOG_TRANSFORM_COLS

    cache = load_cache()
    grid_ids = cache["grid_ids"].tolist()
    target_idx = grid_ids.index(grid_id)
    feat = load_static_features().copy()
    base_emp = float(feat.loc[target_idx, "total_employment"])
    base_office_sec = float(feat.loc[target_idx, "sec6_info_finance"])

    feat.loc[target_idx, "total_employment"] = max(base_emp + delta, 0)
    feat.loc[target_idx, "sec6_info_finance"] = max(base_office_sec + delta, 0)

    # Re-normalize using cached training stats (don't re-fit)
    raw = feat[STATIC_COLS].values.astype(np.float64)
    log_idx = [i for i, c in enumerate(STATIC_COLS) if c in LOG_TRANSFORM_COLS]
    raw[:, log_idx] = np.log1p(raw[:, log_idx])
    static_norm = (raw - cache["static_mean"]) / cache["static_std"]

    # Build feature tensor
    x_seq, edge_index, edge_attr, fp = load_graph_and_features()
    x_seq_new = x_seq.clone()
    F_static = static_norm.shape[1]
    for t in range(24):
        x_seq_new[t, :, :F_static] = torch.tensor(static_norm, dtype=torch.float32)

    # Run model
    model = load_model()
    with torch.no_grad():
        V_jt_new = model(x_seq_new, edge_index, edge_attr)
        t_ij_t = torch.tensor(cache["t_ij_t"])
        log_p = rum_closure(V_jt_new, t_ij_t, beta=0.07)
        F_ij_t = torch.tensor(cache["F_ij_t"])
        origin_total = F_ij_t.sum(dim=2, keepdim=True)
        pred_F_new = log_p.exp() * origin_total

    pred_F_baseline = torch.tensor(cache["pred_F_baseline"])
    delta_F = (pred_F_new - pred_F_baseline).numpy()  # (T, N, N)

    inflow_baseline = pred_F_baseline.sum(dim=(0, 1)).numpy()  # (N,)
    inflow_new = pred_F_new.sum(dim=(0, 1)).numpy()
    delta_inflow = inflow_new - inflow_baseline

    # Plausibility distance
    train_static = cache["static_features"][cache["train_mask"]]
    target_vec = static_norm[target_idx]
    dists = np.sqrt(((train_static - target_vec) ** 2).sum(axis=1))
    plaus = float(np.partition(dists, 10)[:10].mean())

    # Reference plausibility distribution
    train_self = cache["static_features"][cache["train_mask"]]
    sample = train_self[:50]
    train_dists = []
    for i in range(len(sample)):
        rest = np.delete(train_self, i, axis=0)
        d = np.sqrt(((rest - sample[i]) ** 2).sum(axis=1))
        train_dists.append(np.partition(d, 10)[:10].mean())
    p50, p90 = np.percentile(train_dists, [50, 90])

    if plaus < p50:
        tier = "In-distribution (quantitative reading OK)"
        tier_color = "#2e7d32"
    elif plaus < p90:
        tier = "Near-distribution (directional only)"
        tier_color = "#f57f17"
    else:
        tier = "Far OOD (do not interpret quantitatively)"
        tier_color = "#c62828"

    return {
        "grid_id": grid_id,
        "delta": delta,
        "base_emp": base_emp,
        "new_emp": base_emp + delta,
        "delta_inflow_per_grid": delta_inflow,
        "delta_at_target": float(delta_inflow[target_idx]),
        "plausibility": plaus,
        "p50": float(p50),
        "p90": float(p90),
        "tier": tier,
        "tier_color": tier_color,
    }


@solara.component
def CounterfactualTab():
    top_grids = get_top_employment_grids(50)
    options = {
        f"{r['grid_id']}  (lat={r['centroid_lat']:.4f}, current jobs {r['total_employment']:,.0f})": r["grid_id"]
        for _, r in top_grids.iterrows()
    }

    with solara.Card("Scenario settings"):
        solara.Markdown(
            "Pick a high-employment grid (default = City of London). Move the slider to "
            "**add Delta jobs to that grid**, then click the button. The model predicts: "
            "(a) which origin grids will send more commuters to this destination, "
            "(b) the credibility tier of this counterfactual input."
        )
        with solara.ColumnsResponsive(default=[12], small=[6, 6]):
            solara.Select(
                label="Target grid (top 50 by current employment)",
                value=selected_grid_id,
                values=list(options.values()),
            )
            solara.SliderInt(
                label=f"Employment delta = {employment_delta.value:+,d}",
                value=employment_delta, min=-50_000, max=200_000, step=5_000,
            )

        def on_compute():
            result = run_counterfactual(selected_grid_id.value, employment_delta.value)
            last_counterfactual_result.value = result
        solara.Button(
            "Run counterfactual prediction (about 5 seconds, please wait)",
            on_click=on_compute, color="primary",
        )

    res = last_counterfactual_result.value
    if res is None:
        solara.Info("Click the button above. The STGNN forward pass takes about 5 seconds.")
        return

    with solara.Card("Scenario results"):
        with solara.Row():
            with solara.Card("Target grid", style={"flex": 1}):
                solara.Markdown(
                    f"- **{res['grid_id']}**\n"
                    f"- Current employment: {res['base_emp']:,.0f}\n"
                    f"- Counterfactual employment: **{res['new_emp']:,.0f}** "
                    f"(Delta {res['delta']:+,d})\n"
                    f"- Predicted inflow change at target: **{res['delta_at_target']:+.2f}** trips/day"
                )
            with solara.Card("Credibility tier", style={"flex": 1}):
                solara.Markdown(
                    f"- Plausibility distance: **{res['plausibility']:.3f}**\n"
                    f"- Training 50pct = {res['p50']:.3f}, 90pct = {res['p90']:.3f}\n"
                    f"- Tier: <span style='color:{res['tier_color']}; "
                    f"font-weight:bold;'>{res['tier']}</span>",
                    style={"line-height": "1.7"},
                )

    with solara.Card("Counterfactual inflow change map (delta per grid)"):
        di = res["delta_inflow_per_grid"]
        fig, ax = plt.subplots(figsize=(9, 7))
        grid = load_grid_geo().copy()
        grid["_dF"] = di
        cap = max(abs(di).max(), 1e-6)
        grid.plot(
            column="_dF", ax=ax, cmap="RdBu_r",
            vmin=-cap, vmax=cap,
            legend=True, legend_kwds={"shrink": 0.6, "label": "Delta inflow (trips/day)"},
            edgecolor="none",
        )
        cache = load_cache()
        grid_ids = cache["grid_ids"].tolist()
        target_idx = grid_ids.index(res["grid_id"])
        target_geom = grid.iloc[[target_idx]]
        target_geom.boundary.plot(ax=ax, color="black", linewidth=2)
        ax.set_axis_off()
        ax.set_title("Counterfactual inflow change (black outline = target grid)")
        solara.FigureMatplotlib(fig, format="png")
        plt.close(fig)


# =========================================================
# TAB 3: VALIDATION
# =========================================================

@solara.component
def ValidationTab():
    solara.Markdown("""
    ## Model validation

    v2 STGNN trained on London 1km grid with spatial holdout 75 / 10 / 15, 30 epochs.
    """)

    img1 = EVAL / "loss_curve.png"
    if img1.exists():
        with solara.Card("Training trajectory"):
            solara.Image(str(img1))
            solara.Markdown(
                "**NLL decreases monotonically** (17.76 -> 16.50). "
                "CPC slowly rises (0.20 -> 0.22). "
                "Loss has not plateaued — longer training would push it further."
            )

    img2 = EVAL / "cross_check.png"
    if img2.exists():
        with solara.Card("Multi-model cross-check"):
            solara.Image(str(img2))
            solara.Markdown(
                "**STGNN beats gravity by 12%** (CPC 0.226 vs 0.202).\n\n"
                "**But still below 'observed-prior RUM' at 0.41** — the model has not yet "
                "fully learnt the OD signal. More epochs and ablations should close this gap."
            )

    img3 = EVAL / "magnitude_scan.png"
    if img3.exists():
        with solara.Card("Magnitude scan + Plausibility"):
            solara.Image(str(img3))
            solara.Markdown(
                "Counterfactual scan on City of London (L029_026), employment delta = 0..100k:\n\n"
                "- **Left**: monotonic, smooth response — the model behaves consistently in this range.\n"
                "- **Right**: all points are above the 90th-percentile training distance line. "
                "This is far OOD, so we should only interpret the direction, not the magnitude."
            )


# =========================================================
# TAB 4: PLAIN-LANGUAGE SUMMARY (English)
# =========================================================

@solara.component
def MethodologyTab():
    solara.Markdown(load_methodology_summary())


# =========================================================
# TAB 5: HETEROGENEITY (v2.2)
# =========================================================

# Build the 9 (income, mode) profile labels once
_INCOME_LABELS = ["low", "mid", "high"]   # tier 1, 2, 3
_MODE_LABELS = ["car", "pt", "active"]
PROFILE_OPTIONS = [f"{inc} / {mode}" for inc in _INCOME_LABELS for mode in _MODE_LABELS]


def _parse_profile(label: str):
    """'low / pt' -> (income_tier=1, mode='pt', mode_idx=1)."""
    inc_str, mode_str = [s.strip() for s in label.split("/")]
    income_tier = _INCOME_LABELS.index(inc_str) + 1   # 1-indexed
    from models_lib.heterogeneous_utility import MODE_TO_IDX
    return income_tier, mode_str, MODE_TO_IDX[mode_str]


@functools.lru_cache(maxsize=1)
def _hetero_imports():
    """Import models_lib.heterogeneous_utility lazily and cache."""
    import sys
    sys.path.insert(0, str(V2_ROOT))
    from models_lib.heterogeneous_utility import (
        BETA_INCOME, BETA_MODE, MODE_TO_IDX, beta_for,
    )
    return BETA_INCOME.numpy(), BETA_MODE.numpy(), MODE_TO_IDX, beta_for


def _predict_p_given_profile(home_grid_idx: int, hour: int,
                              income_tier: int, mode_idx: int):
    """Predict P(j | i, hour) for one agent profile using cached V_j(t), t_ij(t).

    Uses simplified RUM:  logits_j = V_j(hour) - beta * t_ij(hour)
    where beta = BETA_INCOME[tier-1] * BETA_MODE[mode_idx].

    Note: this does NOT load v2.2 trained weights — those are not yet trained.
    The cached V_j(t) baseline is from the existing STGNN (linear_utility_best.pt
    or best_model.pt). Only β varies across profiles, which is exactly what the
    heterogeneity demo needs to illustrate.
    """
    cache = load_cache()
    BETA_INCOME, BETA_MODE, _, _ = _hetero_imports()
    V_j = cache["V_jt_baseline"][hour]                 # (N,)
    t_ij_h = np.asarray(cache["t_ij_t"])
    if t_ij_h.ndim == 3:
        t_ij_h = t_ij_h[hour]                          # (N, N)
    beta = float(BETA_INCOME[income_tier - 1] * BETA_MODE[mode_idx])
    logits = V_j[None, :] - beta * t_ij_h              # (N, N), broadcast V_j over rows
    row = logits[home_grid_idx]                        # (N,)
    row = row - row.max()                              # softmax stability
    p = np.exp(row)
    p = p / p.sum()
    return p, beta


@solara.component
def HeterogeneityTab():
    solara.Markdown("""
    ## Agent heterogeneity (v2.2)

    Each agent has a personal cost-sensitivity
    **beta = beta_income[tier] * beta_mode[mode]**.
    Low-income agents penalize travel time more heavily; PT and active modes carry
    higher per-minute disutility than car. This page illustrates how that lookup
    table reshapes destination choice.
    """)

    BETA_INCOME, BETA_MODE, MODE_TO_IDX, beta_for = _hetero_imports()

    # ---------- 1. β lookup table heatmap ----------
    with solara.Card("Heterogeneous beta lookup (income x mode, per-minute disutility)"):
        beta_grid = np.outer(BETA_INCOME, BETA_MODE)   # (3, 3)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(beta_grid, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(_MODE_LABELS)))
        ax.set_xticklabels(_MODE_LABELS)
        ax.set_yticks(range(len(_INCOME_LABELS)))
        ax.set_yticklabels([f"{lbl} (tier {i+1})" for i, lbl in enumerate(_INCOME_LABELS)])
        for i in range(beta_grid.shape[0]):
            for j in range(beta_grid.shape[1]):
                ax.text(j, i, f"{beta_grid[i, j]:.3f}",
                        ha="center", va="center", color="black", fontsize=10)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Income tier")
        ax.set_title("beta = beta_income x beta_mode  (redder = higher cost sensitivity)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="beta (per minute)")
        solara.FigureMatplotlib(fig, format="png")
        plt.close(fig)
        solara.Markdown(
            "- **beta_income** = [0.12, 0.07, 0.04]  (low/mid/high; WebTAG VOT priors)\n"
            "- **beta_mode**   = [1.0, 1.2, 1.5]   (car/pt/active multipliers)\n"
            "- High-cost-sensitivity agents (top-left) accept much shorter commutes."
        )

    # ---------- 2. Side-by-side agent comparison ----------
    cache = load_cache()
    grid_ids = cache["grid_ids"].tolist()
    if hetero_home_grid.value not in grid_ids:
        hetero_home_grid.value = grid_ids[0]

    with solara.Card("Compare two agent profiles — predicted P(j | i, hour)"):
        # Pick home grid from top-30 by employment for relevance
        top_origins = get_top_employment_grids(30)
        origin_options = {
            f"{r['grid_id']}  (jobs={r['total_employment']:,.0f})": r["grid_id"]
            for _, r in top_origins.iterrows()
        }
        if hetero_home_grid.value not in origin_options.values():
            origin_options[hetero_home_grid.value] = hetero_home_grid.value

        with solara.ColumnsResponsive(default=[12], small=[4, 4, 4]):
            solara.Select(
                label="Shared home grid (origin i)",
                value=hetero_home_grid,
                values=list(origin_options.values()),
            )
            solara.Select(
                label="Agent A profile (income / mode)",
                value=hetero_profile_a,
                values=PROFILE_OPTIONS,
            )
            solara.Select(
                label="Agent B profile (income / mode)",
                value=hetero_profile_b,
                values=PROFILE_OPTIONS,
            )
        solara.SliderInt(
            label=f"Hour of day (h={hetero_hour.value})",
            value=hetero_hour, min=0, max=23,
        )

        try:
            home_idx = grid_ids.index(hetero_home_grid.value)
        except ValueError:
            home_idx = 0

        tier_a, mode_a, mode_idx_a = _parse_profile(hetero_profile_a.value)
        tier_b, mode_b, mode_idx_b = _parse_profile(hetero_profile_b.value)

        p_a, beta_a = _predict_p_given_profile(home_idx, hetero_hour.value,
                                                tier_a, mode_idx_a)
        p_b, beta_b = _predict_p_given_profile(home_idx, hetero_hour.value,
                                                tier_b, mode_idx_b)

        solara.Markdown(
            f"- **Agent A** ({hetero_profile_a.value}): effective beta = **{beta_a:.4f}** /min\n"
            f"- **Agent B** ({hetero_profile_b.value}): effective beta = **{beta_b:.4f}** /min\n"
            f"- Hour: **{hetero_hour.value}:00**, origin: **{hetero_home_grid.value}**"
        )

        # Top-10 destinations: union of both agents' top-10
        top_a = set(np.argsort(-p_a)[:10].tolist())
        top_b = set(np.argsort(-p_b)[:10].tolist())
        top_union = sorted(top_a | top_b, key=lambda j: -(p_a[j] + p_b[j]))[:10]

        labels = [grid_ids[j] for j in top_union]
        vals_a = [p_a[j] for j in top_union]
        vals_b = [p_b[j] for j in top_union]

        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(labels))
        h = 0.4
        ax.barh(y - h/2, vals_a, height=h,
                label=f"A: {hetero_profile_a.value}", color="#1976d2")
        ax.barh(y + h/2, vals_b, height=h,
                label=f"B: {hetero_profile_b.value}", color="#e53935")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("P(destination j | origin i, hour)")
        ax.set_title(f"Top-10 predicted destinations from {hetero_home_grid.value} "
                     f"at h={hetero_hour.value}")
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        solara.FigureMatplotlib(fig, format="png")
        plt.close(fig)
        solara.Markdown(
            "*Caption*: high-income / car agents (low beta) are willing to commute "
            "farther — their probability mass spreads over distant high-V_j destinations. "
            "Low-income / active agents (high beta) concentrate on grids near home."
        )

    # ---------- 3. Agent population breakdown ----------
    with solara.Card("Agent population breakdown (n=10,000)"):
        agents = load_agent_population()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Pie 1 — income tier
        tier_counts = agents["income_tier"].value_counts().sort_index()
        tier_labels = [f"{_INCOME_LABELS[t-1]} (tier {t})" for t in tier_counts.index]
        axes[0].pie(tier_counts.values, labels=tier_labels, autopct="%1.1f%%",
                    colors=["#ef5350", "#ffb74d", "#81c784"])
        axes[0].set_title("By income tier")

        # Pie 2 — mode
        mode_counts = agents["mode_initial"].value_counts()
        mode_colors_map = {"car": "#1976d2", "pt": "#7e57c2", "active": "#26a69a"}
        axes[1].pie(mode_counts.values, labels=mode_counts.index, autopct="%1.1f%%",
                    colors=[mode_colors_map.get(m, "#999") for m in mode_counts.index])
        axes[1].set_title("By initial mode")

        # Stacked bar — income x mode joint
        joint = (agents.groupby(["income_tier", "mode_initial"])
                       .size().unstack(fill_value=0))
        # Reorder columns to car/pt/active if present
        col_order = [m for m in _MODE_LABELS if m in joint.columns]
        joint = joint[col_order]
        bottom = np.zeros(len(joint))
        x = np.arange(len(joint))
        for m in joint.columns:
            axes[2].bar(x, joint[m].values, bottom=bottom, label=m,
                        color=mode_colors_map.get(m, "#999"))
            bottom += joint[m].values
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"tier {t}\n({_INCOME_LABELS[t-1]})" for t in joint.index])
        axes[2].set_ylabel("Agents")
        axes[2].set_title("Income tier x mode (stacked)")
        axes[2].legend(title="Mode", loc="upper right")

        plt.tight_layout()
        solara.FigureMatplotlib(fig, format="png")
        plt.close(fig)
        solara.Markdown(
            "Joint distribution of (income, mode) feeds the population-weighted "
            "beta_origin used during aggregate training, while each agent uses its "
            "own personal beta at inference."
        )

    solara.Info(
        "Note: the predicted P(j|i,t) above uses cached V_j(t) from the existing "
        "STGNN baseline (best_model.pt / linear_utility_best.pt). The v2.2 model "
        "with heterogeneous beta in the loss has not yet been trained — only the "
        "per-agent beta differs across profiles in this demo, which is the "
        "qualitative effect we want to show."
    )


# =========================================================
# MAIN PAGE
# =========================================================

@solara.component
def Page():
    solara.Style("""
    .v-application { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
    .v-card { margin-bottom: 14px; }
    """)

    with solara.AppBar():
        solara.AppBarTitle("London Full Model v2 — Interactive Demo")

    with solara.lab.Tabs():
        with solara.lab.Tab("Map"):
            MapTab()
        with solara.lab.Tab("Counterfactual"):
            CounterfactualTab()
        with solara.lab.Tab("Validation"):
            ValidationTab()
        with solara.lab.Tab("Heterogeneity"):
            HeterogeneityTab()
        with solara.lab.Tab("About"):
            MethodologyTab()


if __name__ == "__main__":
    pass
