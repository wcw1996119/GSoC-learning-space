"""decision_explainer.py — explain why an ABM agent chose a particular workplace.

For v2.2 of the London commuting ABM. Given a fitted LondonV2Model and an
agent_id, decompose the agent's utility V_ij into named components and
contrast the chosen destination against top alternatives.

Components (raw, post-multiplication):
  1. alpha_V  * V_j(t)               GNN attractiveness of destination at hour t
  2. alpha_occ * OccMatch(soc, j)    affinity between agent's SOC and grid industry mix
  3. - beta(income, mode) * t_ij(t)  travel-time disutility (income & mode specific)
  4. - gamma * log d_ij              distance decay
  5. + bias                           constant offset

Components are reported in *normalized score units* (the units the utility was
trained in). Denormalising V_j, t_ij, log d, and OccMatch back to raw units
would change every component by an additive constant times the parameter, so
the relative magnitude (which is what we visualize) is preserved.

Public API:
  explain_choice(model, agent_id, top_k=5) -> dict
  explain_to_text(explanation) -> str
  explain_to_html(explanation) -> str
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

# Anchor to the v2 root for imports / data loads regardless of caller cwd.
V2_ROOT = Path(__file__).resolve().parent
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))


# Human-readable SOC labels (UK SOC 2010, 1..9)
SOC_LABELS = {
    1: "managers",
    2: "professionals",
    3: "assoc. professional",
    4: "admin/secretarial",
    5: "skilled trades",
    6: "caring/leisure",
    7: "sales/customer service",
    8: "plant/machine operatives",
    9: "elementary",
}

INCOME_LABELS = {1: "low", 2: "mid", 3: "high"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_lookup(model, agent_id: int):
    """Find an agent by either unique_id (mesa) or our explicit unique_id_."""
    for a in model.agents:
        # Mesa 3 sets a.unique_id; we also stash unique_id_ for safety
        if getattr(a, "unique_id_", None) == agent_id or getattr(a, "unique_id", None) == agent_id:
            return a
    raise KeyError(f"No agent with id={agent_id} found in model")


def _grid_id(model, grid_idx: int) -> str:
    """Return the textual grid_id (e.g. L010_020) for a row index."""
    try:
        return str(model.grid_features_df.iloc[grid_idx]["grid_id"])
    except Exception:
        return f"idx{grid_idx}"


def _is_heterogeneous(utility) -> bool:
    """Duck-type detection of HeterogeneousUtility vs LinearUtility."""
    return hasattr(utility, "utility_personal") and hasattr(utility, "beta_income")


def _component_breakdown(
    model,
    origin_idx: int,
    dest_idx: int,
    hour: int,
    soc_idx: int,
    income_tier: int,
    mode: str,
) -> Dict[str, float]:
    """Compute the 5 utility components for one (agent, dest) pair.

    All scalars are detached floats. Inputs are the *normalised* tensors
    that the utility was trained on, so components are in normalised units.
    """
    util = model.utility

    Vj = float(model.V_jt_norm[hour, dest_idx])
    om = float(model.om_per_soc_norm_t[soc_idx, dest_idx])
    tt = float(model.t_ij_t_norm[hour, origin_idx, dest_idx])
    ld = float(model.log_d_ij_norm[origin_idx, dest_idx])

    # Raw values for the narrative (minutes / km units, pre-normalisation)
    Vj_raw = float(model.V_jt_raw[hour, dest_idx])
    tt_raw = float(model.t_ij_t_raw[hour, origin_idx, dest_idx])
    d_km_raw = float(np.exp(float(model.log_d_ij_raw[origin_idx, dest_idx])))
    om_raw = float(model.om_per_soc[soc_idx, dest_idx])

    if _is_heterogeneous(util):
        from models_lib.heterogeneous_utility import MODE_TO_IDX

        mode_idx = MODE_TO_IDX.get(mode, 0)
        alpha_V = float(util.alpha_V.detach())
        alpha_occ = float(util.alpha_occ.detach())
        gamma = float(util.gamma.detach())
        beta = float(util.beta(income_tier, mode_idx).detach())
        bias = float(util.bias.detach())
    else:
        # LinearUtility — single beta, no income/mode dependence
        alpha_V = float(util.alpha_V.detach())
        alpha_occ = float(util.alpha_occ.detach())
        gamma = float(util.gamma.detach())
        beta = float(util.beta.detach())
        bias = float(util.bias.detach())

    c_V = alpha_V * Vj
    c_occ = alpha_occ * om
    c_time = -beta * tt
    c_dist = -gamma * ld
    c_bias = bias
    total = c_V + c_occ + c_time + c_dist + c_bias

    return {
        "dest_grid_idx": int(dest_idx),
        "dest_grid_id": _grid_id(model, dest_idx),
        "alpha_V_term": c_V,
        "alpha_occ_term": c_occ,
        "time_term": c_time,
        "distance_term": c_dist,
        "bias_term": c_bias,
        "total_utility": total,
        # Raw quantities for narrative readability
        "V_j_raw": Vj_raw,
        "OccMatch_raw": om_raw,
        "travel_time_min": tt_raw,
        "distance_km": d_km_raw,
        # Coefficients
        "alpha_V": alpha_V,
        "alpha_occ": alpha_occ,
        "beta": beta,
        "gamma": gamma,
    }


def _vectorised_total_utility(model, agent) -> torch.Tensor:
    """Compute total utility for ALL N destinations for `agent`.

    Returns a 1-D torch tensor of shape (N,) on CPU.
    """
    util = model.utility
    h = agent.departure_hour
    i = agent.home_grid_idx

    Vj = model.V_jt_norm[h]                          # (N,)
    om = model.om_per_soc_norm_t[agent.soc_idx]       # (N,)
    tt = model.t_ij_t_norm[h, i]                     # (N,)
    ld = model.log_d_ij_norm[i]                       # (N,)

    with torch.no_grad():
        if _is_heterogeneous(util):
            from models_lib.heterogeneous_utility import MODE_TO_IDX
            mode_idx = MODE_TO_IDX.get(agent.mode, 0)
            U = util.utility_personal(Vj, om, tt, ld, agent.income_tier, mode_idx)
        else:
            U = util(Vj, om, tt, ld)
    return U  # (N,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_choice(model, agent_id: int, top_k: int = 5) -> Dict[str, Any]:
    """Return a structured explanation of why agent #agent_id chose its workplace.

    The returned dict has:
      "agent_summary": str
      "chosen": (grid_id, breakdown_dict)
      "alternatives": list[ breakdown_dict ]   (top_k by total utility, excluding chosen)
      "narrative": str
      "meta": dict with utility_class, hour, etc.
    """
    agent = _agent_lookup(model, agent_id)
    if agent.work_grid_idx is None:
        raise RuntimeError(
            f"Agent {agent_id} has no work_grid_idx assigned. Call model.step() first."
        )

    soc_num = agent.soc_idx + 1
    summary = (
        f"Agent #{agent_id}, lives {_grid_id(model, agent.home_grid_idx)}, "
        f"SOC{soc_num} {SOC_LABELS.get(soc_num, '?')}, "
        f"{INCOME_LABELS.get(agent.income_tier, '?')} income, "
        f"{agent.mode.upper()}, departs {agent.departure_hour}:00"
    )

    # Total utility for every candidate destination
    U_all = _vectorised_total_utility(model, agent).numpy()

    # Identify chosen + ranked alternatives
    chosen_j = int(agent.work_grid_idx)
    chosen_breakdown = _component_breakdown(
        model, agent.home_grid_idx, chosen_j, agent.departure_hour,
        agent.soc_idx, agent.income_tier, agent.mode,
    )
    chosen_breakdown["rank"] = int((U_all > U_all[chosen_j]).sum()) + 1
    chosen_breakdown["log_p"] = float(
        U_all[chosen_j] - np.log(np.sum(np.exp(U_all - U_all.max()))) - U_all.max()
    )

    # Top-k alternatives (excluding chosen)
    order = np.argsort(-U_all)
    alt_indices = [int(j) for j in order if int(j) != chosen_j][:top_k]
    alternatives = []
    for j in alt_indices:
        bd = _component_breakdown(
            model, agent.home_grid_idx, int(j), agent.departure_hour,
            agent.soc_idx, agent.income_tier, agent.mode,
        )
        bd["rank"] = int((U_all > U_all[j]).sum()) + 1
        alternatives.append(bd)

    # Narrative — contrast chosen vs runner-up
    runner = alternatives[0] if alternatives else None
    if runner is not None:
        diffs = {
            "GNN attractiveness": chosen_breakdown["alpha_V_term"] - runner["alpha_V_term"],
            "occupation match": chosen_breakdown["alpha_occ_term"] - runner["alpha_occ_term"],
            "travel time": chosen_breakdown["time_term"] - runner["time_term"],
            "distance": chosen_breakdown["distance_term"] - runner["distance_term"],
        }
        dominant = max(diffs, key=lambda k: abs(diffs[k]))
        sign = "favoured" if diffs[dominant] > 0 else "penalised"
        narrative = (
            f"Chosen workplace {chosen_breakdown['dest_grid_id']} "
            f"(rank {chosen_breakdown['rank']}, total U={chosen_breakdown['total_utility']:.3f}) "
            f"vs runner-up {runner['dest_grid_id']} (U={runner['total_utility']:.3f}). "
            f"Dominant factor: '{dominant}' {sign} the chosen by "
            f"{abs(diffs[dominant]):.3f} score units. "
            f"Travel time {chosen_breakdown['travel_time_min']:.1f} min vs "
            f"{runner['travel_time_min']:.1f} min; distance "
            f"{chosen_breakdown['distance_km']:.1f} km vs {runner['distance_km']:.1f} km; "
            f"OccMatch {chosen_breakdown['OccMatch_raw']:.3f} vs {runner['OccMatch_raw']:.3f}."
        )
    else:
        narrative = "No alternatives available for comparison."

    util_class = type(model.utility).__name__
    return {
        "agent_summary": summary,
        "chosen": (chosen_breakdown["dest_grid_id"], chosen_breakdown),
        "alternatives": alternatives,
        "narrative": narrative,
        "meta": {
            "utility_class": util_class,
            "hour": agent.departure_hour,
            "n_candidates": int(model.N),
            "agent_id": agent_id,
        },
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_COMPONENTS = [
    ("alpha_V_term", "alpha_V * V_j(t)"),
    ("alpha_occ_term", "alpha_occ * OccMatch"),
    ("time_term", "- beta * t_ij(t)"),
    ("distance_term", "- gamma * log d_ij"),
    ("bias_term", "+ bias"),
]


def _fmt_breakdown_row(bd: Dict[str, Any], label: str) -> str:
    parts = [f"{label:<14s}", f"{bd['dest_grid_id']:<10s}"]
    for key, _ in _COMPONENTS:
        parts.append(f"{bd[key]:+8.3f}")
    parts.append(f" = {bd['total_utility']:+8.3f}")
    return "  ".join(parts)


def explain_to_text(explanation: Dict[str, Any]) -> str:
    """Format the explanation as a plain-text utility breakdown table."""
    lines: List[str] = []
    lines.append("=" * 110)
    lines.append("DECISION EXPLANATION")
    lines.append("=" * 110)
    lines.append(explanation["agent_summary"])
    meta = explanation["meta"]
    lines.append(
        f"Utility class: {meta['utility_class']}    Hour: {meta['hour']}:00    "
        f"Candidates: {meta['n_candidates']}"
    )
    lines.append("")

    chosen_id, chosen_bd = explanation["chosen"]
    header = "  ".join(
        [f"{'role':<14s}", f"{'dest':<10s}"]
        + [f"{label[:8]:>8s}" for _, label in _COMPONENTS]
        + [f"   {'total':>8s}"]
    )
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(_fmt_breakdown_row(chosen_bd, f"CHOSEN(r{chosen_bd['rank']})"))

    for i, alt in enumerate(explanation["alternatives"], 1):
        lines.append(_fmt_breakdown_row(alt, f"alt#{i}(r{alt['rank']})"))

    lines.append("")
    lines.append("Raw destination quantities (CHOSEN):")
    lines.append(
        f"  V_j(t) = {chosen_bd['V_j_raw']:.3f}   OccMatch = {chosen_bd['OccMatch_raw']:.3f}   "
        f"travel_time = {chosen_bd['travel_time_min']:.1f} min   distance = {chosen_bd['distance_km']:.2f} km"
    )
    lines.append(
        f"  Coeffs: alpha_V={chosen_bd['alpha_V']:.3f}, alpha_occ={chosen_bd['alpha_occ']:.3f}, "
        f"beta={chosen_bd['beta']:.4f}, gamma={chosen_bd['gamma']:.3f}"
    )
    lines.append("")
    lines.append("Narrative:")
    lines.append("  " + explanation["narrative"])
    lines.append("=" * 110)
    return "\n".join(lines)


def explain_to_html(explanation: Dict[str, Any]) -> str:
    """Format the explanation as an HTML snippet with CSS classes for styling."""
    chosen_id, chosen_bd = explanation["chosen"]
    meta = explanation["meta"]

    def _row(bd: Dict[str, Any], role: str, css: str) -> str:
        cells = [f"<td class='dx-role'>{role}</td>",
                 f"<td class='dx-dest'>{bd['dest_grid_id']}</td>"]
        for key, _ in _COMPONENTS:
            cls = "dx-pos" if bd[key] >= 0 else "dx-neg"
            cells.append(f"<td class='dx-cell {cls}'>{bd[key]:+.3f}</td>")
        cells.append(f"<td class='dx-total'>{bd['total_utility']:+.3f}</td>")
        cells.append(f"<td class='dx-rank'>#{bd['rank']}</td>")
        return f"<tr class='{css}'>" + "".join(cells) + "</tr>"

    head = (
        "<tr>"
        + "<th>Role</th><th>Dest</th>"
        + "".join(f"<th>{label}</th>" for _, label in _COMPONENTS)
        + "<th>Total U</th><th>Rank</th>"
        + "</tr>"
    )

    rows = [_row(chosen_bd, "CHOSEN", "dx-chosen")]
    for i, alt in enumerate(explanation["alternatives"], 1):
        rows.append(_row(alt, f"alt#{i}", "dx-alt"))

    raw_line = (
        f"V<sub>j</sub>(t) = {chosen_bd['V_j_raw']:.3f} &nbsp;|&nbsp; "
        f"OccMatch = {chosen_bd['OccMatch_raw']:.3f} &nbsp;|&nbsp; "
        f"travel = {chosen_bd['travel_time_min']:.1f} min &nbsp;|&nbsp; "
        f"distance = {chosen_bd['distance_km']:.2f} km"
    )
    coeffs_line = (
        f"&alpha;<sub>V</sub>={chosen_bd['alpha_V']:.3f}, "
        f"&alpha;<sub>occ</sub>={chosen_bd['alpha_occ']:.3f}, "
        f"&beta;={chosen_bd['beta']:.4f}, "
        f"&gamma;={chosen_bd['gamma']:.3f}"
    )

    html = f"""
<div class='decision-explainer'>
  <div class='dx-summary'>{explanation['agent_summary']}</div>
  <div class='dx-meta'>Utility: {meta['utility_class']} &nbsp;|&nbsp; Hour: {meta['hour']}:00 &nbsp;|&nbsp; Candidates: {meta['n_candidates']}</div>
  <table class='dx-table'>
    <thead>{head}</thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <div class='dx-raw'>Raw (chosen): {raw_line}</div>
  <div class='dx-coeffs'>Coefficients: {coeffs_line}</div>
  <div class='dx-narrative'><strong>Why?</strong> {explanation['narrative']}</div>
</div>
""".strip()
    return html


# ---------------------------------------------------------------------------
# Self-test entry point
# ---------------------------------------------------------------------------

def _self_test(n_agents: int = 200, n_iters: int = 1, n_explain: int = 5):
    """Build a tiny LondonV2Model from existing v2.1 artefacts and explain N agents."""
    import pandas as pd
    from models_lib.linear_utility import LinearUtility
    from model import LondonV2Model

    print(f"V2_ROOT = {V2_ROOT}")
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    aux = dict(np.load(V2_ROOT / "data" / "processed" / "training_aux.npz", allow_pickle=True))
    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    agents_df = agents_df.sample(n_agents, random_state=42).reset_index(drop=True)

    rng = np.random.default_rng(42)
    nts = pd.read_csv(V2_ROOT / "data" / "processed" / "nts_commute_departure_time.csv", comment="#")
    pi_t = nts[nts["mode"] == "all"].sort_values("hour")["share"].values
    pi_t = pi_t / pi_t.sum()
    agents_df["departure_hour"] = rng.choice(24, size=len(agents_df), p=pi_t)

    util = LinearUtility()
    util.load_state_dict(torch.load(V2_ROOT / "linear_utility_best.pt", map_location="cpu"))
    util.eval()

    norm_stats = {
        "V_jt_mean": aux["V_jt_mean"], "V_jt_std": aux["V_jt_std"],
        "t_log_mean": aux["t_log_mean"], "t_log_std": aux["t_log_std"],
        "d_log_mean": aux["d_log_mean"], "d_log_std": aux["d_log_std"],
        "om_mean": aux["om_mean"], "om_std": aux["om_std"],
    }

    print(f"Building model with {n_agents} agents...")
    model = LondonV2Model(
        agents_df=agents_df,
        grid_features_df=grid_feats,
        V_jt_baseline=cache["V_jt_baseline"],
        t_ij_t_initial=cache["t_ij_t"],
        log_d_ij=aux["log_d_ij"],
        linear_utility_model=util,
        norm_stats=norm_stats,
    )
    for it in range(n_iters):
        model.step(recompute_congestion=(it < n_iters - 1))
    print(f"Stepped model {n_iters}x. Now explaining {n_explain} agents...\n")

    ids = [int(a.unique_id_) for a in list(model.agents)[:n_explain]]
    for aid in ids:
        ex = explain_choice(model, aid, top_k=4)
        print(explain_to_text(ex))
        print()


if __name__ == "__main__":
    _self_test()
