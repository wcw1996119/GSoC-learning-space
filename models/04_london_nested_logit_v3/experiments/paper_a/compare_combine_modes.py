"""Compare CPC + attribute shares across RUM/GNN combination modes.

Reads JSON results from experiments and produces:
- CPC table (final, mean across stages if backfit)
- Attribute share table (destination-attractor only, excl. IV which dominates by scale)
- Real-world interpretation notes
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path
from collections import OrderedDict

# Force UTF-8 stdout on Windows GBK terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V3_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = V3_ROOT / "evaluation_outputs" / "paper_a"


def load(name: str) -> dict | None:
    p = OUT_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def dest_share(rms: dict) -> dict:
    """Recompute destination-attractor share from RMS dict (excluding mode_choice_IV)."""
    dest = {k: v for k, v in rms.items() if k != "mode_choice_IV"}
    total_var = sum(v ** 2 for v in dest.values()) + 1e-9
    return {k: (v ** 2) / total_var for k, v in dest.items()}


MODES = OrderedDict([
    ("v3l_selfloop_seed0",          "Joint residual 500ep"),
    ("v3l_boosting_4stage_v2_seed0","Backfitting 400ep"),
    ("v3l_mult_500ep_seed0",        "Multiplicative 500ep"),
    ("v3l_ortho5_500ep_seed0",      "Joint+ortho λ=5  500ep"),
    ("v3l_ortho20_500ep_seed0",     "Joint+ortho λ=20 500ep"),
    ("v3l_moe_500ep_seed0",         "MoE per-origin 500ep"),
])

INTERPRETATIONS = {
    "gravity_gamma_logM":   "γ·log(M_j) — Hansen 1959 gravity attractor: bigger destination → more attractive (jobs concentration)",
    "wage_alpha_logW":      "α·log(W_j) — wage-driven attraction: higher-wage destinations attract more (Schwanen 2003 income×destination)",
    "competition_nu_logD":  "ν·log(D_j) — Shen 1998 competition: more competitors at destination → less attractive (job market crowding)",
    "occmatch_delta_M":     "δ·match·log(M_j) — Cervero 1999 OccMatch: destinations matching origin's worker profile get amplified gravity",
    "self_loop_boost":      "Per-(borough, hour) intra-zone bias: CBD lunch-pulse — workers stay in own borough at midday",
    "busy_dest_boost":      "Top-K busy-destination hour boost: CBD-specific attraction pattern (e.g. grid 816 vs 770 within same borough)",
    "push_pull":            "ξ·push_i·log(M_j) — labor surplus at origin amplifies preference for large-jobs destinations (Wilson 1967)",
    "nn_residual":          "GNN-learned residual on logits: any spatial/temporal pattern RUM doesn't structurally encode",
}


def main():
    rows = []
    for fname, label in MODES.items():
        d = load(f"{fname}.json")
        if d is None:
            print(f"  [skip] {fname}.json not found")
            continue
        fd = d["final_diagnostic"]
        rms = fd.get("attribute_rms", {})
        dest_sh = fd.get("attribute_shares_dest") or (dest_share(rms) if rms else {})
        rows.append({
            "label": label,
            "fname": fname,
            "cpc": d["final_cpc"],
            "epochs": d.get("epochs_actual", "?"),
            "fit_s": d.get("fit_time_s", "?"),
            "delta_meas": fd.get("delta_measured", float("nan")),
            "lam_b": fd.get("lambda_b_mean", float("nan")),
            "ortho_cos_sq": fd.get("ortho_cos_sq", float("nan")),
            "dest_share": dest_sh,
            "rms": rms,
            "gnn_mode": fd.get("gnn_mode", "?"),
            "w_NN": fd.get("gnn_residual_scale"),
            "gamma_mult": fd.get("gnn_mult_scale"),
            "gate_mean": fd.get("gate_per_origin_mean"),
            "gate_std": fd.get("gate_per_origin_std"),
            "alpha_wage": fd.get("alpha_wage"),
            "gamma_M": fd.get("gamma_M"),
            "nu_D": fd.get("nu_D"),
            "delta_match": fd.get("delta_match"),
        })

    if not rows:
        print("No JSONs found — runs not finished yet?")
        return

    # ---- CPC table ----
    print("\n=== CPC comparison (seed 0, all modes) ===")
    print(f"  {'Method':<28s}  {'CPC':>7s}  {'δ_meas':>7s}  {'λ_b':>6s}  {'ortho cos2':>10s}")
    print(f"  {'-'*28}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*10}")
    for r in rows:
        print(f"  {r['label']:<28s}  {r['cpc']:>7.4f}  {r['delta_meas']:>7.4f}  "
              f"{r['lam_b']:>6.3f}  {r['ortho_cos_sq']:>10.4f}")

    print("\n=== Method-specific param ===")
    for r in rows:
        info = []
        if r["w_NN"] is not None:    info.append(f"w_NN={r['w_NN']:.3f}")
        if r["gamma_mult"] is not None: info.append(f"γ_mult={r['gamma_mult']:.3f}")
        if r["gate_mean"] is not None: info.append(f"gate_mean={r['gate_mean']:.3f} std={r['gate_std']:.3f}")
        print(f"  {r['label']:<28s}  {' '.join(info)}")

    # ---- Attribute share table ----
    print("\n=== Destination-attractor variance shares (excl. mode_choice_IV) ===")
    all_keys = sorted(set().union(*[set(r["dest_share"].keys()) for r in rows]),
                      key=lambda k: -max(r["dest_share"].get(k, 0) for r in rows))
    header = "  " + f"{'Attribute':<24s}" + "".join(f"{r['label'][:18]:>20s}" for r in rows)
    print(header)
    print("  " + "-" * (24 + 20 * len(rows)))
    for k in all_keys:
        line = f"  {k:<24s}"
        for r in rows:
            v = r["dest_share"].get(k, 0)
            line += f"{v*100:>18.1f}%"
        print(line)

    # ---- Interpretation block ----
    print("\n=== Attribute interpretations ===")
    for k in all_keys:
        if k in INTERPRETATIONS:
            print(f"  {k}: {INTERPRETATIONS[k]}")

    # ---- RUM coefficient table (avg across tiers) ----
    def _mean(v):
        if v is None: return float("nan")
        if isinstance(v, list): return sum(v) / len(v)
        return float(v)
    print("\n=== RUM coefficients (mean across income tiers) ===")
    print(f"  {'Method':<28s}  {'α_w':>6s}  {'γ_M':>6s}  {'ν_D':>6s}  {'δ_m':>6s}")
    for r in rows:
        print(f"  {r['label']:<28s}  "
              f"{_mean(r['alpha_wage']):>+6.3f}  "
              f"{_mean(r['gamma_M']):>+6.3f}  "
              f"{_mean(r['nu_D']):>+6.3f}  "
              f"{_mean(r['delta_match']):>+6.3f}")


if __name__ == "__main__":
    main()
