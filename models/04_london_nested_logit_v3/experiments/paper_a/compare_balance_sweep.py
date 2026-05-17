"""Compare IV-balance sweep results: CPC trade-off vs attribute share variance."""
from __future__ import annotations
import io
import json
import sys
import statistics
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V3_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = V3_ROOT / "evaluation_outputs" / "paper_a"

CASES = [
    ("v3l_ortho5_decomposed_seed0",   "baseline (no balance)"),
    ("v3l_balance_iv070_seed0",       "balance → IV 70%"),
    ("v3l_balance_iv050_seed0",       "balance → IV 50%"),
    ("v3l_balance_iv030_seed0",       "balance → IV 30%"),
]


def share_var(shares: dict) -> float:
    """How balanced are the attribute shares (lower variance = more balanced)?"""
    vals = list(shares.values())
    if len(vals) < 2:
        return 0.0
    return statistics.variance(vals)


def main():
    rows = []
    for fname, label in CASES:
        p = OUT_DIR / f"{fname}.json"
        if not p.exists():
            print(f"  [skip] {fname}.json not found")
            continue
        d = json.load(open(p))
        fd = d["final_diagnostic"]
        ad = fd.get("ablation_drop", {})
        asj = fd.get("attribute_shares_across_j", {})
        rows.append({
            "label": label,
            "cpc": d["final_cpc"],
            "iv_share_across_j": asj.get("mode_choice_IV", float("nan")),
            "abl_iv": ad.get("IV_time_cost_beta_t", float("nan")),
            "abl_gravity": ad.get("gravity_gamma_logM", float("nan")),
            "abl_wage": ad.get("wage_alpha_logW", float("nan")),
            "abl_comp": ad.get("competition_nu_logD", float("nan")),
            "abl_match": ad.get("occmatch_delta_M", float("nan")),
            "abl_self_loop": ad.get("self_loop_boost", float("nan")),
            "abl_nn": ad.get("nn_residual", float("nan")),
            "shares": asj,
            "ablation_drops": ad,
        })

    if not rows:
        print("No JSONs found yet — wait for sweep to finish")
        return

    print("\n=== CPC vs balance target ===")
    print(f"  {'Method':<24s}  {'CPC':>7s}  {'IV share':>9s}  {'share var':>10s}")
    for r in rows:
        # Compute share variance of destination attributes (excl IV) — lower = more balanced
        dest_shares = {k: v for k, v in r["shares"].items()
                       if k in ("gravity_gamma_logM","wage_alpha_logW",
                                "competition_nu_logD","occmatch_delta_M",
                                "nn_residual","self_loop_boost")}
        sv = share_var(dest_shares)
        print(f"  {r['label']:<24s}  {r['cpc']:>7.4f}  {r['iv_share_across_j']*100:>7.1f}%  {sv*1e6:>9.2f}e-6")

    print("\n=== Ablation CPC drops (per attribute) ===")
    print(f"  {'Method':<24s}  {'time β·t':>9s}  {'gravity':>8s}  {'comp':>7s}  {'match':>7s}  {'self_loop':>9s}  {'NN':>6s}")
    for r in rows:
        print(f"  {r['label']:<24s}  {r['abl_iv']:>+9.4f}  {r['abl_gravity']:>+8.4f}  "
              f"{r['abl_comp']:>+7.4f}  {r['abl_match']:>+7.4f}  "
              f"{r['abl_self_loop']:>+9.4f}  {r['abl_nn']:>+6.4f}")

    print("\n=== Across-j shares (paper-grade balanced importance) ===")
    keys = sorted(set().union(*[set(r["shares"].keys()) for r in rows]),
                  key=lambda k: -max(r["shares"].get(k, 0) for r in rows))
    print(f"  {'Attribute':<24s}" + "".join(f"{r['label'][:20]:>22s}" for r in rows))
    for k in keys:
        line = f"  {k:<24s}"
        for r in rows:
            v = r["shares"].get(k, 0)
            line += f"{v*100:>20.2f}%"
        print(line)


if __name__ == "__main__":
    main()
