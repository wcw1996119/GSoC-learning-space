"""Phase B per-tier VOT vs DfT WebTAG TAG A1.3 external-validity check.

Reads the per-tier mean VOT (low/mid/high) recovered by
experiments/paper_a/gnn_ablation_B.py from gnn_ablation_B_summary.json,
compares against TAG A1.3 reference values, and writes a JSON report.

TAG A1.3 reference values used here (toy validation thresholds):
    Low:  8.0  GBP/h
    Mid:  13.0 GBP/h
    High: 22.0 GBP/h

Pass criterion (per tier): relative_error = |VOT_recovered - VOT_TAG| / VOT_TAG < 0.5

Output
------
  evaluation_outputs/paper_a/vot_tag_B.json

CLI
---
  python vot_tag_check_B.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

OUT_DIR = V2_ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "gnn_ablation_B_summary.json"
OUT_PATH = OUT_DIR / "vot_tag_B.json"

TAG_LOW = 8.0
TAG_MID = 13.0
TAG_HIGH = 22.0
PASS_THRESHOLD = 0.5   # 50 % relative error - toy data is loose


def rel_err(recovered: float, reference: float) -> float:
    if reference == 0 or recovered != recovered:   # NaN guard
        return float("nan")
    return abs(recovered - reference) / reference


def main() -> int:
    if not SUMMARY_PATH.exists():
        print(f"[vot_tag_check_B] FAIL: missing {SUMMARY_PATH}")
        print("  Run experiments/paper_a/gnn_ablation_B.py first.")
        return 1

    summary = json.loads(SUMMARY_PATH.read_text())
    vot_low = float(summary.get("phase_B_vot_low_mean", float("nan")))
    vot_mid = float(summary.get("phase_B_vot_mid_mean", float("nan")))
    vot_high = float(summary.get("phase_B_vot_high_mean", float("nan")))

    tiers = [
        ("low", vot_low, TAG_LOW),
        ("mid", vot_mid, TAG_MID),
        ("high", vot_high, TAG_HIGH),
    ]

    results: list[dict] = []
    for name, recovered, reference in tiers:
        err = rel_err(recovered, reference)
        passes = (err == err) and (err < PASS_THRESHOLD)   # NaN -> not pass
        results.append({
            "tier": name,
            "vot_recovered_gbp_per_h": recovered,
            "vot_tag_gbp_per_h": reference,
            "relative_error": err,
            "pass_threshold": PASS_THRESHOLD,
            "passes": bool(passes),
        })

    n_pass = sum(1 for r in results if r["passes"])
    overall_pass = bool(n_pass == len(results))

    out = {
        "tag_reference": {"low": TAG_LOW, "mid": TAG_MID, "high": TAG_HIGH},
        "pass_threshold_relative": PASS_THRESHOLD,
        "tiers": results,
        "n_tiers_passed": n_pass,
        "n_tiers_total": len(results),
        "overall_pass": overall_pass,
        "source_summary": str(SUMMARY_PATH),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------------
    # Stdout report (ASCII only - GBP not the symbol).
    # ------------------------------------------------------------------
    print("[vot_tag_check_B] WebTAG TAG A1.3 per-tier comparison")
    print(f"  source: {SUMMARY_PATH}")
    print(f"  threshold: relative_error < {PASS_THRESHOLD}")
    print("")
    print(f"  {'tier':<6} {'recovered':>12} {'TAG':>8} {'rel_err':>10}  {'verdict':>8}")
    for r in results:
        verdict = "[OK]" if r["passes"] else "[FAIL]"
        rec_str = f"{r['vot_recovered_gbp_per_h']:.2f}" \
            if r['vot_recovered_gbp_per_h'] == r['vot_recovered_gbp_per_h'] else "NaN"
        err_str = f"{r['relative_error']:.3f}" \
            if r['relative_error'] == r['relative_error'] else "NaN"
        print(f"  {r['tier']:<6} {rec_str:>12} {r['vot_tag_gbp_per_h']:>8.2f} "
              f"{err_str:>10}  {verdict:>8}")

    print("")
    print(f"[vot_tag_check_B] passed {n_pass}/{len(results)} tiers; "
          f"overall_pass={overall_pass}")
    print(f"[vot_tag_check_B] wrote {OUT_PATH}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
