"""Analyze hierarchical pooling shrinkage on reliability_scan results.

Reads reliability_scan.csv (raw cell-level cluster × magnitude experiments)
and applies hierarchical pooling: shrink aggregate prediction toward MC
prediction with weight alpha = baseline_emp / (baseline_emp + k).

Reports for several k values:
  - Reliable/caveat/unreliable fraction (with pooling vs without)
  - Mean bias by (cluster baseline × magnitude) bin
  - Best k recommendation

Mathematically, with pooled = alpha*agg + (1-alpha)*mc:
    bias_pooled = |pooled - mc| / |mc|
                = alpha * |agg - mc| / |mc|
                = alpha * bias_original

So small-cluster bias shrinks toward 0 (alpha → 0), large-cluster bias
preserved (alpha → 1).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "evaluation_outputs" / "paper_a" / "reliability_scan.csv"


def trust_region_stats(bias_col: pd.Series) -> dict:
    """Compute reliable / caveat / unreliable share."""
    n = len(bias_col)
    n_rel = (bias_col < 5).sum()
    n_cav = ((bias_col >= 5) & (bias_col < 20)).sum()
    n_unr = (bias_col >= 20).sum()
    return {
        "reliable_pct": 100 * n_rel / n,
        "caveat_pct": 100 * n_cav / n,
        "unreliable_pct": 100 * n_unr / n,
        "n_total": n,
    }


def bin_means(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["emp_bin"] = pd.cut(df["baseline_emp"],
                            bins=[0, 5e3, 2e4, 5e4, 2e5, 1e7],
                            labels=["<5k", "5k-20k", "20k-50k", "50k-200k", ">200k"])
    df["mag_bin"] = pd.cut(df["magnitude_multiplier"],
                            bins=[0, 0.20, 0.50, 1.50, 10],
                            labels=["+10-20%", "+30-50%", "+100-150%", "+200%+"])
    return df.pivot_table(index="emp_bin", columns="mag_bin",
                            values=value_col, aggfunc="mean", observed=True).round(1)


def main():
    df = pd.read_csv(CSV)
    print(f"loaded {len(df)} cells\n")

    # Original baseline (no pooling)
    base = trust_region_stats(df["abs_jensen_bias_pct"])
    print("=" * 70)
    print(f"BASELINE (no pooling) — k = inf, alpha = 1 always")
    print(f"  reliable    : {base['reliable_pct']:>5.1f}%")
    print(f"  caveat      : {base['caveat_pct']:>5.1f}%")
    print(f"  unreliable  : {base['unreliable_pct']:>5.1f}%")
    print(f"  reliable + caveat: {base['reliable_pct'] + base['caveat_pct']:>5.1f}%")
    print()
    print("mean |bias|% by bin:")
    print(bin_means(df, "abs_jensen_bias_pct"))

    # Sweep k
    print("\n" + "=" * 70)
    print(f"HIERARCHICAL POOLING sweep")
    print(f"  alpha = baseline_emp / (baseline_emp + k)")
    print(f"  bias_pooled = alpha * bias_raw")
    print()

    results = []
    for k in [1_000, 5_000, 10_000, 20_000, 50_000]:
        df_pooled = df.copy()
        df_pooled["alpha"] = df_pooled["baseline_emp"] / (df_pooled["baseline_emp"] + k)
        df_pooled["abs_pooled_bias_pct"] = df_pooled["alpha"] * df_pooled["abs_jensen_bias_pct"]

        stats = trust_region_stats(df_pooled["abs_pooled_bias_pct"])
        results.append({
            "k": k,
            "reliable_pct": stats['reliable_pct'],
            "caveat_pct": stats['caveat_pct'],
            "unreliable_pct": stats['unreliable_pct'],
            "rel_plus_caveat": stats['reliable_pct'] + stats['caveat_pct'],
        })

        print(f"k = {k:>6,}  shrinkage when baseline = 5k: alpha = {5000/(5000+k):.2f}")
        print(f"  reliable    : {stats['reliable_pct']:>5.1f}%  ({stats['reliable_pct']-base['reliable_pct']:+.1f})")
        print(f"  caveat      : {stats['caveat_pct']:>5.1f}%")
        print(f"  unreliable  : {stats['unreliable_pct']:>5.1f}%  ({stats['unreliable_pct']-base['unreliable_pct']:+.1f})")
        print(f"  rel + caveat: {stats['reliable_pct']+stats['caveat_pct']:>5.1f}%")
        print(f"  bin-mean |bias|%:")
        print(bin_means(df_pooled, "abs_pooled_bias_pct").to_string())
        print()

    # Best k by reliable + caveat
    best = max(results, key=lambda r: r["rel_plus_caveat"])
    print("=" * 70)
    print(f"BEST: k = {best['k']:,}")
    print(f"  reliable    : {best['reliable_pct']:>5.1f}%")
    print(f"  caveat      : {best['caveat_pct']:>5.1f}%")
    print(f"  unreliable  : {best['unreliable_pct']:>5.1f}%")
    print(f"  reliable+caveat: {best['rel_plus_caveat']:>5.1f}%   "
          f"(baseline: {base['reliable_pct']+base['caveat_pct']:.1f}%)")

    # Save final pooled CSV with best k
    df_final = df.copy()
    df_final["alpha"] = df_final["baseline_emp"] / (df_final["baseline_emp"] + best["k"])
    df_final["delta_pooled_inflow"] = (
        df_final["alpha"] * df_final["delta_agg_inflow"]
        + (1 - df_final["alpha"]) * df_final["delta_mc_inflow"]
    )
    df_final["abs_pooled_bias_pct"] = (
        df_final["alpha"] * df_final["abs_jensen_bias_pct"]
    )
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "reliability_scan_pooled.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
