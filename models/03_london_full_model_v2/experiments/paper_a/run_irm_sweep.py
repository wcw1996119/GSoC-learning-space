"""IRM λ sweep — Module 1 (Paper A, 2026-05-13).

Sweeps 2 variants × 5 lambdas × 3 seeds × 200 epochs on Config 0 (orig22
baseline DUAL_HET). Tags output files with variant + λ so they don't clobber.

Expected T4 cost: ~30 runs × 200s ≈ 1.7 hr.

After sweep, run with --summarize to compile a leaderboard table.

Usage::

    # Full sweep (Colab T4)
    python experiments/paper_a/run_irm_sweep.py --device cuda

    # Single variant quick test (mesa-demo CPU)
    python experiments/paper_a/run_irm_sweep.py --variants rex \\
        --lambdas 1 10 100 --seeds 0 --epochs 50 --device cpu

    # Summarize after sweep
    python experiments/paper_a/run_irm_sweep.py --summarize \\
        --out_dir evaluation_outputs/paper_a/irm_sweep
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "experiments" / "paper_a" / "run_dual_het_ablation.py"

# V-REx and IRM-v1 penalties live on different magnitude scales — calibrated
# in smoke_test_irm.py: V-REx ≈ 65, IRM-v1 ≈ 1.8e+4 (≈ 270×). Use different
# sweeps so each variant gets penalty contribution in the same ratio band.
DEFAULT_LAMBDAS_REX = [0.1, 1.0, 10.0, 100.0, 1000.0]
DEFAULT_LAMBDAS_IRMV1 = [0.001, 0.01, 0.1, 1.0, 10.0]
# IRM-v1 disabled by default: 33-borough × second-order autograd (`create_graph
# True` per env) OOM-ed on Colab T4 (14 GiB) in 2026-05-13 sweep — every IRM-v1
# config crashed allocating 274 MiB for the per-env Hessian-vector products.
# V-REx is the Krueger 2021 recommended variant anyway. To re-enable IRM-v1,
# pass `--variants irmv1 rex` AND switch to a smaller env partition (e.g.,
# 5-region super-borough) or use sub-batched env sampling.
DEFAULT_VARIANTS = ["rex"]
DEFAULT_SEEDS = [0, 1, 2]


def lambdas_for(variant: str, override: list | None):
    """Pick the calibrated λ range per variant unless caller overrides."""
    if override is not None:
        return override
    return DEFAULT_LAMBDAS_REX if variant == "rex" else DEFAULT_LAMBDAS_IRMV1


def run_sweep(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(lambdas_for(v, args.lambdas)) for v in args.variants)
    i = 0
    for variant in args.variants:
        for lam in lambdas_for(variant, args.lambdas):
            i += 1
            print(f"\n{'='*80}")
            print(f"[{i}/{total}] IRM {variant}  λ={lam}  seeds={args.seeds}  "
                  f"epochs={args.epochs}  warmup={args.warmup}")
            print('=' * 80)
            cmd = [
                sys.executable, str(ABLATION),
                "--config", args.config,
                "--seeds", *map(str, args.seeds),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--cache_name", args.cache_name,
                "--device", args.device,
                "--out_dir", str(out_dir),
                "--irm_variant", variant,
                "--irm_lambda", str(lam),
                "--irm_warmup_epochs", str(args.warmup),
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED variant={variant} λ={lam}: {e}")
                continue


def summarize(args):
    """Read all ablation_*.json under out_dir, build CPC leaderboard."""
    out_dir = Path(args.out_dir)
    rows = []
    for fp in sorted(out_dir.glob("ablation_*.json")):
        with open(fp) as f:
            r = json.load(f)
        rows.append({
            "file": fp.name,
            "config": r.get("config"),
            "irm_variant": r.get("irm_variant", "none"),
            "irm_lambda": r.get("irm_lambda", 0.0),
            "warmup": r.get("irm_warmup_epochs", 0),
            "n_seeds": r.get("n_seeds"),
            "cpc_mean": r.get("cpc_mean"),
            "cpc_std": r.get("cpc_std"),
            "cache": r.get("cache_name"),
        })
    rows.sort(key=lambda x: (x["irm_variant"], x["irm_lambda"]))

    print(f"\n{'variant':<10}{'λ':>10}{'warmup':>10}"
          f"{'CPC mean':>12}{'std':>10}{'n':>4}{'cache':>26}")
    print("-" * 82)
    for r in rows:
        print(f"{r['irm_variant']:<10}{r['irm_lambda']:>10.4g}{r['warmup']:>10}"
              f"{r['cpc_mean']:>12.4f}{r['cpc_std']:>10.4f}"
              f"{r['n_seeds']:>4}{(r['cache'] or 'n/a'):>26}")

    # Best per variant
    print("\nBest λ per variant (by CPC mean):")
    by_variant = {}
    for r in rows:
        v = r["irm_variant"]
        if v == "none" or r["irm_lambda"] == 0:
            continue
        if v not in by_variant or r["cpc_mean"] > by_variant[v]["cpc_mean"]:
            by_variant[v] = r
    for v, r in by_variant.items():
        print(f"  {v}: λ={r['irm_lambda']:g}  CPC={r['cpc_mean']:.4f} ± {r['cpc_std']:.4f}")

    # Compare to ERM baseline (if present in same dir or hardcoded reference)
    erm_baselines = [r for r in rows if r["irm_lambda"] == 0 or r["irm_variant"] == "none"]
    if erm_baselines:
        print("\nERM baselines (λ=0):")
        for r in erm_baselines:
            print(f"  config={r['config']}  CPC={r['cpc_mean']:.4f} ± {r['cpc_std']:.4f}")
    else:
        print("\n(no ERM baseline in this dir; reference: Phase 1a orig22 = 0.4833 ± 0.0038)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", action="store_true",
                   help="Skip running; just read JSONs and compile leaderboard.")
    p.add_argument("--config", type=str, default="baseline",
                   help="Base config (default baseline DUAL_HET).")
    p.add_argument("--cache_name", type=str, default="demo_cache.npz",
                   help="orig22 by default (Config 0 baseline pick from 2026-05-13).")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   choices=["rex", "irmv1"])
    p.add_argument("--lambdas", nargs="+", type=float, default=None,
                   help="Override λ list. Default uses calibrated ranges: "
                   "V-REx [0.1..1000], IRM-v1 [0.001..10].")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--warmup", type=int, default=50,
                   help="Warm-up epochs at λ=0 before penalty kicks in.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=str,
                   default=str(ROOT / "evaluation_outputs" / "paper_a" / "irm_sweep"))
    args = p.parse_args()

    if args.summarize:
        summarize(args)
    else:
        run_sweep(args)
        print(f"\nAll runs done. Summarize with:\n  "
              f"python {Path(__file__).name} --summarize --out_dir {args.out_dir}")


if __name__ == "__main__":
    main()
