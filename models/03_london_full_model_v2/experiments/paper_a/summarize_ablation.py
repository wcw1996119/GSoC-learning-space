"""Summarise DUAL_HET frozen-component ablation: A baseline vs B frozen_gnn vs C frozen_rum.

Reads:
  evaluation_outputs/paper_a/distance_aware_results.json   (baseline A, 200ep, 3 seeds)
  evaluation_outputs/paper_a/ablation_frozen_gnn.json       (config B)
  evaluation_outputs/paper_a/ablation_frozen_rum.json       (config C)

Optional:
  evaluation_outputs/paper_a/ablation_baseline.json         (CPU baseline at matched epochs)

Writes:
  evaluation_outputs/paper_a/ablation_summary.json
  evaluation_outputs/paper_a/ablation_summary.png
  evaluation_outputs/paper_a/ablation_summary.md

Judgement (per memory project_session_2026_05_11.md):
  - B CPC ≈ 0.20–0.30 + C drops too → RUM head carries signal → counterfactual OK
  - B CPC ≈ baseline (~0.4+)         → GNN ate everything → counterfactual fails (Mozolin warning)
  - B middle (~0.35)                 → both contribute, GNN dominant → need δ control
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER_A = ROOT / "evaluation_outputs" / "paper_a"


def load_baseline_colab():
    """Read DUAL_HET aggregate (3 seeds) from distance_aware_results.json."""
    path = PAPER_A / "distance_aware_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]["DUAL_HET"]
    per_seed = data["per_seed"]["DUAL_HET"]
    return {
        "cpc_mean": s["cpc"]["mean"],
        "cpc_std": s["cpc"]["std"],
        "cpc_all": s["cpc"]["all"],
        "epochs": data["config"]["epochs"],
        "n_seeds": len(per_seed),
        "device": data["config"].get("device", "cuda"),
        "source": "distance_aware_results.json",
    }


def load_ablation(config):
    path = PAPER_A / f"ablation_{config}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def judge(cpc_baseline, cpc_frozen_gnn, cpc_frozen_rum):
    """Return verdict string based on ablation pattern."""
    # Reasoning gates (with some slack for noise):
    gap_b_vs_baseline = cpc_baseline - cpc_frozen_gnn       # how much B drops vs A
    gap_c_vs_baseline = cpc_baseline - cpc_frozen_rum       # how much C drops vs A

    if cpc_frozen_gnn > cpc_baseline - 0.05:
        # B barely drops → GNN ate everything
        return ("FAIL", "B (frozen_gnn) ≈ A baseline → GNN carries signal alone. "
                "Mozolin 2000 warning复发：NN cannot extrapolate beyond training. "
                "Counterfactual stance is fragile.")
    if cpc_frozen_gnn < cpc_baseline - 0.15:
        if cpc_frozen_rum < cpc_baseline - 0.05:
            return ("PASS", "B (frozen_gnn) dropped substantially AND C (frozen_rum) "
                    "also dropped → both components carry signal, RUM head "
                    "non-trivially. Counterfactual stance is defensible.")
        return ("PASS_RUM_ONLY", "B drops a lot but C barely drops → "
                "RUM head carries most signal, GNN nearly free. "
                "Counterfactual based on RUM is safe.")
    return ("MARGINAL", f"B drops only {gap_b_vs_baseline:.3f} below baseline "
            f"(GNN partly carrying signal). Consider adding δ-control "
            "(Wang TB-ResNet style) to enforce (1-δ)V_RUM + δV_GNN.")


def main():
    baseline_colab = load_baseline_colab()
    gnn_ab = load_ablation("frozen_gnn")
    rum_ab = load_ablation("frozen_rum")
    baseline_cpu = load_ablation("baseline")  # optional, CPU-matched

    if gnn_ab is None or rum_ab is None:
        print("[summarize] frozen_gnn or frozen_rum JSON missing — run ablation first.")
        if gnn_ab is None:
            print("  missing: ablation_frozen_gnn.json")
        if rum_ab is None:
            print("  missing: ablation_frozen_rum.json")
        sys.exit(1)

    cpc_baseline = baseline_colab["cpc_mean"] if baseline_colab else None
    cpc_baseline_cpu = baseline_cpu["cpc_mean"] if baseline_cpu else None
    cpc_b = gnn_ab["cpc_mean"]
    cpc_c = rum_ab["cpc_mean"]

    # Prefer CPU-matched baseline for the verdict if available; else fall back to Colab.
    cpc_a_for_judge = cpc_baseline_cpu if cpc_baseline_cpu is not None else cpc_baseline

    verdict, reasoning = judge(cpc_a_for_judge, cpc_b, cpc_c)

    summary = {
        "verdict": verdict,
        "reasoning": reasoning,
        "configs": {
            "baseline_colab_200ep": baseline_colab,
            "baseline_cpu_matched": baseline_cpu,
            "frozen_gnn": {
                "cpc_mean": gnn_ab["cpc_mean"],
                "cpc_std": gnn_ab.get("cpc_std", 0.0),
                "epochs": gnn_ab["epochs"],
                "n_seeds": gnn_ab["n_seeds"],
                "beta_recovered": [
                    (r["beta_car"], r["beta_transit"], r["beta_walk"])
                    for r in gnn_ab["results"]
                ],
                "gamma_recovered": [r["gamma"] for r in gnn_ab["results"]],
            },
            "frozen_rum": {
                "cpc_mean": rum_ab["cpc_mean"],
                "cpc_std": rum_ab.get("cpc_std", 0.0),
                "epochs": rum_ab["epochs"],
                "n_seeds": rum_ab["n_seeds"],
                "beta_fixed": [
                    (r["beta_car"], r["beta_transit"], r["beta_walk"])
                    for r in rum_ab["results"]
                ],
                "gamma_fixed": [r["gamma"] for r in rum_ab["results"]],
            },
        },
    }

    summary_path = PAPER_A / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summarize] wrote {summary_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = []
        means = []
        stds = []
        if baseline_colab is not None:
            labels.append(f"A baseline\n(Colab 200ep ×{baseline_colab['n_seeds']})")
            means.append(baseline_colab["cpc_mean"])
            stds.append(baseline_colab["cpc_std"])
        if baseline_cpu is not None:
            labels.append(f"A baseline\n(CPU {baseline_cpu['epochs']}ep ×{baseline_cpu['n_seeds']})")
            means.append(baseline_cpu["cpc_mean"])
            stds.append(baseline_cpu.get("cpc_std", 0.0))
        labels.append(f"B frozen_gnn\n({gnn_ab['epochs']}ep ×{gnn_ab['n_seeds']})")
        means.append(gnn_ab["cpc_mean"])
        stds.append(gnn_ab.get("cpc_std", 0.0))
        labels.append(f"C frozen_rum\n({rum_ab['epochs']}ep ×{rum_ab['n_seeds']})")
        means.append(rum_ab["cpc_mean"])
        stds.append(rum_ab.get("cpc_std", 0.0))

        colors = ["#1976d2", "#1976d2", "#e57373", "#81c784"][-len(means):]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.005,
                    f"{m:.3f}", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("CPC (validation)")
        ax.set_title(f"DUAL_HET frozen-component ablation — verdict: {verdict}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(means) * 1.15)
        plt.tight_layout()
        png_path = PAPER_A / "ablation_summary.png"
        plt.savefig(png_path, dpi=140)
        print(f"[summarize] wrote {png_path}")
    except ImportError:
        print("[summarize] matplotlib not available — skipped PNG")

    # Human-readable markdown
    md_lines = [
        "# DUAL_HET 冻结组件 ablation 汇总",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"{reasoning}",
        "",
        "## CPC 对比",
        "",
        "| Config | CPC (mean ± std) | Epochs | Seeds | Note |",
        "|---|---|---|---|---|",
    ]
    if baseline_colab is not None:
        md_lines.append(
            f"| A baseline (Colab) | {baseline_colab['cpc_mean']:.4f} ± "
            f"{baseline_colab['cpc_std']:.4f} | {baseline_colab['epochs']} | "
            f"{baseline_colab['n_seeds']} | full GNN + full RUM |"
        )
    if baseline_cpu is not None:
        md_lines.append(
            f"| A baseline (CPU matched) | {baseline_cpu['cpc_mean']:.4f} ± "
            f"{baseline_cpu.get('cpc_std', 0):.4f} | {baseline_cpu['epochs']} | "
            f"{baseline_cpu['n_seeds']} | full GNN + full RUM, fair epoch budget |"
        )
    md_lines.append(
        f"| B frozen_gnn | {gnn_ab['cpc_mean']:.4f} ± "
        f"{gnn_ab.get('cpc_std', 0):.4f} | {gnn_ab['epochs']} | "
        f"{gnn_ab['n_seeds']} | encoder at random init (frozen); only RUM trains |"
    )
    md_lines.append(
        f"| C frozen_rum | {rum_ab['cpc_mean']:.4f} ± "
        f"{rum_ab.get('cpc_std', 0):.4f} | {rum_ab['epochs']} | "
        f"{rum_ab['n_seeds']} | β=-0.05, γ=-0.5, δ=0, κ=1 (literature priors, frozen); only GNN trains |"
    )
    md_lines.extend([
        "",
        "## 解读规则 (来自 2026-05-11 session memory)",
        "",
        "- B ≈ 0.20-0.30，C 也大跌 → **RUM 在 carry**，counterfactual 站得住 ✅",
        "- B ≈ A baseline → **GNN 一个人吃了所有信号** → Mozolin 灾难复发 ❌",
        "- B 中等 (~0.35) → 两者都贡献但 GNN 偏多 → 需 δ 控制机制 🟡",
        "",
        "## frozen_gnn 学到的 RUM 参数（看 RUM 是否回收了合理的经济学参数）",
        "",
    ])
    for r in gnn_ab["results"]:
        md_lines.append(
            f"- seed {r['seed']}: β_car={r['beta_car']:+.4f}, "
            f"β_transit={r['beta_transit']:+.4f}, "
            f"β_walk={r['beta_walk']:+.4f}, γ={r['gamma']:+.4f}, "
            f"κ={[round(k, 2) for k in r['kappa']]}"
        )

    md_path = PAPER_A / "ablation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[summarize] wrote {md_path}")

    print("\n=== Verdict ===")
    print(f"  {verdict}: {reasoning}")
    print(f"\n  A baseline (Colab): {cpc_baseline:.4f}" if cpc_baseline else "")
    if cpc_baseline_cpu:
        print(f"  A baseline (CPU):   {cpc_baseline_cpu:.4f}")
    print(f"  B frozen_gnn:       {cpc_b:.4f}")
    print(f"  C frozen_rum:       {cpc_c:.4f}")


if __name__ == "__main__":
    main()
