"""Compare ABM-based Scenario B (behavioral) vs aggregate-based Scenario B
(mechanical intervention on congestion_z + row sums).

Reads:
  evaluation_outputs/paper_a/scenario_B_abm_dual_het.json              (flex_frac 0.30)
  evaluation_outputs/paper_a/abm_scenB_f50/scenario_B_abm_dual_het.json (flex_frac 0.50)
  evaluation_outputs/paper_a/abm_scenB_f100/scenario_B_abm_dual_het.json (flex_frac 1.00)
  evaluation_outputs/paper_a/scenario_B_dual_het.json                   (old aggregate)

Writes:
  evaluation_outputs/paper_a/scenario_B_abm_vs_aggregate.png
  evaluation_outputs/paper_a/scenario_B_abm_vs_aggregate.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation_outputs" / "paper_a"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    abm30 = load(OUT / "scenario_B_abm_dual_het.json")
    abm50 = load(OUT / "abm_scenB_f50" / "scenario_B_abm_dual_het.json")
    abm100 = load(OUT / "abm_scenB_f100" / "scenario_B_abm_dual_het.json")
    aggregate = load(OUT / "scenario_B_dual_het.json")

    # Per-hour flow (24 hours)
    hours = np.arange(24)
    base_abm = np.array(abm30["baseline"]["per_hour"])
    f30 = np.array(abm30["flex"]["per_hour"])
    f50 = np.array(abm50["flex"]["per_hour"])
    f100 = np.array(abm100["flex"]["per_hour"])

    # Aggregate per-hour from old scenario
    agg_base = np.array([r["flow_base"] for r in aggregate["per_hour_flow"]])
    agg_scen = np.array([r["flow_scen"] for r in aggregate["per_hour_flow"]])

    # Normalise to share for comparison
    base_abm_pct = base_abm / base_abm.sum()
    f30_pct = f30 / f30.sum()
    f50_pct = f50 / f50.sum()
    f100_pct = f100 / f100.sum()
    agg_base_pct = agg_base / agg_base.sum()
    agg_scen_pct = agg_scen / agg_scen.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ABM versions
    ax = axes[0]
    ax.plot(hours, base_abm_pct, "k-",  lw=2, label="Baseline (NTS dist)")
    ax.plot(hours, f30_pct,      "C0-", lw=2, label="ABM flex 30%")
    ax.plot(hours, f50_pct,      "C1-", lw=2, label="ABM flex 50%")
    ax.plot(hours, f100_pct,     "C3-", lw=2, label="ABM flex 100%")
    ax.axvspan(7, 9.5, alpha=0.15, color="red", label="7-9am peak")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of daily trips")
    ax.set_title("ABM-based Scenario B (behavioural flex)\n"
                 "agent-level: change p% of agents' departure_hour distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, 24, 2))

    # Right: aggregate version (rescaled to share for fair comparison)
    ax = axes[1]
    ax.plot(hours, agg_base_pct, "k-",  lw=2, label="Baseline (NTS dist)")
    ax.plot(hours, agg_scen_pct, "C2-", lw=2, label="Aggregate scenario\n(congestion_z × 0.5 peak / × 1.5 shoulder\n+ row-sum scaling)")
    ax.axvspan(7, 9.5, alpha=0.15, color="red", label="7-9am peak (×0.5)")
    ax.axvspan(10, 15.5, alpha=0.15, color="green", label="10-15 shoulder (×1.5)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of daily trips")
    ax.set_title("Aggregate-based Scenario B (mechanical)\n"
                 "no behavioural basis — directly scale demand mass per hour")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, 24, 2))

    plt.tight_layout()
    png_path = OUT / "scenario_B_abm_vs_aggregate.png"
    plt.savefig(png_path, dpi=140)
    plt.close()
    print(f"[compare] wrote {png_path}")

    # Markdown summary
    md = []
    md.append("# Scenario B — ABM (behavioral) vs Aggregate (mechanical) 对比")
    md.append("")
    md.append("**核心问题**：flex work 反事实该干预哪一层？")
    md.append("")
    md.append("- ABM 版本：选 p% 的 agent，把他们的 departure_hour 从 NTS 分布换成 7-11 均匀。"
              "保留剩下 (1-p)% 的 agent 不变。**有行为基础**。")
    md.append("- Aggregate 版本：直接对 origin row sums × 0.5 (peak hours 7-9) / × 1.5 (shoulder 10-15)，"
              "同时把 congestion_z 也按同样系数缩放。**没有行为对应**，机械强加。")
    md.append("")
    md.append("## 数值对比")
    md.append("")
    md.append("| 干预 | flex_frac / 系数 | 7-9am peak share | Δ peak share | dest corr |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Baseline (NTS) | — | 0.350 | — | 1.000 |")
    md.append(f"| **ABM flex 30%** | 30% agents | {abm30['flex']['morning_peak_share_7_9am']:.3f} | "
              f"{abm30['delta']['morning_peak_share_7_9am_change']:+.3f} | "
              f"{abm30['delta']['destination_correlation_pearson']:.3f} |")
    md.append(f"| **ABM flex 50%** | 50% agents | {abm50['flex']['morning_peak_share_7_9am']:.3f} | "
              f"{abm50['delta']['morning_peak_share_7_9am_change']:+.3f} | "
              f"{abm50['delta']['destination_correlation_pearson']:.3f} |")
    md.append(f"| **ABM flex 100%** | 100% agents | {abm100['flex']['morning_peak_share_7_9am']:.3f} | "
              f"{abm100['delta']['morning_peak_share_7_9am_change']:+.3f} | "
              f"{abm100['delta']['destination_correlation_pearson']:.3f} |")
    md.append(f"| **Aggregate (old)** | mass ×0.5 peak / ×1.5 shoulder | "
              f"{aggregate['peak_share_scenario_pct']/100:.3f} | "
              f"-{(aggregate['peak_share_baseline_pct'] - aggregate['peak_share_scenario_pct'])/100:.3f} (peak_share_pct: "
              f"{aggregate['peak_share_baseline_pct']:.1f}% → {aggregate['peak_share_scenario_pct']:.1f}%) | "
              f"未报告 |")
    md.append("")
    md.append("注：aggregate 的 peak_share 指标定义和 ABM 略不同（aggregate 是 8 hour 总 flow 中 peak hours 的占比；"
              "ABM 是 24 hour 总 trips 中 7-9am 的占比）。但方向是清楚的。")
    md.append("")
    md.append("## 关键观察")
    md.append("")
    md.append("1. **量级差距**：ABM-flex 100% 把 7-9am peak share 从 0.350 削到 0.233（-33% relative），"
              "已经接近行为可能的上限（所有人都成 flex 工作者）。aggregate 版本则把对应指标削到 7.9% "
              "（-55% relative）——比行为最大效应还猛。")
    md.append("")
    md.append("2. **机制差异**：")
    md.append("   - ABM：改 agent.departure_hour → DUAL_HET head 在新 hour 选 destination → "
              "destination 选择**自动适应**新 hour 的 V_jt 和 congestion")
    md.append("   - Aggregate：改 X_dynamic[congestion_z] + row sums → encoder + RUM 接受改过的输入 → "
              "destination 选择**被动接受**人为指定的拥堵和 mass")
    md.append("")
    md.append("3. **destination 选择稳定性**：ABM 30%/50%/100% 的 dest correlation 分别是 "
              "0.907/0.903/0.887。即使全员 flex，destination pattern 仍 88.7% 相关——说明 destination 选择"
              "**主要由 origin + mode + tier 决定，hour 是次要因子**。这符合直觉：你工作地点变不变跟你 8 点还是 10 点出门关系不大。")
    md.append("")
    md.append("## Paper §3 写作建议")
    md.append("")
    md.append("**建议用 ABM 版本作为 Scenario B 主报告**。理由：")
    md.append("")
    md.append("- ABM 有清晰的行为反事实语义（\"30% 的工作者改了出发时间\"）")
    md.append("- Aggregate 版本的 ×0.5 / ×1.5 系数是 ad-hoc 设定，不容易在 paper 里 defend")
    md.append("- ABM 利用了项目的 agent 层架构（agent.departure_hour），不浪费现有设计")
    md.append("")
    md.append("Aggregate 版本可以放 Supplementary 作 alternative formulation，"
              "或者放 §3.5 sensitivity analysis 里。")

    md_path = OUT / "scenario_B_abm_vs_aggregate.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[compare] wrote {md_path}")


if __name__ == "__main__":
    main()
