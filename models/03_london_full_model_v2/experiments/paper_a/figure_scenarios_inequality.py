"""F6 Figure: Scenario A vs B inequality contrast.

Reads from scenario_A_accessibility.json + scenario_B_temporal.json.
Visualises the central narrative: spatial intervention is regressive,
temporal intervention is progressive.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


def main():
    sa = json.load(open(OUT_DIR / "scenario_A_accessibility.json"))
    sb = json.load(open(OUT_DIR / "scenario_B_temporal.json"))

    # Scenario A: gini change baseline -> scenario_BPR
    sa_rows = sa["rows"]
    sa_base_gini = sa_rows[0]["pop_gini"]
    sa_bpr_gini = sa_rows[2]["pop_gini"]
    sa_base_palma = sa_rows[0]["pop_palma"]
    sa_bpr_palma = sa_rows[2]["pop_palma"]
    sa_base_atk = sa_rows[0]["pop_atkinson_e0_5"]
    sa_bpr_atk = sa_rows[2]["pop_atkinson_e0_5"]

    # Scenario B: gini change baseline -> scenario at h08
    sb_h8 = sb["accessibility_h8"]
    sb_base_gini = sb_h8["gini_base"]
    sb_scen_gini = sb_h8["gini_scen"]
    sb_base_palma = sb_h8["palma_base"]
    sb_scen_palma = sb_h8["palma_scen"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Scenario A
    ax = axes[0]
    metrics = ["Gini", "Palma", "Atkinson\n(ε=0.5)"]
    base = [sa_base_gini, sa_base_palma, sa_base_atk]
    scen = [sa_bpr_gini, sa_bpr_palma, sa_bpr_atk]
    deltas_pct = [(s - b) / b * 100 for s, b in zip(scen, base)]
    xs = np.arange(len(metrics))
    width = 0.35
    ax.bar(xs - width/2, base, width, label="baseline", color="#aaaaaa")
    ax.bar(xs + width/2, scen, width, label="scenario A (BPR eq.)", color="#d62728")
    for i, (b, s, d) in enumerate(zip(base, scen, deltas_pct)):
        ax.text(i + width/2, s + 0.005, f"{d:+.2f}%", ha="center", va="bottom",
                fontsize=9, color="#d62728")
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Inequality index value", fontsize=11)
    ax.set_title("Scenario A: +65k jobs at OOC cluster\n(REGRESSIVE: all indices INCREASE)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Panel B: Scenario B
    ax = axes[1]
    metrics = ["Gini", "Palma"]
    base = [sb_base_gini, sb_base_palma]
    scen = [sb_scen_gini, sb_scen_palma]
    deltas_pct = [(s - b) / b * 100 for s, b in zip(scen, base)]
    xs = np.arange(len(metrics))
    ax.bar(xs - width/2, base, width, label="baseline", color="#aaaaaa")
    ax.bar(xs + width/2, scen, width, label="scenario B (flex work)", color="#2ca02c")
    for i, (b, s, d) in enumerate(zip(base, scen, deltas_pct)):
        ax.text(i + width/2, s + 0.005, f"{d:+.2f}%", ha="center", va="bottom",
                fontsize=9, color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Inequality index value (h=08)", fontsize=11)
    ax.set_title("Scenario B: 50% peak shift to off-peak\n(PROGRESSIVE: all indices DECREASE)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Two policy scenarios contrast on accessibility inequality",
                  fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "F6_scenarios_inequality.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
