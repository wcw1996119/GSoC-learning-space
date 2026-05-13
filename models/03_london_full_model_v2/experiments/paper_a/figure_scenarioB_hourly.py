"""F7 Figure: Scenario B per-hour flow + congestion comparison."""
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
    sb = json.load(open(OUT_DIR / "scenario_B_temporal.json"))
    rows = sb["per_hour_flow"]
    bpr = sb["per_hour_bpr"]

    hours = [r["hour"] for r in rows]
    flow_base = [r["flow_base"] for r in rows]
    flow_scen = [r["flow_scen"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Top panel: per-hour flow base vs scen
    ax = axes[0]
    ax.bar(np.array(hours) - 0.2, flow_base, width=0.4, label="baseline", color="#aaaaaa")
    ax.bar(np.array(hours) + 0.2, flow_scen, width=0.4, label="scenario B (flex work)", color="#2ca02c")
    ax.set_ylabel("Total flow per hour", fontsize=11)
    ax.set_title("Scenario B — flexible work spreads peak demand to off-peak hours\n(50% peak reduction, 150% off-peak increase, mass conserved per origin)",
                  fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    # Highlight peak band
    ax.axvspan(6.5, 9.5, alpha=0.1, color="red")
    ax.axvspan(9.5, 15.5, alpha=0.1, color="green")
    ax.text(8, max(flow_base) * 0.95, "PEAK\n(7-9)", ha="center", color="red", fontsize=9)
    ax.text(12.5, max(flow_base) * 0.95, "SHOULDER\n(10-15)", ha="center", color="green", fontsize=9)

    # Bottom panel: BPR per-hour congestion (only sampled hours)
    ax = axes[1]
    bpr_hours = [b["hour"] for b in bpr]
    base_t = [b["mean_t_base"] for b in bpr]
    scen_t = [b["mean_t_scen"] for b in bpr]
    delta_pct = [b["delta_pct"] for b in bpr]
    xs = np.arange(len(bpr_hours))
    width = 0.35
    ax.bar(xs - width/2, base_t, width, label="baseline", color="#aaaaaa")
    ax.bar(xs + width/2, scen_t, width, label="scenario B", color="#2ca02c")
    for i, (b, s, d) in enumerate(zip(base_t, scen_t, delta_pct)):
        c = "#d62728" if d > 0 else "#2ca02c"
        ax.text(i + width/2, s + 0.1, f"{d:+.2f}%", ha="center", va="bottom",
                fontsize=8, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"h{h:02d}" for h in bpr_hours])
    ax.set_ylabel("Mean t_ij (min, BPR equilibrium)", fontsize=11)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_title("Per-hour BPR-equilibrium congestion (capacity = baseline peak inflow)",
                  fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    # Set y limits to focus on differences
    ax.set_ylim(min(scen_t) * 0.97, max(base_t) * 1.05)

    plt.tight_layout()
    out = OUT_DIR / "F7_scenarioB_hourly.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
