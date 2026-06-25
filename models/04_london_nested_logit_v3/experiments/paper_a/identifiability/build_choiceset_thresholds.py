# -*- coding: utf-8 -*-
"""Behavioural finding: the consideration-set thresholds the model learned from
flows alone. Commute-time tolerance and travel-cost budget both rise with income
— the choice-set story (Bhat 1995), not an identifiability table."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

tiers = ["lower\nincome", "middle\nincome", "higher\nincome"]
Tmax = [59.0, 87.9, 122.6]            # commute tolerance (min), softplus(raw_T_max)
budget = [2.02, 2.63, 7.70]           # travel-cost budget, softplus(raw_cost_budget)
x = np.arange(3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))

# Panel A: commute tolerance
c = ["#9ecae1", "#4292c6", "#08519c"]
b1 = ax[0].bar(x, Tmax, color=c, width=0.62, edgecolor="k", lw=.5)
for xi, v in zip(x, Tmax): ax[0].text(xi, v+2, f"{v:.0f} min", ha="center", fontsize=10, fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels(tiers, fontsize=10)
ax[0].set_ylabel("furthest commute a worker will consider  (minutes)", fontsize=10)
ax[0].set_title("(a) Commute-time tolerance rises with income", fontsize=11.5, loc="left", fontweight="bold")
ax[0].set_ylim(0, 145)
ax[0].text(0.5, 134, f"learned ratio  1 : {Tmax[1]/Tmax[0]:.1f} : {Tmax[2]/Tmax[0]:.1f}\n"
                     "Bhat (1995)   1 : 1.8 : 2.5   ✓ same pattern",
           fontsize=9.5, color="#08306b", ha="center",
           bbox=dict(boxstyle="round,pad=0.35", fc="#eef5fb", ec="#4292c6"))
ax[0].grid(alpha=.2, axis="y")

# Panel B: cost budget
b2 = ax[1].bar(x, budget, color=["#c7e9c0", "#74c476", "#238b45"], width=0.62, edgecolor="k", lw=.5)
for xi, v in zip(x, budget): ax[1].text(xi, v+0.15, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax[1].set_xticks(x); ax[1].set_xticklabels(tiers, fontsize=10)
ax[1].set_ylabel("travel-cost budget  (relative units)", fontsize=10)
ax[1].set_title("(b) Travel-cost budget rises with income", fontsize=11.5, loc="left", fontweight="bold")
ax[1].set_ylim(0, 9)
ax[1].text(0.5, 8.3, "higher earners tolerate costlier commutes", fontsize=9.5, color="#1a6e36", ha="center")
ax[1].grid(alpha=.2, axis="y")

fig.suptitle("What workplaces commuters even consider — choice-set thresholds the model learned from flows alone",
             fontsize=12, y=1.02)
fig.tight_layout()
out = r"D:\GIT\mesa Gsoc\GSoC-learning-space\models\04_london_nested_logit_v3\report\figures\choiceset_thresholds.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"), bbox_inches="tight")
print("saved", out)
