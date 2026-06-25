# -*- coding: utf-8 -*-
"""Identifiability map of the complete model: for each behavioural parameter, how
well the data RESTORES its structure after scrambling, and WHICH data does it —
commuting flows, injected external moments, or neither (a literature prior).
From the complete-loss finding-vs-prior test (flows + occupation census + class
census + mode shares)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, structure-recovered, source)  source: flows / external / prior
# recovered = std(refit deviations)/std(trained deviations) under the COMPLETE loss
rows = [
    ("Distance decay",                 1.32, "flows"),
    ("Cost tolerance threshold",       1.51, "flows"),
    ("Pull toward job-dense areas",    1.39, "flows"),
    ("Sensitivity to travel time",     0.65, "flows"),
    ("Avoiding crowded workplaces",    0.44, "flows"),
    ("Occupation - matching workplaces", 1.00, "external"),   # census (in-loss ρ→1.0); fvp run cut off
    ("Mode choice x car access",       0.93, "external"),
    ("Baseline mode preference",       0.76, "external"),
    ("Pull to high-pay areas (class)", 0.19, "external"),     # class census; direction only (weak)
    ("Travel-cost budget",             0.23, "prior"),
    ("Commute-time tolerance",         0.03, "prior"),
]
col = {"flows": "#1a9850", "external": "#2c6fbb", "prior": "#9a9a9a"}
src_lab = {"flows": "identified by commuting flows",
           "external": "identified by injected external data",
           "prior": "not identified — literature prior"}
# order: flows block (desc), external block (desc), prior block (desc)
order = ([r for r in rows if r[2]=="flows"] +
         [r for r in rows if r[2]=="external"] +
         [r for r in rows if r[2]=="prior"])
order = order[::-1]   # bottom-up so flows on top
labels = [r[0] for r in order]; vals = [min(r[1],1.6) for r in order]; cols = [col[r[2]] for r in order]
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(10.5, 7))
ax.barh(y, vals, color=cols, edgecolor="k", lw=.4, height=0.7)
ax.axvline(0.4, color="#cc3333", ls="--", lw=1.3)
ax.text(0.4, -0.85, "← weakly / not identified      well identified →", fontsize=8.5,
        color="#cc3333", ha="center")
for yi, r in zip(y, order):
    note = " *" if r[0].startswith("Occupation") else ("  (direction only)" if r[1] < 0.25 and r[2]=="external" else "")
    ax.text(min(r[1],1.6)+0.02, yi, f"{r[1]:.2f}{note}", va="center", fontsize=8.5,
            color="#cc3333" if r[2]=="prior" else "#333")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("how well the data recovers the parameter after it is scrambled\n"
              "(structure recovered, complete-model loss)", fontsize=10)
ax.set_xlim(0, 1.75)
ax.set_title("What identifies each behavioural parameter — flows, external data, or neither",
             fontsize=12.5, fontweight="bold", loc="left")
ax.legend(handles=[Patch(facecolor=col[k], label=src_lab[k]) for k in ["flows","external","prior"]],
          fontsize=9.5, loc="lower right", frameon=True)
fig.tight_layout()
out = r"D:\GIT\mesa Gsoc\GSoC-learning-space\models\04_london_nested_logit_v3\report\figures\identifiability_map.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"), bbox_inches="tight")
print("saved", out)
