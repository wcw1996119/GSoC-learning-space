# -*- coding: utf-8 -*-
"""Second pass: resolve [confirm] markers using values read from the actual code."""
import pypandoc, io, re
md = r"C:\Users\Chengwei\Downloads\chapter5_final_0611.md"
out_docx = r"C:\Users\Chengwei\Downloads\Chapter5_Model_Specification.docx"
with io.open(md, encoding="utf-8") as f: t = f.read()

R = [
("AM peak ([confirm: window]), PM peak ([confirm: window]), midday ([confirm: window]) and night ([confirm: window]).",
 "AM peak (07:00–09:00), PM peak (17:00–19:00), midday (10:00–16:00) and night (20:00–06:00)."),

("divided by an assumed walking speed of 4.5 km/h, with a maximum threshold of [confirm: value] minutes beyond which walking is excluded.",
 "divided by an assumed walking speed of 5 km/h. No hard cap is applied to the walking time itself; the distance-aware mode-share prior sets the walking share to zero beyond 5 km, and the commute-time gate of Section 5.3.2 screens out implausibly long walking alternatives."),

("controlling the sharpness of the threshold [confirm: value of $\\delta$ or how it is set]",
 "controlling the sharpness of the threshold (in the implementation this sharpness is a learnable per-gate slope rather than a single fixed $\\delta$; the time gate's slope is initialised at 0.2 per minute and estimated jointly with the other parameters)"),

("[confirm: the implementation also includes a soft travel-cost screen alongside the time and occupation screens of this section.]",
 "In the implementation the consideration filter applies three soft screens together: the occupational-match and commute-time screens of this section and a soft travel-cost screen (a sigmoid in a cost-to-budget ratio)."),

("$$\\mathcal{L}_{\\text{mode}} = -\\sum_{m} s_{m}^{\\text{survey}} \\, \\ln \\hat{s}_{m} \\qquad \\text{[confirm: matched globally or by occupation group]}$$",
 "$$\\mathcal{L}_{\\text{mode}} = -\\sum_{m} s_{m}^{\\text{survey}} \\, \\ln \\hat{s}_{m}$$\n\nmatched against the global survey mode-split target $(0.21, 0.35, 0.44)$ for car, transit and walk."),

("Their values are [confirm: the values used, or the validation procedure by which they are chosen].",
 "In the main specification the anchoring weights are $\\lambda_{\\text{transit}} = 1.0$ and $\\lambda_{\\text{mode}} = 2.0$. [confirm: $\\lambda_{\\text{reg}}$.]"),

("initialised at [confirm: literature-derived or random starting points]. The GNN weights are initialised with [confirm: Xavier/Kaiming]. The consideration temperature $\\delta$ is [confirm: initial value and whether it is updated during training].",
 "initialised at [confirm: literature-derived or random starting points]. The GNN weights use the default PyTorch (Kaiming-uniform) initialisation. The commute-tolerance thresholds are initialised at 40, 60 and 90 minutes and the gate slopes at small positive values; both are updated during training and the thresholds converge close to the census-derived 42, 64 and 95 minutes."),

("that stabilises convergence [confirm: value of $\\eta$].",
 "that stabilises convergence [confirm: value of $\\eta$. Note: the implemented congestion model is first-order — a single free-flow all-or-nothing assignment with a capacity-capped BPR delay (volume/capacity capped at 3) — rather than a fully equilibrated fixed point; reconcile Sections 5.5.3–5.5.4 with this implementation]."),

("[confirm: reconcile with the implementation, in which $\\mathcal{D}_o$ is the set of destinations that have a finite recorded travel time in the Gaode matrix.]",
 "In the implementation, $\\mathcal{D}_o$ is the set of destination cells that have a finite recorded travel time in the Gaode matrix (the reachable set); this already contains every destination receiving observed flow, so subsets (ii) and (iii) above are subsumed by it and no random tail sample is drawn."),

("set at $\\alpha = 0.15$ and $\\gamma = 4$ (Bureau of Public Roads, 1964).",
 "set at $\\alpha = 0.15$ and $\\gamma = 4$ (Bureau of Public Roads, 1964), with the volume/capacity ratio capped at 3 before the power term."),
]

miss=[]
for old,new in R:
    if old in t: t=t.replace(old,new,1)
    else: miss.append(old[:70])

with io.open(md,"w",encoding="utf-8") as f: f.write(t)
print("pass2 applied", len(R)-len(miss), "/", len(R))
for m in miss: print("  MISS:", m)
print("remaining [confirm markers:", len(re.findall(r"\[confirm", t)))
pypandoc.convert_file(md, "docx", outputfile=out_docx)
print("docx rewritten:", out_docx)
