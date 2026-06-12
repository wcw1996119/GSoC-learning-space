# -*- coding: utf-8 -*-
"""Fill grounded blanks in chapter5_draft and export to docx. Leaves genuine unknowns as [confirm:]."""
import pypandoc, sys, io
src = r"C:\Users\Chengwei\Downloads\chapter5_draft_0611.md"
out_md = r"C:\Users\Chengwei\Downloads\chapter5_final_0611.md"
out_docx = r"C:\Users\Chengwei\Downloads\Chapter5_Model_Specification.docx"

with io.open(src, encoding="utf-8") as f:
    t = f.read()

R = [
("(Table 4.1, ID 15) in [three / four] steps.",
 "(Table 4.1, ID 15) in three steps."),

("least one outbound commuting trip. [Describe any filtering: minimum-flow threshold, removal of intra-cell trips, etc.] The records carry a timestamp",
 "least one outbound commuting trip. Cells with no outbound commuting record are excluded, leaving these 11,371 active origins; no minimum-flow threshold is applied to individual pairs, and intra-cell (within-grid) trips are retained, since they identify the same-cell term of Section 5.3.3. The records carry a timestamp"),

("commuting trips are identified as [describe the commuting extraction rule: e.g. repeated home-work pairs over N weekdays, morning departure window, etc.].",
 "commuting trips are identified by extracting stable home–workplace cell pairs from the records, with residence inferred from night-time presence and workplace from regular daytime presence across weekdays. [confirm: exact extraction window and stay-duration thresholds]"),

("AM peak ([time window]), PM peak ([time window]), midday ([time window]) and night ([time window]).",
 "AM peak ([confirm: window]), PM peak ([confirm: window]), midday ([confirm: window]) and night ([confirm: window])."),

("rather than variation in preferences. [If parameters do vary by period, state how.]",
 "rather than variation in preferences. The behavioural parameters $\\beta^c$ and $\\mu$ do not vary by period; period enters the behavioural core only through the travel-time argument $tt^t$. (The network residual of Section 5.4.2 is period-specific, as it encodes the diurnal population and congestion dynamics.)"),

("described in Section 5.2.4. [State any row/column normalisation or scaling applied to Y before estimation.]",
 "described in Section 5.2.4. No normalisation is applied to $\\mathbf{Y}$: observed flows enter the loss as counts, and predicted flows are reconstructed by scaling each origin's choice probabilities by its observed outbound total $N_o$ (Section 5.3.4), so the model fits the conditional destination distribution while the origin marginals are taken as given."),

("where $\\text{CI}_{od}$ is the congestion index for the pair. [Confirm or correct this formula.]",
 "where $\\text{CI}_{od}$ is the congestion index for the pair. [confirm: in the implementation the free-flow time is read directly as the $t_{ff}$ field of the Gaode matrix, and the period-specific congested time uses $t_{obs}$ at the AM and PM peaks, $t_{ff}$ at night, and their average at midday; reconcile with the formula above]"),

("queried for [describe query protocol: departure times, number of queries, sampling strategy].",
 "queried for [confirm: departure times sampled per period, number of queries, and sampling strategy]."),

("calibrated to operating speeds measured from smart-card records [describe calibration briefly]",
 "calibrated to an average operating speed of approximately 13.4 km/h estimated from the bus vehicle-status (GPS) records"),

("transfer time [if applicable], and egress time",
 "transfer time [confirm: whether transfers are modelled], and egress time"),

("or a simpler heuristic.] Pairs for which no transit path exists within [threshold] minutes",
 "or a simpler heuristic.] Pairs for which no transit path exists within [confirm: threshold] minutes"),

("divided by [assumed walking speed, e.g. 4.5 km/h], with a maximum threshold of [value] minutes",
 "divided by an assumed walking speed of 4.5 km/h, with a maximum threshold of [confirm: value] minutes"),

("who belong to group $c$. [If any spatial smoothing or downscaling is applied, describe it.]",
 "who belong to group $c$. No spatial smoothing or downscaling is applied: $\\pi_o^c$ is therefore constant within a district, the resolution limit discussed in Section 4.3.3."),

("from which the tolerable commuting times above are derived as [describe: e.g. the 95th percentile of observed durations within each tier].",
 "from which the tolerable commuting times above are derived from the upper tail of the tier-specific commuting-duration distributions (the tier means are about 21, 34 and 44 minutes, against the tolerances 42, 64 and 95 minutes). [confirm: exact percentile or scaling rule]"),

("Industries are mapped to the seven occupation categories [describe the mapping briefly], and the resulting surface",
 "Industries are mapped to the seven occupation categories through a semantic crosswalk between the 2002 industrial classification used by the 2008 census and the occupation categories (matching on business content rather than code position), so that, for instance, manufacturing and transport industries map to production–transport occupations while information and finance industries map to professional and clerical occupations, and the resulting surface"),

("even though the target tensor $\\mathbf{Y}$ does not distinguish modes. [Describe how this enters the loss function or the estimation: e.g. as a KL-divergence penalty on the predicted transit OD normalised to a distribution, matched against the normalised smart-card OD.]",
 "even though the target tensor $\\mathbf{Y}$ does not distinguish modes. It enters the loss of Section 5.4.3 as a KL-divergence penalty between the predicted transit OD, normalised to a distribution, and the normalised smart-card OD, so that only the spatial pattern, not the volume, is matched."),

("all cells within [radius or time threshold] of $o$",
 "all cells within [confirm: radius or time threshold] of $o$"),

("a random sample of [N] cells from the remainder",
 "a random sample of [confirm: N] cells from the remainder"),

("stratified by [district / distance band], to represent the tail of the distribution. [Adjust to match the actual construction.]",
 "stratified by [confirm: district / distance band], to represent the tail of the distribution. [confirm: reconcile with the implementation, in which $\\mathcal{D}_o$ is the set of destinations that have a finite recorded travel time in the Gaode matrix.]"),

("In practice, [state the typical size of $\\mathcal{D}_o$ and any evidence that further enlargement does not change the estimates, e.g. from a pilot run with doubled candidate sets].",
 "In practice, the candidate set averages about 2,674 destinations per origin, and the candidate sets together cover 97.2\\% of all observed commuting flow. [confirm: evidence from an enlarged-candidate pilot, if run]."),

("$\\delta$ is a temperature parameter controlling the sharpness of the threshold [state value or how it is set]",
 "$\\delta$ is a temperature parameter controlling the sharpness of the threshold [confirm: value of $\\delta$ or how it is set]"),

("[If the consideration mechanism has additional layers, e.g. a mode-availability screen based on car ownership or transit coverage, describe them here.]",
 "[confirm: the implementation also includes a soft travel-cost screen alongside the time and occupation screens of this section.]"),

("The nest parameter $\\mu$ is [estimated / fixed at a value; if estimated, state whether it varies by group].",
 "The nest parameter $\\mu$ is estimated jointly with the other parameters as a single value shared across groups, constrained to $(0, 1]$ by the reparameterisation $\\mu = 0.05 + 0.95\\,\\sigma(\\tilde{\\mu})$."),

("where $N_\\beta$ is the number of parameters per group. [Confirm or adjust the exact parameterisation; add any interaction terms, e.g. time $\\times$ period, if present.]",
 "where $N_\\beta$ is the number of parameters per group, leaving two free mode constants and three slope parameters per group. [confirm: exact $N_\\beta$, and reconcile with the implementation, in which destination attractiveness enters through the occupation-typed employment surface and the time sensitivity varies with distance.]"),

("$\\text{cost}_{od}^m$ is travel cost [define: e.g. fuel cost for car, fare for bus, zero for walking], $E_d^c$",
 "$\\text{cost}_{od}^m$ is travel cost [confirm: e.g. distance-based fuel cost for car, fare for bus, zero for walking], $E_d^c$"),

("is a mode-specific constant for group $c$ [state which mode is normalised to zero].",
 "is a mode-specific constant for group $c$ (walking is the reference mode, $\\text{ASC}_{\\text{walk}}^c = 0$)."),

("[If additional constraints are imposed, e.g. that higher-income groups have smaller (less negative) time coefficients in absolute terms, or that the value of time rises with education tier, state them and their implementation.]",
 "No further sign or ordering constraints are imposed: the value of travel time is left free across groups, so any education gradient in time sensitivity is an estimation outcome rather than an assumption. [confirm]"),

("the edges $\\mathcal{E}$ connect [describe the graph construction: e.g. k-nearest neighbours, cells sharing a boundary, cells within a distance threshold, or edges defined by nonzero observed flows].",
 "the edges $\\mathcal{E}$ connect each cell to its $k = 8$ nearest neighbours by centroid distance, made symmetric."),

("composed of [list the node features: e.g. employment count, residential population, land-use mix, distance to CBD, district indicators]. The GNN propagates information across the graph through [number] message-passing layers:",
 "composed of the log occupation-typed employment, the log competition (residents per job), an income index, the log wage proxy, the standardised log job and resident counts, and the two standardised centroid coordinates — eight features in total. The GNN propagates information across the graph through two message-passing layers:"),

("and Update and Aggregate are [describe the specific GNN variant used: e.g. GraphSAGE mean aggregator, GAT attention, GCN spectral convolution]. After [L] layers,",
 "and Update and Aggregate form a GraphSAGE layer: a linear transform of the node's own representation summed with a linear transform of the mean of its neighbours' representations, followed by a ReLU. After two layers,"),

("where $\\mathbf{W}_r$ is a learnable $[H \\times H]$ matrix and $H$ is the dimension of the node representation.",
 "where $\\mathbf{W}_r$ is a learnable $H \\times H$ matrix and $H = 16$ is the dimension of the node representation."),

("[If the form is different, e.g. a concatenation followed by an MLP, describe it instead.]",
 "[confirm: in the implementation the edge score is a dot product of period-specific origin and destination embeddings produced by a dual static/dynamic encoder; reconcile with the single-matrix bilinear form above.]"),

("[State the specific form: e.g. Poisson log-likelihood, KL divergence, or weighted MSE on log-flows:]",
 "The primary objective is a flow-weighted cross-entropy over each origin's destination distribution: every observed trip contributes the negative log of the predicted probability of its destination, summed over candidate pairs and the four periods:"),

("$$\\mathcal{L}_{\\text{OD}} = [- \\sum_{o,d,t} \\left( y_{od}^t \\ln \\hat{y}_{od}^t - \\hat{y}_{od}^t \\right) \\quad \\text{(Poisson)}, \\text{ or describe alternative}]$$",
 "$$\\mathcal{L}_{\\text{OD}} = -\\sum_{o,d,t} y_{od}^t \\, \\ln \\hat{P}(d \\mid o; \\, tt^t)$$"),

("$$\\mathcal{L}_{\\text{mode}} = \\sum_{o, m} [\\text{describe form: e.g. } (\\hat{s}_{om} - s_{om}^{\\text{survey}})^2 \\text{ averaged over origins}]$$",
 "$$\\mathcal{L}_{\\text{mode}} = -\\sum_{m} s_{m}^{\\text{survey}} \\, \\ln \\hat{s}_{m} \\qquad \\text{[confirm: matched globally or by occupation group]}$$"),

("$$\\mathcal{L}_{\\text{transit}} = [KL\\!\\left(\\hat{p}_{od}^{\\text{bus}} \\,\\|\\, p_{od}^{\\text{card}}\\right) \\text{ or describe alternative}]$$",
 "$$\\mathcal{L}_{\\text{transit}} = \\mathrm{KL}\\!\\left(p_{od}^{\\text{card}} \\,\\|\\, \\hat{p}_{od}^{\\text{bus}}\\right)$$"),

("[Describe any spatial aggregation applied before comparison, e.g. matching at the district-pair level rather than cell-pair level.]",
 "[confirm: aggregation level before comparison — cell-pair or district-pair.]"),

("[Describe the regularisation terms: e.g. L2 penalty on GNN weights, smoothness penalty on the residual $r_{od}$ across neighbouring pairs, penalty anchoring $\\beta^c$ magnitudes to literature values. State which literature values are used.]",
 "[confirm: regularisation terms — e.g. an L2 penalty on the GNN weights and any anchoring of $\\beta^c$ magnitudes to literature values.]"),

("Their values are [state values or describe how they are chosen, e.g. by validation performance on held-out origins].",
 "Their values are [confirm: the values used, or the validation procedure by which they are chosen]."),

("The model is implemented in [PyTorch / JAX / other] with [GNN library, e.g. PyTorch Geometric / DGL]. All parameters are optimised jointly using [optimiser, e.g. Adam] with learning rate [value], batch size [value over origins / mini-batch strategy], and [any learning-rate schedule].",
 "The model is implemented in PyTorch with a custom GraphSAGE-style message-passing implementation. All parameters are optimised jointly using Adam with learning rate 0.03, processing origins in chunks (sixteen origin-chunks per forward pass to bound memory). [confirm: learning-rate schedule, if any.]"),

("Training proceeds for [N] epochs or until [convergence criterion: e.g. the relative change in $\\mathcal{L}$ falls below $10^{-k}$ for $n$ consecutive epochs]. [Describe any curriculum or staged training: e.g. warming up the parametric branch before activating the GNN, or annealing the regularisation weight.]",
 "Training proceeds for 300 epochs, retaining the checkpoint with the best held-out flow similarity. [confirm: any staged training or warm-up.]"),

("initialised at values corresponding to [describe initialisation: e.g. literature-derived starting points for each group's time and cost sensitivities, or random draws within a plausible range]. The GNN weights are initialised with [Xavier / Kaiming / other]. The consideration temperature $\\delta$ is initialised at [value] and [is / is not] updated during training.",
 "initialised at [confirm: literature-derived or random starting points]. The GNN weights are initialised with [confirm: Xavier/Kaiming]. The consideration temperature $\\delta$ is [confirm: initial value and whether it is updated during training]."),

("[Describe the hardware: GPU type, training time. Describe any data-parallel or distributed training if applicable.]",
 "Each estimation runs on a single NVIDIA RTX 4090 (24 GB) and is repeated over three random seeds. [confirm: wall-clock training time.]"),

("Transit and walking times remain exogenous [confirm or state if transit is also endogenised].",
 "Transit and walking times remain exogenous; only car travel times are endogenised through congestion."),

("set at $\\alpha = [0.15]$ and $\\gamma = [4]$ [state the values used and their source, e.g. BPR, 1964; Spiess, 1990].",
 "set at $\\alpha = 0.15$ and $\\gamma = 4$ (Bureau of Public Roads, 1964)."),

("that stabilises convergence [state value, e.g. $\\eta = 0.3$].",
 "that stabilises convergence [confirm: value of $\\eta$]."),

("for a tolerance $\\epsilon = [10^{-4}]$, or until a maximum of [N] iterations is reached.",
 "for a tolerance $\\epsilon = 10^{-4}$, or until a maximum of [confirm: N] iterations is reached."),

("The implementation uses [JAX / PyTorch / other].",
 "The implementation uses PyTorch."),

("$$\\mathcal{L}_{\\text{OD}}^{\\text{eq}} = [\\text{same form as Section 5.4.3, but evaluated at equilibrium times}]$$",
 "$$\\mathcal{L}_{\\text{OD}}^{\\text{eq}} = -\\sum_{o,d,t} y_{od}^t \\, \\ln \\hat{P}(d \\mid o; \\, tt^*)$$"),

("The results belong to [Chapter 6 / later chapters]; what follows is the protocol.",
 "The results belong to Chapter 6; what follows is the protocol."),

("Run the full Stage A estimation from [N, e.g. 5] different random seeds, holding all other settings constant.",
 "Run the full Stage A estimation from three different random seeds, holding all other settings constant."),

("The relative error should be below [threshold, e.g. $10^{-4}$] for each tested parameter.",
 "The relative error should be below $10^{-4}$ for each tested parameter."),

("Report [metrics: e.g. RMSE, $R^2$, Sørensen similarity] at the cell-pair level and at the district-pair level.",
 "Report the Sørensen similarity (CPC), RMSE and $R^2$ at the cell-pair level and at the district-pair level."),

("[Chapter 6 / The following chapter] reports the results.",
 "Chapter 6 reports the results."),

("Training of Stage A is performed on [GPU type, number of GPUs, approximate wall-clock time].",
 "Training of Stage A is performed on a single NVIDIA RTX 4090 (24 GB). [confirm: number of GPUs and wall-clock time.]"),
]

miss=[]
for old,new in R:
    if old in t: t = t.replace(old,new,1)
    else: miss.append(old[:70])

with io.open(out_md,"w",encoding="utf-8") as f: f.write(t)
print("applied", len(R)-len(miss), "/", len(R), "replacements")
if miss:
    print("---- NOT MATCHED ----")
    for m in miss: print("  *", m)

# remaining [confirm: / instruction brackets count
import re
rem = re.findall(r"\[[^\]]+\]", t)
print("remaining bracketed markers:", len(rem))

pypandoc.convert_file(out_md, "docx", outputfile=out_docx)
print("docx written:", out_docx)
