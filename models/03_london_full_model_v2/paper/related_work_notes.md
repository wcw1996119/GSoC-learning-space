# Related Work Notes — Paper A bibliography prep

> **Purpose**: Structured 1-paragraph reading notes per work. Bibliography prep, NOT a polished section. The condensed §2 of the paper draws from these.
> **Format per entry**: Summary → Specific contribution → How Paper A differs/extends → 2–3 quotable lines.

## Table of Contents

1. [Yao et al. 2021 — SI-GCN](#1-yao-et-al-2021--si-gcn)
2. [Simini et al. 2021 — Deep Gravity](#2-simini-et-al-2021--deep-gravity)
3. [Pappalardo et al. 2023 — Mobility Survey](#3-pappalardo-et-al-2023--human-mobility-survey)
4. [Train 2009 — Discrete Choice Methods](#4-train-2009--discrete-choice-methods-with-simulation)
5. [Amos & Kolter 2017 — OptNet](#5-amos--kolter-2017--optnet)
6. [Blondel et al. 2022 — jaxopt](#6-blondel-et-al-2022--jaxopt)
7. [Ziebart 2008 — MaxEnt IRL](#7-ziebart-et-al-2008--maximum-entropy-irl)
8. [Schölkopf et al. 2021 — Causal Representation Learning](#8-sch%C3%B6lkopf-et-al-2021--causal-representation-learning)
9. [Pearl 2009 — Causality](#9-pearl-2009--causality)
10. [Manski 2003 — Partial Identification](#10-manski-2003--partial-identification)
11. [Geurs & van Wee 2004 — Accessibility](#11-geurs--van-wee-2004--accessibility-measures)
12. [Lenormand et al. 2016 — CPC metric](#12-lenormand-et-al-2016--cpc-metric)
13. [McFadden 1978 — Sampling of Alternatives](#13-mcfadden-1978--sampling-of-alternatives)
14. [Cinelli & Hazlett 2020 — Robustness Value](#14-cinelli--hazlett-2020--robustness-value)
15. [Imbens 2003 — Sensitivity to Exogeneity](#15-imbens-2003--sensitivity-to-exogeneity-assumptions)
16. [Veličković et al. 2018 — GAT](#16-veli%C4%8Dkovi%C4%87-et-al-2018--graph-attention-networks)
17. [Wilson 1971 — Entropy-Maximising Gravity](#17-wilson-1971--entropy-maximising-doubly-constrained-gravity)

---

## 1. Yao et al. 2021 — SI-GCN

**Citation**: Yao, X., Gao, Y., Zhu, D., Manley, E., Wang, J., & Liu, Y. (2021). "Spatial Origin–Destination Flow Imputation Using Graph Convolutional Networks." *IEEE Transactions on Intelligent Transportation Systems*, 22(12), 7474–7484.

**Summary**: Proposes SI-GCN, a graph-convolutional model for imputing missing OD flows. Treats OD prediction as a link-prediction problem on a bipartite zone-zone graph; uses two GCN encoders for origins and destinations, then a bilinear decoder for flow magnitude. Reports strong imputation accuracy on Chinese metropolitan OD data.

**Specific contribution**: Demonstrates that GNN-based encoders can capture spatial-network structure better than gravity-style hand-engineered features for flow imputation.

**How Paper A differs**:
- SI-GCN's loss is direct flow regression (MSE on Y_ij). Paper A's loss is RUM-implied softmax NLL — preserves choice-theoretic interpretation.
- SI-GCN does not identify behavioural parameters; the encoder is a black box. Paper A's GNN output is interpretable as a structural attractiveness scalar V_jt.
- SI-GCN does not consider counterfactuals. Paper A's L2-conditional protocol gives a principled way to use the trained model for `do()` queries.

**Quotable**:
> "the bilinear decoder predicts the flow magnitude as f(h_o, h_d) = h_o^T M h_d" — Paper A: we replace the bilinear flow decoder with a softmax over destination utilities, retaining the GNN encoder design.
> "GCN outperforms the gravity model" — we cite this to motivate GNN attractiveness over `log(employment_j)`.

---

## 2. Simini et al. 2021 — Deep Gravity

**Citation**: Simini, F., Barlacchi, G., Luca, M., & Pappalardo, L. (2021). "A Deep Gravity model for mobility flows generation." *Nature Communications*, 12, 6576.

**Summary**: Replaces the parametric gravity model with a deep neural network that takes origin/destination feature vectors and distance, outputting a flow estimate. Trained on OD data from multiple countries; generalises across geographies without retraining.

**Specific contribution**: Establishes that deep models substantially outperform classical gravity on flow prediction across many geographies, and that a single learned model transfers across cities.

**How Paper A differs**:
- Deep Gravity is a **flow predictor** — no identified behavioural parameters. We train a **structural** model whose parameters have economic meaning (VOT, distance-deterrence).
- Deep Gravity has no causal claim — it's a regression. Paper A is explicit about L2-conditional and ICM.
- We share the cross-city transfer ambition (forward reference to our Paper B Beijing transfer).

**Quotable**:
> "Deep Gravity can generate flows in geographic areas with no historical data" — we cite to motivate that GNN attractiveness functions can transfer; we then show how to make them carry **identified behavioural parameters**.
> "a single neural network model trained on multiple countries" — points toward multi-city training as a falsification of the ICM assumption (Paper B framing).

---

## 3. Pappalardo et al. 2023 — Human Mobility Survey

**Citation**: Pappalardo, L., Manley, E., Sekara, V., & Alessandretti, L. (2023). "Future directions in human mobility science." *Nature Computational Science*, 3, 588–600.

**Summary**: Review article surveying the state of human mobility science, including OD modelling, individual trajectory prediction, and policy applications. Identifies open problems in interpretability, transferability, and causal inference for mobility models.

**Specific contribution**: Maps the open problems Paper A directly addresses (interpretability + causality at urban scale).

**How Paper A differs / connects**:
- We position Paper A as targeting two of the survey's open problems: (a) "interpretability of deep mobility models" and (b) "causal inference from observational mobility data".
- The survey explicitly calls for hybrid mechanistic–neural approaches; Paper A's GNN-as-structural-function inside a RUM is exactly this.

**Quotable**:
> "future mobility models should be both predictive and explanatory" — we cite at the start of §1 as the gap statement.
> "causal inference remains underexplored" — we cite to position Paper A's L2-conditional contribution.

---

## 4. Train 2009 — Discrete Choice Methods with Simulation

**Citation**: Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.

**Summary**: Standard graduate-level reference for discrete choice estimation. Chapter 4 covers MNL, nested logit, and identification under linear-in-parameters utility. Chapter 6 surveys mode-choice elasticity ranges from US/EU literature.

**Specific contribution**: Establishes identification theory and standard estimation procedures for RUM models; provides elasticity benchmark ranges for mode choice and VOT.

**How Paper A uses**:
- Chapter 4: foundation for our identification arguments; we extend the linear-parameter case to GNN-parameterised utility.
- Chapter 6: source of the literature elasticity bracket in `validation/paper_a/literature_elasticity_db.py`.

**Quotable**:
> "the magnitude of the parameters is identified up to scale, with the scale determined by the variance of the unobserved utility component" — paraphrased in Appendix A.
> Table 6.2 (mode-choice β̂_t ranges): -0.08 to -0.02 per minute → enters our literature bracket directly.

---

## 5. Amos & Kolter 2017 — OptNet

**Citation**: Amos, B., & Kolter, J. Z. (2017). "OptNet: Differentiable Optimization as a Layer in Neural Networks." *ICML 2017*.

**Summary**: Introduces the OptNet layer: a differentiable QP-solving layer where forward pass solves an argmin and the backward pass uses implicit differentiation through KKT conditions. Enables embedding optimisation as a learnable component.

**Specific contribution**: Establishes the implicit-differentiation toolkit for embedding constrained optimisation in deep learning.

**How Paper A uses**:
- Theoretical foundation for our implicit-diff through the BPR equilibrium fixed point.
- We use `jaxopt` (Blondel et al. 2022) which generalises OptNet's idea beyond QPs to fixed points.

**Quotable**:
> "differentiating through the optimum lets the optimisation problem itself become a learnable component of the architecture" — we cite as the conceptual ancestor of our setup.

---

## 6. Blondel et al. 2022 — jaxopt

**Citation**: Blondel, M., Berthet, Q., Cuturi, M., Frostig, R., Hoyer, S., Llinares-López, F., Pedregosa, F., & Vert, J.-P. (2022). "Efficient and Modular Implicit Differentiation." *NeurIPS 2022*.

**Summary**: Generalises OptNet to arbitrary fixed-point and root-finding problems via the implicit function theorem; provides a Python library (jaxopt) for declaring optimality conditions and getting backward gradients automatically.

**Specific contribution**: Practical, modular, scalable implicit differentiation in JAX. O(K) memory regardless of forward iteration count.

**How Paper A uses**:
- Direct dependency: our BPR-fixed-point backward pass uses `jaxopt.FixedPointIteration` with `implicit_diff=True`.
- Citation justifying that gradients are exact (up to convergence tolerance), not unrolled approximations.

**Quotable**:
> "implicit differentiation decouples the forward solver from the backward pass, enabling exact gradients with constant memory cost" — we cite for our memory-efficiency claim.

---

## 7. Ziebart et al. 2008 — Maximum Entropy IRL

**Citation**: Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). "Maximum Entropy Inverse Reinforcement Learning." *AAAI 2008*.

**Summary**: Introduces MaxEnt IRL: recover a reward function from demonstrations by maximising the entropy of the trajectory distribution subject to feature-matching constraints. Equivalent in single-step settings to logit choice estimation under a Gumbel error.

**Specific contribution**: Formal connection between IRL and discrete choice; Gumbel/softmax structure shared.

**How Paper A connects**:
- Our setup is structurally identical to MaxEnt IRL with **states = origins, actions = destination choices, rewards = U_ij**. Calling this out positions Paper A within the IRL literature for ML reviewers.
- The crucial difference: we operate at urban *aggregate* scale (10⁶ trips, 10³ "actions") whereas IRL is typically trained on individual trajectory data.

**Quotable**:
> "the maximum entropy distribution over trajectories is consistent with the principle of insufficient reason" — paraphrased.
> "MaxEnt IRL recovers reward functions from observed behaviour without requiring a stochasticity model beyond the maximum-entropy assumption" — useful for arguing that aggregate OD plus Gumbel softmax is enough to identify utility.

---

## 8. Schölkopf et al. 2021 — Causal Representation Learning

**Citation**: Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). "Toward Causal Representation Learning." *Proceedings of the IEEE*, 109(5), 612–634.

**Summary**: Reviews the connection between causal inference and representation learning; articulates the **Independent Causal Mechanisms (ICM) principle** — that the structural mechanism for each variable is invariant under interventions on other variables.

**Specific contribution**: Defensible non-randomisation route to L2 claims: invoke ICM as an explicit assumption rather than randomisation or back-door identification.

**How Paper A uses**:
- Central citation for the framing of "L2 conditional on stated SCM" — we adopt ICM as the assumption underlying scenario simulation.
- We do NOT claim ICM holds; we **assume** it and bound violations via an Imbens-style omitted-variable sensitivity argument (Imbens 2003); see §15 below.

**Quotable**:
> "the independent causal mechanisms principle states that the conditional distribution of each variable given its causal parents is invariant under interventions on other variables" — direct citation in §2.5 of v2_5_paperA_inverse_choice.md.
> "ICM is an assumption, not a theorem; its violation is empirically testable but only via interventional data" — supports our sensitivity-bound approach.

---

## 9. Pearl 2009 — Causality

**Citation**: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press.

**Summary**: Canonical reference for the structural-causal-models / do-calculus framework. Defines the L1/L2/L3 hierarchy (associational, interventional, counterfactual). Provides identification theorems for L2 from observational data under back-door, front-door, and IV conditions.

**Specific contribution**: Formal definition of the ladder rungs Paper A is positioning against.

**How Paper A uses**:
- Cite for the L1/L2/L3 hierarchy in §1 and §8.
- Cite for the strict definition that justifies why our claim is "L2 conditional" not "L2 unconditional".
- Cite for the do-operator notation `do(X_OOC.employment += 65k)`.

**Quotable**:
> "interventions cannot be evaluated from observational data alone unless one assumes a specific structural model and verifies its identification conditions" — paraphrased to support our explicit-SCM stance.
> Pearl's L1/L2/L3 ladder appears in many figure captions / discussion paragraphs.

---

## 10. Manski 2003 — Partial Identification

**Citation**: Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer Series in Statistics.

**Summary**: Develops the theory of partial (set) identification when point identification is impossible — derive bounds rather than insist on point estimates. Influential for honest reporting of what data can and cannot say.

**Specific contribution**: Conceptual basis for "what is and is not identified" tables; cautions against overclaiming under untestable assumptions.

**How Paper A uses**:
- Frames Section 8.3 ("structurally identified, not causally identified") and the Table in `identifiability_argument.md` §6.
- Methodologically motivates the Imbens-style omitted-variable bound (Imbens 2003) used in §4 of `identifiability_argument.md`: we bound the unidentified portion rather than ignore it.

**Quotable**:
> "what data alone can reveal differs sharply from what data combined with assumptions can reveal" — direct quote, used in §8.

---

## 11. Geurs & van Wee 2004 — Accessibility Measures

**Citation**: Geurs, K. T., & van Wee, B. (2004). "Accessibility evaluation of land-use and transport strategies: review and research directions." *Journal of Transport Geography*, 12(2), 127–140.

**Summary**: Review of accessibility measure definitions: location-based (gravity, cumulative-opportunity), person-based (space-time geography), utility-based (logsum from MNL). Discusses tradeoffs and use cases.

**Specific contribution**: Establishes utility-based accessibility (the MNL logsum) as the theoretically principled measure; gravity-based measures are special cases.

**How Paper A uses**:
- Justifies our reporting of accessibility as the choice-set logsum: `A_i = log Σ_j exp(α V_j + β t_ij + γ log d_ij)`.
- Cites for the macro-validation that our recovered model produces sensible accessibility patterns (§5 of paper outline).

**Quotable**:
> "utility-based accessibility measures are derived directly from random utility theory and have welfare-economic interpretation" — supports our use of logsum as a metric.
> "gravity-based accessibility is a special case of utility-based accessibility under particular utility specifications" — useful for connecting our approach to the broader accessibility literature.

---

## 12. Lenormand et al. 2016 — CPC metric

_(added per R5: missing related-work entry.)_

**Citation**: Lenormand, M., Bassolas, A., & Ramasco, J. J. (2016). "Systematic comparison of trip distribution laws and models." *Journal of Transport Geography*, 51, 158–169.

**Summary**: Empirical comparison of trip-distribution laws (gravity, intervening-opportunities, radiation) across multiple European cities. Defines the **Common Part of Commuters (CPC)** metric — `CPC(Y_obs, Y_pred) = 2 Σ min(Y_obs_ij, Y_pred_ij) / (Σ Y_obs_ij + Σ Y_pred_ij)` ∈ [0, 1] — and uses it as the primary aggregate-flow goodness-of-fit measure.

**Specific contribution**: Establishes CPC as the standard aggregate-OD goodness-of-fit metric for European cities and provides empirical reference values across cities.

**How Paper A uses**:
- CPC is the W8 milestone gate (≥ 0.5 on held-out test origins; see `methodology/v2_5_paperA_inverse_choice.md` §10).
- We cite Lenormand 2016's distance-deterrence γ bracket [-1.5, -0.8] in the literature elasticity database (`preregistration.md` §5.2).

**Quotable**:
> "the Common Part of Commuters provides a symmetric, bounded similarity between predicted and observed flow distributions" — paraphrased in §5.2 of paper outline.

---

## 13. McFadden 1978 — Sampling of Alternatives

_(added per R5: missing related-work entry.)_

**Citation**: McFadden, D. (1978). "Modeling the Choice of Residential Location." In Karlqvist, A., et al. (eds.), *Spatial Interaction Theory and Planning Models*, North-Holland, 75–96.

**Summary**: Establishes the **sampling-of-alternatives** correction for MNL estimation when the full choice set is too large to enumerate. Shows that the conditional MLE remains **consistent** when each agent's choice set is restricted to a random subset that includes the chosen alternative, provided the sampling probability is independent of the choice (the "uniform conditioning property") or correctable via an additive log-sampling-probability term.

**Specific contribution**: Theoretical justification for top-K choice sets in spatial choice estimation.

**How Paper A uses**:
- Direct theoretical foundation for our **Top-K=50 choice set** construction in `methodology/v2_5_paperA_inverse_choice.md` §3.4.
- We sample by minimum-time rather than uniformly; consistency is preserved as long as the sampling rule does not depend on the chosen destination's identity, which we verify by including the observed destination in every training sample.

**Quotable**:
> "consistent estimates of the parameters of a multinomial logit model can be obtained from a randomly selected subset of alternatives provided the chosen alternative is included" — direct paraphrase used in §3.4.

---

## 14. Cinelli & Hazlett 2020 — Robustness Value

_(added per R5: missing related-work entry; related to Imbens 2003 framework in §15.)_

**Citation**: Cinelli, C., & Hazlett, C. (2020). "Making sense of sensitivity: extending omitted variable bias." *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 82(1), 39–67.

**Summary**: Extends the omitted-variable-bias framework with the **robustness value** — a single summary statistic capturing how strong an unobserved confounder must be (jointly with treatment and outcome) to overturn a regression coefficient. Provides bias contour plots that visualise the joint (R²_treatment, R²_outcome) confounder strength needed to flip a conclusion.

**Specific contribution**: A modern, interpretable extension of the Imbens 2003 sensitivity framework with practical visualisation tools.

**How Paper A connects**:
- Cited in `identifiability_argument.md` §4.5 as a v2.6 extension target — non-linear / interaction-confounder sensitivity beyond the Imbens 2003 linear-in-U formula we use in §4.2.
- We do **not** implement Cinelli–Hazlett in v2.5 Paper A (scope cut); we report the simpler Imbens 2003 bound and flag the extension.

**Quotable**:
> "the robustness value is the smallest amount of confounder-strength jointly with both treatment and outcome that overturns the conclusion" — paraphrased; useful framing for §8.2.

---

## 15. Imbens 2003 — Sensitivity to Exogeneity Assumptions

_(added per R5 §4: this is the **actual citation** for the sensitivity bound used in `identifiability_argument.md` §4 — replacing the prior incorrect "Rosenbaum 2002" citation.)_

**Citation**: Imbens, G. W. (2003). "Sensitivity to Exogeneity Assumptions in Program Evaluation." *American Economic Review*, 93(2), 126–132.

**Summary**: Proposes a regression-based sensitivity-analysis framework for observational studies in program evaluation. Parametrises an unobserved confounder U by its partial correlations with the treatment-analog and the outcome, and derives a closed-form bound on the resulting bias. Designed for continuous-covariate regression settings, distinct from the Rosenbaum (2002) Γ-bound for matched-pair propensity-score designs.

**Specific contribution**: Closed-form omitted-variable bias bound (the "ω-formula") usable in any linear-in-parameters regression — directly applicable to our linear-in-(α, β, γ) utility specification.

**How Paper A uses**:
- **Direct foundation for `identifiability_argument.md` §4** Imbens-style omitted-variable sensitivity bound.
- We use the regression form (not the matched-pair form) because our setting is regression-based: we fit (α, β, γ) by conditional MLE on the linear utility, and want to bound how much of β̂ could be unobserved-confounder-driven.
- **Distinct from Rosenbaum 2002**: Rosenbaum bounds odds-ratio shifts in matched-pair designs; we are not matching pairs, we are fitting a continuous-covariate regression. Imbens 2003 is the appropriate citation.

**Quotable**:
> "the sensitivity of treatment-effect estimates to violations of unconfoundedness can be summarised by the partial-correlation strengths an omitted confounder would need to overturn the conclusion" — paraphrased in `identifiability_argument.md` §4.2.

---

## 16. Veličković et al. 2018 — Graph Attention Networks

_(added per R5: missing related-work entry; relevant for the §4.4 ablation alternative to GraphSAGE.)_

**Citation**: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). "Graph Attention Networks." *ICLR 2018*.

**Summary**: Introduces the **GAT** layer, which computes node representations by attending over a node's neighbours with learnable attention weights, instead of using fixed (e.g., degree-normalised) aggregation as in GCN. Multi-head attention stabilises training and increases expressivity.

**Specific contribution**: A widely-adopted alternative to GCN/GraphSAGE that lets the model learn neighbour importance per node.

**How Paper A uses**:
- **Default GNN architecture** in `methodology/v2_5_paperA_inverse_choice.md` §3.2 (2-layer GAT, hidden=64, 4 heads).
- In the §4.4 ablation, **GraphSAGE** is the alternative GNN class we compare against (along with linear and MLP utility classes); GAT vs SAGE comparison localises the contribution of attention-based aggregation.
- We cite both Veličković 2018 and Hamilton et al. 2017 (GraphSAGE) so the ablation reads as a fair comparison of two mainstream GNN families.

**Quotable**:
> "GAT layers compute attention-weighted neighbour aggregations, allowing the model to adaptively focus on the most relevant neighbours" — paraphrased.

---

## 17. Wilson 1971 — Entropy-Maximising Doubly-Constrained Gravity

_(added per R5: missing related-work entry; underpins the multi-model triangulation baseline.)_

**Citation**: Wilson, A. G. (1971). "A family of spatial interaction models, and associated developments." *Environment and Planning A*, 3(1), 1–32.

**Summary**: Derives the doubly-constrained gravity model from an entropy-maximisation principle subject to row-sum (origin trip totals) and column-sum (destination trip totals) constraints. Established the family of singly-constrained / doubly-constrained / unconstrained gravity models that has dominated trip-distribution modelling for fifty years.

**Specific contribution**: Theoretical justification for the doubly-constrained gravity model as the maximum-entropy distribution consistent with marginal constraints.

**How Paper A uses**:
- **Multi-model triangulation baseline** (§5.5 of `methodology/v2_5_paperA_inverse_choice.md` and §6.4 of `paper/outline.md`): the doubly-constrained Wilson gravity is model (c) in our three-way comparison.
- Cited in `paper/preregistration.md` §H2 as the gravity-baseline against which we expect Scenario A to agree on **sign** (positive ΔY_·,OOC).

**Quotable**:
> "the doubly-constrained gravity flow distribution is the maximum-entropy distribution consistent with given row and column marginals" — paraphrased.

---

## Bibliography Format Note

For paper submission, format references in the venue-specific style. The notes here use loose Author-Year format as agreed. Cross-reference exact venue/style choice with `paper/outline.md` once venue is finalised (currently "NeurIPS Datasets & Benchmarks / KDD Applied Data Science / Transportation Research Part B" — three target venues with different reference styles).

---

*End related-work notes. Companions: `outline.md`, `preregistration.md`, `../methodology/v2_5_paperA_inverse_choice.md`.*
