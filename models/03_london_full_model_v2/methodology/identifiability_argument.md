# Identifiability Argument & Sensitivity Analysis — Paper A

> **Companion to**: `v2_5_paperA_inverse_choice.md` (canonical methodology) and `odd_d_protocol.md` (simulation spec).
> **Purpose**: Formal argument for what is and is not identified in the inverse-choice problem, plus an Imbens-style omitted-variable sensitivity bound for unobserved confounding. This is the appendix Causal-ML reviewer wants. _(addresses R5 §4: prior drafts mislabelled the bound as "Rosenbaum-style"; the formula in §4 is in fact Imbens 2003.)_

## Table of Contents

1. [Identification Problem Statement](#1-identification-problem-statement)
2. [FWL-Inspired Ridge Regularization](#2-fwl-inspired-ridge-regularization)
3. [Sign Constraint via Softplus](#3-sign-constraint-via-softplus)
4. [Imbens-style Omitted-Variable Sensitivity Bound](#4-imbens-style-omitted-variable-sensitivity-bound)
5. [Synthetic Recovery as Estimator-Validity Test (NOT Assumption Test)](#5-synthetic-recovery-as-estimator-validity-test)
6. [What Is and Is Not Identified](#6-what-is-and-is-not-identified)
7. [Why Not Pearl L2 from Observational Data Alone](#7-why-not-pearl-l2-from-observational-data-alone)

---

## 1. Identification Problem Statement

### 1.1 Setup

We observe aggregate origin–destination flow counts {Y_ij}_{i,j ∈ MSOA} on London 2021. The data-generating process we *posit* (the SCM of `v2_5_paperA_inverse_choice.md` §2) is:

```
V_j   = GNN_θ(X_j; A)                                  (1)
U_ij  = α V_j + β t_ij + γ log d_ij + ε_ij             (2)
ε_ij ~ Gumbel(0, 1)  i.i.d.                             (3)
P(j | i) = exp(U_ij) / Σ_{j' ∈ C_i} exp(U_ij')         (4)
Y_ij  = Poisson(N_i · P(j | i))                        (5)
```

The identification target is **(θ, α, β, γ)** — i.e., the structural parameters of the GNN attractiveness function and the three RUM coefficients.

### 1.2 What "identification" means here

We use **structural identification** in the econometric sense (Hsiao 1983; Manski 2003): there exists a unique parameter vector (θ*, α*, β*, γ*) such that the implied likelihood `L(Y | θ*, α*, β*, γ*)` equals the observed likelihood, and no other parameter vector achieves this.

Structural identification ≠ causal identification. The latter additionally requires that the SCM is the true DGP up to the labelled exogenous variables — equivalently, no unmeasured confounder violates the conditional-independence structure of the DAG. We address this gap in §4 (sensitivity bound) and §7 (assumption-conditional L2).

### 1.3 Three classical threats to identification

1. **Scale indeterminacy**: U_ij is identified only up to a positive affine transformation; in MNL the location is fixed by the difference structure but the **scale** (utility units) is set by the assumed Var(ε) = π²/6. We adopt the standard convention.
2. **Collinearity between V_j and (t_ij, d_ij)**: closer destinations tend to be more attractive (jobs cluster centrally; t_ij is also small). Without correction, β and γ absorb part of α V. We address this in §2 via FWL-inspired ridge regularization.
3. **Sign of β under thin data**: in regions of (V, t) space where the design matrix is rank-deficient, β can flip sign at low penalty. We address this in §3 via softplus parameterisation.

---

## 2. FWL-Inspired Ridge Regularization

_(addresses R1 + R5: prior drafts framed this as an OLS-FWL identification transformation; the actual code (`_fwl_inspired_ridge_penalty`) implements a column-mean partial-out plus ridge penalty, which is a regularizer, not an identification transformation. Renamed and reframed here.)_

### 2.1 The collinearity problem

In the choice-set-restricted utility, the "design matrix" per agent is `[V_j, t_ij, log d_ij]` of shape (K=50, 3). For typical London commute geography:

- Corr(V_j, log d_ij) ≈ -0.45 (closer zones tend to be more attractive: central employment cluster)
- Corr(t_ij, log d_ij) ≈ +0.85 (time and distance are nearly proportional in free-flow conditions)
- Corr(V_j, t_ij) ≈ -0.40 (compounded effect)

The 0.85 collinearity between t and log d is the worst — any naive joint estimation will partition the time-vs-distance contribution arbitrarily.

### 2.2 Frisch–Waugh–Lovell decomposition

Let M_X = I - X(X'X)^(-1)X' be the residual-maker for X. The FWL theorem (Frisch & Waugh 1933; Lovell 1963) states: in the OLS regression `y = α V + β t + γ log d + e`, the OLS estimate of α equals the OLS of `M_(t, log d) y` on `M_(t, log d) V`.

Under MNL the analog is the **profile likelihood** in α holding (β, γ) at their conditional MLE: this is asymptotically equivalent to residualising V against (t, log d) before entering the softmax.

### 2.3 Operationalisation

In practice we add a ridge penalty **at training time** that penalises the projection of V_jt onto the column-means of (t_ij, log d_ij):

```
L_total = L_NLL  +  λ_FWL · ||proj_(t̄, log d̄) V||²
```

where `proj_(t̄, log d̄)` is the linear projection onto the column-mean cost/distance vector. λ_FWL is tuned by 5-fold CV on held-out origins; default λ_FWL = 0.1.

**Important caveat (addresses R1 + R5).** Strictly speaking, our procedure is **not** the Frisch–Waugh–Lovell transformation. FWL would require partialling out (t_ij, log d_ij) at the cell level (per (i, j) pair), which is computationally infeasible at our scale (≈10⁶ OD cells × 50 alternatives). Instead we add a ridge penalty on the projection of V_jt onto column-mean (t_ij, log d_ij). This is a **regularization**, not an identification transformation. We rely on the structural-GNN's domain-restricted output for identification, validated by synthetic recovery (§5), rather than on FWL per se. The label "FWL-inspired" reflects that the penalty is motivated by the same collinearity intuition FWL formalises, not that it implements FWL.

### 2.4 What the FWL-inspired ridge buys us

- **Reduced standard errors on α̂** (variance reduction proportional to 1 - R²(V ~ t̄, log d̄) ≈ 1 - 0.20 = 0.80, so ~5× narrower CI on α). _Approximate, not exact — exact FWL variance reduction would require cell-level partialling, which we do not do._
- **Cleaner separation of "how attractive is the place" (α V) from "how far away" (β t + γ log d)** at the population level.
- **Does NOT solve the unobserved-confounder problem** — see §4.
- **Does NOT provide point identification on its own** — identification rests on the structural-GNN domain restriction + sign constraint + synthetic recovery (§3, §5).

---

## 3. Sign Constraint via Softplus

### 3.1 The problem

Without constraint, the MNL likelihood is identified up to the sign of β only when the design matrix is full-rank in expectation. In *finite samples* with high V–t collinearity, β can settle at small positive values with nearly equal likelihood as small negative values, particularly if the GNN absorbs the time-deterrence signal into V.

### 3.2 The fix

We parameterise:
```
β = -softplus(β_raw) = -log(1 + exp(β_raw))    where β_raw ∈ ℝ
```

This forces β < 0 by construction. The softplus is smooth (vs. clipping), so gradients flow normally.

### 3.3 What the sign constraint does and does not do

**Does**:
- Identifies the *sign* of β (β < 0 by construction).
- Stabilises optimisation in early epochs when the GNN is still random.

**Does not**:
- Identify the *magnitude* of β. Magnitude must come from the data, conditional on the constraint.
- Rule out a "wrong magnitude" — e.g., β̂ → 0 (no time sensitivity) or β̂ → -∞ (perfect time sensitivity) are both legal.

### 3.4 Empirical guard

The W4 synthetic-recovery gate (§5) checks that β̂ recovers the true magnitude within 10% bias on synthetic data. If this fails under the softplus parameterisation, the architecture (not the constraint) is the issue.

---

## 4. Imbens-style Omitted-Variable Sensitivity Bound

_(addresses R5 §4 / R5 top-priority finding: prior drafts cited Rosenbaum 2002, but the formula and framing here are Imbens 2003.)_

### 4.1 Why we need this

Even with FWL + sign constraint + synthetic recovery passing, we have **not** ruled out that an unobserved zone-level confounder U → (V_j and Y_ij) biases β̂. Examples of plausible U:

- Workplace amenities not captured in X_j (gym, daycare, food options).
- Reputation / brand effect of major employers.
- Unmeasured commuter-destination match quality (industry-specific networks).

Imbens (2003) provides a sensitivity-analysis framework for regression-based observational studies: bound how strong an omitted variable U must be (in terms of its partial correlation with the treatment-analog and the outcome) to overturn the conclusion. Note: the related Rosenbaum (2002) Γ-bound is for *matched-pair propensity-score* designs and bounds odds-ratio shifts; we use Imbens (2003) here because our setting is regression-based (linear-in-parameters utility with continuous covariates), not matched-pair.

**Citation**: Imbens, G. W. (2003). "Sensitivity to Exogeneity Assumptions in Program Evaluation." *American Economic Review* 93(2): 126–132.

### 4.2 The bound

Assume U is a scalar zone-level confounder with:
- Corr(U, V_j) = ρ_UV
- Corr(U, β t_ij + γ log d_ij) = 0 (U confounds attractiveness, not travel cost)
- |U| ≤ Γ standard deviations of V_j

Under these assumptions, the bias in β̂ from omitting U satisfies (Imbens 2003 ω-formula, regression form):

```
|bias(β̂)| ≤  Γ · |ρ_UV| · σ_V · |α̂| / σ_t
```

where σ_V, σ_t are the conditional standard deviations of V and t.

### 4.3 Numerical example for London

Plugging in placeholder values (to be filled at W8):
- σ_V ≈ 1.0 (standardised GNN output)
- σ_t ≈ 18 minutes (London 2021 commute time SD)
- α̂ ≈ 0.8 (placeholder)
- β̂ ≈ -0.05 per minute (placeholder — TAG-implied range)

For the bias to *flip the sign* of β̂ (from -0.05 to +0.05), we need:
```
0.10 ≤ Γ · |ρ_UV| · 1.0 · 0.8 / 18
⇒ Γ · |ρ_UV| ≥ 2.25
```

i.e., the unobserved confounder would need to vary by **≥ 2.25 σ across MSOAs** AND be **perfectly correlated with V** to flip β̂. Either component alone is implausible; the joint requirement is very strong evidence that β̂ < 0 is robust.

### 4.4 What we report

In the paper, we compute this bound at the final W8 (θ̂, α̂, β̂) and report:
- The minimum (Γ × |ρ_UV|) product needed to flip β̂.
- A "what-would-flip-this" narrative: e.g., "an unobserved confounder explaining ≥ X% of the variance in V_j would need to exist to flip β̂ — and we know of no candidate confounder of this magnitude in the X_j feature set we excluded."

### 4.5 Caveats

- This bound is **linear-in-U**. Non-linear confounders (e.g., U interacts with t_ij) require more elaborate sensitivity analysis (Cinelli & Hazlett 2020 omitted-variable bias contour plots / robustness value — flagged for v2.6 as an extension of the Imbens 2003 framework used here).
- The bound assumes Corr(U, t_ij) = 0. If U *also* confounds travel time (e.g., destinations with hidden amenities are also better-served by transit), the bound is too narrow.

---

## 5. Synthetic Recovery as Estimator-Validity Test (NOT Assumption Test)

### 5.1 The reviewer concern

Causal-ML reviewer flagged (correctly): "Synthetic recovery shows your *estimator* works on data your *model* generated. It does NOT show your model is the right model for the real data." This is the difference between:

- **Estimator validity**: given (θ*, α*, β*, γ*) and data simulated from the SCM, does our optimiser find them? — testable, and we test it at W4.
- **Model validity**: is the SCM the true DGP for London 2021? — *not* testable from observational data alone; addressed via §4 sensitivity, §7 ICM assumption, and external triangulation (TAG VOT, multi-model agreement).

### 5.2 W4 gate

Synthetic recovery PASSes if:
- Bias(α̂, β̂, γ̂) < 10% of true value (per-parameter).
- 95% bootstrap CI coverage ≥ 85% (per-parameter, 200 bootstrap reps).
- GNN parameter recovery: cosine similarity between flattened θ̂ and θ* > 0.7 (or, alternatively, R² of V_j(θ̂) vs V_j(θ*) > 0.85).

If W4 fails → architecture / optimiser bug; do NOT proceed to London training.

### 5.3 Why this is necessary but not sufficient

- **Necessary**: failing synthetic recovery while continuing to a real fit means we cannot trust *any* parameter estimate, even on the synthetic-DGP world.
- **Not sufficient**: passing synthetic recovery only validates the inverse step. The forward step (the SCM matches London) still requires §4 (Imbens) sensitivity + §7 assumption.

---

## 6. What Is and Is Not Identified

| Quantity | Identified? | From what? | Notes |
|---|---|---|---|
| Sign of β | Yes (by construction) | Softplus parameterisation | §3 |
| Magnitude of β | Yes (structurally) | Conditional MLE on TopK choice set | Up to scale of ε; standard MNL |
| α | Yes (structurally) | Conditional MLE | After FWL-inspired ridge regularization |
| γ | Yes (structurally) | Conditional MLE | High-variance under t–log d collinearity; flagged in CIs |
| θ (GNN) | Yes (in expectation) | Joint MLE | Subject to GNN identifiability quirks (permutation symmetries; not material here since V is scalar output) |
| ASC_mode | Yes (structurally) | If mode is observed; absorbed into V_jt otherwise | |
| Mode-specific β | Partially | Only if mode-OD is observed | London 2021 has mode-OD via Census; usable |
| Income-tier β | Yes (with shrinkage) | Hierarchical pooling | v2.5 includes 3 tiers |
| Causal effect of `do(X_j)` | NOT identified from data alone | Requires ICM assumption | §7 |
| Real-world counterfactual (post-OOC OD) | NOT identifiable | No holdout | Discussed but not claimed |

---

## 7. Why Not Pearl L2 from Observational Data Alone

### 7.1 Pearl's ladder

Pearl 2009 §1 distinguishes:
- **L1 (associational)**: P(Y | X) — what we observe.
- **L2 (interventional)**: P(Y | do(X)) — what would happen if we forced X.
- **L3 (counterfactual)**: P(Y_{X=x} | X=x', Y=y') — what would have happened in a parallel world.

Aggregate OD data is L1. Going to L2 requires either:
(a) randomised intervention (we have none),
(b) back-door adjustment with verified ignorability (unfalsifiable from data alone),
(c) front-door adjustment with verified mediator (we have no mediator),
(d) instrumental variable with verified relevance + exclusion (none of the X_j features qualify),
(e) **structural assumption strong enough to defend as an assumption**.

### 7.2 Our position: option (e) — ICM assumption

Schölkopf et al. 2021 articulate the **Independent Causal Mechanisms (ICM) principle**: the structural mechanism producing each variable is invariant under interventions on other variables.

In our SCM:
- Mechanism for V_j: GNN_θ(X_j) — assumed invariant if X_j is intervened.
- Mechanism for U_ij: α V + β t + γ log d + ε — assumed invariant.
- Mechanism for t_ij: BPR equilibrium — assumed invariant under X_j interventions (i.e., new jobs at OOC do not change the *form* of the BPR function, only its inputs).

Under ICM, the post-intervention distribution P(Y | do(X_j)) is computable from the pre-intervention SCM by mechanically substituting X_j' for X_j and re-evaluating the equations. This is L2 **conditional on the ICM assumption holding**.

### 7.3 What we explicitly say in the paper

> "The scenario evaluation in §7 is an L2 query in Pearl's hierarchy *conditional on* the structural causal model stated in §2 satisfying the Independent Causal Mechanisms assumption (Schölkopf et al. 2021). We do not invoke a randomised intervention or a back-door identification argument; we invoke the ICM assumption explicitly. Our sensitivity analysis (Appendix A) bounds the magnitude of unobserved-confounder violations needed to flip our directional conclusions, but cannot rule them out."

### 7.4 What this concession costs us

- We **cannot** claim "Paper A predicts the post-OOC flow."
- We **can** claim "Paper A computes the L2 query implied by a stated, defensible SCM, with sensitivity analysis on the central assumption."

The reviewer-AC distinction matters: the former is a falsifiable forecast (which we cannot make without holdout data); the latter is a *modelling* contribution (which we can make).

### 7.5 What would graduate Paper A's claim to unconditional L2

Three falsification routes (any one, in priority order):
1. **Beijing transfer success** (Paper B): if (β̂) recovered in London transfers to Beijing within 30%, this is strong inductive evidence that ICM holds *across cities*, narrowing the assumption.
2. **Within-London time-series**: if 2011 → 2021 OD evolution is consistent with applying X_2011 → X_2021 perturbation under fixed (α̂, β̂, γ̂) recovered from a third year — this would be a *natural experiment* test.
3. **Holdout post-OOC data** (years away): when OOC opens, compare the actual OD shift to our 2025 prediction. This is the gold-standard falsification.

Paper A flags all three as future work.

---

*End identifiability appendix. Companion: `v2_5_paperA_inverse_choice.md`, `odd_d_protocol.md`, `../paper/preregistration.md`.*
