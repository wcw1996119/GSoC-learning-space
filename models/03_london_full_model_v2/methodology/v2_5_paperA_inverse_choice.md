# v2.5 Paper A — Differentiable Inverse Choice Learning at Urban Scale

> **Status**: Canonical Paper A methodology. Supersedes `00_v2_architecture.md` for the Paper A submission scope.
> **Title direction (validator-approved)**: *Differentiable Inverse Choice Learning at Urban Scale: Recovering Behavioral Parameters and a Structural Utility Mechanism from Aggregate OD Flows*
> **Note**: "counterfactual" is intentionally REMOVED from the title per Area Chair refinement. The paper claims **L2 conditional on a stated SCM**, not "L1.5" and not unconditional L2.

## Table of Contents

0. [Prerequisites — Pre-W1 Skill Ramp (3–4 weeks)](#0-prerequisites--pre-w1-skill-ramp)
1. [Goal & Framing](#1-goal--framing)
2. [SCM Specification](#2-scm-specification)
3. [Architecture](#3-architecture)
4. [Inverse Training](#4-inverse-training)
5. [Validation Regime](#5-validation-regime)
6. [Scenario A Protocol](#6-scenario-a-protocol)
7. [Pre-registration Commitment](#7-pre-registration-commitment)
8. [Limitations](#8-limitations)
9. [Beijing Transfer Plan](#9-beijing-transfer-plan)
10. [Milestone Gates](#10-milestone-gates)
11. [Cut-or-Deliver Protocol](#11-cut-or-deliver-protocol)

---

## 0. Prerequisites — Pre-W1 Skill Ramp

_(addresses R5 / refinement #14: the strategy advisor flagged that none of the 6 docs described a 3–4 week pre-W1 skill ramp. This section makes that prerequisite explicit before the W1 milestone clock starts.)_

Before week W1 begins (i.e., before the 12-week W1–W12 schedule in §10), the author commits a 3–4 week skill-ramp covering the technical prerequisites for the methods used in this paper. Without this ramp, the W4 synthetic-recovery gate is at high risk regardless of how clean the spec is.

### 0.1 Implicit differentiation toolkit (≈1.5 weeks)

- **Implicit function theorem** (Krantz & Parks 2002): the mathematical foundation for differentiating through fixed points and argmin operators.
- **`torch.autograd.Function` custom backward**: how PyTorch lets you specify a backward pass manually rather than tracing through the forward.
- **Conjugate gradients (CG)**: linear-system solver used inside implicit-diff backward passes (avoids forming the Jacobian explicitly).
- **OptNet** (Amos & Kolter 2017): differentiable QP layer; original concrete instantiation of implicit-diff in deep learning.
- **jaxopt** (Blondel et al. 2022): the modular implicit-diff library this paper actually uses (`FixedPointIteration`, `implicit_diff=True`). Goal: be comfortable enough to debug a backward pass that diverges.

### 0.2 Discrete choice fundamentals (≈1 week)

- **Train (2009)** *Discrete Choice Methods with Simulation*, **chapter 4** (MNL identification, IIA, scale parameter) and **chapter 6** (nested logit, dissimilarity parameter, GEV class).
- Replicate **one Apollo nested-logit example** end-to-end (Hess & Palma 2019; Apollo R package). This is a forcing function for understanding the nested logit's identification quirks before we commit to it as a stretch goal in §3.5.

### 0.3 Compute infrastructure (≈0.5–1 week)

- **PACE GPU access**: queue submission, A100 allocation, debugging GPU OOM.
- **Checkpointing**: `torch.save`/`torch.load` round-trip for training resumption; jaxopt fixed-point state serialisation.
- **Conda env reproducibility**: `conda env export --no-builds`, lockfiles, pinning CUDA/cuDNN versions; ensure W12 reproducibility hash matches W1 environment hash.

### 0.4 Why this is non-negotiable

The W4 milestone in §10 assumes implicit-diff and discrete-choice fundamentals are already fluent. Skipping the ramp would push the architecture-debug load *into* W1–W4, and the synthetic-recovery gate (bias < 10%, coverage ≥ 85%) is unforgiving when the underlying solver is not understood.

---

## 1. Goal & Framing

### 1.1 What this paper claims

Paper A demonstrates that **aggregate origin–destination (OD) flow data alone identifies a coherent set of structurally-identified parameters** — a value-of-time coefficient β, a distance-deterrence γ, an attractiveness-weighting α, and the parameters θ of a Graph Neural Network (GNN) that maps zone features to a structural attractiveness scalar V_jt — when those behaviorally-grounded structural parameters are embedded in a Random Utility Model (RUM) and trained by **implicit differentiation through the softmax choice probabilities**. _(addresses R5 / refinement #10: "primitives" replaced by "structurally-identified parameters" / "behaviorally-grounded structural parameters" throughout.)_

This is a **structural-parameter recovery** claim, not a "counterfactual prediction" claim:

- **Recovery**: given observed OD flows {y_ij}, we recover (θ̂, α̂, β̂, γ̂) such that the RUM-implied flows match {y_ij} and the recovered parameters are consistent with prior estimates from stated-preference surveys (DfT TAG A1.3 VOT) and revealed-preference literature (Train 2009 elasticity ranges).
- **Scenario simulation under recovered parameters** (NOT counterfactual prediction in the unconditional sense): given an intervention `do(employment_OOC += 65k)` on the Old Oak Common node, we forward-evaluate the GNN and re-solve the choice equilibrium with frozen (θ̂, α̂, β̂, γ̂). The resulting flow change is a **L2 query conditional on the stated SCM** (§2). It is NOT a real-world counterfactual — no holdout natural experiment is invoked.

### 1.2 Why naming matters

Per Causal-ML reviewer refinement: any "counterfactual prediction" framing of an observational ABM/IRL pipeline trades on Pearl's L3 ladder rung that requires either (a) a randomised intervention, (b) front-door/back-door identification with verified ignorability, or (c) a structural assumption strong enough to be defensible *as an assumption*. We adopt option (c) and explicitly state the SCM (§2) and the invariance conditions (Schölkopf 2021 ICM principle) under which the recovered parameters extrapolate to the intervened world.

Therefore:
- **Title** says "scenario simulation" / "structural utility mechanism" — not "counterfactual".
- **Abstract** says "scenario simulation under recovered structural parameters" — never "counterfactual prediction".
- **Discussion** explicitly distinguishes "structurally identified" (assumption-conditional) from "causally identified" (assumption-light or randomisation-based).

### 1.3 Why aggregate OD identifies the parameters at all

The intuition is McFadden 1978: under an i.i.d. Gumbel utility shock, the conditional choice probability has a closed-form softmax over alternative-specific utilities. Aggregating Poisson-distributed individual choices to OD counts is invertible (up to a normalising constant) when the utility function is linear in observed covariates and the choice set is fixed. We extend this in three directions:

1. **GNN-parameterised attractiveness** V_jt = GNN_θ(X_j): the attractiveness term is a learned non-linear function of zone features rather than a hand-coded `log(employment_j)`. θ is identified jointly with (α, β, γ).
2. **Top-K=50 sampling-of-alternatives** (McFadden 1978 §4): the full London choice set is ~983 MSOAs; we restrict each commuter's choice set to their 50 nearest-by-time alternatives plus their observed destination. This preserves consistency of the MLE under standard regularity conditions.
3. **Implicit differentiation** (Amos & Kolter 2017 OptNet; Blondel et al. 2022 jaxopt): the equilibrium choice probabilities solve a fixed-point in the presence of congestion feedback (mean t_ij depends on the mean choice). We differentiate through the fixed point rather than through an unrolled iteration, giving exact gradients at machine precision.

---

## 2. SCM Specification

### 2.1 Variables

| Symbol | Type | Domain | Source |
|---|---|---|---|
| X_j | exogenous | ℝ^d | Zone features: employment count, employment density, sectoral mix, retail floor area, transit accessibility, land-use entropy |
| t_ij | endogenous | ℝ_+ | Travel time (minutes) home i → work j; depends on mode share and congestion |
| d_ij | exogenous | ℝ_+ | Network distance (km) — fixed by geography |
| C_ij | endogenous | ℝ_+ | Congestion ratio on link ij (BPR equilibrium) |
| V_jt | endogenous | ℝ | Attractiveness scalar for zone j at hour t; structural function V_jt = GNN_θ(X_j) |
| U_ijt | endogenous | ℝ | Random utility of commuter at i choosing destination j at hour t |
| ε_ij | exogenous | i.i.d. Gumbel(0, 1) | Idiosyncratic taste shock |
| Y_ij | observed | ℕ | Observed annual OD flow count |
| do(X_j) | intervention | ℝ^d | Counterfactual zone feature value — the only handle we operate on |

### 2.2 Structural equations

```
X_j         := exogenous (or do(X_j) under intervention)
d_ij        := exogenous (geography)
V_jt        := GNN_θ(X_j, A_graph)              # graph A is fixed across scenarios
t_ij        := BPR-equilibrium(mode_share, C_ij) # solved jointly with choice
U_ijt       := α · V_jt + β · t_ij + γ · log(d_ij) + ε_ij
P(j | i, t) := softmax over j' ∈ TopK(i)  of U_ij't
Y_ij        := Σ_t Poisson(N_i · P(j | i, t))
```

with sign constraints:
- α > 0 (more attractive zones are more chosen)
- β < 0 (longer travel time deters; encoded via softplus reparameterisation, see §4.3)
- γ < 0 (longer distance deters)

### 2.3 DAG (ASCII)

```
                    ┌──────────┐
                    │  X_j     │ (zone features; do() target)
                    └────┬─────┘
                         │ GNN_θ
                         ▼
                    ┌──────────┐
        ┌──────────►│  V_jt    │
        │           └────┬─────┘
        │                │ α
   ┌────┴─────┐          ▼
   │ d_ij     │──γ──►┌────────┐
   └──────────┘      │ U_ijt  │◄──ε_ij  (Gumbel)
   ┌──────────┐      └───┬────┘
   │ t_ij     │──β──────►│
   └────▲─────┘          │ softmax / TopK
        │                ▼
        │           ┌──────────┐
        │           │ P(j|i,t) │
        │           └────┬─────┘
        │                │ Poisson · N_i
        │                ▼
        │           ┌──────────┐
        │           │  Y_ij    │ (observed)
        │           └────┬─────┘
        │                │ aggregates → mode share
        └────────────────┘  (BPR feedback loop)
```

### 2.4 Intervention nodes

The only intervention we evaluate in Paper A is `do(X_OOC)` — additively perturbing the Old Oak Common employment count by +65k jobs (Crossrail HS2 interchange business case). All downstream variables (V_jt, t_ij, P, Y) are re-solved under the frozen recovered parameters (θ̂, α̂, β̂, γ̂).

### 2.5 Why this is L2 conditional, not unconditional L2

To push observational OD data to L2 unconditionally, we would need to verify:
(i) no unobserved confounder U → (V_j, t_ij, Y_ij);
(ii) the structural equations are correctly specified;
(iii) the GNN class is rich enough that V_jt = GNN_θ̂(X_j) coincides with the true attractiveness in the post-intervention world.

We can defend (ii) and (iii) by ablation and synthetic recovery (§5), but (i) is unfalsifiable from observational data alone. We therefore *assume* (i) — equivalently, the Independent Causal Mechanisms assumption (Schölkopf et al. 2021): each structural mechanism is invariant under interventions on other mechanisms — and report an Imbens-style omitted-variable sensitivity bound (Imbens 2003) in `identifiability_argument.md` for how strong an unobserved confounder would need to be to flip our conclusions.

---

## 3. Architecture

### 3.1 Overview

```
        X_j (∈ ℝ^d, d=18)
            │
            ▼
   ┌──────────────────────┐
   │  GNN_θ (2-layer GAT) │   ← structural function for attractiveness
   │  hidden = 64         │
   └──────────┬───────────┘
              │
              ▼  V_jt ∈ ℝ
              │
   ┌──────────────────────┐
   │  RUM utility:        │
   │  U_ij = α V_jt       │
   │      + β t_ij        │   ← (α, β, γ) are scalar globals (or income-tier in v2.6)
   │      + γ log d_ij    │
   │      + ε             │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Top-K=50 softmax    │
   │  P(j|i) on K alts    │
   └──────────┬───────────┘
              │
              ▼  fit to observed OD via NLL
   ┌──────────────────────┐
   │  Implicit-diff ∇θ,α,β,γ
   └──────────────────────┘
```

### 3.2 GNN_θ as structural function

We use a 2-layer GAT (Veličković et al. 2018) with hidden width 64 and 4 attention heads. The graph A is a 1km × 1km grid adjacency over Greater London (≈983 MSOAs in the simpler aggregation, or 8.6k 1km cells in the dense aggregation; v2.5 uses MSOAs for tractability). Node features X_j (d = 18) include:

- log(employment_j), employment_density_j
- sectoral employment shares (4 broad SOC groups)
- retail floor area, office floor area
- log(population), median income
- transit accessibility index, road network density
- land-use entropy (Shannon, 5 land uses)

Output V_jt is a scalar per zone per hour. Time variation enters via hour-specific bias terms; X_j itself is annual.

### 3.3 RUM utility

```
U_ij = α · V_jt(θ) + softplus_neg(β_raw) · t_ij + γ · log(d_ij) + ε_ij
```

`softplus_neg(x) = -log(1 + exp(x))` ensures β < 0 (see §4.3).

### 3.4 Top-K choice set construction

For each origin i, the choice set C_i = {top-50 destinations by t_ij} ∪ {observed_destination_i_in_training_data}. This is the McFadden 1978 sampling-of-alternatives correction; the conditional MLE remains consistent provided the sampling probability is independent of the choice. We sample by minimum-time rather than uniformly, matching the empirical practice in Ben-Akiva & Lerman 1985 §9.

### 3.5 Nested logit (stretch goal)

If time permits before W8: replace the flat MNL with a nested logit where the upper nest is workplace MSOA and the lower nest is mode {car, PT, active}. The dissimilarity parameter λ is identified separately from β under standard regularity (Train 2009 ch.4). This is **a stretch goal**, dropped first under §11 cut protocol.

---

## 4. Inverse Training

### 4.1 Loss

```
L(θ, α, β, γ) = -Σ_{(i,j) ∈ obs} y_ij · log P_θαβγ(j | i)  +  λ · ||θ||²
```

i.e. NLL of observed OD counts under the RUM-implied multinomial. y_ij is the integer flow count; total commuters per origin N_i = Σ_j y_ij is treated as exogenous (we model destination conditional on departure, not departure rate).

### 4.2 Implicit differentiation through the softmax fixed point

In the static (no-congestion) version, P_θαβγ(j|i) is a closed-form softmax — gradients are direct. Once we add BPR congestion feedback (mean t_ij depends on the mean P), the equilibrium is the fixed point of:

```
t_ij^(k+1) = BPR(volume(P^(k)))
P^(k+1)    = softmax(α V + β t_ij^(k+1) + γ log d)
```

We use jaxopt's `FixedPointIteration` + `custom_vjp` (Blondel et al. 2022) to differentiate through the converged fixed point in O(K) memory regardless of iteration count. This avoids the unrolled-backprop memory blow-up and gives exact gradients (up to convergence tolerance).

### 4.3 Sign constraint via softplus reparameterisation

Identifying the *sign* of β is non-trivial under high collinearity between V_j and t_ij (close zones tend to be more attractive in any reasonable city). We enforce β < 0 by training `β_raw ∈ ℝ` and using `β = -softplus(β_raw)`. This identifies the sign by construction; the *magnitude* is then identified by the data via the conditional likelihood. See `identifiability_argument.md` §3 for the formal argument.

### 4.4 FWL-inspired ridge regularization for identifiability

_(addresses R1 + R5: prior drafts described this as OLS-FWL residualisation; the implemented code is a column-mean partial-out plus ridge penalty.)_

In the linear-in-(α, β, γ) part of the utility, V_jt and t_ij are partially collinear (longer travel time tends to correlate with less attractive zones). To reduce variance of (α̂, β̂), we add a ridge penalty on the projection of V_jt onto the column-mean cost/distance vector (`_fwl_inspired_ridge_penalty` in code). This is **not** the Frisch–Waugh–Lovell transformation — exact FWL would require cell-level partialling, infeasible at our scale — but is motivated by the same collinearity intuition.

Identifiability in this paper is therefore supported by:
(a) **ridge regularization** on V_jt's projection onto cost/distance column means [code: `_fwl_inspired_ridge_penalty`] — a variance-reducer, not an identification step;
(b) **softplus sign constraint** on β_t (§4.3) — identifies the sign of β by construction;
(c) **synthetic recovery validation** (§5.1, W4 gate) — empirical check that the estimator recovers (α*, β*, γ*, θ*) on data simulated from the SCM.

See `identifiability_argument.md` §2 for the full argument.

### 4.5 Optimisation

- Optimiser: Adam(lr=1e-3, β1=0.9, β2=0.999)
- Batch: 512 origins per step
- Epochs: 50 (early stop on validation NLL)
- Hardware: 1× A100 40GB (university shared cluster)
- Wall-clock: ~6 hours for London 2021

---

## 5. Validation Regime

Five-tier validation, ordered by strictness:

### 5.1 Synthetic recovery (W4 deliverable)

Generate synthetic OD with known (θ*, α*, β*, γ*); train inverse model; check (θ̂, α̂, β̂, γ̂) recovers the truth within 10% bias and 85% coverage of 95% bootstrap CIs. **This is the W4 milestone gate.** See §10.

This validates the *estimator*, not the *assumption* — per Causal-ML reviewer. If synthetic recovery fails, the architecture is misspecified. If it passes, we still must defend the SCM assumption separately (§2.5, identifiability appendix §4 Imbens-style omitted-variable bound).

### 5.2 VOT vs DfT TAG A1.3

Implied value-of-time = -β / β_cost (where β_cost is the marginal utility of money; we use the income-tier reparameterisation to back this out). Compare to UK DfT TAG Unit A1.3 (commuting VOT ~£11–14/hr in 2021 prices). If our recovered VOT lies within the TAG bracket, this is converging external evidence (NOT proof — TAG itself is an estimate).

### 5.3 100-draw multiverse (Steegen et al. 2016, structural-prior variant)

_(addresses R4 doc-code mismatch: aligned with the actual `validation/paper_a/multiverse_runner.py`, which samples continuous Gaussian priors over the structural parameters. Pre-reg detail in `paper/preregistration.md` §4.)_

Sample N_DRAWS = 100 continuous prior draws (seed = 42) over:
- β_t ~ Normal(-0.07, 0.02²)
- β_c ~ Normal(-1.0, 0.2²)
- α_V ~ Normal(1.0, 0.25²)
- γ ~ Normal(-1.5, 0.4²)
- K_choice ~ DiscreteUniform({25, 50, 100, 200})

Report 90% percentile interval of recovered β across the 100 runs. If the interval excludes zero AND includes the TAG bracket, this is robustness evidence.

### 5.4 Literature elasticity bracketing

For each recovered parameter, check it falls within the bracket of prior published estimates compiled in `validation/paper_a/literature_elasticity_db.py` (Train 2009 ch.6 tables; Small & Verhoef 2007; Cervero 2002; etc.). This is bracketing not a point comparison.

### 5.5 Multi-model triangulation

Report (θ̂, β̂) from three model classes on the same data:
- (a) the proposed differentiable inverse model
- (b) a standard MNL with hand-coded `log(employment_j)` (no GNN)
- (c) a doubly-constrained gravity model (Wilson 1971)

If (a) and (b) agree on β within 20% but disagree with (c), this localises the GNN's contribution to the V_jt term, not to β. The paper's claim is precisely this localisation.

---

## 6. Scenario A Protocol

### 6.1 Intervention

```
do(X_OOC.employment += 65k jobs)
```

motivated by HS2/Crossrail Old Oak Common business case (DfT 2018 Outline Business Case). All other zones' X_j are held fixed.

### 6.2 Forward evaluation under frozen parameters

```
1. Freeze (θ̂, α̂, β̂, γ̂) from the W8 London fit
2. Recompute V'_jt = GNN_θ̂(X' )   for all j (note: GNN is graph-aware,
                                              so X_OOC perturbation
                                              propagates to neighbours)
3. Re-solve the BPR-equilibrium fixed point with the new V'
4. Read off ΔP(j | i) and ΔY_ij
```

### 6.3 What we report

- Top 20 origin MSOAs by ΔY_i,OOC
- Aggregate mode share shift on the Acton–Park Royal corridor
- 90% multiverse CI on the predicted ΔY_OOC
- **Comparison vs gravity-model baseline** (claim: same direction, GNN provides finer spatial structure)

### 6.4 What we do NOT claim

- We do NOT claim this predicts the actual real-world post-OOC flow. No holdout natural experiment is invoked.
- We DO claim this is the L2 query implied by the stated SCM, evaluated at the recovered parameters. Any deviation from real outcomes, when OOC opens in 2030+, is attributable either to (a) failure of the ICM assumption, (b) parameter drift, or (c) action mis-specification — and we cannot disentangle these without the actual data.

---

## 7. Pre-registration Commitment

Before running any scenario evaluation, we freeze:

- Model architecture (commit hash placeholder in `paper/preregistration.md`)
- Loss function and optimisation hyperparameters
- Multiverse grid (100 specific combinations)
- Literature elasticity bracket ranges
- Pass/fail thresholds for each hypothesis (H1–H3)
- Stopping rules (W4 / W8 gates)

See `paper/preregistration.md` for the OSF-style freeze. Any post-hoc deviation from the pre-registration is reported in the paper's "Deviations from Pre-registration" section (Nosek et al. 2018).

---

## 8. Limitations

### 8.1 Single city, single year

The recovery is on London 2021 only. We cannot test cross-city or cross-time invariance of (α, β, γ) within Paper A. This is the explicit motivation for Paper B (Beijing transfer; §9).

### 8.2 No real-world counterfactual

We have no holdout post-OOC OD to validate against. Scenario A is L2-conditional, not L2-validated.

### 8.3 Structurally-identified parameters are *structurally identified*, not *causally identified*

Per Causal-ML reviewer: structural identification (parameters are uniquely determined by the model + likelihood) is necessary but not sufficient for causal identification (parameters extrapolate to interventions). The latter requires the ICM/invariance assumption (§2.5), which is unfalsifiable from observational data.

### 8.4 Static graph

The GNN graph A is fixed. We do not model network expansion (e.g., new transit lines as graph rewiring); the OOC scenario only perturbs node features, not edges. This is a deliberate scope cut.

### 8.5 Income heterogeneity is coarse

v2.5 uses 3 income tiers for β, not full Axtell-style continuous heterogeneity. Tier definition is by ONS LSOA mean equivalised income terciles. This is deferred to v2.6 / Paper B.

### 8.6 No dynamic learning / adaptation

ODD+D §2.2: agents do not learn or adapt within a simulation. The choice probabilities are equilibrium probabilities, not converged behaviour from iterative learning.

### 8.7 Mode set is fixed at {car, PT, active}

No micromobility, no rideshare, no work-from-home elasticity. WFH is a known confounder for 2021 London data; we report sensitivity to WFH fraction in the multiverse.

---

## 9. Beijing Transfer Plan (forward reference to Paper B)

Paper B will test whether (α, β, γ) recovered in London transfer to Beijing 2020 OD. Three regimes:

1. **Frozen-θ frozen-β**: pure transfer; failure tells us nothing about the ICM assumption (could be data heterogeneity).
2. **Frozen-θ refit-β**: tests whether the GNN attractiveness function generalises.
3. **Refit-θ refit-β**: tests whether the architecture (not parameters) generalises.

A successful (β consistent across cities within 30%) is strong evidence for the ICM assumption; a failure is evidence against (or against the architecture). Paper B is out of scope for Paper A but the Paper A discussion (§8 of Paper A) flags this as the next falsification step.

---

## 10. Milestone Gates

| Week | Deliverable | PASS criterion | NO-GO action |
|---|---|---|---|
| W4 | Synthetic recovery experiments | bias < 10% AND CI coverage ≥ 85% on (α, β, γ) | Reframe Paper A as "estimator development" methods paper; drop scenarios entirely |
| W8 | London 2021 OD fit | CPC (Common Part of Commuters, Lenormand 2016) ≥ 0.5 on held-out 20% origins | Drop Scenario A; ship as "London structural recovery" paper without scenarios |
| W10 | Multiverse + VOT-TAG | 90% CI excludes zero AND VOT in TAG bracket | Report negative findings honestly; reframe as "what aggregate OD does NOT identify" |
| W12 | Final paper draft | All §5 validations pass | Submit short paper variant |

---

## 11. Cut-or-Deliver Protocol

If we fall behind schedule, drop in this order (most cuttable first):

1. **Drop nested logit** (§3.5) — keep flat MNL. Discussion notes this as future work.
2. **Drop multi-model triangulation** (§5.5 model (c) gravity comparison). Keep (a) and (b) only — preserves the "GNN contribution localisation" claim.
3. **Drop Scenario A** entirely. Reframe as a structural-recovery paper without scenarios. Title becomes "Recovering Behavioral Parameters from Aggregate OD via Differentiable Inverse Choice".
4. **Drop multiverse**, keep point estimates with bootstrap CIs only.
5. **Drop GNN entirely**, fall back to MNL with hand-coded V_j. At this point Paper A is no longer novel; this is the abandon-and-write-Paper-B protocol.

The W4 synthetic recovery gate is the hardest line: if it fails, **we do not proceed to London training** — we go back to architecture diagnosis. Failing synthetic recovery while continuing to a real-data fit would be scientific malpractice (we'd be fitting noise with a known-broken estimator).

---

*End of v2.5 Paper A methodology spec. See companion files: `odd_d_protocol.md`, `identifiability_argument.md`, `../paper/preregistration.md`, `../paper/outline.md`, `../paper/related_work_notes.md`.*
