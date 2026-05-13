# Pre-registration — Paper A (Differentiable Inverse Choice Learning)

> **Style**: OSF-style pre-registration. Frozen *before* running London 2021 fit and any scenario evaluation.
> **Commit hash freeze**: `<COMMIT_HASH_PLACEHOLDER — fill at W4 freeze>`
> **Date frozen**: `<DATE_PLACEHOLDER — fill at W4 freeze>`
> **Authors**: Chengwei Wang (sole author for Paper A v1)

## Table of Contents

1. [Hypotheses](#1-hypotheses)
2. [Sample](#2-sample)
3. [Method (frozen)](#3-method-frozen)
4. [Multiverse Parameter Grid (100 combinations)](#4-multiverse-parameter-grid)
5. [Literature Elasticity Ranges](#5-literature-elasticity-ranges)
6. [Pass/Fail Criteria per Hypothesis](#6-passfail-criteria-per-hypothesis)
7. [Stopping Rules](#7-stopping-rules)

---

## 1. Hypotheses

We register three primary hypotheses. All are pre-specified before viewing any London 2021 result.

### H1 — Recovered β consistent with TAG VOT

**Statement**: The recovered coefficient β̂ on travel time, expressed as an implied value-of-time (VOT) per income tier, lies within the bracket published in DfT TAG Unit A1.3 (commuting VOT, 2021 prices, ±25% tolerance).

**Direction**: two-sided test (β̂ could come in too small or too large; both are interesting failures).

**Rationale**: TAG values are independently estimated from stated-preference surveys. Convergence of revealed-preference β̂ from aggregate OD with stated-preference TAG is meaningful triangulation.

### H2 — Scenario A direction matches gravity baseline

**Statement**: Under `do(X_OOC.employment += 65k jobs)`, the recovered model predicts a **positive ΔY_·,OOC** (more flows into OOC) with the **same sign** as a doubly-constrained gravity model (Wilson 1971) given the same intervention.

**Direction**: directional test (we expect agreement on sign, not necessarily magnitude).

**Rationale**: a sign disagreement with the gravity baseline would suggest the GNN is doing something semantically different from "places with more jobs attract more commuters" — which would be an alarming finding requiring separate diagnosis.

### H3 — Multiverse 90% CI excludes zero for β̂

**Statement**: Across the 100-combination multiverse (§4), the 90% percentile interval of recovered β̂ excludes zero.

**Direction**: one-sided (we pre-commit to expecting β̂ < 0; H3 is rejected if the interval crosses zero).

**Rationale**: robustness to modelling-degree-of-freedom variation. If β̂ flips sign or hits zero under reasonable variations of the multiverse axes, the recovery is not robust.

---

## 2. Sample

### 2.1 Population

**London 2021 working population**: ≈ 3.05M commuters from Census 2021 ODWP01EW table, restricted to home-MSOA AND work-MSOA both within Greater London (33 boroughs).

### 2.2 Spatial unit

**MSOA21CD** (≈ 983 zones for Greater London after dropping zones with <5 commuters).

### 2.3 Sample subset for training

- **Train**: 80% of origin MSOAs randomly held in (seed=20260504).
- **Validation**: 10% origin MSOAs (early stopping).
- **Test**: 10% origin MSOAs, never seen during training. CPC reported on test only.

### 2.4 Exclusions (pre-specified)

- Drop MSOA pairs with Y_ij = 0 — softmax denominator handles them, but they contribute zero to the NLL.
- Drop home MSOAs with N_i < 50 commuters (small-cell unreliability).
- Drop the City of London MSOAs from origin set (pre-specified) — they are dominantly tourist/transient; not residential commuters.

### 2.5 Sample size for agent rollout

**N = 50 000** sampled commuter agents (representing ~3M actual commuters at 60:1 ratio). Sample stratified by home-MSOA × SOC × income-tier with weights = working-pop counts.

---

## 3. Method (frozen)

The full method specification is in `methodology/v2_5_paperA_inverse_choice.md`. Key freezes:

- **Architecture**: 2-layer GAT, hidden=64, 4 heads. No architecture search after freeze.
- **Loss**: NLL with FWL-inspired ridge penalty (`_fwl_inspired_ridge_penalty`) λ_FWL = 0.1; L2 weight decay 1e-4. _(Renamed per R1 + R5: this is a ridge regularizer on V_jt's projection onto cost/distance column means, not OLS-FWL residualisation. See `methodology/identifiability_argument.md` §2.)_
- **Optimiser**: Adam(lr=1e-3, β₁=0.9, β₂=0.999), batch=512 origins, max 50 epochs, early-stop patience=5.
- **Choice set**: Top-K=50 by minimum t_ij + observed destination.
- **Sign constraint**: β = -softplus(β_raw).
- **BPR**: α_BPR=0.15, β_BPR=4 (standard).
- **Hyperparameter sweep**: NONE on the real London data after freeze. All hyperparameter selection is via 5-fold CV on the *synthetic* W4 data.

### 3.1 Code freeze

The exact code state is captured at git commit `<COMMIT_HASH_PLACEHOLDER>`. Any post-freeze code change is logged in `paper/deviations_from_preregistration.md`.

### 3.2 Random seeds

`master_seed = 20260504`. All sub-seeds derived deterministically (see `methodology/odd_d_protocol.md` §5.1).

---

## 4. Multiverse Parameter Grid

_(addresses R4 doc-code mismatch: prior pre-reg drafts described a discrete LHS over (β prior, K, FWL on/off, graph A, employment def). The actual `validation/paper_a/multiverse_runner.py` code samples continuous Gaussians over (β_t, β_c, α_V, γ, K_choice). Per the F2/F3 decision, we update the pre-registration here to match the runner — this is simpler, aligns with the registration_hash that is computed over the runner's draws, and reflects the structural-prior literature standard. The earlier draft's discrete grid is **superseded** by the continuous Gaussian priors below.)_

The multiverse uses **continuous Gaussian priors** over the structural parameters (Steegen et al. 2016 multiverse analysis, structural-prior variant). Implemented in `validation/paper_a/multiverse_runner.py`.

### 4.1 Priors (continuous)

| Parameter | Prior | Interpretation |
|---|---|---|
| β_t (time) | Normal(-0.07, 0.02²) | Per-minute time disutility; centred on Train 2009 ch.6 mid-bracket; SD covers TAG ±25%. |
| β_c (cost) | Normal(-1.0, 0.2²) | Per-£ cost disutility; standardised cost units. |
| α_V (attractiveness) | Normal(1.0, 0.25²) | Loading on GNN attractiveness scalar V_jt; centred at unit identification. |
| γ (distance-deterrence) | Normal(-1.5, 0.4²) | Log-distance coefficient; centred on Lenormand 2016 / Train 2009 ch.6 bracket midpoint. |
| K_choice (choice-set size) | DiscreteUniform({25, 50, 100, 200}) | Number of TopK alternatives per origin. |

### 4.2 Sampling

- **N_DRAWS = 100** independent draws.
- **seed = 42**.
- Prior draws are independent across parameters (no joint covariance specified).
- The 100 (β_t, β_c, α_V, γ, K_choice) tuples are deterministically reproducible from (seed, N_DRAWS) and are logged in `validation/paper_a/multiverse_grid.json` at the freeze commit hash.

### 4.3 Rationale for continuous over discrete grid

- **Standard in the discrete-choice literature**: structural priors on (β_t, β_c, α_V, γ) are typically reported as Gaussians centred on prior estimates with SDs reflecting reported uncertainty (Train 2009 ch.6; Hensher & Greene 2003 mixed logit).
- **Aligns with `registration_hash` semantics**: the runner's hash is computed over the continuous draws, not over a categorical axis index. Aligning the pre-reg with the runner avoids a post-hoc deviation.
- **Coverage**: 100 continuous draws give better coverage of the prior tail than a 100-point LHS over a categorical 270-combination grid.

### 4.4 Reporting

For each of the 3 hypotheses we report:
- Median estimate across 100 runs
- 90% percentile CI
- 5%/95% extreme-run inspection
- Specification-curve plot (Simonsohn et al. 2020 style)

---

## 5. Literature Elasticity Ranges

The bracket ranges below are loaded from `validation/paper_a/literature_elasticity_db.py`.

### 5.1 VOT (commuting, 2021 GBP/hour)

| Source | Low | High |
|---|---|---|
| DfT TAG A1.3 (2021) | 11.0 | 14.5 |
| Mackie et al. 2003 (UK meta-analysis) | 9.0 | 16.0 |
| Small & Verhoef 2007 ch.2 (US, GBP-converted) | 10.0 | 18.0 |
| **Used bracket (intersection)** | **11.0** | **14.5** |

### 5.2 Distance-deterrence γ (gravity-model literature)

| Source | Low | High |
|---|---|---|
| Lenormand et al. 2016 (European cities) | -1.5 | -0.8 |
| Simini et al. 2012 (radiation model) | n/a (parameter-free) | n/a |
| Train 2009 ch.6 table | -1.4 | -0.6 |
| **Used bracket** | **-1.5** | **-0.6** |

### 5.3 Mode-choice β_t (per minute)

| Source | Low | High |
|---|---|---|
| Train 2009 ch.6 (US transit) | -0.08 | -0.02 |
| Wardman 2014 (UK rail meta) | -0.06 | -0.03 |
| **Used bracket** | **-0.08** | **-0.02** |

### 5.4 Use of brackets in the paper

A recovered parameter is "**within bracket**" if the point estimate AND the 95% bootstrap CI lower bound both fall inside. We report bracket containment rate across the multiverse.

---

## 6. Pass/Fail Criteria per Hypothesis

### H1 (β̂ consistent with TAG)

**PASS**: implied VOT (median across multiverse) ∈ [11.0, 14.5] GBP/hour at the mid-income tier.
**FAIL**: VOT outside this bracket.
**Reporting**: pass/fail + magnitude of deviation; no p-value (this is a triangulation, not a test).

### H2 (Scenario A direction)

**PASS**: ΔY_·,OOC > 0 in our model AND > 0 in gravity baseline AND |sign(our) - sign(gravity)| = 0.
**FAIL**: signs disagree; we report the disagreement and investigate (this is a research finding, not a paper-killer).
**Reporting**: directional comparison + magnitude ratio (our / gravity).

### H3 (Multiverse β̂ CI excludes zero)

**PASS**: 5th and 95th percentiles of β̂ across 100 multiverse runs are both negative.
**FAIL**: any percentile crosses zero. Report and investigate.
**Reporting**: full multiverse histogram + specification curve.

### Composite outcome

| H1 | H2 | H3 | Paper outcome |
|---|---|---|---|
| Pass | Pass | Pass | Submit as planned. |
| Pass | Pass | Fail | Submit with multiverse caveat highlighted; reframe as "sensitive but central estimate is consistent". |
| Pass | Fail | Pass | Investigate gravity disagreement; submit if explained. |
| Fail | * | * | Re-examine SCM and identifiability; consider Paper A reframe to "what aggregate OD does NOT identify". |
| * | * | Fail | Same as above — robustness failure is a major finding either way. |

Critically: **all four outcomes lead to a paper**, just different papers. We do not file-drawer.

---

## 7. Stopping Rules

### W4 — Synthetic recovery checkpoint

**Stop and reframe** if:
- bias on (α̂, β̂, γ̂) > 10% on synthetic data, OR
- 95% CI coverage < 85% (200 bootstrap reps), OR
- GNN R² of V̂ vs V_true < 0.85.

**Reframe target**: "Estimator development for differentiable inverse choice — methods paper." Drop London fit and scenarios entirely.

### W8 — London CPC GO/NO-GO

**GO** if CPC on held-out test origins ≥ 0.5.

**NO-GO** if CPC < 0.5: drop Scenario A; submit as "London structural recovery" without scenarios.

### W10 — Multiverse + VOT-TAG checkpoint

If H1 fails by > 50% deviation OR H3 fails outright: stop, write up negative findings honestly. Do not p-hack the multiverse grid post-hoc.

### W12 — Final draft

Whatever combination of hypotheses passes, freeze the paper. Any post-freeze re-run is logged in `deviations_from_preregistration.md`.

---

## Appendix — Pre-registration Provenance

This document is committed at git hash `<COMMIT_HASH_PLACEHOLDER>` with the file checksum recorded as `<SHA256_PLACEHOLDER>`. Any modification after the freeze date is reported in the published paper as a deviation.

---

*End pre-registration. Companion: `methodology/v2_5_paperA_inverse_choice.md`, `methodology/odd_d_protocol.md`.*
