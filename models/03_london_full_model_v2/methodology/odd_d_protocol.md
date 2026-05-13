# ODD+D Protocol — London Commuting Choice Model (v2.5 Paper A)

> **Standard**: Müller et al. (2013) ODD+D extension of Grimm et al. (2010) ODD.
> **Scope**: This document specifies the agent-based simulation layer of the v2.5 model that wraps the recovered structural parameters (θ̂, α̂, β̂, γ̂) from `v2_5_paperA_inverse_choice.md`. It is the reproducibility specification.

## Table of Contents

1. [Overview](#1-overview)
   - 1.1 Purpose
   - 1.2 Entities, state variables, scales
   - 1.3 Process overview and scheduling
2. [Design Concepts](#2-design-concepts)
   - 2.1 Theoretical and empirical background
   - 2.2 Individual decision-making
   - 2.3 Learning
   - 2.4 Individual sensing
   - 2.5 Individual prediction
   - 2.6 Interaction
   - 2.7 Collectives
   - 2.8 Heterogeneity
   - 2.9 Stochasticity
   - 2.10 Observation
3. [Details](#3-details)
   - 3.1 Implementation details
   - 3.2 Initialisation
   - 3.3 Input data
   - 3.4 Sub-models
4. [ODD+D Decision Sub-model (Müller 2013 §3 spec)](#4-odd-d-decision-sub-model)
5. [Reproducibility](#5-reproducibility)

---

## 1. Overview

### 1.1 Purpose

The model's purpose is **scenario-aware spatial choice modelling**: given (a) zone features X_j, (b) a recovered structural attractiveness function GNN_θ̂, and (c) recovered behavioural coefficients (α̂, β̂, γ̂) from London 2021 OD, simulate the equilibrium destination-and-mode choice distribution under **scenario simulation using recovered structural parameters** — i.e., exploratory scenario analysis under stated zone-feature interventions `do(X_j)`. _(addresses R5 / refinement #6: removed "counterfactual" leak from Purpose to keep ODD+D wording consistent with the L2-conditional framing of Paper A §1.2.)_

The model is **not** intended to:
- Predict individual-level commuting decisions (no individual ID is identifiable).
- Forecast OD flows for unobserved years from time-series alone.
- Validate causal claims absent the SCM stated in `v2_5_paperA_inverse_choice.md` §2.

### 1.2 Entities, state variables, scales

**Entities**:
- **Commuter agent** — one per ~5 sampled commuters from the 2021 Census working population (default N = 50 000 agents representing ~3M commuters).
- **Grid cell** — 1 km × 1 km cell aggregated into MSOA (≈983 MSOAs covering Greater London).
- **Borough** — 33 London boroughs (used for congestion attribution from TomTom).
- **OD pair** — directed (i, j) with i, j ∈ MSOAs; observed flow Y_ij.

**Per-agent state variables**:

| Variable | Type | Domain | Source |
|---|---|---|---|
| `home_grid_idx` | int | {0, ..., 982} | Sampled from 2021 working-population distribution |
| `soc` | int | {1, ..., 9} | SOC2020 major group; sampled from MSOA-conditional distribution |
| `income` | float | ℝ_+ (£/yr) | Drawn from LSOA equivalised income (ONS) given home_grid_idx |
| `age_band` | int | {16-24, 25-34, 35-49, 50-64, 65+} | Census 2021 conditional on MSOA |
| `mode` | enum | {car, PT, active} | Chosen each step from RUM |
| `departure_hour` | int | {6, ..., 21} | Sampled from MSOA-conditional NTS time-of-day distribution |
| `chosen_work_msoa` | int | {0, ..., 982} | Chosen each step from RUM (re-evaluated under scenario simulation) |
| `commute_time_minutes` | float | ℝ_+ | Computed from BPR given chosen mode and hour |

**Spatial scale**: MSOA (≈ 7000 residents per zone; mean area ≈ 1.7 km²).
**Temporal scale**: 1 hour per step; 16 steps per day (06:00–21:00); a "run" is 1 representative weekday.

### 1.3 Process overview and scheduling

```
For each step t in {06, 07, ..., 21}:
   1. (For each agent active at t) Sense: read V_jt for all j in TopK(home),
      read t_ij^t for all j in TopK, read own income.
   2. (For each agent active at t) Decide: sample (j, mode) from softmax over
      U_ij = α V_jt + β t_ij + γ log d_ij + ε  (Gumbel ε).
   3. (Aggregate) Sum mode-specific volumes per link.
   4. (Update congestion) BPR fixed-point: t_ij^{t+1} = t_ff · (1 + 0.15 (V/C)^4).
   5. (Output) DataCollector logs Mean_Commute_Time, Accessibility_Gini,
      Mean_Accessibility, Validation_Correlation, mode_share, chosen_work_msoa
      histogram per origin.
```

Step 1–4 form an inner fixed-point loop (typically converges in 5–10 iterations per step). Step 5 runs once per step.

---

## 2. Design Concepts

### 2.1 Theoretical and empirical background

The choice model is grounded in **McFadden 1974, 1978** Random Utility Maximisation (RUM); the spatial attractiveness function is grounded in **Yao et al. 2021** SI-GCN and **Simini et al. 2021** Deep Gravity. Empirical calibration is to **2021 London Census OD** (Travel-to-Work table TS061 + ODWP01EW).

### 2.2 Individual decision-making

**Decision rule**: Each agent picks (j, mode) by maximising
```
U_ij,m = α̂ V_jt + β̂ t_ij,m + γ̂ log d_ij + ASC_m + ε_ij,m
```
where ε_ij,m is i.i.d. Gumbel(0, 1), ASC_m is a mode-specific constant, and (α̂, β̂, γ̂) are the recovered parameters from the inverse training step (Paper A, §4).

**Key invariance assumption**: Agents in the scenario-simulation world `do(X_OOC)` use the **same** (α̂, β̂, γ̂) as in the observed world. This is the ICM/Schölkopf 2021 assumption explicitly stated in `v2_5_paperA_inverse_choice.md` §2.5. It is the central methodological claim and the central limitation. _(Wording: "scenario simulation" rather than "counterfactual" per R5 refinement #6; the underlying L2-conditional claim is unchanged.)_

**Frozen θ̂**: GNN_θ̂ is forward-evaluated in the scenario simulation; θ̂ is **not** retrained. Equivalently: the *mechanism* mapping zone features to attractiveness is invariant; only the input X_j changes.

### 2.3 Learning

**No within-simulation learning.** Agents do not update beliefs, do not have memory, do not adapt. This is a deliberate limitation of v2.5 Paper A — a one-shot equilibrium model. Iterative learning (e.g., reinforcement-learning-style adaptation, or a la Manski 1993 social interactions) is deferred to future work.

This is a stronger limitation than it sounds: the BPR congestion fixed-point is solved analytically each step, but agents do *not* iteratively re-route based on yesterday's congestion. Mean-field equilibrium is assumed instantaneous.

### 2.4 Individual sensing

Each agent senses:
- **The full attractiveness vector** {V_jt}_{j ∈ TopK(home)} (50 destinations).
- **The travel-time vector** {t_ij,m}_{j ∈ TopK, m ∈ {car, PT, active}}.
- **Own income, age band, SOC** (used in the income-tiered β if active).

Agents do NOT sense:
- Other agents' choices directly (only via the congestion proxy in t_ij).
- Their own historical choices (no memory).
- Aggregate quantities like mode-share or accessibility.

### 2.5 Individual prediction

Agents are myopic: they pick the choice with maximum current expected utility, no forward-looking. Each step is independent given the equilibrium congestion field.

### 2.6 Interaction

**Indirect only, via congestion.** Agents interact only through the BPR equilibrium: more agents picking mode car on link ij → higher V/C ratio → higher t_ij,car → next iteration of the inner fixed-point shifts agents away. No direct interaction (no carpooling, no shared decisions, no peer effects).

### 2.7 Collectives

None modelled. Households, firms, and social networks are out of scope.

### 2.8 Heterogeneity

**Income tier (3 levels)**: β is allowed to vary across {low, mid, high} income terciles, with prior N(β̂_pooled, σ²) shrinkage. This is the v2.5 minimum heterogeneity. **All other parameters (α, γ, θ) are global.**

Heterogeneity in (mode set, ASC, choice set size) is *not* modelled in v2.5 Paper A — deferred to v2.6 / Paper B.

### 2.9 Stochasticity

Two sources:
- **Gumbel choice error** ε_ij ~ Gumbel(0, 1), i.i.d. per agent per alternative per step. Drives the softmax.
- **Poisson output sampling**: when reporting OD counts Y_ij at the aggregate level, we Poisson-sample N_i · P(j|i) rather than reporting expected counts. This is for matching the observed integer OD format.

Distance d_ij and graph A are deterministic.

### 2.10 Observation

Output variables logged each step:
- `Mean_Commute_Time` (minutes)
- `Mean_Accessibility` (gravity-weighted)
- `Accessibility_Gini`, `Accessibility_Palma`
- `Validation_Correlation` (sim vs observed accessibility)
- `Mode_Share_{car,pt,active}`
- `OD_count_matrix` (sparse, top 5% pairs only, for memory)
- `Chosen_Work_MSOA_histogram` per home MSOA (for choropleth visualisation)

---

## 3. Details

### 3.1 Implementation details

- **Language**: Python 3.11
- **Core libs**: PyTorch 2.1 (GNN), JAX 0.4 + jaxopt 0.8 (implicit diff), Mesa 2.3 + mesa-geo 0.7 (agent layer), GeoPandas 0.14, NetworkX 3.2.
- **Compute**: 1× A100 40GB for training; CPU (≥16 cores, ≥64GB RAM) for ABM rollouts.
- **Reproducibility**: all random calls use a single `np.random.default_rng(seed)` passed top-down. JAX uses `jax.random.split` from a master key.

### 3.2 Initialisation

```
1. Load processed data files (§3.3).
2. Build graph A (MSOA-level, queen-contiguity).
3. Compute X_j (18-dim node feature matrix).
4. Sample N agents:
   - home_grid_idx ~ Categorical(working_pop_share)
   - given home_grid_idx: SOC, income, age_band ~ Census conditional
   - departure_hour ~ NTS conditional on (SOC, distance band)
5. Load recovered parameters (θ̂, α̂, β̂, γ̂) from `weights/paperA_w8.pt`.
6. Forward-pass GNN_θ̂ to get V_jt for all (j, t).
7. Initialise t_ij^0 with free-flow times.
```

### 3.3 Input data

| File | Source | Geography | Vintage |
|---|---|---|---|
| `data/processed/london_OD_travel2work.csv` | ONS Census ODWP01EW | MSOA21CD | 2021 |
| `data/processed/london_msoa_boundaries.geojson` | ONS Open Geography | MSOA21CD | 2021 |
| `data/processed/london_commute_mode_msoa.csv` | Census 2011 (legacy) | MSOA11CD | 2011 |
| `data/processed/msoa_hourly_congestion.csv` | TomTom Traffic Stats | MSOA21CD | 2022–23 |
| `data/processed/london_bres_msoa.csv` | NOMIS BRES | MSOA21CD | 2024 |
| `data/processed/london_occupation_msoa.csv` | Census TS062 | MSOA21CD | 2021 |
| `data/processed/soc_work_attraction.json` | Derived | MSOA21CD × SOC | 2021 |

**Known geography mismatch** (carried over from v1 / `02_london_commuting_model/CLAUDE.md`): commute mode data uses `MSOA11CD` while OD and boundaries use `MSOA21CD`. The model falls back to London-wide defaults `(car=0.341, pt=0.519, active=0.132)` for unmatched zones.

### 3.4 Sub-models

#### 3.4.1 GNN attractiveness V_jt

```
H^(0) = X
H^(1) = ELU(GAT_layer1(H^(0), A))   # 4 heads, hidden=64
H^(2) = GAT_layer2(H^(1), A)        # 1 head, hidden=1
V_j   = H^(2)  + bias_t              # bias_t is hour-specific scalar
```

Trained jointly with (α, β, γ) in Paper A §4 via implicit-diff NLL on observed OD.

#### 3.4.2 RUM choice

```
U_ij = α̂ V_jt + β̂_tier(income) · t_ij + γ̂ log d_ij + ASC_mode + ε
P(j, m | i) = exp(U_ij,m) / Σ_{j' ∈ TopK, m'} exp(U_ij',m')
```

#### 3.4.3 BPR congestion equilibrium

```
t_ij,car = t_ff,ij · (1 + 0.15 · (V_ij / C_ij)^4)
t_ij,pt   = t_pt_baseline,ij · (1 + λ_crowd · 𝟙{peak})
t_ij,active = t_ff,ij,active   (no congestion)
```

Solved as fixed-point per step (5–10 inner iterations to convergence ε < 1e-3 on mean t).

---

## 4. ODD+D Decision Sub-model

Per Müller et al. 2013 §3, the decision sub-model is documented in additional structured detail.

### 4.1 Subject of decision

Each commuter decides, *jointly*, **(workplace MSOA, mode)** for one representative weekday, given their home MSOA and personal attributes (income, SOC, age band).

### 4.2 Decision-makers' objective

Maximise random utility `U_ij,m = α V_jt + β t_ij,m + γ log d_ij + ASC_m + ε`. The objective is **not** explicitly cast as a normative goal (e.g., "minimise commute time"); it is a positive description of revealed-preference behaviour, with the structural parameters identifying the relative weights.

### 4.3 Decision rule

Single-shot maximum-utility selection from the random-utility realisation. Equivalently: sample (j, m) from `softmax(U_·)` over the choice set TopK(i) × {car, PT, active}.

### 4.4 Information used

Strict subset of the agent's sensing scope (§2.4):
- Attractiveness V_jt (from GNN_θ̂; the agent does not "see" X_j directly, only the GNN output)
- Travel time t_ij,m (from BPR equilibrium at current step)
- Distance d_ij (geographic, fixed)
- Own income tier (selects β̂_tier)

### 4.5 Spatial level of decision

**MSOA-level workplace choice.** Within-MSOA destination is not modelled (no street-level destination, no specific workplace).

### 4.6 Temporal level of decision

**Once per step (1 hour).** Agents with `departure_hour = t` make their choice at step t; agents with other departure hours are dormant at step t (their previously-made choice persists in the congestion accounting).

### 4.7 Adaptation

**None.** Agents do not update β̂_tier within a run, do not learn from past commute times, do not switch modes mid-day. This is the fixed-parameter equilibrium assumption.

### 4.8 Heterogeneity

- β̂ varies by income tier (3 levels).
- ASC_mode is global.
- α̂, γ̂ are global.
- θ̂ is global (one GNN for all agents).

### 4.9 Memory and learning

None — see §2.3, §4.7.

### 4.10 Uncertainty

**Aleatoric**: Gumbel ε on each (i, j, m) triple realises a stochastic choice. Aggregating over 50 000 agents averages out individual ε but preserves macroscopic Poisson noise.
**Epistemic**: parameter uncertainty in (α̂, β̂, γ̂, θ̂) is propagated via the multiverse (§5.3 of `v2_5_paperA_inverse_choice.md`) and bootstrap CIs.

### 4.11 Decision interactions

Agents interact only via congestion (mean-field). No game-theoretic / strategic decision-making.

---

## 5. Reproducibility

### 5.1 Random seeds

| Seed name | Default | Used for |
|---|---|---|
| `master_seed` | 20260504 | Top-level seed; all sub-seeds derived |
| `agent_init_seed` | derived | Agent attribute sampling |
| `gnn_init_seed` | derived | PyTorch GNN weight init |
| `gumbel_seed` | derived | Gumbel ε realisations per step |
| `multiverse_seed` | derived | Multiverse run-id offset |

### 5.2 Hyperparameter table (frozen for Paper A submission)

| Hyperparameter | Value | Notes |
|---|---|---|
| GNN architecture | 2-layer GAT | hidden=64, heads=4 |
| GNN dropout | 0.1 | |
| GNN weight decay | 1e-4 | L2 |
| Optimiser | Adam | lr=1e-3, betas=(0.9, 0.999) |
| Batch size (origins) | 512 | |
| Epochs | 50 | early-stop patience=5 |
| Choice set K | 50 | top-K by minimum t_ij + observed dest |
| BPR α (capacity) | 0.15 | standard |
| BPR β (power) | 4 | standard |
| Inner fixed-point tol | 1e-3 | mean Δt convergence |
| Max inner iters | 30 | failsafe |
| N agents | 50 000 | scaled from ~3M census commuters |
| Number of hours simulated | 16 | 06:00 – 21:00 |
| Multiverse runs | 100 | continuous Gaussian priors over (β_t, β_c, α_V, γ, K_choice); see `paper/preregistration.md` §4 |
| Bootstrap reps | 200 | for parameter CIs |

### 5.3 Software versions (frozen)

```
python==3.11.7
torch==2.1.0+cu121
jax==0.4.20
jaxopt==0.8.1
mesa==2.3.0
mesa-geo==0.7.0   # NOTE: v2.5 has migrated off mesa-geo (see commit f60aafb);
                  # spec retained here for cross-version reproducibility
geopandas==0.14.1
networkx==3.2.1
torch-geometric==2.4.0
solara==1.30.1
```

### 5.4 Hardware

- **Training**: 1× NVIDIA A100 40GB, CUDA 12.1, ~6h wall-clock for London 2021 fit
- **ABM rollout / multiverse**: 16-core CPU, 64GB RAM, ~30 min per multiverse run
- **Storage**: ~10 GB raw data, ~2 GB processed, ~5 GB model checkpoints + multiverse logs

### 5.5 Determinism caveats

- PyTorch CUDA reductions are non-deterministic by default. We set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Trade-off: ~15% slower training.
- jaxopt's `FixedPointIteration` with `implicit_diff=True` is deterministic for a fixed seed.
- Mesa scheduler is deterministic given a fixed seed.

### 5.6 Provenance

Every output figure/table in the Paper A submission carries a `(commit_hash, seed, run_id)` triple in its caption. The pre-registration freeze hash is recorded in `paper/preregistration.md`.

---

*End ODD+D protocol. See companion files: `v2_5_paperA_inverse_choice.md`, `identifiability_argument.md`, `../paper/preregistration.md`.*
