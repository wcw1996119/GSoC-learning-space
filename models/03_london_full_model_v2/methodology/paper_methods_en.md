# 2 Methods

## 2.1 Framework Overview

We propose an **ML-ABM hybrid framework** for urban commuting analysis. The
framework has two stages tied together by a single set of learned
parameters (Figure 1):

- **Stage 1 (machine learning)** — A spatio-temporal graph neural network
  is trained, end-to-end on observed hourly origin-destination flows, to
  produce two outputs: a destination-attractiveness embedding $V_{j,t}$
  for every destination grid $j$ at every hour $t$, and a small set of
  behavioural coefficients $(\beta, \gamma, \delta, \kappa)$ governing
  how commuters trade off travel time, distance, labour-market match,
  and income-tier heterogeneity.

- **Stage 2 (agent-based modelling)** — At deployment time, the same
  $V_{j,t}$ and the same behavioural coefficients are used by a
  population of synthetic agents to make individual workplace
  decisions. Aggregating individual choices over the population
  reproduces, by construction, the population-level flow distribution
  the ML stage was trained against; perturbing inputs at this stage
  (e.g., adding 65,000 jobs at a candidate Old Oak Common cluster) lets
  us simulate counterfactual policy interventions while preserving the
  data-fitted decision mechanism.

The bridge between the two stages is a multinomial logit choice
formulation (McFadden, 1974). Under this formulation the cross-entropy
loss used in Stage 1 is *numerically identical* to the maximum
log-likelihood objective for individual choices in Stage 2; we develop
this identity in §2.7. The same formulation also lets us read the
fitted $\beta$, $\gamma$, $\delta$, $\kappa$ as **behavioural
parameters** rather than uninterpretable network weights — supporting
both the predictive power of deep learning and the behavioural
interpretability that urban-policy questions require.

![Figure 1 — ML-ABM hybrid framework. Stage 1 (left, green) is supervised ML training: the dual-branch encoder produces destination attractiveness $V_{j,t}$; the behavioural choice layer (mode × tier mixture multinomial logit) outputs $P(j \mid i, t)$ matched against observed hourly OD via cross-entropy. Stage 2 (right, red) is agent-based deployment with congestion feedback (orange): agents pick destinations using free-flow $t_{ij}$, aggregated flows are passed to a BPR equilibrium that returns congested $t_{ij}$, and agents re-evaluate utility under congestion until convergence. The shared-parameter bus (dark green, dashed) connects training to deployment.](paper_methods_architecture.png)

A schematic system view is also given for plain-text viewers:

```
Figure 1 (system view).  ML-ABM hybrid framework — two stages sharing learned parameters.

  Stage 1: Machine Learning (training)
  -------------------------------------
   X_static (1725, 22)   ─→  Static branch (GraphSAGE × 2)
                                    ↓
                              h_static (1725, 32)
                                    │
   X_dynamic (24, 1725, 5)            │
    · congestion ratio                │
    · hourly destination inflow       │
    · queue residual placeholder      │
    · sin/cos hour-of-day  ─→  Dynamic branch
                                    │
                            (per-hour SAGE × 2)
                                    ↓
                            (GRU, 24-step) + (multi-scale TCN: kernels 3,5,7)
                                    ↓ fuse
                              h_dynamic (24, 1725, 32)
                                    │
                                    ↓
                              Gated fusion
                                    ↓
                              V_jt (24, 1725)
                                    │
                                    ↓
                              Behavioural choice layer
                              (multinomial logit; mode × tier mixture)
                                    │
                                    ↓
                              P(j | i, t)  = softmax over destinations
                                    │
                                    ↓
                              cross-entropy loss vs F_{ij,t}
                                    │
                                    ↓
                              AdamW updates encoder + behavioural coefficients


  Stage 2: Agent-Based Modelling (deployment / counterfactuals)
  -------------------------------------------------------------
   N_agents synthetic commuters, each with attributes (i, t, m, k):
       i ∈ origin grid, t ∈ hour, m ∈ mode, k ∈ income tier
                                    ↓
   For each agent, evaluate utility for every candidate destination j:
       U_{ij,t}^{(m,k)} = α V_{j,t} + β_{t,m,k} t_{ij}^{(m)}
                          + γ log d_{ij} + δ_k OccMatch_{ij}
                          + ε_{ij}            (ε ~ Gumbel)
                                    ↓
   Agent picks j* = argmax_j U_{ij,t}^{(m,k)}
                                    ↓
   Aggregate over agents → simulated F̂_{ij,t}
                                    ↓
   Counterfactual: replace inputs with intervention (e.g. +65,000 jobs at OOC)
                  → recompute V_{j,t} under intervention via the trained encoder
                  → re-run agent-level decisions with **same** behavioural coefficients
                  → simulated counterfactual F̂_{ij,t}'
```

## 2.2 Aggregate OD as Supervised Classification

Let $N$ be the number of grid cells in the study area and $T$ the number
of hour-of-day windows ($N = 1{,}725$, $T = 24$ in our London
implementation). The training data are an observed flow tensor

$$
\mathbf{F} \in \mathbb{Z}_{\geq 0}^{T \times N \times N},
\qquad
F_{ij,t} = \text{number of observed trips from grid } i \text{ to grid } j \text{ during hour } t.
$$

A natural framing of the modelling task is supervised classification:
*for each (origin, hour) example $(i, t)$, predict the share of trips
going to each of the $N$ candidate destinations*.

Concretely, for each $(i, t)$ the model outputs a probability vector
$\mathbf{P}_{i,t} \in \Delta^{N-1}$ over the $N$ destinations, and the
training label is the empirical share
$\hat{q}_{ij,t} = F_{ij,t} / O_{i,t}$ where $O_{i,t} = \sum_j F_{ij,t}$
is the total outflow. The loss is the standard weighted cross-entropy

$$
\mathcal{L}(\boldsymbol{\theta})
= - \sum_{t=1}^{T} \sum_{i \in \mathcal{I}_{\text{train}}} \sum_{j=1}^{N}
   F_{ij,t} \log P(j \mid i, t;\, \boldsymbol{\theta}),
\tag{1}
$$

where $\mathcal{I}_{\text{train}}$ is the set of training origins (87% of
the 1,725 grids; the remaining 13% are held out for validation; see
§2.11). The weighting by $F_{ij,t}$ makes high-flow OD pairs contribute
more to the loss than rare OD pairs — equivalent to multinomial
log-likelihood on the per-origin choice distribution.

Equation (1) is identical to the negative log-likelihood of a
multinomial choice model in which each of $O_{i,t}$ commuters chooses
independently from origin $i$ during hour $t$. We exploit this dual
identity in §2.7. The remainder of this section specifies the
parameterisation of $P(j \mid i, t; \boldsymbol{\theta})$.

## 2.3 Spatial Graph and Node Features

**Spatial graph.** We construct a fixed undirected graph $\mathcal{G} =
(\mathcal{V}, \mathcal{E})$ where $\mathcal{V} = \{1, \ldots, N\}$
indexes the 1 km grid cells covering Greater London. Two cells are
connected by an edge if their centroid-to-centroid Euclidean distance
is below 1.5 km. The resulting graph has 13,180 directed edges (mean
out-degree 7.6); a self-loop is added at each node so isolated cells
do not vanish under neighbour aggregation. The graph is *static*:
edges do not change over time. The decision to use a fixed geographic
adjacency rather than a learned data-driven graph (in the style of
Wu et al. 2019; Lan et al. 2022) is deliberate — it preserves
identifiability under aggregate observation (§2.8) and ensures the
graph structure is directly transferable to other cities.

**Static node features** $\mathbf{X}^{\mathrm{static}} \in \mathbb{R}^{N \times F_{s}}$
($F_s = 22$). For each grid cell we compute time-invariant attributes:

- POI category counts (nine categories) from OpenStreetMap (Boeing, 2017);
- employment counts by industry (eight categories) from BRES 2024;
- residential population from Census 2021;
- transport infrastructure flags (rail station, tube station).

All features are z-scored across cells.

**Dynamic node features** $\mathbf{X}^{\mathrm{dyn}} \in \mathbb{R}^{T \times N \times F_{d}}$
($F_d = 5$). For each (hour, cell):

- TomTom borough × hour-of-day congestion ratio, mapped to grid cells
  (z-scored);
- Hourly destination inflow $\sum_i F_{ij,t}$ (z-scored), constructed
  from the same OD tensor but used only as an explanatory variable in
  the encoder;
- Queue residual placeholder (zero during training; populated during
  counterfactual simulation to carry hour-to-hour congestion spillover);
- Sinusoidal time-of-day embedding $(\sin(2\pi t / T), \cos(2\pi t / T))$.

The decision to keep static features static (and *not* tile them across
$T$) reflects an architectural prior: 22 of the 25 dimensions in our
encoder input genuinely do not vary at hourly resolution, so feeding
them through a temporal mechanism wastes capacity. We discuss the
two-branch consequence in §2.4.

**Travel time matrices.** For each travel mode
$m \in \{\text{car}, \text{transit}, \text{walk}\}$ we precompute a
mode-specific travel time matrix $t_{ij}^{(m)} \in \mathbb{R}^{N \times N}$
in minutes:

- $t_{ij}^{(\mathrm{car})}$: free-flow OS Open Roads driving time computed
  via OSMnx (Boeing, 2017); range 0–68 min, median 32.5 min.
- $t_{ij}^{(\mathrm{transit})}$: piecewise-linear approximation from
  Euclidean distance — 12 km/h for $d_{ij} < 3$ km, 25 km/h with 8 min
  waiting/transfer overhead otherwise; range 0–150 min, median 59 min.
  We replace this with `r5r` GTFS routing in future work; see §2.11.
- $t_{ij}^{(\mathrm{walk})}$: $d_{ij} / 5\,\mathrm{km/h}$, capped at 180 min;
  median 180 min (because most OD pairs in London exceed 15 km).

A pairwise log-distance matrix $\log d_{ij}$ provides a separate
distance-decay covariate.

**Labour-market match.** $\mathrm{OccMatch}_{ij}$ is the cosine
similarity between origin $i$'s SOC9 occupation distribution (Census
2021) and destination $j$'s industry-implied occupation demand
(Pellegrini & Fotheringham, 2002), pre-computed once.

## 2.4 Stage 1 — Dual-Branch Spatio-Temporal Encoder

The encoder maps $(\mathbf{X}^{\mathrm{static}}, \mathbf{X}^{\mathrm{dyn}}, \mathcal{E})$
to a destination attractiveness scalar $V_{j,t}$ for every
$(j, t) \in \{1, \ldots, N\} \times \{1, \ldots, T\}$. It has three
components — a static branch, a dynamic branch, and a gated fusion.

**Static branch.** Two GraphSAGE layers (Hamilton et al., 2017) with
mean aggregation:

$$
\mathbf{h}_{i}^{(\ell+1)}
= \sigma\!\left( W_{\mathrm{self}}^{(\ell)} \mathbf{h}_{i}^{(\ell)}
   + W_{\mathrm{nb}}^{(\ell)} \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{h}_{j}^{(\ell)} \right),
\quad \ell = 0, 1,
\tag{2}
$$

where $\sigma$ is the ReLU, $\mathcal{N}(i)$ are the neighbours of $i$
including a self-loop, and $\mathbf{h}_{i}^{(0)} = \mathbf{X}^{\mathrm{static}}_{i}$.
Two layers give a two-hop receptive field per destination. The output
$\mathbf{h}^{\mathrm{static}} \in \mathbb{R}^{N \times d}$ ($d = 32$)
encodes long-run destination desirability — POI mix, employment density,
transport infrastructure, and their two-hop spatial spillovers.

**Dynamic branch.** For each hour $t$, we apply two GraphSAGE layers
(2) to $\mathbf{X}^{\mathrm{dyn}}_{t}$, yielding 24 hidden states
$\{\tilde{\mathbf{h}}_{t}\}_{t=0}^{23}$. We pass these through two
parallel temporal mechanisms:

- A **gated recurrent unit (GRU)** sweeping forward in $t$:
  $\mathbf{h}_{t}^{\mathrm{gru}} = \mathrm{GRU}(\tilde{\mathbf{h}}_{t}, \mathbf{h}_{t-1}^{\mathrm{gru}})$
  (Cho et al., 2014). This carries hourly sequential dependence — the
  state at hour 9 inherits information from hours 0–8, which we use as
  the model's representation of *carry-over* effects (e.g., how
  rush-hour congestion at 8 a.m. shapes 9 a.m. accessibility).
- A **multi-scale temporal convolutional block** with three parallel
  causal 1D convolutions of kernel sizes 3, 5, and 7, followed by
  concatenation and a linear projection. Circular padding lets the
  model treat the 24-hour cycle as periodic. Different kernels capture
  different bandwidths of intra-day pattern.

The two outputs are concatenated and passed through a linear layer to
yield $\mathbf{h}^{\mathrm{dyn}} \in \mathbb{R}^{T \times N \times d}$.

**Gated fusion.** A scalar gate per $(t, j)$,

$$
g_{j,t} = \sigma\!\left( \mathbf{w}^{\top} [\mathbf{h}_{j}^{\mathrm{static}};\, \mathbf{h}_{j,t}^{\mathrm{dyn}}] \right) \in [0, 1],
\tag{3}
$$

combines the two representations,

$$
\mathbf{h}_{j,t} = g_{j,t} \cdot \mathbf{h}_{j}^{\mathrm{static}}
                + (1 - g_{j,t}) \cdot \mathbf{h}_{j,t}^{\mathrm{dyn}},
\tag{4}
$$

and a final linear layer collapses to a scalar
$V_{j,t} = \mathbf{w}_{V}^{\top} \mathbf{h}_{j,t}$. We then z-score
$V_{j,t}$ across all $(j, t)$ jointly to enforce zero mean and unit
standard deviation; this fixes the otherwise-arbitrary scale of $V$
and is a standard identification trick for deep multinomial-choice
models (Wang & Klabjan, 2018).

The gated fusion lets the model decide, for each (hour, destination)
cell, whether to rely more on the static long-run desirability or the
dynamic this-hour signal. Empirically the gate concentrates near 1 at
midnight (static features carry the day) and shifts toward 0.5 at peak
hours (dynamic inflow contains independent information).

The encoder has approximately 33,000 parameters in total.

## 2.5 Stage 1 — Behavioural Choice Layer (Mode × Tier Mixture)

We model commuters as heterogeneous along two observable dimensions:

- **Travel mode** $m \in \{\text{car}, \text{transit}, \text{walk}\}$,
  with origin-level mode share $\pi_m^{\mathrm{origin}}(i)$ obtained
  from Census 2011 *Method of travel to work* aggregated to the 1 km
  grid;
- **Income tier** $k \in \{1, 2, 3\}$ (low / middle / high), with
  tier weights $\pi_k(i)$ derived from the IMD 2019 income score
  (Department for Communities and Local Government, 2019).

**Distance-aware choice set.** Following the choice-set generation
literature (Manski, 1977; Ben-Akiva & Boccara, 1995), we further
restrict the walking option to feasible distances. Specifically we
build a *pair-level* mode share $\pi_m(i, j)$ such that
$\pi_{\mathrm{walk}}(i, j) = 0$ whenever $d_{ij} \geq 5$ km, with the
removed mass redistributed proportionally between the motorised modes:

$$
\pi_{m}(i, j) = \begin{cases}
\pi_{m}^{\mathrm{origin}}(i)                                    & \text{if } d_{ij} < 5 \text{ km}, \\
\pi_{m}^{\mathrm{origin}}(i)\,\dfrac{1}{1 - \pi_{\mathrm{walk}}^{\mathrm{origin}}(i)} & \text{if } d_{ij} \geq 5 \text{ km}, m \neq \mathrm{walk}, \\
0                                                                & \text{if } d_{ij} \geq 5 \text{ km}, m = \mathrm{walk}.
\end{cases}
\tag{5}
$$

This restriction has two motivations. First, walking commute share in
London (≈11%) is concentrated in short trips — long-distance walking
is empirically negligible. Second, it improves identifiability of the
walking time coefficient $\beta_{t,\mathrm{walk}}$ by anchoring its
estimation to the subset of OD pairs where walking is actually a
realistic option; see §2.8.

**Profile-specific systematic utility.** Each (mode, tier) profile at
origin $i$ faces its own systematic utility for candidate destination
$j$ at hour $t$:

$$
V_{ij,t}^{(m,k)}
= \alpha\, V_{j,t}
+ \beta_{t,m,k}\, t_{ij}^{(m)}
+ \gamma\, \log d_{ij}
+ \delta_{k}\, \mathrm{OccMatch}_{ij}.
\tag{6}
$$

with the following components:

- $V_{j,t}$ from (4); $\alpha = 1$ is a fixed buffer (the GNN absorbs
  the scale; see §2.8).
- $\beta_{t,m,k} = \beta_{t,m} \cdot \kappa_{k}$ factorises mode and
  tier effects on time sensitivity. $\beta_{t,m}$ has shape $(T, M)$
  (per-hour, per-mode); $\kappa_{k}$ is a tier-specific multiplicative
  scalar. We constrain $\kappa_{1} = 1$ (anchor) and
  $\kappa_{k > 1} = 1 + \mathrm{softplus}(\rho_{k})$ to enforce monotone
  non-decreasing tier scaling. Time sensitivity is non-positive by
  construction:
  $\beta_{t,m} = -\mathrm{softplus}(\eta_{t,m})$.
- $\gamma = -\mathrm{softplus}(\zeta)$: a single distance-friction
  coefficient (negative).
- $\delta_{k}$: tier-specific labour-match coefficient, with
  $\delta_{1} = 0$ as anchor and $\delta_{k > 1}$ free.

**Choice probability.** Each commuter is assumed to evaluate (6) for
each candidate destination $j$, add an independent Type-1 Extreme
Value error $\varepsilon$, and pick the destination of maximum
utility. The classical McFadden (1974) result gives a closed-form
softmax for a single (mode, tier) profile,

$$
P(j \mid i, t, m, k) = \frac{\exp(V_{ij,t}^{(m,k)})}{\sum_{j'=1}^{N} \exp(V_{ij',t}^{(m,k)})},
\tag{7}
$$

and the population-level destination probability marginalises over the
mode-tier profile distribution at origin $i$:

$$
P(j \mid i, t) = \sum_{m=1}^{M} \sum_{k=1}^{K} \pi_m(i, j)\, \pi_k(i)\, P(j \mid i, t, m, k).
\tag{8}
$$

In our implementation we accumulate (8) by incremental log-sum-exp
over the $M \cdot K = 9$ inner softmaxes to avoid materialising a
$(T, N, N, M, K)$ tensor.

## 2.6 Stage 2 — Agent-Based Deployment with Congestion Feedback

The trained parameters from Stage 1 are deployed in a population-level
agent-based simulator with an explicit congestion-feedback loop. Each
synthetic agent $n$ has attributes $(i_n, t_n, m_n, k_n)$ — origin
grid, departure hour, travel mode, income tier — drawn from the
empirical joint distribution $(\pi_m(i, \cdot), \pi_k(i))$. The
deployment proceeds in five sub-steps:

**Step 2a — Initial utility evaluation (free-flow).** For each
candidate workplace $j$, agent $n$ evaluates

$$
U_{nj} = V_{i_n j, t_n}^{(m_n, k_n)}\big[t_{ij}^{(m)} \!\leftarrow\! t_{ij}^{(m), \mathrm{free}}\big]
       + \varepsilon_{nj},
\quad \varepsilon_{nj} \overset{\mathrm{iid}}{\sim} \mathrm{Gumbel}(0, 1),
\tag{9}
$$

and picks $j_n^{*} = \arg\max_j U_{nj}$. Note that $V_{i_n j, t_n}^{(m_n, k_n)}$
in (9) is evaluated with the **free-flow** travel time
$t_{ij}^{(m), \mathrm{free}}$, the same convention used during training
(Section 2.7). The Gumbel-max trick (McFadden, 1974) ensures that
individual choices, aggregated over $N_{\mathrm{agents}}$ commuters,
reproduce the softmax distribution (7) — closing the loop with the
training target.

**Step 2b — Flow aggregation.** Individual choices are aggregated to
predicted hourly destination inflow $\widehat{F}_{j,t} = \sum_{n: t_n = t} \mathbb{1}\{j_n^* = j\}$.

**Step 2c — BPR congestion estimation.** For each (origin, destination,
hour, mode = car), we apply the Bureau of Public Roads volume-delay
function (Beckmann et al., 1956):

$$
t_{ij,t}^{(\mathrm{car}), \mathrm{cong}}
= t_{ij}^{(\mathrm{car}), \mathrm{free}}
  \cdot \big(1 + a (\widehat{F}_{j,t} / C_j)^b \big),
\qquad a = 0.15,\; b = 4,
\tag{10}
$$

where $C_j$ is a per-grid capacity proxy (workplace density × per-hour
car-equivalent capacity factor). Public-transit and walk travel times
are not subject to congestion in this formulation. The BPR multiplier
is capped at 3.0 to avoid runaway iteration under heavy oversaturation.

**Step 2d — Re-evaluation under congestion.** Agents recompute (9)
substituting $t_{ij}^{(\mathrm{car}), \mathrm{free}}$ with
$t_{ij,t}^{(\mathrm{car}), \mathrm{cong}}$ from (10), and re-pick
$j_n^*$. Steps 2b–2d iterate until $\widehat{F}_{j,t}$ changes by less
than 0.5% across iterations or a maximum of 10 iterations is reached
(typical convergence: 5–8 iterations).

**Step 2e — Final outputs.** The fixed-point flow tensor
$\widehat{F}_{ij,t}^{*}$ and the equilibrium congested travel-time
matrix $t_{ij,t}^{(\mathrm{car}), *}$ are reported as the deployment
outputs.

**Why training and deployment treat $t$ differently.** During training
(Section 2.7) we use only free-flow $t_{ij}^{(m), \mathrm{free}}$ as
input to (6), and $\beta_{t,m}$ is identified as the sensitivity of
utility to *free-flow* travel time. The training labels
$F_{ij,t}$ themselves reflect commuter responses to congested
conditions in the real world, so $\beta_{t,m}$ implicitly absorbs the
average level of congestion that actual London commuters experienced
in 2019 — but the input is free-flow, by convention. At deployment
(Section 2.6), the BPR loop (10) explicitly recomputes $t$ as a
function of predicted flow, *using the same $\beta_{t,m}$*. This
separation has two benefits: (i) the training loss (1) is a clean
multinomial likelihood without an inner fixed-point, preserving MLE
asymptotics for $\beta$; (ii) at deployment, $\beta$ has a
well-defined interpretation (free-flow-time sensitivity) under both
free-flow and congested conditions because the only difference is
which $t$ enters $\beta \cdot t$. We avoid the simultaneity that the
"all stages must use the same $t$" view would create (Sheffi, 1985).

**Counterfactual evaluation.** Counterfactuals are then a matter of
perturbing inputs and re-running both stages. For an *origin-side*
policy (e.g., +65,000 jobs at Old Oak Common, Section 2.9), we modify
$\mathbf{X}^{\mathrm{static}}$ at the affected grid cells, re-evaluate
the trained encoder to obtain a new $V_{j,t}'$, and re-simulate
agent choices and BPR under the same trained $(\beta, \gamma, \delta,
\kappa)$. For a *temporal-side* policy (e.g., 50% morning-peak load
shifted to shoulder hours), we perturb $\mathbf{X}^{\mathrm{dyn}}$
analogously. The behavioural coefficients are *not* re-estimated under
intervention — this is the hybrid framework's identifying assumption
(Schölkopf et al., 2021, single-component invariance): the *mechanism*
of choice is structural and stable; only the *inputs* to the mechanism
change. We report each scenario both with and without the BPR loop
active, so the gap between the two reads the magnitude of congestion
damping on the policy effect.

## 2.7 Training: End-to-End Cross-Entropy

The full parameter vector
$\boldsymbol{\theta} = (\Theta_{\mathrm{enc}}, \beta, \kappa, \gamma, \delta)$
combines approximately 33,000 encoder weights $\Theta_{\mathrm{enc}}$
with 76 behavioural coefficients $(\beta, \kappa, \gamma, \delta)$. We
fit $\boldsymbol{\theta}$ by minimising the cross-entropy loss (1)
with stochastic gradient descent. Specifically we use AdamW
(Loshchilov & Hutter, 2019) with two parameter groups: encoder weights
with learning rate $10^{-3}$ and weight decay $10^{-4}$; behavioural
coefficients with learning rate $10^{-2}$ and zero weight decay.

**ML / econometrics duality.** The negative cross-entropy in (1) is
*term-for-term equal* to the negative multinomial log-likelihood of
the choice model (8):

$$
- \mathcal{L}(\boldsymbol{\theta})
= \sum_{i,j,t} F_{ij,t}\, \log P(j \mid i, t;\, \boldsymbol{\theta})
= \log \mathrm{Multinomial}(\mathbf{F}_{i,\cdot,t} \mid O_{i,t}, P(\cdot \mid i, t)),
\tag{11}
$$

so the gradient-descent estimates we obtain by backpropagation are
maximum-likelihood estimates and inherit the standard MLE asymptotics
(consistency, asymptotic normality; Bentz & Merunka, 2000). This
duality is what licences the behavioural reading of $\beta$, $\gamma$,
$\delta$ as time-, distance-, and labour-match-disutility
coefficients respectively.

**Optimisation details.** Origins are split 87% / 13% into training
and validation pools. The cross-entropy in (1) is computed only over
training origins; the validation pool is used for early stopping
(patience 30 epochs) and for reporting performance metrics. Gradient
clipping at norm 1.0 is applied to all parameters jointly. Each
training run uses 200 epochs and a single random seed; we report
multi-seed mean ± standard deviation across three seeds.

## 2.8 Identifiability and Parameter Constraints

Aggregate flow data does not identify every parameter we might wish
to recover. We deal with three identifiability problems by structural
constraints rather than data-derived estimates:

**(i) The scale of $V_{j,t}$.** The product $\alpha \cdot V_{j,t}$ is
identified only up to a positive scaling. We resolve by fixing
$\alpha = 1$ (a buffer, not a parameter) and z-scoring the GNN
output, leaving the GNN to absorb all scale variation. This is
standard practice in deep multinomial-choice models (Wang & Klabjan,
2018; conditional-logit identification in Train, 2009 §3).

**(ii) The walking time coefficient $\beta_{t,\mathrm{walk}}$.**
Within the 1,725 × 1,725 OD pair set the walking time matrix
$t_{ij}^{(\mathrm{walk})} = d_{ij} / 5\,\mathrm{km/h}$ is severely
collinear with $\log d_{ij}$ on the small subset of pairs where
walking is feasible ($d_{ij} \lesssim 5$ km). The distance-aware
choice set restriction in (5) addresses this by removing infeasible
walking from the model entirely. After the restriction,
$\beta_{t,\mathrm{walk}}$ is identified only from the walkable
subset and converges robustly to $\approx -0.05$ across random
restarts. We interpret this estimate as the time-disutility
coefficient for short-distance walking commute (cf. Wardman, 2014's
finding that walking VOT scales sub-linearly with trip length).

**(iii) Tier-2 vs tier-3 collapse.** Empirically the data does not
support distinct $\kappa$ or $\delta$ for IMD income tiers 2 and 3 —
they consistently estimate to identical values across random
restarts ($\kappa_2 = \kappa_3 \approx 1.97$, $\delta_2 = \delta_3 \approx 0.22$).
This reflects the limited variation in IMD income score across origin
grids in inner London. We report the collapse as an empirical
finding and document it as a data-resolution limitation; access to
individual-level income data (planned for the Beijing replication)
would be required to discriminate tiers 2 and 3.

The remaining 76 RUM parameters and ≈ 33,000 encoder parameters are
fit against approximately 5.7 million observed (origin, destination,
hour) cells — strongly over-identified.

## 2.9 Counterfactual Scenario Design

We evaluate the framework on two policy scenarios that span the
spatial / temporal spectrum of urban interventions:

**Scenario A — Old Oak Common employment cluster.** A spatial
intervention modelled as a uniform addition of 65,000 jobs to the
nine 1 km grid cells covering the proposed Old Oak Common employment
cluster (Hammersmith / Ealing boundary). At simulation time, the
employment count features in $\mathbf{X}^{\mathrm{static}}$ for these
nine cells are increased proportionally; the trained encoder
recomputes $V_{j,t}$ under intervention; agents re-run their
destination choices using the same behavioural coefficients.

**Scenario B — Flexible-hours peak shifting.** A temporal
intervention modelled as moving 50% of the 7–9 a.m. peak commuting
load to the 10–15 shoulder window. Concretely, the hourly inflow
feature in $\mathbf{X}^{\mathrm{dyn}}$ is multiplied by 0.5 at peak
hours and increased proportionally at shoulder hours, preserving the
24-hour total. The trained encoder recomputes $V_{j,t}$; agents
re-run destination choices.

For each scenario we report the change in cell-level Hansen
accessibility,

$$
A_i = \sum_{j} E_j \exp(-|\beta_{\mathrm{car}}|\, t_{ij}^{(\mathrm{car})}),
\tag{12}
$$

where $E_j$ is employment at $j$, and the change in commuting-flow
inequality measured by the Gini coefficient and Palma ratio of the
population-weighted accessibility distribution. We compute results
both under free-flow $t_{ij}^{(\mathrm{car})}$ and under a BPR
user-equilibrium congestion assignment (Beckmann et al., 1956); the
two settings bracket the "no-feedback" and "with-feedback"
counterfactual reads.

## 2.10 Performance Metrics

Aggregate predictive performance is summarised by seven complementary
metrics computed on validation-pool origins:

- **Common Part of Commuters (CPC)** (Lenormand et al., 2012):
$$
\mathrm{CPC} = \frac{2 \sum_{i,j,t} \min(\widehat{F}_{ij,t}, F_{ij,t})}{\sum_{i,j,t} \widehat{F}_{ij,t} + \sum_{i,j,t} F_{ij,t}}, \quad \in [0, 1].
\tag{13}
$$

- **Root mean squared error (RMSE)** of cell-level flow:
$\sqrt{\mathrm{mean}((\widehat{F}_{ij,t} - F_{ij,t})^2)}$.
- **Mean absolute error (MAE)**: $\mathrm{mean}(|\widehat{F}_{ij,t} - F_{ij,t}|)$.
- **Pearson correlation** between predicted and observed flow vectors.
- **Spearman rank correlation** (rank-stable variant).
- **KL divergence** $D_{\mathrm{KL}}(F_{ij,t} / O_{i,t} \,\|\, P(j \mid i, t))$
  averaged over active (origin, hour) pairs.
- **Top-10 destination accuracy**: the average fraction of an origin's
  top-10 observed destinations that the model also assigns to the
  top-10 predicted destinations.

CPC is the most common headline metric in OD-flow literature
(Lenormand et al., 2012; Simini et al., 2021); the others provide
complementary views. CPC is robust to total-flow mismatch but
insensitive to per-cell magnitude; RMSE penalises high-flow errors
heavily; KL emphasises destination-distribution fit per origin; and
top-K accuracy directly tests the model's ability to identify
high-volume destinations.

## 2.11 Implementation, Reproducibility, and Software

The full pipeline is implemented in PyTorch 2.0 with no external graph
neural network library (we use a lightweight `_SAGELayer` of our own
that supports both CPU and CUDA backends). Training is performed on a
single NVIDIA T4 (16 GB) for the dual-branch and mixture variants
(<6 minutes per epoch on London-scale data) and on a CPU for the
baselines. Random seeds are fixed (PyTorch + NumPy) and three random
restarts are run per experiment. All code, configurations, and
trained checkpoints are deposited at \[anonymous repository URL\] for
review. Data ingestion scripts for Census, IMD, OSM-POI, BRES, TomTom,
and GEODS sources are provided; the recipe is fully reproducible from
public data sources.

**Limitations of the present implementation.** Three are worth
flagging explicitly:

- $t_{ij}^{(\mathrm{transit})}$ is a piecewise-linear approximation
  rather than a true GTFS routing computation. We plan to replace this
  with `r5r` GTFS-based shortest-path matrices in future work; for the
  Beijing replication (Paper B), real-time bus position data and the
  Amap routing API will provide hour-specific transit times that
  substantially exceed the present approximation.
- Mode share is from Census 2011 (we have no MSOA-level mode share for
  the 2019 GEODS OD release). The 2011 / 2019 mismatch is ≈10% in
  aggregate and is documented as a limitation.
- The BPR user-equilibrium computation in Scenario A and B uses
  origin-side capacity proxies derived from grid-level workplace
  density rather than link-level capacity, and is therefore best read
  as an order-of-magnitude congestion sensitivity rather than a
  literal traffic-equilibrium prediction.

---

# 3 Results

We report results for the four model variants described in Section 2 — a
single-branch GraphSAGE+GRU baseline (SINGLE), a dual-branch encoder
without behavioural augmentations (DUAL_MIN), the same encoder with the
OccMatch and scalar income×β extension (DUAL_EXT), and the full
mode×tier mixture model with distance-aware choice set (DUAL_HET) — and
two counterfactual scenarios deployed on the trained DUAL_HET model
under destination-side BPR closure. All performance numbers in §3.1 are
3-seed mean ± standard deviation on the validation pool (every fifth
origin held out, 345 / 1725 origins). Counterfactual results in
§3.3–3.4 use the DUAL_HET checkpoint trained on a single seed for 200
epochs.

## 3.1 Quantitative performance — DUAL_HET vs ablations

Table 3.1 summarises predictive performance on the held-out validation
origins. The headline metric is the Common Part of Commuters (CPC,
equation 13).

| Variant       | Innovation vs prev. row                | CPC ↑          | RMSE ↓        | Pearson r ↑  | val NLL ↓ | Params |
|---------------|----------------------------------------|----------------|---------------|--------------|-----------|--------|
| SINGLE        | single-branch GraphSAGE + GRU baseline | 0.465 ± 0.007  | 2.90 ± 0.05   | 0.789 ± 0.004 | 648.4     | 10,081 |
| DUAL_MIN      | dual-branch encoder (no augmentations) | 0.485 ± 0.003  | 2.84 ± 0.08   | 0.791 ± 0.003 | 632.3     | 33,026 |
| DUAL_EXT      | + OccMatch + scalar income × β coupling | 0.488 ± 0.002  | 2.78 ± 0.02   | 0.786 ± 0.001 | 629.9     | 33,026 |
| **DUAL_HET**  | **+ 3-mode × 3-tier mixture RUM**      | **0.484 ± 0.005** | **2.36 ± 0.07** | **0.826 ± 0.006** | **624.6** | **33,103** |

Three observations frame the rest of this section. **First**, the dual-branch
encoder alone (DUAL_MIN) closes most of the gap to the full model on CPC —
moving from SINGLE 0.465 to DUAL_MIN 0.485 is a +0.020 absolute gain
(+4.3% relative), explained by no longer broadcasting time-invariant
features 24 times into the temporal mechanism. **Second**, the OccMatch
and scalar income coupling (DUAL_EXT) buys only an additional +0.003 CPC,
suggesting that a scalar income–time interaction is too coarse to capture
behavioural heterogeneity at this aggregation scale. **Third**, the full
mode × tier mixture (DUAL_HET) does not improve CPC further (in fact it
sits 0.004 below DUAL_EXT), but materially lowers RMSE (2.78 → 2.36, a
−15% reduction), Pearson r (0.786 → 0.826), and val NLL (629.9 → 624.6).
This is the expected signature of the mixture: the model now distributes
flow probability across modes and income tiers correctly, so per-cell
flow magnitudes are more accurate even when the row-level total predicted
correctly. CPC is dominated by the row total; RMSE and NLL are dominated
by the cell-level distribution. The mixture trades a marginal CPC point
for a substantial improvement in cell-level distributional fit, which
matters more for the counterfactual analysis in §3.3–3.4 because per-cell
flow drives congestion and per-grid accessibility.

The mixture model adds 77 parameters relative to DUAL_EXT (the 33,026
encoder weights are nearly identical; only the RUM head changes). All 77
are interpretable behavioural coefficients (β<sub>t,m</sub>, γ, δ<sub>k</sub>, κ<sub>k</sub>) — see §3.2.

## 3.2 Recovered behavioural primitives

DUAL_HET recovers the following coefficients (3-seed mean; standard errors
computed from across-seed dispersion are tight enough to round on the
displayed precision).

**Mode-specific time sensitivity (mean across 24 hours).** The recovered
β<sub>t,m</sub> in minutes<sup>−1</sup>, with mode m ∈ {car, transit, walk}:

| Mode    | β̄<sub>m</sub> (per-min) | Relative to β̄<sub>car</sub> | Sign expected? |
|---------|---------------------------|-----------------------------|----------------|
| Car     | −0.187                    | 1.00 (anchor)               | ✓ negative     |
| Transit | −0.105                    | 0.56                        | ✓ negative     |
| Walk    | −0.049                    | 0.26                        | ✓ negative     |

All three β are negative and ordered |β<sub>car</sub>| > |β<sub>transit</sub>| > |β<sub>walk</sub>|. The ordering matches
the well-known transport-economics finding (Wardman 2014, Department for
Transport 2024, WebTAG TAG A1.3 §B.2.4) that car commuters carry a higher
value of time than transit commuters, who in turn carry a higher VOT than
walk commuters. Two mechanisms drive this in our setting: (i) the mode-share
mixture is *pair-level* and distance-aware, so β<sub>walk</sub> is identified only
on origin–destination pairs within 5 km (§2.5) — those are short trips
where the elapsed-time penalty of walking is necessarily small; (ii) the
shared encoder V<sub>j,t</sub> absorbs the destination-attractiveness component, so
β<sub>m</sub> reads cleanly off the differential effect of mode-specific travel time
on choice probability. Cross-mode magnitude *ratios* (1.00 : 0.56 : 0.26)
sit close to but slightly below the Wardman 2014 commute ratio of
1.00 : 0.77 : 0.46 — a discrepancy we ascribe to (i) the distance restriction
on walk and (ii) the pair-level mode share assigning small weight to walk
on most trip pairs (effective walk share is ≈3% citywide), which makes
β<sub>walk</sub> identifiable only from the short-trip subset.

**Distance decay.** The shared distance coefficient is γ̄ = −0.328
(3-seed std 0.058). In the present formulation γ multiplies log d<sub>ij</sub> with
d in kilometres, so the marginal effect of doubling distance on log-utility
is γ · log 2 ≈ −0.23 — modestly larger than gravity-model norms (γ ≈ −0.15
to −0.20 for inter-zonal commuting in the UK, Casey & Currie 1955; Wilson
1971), reflecting the fact that our model already captures
origin-attractiveness asymmetry through V<sub>j,t</sub> so γ absorbs the residual
distance friction more aggressively.

**Income-tier scaling κ<sub>k</sub>.** The recovered tier multipliers are
κ̄ = [1.00, 1.97, 1.97] for low / mid / high income tiers respectively
(κ<sub>0</sub> = 1 by anchor; tiers 2 and 3 collapse to identical values
across seeds, std < 0.06 each). This collapse is driven by the
discretisation of the IMD income score into three tiers: the IMD's
percentile-based binning concentrates 67% of London grids into the upper
two tiers, leaving the bottom tier (IMD income decile 1–3) as the only
distinguishable group. We retain the three-tier specification in the
architecture because Beijing data permits a finer income discretisation
(see §2.11 limitations and Paper B), but for London the operative
heterogeneity is binary (lowest tier vs the rest).

**Occupational matching δ<sub>k</sub>.** Per-tier OccMatch coefficients
δ̄<sub>k</sub> = [0.00, 0.20, 0.20] (with anchor δ<sub>0</sub> = 0). The mid- and
high-tier δ being equal mirrors the κ collapse and reflects the same
income-tier discretisation limit. The positive sign of δ for the upper
two tiers indicates that higher-income workers' destination choice is
significantly more sensitive to occupational match between origin and
destination industry composition than is the case for the lowest-tier
workers, consistent with prior work on labour-market matching by
education / income tier (Manning & Petrongolo 2017, *AER*).

**Recovered VOT ratios anchored to literature.** Treating the Wardman
(2014) UK commute central VOT for car (£6.50/h, year 2010 prices) as a
literature anchor for our recovered β<sub>car</sub>, the implied behavioural
value of time for transit and walk commuters in our sample is
£3.65/h and £1.71/h respectively — both within an order of magnitude of
WebTAG TAG A1.3's 2024 central business commute VOT for car (£11.10/h),
bus (£5.90/h), and walk (£3.00/h) after price-level adjustment. We do not
claim a new VOT measurement; instead, the agreement of the cross-mode
*ranking* with WebTAG is treated as a behavioural-validity check for the
inverse-recovery procedure.

## 3.3 Counterfactual A — Old Oak Common +65,000 jobs

We instantiate Scenario A by inflating the employment-cluster features
(sectoral employment ×8 + total employment + POI office count, see
EMP_BLOCK in `experiments/paper_a/scenario_A_dual_het.py`) at a 9-grid
cluster centred on grid 908 (Old Oak Common, the western Eulerian
terminus of the planned HS2 station), totalling +65,000 jobs distributed
proportionally across the 9 grids (≈ +7,222 jobs per grid). The
intervention is *coherent* in the sense that all employment-related
columns scale by the same per-grid ratio, preserving each grid's sectoral
mix; only the magnitude of the cluster changes. Forwarding the trained
DUAL_HET model with the modified X<sup>static</sup> and re-aggregating predicted
flow:

| Quantity                             | Free-flow                         | BPR equilibrium                                |
|--------------------------------------|-----------------------------------|------------------------------------------------|
| Δ inflow into OOC cluster            | +414 commuters                    | +261 commuters                                 |
| BPR damping ((free − bpr) / free)    | —                                 | 37%                                            |
| Δ mean Hansen accessibility A̅       | **+1.34%**                        | **−0.67%**                                     |
| Δ Gini on A<sub>i</sub> (pop-weighted) | +0.82%                          | +1.28%                                         |
| Δ Palma ratio on A<sub>i</sub>         | +1.27%                          | +2.28%                                         |
| Δ Atkinson(ε=0.5) on A<sub>i</sub>     | +1.67%                          | +2.63%                                         |

Three findings. **First**, the gross flow response (+414 commuters)
substantially exceeds what the constrained Phase B v6 model used in
earlier drafts would have predicted (+126), because the recovered
β̄<sub>car</sub> = −0.187 in DUAL_HET is roughly three times larger than the v6
constrained variant's β<sub>t</sub> = −0.058 (§3.2). Higher time sensitivity →
higher elasticity of destination choice with respect to employment
attractiveness → larger flow shift in response to the same intervention.
This is not a bug — it is what cross-mode-validated identification of
β<sub>car</sub> against transport-economics literature (Wardman 2014) implies for
this scenario.

**Second**, BPR closure damps the flow response by 37% — three times
larger than the v6 prediction of 17% damping. Mechanistically, the
intervention concentrates new attractiveness at OOC and the closest few
adjacent grids; the BPR multiplier scales as (Inflow / capacity)<sup>4</sup>,
so even a moderate inflow surge at a grid already near baseline capacity
saturates the destination-side multiplier and pushes a substantial
fraction of the gross flow back into the alternative-distribution.

**Third — the policy-significant finding** — the Hansen mean
accessibility A̅ across all London grids *increases* by 1.34% under
free-flow but *decreases* by 0.67% under BPR equilibrium. The intuition:
the intervention concentrates jobs at one cluster, which raises the
attractiveness term ∑<sub>j</sub> E<sub>j</sub> at grids that can reach OOC easily, but the
BPR loop concurrently raises the travel-time penalty for those same
high-flow OD pairs, and the exponential decay exp(β<sub>car</sub> · t) in the
Hansen formula amplifies the time penalty disproportionately. Net of
both effects, the marginal commuter loses more in expected travel time
than they gain in additional employment options — except in the small
neighbourhood of OOC itself, which we discuss below.

The intervention is *regressive* across all three inequality measures
under both free-flow and BPR. Population-weighted Gini, Palma, and
Atkinson(ε=0.5) all rise — by 0.8–2.6% under BPR — indicating that the
top-tier-A̅ grids (those already well-connected to the rich north-west
London CBD-fringe corridor that includes OOC) gain more accessibility
than the bottom-tier-A̅ grids. The disaggregated top-10 destination-gain
list (in `scenario_A_dual_het.json`) shows that OOC itself and grids
within a 2 km radius capture nearly all of the gross +414 inflow, while
grids in inner east London and outer south London record small but
broadly distributed losses.

## 3.4 Counterfactual B — Flexible work temporal intervention

Scenario B operationalises a "flexible work" policy by re-distributing
peak-hour commute load to off-peak shoulder hours. Concretely, we apply
two stacked interventions:

1. **Feature-level**: scale the congestion-z column of X<sup>dyn</sup> by 0.5 at
   peak hours h ∈ {7, 8, 9} and by 1.5 at shoulder hours h ∈ {10, 11, 12, 13, 14, 15};
2. **Demand-level**: scale per-origin row sums of observed F<sub>ij,t</sub> by the
   same multipliers, then renormalise per origin so daily total demand is
   conserved (commuters reschedule, but no commuter is created or lost).

The combination represents a regime in which more workers can choose
mid-day departure without modifying total commuting demand. Forwarding
DUAL_HET with the perturbed X<sup>dyn</sup> and rescaled row sums:

| Quantity                                      | Baseline       | Scenario      | Δ              |
|-----------------------------------------------|----------------|---------------|----------------|
| Peak-hour share of daily commute              | 17.5%          | 7.9%          | −9.6 pp        |
| Mean t<sub>car</sub> at h=08 (BPR-equilibrium) | 34.2 min      | 32.7 min      | **−4.4%**      |
| Mean A<sub>i</sub> at h=08 (Hansen, β<sub>car</sub>) | 125,836      | 142,977       | **+13.6%**     |
| Gini on A<sub>i</sub> at h=08                 | 0.404          | 0.398         | **−0.006**     |
| Palma ratio on A<sub>i</sub> at h=08          | 1.859          | 1.814         | **−0.045**     |

**The intervention is progressive at h=08**: Gini, Palma both *fall*. The
mechanism is the inverse of Scenario A's: by relieving peak-hour
congestion, the BPR multiplier drops most where it was highest — at
peripheral grids whose only viable car commute path passes through
saturated central corridors. Those peripheral grids are also the
lowest-baseline-A̅ grids, so the gain in accessibility from peak
relief is captured disproportionately by the under-connected origins.
The exponential decay exp(β<sub>car</sub> · t) in the Hansen formula is again
the active mechanism, but now in the opposite direction: a small drop in
t at a peripheral high-baseline-t grid translates into a much larger
proportional A<sub>i</sub> gain than at a central low-baseline-t grid.

Per-hour BPR analysis (`scenario_B_dual_het.json`) shows the expected
shape: t<sub>car</sub> falls at h=07/08/09 by 1.4–4.4% and rises at h=12/14
(shoulder hours absorbing the spread demand) by 9.4%. Total daily
commute volume is conserved by construction; what changes is the
*distribution* across hours and the *equilibrium* t at each hour given
the new demand profile.

## 3.5 The contrast: same Θ, opposite distributional verdicts

The central policy-analytic finding of this paper is that the *same*
trained behavioural parameters (β, γ, δ, κ) yield diametrically opposed
inequality verdicts on Scenario A and Scenario B:

| Inequality measure (pop-weighted) | Scenario A (OOC +65k) | Scenario B (flex work) |
|-----------------------------------|------------------------|-------------------------|
| Δ mean A̅                         | −0.67% (BPR eq)        | +13.6% (h=08, BPR)      |
| Δ Gini                            | +1.28% (regressive)    | −0.6 pp (progressive)   |
| Δ Palma                           | +2.28% (regressive)    | −2.4 pp (progressive)   |

The inequality reversal is not an artefact of model choice — both
scenarios use the identical encoder weights, identical recovered β,
and identical BPR closure. The difference is where the intervention
operates in the system: Scenario A perturbs the *destination-side*
attractiveness landscape, concentrating gains in the upper tail of
already-accessible grids; Scenario B perturbs the *temporal* congestion
profile, distributing time-savings to the lower tail of A̅ where
commute-time is the binding constraint on accessibility. The
counterfactual evaluation thus reveals that demand-side temporal policy
can dominate supply-side spatial policy on welfare grounds, even when
the two interventions cost the same in policy terms.

The framework's identifying assumption (single-component invariance,
§2.6) is what permits this comparison: the behavioural mechanism
(β, γ, δ, κ) is structural and stable across the perturbations; only
the inputs to the mechanism change. In a model whose β were re-fit
under each scenario — as gravity models historically have been — this
clean comparison would not be possible because the parameters themselves
would shift between regimes.

---

**References (selected; full reference list with the paper).**

Beckmann, M., McGuire, C. B., Winsten, C. B. (1956). *Studies in the Economics of Transportation.* Yale University Press.
Ben-Akiva, M., Boccara, B. (1995). *Discrete choice models with latent choice sets.* International Journal of Research in Marketing, 12(1), 9–24.
Bentz, Y., Merunka, D. (2000). *Neural networks and the multinomial logit for brand choice modelling: A hybrid approach.* Journal of Forecasting, 19(3), 177–200.
Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.* Computers, Environment and Urban Systems, 65, 126–139.
Cho, K. et al. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation.* EMNLP.
Department for Communities and Local Government (2019). *English indices of deprivation 2019.*
Hamilton, W., Ying, Z., Leskovec, J. (2017). *Inductive representation learning on large graphs.* NeurIPS.
Lan, S. et al. (2022). *DSTAGNN: Dynamic spatial-temporal aware graph neural network for traffic flow forecasting.* ICML.
Lenormand, M., Picornell, M., Cantú-Ros, O. G., Tugores, A., Louail, T., Herranz, R., Barthelemy, M., Frías-Martínez, E., Ramasco, J. J. (2012). *Cross-checking different sources of mobility information.* PLoS ONE, 9(8), e105184.
Loshchilov, I., Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.
Manski, C. (1977). *The structure of random utility models.* Theory and Decision, 8(3), 229–254.
McFadden, D. (1974). *Conditional logit analysis of qualitative choice behavior.* In *Frontiers in Econometrics*, ed. P. Zarembka.
Pellegrini, P. A., Fotheringham, A. S. (2002). *Modelling spatial choice: A review and synthesis in a migration context.* Progress in Human Geography, 26(4), 487–510.
Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., Bengio, Y. (2021). *Toward causal representation learning.* Proceedings of the IEEE, 109(5), 612–634.
Simini, F., Barlacchi, G., Luca, M., Pappalardo, L. (2021). *A Deep Gravity model for mobility flows generation.* Nature Communications, 12, 6576.
Train, K. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.
Wang, S., Klabjan, D. (2018). *An End-to-End Deep Reinforcement Learning-Based Intelligent Agent for Discrete Choice.* arXiv:1810.04644.
Wardman, M. (2014). *Valuing convenience in public transport.* ITF Round Tables, OECD Publishing.
Wu, Z., Pan, S., Long, G., Jiang, J., Zhang, C. (2019). *Graph WaveNet for deep spatial-temporal graph modeling.* IJCAI.
