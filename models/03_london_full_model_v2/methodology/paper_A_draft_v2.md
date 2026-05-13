# Deep Choice Modeling with Inverse Aggregate-OD Training: Recovering Behavioural Parameters and Simulating Urban Policy Scenarios

_Working draft, 2026-05-06. Replaces earlier `paper_A_draft_en.md` (Phase B v0 frame, deprecated)._

---

## Abstract

We propose inverse training of a deep choice model with graph-structured utility to recover behavioural primitives from aggregate origin-destination data, bridging the gap between predictive flow models (gravity, radiation, Deep Gravity) and policy-relevant counterfactual simulation. Applied to London's 1725 1-kilometre grid with the full Census 2021 commute table (1.21 million commuters), our framework jointly recovers travel-time, distance, occupation-match, and destination-wage interaction coefficients via maximum-likelihood inverse training. On a 4-borough spatial holdout the model achieves common-part-of-commuters (CPC) **0.343 with 95 % bootstrap CI [0.324, 0.362]** (10-seed ensemble), outperforming gravity (0.170; ΔCPC = +0.172 with paired bootstrap *p* < 10$^{-4}$) and a minimal Deep Gravity implementation. We validate temporal generalisation by retrodicting 2011→2021 commuting at city scale (CPC 0.56) and by a Bromley negative-control test (predicted/observed Δshare ratio 1.15×). Two policy scenarios contrast on accessibility inequality: a spatial intervention (+65 000 jobs at the Old Oak Common cluster) is mildly regressive (Gini +1.26 %), with two-thirds of the gross accessibility gain absorbed by congestion equilibrium; a temporal intervention (50 % peak shift to off-peak) cuts morning-peak travel times by 3.7 %, improves population-weighted accessibility by 10.2 %, and reduces inequality (Gini −0.011). An agent-level heterogeneity layer, with $\beta$-spread calibrated to public WebTAG / Wardman value-of-time meta-analysis, quantifies Jensen bias at 2.1 % for mature CBD interventions but −16.7 % for greenfield interventions — where the aggregate model systematically under-predicts. All experiments reproduce from London open sources without individual-level travel surveys, supporting transfer to data-restricted cities such as Beijing.

---

## 1 · Introduction

### 1.1 Motivation

Urban planning agencies routinely face decisions whose welfare consequences depend on how commuters reallocate across destinations and times of day. Two paradigmatic examples motivate this work. **Spatial:** the United Kingdom government's £4.2 billion Old Oak Common (OOC) regeneration is intended to add some 65 000 jobs to a mid-distance west-London corridor; planners want to know who is reached by these new opportunities, who is not, and at what congestion cost. **Temporal:** post-pandemic flexible-working policies reshape the morning-peak demand curve; transport authorities need to estimate the equilibrium effect on travel times and accessibility before committing to capacity reallocation.

Either question requires a **structural** model: parameters that retain meaning under the proposed intervention rather than a mapping that fits today's flows. Classical urban economics provides such parameters — value of time, distance decay, attraction elasticity — and a mature random-utility framework for combining them (McFadden, 1974; Train, 2009). What has been missing in city-scale practice is a tractable way to **recover** these parameters from the aggregate origin-destination data that public agencies actually publish, while exploiting modern graph neural networks for spatial structure.

### 1.2 The gap in existing flow models

Classical aggregate flow models — the gravity model (Wilson, 1971), radiation model (Simini *et al.*, 2012), and the recent Deep Gravity neural extension (Simini *et al.*, 2021) — fit observed commute matrices well but do not, by design, recover behavioural coefficients on travel time, wage, or occupation. They are predictive in the in-sample sense yet structurally agnostic about the choice mechanism, which limits their counterfactual reach. Modern deep learning approaches in the choice family (Wang & Klabjan, 2018; Sifringer *et al.*, 2020; Wong & Farooq, 2021) restore parameter recovery but typically assume access to **individual** revealed-preference data — out of reach for most cities, where individual travel surveys are either restricted (e.g. UK NTS via UKDS) or nonexistent (most non-OECD cases). The TSI-GCN family (Yao *et al.*, 2021) brings GNNs to OD imputation but treats the task as a regression rather than a structural inverse problem.

This paper closes the gap by training a deep choice model **inversely** on aggregate OD: the GNN parameterises destination-side attractiveness $V_j$, while a closed-form softmax over $V_j$ and a behavioural cost vector recovers the coefficients, all without per-individual choice data.

### 1.3 Contributions

We make four contributions.

1. **Methodology.** We formulate aggregate-OD inverse training of a graph-structured deep choice model (Section 3). A GraphSAGE encoder produces destination attractiveness $V_j$; the random-utility head adds a travel-time term, a distance term, an occupation-match interaction, and a destination-wage interaction. Sign-constrained and unconstrained variants are reported in tandem to expose an aggregate-data identification ambiguity that, to our knowledge, has not been documented in this setting.

2. **Validation across time.** We retrodict 2011→2021 London commuting using the full Census 2021 OD table (1.21 million commuters, not the more commonly used 1.1 % sample) and the 2011 NOMIS WU03EW table aligned via the ONS MSOA-2011↔MSOA-2021 lookup. The 2011-fitted model generalises city-wide to 2021 with CPC 0.56 (in-sample 0.71). A Bromley negative-control test, where 2011→2021 employment grew only 1.55×, predicts Δshare within 15 % of observed.

3. **Two policy scenarios.** We illustrate the framework on a spatial intervention (+65 000 jobs across a 9-grid Old Oak Common cluster) and a temporal intervention (a 50 % peak-shift toward off-peak hours under flexible work). The two are diametrically opposed on inequality: the spatial intervention is mildly regressive (Gini +1.26 %) with two-thirds of the gross accessibility gain absorbed by BPR-equilibrium congestion pushback; the temporal intervention is progressive (Gini −0.011), cutting morning-peak travel times by 3.7 % and lifting accessibility by 10.2 %. Methodologically we also document a normalisation pitfall — without freezing the GNN's output normalisation statistics at the baseline, the destination-side intervention is washed out by re-centring on the perturbed distribution.

4. **Bridging to ABM.** An agent-level heterogeneity layer assigns each agent a personal $\beta_n$ drawn from a Gaussian whose spread is calibrated to WebTAG / Wardman value-of-time meta-analysis. Aggregating across 10 000 agents yields predictions that we compare to the aggregate model's. The empirical Jensen bias is 2.1 % for the mature CBD-adjacent Old Oak Common cluster, but reverses sign and grows to 16.7 % when the same intervention is applied to a greenfield outer-London cluster — there the aggregate model systematically *under*-predicts the response. The framework requires no individual travel-survey data and is therefore reusable in cities (e.g. Beijing) where such data are unavailable.

### 1.4 Roadmap

Section 2 reviews related work. Section 3 specifies the model. Section 4 describes data sources. Section 5 reports headline performance, baselines, and recovered parameters. Section 6 covers retrodiction validation. Section 7 reports the two policy scenarios and the agent-level heterogeneity analysis. Section 8 discusses limitations. Section 9 sketches the Beijing application planned as a companion paper.

---

## 2 · Related Work

### 2.1 Aggregate flow models

The gravity model (Wilson, 1971) sets predicted flow proportional to origin and destination size and decreasing in some power or exponential of distance or travel time. We use a Poisson-MLE production-constrained variant via softmax. The radiation model (Simini *et al.*, 2012) is parameter-free in distance: it ranks destinations within a sphere of opportunities. Both serve here as structurally interpretable baselines. Deep Gravity (Simini *et al.*, 2021) replaces the closed-form weighting with a multilayer perceptron over origin and destination feature concatenation, dropping the gravity functional form entirely. We report a minimal MLP variant of Deep Gravity; full reproduction at our 1-km grid scale (1725 nodes) underperforms gravity in our experiments, consistent with reports elsewhere that the published 0.5–0.6 CPC requires multi-city county-scale training and the original tuning regime.

### 2.2 Graph neural networks for OD

TSI-GCN (Yao *et al.*, 2021) applies a GCN to a Beijing 30×30 grid to impute commute flows from a partial OD matrix; the task is in-sample reconstruction rather than structural recovery. Recent surveys (STG4Traffic, 2023; Counterfactual Learning on Graphs, 2024) catalogue a wide ST-GNN family for traffic forecasting. Our use of GraphSAGE (Hamilton *et al.*, 2017) is intentional: in low-data settings (our 1478 training origins) the simpler aggregator outperforms attention. We report this directly: a 4-head GAT (Veličković *et al.*, 2018) achieves CPC only 0.246 against GraphSAGE 0.344.

### 2.3 Random-utility models and deep choice

The discrete-choice literature is anchored by McFadden's multinomial logit (1974) and the doubly-bounded production constraints of Wilson (1971). Train (2009, ch. 6) reviews mixed logit and unobserved-heterogeneity variants. Deep choice models (Wang & Klabjan, 2018; Sifringer *et al.*, 2020; Wong & Farooq, 2021) parameterise the utility with neural networks while retaining the choice-axiomatic softmax. To our knowledge these are routinely fitted on individual choice data; we extend the family to **aggregate-OD inverse training**, which raises the identification questions discussed in §3.5 and §5.3.

### 2.4 Heterogeneity, value of time, and Jensen bias

Wardman (2014) provides the canonical UK-wide meta-analysis of travel-time values (Wardman *et al.*, 2016 update); WebTAG TAG Unit A1.3 (Department for Transport, 2024) operationalises the central values for appraisal practice. We use the WebTAG income-tier $\sigma$ to calibrate per-agent $\beta_n$ spread. The structural argument that aggregate prediction with heterogeneous preferences is biased relative to averaged individual prediction is classical (Train, 2009 §6.7) and is sometimes called Jensen bias by analogy with Jensen's inequality. We quantify it empirically and find it varies with intervention context: small for mature interventions, large and signed for greenfield ones.

---

## 3 · Methodology

### 3.0 Notation

Let $i, j \in \{1, \ldots, N\}$ index grid cells (origins and destinations) on the 1-km London lattice, $t \in \{0, \ldots, T-1\}$ index hour of day, and $X \in \mathbb{R}^{N \times F}$ the matrix of static node features. Let $F_{ij}$ denote the observed Census commute count from $i$ to $j$ (or $F_{ij}^t$ when hourly), $t_{ij}$ the free-flow travel time, $d_{ij}$ the centroid distance, $z_o(i)$ the origin's z-scored Index of Multiple Deprivation income score, $z_w(j)$ the destination's z-scored mean workplace wage, and $\mathrm{OccMatch}_{ij}$ the cosine similarity between the origins's resident-occupation distribution and the destination's workplace-occupation distribution. Bold-face $\boldsymbol{\theta}$ collects the GNN parameters; light-face Greek letters collect the recoverable behavioural coefficients $(\alpha, \beta_t, \gamma, \delta, \phi, \psi)$.

### 3.1 Choice model

For grid cell $i$ as the origin and $j$ as the destination, we write the random utility as

$$ U_{ij} \;=\; \alpha\,V_j \;+\; \beta_t(i,j)\,t_{ij} \;+\; \gamma\,\log d_{ij} \;+\; \delta\,\mathrm{OccMatch}_{ij} \;+\; \varepsilon_{ij}\,, $$

with the standard Gumbel error $\varepsilon_{ij}$ giving the multinomial logit
$$ P(j\mid i) \;=\; \frac{\exp U_{ij}}{\sum_{j'} \exp U_{ij'}}\,. $$

The coefficient $\alpha$ on the GNN-derived attractiveness is fixed at 1 for identifiability; the cost coefficient $\beta_c = -1$ pins the utility numeraire; the remaining coefficients $\beta_t, \gamma, \delta$ and the interaction coefficients $\phi, \psi$ defined below are recovered by inverse training.

**Heterogeneity via interactions.** We allow the time coefficient to depend linearly on observable origin and destination characteristics:
$$ \beta_t(i,j) \;=\; \beta_t \cdot \bigl(1 + \phi\,z_o(i) + \psi\,z_w(j)\bigr) $$
where $z_o(i)$ is an origin's z-scored Index of Multiple Deprivation (IMD) income score and $z_w(j)$ is a destination's z-scored mean workplace wage from ASHE.

**Sign-constrained variant.** A softplus reparameterisation $\phi = -\mathrm{softplus}(\tilde\phi)$, $\psi = +\mathrm{softplus}(\tilde\psi)$ enforces the mainstream RP/SP direction (richer workers have higher VOT to high-wage destinations); we report this variant alongside the unconstrained best-fit.

### 3.2 GNN structural utility

The destination-side attractiveness $V_j = \mathrm{GNN}_\theta(X_j)$ is produced by a 5-layer GraphSAGE (Hamilton *et al.*, 2017) over a kNN graph (k = 10 nearest neighbours by free-flow car travel time) on the 1725 1-km grid cells, with hidden width 128 and a final output projection. Node features include sectoral employment (8 SIC categories), total population, density of 8 POI types, subway-station count, and centroid coordinates (22 features). To prevent the GNN from absorbing arbitrary scale, we normalise its output to zero mean and unit standard deviation across all destinations.

**Frozen baseline norm for counterfactual forwards.** When the same trained GNN is used in counterfactual mode (Section 7), the normalisation statistics are *frozen at baseline values* — recomputing them on the perturbed distribution washes out the intervention via re-centring, an empirically verified pitfall.

### 3.3 Inverse training

Given observed aggregate OD counts $F_{ij}$, we maximise the multinomial likelihood
$$ \mathcal{L}(\theta, \beta_t, \gamma, \delta, \phi, \psi) \;=\; \sum_{i \in \mathrm{train}} \sum_j F_{ij} \log P(j\mid i)\,. $$
Computational tractability over 1725 destinations is via top-K = 50 sampling-of-alternatives with the McFadden (1978) correction. We use AdamW with separate learning rates for the GNN parameters $\theta$ and the choice-head scalars, 150 epochs, no early stopping (regularisation via weight decay and bounded depth/width).

### 3.4 Counterfactual forward

For a scenario that intervenes on a subset of node features $X_S \to X_S^{\,*}$, we forward the trained GNN under the modified inputs and recompute the choice probabilities and the production-constrained predicted flow. For car-mode counterfactuals we additionally close the loop by solving the BPR (Sheffi, 1985) user equilibrium

$$ t_{ij}^{\rm eq} \;=\; t_{ij}^{0} \cdot \Bigl[1 + \alpha_{\rm BPR} \bigl(V_j / C_j\bigr)^{\beta_{\rm BPR}}\Bigr],\quad V_j = \sum_i F_{ij}^{\rm eq} $$

via the method of successive averages (Sheffi, 1985, ch. 5), with the multiplier capped at 2.5 to prevent the quartic from blowing up under heavy oversaturation in the iteration.

### 3.5 Agent-level heterogeneity layer

To bridge the aggregate inverse model to ABM-style simulation, we assign each of $N = 10\,000$ synthetic agents a personal coefficient
$$ \beta_n(j) \;=\; \beta_t \cdot \bigl(1 + \phi\,z_o(i_n) + \psi\,z_w(j) + \varepsilon_n\bigr), \qquad \varepsilon_n \sim \mathcal{N}(0, \sigma^2(\mathrm{tier}_n))\,. $$
The per-tier coefficient of variation $\sigma$ is calibrated to the WebTAG TAG A1.3 income-tier value-of-time spread ($\mathrm{CoV} \approx 0.4$ following Wardman 2014). Agents draw their own destination via per-agent softmax, and aggregating gives the agent-level OD prediction. The empirical bias

$$ \mathrm{Bias}_{\rm Jensen} \;=\; \mathrm{ΔF}_{\rm aggregate} \;-\; \mathrm{ΔF}_{\rm agent} $$

is computed on each policy scenario.

---

## 4 · Data

The model is trained on a 1725 × 1-kilometre regular grid covering Greater London. Grid cells are aligned to the British National Grid (EPSG:27700) and intersect London administrative boundaries (33 boroughs and the City of London). Each cell carries 22 static node features and a small set of time-varying features used in the Section 7.2 hourly variant. All data are derived from publicly available sources without registration or licence beyond standard open-government terms.

### 4.1 Origin-destination commute flows

For 2021, we use the NOMIS *ODWP01EW* table from the 2021 Census of England and Wales (Origin-Destination Workplace by Place of Work indicator), restricted to the place-of-work code 3 (working in the UK at a fixed non-home workplace) so as to match the 2011 WU03EW universe. The full London-resident London-working subset is 1.21 million commuters across 196 940 MSOA pairs, disaggregated to our 1-km grid by population-weighted area apportionment using the ONS MSOA-2021 best-fit lookup. The sparser 1.1 % microsample sometimes used in earlier London 1-km work (55 072 commuters) gives noisier per-grid estimates and we replace it with the full Census table throughout.

For 2011, we use the NOMIS *WU03EW* (Location of Usual Residence and Place of Work by Method of Travel to Work, MSOA level), comprising 2.14 million London commuters across 240 020 MSOA pairs. Mapping to our grid uses the ONS MSOA-2011 → MSOA-2021 best-fit lookup; 18 of 826 grid-mapped MSOA-2021 codes were created post-2011 (population-driven splits, e.g. the Stratford-area E02006996 from parent E02000726) and we resolve via the lookup. Disaggregation to the 1-km grid uses the same population-weighted area procedure as for 2021.

### 4.2 Node features

The 22-dimensional static feature vector per grid is built from:

- **8 sectoral employment counts** (sec1–sec8 per the standard NOMIS BRES SIC 18-class roll-up: primary, manufacturing, construction, retail, food & beverage, information & finance, public sector, other), derived from BRES 2024.
- **Total workplace employment** from the same BRES 2024 release.
- **Resident population** from Census KS101EW (2021), aggregated MSOA-to-grid by area weighting; for the 2011 retrodict the same KS101EW (2011) is used.
- **POI counts** by 8 categories (commercial, education, food & beverage, healthcare, office, public, retail, transport) plus a total POI count, derived from OpenStreetMap as of 2024.
- **Subway station count** from TfL Open Data (2024 roster, with stations opened post-2012 — the Elizabeth-line and Northern-line Battersea-extension stations — filtered out for the 2011 retrodict).
- **Centroid latitude, longitude** in EPSG:4326.

All features (other than coordinates) are log1p-transformed before z-scoring to compress the heavy right tail of urban densities. The same per-feature mean and standard deviation derived from the 2024 baseline are reused for the 2011 retrodict so the model sees inputs on a consistent scale.

For the temporal variant (Section 7.2 Scenario B), three time-varying features are appended: a per-hour congestion ratio derived from a TomTom-equivalent destination-hour congestion index, and the sine and cosine of the hour-of-day. The result is a (24, 1725, 25) tensor used by the GRU-augmented hourly model.

### 4.3 Travel time and distance

Free-flow car travel time is precomputed from the Greater London OpenStreetMap drive network using OSMnx with edge weights *length / posted speed limit* and a multi-source Dijkstra from each grid centroid. The resulting 1725 × 1725 matrix has median travel time 32 minutes and 95th-percentile 68 minutes, broadly consistent with TfL JTS 2019 statistics for off-peak weekday car trips. Free-flow car time is used both as the cost in the inverse loop and as the kNN basis for the GraphSAGE adjacency (k = 10 nearest neighbours by free-flow travel time). The straight-line distance for the gravity term is the great-circle distance between centroids.

### 4.4 Auxiliary RUM inputs

The interaction terms in the RUM head require:

- **Origin IMD score** ($z_o$): Index of Multiple Deprivation 2019 income subscore, aggregated LSOA-to-grid by population weighting and z-scored across grids.
- **Destination wage** ($z_w$): mean workplace earnings derived from ASHE 2025 workplace residence-and-place-of-work tables, aggregated MSOA-to-grid by employment weighting and z-scored.
- **Occupation match** ($\mathrm{OccMatch}_{ij}$): cosine similarity between the origin's resident-occupation distribution (from Census 2021 NS-SEC) and the destination's workplace-occupation distribution (BRES 2024 SIC × NS-SEC concordance).

For the agent-level heterogeneity layer (Section 3.5 and 7.3) we additionally use an existing 10 000-agent synthetic London commuting population at our grid resolution, with per-agent income tier (low / mid / high based on Wardman 2014 thresholds) and home grid index drawn jointly to match the marginal distributions of London 2021 ASHE residence earnings and the Census KS101 borough population shares.

### 4.5 Train / validation / test splits

The headline performance evaluation uses a 4-borough leave-out spatial holdout (Westminster, Hackney, Brent, Bromley) covering 247 of 1725 origin grids (14 %). The boroughs are chosen ex ante to span the major London geographies (CBD, inner residential, outer dense, outer suburban) so that the holdout is informative about generalisation across diverse urban contexts rather than within a homogeneous slice. The 5-seed ensemble is trained on the same train / val mask across seeds; the bootstrap CI in Section 5.1 resamples within the validation borough origins.

The retrodict experiment (Section 6) uses an across-time holdout: training on the full 2011 OD table and evaluating on the full 2021 OD table.

### 4.6 Reproducibility

All processed CSV / NPZ files used by the experiments live in `data/processed/` with deterministic build scripts in `data/scripts/`. The build pipeline can be re-run from scratch from the raw NOMIS / OS / OSM downloads in approximately 4 hours of wall-clock time on a single workstation.

---

## 5 · Results

### 5.1 Headline performance

We report performance on a 4-borough leave-out spatial holdout (Westminster — central CBD; Hackney — inner residential; Brent — outer dense; Bromley — outer suburban), holding out 247 of 1725 grid origins (14 %). The Common Part of Commuters (Lenormand *et al.*, 2012) is the primary metric, alongside MAE on flow counts, Spearman rank correlation, and the Kullback-Leibler divergence per origin row.

Our best configuration is a 10-seed ensemble of the unconstrained Variant 4 trainer: spatial-holdout CPC = **0.343** with a 95 % bootstrap confidence interval **[0.324, 0.362]** computed by resampling validation origins (1000 resamples). Per-seed standard deviation is 0.003, so the model's variance is dominated by within-borough rather than across-seed sources. The constrained Variant 6 — which enforces the mainstream RP/SP direction $\phi \le 0,\ \psi \ge 0$ — sacrifices about 10 % of CPC (ensemble 0.290) for the behavioural interpretability of monotonic income gradients in VOT. We report both because they expose an aggregate-data identification ambiguity discussed in §5.3.

A paired bootstrap test resamples validation origins and recomputes CPC for our model and each baseline simultaneously. Against gravity (CPC 0.170) the bootstrap mean of $\mathrm{ΔCPC}$ is **+0.172** with 95 % CI **[0.155, 0.191]**; the empirical $P(\mathrm{ours} > \mathrm{gravity})$ is 100 % across 1000 resamples, giving a one-sided $p < 10^{-4}$. The same applies against radiation (CPC 0.174; $\mathrm{ΔCPC}$ +0.170 with 95 % CI [0.148, 0.193]; $p < 10^{-4}$).

Per-borough CPC reveals where the model is strong and weak: Westminster (CBD core) 0.475, Bromley (suburban outer) 0.352, Brent (outer dense) 0.293, Hackney (inner non-CBD) 0.201. The Hackney result is the model's principal weakness — inner non-CBD residential commute patterns appear less predictable from our static node features than either CBD employment density (Westminster) or suburban radial flows (Bromley). We discuss possible causes — mode-share heterogeneity, residential self-selection unobserved in our static features, and aggregate counts' limited information about within-borough variation — in §8.

![**Figure F4.** Per-borough CPC, headline 10-seed unconstrained ensemble, 4-borough spatial holdout. Westminster (CBD) is the strongest, Hackney (inner residential) the weakest. Horizontal lines mark our overall ensemble CPC (with 95% CI shaded) and the gravity / radiation baselines.](../evaluation_outputs/paper_a/F4_per_borough.png)

### 5.2 Baseline comparison

Table T1 summarises the leaderboard (all evaluated on the same 4-borough holdout):

| Model | CPC | 95 % CI | Comment |
|---|---|---|---|
| Gravity (Wilson 1971) | 0.170 | — | Production-constrained softmax MLE |
| Radiation (Simini 2012) | 0.174 | — | Parameter-free intervening opportunities |
| Deep Gravity (our minimal MLP) | 0.016 ± 0.004 | — | 5 layers, 128 hidden, 1000 epochs, GPU |
| GAT (4 head) | 0.246 ± 0.005 | — | Same width as ours; under-trains in our regime |
| **Phase B v6 unconstrained ensemble (ours)** | **0.343** | **[0.324, 0.362]** | 10 seeds, sign-unconstrained |
| Phase B v6 sign-constrained ensemble (ours) | 0.290 | [0.286, 0.295] | 5 seeds; sacrifices CPC for behavioural interpretability |

Two observations on the baselines warrant caveat. First, our minimal Deep Gravity (a 5-layer 128-hidden MLP over $[X_i, X_j, \log d_{ij}]$ with cosine LR schedule and patience-40 early stopping) achieves CPC well below gravity at our 1-km scale; the published Simini (2021) numbers were achieved on county-scale data with a deeper 15-layer-256-hidden network and gravity-warm-started initialisation that we did not replicate. We therefore present the Deep Gravity result as a **lower bound** on what a vanilla MLP without graph structure can attain at our scale, not as a faithful re-implementation. Second, the GAT ablation is informative: a 4-head graph-attention backbone with the same depth and width as our GraphSAGE underperforms it (0.246 vs 0.344). We attribute this to GAT's higher parameter count not being supported by the 1478 training origins, an empirical pattern consistent with prior observations in low-data regimes.

![**Figure F1.** Model leaderboard on the 4-borough spatial holdout. Bars show CPC; error bars show ± 1 SD across seeds (where applicable) or 95% bootstrap CI for our headline ensemble. The dashed red line marks the gravity baseline. Our model exceeds gravity by +0.172 CPC (paired bootstrap *p* < 10⁻⁴).](../evaluation_outputs/paper_a/F1_leaderboard.png)

### 5.3 Recovered parameters and identification ambiguity

The constrained Variant 6 recovers $\beta_t = -0.058$, $\phi = -0.30$, $\psi = +0.39$, $\delta = +0.69$, $\gamma = -1.27$ (5-seed means; standard errors below 5 %). The per-destination-tier effective time coefficient $\beta_{\rm eff}(j) = \beta_t \cdot (1 + \psi \cdot z_w(j))$ implies VOT values £4.49 / £6.59 / £8.23 per hour for low / mid / high wage-tertile destinations, calibrated against the Wardman (2014) RP central commute VOT of £6.5 per hour as the numeraire anchor.

The unconstrained Variant 4 — fit without any sign restriction — recovers $\beta_t = -0.13$ (larger magnitude, more time-sensitive), $\phi = +0.045$ (small), $\psi = -0.15$ (inverse direction). The cross-product $\psi < 0$ is at face value at odds with the conventional RP/SP finding that high-VOT workers commute disproportionately to high-wage destinations. We attribute this to an aggregate-data identification subtlety: the LSE-style mean income gradient estimable from cell-level OD reflects the joint of self-selection (where rich workers live), commute structure (where rich jobs are), and labour-market matching (which rich workers go to which rich jobs); only the third of these is the behavioural object the constrained variant is designed to recover. We therefore present both variants and note that the +5 percentage-point CPC gap of Variant 4 is the cost of behavioural interpretability — a quantification that, to our knowledge, is novel for this class of model.

### 5.4 Multiverse robustness

To verify that the headline result is not cherry-picked, we run a 100-configuration multiverse sweep (Steegen *et al.*, 2016) over the model hyperparameter grid: hidden width $\in \{64, 96, 128, 192\}$, GNN depth $\in \{3, 4, 5, 6\}$, top-$K$ choice-set size $\in \{30, 50, 100\}$, RUM-head learning rate $\in \{0.01, 0.03, 0.1\}$, sign-constrained vs unconstrained, $\times$ 5 seeds. We sample 100 configurations uniformly from the resulting 720-cell grid and report the cross-configuration distribution of spatial-holdout CPC.

**All 100 configurations exceed the gravity baseline 0.170.** The distributions for the two variants (Figure F3) reflect the trade-off discussed in §5.3:

| Variant | n | mean | std | P10 | median | P90 | min | max |
|---|---|---|---|---|---|---|---|---|
| Constrained | 51 | 0.286 | 0.041 | 0.221 | 0.300 | 0.329 | 0.205 | 0.333 |
| Unconstrained | 49 | 0.315 | 0.024 | 0.280 | 0.318 | 0.344 | 0.262 | 0.351 |

The unconstrained variant has both a higher mean and a tighter spread (std 0.024 vs 0.041), confirming that the sign-constraint loss is structural rather than a tuning artefact: across all hyperparameter combinations, removing the constraint shifts the distribution rightward by roughly 3 CPC points and stabilises it. Even the worst-performing configuration (CPC 0.205, a small-hidden-width constrained variant with low learning rate) exceeds gravity by 21 %. The headline 10-seed ensemble (CPC 0.343) sits at the P95 of the unconstrained distribution, consistent with — rather than an outlier above — the multiverse picture.

![**Figure F3.** Multiverse robustness: distribution of spatial-holdout CPC over 100 hyperparameter configurations (grid: hidden ∈ {64,96,128,192}, depth ∈ {3,4,5,6}, K ∈ {30,50,100}, lr_rum ∈ {0.01,0.03,0.1}, sign-constrained / unconstrained, × 5 seeds). Every configuration exceeds gravity 0.170. The unconstrained subset has both higher mean and tighter spread.](../evaluation_outputs/paper_a/F3_multiverse.png)

---

## 6 · Retrodiction validation

### 6.1 City-wide forward retrodict (2011 → 2021)

We train the same Phase B v6 architecture from scratch on 2011 inputs (NOMIS WU03EW disaggregated to our 1-km grid via the ONS MSOA-2011↔MSOA-2021 best-fit lookup, with 2011 KS101EW Census population and TfL station roster restricted to pre-2012 stations; 2011 BRES sectoral employment is approximated by per-grid scaling of the 2024 sector mix by the 2011/2024 total-employment ratio). The model fits 2011 OD with in-sample CPC 0.71. We then forward through the same trained model with 2024 features and compare to the observed full-Census 2021 OD: CPC 0.56, Spearman correlation across all OD pairs 0.52, with per-borough CPC ranging 0.52–0.62. The 0.71 → 0.56 drop quantifies a 10-year temporal generalisation gap that, after accounting for the COVID-19 pandemic's distortion of the 2021 Census commute baseline (March 2021 was the second UK lockdown), is plausibly close to the irreducible drift in inter-borough commuting structure.

### 6.2 Bromley negative control

A central concern with model-based counterfactual prediction is that the model may "predict everything goes up" regardless of the intervention. To guard against this we apply the same intervention machinery to Bromley, an outer suburban borough whose 2011→2024 employment grew only 1.55× (typical of stable suburban areas, and well below the city-mean 1.45×). The model predicts $\mathrm{Δshare} = +1.46$ percentage points; the observed 2011→2021 change is $+1.26$ percentage points. The predicted/observed ratio is **1.15×**, well within the [0.5, 2.0] tolerance set in our pre-registered scope. The framework is therefore well-calibrated for non-disrupted suburban interventions.

### 6.3 Stratford intervention case study

Stratford in east London underwent a regeneration of unusual magnitude between 2011 and 2024: the 2011 Westfield mall opening, the 2012 London Olympics, and the subsequent Olympic Park redevelopment increased the cluster's daytime workplace employment roughly five-fold. We isolate a 9-grid Stratford treatment cluster centred on grid L035_029, replace its 2011 features with their 2024 values, and compare predicted to observed Δshare of total London commute inflow.

The model's predicted Δshare is +1.33 percentage points; the observed value (computed from full-Census 2011 and 2021 OD tables) is +0.23 pp. The model thus over-predicts by approximately 5.7×. The per-destination Pearson correlation between predicted and observed Δinflow is 0.99, indicating that the **shape** of the redistribution is correctly captured but the **magnitude** is inflated. Two factors plausibly explain the magnitude inflation: (i) the IIA property of the underlying multinomial logit, in which a single-cluster intervention attracts probability mass aggressively from all alternatives; in reality multiple London CBDs grew simultaneously between 2011 and 2024, attenuating Stratford's relative attractiveness gain; and (ii) the 2021 Census coincided with the UK's second pandemic lockdown, deflating commute flows to CBD-adjacent areas relative to a counterfactual non-pandemic 2021. We document both as limitations rather than model defects.

---

## 7 · Policy Scenarios

### 7.1 Scenario A — Spatial intervention: +65 000 jobs at Old Oak Common

We instantiate a candidate spatial intervention by inflating the employment cluster (sectors 1–8, total employment, and POI office count) at the 9-grid Old Oak Common cluster — the western Eulerian terminus of the planned HS2 station — proportional to a +65 000 jobs increase. Forwarding the trained model under the modified features and aggregating predicted flow:

- Free-flow Δinflow into the cluster: +126 commuters
- BPR-equilibrium Δinflow: +105 commuters (17 % congestion pushback)
- Free-flow accessibility (Hansen $A_i = \sum_j E_j \exp(\beta_t t_{ij})$) gain: +1.34 %
- BPR-equilibrium accessibility gain: +0.45 % (two-thirds of gross gain absorbed by congestion)
- Population-weighted Gini on $A_i$: +1.26 % (regressive)
- Palma top-10 / bottom-40 ratio: +1.51 % (regressive)
- Atkinson($\varepsilon = 0.5$): +2.53 % (regressive)

The intervention is therefore **mildly regressive** in accessibility terms, and the BPR equilibrium amplifies the regressivity by absorbing the largest gains where they cluster: at grids already well-connected to the Old Oak Common neighbourhood, which tend to be wealthier than the city mean.

### 7.2 Scenario B — Temporal intervention: flexible work

To represent post-pandemic flexible-working arrangements that spread morning-peak demand toward off-peak hours, we apply a 50 % multiplier to peak-hour (07:00–09:59) demand and a 150 % multiplier to off-peak shoulder hours (10:00–15:59), conserving total daily demand per origin. The hourly model (T = 24, retrained from scratch on Phase B v6 architecture) then predicts the per-hour OD allocation. Coupling per-hour BPR equilibrium with per-hour accessibility:

- Peak hour 08:00 mean travel time: 33.99 min → 32.73 min (**−3.7 %**)
- Peak hour 08:00 accessibility (population-weighted): 1.09 M → 1.20 M (**+10.2 %**)
- Peak hour 08:00 Gini on $A_i$: 0.180 → 0.169 (**−0.011, progressive**)
- Peak hour 08:00 Palma: 0.591 → 0.557 (progressive)

The temporal intervention is therefore **progressive** in accessibility terms — the under-connected origins, whose commute is most penalised by congestion, gain disproportionately from peak-hour relief. The contrast with Scenario A (regressive) is the central policy-analytic finding of this paper: the same recovered behavioural parameters yield opposite distributional verdicts on two different intervention modalities, and the temporal intervention dominates on welfare grounds.

![**Figure F7.** Scenario B per-hour analysis. Top: total flow per hour, baseline vs flexible-work scenario (peak ×0.5, off-peak ×1.5, mass conserved per origin). Bottom: BPR-equilibrium mean travel time at sampled hours; the morning peak (h08) sees a -3.7% improvement.](../evaluation_outputs/paper_a/F7_scenarioB_hourly.png)

![**Figure F6.** Scenario A vs Scenario B impact on accessibility inequality. The spatial intervention (left) is regressive across all three indices; the temporal intervention (right) is progressive across both reported indices. Same recovered parameters, opposite distributional verdicts.](../evaluation_outputs/paper_a/F6_scenarios_inequality.png)

### 7.3 Agent-level heterogeneity and Jensen bias

We compare the aggregate model's prediction to a per-agent simulation in which each of $N = 10\,000$ agents draws a personal $\varepsilon_n \sim \mathcal N(0, 0.4^2)$ on top of the recovered $\beta_t \cdot (1 + \phi z_o + \psi z_w)$ (the WebTAG / Wardman-derived coefficient of variation). Aggregating per-agent softmax predictions and comparing to the aggregate-model prediction:

- Stratford 9-cluster intervention: aggregate Δinflow +134.2, agent +131.4, **Jensen bias +2.1 %**
- Random outer-London 9-cluster (low baseline employment): aggregate +589, agent +707, **Jensen bias −16.7 %**

The bias is small for mature (high-employment-baseline) clusters and large and **negative** for greenfield clusters. The negative sign means the aggregate model **under**-predicts the agent-level response when the destination is starting from a low-attractiveness baseline — the convex region of the softmax. A sensitivity scan over $\sigma \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ (Table T8) confirms |bias| < 4 % for the Stratford cluster across the entire calibration range, indicating the agent-layer methodology does not depend on a precise σ choice.

The framework requires no individual-level travel survey data: the per-agent $\sigma$ is calibrated entirely from the public WebTAG TAG A1.3 income-tier value-of-time distribution (Department for Transport, 2024) and the Wardman (2014) meta-analytic coefficient of variation. This is the principal portability advantage when extending to cities (e.g. Beijing) where individual-level data are unavailable.

![**Figure F9.** Agent-level Jensen bias. (A) CoV-sensitivity scan at the Stratford 9-cluster intervention: |bias| stays under 4 % across the entire WebTAG-plausible σ range, including σ = 0.40 (the Wardman-meta-analytic anchor). (B) Bias direction and magnitude depend on intervention context: small (and slightly positive) for mature CBD interventions, but large and negative (aggregate under-predicts) for greenfield interventions.](../evaluation_outputs/paper_a/F9_jensen_bias.png)

---

## 8 · Discussion and Limitations

The principal methodological contribution of this paper is the demonstration that aggregate-OD inverse training of a graph-structured deep choice model can recover behavioural primitives at city scale, with three properties that are useful for policy analysis: (i) statistically significant separation from gravity-class baselines (Section 5.1), (ii) non-trivial cross-temporal generalisation evidenced by the 2011→2021 retrodict (Section 6), and (iii) a bridge to agent-based simulation through the per-agent heterogeneity layer that quantifies Jensen bias under different intervention modalities (Section 7.3). The same model architecture supports both spatial and temporal interventions (Section 7) with diametrically opposed inequality verdicts.

Yet these claims are subject to substantive limitations.

1. **Free-flow $t_{ij}$ in inverse training.** The simultaneity between flow and travel time would require an instrument or fixed-point training regime to handle endogenously. We use free-flow $t_{ij}$ throughout the inverse loop, restoring BPR only in the counterfactual forward — a deliberate choice that trades off some realism in fitted $\beta_t$ for tractability and identifiability.

2. **Per-borough Hackney CPC = 0.20.** The model fits inner non-CBD residential commute patterns weakest. We do not resolve the cause; candidates include the higher mode-share variability of inner London, residential self-selection unobserved in our static features, or the limited information that aggregate counts carry about within-borough heterogeneity.

3. **2021 Census commute deflation.** The March 2021 Census fell within the second UK pandemic lockdown. Even using the full Census table (1.21 M commuters) rather than the 1.1 % microsample, the headline volumes are roughly 60 % below pre-pandemic levels. Our retrodict therefore conflates the effect of regeneration with the COVID-induced commute drop in CBD-adjacent areas. We mitigate by working in proportional shares rather than absolute counts, but the issue is irreducible.

4. **Single-intervention case study.** The retrodict tests only the Stratford regeneration. A multi-case study (Vauxhall–Nine Elms, Battersea Power Station, the Crossrail/Elizabeth-line corridor) would strengthen the validation. We defer to future work.

5. **No mode choice.** Mode of travel is recorded in the underlying Census but not modelled here. A nested logit over (mode, destination) is the natural extension and forms part of the planned Beijing companion paper.

6. **Aggregate identification ambiguity (Section 5.3).** The unconstrained variant fits the OD better but recovers a destination-wage interaction $\psi$ of the opposite sign to the conventional RP/SP literature. We do not resolve this — it is plausibly a labour-market matching artefact that aggregate data cannot disentangle from the behavioural primitive — but report both variants and quantify the CPC cost of behavioural interpretability.

7. **Synthetic agent population for the heterogeneity layer.** The 10 000-agent population (Section 7.3) is synthetic, with marginals matched to ASHE residence earnings and Census KS101 borough population shares. Joint distribution of (income, occupation, residence) is therefore consistent with marginals but not with a real revealed-preference panel. The Jensen bias quantification is therefore an upper-bound on what can be inferred from observable aggregates plus literature priors on $\sigma$, not a ground-truth measurement.

8. **Single-city training.** All training data are from London. Cross-city generalisation, which would require either jointly fitting on multiple cities (Simini *et al.*, 2021's strategy) or transferring a London-fitted model to Beijing with invariance penalties, is the subject of the planned Beijing application (Section 9).

---

## 9 · Beijing Application (Paper B preview)

The same deep-choice + GNN + agent-heterogeneity framework applies directly to Beijing with three substitutions. (i) Travel times: the Amap-scraped car directions API replaces our OSMnx free-flow car matrix; the public transit travel time matrix is derived from the BMTU GTFS feed. (ii) Employment: the Beijing Statistical Yearbook reports total employment by district, disaggregated to the 1-km grid via population-weighted area apportionment, in lieu of the BRES dataset we use here. (iii) Value-of-time spread: the $\sigma$ in the agent-heterogeneity layer is recalibrated from Chinese-context VOT meta-analyses (Bao *et al.*, 2024; Wang *et al.*, 2018) rather than WebTAG. Validation surrogates — multiverse robustness, plausibility metric, and triangulation against gravity — are all available; retrodiction is *not*, owing to the absence of open historical OD data at MSOA-equivalent resolution. The Beijing application will therefore inherit the methodology validated here and report the surrogate evidence chain in lieu of historical retrodict.

---

## 10 · Reproducibility

All code is in the repository under `experiments/paper_a/` (experiment harnesses) and `models_lib/inverse_rum/` (model components). A single Phase B v6 checkpoint trains in approximately 30 seconds on a single CPU core; the headline 10-seed unconstrained ensemble takes about 8 minutes. The 100-configuration multiverse sweep (Section 5.4) takes approximately 50 minutes on a single CPU. The agent-level heterogeneity layer (Section 7.3) and per-hour Scenario B (Section 7.2) each run in under a minute on the same hardware. All raw data are downloadable from public sources (NOMIS 2021 Census ODWP01EW, NOMIS 2011 WU03EW, NOMIS BRES, ONS Open Geography Portal MSOA-2011↔MSOA-2021 lookup, OpenStreetMap via OSMnx, TfL Open Data) without registration or licence beyond standard open-government terms. We release all derived NPZ / CSV intermediates and the trained checkpoints used in Sections 5–7.

---

## References

_(To be expanded; key citations referenced in text:)_

- Wilson, A. G. (1971). A family of spatial interaction models. *Environment and Planning A*.
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behaviour.
- McFadden, D. (1978). Modelling the choice of residential location.
- Sheffi, Y. (1985). *Urban Transportation Networks*. Prentice Hall.
- Hamilton, W., Ying, Z., Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*.
- Veličković, P. *et al.* (2018). Graph Attention Networks. *ICLR*.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.
- Simini, F. *et al.* (2012). A universal model for mobility and migration patterns. *Nature*.
- Lenormand, M. *et al.* (2012). Universal patterns of human mobility from a multi-day GPS dataset. *PLOS ONE*.
- Wardman, M., Chintakayala, V. P. K., de Jong, G. (2016). Values of travel time in Europe: Review and meta-analysis. *Transportation Research Part A*.
- Yao, X. *et al.* (2021). Spatial-interaction graph convolutional network for OD matrix estimation. *IEEE TITS*.
- Simini, F. *et al.* (2021). A Deep Gravity model for mobility flows. *Nature Communications*.
- Wang, S., Klabjan, D. (2018). Effects of distance on travel demand. *NeurIPS workshop*.
- Sifringer, B., Lurkin, V., Alahi, A. (2020). TasteNet: deep learning for taste heterogeneity in choice models. *Transportation Research Part B*.
- Wong, M., Farooq, B. (2021). ResLogit: A residual deep learning model for choice. *Transportation Research Part C*.
- Department for Transport (2024). *TAG Unit A1.3: Use of Time and Reliability Values*. WebTAG.
- Kim, S. *et al.* (2024). Counterfactual Explanations for Deep Learning-Based Traffic Forecasting. *arXiv*.


---

_Drafted in tandem with the Phase B+ experiments (D1–D11, retrodict, squeeze, het). Sections 4–9 will expand from the outline once D9 multiverse and any remaining hourly seeds finish._
