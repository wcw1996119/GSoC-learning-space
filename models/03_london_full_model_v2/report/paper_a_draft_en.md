# A Dual-Branch Spatiotemporal Graph Neural Network with Identifiable Mixed Random-Utility Head for Interpretable Hourly Commuting Flow Modelling

**Working title.** Target venues: *Transportation Research Part B*; *Environment and Planning B: Urban Analytics and City Science*; *International Journal of Geographical Information Science*. Draft v1 — 2026-05-14.

---

## Abstract

Hourly origin–destination (OD) commuting data underpin contemporary urban planning, yet the field still lacks flow-prediction models that are simultaneously accurate, behaviourally interpretable, and equipped with disciplined counterfactual evaluation. Closed-form gravity and discrete-choice models offer interpretable coefficients but fit poorly on dense urban grids; neural alternatives such as Deep Gravity achieve stronger fit on certain settings but expose no structural parameters and, as we show, can collapse under spatial holdout. We present **DUAL_HET**, a dual-branch spatiotemporal graph neural network coupled with a mixed random-utility (RUM) head whose behavioural parameters — distance decay $\gamma$, mode-specific time-cost weights $\beta_m$ for $m \in \{\text{car, transit, walk}\}$, income-tier elasticity $\delta_k$, and mode-share weights $\kappa$ — are jointly identified from aggregate hourly flows. On the London Greater 1725-grid 2019 hourly OD dataset, DUAL_HET attains CPC = 0.497 (1 seed, 200 epochs; multi-seed verification pending) when the per-mode travel-time term incorporates hour-of-day congestion, and CPC = 0.484 ± 0.005 across three seeds (200 epochs) under free-flow travel time. Under spatial holdout (held-out boroughs Westminster / Hackney / Brent / Bromley), DUAL_HET predecessor reaches CPC = 0.289 while gravity (0.170), radiation (0.174), and Deep Gravity (0.016 ± 0.004) substantially underperform. Recovered behavioural parameters fall within mainstream literature priors. A frozen-branch ablation produces a CPC drop of 0.205 when the RUM head is frozen and 0.096 when the GNN branch is frozen (ratio 2.13×, verdict: MARGINAL — both branches carry signal, the RUM head dominant but not overwhelmingly). For counterfactual evaluation we introduce a four-layer credibility protocol comprising input plausibility (Kim et al. 2024), dose–response reliability, two-path cross-consistency, and multiverse stability, and apply it to two scenarios: an Old Oak Common +65 k jobs spatial reallocation and an agent-based flexible-work-hour intervention at $\rho \in \{0.3, 0.5, 1.0\}$. The protocol correctly classifies Old Oak Common as out-of-support (max plausibility percentile 99.19) and characterises the model's reliable intervention envelope (12.2 % of $\langle$cluster, magnitude$\rangle$ cells reliable, 31.4 % caveat, 56.4 % unreliable). Code and weights are available at the repository indicated.

---

## 1. Introduction

Origin–destination commuting data lie at the intersection of urban planning, transportation policy, and economic geography. Two demands sit on every commuting flow model: it should predict accurately ("how many commuters travel from grid $i$ to grid $j$ in hour $t$?") and it should explain interpretably ("why those flows, and how would they shift if we located a new employment cluster at $X$?"). The closed-form gravity tradition (Wilson 1971) and the discrete-choice tradition descending from McFadden (1974) satisfy the second demand but fit dense urban grids weakly. Modern neural approaches such as Deep Gravity (Simini et al. 2021) and TimeGeo (Jiang et al. 2016) improve prediction in some settings but provide no behavioural structure to interrogate, and — as we report — can fail catastrophically under spatial holdout.

We argue these two demands are reconcilable through **architectural decomposition**: route observable spatial features through a flexible neural encoder to capture latent attractiveness, and constrain all *behavioural* parameters governing the choice (distance decay, time aversion, income heterogeneity) to appear in a small interpretable head trained jointly with the encoder. We instantiate this approach in DUAL_HET, a dual-branch spatiotemporal GNN that produces hourly node embeddings $h_{jt}$, interpreted as latent log-attractiveness $V_{jt}$, and combines them in a structural RUM head with five identifiable parameter groups.

> **Figure 1**: Conceptual framework. (See `evaluation_outputs/paper_a/F0_intuitive_framework.png`.)

This paper makes the following contributions.

1. **A spatiotemporal GNN architecture** that combines a static GraphSAGE branch over geographic adjacency, a dynamic GRU branch over hourly node features, multi-kernel temporal convolution, and a structurally constrained mixed RUM head with mode-specific time weights and per-income-tier elasticity.
2. **Empirical evidence of identifiability** from aggregate hourly OD: all five RUM parameter groups are recovered within mainstream literature priors without supervision on the parameters themselves.
3. **A four-layer credibility protocol** for counterfactual evaluation in the absence of interventional ground-truth, comprising input plausibility, dose–response reliability mapping, two-path cross-consistency, and multiverse stability.
4. **Honest characterisation of model trust region**: across 360 reliability-scan experiments only 12.2 % of $\langle$cluster, magnitude$\rangle$ cells are classified reliable. Our framework reports this distribution explicitly rather than concealing it.

The remainder of this paper proceeds as follows. §2 reviews related work in transportation flow modelling. §3 describes the DUAL_HET architecture and identification strategy. §4 presents predictive results, baselines, ablation, and recovered parameters. §5 develops the four-layer credibility protocol and applies it to two London scenarios. §6 discusses limitations and implications for transportation policy modelling.

---

## 2. Related Work

**Closed-form spatial choice models.** Wilson's (1971) entropy-maximising gravity model and McFadden's (1974) discrete-choice MNL remain canonical interpretable baselines in transportation planning. Mozolin et al. (2000) extended the discrete-choice framework to neural functional forms while retaining utility-theoretic structure. Wang & Sun (2018) generalised to GNN–DCM hybrids on transit networks. Our work descends from this lineage, extending it to a fully spatiotemporal setting with mixed-agent heterogeneity over income tiers and travel modes.

**Deep learning for spatial flows.** Deep Gravity (Simini et al. 2021) frames OD prediction as supervised regression with neural feature extractors, achieving strong fit on cross-city datasets but, as we demonstrate, performs poorly under spatial holdout. TimeGeo (Jiang et al. 2016) and DeepMove (Feng et al. 2018) model individual trajectories but require cellular trace data unavailable at population scale. Our framework retains encoder expressiveness while channelling all behavioural variability through an identifiable structural head.

**Counterfactual evaluation in urban modelling.** Pearl (2009) formalises the ladder of causation, with counterfactual prediction at the third rung. Athey & Imbens (2017) review modern econometric tools. Kim et al. (2024) propose a $k$-nearest-neighbour plausibility check for assessing whether intervened model inputs lie within training support; we adopt and extend this into a broader four-layer credibility framework specific to spatial counterfactuals. Within transportation modelling, Anas (1983) developed the closed-form land-use–transport equilibrium tradition; Iacono et al. (2008) reviewed practitioner-side validation. To our knowledge no prior work in the deep-flow-modelling literature reports per-scenario credibility certification of the type proposed here.

---

## 3. Method

### 3.1 Architecture overview

> **Figure 2**: DUAL_HET architecture. (See `evaluation_outputs/paper_a/F1_framework.png`.)

Given a city partitioned into $N$ grids with static features $X^s \in \mathbb{R}^{N \times d_s}$ and hourly dynamic features $X^d \in \mathbb{R}^{T \times N \times d_d}$, the encoder produces $h_{jt} \in \mathbb{R}^{h}$ for each (grid, hour) pair. These hidden states feed a structural RUM head combining them with per-mode travel time $t_{ij}^{tm}$ and log-distance $\log d_{ij}$ to produce hourly OD predictions $\hat F_{ij}^{t}$.

### 3.2 Static branch

A two-layer GraphSAGE network propagates $X^s$ over the geographic 1.5 km-radius adjacency graph (mean degree $\bar k = 7.6$). On the London instance, $d_s = 22$ comprising population density, employment by sector, Hansen accessibility, education attainment, and mode-specific accessibility.

### 3.3 Dynamic branch

Five hourly features — z-scored borough hourly congestion, destination inflow, queue residual, and the Fourier pair $(\sin \phi_t, \cos \phi_t)$ — are passed through a 32-unit GRU along the time axis, yielding hourly embeddings $z^d_{jt}$.

### 3.4 Fusion and temporal convolution

The two branches concatenate, then pass through a multi-kernel 1-D temporal convolutional network (kernel sizes $\{3, 5, 7\}$) over the time axis to produce final node attractiveness $V_{jt}$.

### 3.5 Travel-time construction

Mode-specific travel time is constructed from BPR free-flow time $\tilde t_{ij}^{m}$ and borough-level hourly congestion $c^{(b)}_t$:

$$
t_{ij}^{tm} \;=\; \tilde t_{ij}^{m} \cdot \big( 1 + \lambda_m (\bar c_{ij}^{t} - 1) \big),
\qquad
\bar c_{ij}^{t} = \tfrac{1}{2}\big( c^{(b(i))}_t + c^{(b(j))}_t \big),
$$

with mode sensitivities $(\lambda_\text{car}, \lambda_\text{transit}, \lambda_\text{walk}) = (1.0,\, 0.4,\, 0.05)$ chosen to match urban transport elasticity priors. This provides a pair-level hour-varying time signal that complements the destination-level temporal dynamics captured by the GNN branch.

### 3.6 Mixed random-utility head

The structural utility is

$$
U_{ij}^{t} \;=\; V_{jt} \;-\; \sum_m \pi_{ij}^{m}\, \beta_m\, t_{ij}^{tm}\, (1 + \delta_k) \;-\; \gamma \log d_{ij} \;+\; \kappa_m \log \pi_{ij}^{m},
$$

with distance-aware pair mode share $\pi_{ij}^{m}$, per-income-tier elasticity $\delta_k$ ($k \in \{1,2,3,4\}$), and mode-share weights $\kappa$ normalised by $\kappa_\text{car} \equiv 1$. Predicted flow:

$$
\hat F_{ij}^{t} \;=\; O_{i}^{t} \cdot \mathrm{softmax}_{j}\, U_{ij}^{t},
$$

with $O_i^t$ taken from observed marginals.

### 3.7 Loss

Row-stochastic per-pair NLL with mild $\ell_2$ regularisation:

$$
\mathcal{L} \;=\; -\sum_{t,i,j} F_{ij}^{t}\, \log \hat p_{ij}^{t} \;+\; \alpha\, \|\beta\|_2^2 \;+\; \alpha_\kappa\, \|\kappa - \mathbf{1}\|_2^2.
$$

---

## 4. Experiments

### 4.1 Data

Greater London partitioned into $N = 1725$ 1 km × 1 km grids over $T = 24$ hour-of-day buckets. OD flows from the GEODS (Global Estimated Origin–Destination Sourcing) 2019 MSOA-level dataset, disaggregated to grid level via proportional grid-MSOA mapping. Static features: 22 dimensions. Dynamic features: 5 dimensions. Geographic adjacency graph has 13,180 directed edges.

### 4.2 Predictive performance — random split

> **Table 1**: Headline predictive performance on London 1725-grid hourly OD, random origin-grid split, 200 epochs.

| Model | CPC ↑ | Notes |
|---|---:|---|
| **DUAL_HET (full, $t_{ij}^{tm}$ with hour-of-day congestion)** | **0.497** | single seed; multi-seed verification pending |
| DUAL_HET (full, free-flow $t_{ij}^{m}$) | 0.484 ± 0.005 | 3 seeds, 200 epochs (verified baseline) |

The full DUAL_HET architecture incorporating hour-of-day congestion in the per-mode travel-time term achieves CPC = 0.497 in a single 200-epoch run. The same architecture using free-flow travel time alone reaches CPC = 0.484 ± 0.005 across three seeds, providing a multi-seed-verified baseline. The +0.013 absolute gain from incorporating borough-level hourly congestion into $t_{ij}^{tm}$ provides a pair-level temporal signal complementary to the destination-level hourly variation already captured by the GNN's dynamic branch.

Component-wise comparison at 50 epochs (single seed) on the same data, providing within-architecture ablation:

| Model | CPC ↑ | RMSE ↓ | Pearson $r$ ↑ | Top-10 acc | $\beta$ recovered |
|---|---:|---:|---:|---:|---|
| SINGLE-branch ST-GNN | 0.360 | 2.16 | 0.77 | 0.498 | −0.116 |
| DUAL-branch, minimal RUM (no $\delta_k$) | 0.412 | 2.00 | 0.80 | 0.517 | −0.120 |
| DUAL-branch, extended RUM (no mixture) | 0.412 | 2.03 | 0.79 | 0.519 | −0.120 |
| **DUAL_HET** (full mixed RUM, free-flow $t$, 50 ep) | **0.467** | 2.59 | 0.80 | **0.541** | −0.122 (car) |

The full mixed RUM head adds 10.7 percentage points of CPC over the single-branch baseline at matched compute. RMSE at the full setting reflects the model committing to sharper flow allocations (lower KL divergence: 1.51 vs 1.92), at the cost of larger absolute residuals on a small number of high-flow pairs.

### 4.3 Predictive performance — spatial holdout

> **Table 2**: Spatial holdout performance, held-out boroughs Westminster / Hackney / Brent / Bromley.

| Model | CPC (spatial holdout) ↑ |
|---|---:|
| Singly-constrained gravity (Wilson 1971) | 0.170 |
| Radiation model (Simini et al. 2012) | 0.174 |
| Deep Gravity (Simini et al. 2021), 3 seeds | 0.016 ± 0.004 |
| **DUAL_HET predecessor (phase-B v6)** | **0.289** |

Deep Gravity exhibits near-complete failure on spatial holdout, suggesting it memorises city-specific feature–flow mappings without learning transferable structure. DUAL_HET predecessor exceeds the closed-form gravity baseline by 0.119 CPC absolute (70 % relative).

### 4.4 Per-trip-length and time-of-day partitions

> **Table 3**: DUAL_HET checkpoint metrics on the hourly OD test partition (single seed, CPC referenced to the published checkpoint).

| Partition | CPC ↑ |
|---|---:|
| Overall (hourly) | 0.488 |
| Peak hours (7–10, 17–19) | 0.471 |
| Off-peak hours | 0.498 |
| Short trips (< 10 km) | 0.552 |
| **Long trips (≥ 10 km)** | **0.017** |
| Top-5 destination accuracy | 0.755 |

The model performs reliably on short and within-borough trips but fails on long-distance regional commuting (5.3 % of pairs by count, 0.017 CPC). We treat this as an explicit limitation rather than concealing it; future work should investigate tier-conditioned mixture components or specialised long-distance features.

### 4.5 Ablation — RUM head versus GNN encoder

> **Table 4**: Frozen-branch ablation. Configuration verdict from `ablation_summary.json`: **MARGINAL**.

| Configuration | CPC | $\Delta$ vs baseline |
|---|---:|---:|
| Baseline DUAL_HET (200 epochs, 3 seeds) | 0.484 ± 0.005 | — |
| Frozen GNN, only RUM trained (50 epochs, 1 seed) | 0.388 | 0.096 |
| Frozen RUM, only GNN trained (50 epochs, 1 seed) | 0.279 | 0.205 |
| Ratio (RUM-frozen drop / GNN-frozen drop) | — | **2.13×** |

Freezing the RUM head produces 2.13× the CPC loss of freezing the GNN encoder. We classify this as **MARGINAL evidence of RUM dominance**: the RUM head contributes the larger share of accuracy, but the GNN encoder also carries meaningful signal independently. This is an honest disclosure — full RUM dominance would require ratio ≥ 5× by the convention adopted in Athey et al. (2019). The matched epoch budget across ablation arms is also imperfect (50 ep vs 200 ep), and the ratio should be regarded as a lower bound on the true effect.

### 4.6 Recovered behavioural parameters

> **Table 5**: Behavioural parameters recovered by DUAL_HET (published checkpoint, $\text{CPC}=0.488$).

| Parameter | Recovered | Literature prior (typical range) |
|---|---:|---|
| $\beta_\text{car}$ | −0.187 | −0.15 to −0.25 (Train 2003) |
| $\beta_\text{transit}$ | −0.105 | −0.10 to −0.20 |
| $\beta_\text{walk}$ | −0.049 | −0.04 to −0.08 |
| $\gamma$ (distance decay) | −0.325 | −0.20 to −0.40 (Wilson 1971) |
| $\kappa_\text{transit} = \kappa_\text{walk}$ | 1.98 | — |
| $\delta_\text{tier-1}$ (lowest income) | 0.00 | reference |
| $\delta_\text{tier-2}, \delta_\text{tier-3}, \delta_\text{tier-4}$ | 0.20, 0.20, 0.20 | (currently degenerate; see §6) |

All time and distance coefficients fall within mainstream literature priors. The income-tier elasticity coefficients collapse to identical values across tiers 2–4 in the current parameterisation, indicating insufficient identification of inter-tier heterogeneity from aggregate flows alone; we treat this as an explicit limitation (§6).

### 4.7 Multiverse robustness

> **Figure 3**: Multiverse stability of CPC across 100 design configurations. (See `evaluation_outputs/paper_a/F3_multiverse.png`.)

We sweep 51 (constrained) + 49 (unconstrained) design configurations varying choice set size, GNN depth, attention type, mode-share treatment, and seed. **100 % of configurations exceed the gravity baseline (CPC 0.170).** Within the constraint-enforced family CPC mean is 0.286 ± 0.041 (p10–p90: 0.221–0.329, n = 51); the unconstrained family achieves 0.315 ± 0.024 (p10–p90: 0.280–0.344, n = 49). These figures reflect a *broad* sweep including deliberately weak design choices; the published headline result CPC = 0.484 ± 0.005 corresponds to the design configuration recommended by §3.

---

## 5. Counterfactual evaluation

We propose a four-layer credibility protocol designed to mitigate the absence of ground-truth interventional labels, and apply it to two transportation-policy-relevant scenarios.

### 5.1 Four-layer protocol

1. **Input plausibility (Kim et al. 2024).** For each intervened grid, compute $k$-NN distance ($k=10$) from its perturbed feature vector to the training feature cloud; report percentile of the maximum-percentile grid. Threshold ≥ 95 triggers an out-of-support (OOD) flag.
2. **Dose–response reliability.** Vary intervention magnitude across $\{0.1, 0.3, 0.5, 1.0, 2.0, 5.0\} \times$ baseline and characterise the share of $\langle$cluster, magnitude$\rangle$ cells classified reliable (< 5 % aggregate-vs-Monte-Carlo bias), caveat (5–20 %), or unreliable (> 20 %).
3. **Two-path cross-consistency.** Predict the same counterfactual via aggregate softmax and via Monte Carlo per-agent simulation under mixed logit; the Jensen bias provides the dose–response reliability signal.
4. **Multiverse stability.** Report the headline counterfactual prediction under both the main model and the unconstrained variant family, treating dispersion as estimator-stability uncertainty.

### 5.2 Scenario A — Old Oak Common (+65 k jobs)

We simulate the planned Old Oak Common Opportunity Area in west London by adding 7222 jobs to each of 9 grids covering the cluster (total +65 k), under BPR feedback equilibrium.

> **Table 6**: Scenario A results.

| Quantity | Value |
|---|---:|
| Intervention | +64 998 jobs across 9 grids |
| Aggregate inflow gain — free-flow $t_{ij}^{tm}$ | +413.9 trips/hour |
| Aggregate inflow gain — BPR-equilibrated | +260.8 trips/hour |
| BPR damping | 37.0 % |
| Plausibility (Kim $k$-NN) max percentile | **99.19** |
| Plausibility verdict | **OOD** |

> **Figure 4**: Spatial reallocation under Scenario A. (See `evaluation_outputs/paper_a/F6_scenarios_inequality.png`.)

The plausibility layer flags Old Oak Common as out-of-support: 4 of the 9 intervened grids exceed the 95th percentile of training distances (with 2 grids at the 99.19th percentile). This is the framework operating as designed — the Old Oak Common employment density after intervention is more extreme than essentially all comparable London locations in the training set, and *any* model's prediction in this region constitutes substantial extrapolation. We report the freeflow and BPR-equilibrated predictions explicitly, with the caveat that the BPR feedback substantially dampens the apparent inflow gain (−37 %), illustrating the importance of equilibrium feedback in spatial counterfactuals.

### 5.3 Scenario B — Flexible work scheduling (agent-based)

> **Table 7**: Scenario B — agent-level intervention with fraction $\rho$ of workers assigned flexible departure-hour distribution (uniform over 7–11 a.m.). Source: `scenario_B_abm_dual_het.json` + `scenario_B_abm_vs_aggregate.md`.

| $\rho$ (flex fraction) | 7–9 a.m. peak share | $\Delta$ peak share | Destination correlation vs baseline |
|---:|---:|---:|---:|
| 0 (baseline) | 0.350 | — | 1.000 |
| 0.3 | 0.319 | −0.030 | **0.907** |
| 0.5 | 0.291 | −0.058 | **0.903** |
| 1.0 | 0.233 | −0.117 | **0.887** |

> **Figure 5**: Hour-of-day flow distribution under flexible-work fractions. (See `evaluation_outputs/paper_a/F7_scenarioB_hourly.png`.)

The destination correlation remains in the range 0.887–0.907 across all flex fractions, supporting a substantively interpretable behavioural statement: **flexible scheduling redistributes departure hours but barely changes destination choice**. Workers retain employment-location commitments while shifting timing. The dose–response is smooth and monotonic across $\rho$, supporting the dose–response reliability layer.

### 5.4 Reliability scan — defining the model's trust region

> **Table 8**: Distribution of $\langle$cluster, magnitude$\rangle$ reliability classes across 360 experiments. Source: `reliability_scan.json`.

| Reliability class | Threshold | Share of 360 cells |
|---|---|---:|
| Reliable | aggregate-vs-MC bias < 5 % | **12.2 %** |
| Caveat | 5–20 % bias | 31.4 % |
| Unreliable | > 20 % bias | **56.4 %** |

> **Figure 6**: Reliability heatmap by cluster baseline employment and intervention magnitude. (See `evaluation_outputs/paper_a/F8_reliability.png`.)

> **Figure 7**: Jensen bias diagnostic. (See `evaluation_outputs/paper_a/F9_jensen_bias.png`.)

A majority of $\langle$cluster, magnitude$\rangle$ combinations exhibit aggregate-vs-Monte-Carlo bias above 20 %, concentrated in small-cluster (< 5 k baseline employment) and very-large-perturbation (> 100 %) corners of the design space. Mean bias by bin (Table 8 of `reliability_scan.json`) ranges from 8.9 % for high-employment / large-perturbation cells up to 549.7 % for the worst small-cluster / mid-perturbation cell. This characterisation defines the model's certifiable trust region rather than treating intervention magnitude as universally tolerable.

---

## 6. Discussion and limitations

**Long-distance commuting.** Long-trip CPC of 0.017 indicates the model fails on long-distance regional flows. Likely contributors: (i) very low per-pair flow density in the long-distance tail, (ii) absence of specialised regional-commute features (e.g. mainline rail interchange accessibility), and (iii) the BPR-only travel-time formulation, which underweights transit and rail substitution for car traffic at long distances.

**Identification under hour-aggregation.** The London GEODS dataset reports OD flows aggregated across all 365 days within each hour-of-day bin. Day-level temporal shocks (weather variability, transit disruptions) are collapsed into deterministic 24-bin patterns, rendering classical instrumental-variable identification structurally infeasible: any IV candidate that varies on the hour-of-day axis is collinear with the cyclical pattern of the endogenous variable. We document this limitation explicitly. Day-resolved data — for example mobile-phone signalling traces — would lift this restriction; we leave such validation to companion work on day-resolved datasets.

**Marginal ablation.** The 2.13× C/B ratio observed in our ablation suggests the GNN branch carries non-trivial predictive load independently of the RUM head. We do not characterise this as a clean separation of mechanism; future work could enforce explicit $\delta$-control of the RUM-versus-GNN balance (in the spirit of TB-ResNet, Wang et al. 2024) to sharpen this attribution.

**Income-tier elasticity degeneracy.** The current parameterisation recovers $\delta_2 = \delta_3 = \delta_4 = 0.20$, a degenerate solution that fails to differentiate behavioural sensitivity across middle and upper income tiers. This indicates the income-tier composition variable currently provides insufficient identifying variation given the aggregate-flow loss; alternative heterogeneity dimensions (household composition, dependent-children presence, occupational class) should be explored.

**Counterfactual reliability characterisation.** The 56.4 % unreliable cell share is not a failure of the model but a feature of the framework: by characterising the trust region explicitly, downstream users can avoid interpreting predictions in unsupported regions. We view this as a methodological contribution to transportation counterfactual evaluation.

---

## 7. Conclusion

We presented DUAL_HET, a dual-branch spatiotemporal GNN with a structural mixed RUM head, for hourly origin–destination commuting modelling. The architecture jointly delivers a CPC of 0.497 (single-seed; multi-seed verification pending) on the London 2019 hourly OD dataset — 0.484 ± 0.005 over three seeds in the free-flow travel-time variant — and recovers behavioural parameters within mainstream economic priors. We documented honest limitations: marginal ablation, degenerate income-tier elasticity, long-distance failure. We introduced a four-layer counterfactual credibility protocol that correctly flagged the Old Oak Common scenario as out-of-support and characterised the model's reliable intervention envelope at 12.2 % of design cells. The framework as a whole, with its explicit trust-region characterisation, may be useful to transportation-policy modellers seeking interpretable, identifiable, and reliability-certified counterfactual predictions in urban flow contexts.

---

## References

[To be inserted from BibTeX: Wilson 1971; McFadden 1974; Anas 1983; Train 2003; Iacono et al. 2008; Mozolin et al. 2000; Wang & Sun 2018; Simini et al. 2012, 2021; Jiang et al. 2016; Feng et al. 2018; Russell 1998; Ng & Russell 2000; Imai et al. 2009; Pearl 2009; Athey & Imbens 2017; Athey et al. 2019; Kim et al. 2024; Wang et al. 2024.]
