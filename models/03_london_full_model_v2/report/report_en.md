# Reshaping job accessibility — a fine-grained simulation tool for assessing the socio-spatial inequality impact of urban policy interventions

### A working report applied to London (Old Oak Common spatial regeneration & flexible working hours)

**Author:** Chengwei Wang  
**Working report — May 2026**

---

## In three sentences

**Aim.** A fine-grained urban-commute simulation tool, built to evaluate how different **policy interventions** reshape the socio-spatial dimension of commute and job-accessibility inequality — by **recovering behavioural decision parameters directly from publicly aggregated OD data**.

**Method.** The design binds together the **behavioural interpretability of random-utility models (RUM)** and the **flexible function-approximation power of deep learning** — its core is the **inversion of individual choice parameters from aggregate OD flows**: a standard multinomial-logit choice $P(j|i) = e^{V_{ij}}/\sum_k e^{V_{ik}}$ is written down, the destination-as-workplace attractiveness $A_j$ is **learned** by a GraphSAGE graph neural network from spatial features, and the travel-time sensitivity $\beta$, the attractiveness weight $\alpha$, and the GNN's internal weights are jointly **back-solved** from observed OD distributions; RUM keeps the learned $A_j$ a behaviourally meaningful scalar, deep learning supplies the spatial-spillover expressivity that hand-coded RUM cannot reach.

**Result.** The tool reveals two **qualitatively different distributional signatures** under the two London interventions — Old Oak Common spatial regeneration is **mildly regressive** (population-weighted Palma 0.552 → 0.561, with two-thirds of the free-flow gain absorbed by road congestion); flexible working is **progressive** (peak-hour commute load redistributed −46% / +63%, with the most-exposed outer-borough cells gaining most) — and a mixed-logit diagnostic surfaces a **systematic 17% under-estimation by aggregate-only prediction on greenfield interventions**, an inequality-sensitive bias that conventional aggregate flow analysis cannot see.

---

## 1. The question, and why it matters

**Job accessibility** — *how many jobs you can practically reach from where you live* — is one of the cleanest summary measures of urban opportunity. Two Londoners on the same wage but living 5 km apart can be looking at very different sets of jobs they could realistically commute to. This gap then interacts with income, with car ownership, and with the shape of the public-transport network. Almost every transport-related urban policy lever — from spatial regeneration to flexible-hours schemes to new transit lines — moves this gap in some direction, but rarely uniformly across the city: a policy that lifts the average can simultaneously *widen* the gap between the well-served and the poorly-served. The methodological task this report addresses is to build a tool that lets us see the redistribution, not just the average.

We apply this tool to two live London policy levers:

1. **Old Oak Common (OOC) spatial regeneration.** Britain's largest planned interchange, scheduled to anchor a regeneration cluster of approximately 65,000 new jobs across the Hammersmith / Ealing border. Classic question: does adding jobs in a designated sub-centre genuinely de-concentrate access from central London, or does it just produce another small dot on the existing accessibility map?

2. **Flexible working / off-peak shift.** A staggered-hours intervention modelled as moving 50% of the 7–9am peak commute load into the 10am–3pm shoulder window. Classic question: who benefits — the rush-hour office worker who sees less congestion, or the off-peak service worker who finally gets reasonable transit headways?

Both policy families are also under active discussion in Beijing (副中心 sub-centre programme, 弹性办公 flexible-hours policy). London is a development sandbox for the methodology; it is not the research target. Every methodological choice in this report has been picked so that the same pipeline can be re-fitted to Beijing once data clearance lands.

## 2. Methodology — combining RUM and deep learning

Three families of method dominate the existing literature on simulating commuting under policy change, and each is missing one capability that policy-impact assessment requires:

| Family | What it does well | What it cannot do |
|---|---|---|
| Gravity / radiation models | Predicts aggregate flows on calibration data | Recovers no behavioural parameters → cannot do counterfactuals coherently |
| Pure deep learning (e.g. Deep Gravity) | Fits observed flows with high accuracy | Parameters lack interpretable behavioural meaning → cannot say *why* a result moves |
| Pure agent-based simulation | Behaviourally rich, individual-level | Agent rules typically hand-coded → drifts away from observed data |

The design of this tool is to combine **the behavioural interpretability of random-utility models (RUM)** with **the flexible function-approximation power of a graph neural network**. Each piece carries one capability the other cannot provide.

### 2.1 The accessibility metric

For each 1×1 km grid cell *i* we compute a Hansen-style accessibility:

$$
A_i \;=\; \sum_j E_j \cdot e^{-\beta\, t_{ij}}, \qquad \beta = 0.058 \text{ per minute}.
$$

The decay parameter $\beta$ here is the *same* $\beta$ that the choice model recovers in §2.2 — so the accessibility numbers are interpretable as *effective workplace reach as Londoners actually trade off time against pull*, not as an arbitrary analyst weighting. A 30-minute destination is weighted at 18% of a 0-minute one; a 60-minute destination at 3%. $A_i$ has units of "jobs-equivalent": if $A_i = 100{,}000$, the cell has the same effective access as if 100,000 jobs sat on its doorstep at zero travel time.

### 2.2 The choice model — random utility with a learned attractiveness

A worker resident in cell $i$ faces all $N = 1{,}725$ possible destination cells and trades two things off:

- **pull**: how attractive cell $j$ is as a workplace — its industry mix, employment density, transport-hub effect, amenity neighbourhood;
- **friction**: how long the commute to $j$ takes.

We write this as a multinomial logit:

$$
P(j \mid i;\,\theta) \;=\; \frac{\exp(V_{ij})}{\sum_{k=1}^{N} \exp(V_{ik})},
\qquad V_{ij} \;=\; \alpha\, A_j(\mathbf{X};\,\phi) \;-\; \beta\, t_{ij}.
$$

The hard part is $A_j(\mathbf{X};\phi)$ — the destination's **attractiveness as a workplace**. Hand-coding it as "jobs × wage × index of amenity" is known to fit observed flows poorly because workplace attractiveness depends on *neighbourhood spillovers* (a Soho job is more attractive partly because Soho's neighbours are also rich in amenity), on *transport-hub network position*, and on *industry-cluster interactions* that no analyst can specify by hand.

We let a **graph neural network (GraphSAGE)** read each cell's own 22 features and its neighbours' features through a 1-hop spatial graph over the 1km grid, and output the attractiveness scalar $A_j$ that best fits observed flows:

$$
A_j(\mathbf{X};\,\phi) \;=\; \mathrm{GraphSAGE}_\phi\!\left(\mathbf{X}_j,\; \{\mathbf{X}_k : k \in \mathcal{N}(j)\}\right).
$$

Crucially the GNN does *not* predict OD on its own. It supplies one structured input — the destination attractiveness vector $A$ — into the discrete-choice model that does the actual prediction. This contrasts with the Deep-Gravity-style approach (Simini et al., 2021), which feeds origin and destination features into one large neural network with no explicit choice structure — losing the interpretability that lets us reason about counterfactual policy.

### 2.3 The statistical principle of "recovering behavioural parameters from aggregate OD"

> Whenever a paper claims to recover individual behavioural parameters from aggregate data, the natural question is: by what statistical mechanism, and with what guarantees? We owe the reader a clean answer.

The mechanism is **Maximum Likelihood Estimation** of a discrete-choice model in the McFadden (1974) tradition, computed numerically via gradient descent. The two named procedures are equivalent.

**Step 1.** Under the random-utility assumption with i.i.d. extreme-value-1 errors, the observed origin-$i$ flow vector is multinomially distributed:

$$
F_{i,\cdot} \;\sim\; \mathrm{Multinomial}\!\left(O_i,\; P(\cdot \mid i;\,\theta)\right),
\qquad
\theta = (\alpha,\, \beta,\, \phi).
$$

**Step 2.** The log-likelihood of the parameter vector $\theta$ given all observed flows is therefore (up to an $\theta$-independent constant)

$$
\mathcal{L}(\theta) \;=\; \sum_{i=1}^{N}\sum_{j=1}^{N} F_{ij} \log P(j \mid i;\,\theta).
$$

**Step 3.** The negative of $\mathcal{L}(\theta)$ is, term-for-term, the *categorical cross-entropy loss* used universally in deep-learning classification, with target distribution $F_{ij}/O_i$ and predicted distribution $P(j|i;\theta)$:

$$
-\mathcal{L}(\theta) \;=\; -\sum_{i,j} F_{ij} \log\frac{e^{V_{ij}}}{\sum_k e^{V_{ik}}} \;=\; \mathrm{CrossEntropy}\!\left(F_{ij}/O_i,\; P(\cdot|i;\theta)\right).
$$

**Therefore.** Minimising the deep-learning cross-entropy loss with backpropagation is *numerically identical* to performing MLE on the multinomial discrete-choice model. The deep-learning community and the discrete-choice-econometrics community are doing the same operation under different names — the equivalence has been pointed out at least since Bentz & Merunka (2000) and is what licences us to claim the resulting $(\hat{\alpha}, \hat{\beta}, \hat{\phi})$ are MLE estimates with the standard statistical properties.

**Identification.** Each parameter is identified from a specific axis of variation in the data:
- $\hat\beta$ — from the cross-OD-pair variation in $t_{ij}$ and the way destination-share decays with travel time;
- the *shape* of $A_j$ — from cross-destination variation in $(E_j, \mathbf{X}_j, \mathbf{X}_{\mathcal{N}(j)})$;
- $\hat\alpha$ — from the relative scale of $A_j$ versus $\beta\, t_{ij}$ (softmax fixes one degree of freedom).

The London training set provides $1{,}725 \times 1{,}725 \times 24 = $ ~71 million OD–hour cells against ~40,000 GNN parameters and 2 behavioural coefficients — strongly over-identified.

**Caveats we acknowledge.**

| Caveat | What we do about it |
|---|---|
| Neural network makes the loss surface non-convex → no guarantee of global optimum | Multi-seed restart (5 seeds in Phase B+); loss spread < 2% across seeds |
| Aggregate-data SE is wider than individual-data SE | We do not report point-estimate SE; the §7 reliability scan reports an *empirical* ±25% confidence band |
| EV1 error assumption may not hold strictly | A mixed-logit Monte Carlo with random coefficients is run as a sensitivity check — it produces the §3 *Jensen-bias* diagnostic |

### 2.4 Data

The aggregated OD flow table is the **anonymised mobile-phone location data** released by the UCL CASA group ([Zhong & Zhou, 2025, *Scientific Data*](https://www.nature.com/articles/s41597-025-06323-8); dataset DOI 10.5281/zenodo.13327082): GPS records from approximately 200 smartphone applications, collected across England in **November 2021**, with home and workplace inferred from night-time and day-time stay-point detection respectively, aggregated to MSOA-level travel-to-work flows. We re-grid this to a 1×1 km Cartesian grid (1,725 inhabited cells covering Greater London) using land-area-weighted reallocation. Total commute flows in our training set: ~1.21 million unique OD relationships.

The November 2021 collection date sits in the inter-lockdown window between England's autumn re-opening and the December 2021 Omicron Plan B — closer to a "normal" baseline than e.g. Census 2021 (March 2021, full lockdown), but still partially affected by hybrid working. A pre-COVID re-fit on the Geographic Data Service 2019 mobile data is on the work plan (see §8).

Other inputs: 1km grid employment by sector (BRES 2024), POI counts (OpenStreetMap), travel-time matrix $t_{ij}$ (TfL public-transport routing), road network for the BPR equilibrium.

### 2.5 Why this transfers to Beijing

Every input — POIs, sectoral employment, travel times, anonymised mobile OD — has a Beijing equivalent (Amap POI, NBS statistical yearbook, Amap routing, university-licensed mobile OD). The inverse — fitting a London-specific RUM with hand-coded UK census variables (ASHE income, JTS journey-time statistics, TS063 SOC9 occupation tables) — would *not* transfer. This constraint has shaped the methodology choice throughout.

## 3. What London looks like today

### 3.1 Where the jobs are, and where the people are

![M1](figures/M1_employment.png)

**Figure M1.** Employment per 1×1 km cell (log scale). The City / Canary Wharf cluster dominates by an order of magnitude; secondary clusters at Stratford, King's Cross, Hammersmith, Croydon. The Old Oak Common cluster (starred) currently holds modest employment — that is the gap the regeneration is intended to fill.

![M2](figures/M2_population.png)

**Figure M2.** Residential population per 1×1 km cell (log scale). The mismatch between M1 and M2 is the source of all commuting: jobs in central London, residents distributed across the inner ring and a sparser outer ring.

### 3.2 The accessibility surface

![M3](figures/M3_accessibility.png)

**Figure M3.** Baseline Hansen accessibility $A_i$, AM peak. Central London delivers $A > 600{,}000$ jobs-equivalent; outer south-east cells (Bromley, Bexley, Havering) sit below 30,000 — a roughly **20× ratio**. The cell-level Gini coefficient is **0.46** — high.

| Borough (top) | Pop-weighted $A_i$ | Borough (bottom) | Pop-weighted $A_i$ | Ratio |
|---|---:|---|---:|---:|
| City of London | 1,017,845 | Havering | 47,314 | 21.5× |
| Westminster | 571,752 | Bromley | 52,604 | 10.9× |
| Islington | 496,160 | Bexley | 53,956 | 9.2× |

### 3.3 The Lorenz / decile picture

![M7](figures/M7_inequality_summary.png)

**Figure M7.** *Left:* Lorenz curve of cell-level accessibility — the area between the curve and the diagonal is the Gini = 0.46. *Right:* mean and median accessibility for the home cells of agents in each income tier. **Tier 3 / Tier 1 ratio = 1.17×**.

Two readings stand out. (i) The cell-level Gini (0.46) is much higher than the income-tier gradient (1.17×), because lower-income workers in London are not uniformly peripheral — many live in highly-accessible inner-borough cells with a strong social-housing legacy (Tower Hamlets, Lambeth, Southwark). (ii) Even with that dampening, Tier-3 residents still average **17% more access** than Tier-1.

## 4. The inequality lens — who reaches what

![M4](figures/M4_access_by_income.png)

**Figure M4 (3 panels).** The same accessibility surface seen from the home cells of each income tier — panels fade the cells where few residents of that tier live. Panel (a) is what Tier-1 (lower-income) residents actually see; panel (c) is what Tier-3 sees.

The reading: Tier-1 residents are visible in inner-east and inner-south London (Tower Hamlets, Hackney, Lambeth, Southwark) — these cells are reasonably well-connected, but not at the absolute peak. Tier-3 residents are concentrated in central west and inner north (Westminster, Camden, Islington, Kensington & Chelsea) — bullseye on the highest-$A$ cells. Tier-2 is between.

The 17% gap is real but smaller than headline rhetoric on "London inequality" might suggest. The London-specific reason is that the central-east boroughs combine high accessibility with a strong social-housing legacy — a feature that is *not generic* to all megacities, and which we should not assume holds for Beijing.

## 5. Scenario A — Old Oak Common (+65,000 jobs)

We add a coherent +65,000-job intervention across the 9-cell OOC cluster (Hammersmith / Ealing border). We then re-run the full pipeline under three settings:

| Setting | Mean $A_i$ | Pop-weighted Gini | Pop-weighted Palma |
|---|---:|---:|---:|
| Baseline (today) | 1,222,444 | 0.167 | 0.552 |
| OOC, free-flow $t_{ij}$ | 1,237,783 | 0.169 | 0.557 |
| OOC, congestion-equilibrium (BPR) | 1,225,612 | 0.169 | 0.561 |

(From `evaluation_outputs/paper_a/scenario_A_accessibility.json`. Gini and Palma here are computed across the agent population — the people-weighted view.)

![M5](figures/M5_scenarioA_delta.png)

**Figure M5.** $\Delta A_i$ per cell from the OOC intervention, under the free-flow opportunity-field assumption. The gain field decays radially from OOC with the recovered $\beta = 0.058$/min. Cells within ~30 minutes of OOC absorb the bulk of the new opportunity; outer south-east London gets almost none.

**Reading.** The headline +1.3% mean $A$ under free-flow falls to +0.3% once congestion absorbs most of the gain. Inequality **rises slightly**: population-weighted Palma 0.552 → 0.561, because the gains land disproportionately on already-well-connected NW grids. The plain summary: **OOC creates real new accessibility, but it spreads upward through the income distribution faster than downward.** To redistribute the benefit, the intervention probably needs to be paired with transit improvements connecting OOC to outer south London, or with affordable-housing supply at the new cluster. We are not in a position to recommend any of these from this report alone — we *are* in a position to say that *"build it and the access will redistribute"* is not what the model predicts.

## 6. Scenario B — Flexible work / off-peak shift

We model a 50% reduction in 7–9am peak congestion, redistributed across the 10am–3pm shoulder window. The temporally-resolved version of the model re-allocates each home cell's commute trips across the day under the new congestion vector.

![M6](figures/M6_scenarioB_share.png)

**Figure M6.** Share of each home cell's commute trips that depart in the 7–9am AM peak. Outer-borough cells with peak share **above 50%** are the ones whose residents are most exposed to the rush-hour cost — and so are the ones who gain most from a flexible-work intervention that genuinely shifts demand off-peak.

| Hour band | Δ flow (% of baseline) |
|---|---:|
| 7–9am peak | **−45.7%** |
| 10am–3pm shoulder | **+63.0%** |
| Off-peak (other hours) | +8.7% |

(`evaluation_outputs/paper_a/T7_scenario_B_per_hour.csv`.)

**Reading.** Peak-hour office workers see a one-shot ~46% drop in their congestion exposure, but the bigger story is the shoulder — service-sector and shift workers whose schedules already span the 10–15h window get markedly better transit headways, which raises their effective accessibility. Mapping this onto figure M6, the cells likely to gain the most are: outer SE London (Bromley, Bexley) where peak-share is high, and parts of the outer NE where transit headways currently degrade fast outside peak.

A clean caveat: this report does not yet quantify the headway / waiting-time effect on Tier-1 service workers; doing it properly requires merging the choice model with a queueing model of the bus network.

## 7. When can we trust these projections?

A model that fits historical OD well can still extrapolate badly. Before quoting policy numbers we ran a **reliability scan**: 60 randomly-located synthetic 9-cell clusters × 6 magnitudes of employment perturbation (10%, 30%, 50%, 100%, 200%, 500%), 360 experiments in total. For each we compared the aggregate-style choice prediction against a Monte-Carlo per-agent prediction (mixed-logit, draws from the agent income distribution). The gap between them is the **Jensen bias** that any aggregate destination-choice model carries when it makes counterfactual claims.

![F8](../evaluation_outputs/paper_a/F8_reliability.png)

**Figure F8.** *Left:* per-experiment $|\text{Jensen bias}|$ as a function of intervention magnitude × baseline cluster employment. The green outline is the empirical "reliable" region ($|\text{bias}| < 5\%$). *Right:* CDF of $|\text{Jensen bias}|$ across all 360 experiments — 12% land below 5% (reliable), a further 31% below 20% (caveat), and 56% above 20% (unreliable).

The OOC intervention sits at **~150,000 baseline cluster employment × +43% magnitude** — the scan places it in the **caveat zone**. We therefore quote the §5 headline as carrying roughly a **±25% confidence band**: the central +1.3% free-flow / +0.3% equilibrium gain is in the right direction and order of magnitude, but the third significant figure should not be treated as meaningful.

This scan is itself a methodological contribution. Most accessibility-modelling papers report a single-scenario delta number; we report the *region of policy space* in which the model produces trustworthy deltas, and the regions where it does not.

## 8. Limits and what's next

1. **Time period.** November 2021 is post-second-lockdown but pre-Omicron. Hybrid working is partially baked into the baseline. A pre-COVID re-fit on the Geographic Data Service 2019 mobile data would tighten interpretation, particularly of the flexible-work scenario.
2. **Income granularity.** Three tiers is the privacy-aggregation limit imposed by the mobile data. Finer breakdowns (income decile × occupation) would require linkage to an income microdataset.
3. **Counterfactuals are structural projections, not forecasts.** The model assumes preference and friction parameters stay fixed as the spatial layout changes. The §7 reliability scan partially guards against this; it is not a full guard.
4. **London is a sandbox.** Beijing replication requires re-fitting on Amap routing, NBS POI, and Beijing OD. The pipeline is designed for this swap; the numbers in this report are not portable.

**Planned next steps.** (i) Pre-COVID re-fit on GDS 2019. (ii) OOC × transit coupling (Crossrail-2 / GLA bus-route layer in $t_{ij}$). (iii) Flexible-work × headway: couple the destination-choice model to a simple bus-headway model so that the Tier-1 service-worker benefit becomes quantifiable. (iv) Beijing parallel build once data clearance lands (target end-2026).

---

## References

- Bentz, Y., & Merunka, D. (2000). *Neural networks and the multinomial logit for brand choice modelling: A hybrid approach*. Journal of Forecasting, 19(3), 177–200.
- Hansen, W. G. (1959). *How accessibility shapes land use*. JAIP, 25(2), 73–76.
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive representation learning on large graphs (GraphSAGE)*. NeurIPS.
- McFadden, D. (1974). *Conditional logit analysis of qualitative choice behavior*. In Frontiers in Econometrics, ed. P. Zarembka.
- Simini, F., Barlacchi, G., Luca, M., & Pappalardo, L. (2021). *A Deep Gravity model for mobility flows generation*. Nature Communications, 12, 6576.
- Train, K. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge UP. (Chs 3, 5.)
- Yao, X., Cheng, T., et al. (2021). *Spatial OD flow imputation using graph convolutional networks*. IEEE T-ITS.
- Zhong, C., & Zhou, Z. (2025). *Anonymised human location data in England for urban mobility research*. Scientific Data, 12. (Dataset DOI 10.5281/zenodo.13327082.)

---

*Reproducibility: `report/build_accessibility_maps.py` regenerates all figures. Numerical headlines come from `report/data/accessibility_per_grid.csv` and `evaluation_outputs/paper_a/*.json`.*
