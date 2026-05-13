# Paper A Outline v2

_Written 2026-05-06 EOD after Phase B+ completed D1, D2, D3, D5, D6, D7, D8, D10, D11 + retrodict + squeeze + het + D4. D9 multiverse running, D12 = this document._

## Title (working)

**"Deep Choice Modeling with Inverse Aggregate-OD Training: Recovering Behavioral Parameters and Simulating Urban Policy Scenarios"**

(No invented method names; positions in established **deep choice family** — Wang & Klabjan 2018, Sifringer 2020, Wong & Farooq 2021.)

## Target venues (in priority order)

1. NeurIPS 2026 workshop (Differentiable Almost Everything / AI4Science) — 55-65% accept estimate
2. TMLR — 40-50%
3. GeoAI / IJGIS / TGIS — 40-55%

## 1-sentence contribution

> "We recover behaviorally-grounded utility parameters (β, ψ, δ) from aggregate London Census OD by inverse training of a GNN-parameterised deep choice model; recovered parameters generalise across time (2011-2021 retrodict CPC 0.56), support two policy scenarios with quantified Jensen bias, and outperform classical (gravity 0.17) and modern (Deep Gravity) baselines (ours 0.34, p<0.0001 paired bootstrap)."

---

## Section structure

### Abstract (200 words)

Three sentences summary + key numbers + reproducibility note.

### 1. Introduction (1.5 pages)

- Motivation: urban policy interventions need counterfactual tools, not just OD prediction
- Gap: gravity / radiation / Deep Gravity FIT OD but don't RECOVER behaviour
- Our contribution:
  - (i) Inverse training of deep choice with GNN structural utility (Section 3)
  - (ii) Validation via 2011→2021 retrodiction + Bromley negative control (Section 4)
  - (iii) Two policy scenarios: spatial (Scenario A: +65k jobs at OOC) and temporal (Scenario B: flexible work) (Section 5)
  - (iv) Agent-level heterogeneity layer with WebTAG-derived spread to bridge to ABM and quantify Jensen bias (Section 6)
- Position in literature: deep choice family extension to GNN-parameterised utility on aggregate data

### 2. Related Work (1 page)

- Gravity / radiation: Wilson 1971, Simini 2012
- Deep Gravity: Simini 2021 — fully data-driven, no behavioural parameters
- TSI-GCN: Yao 2021 — GNN for OD imputation, single-city
- RUM literature: McFadden 1974, Train 2009, Ben-Akiva & Lerman 1985
- Deep choice models: Wang & Klabjan 2018, Sifringer 2020 (TasteNet), Wong & Farooq 2021 (ResLogit)
- Mixture & heterogeneity: Train 2009 ch. 6, Wardman 2014 VOT meta-analysis

### 3. Methodology (3 pages)

#### 3.1 Deep Choice Architecture
- V_j = GNN_θ(X_j) — GraphSAGE, depth 5, hidden 128
- U_ij = α·V_j + β_t·t_ij + γ·log_d_ij + δ·OccMatch_ij
- β_eff(i, j) = β_t · (1 + φ·z_o(i) + ψ·z_w(j)) — interaction-form heterogeneity
- Two variants:
  - Variant 4 (unconstrained): best-fit, CPC 0.34
  - Variant 6 (sign-constrained, mainstream direction φ≤0, ψ≥0): defensible interpretation, CPC 0.29

#### 3.2 Inverse Training
- Aggregate OD likelihood: L = Σ_i F_ij · log P(j|i)
- Top-K=50 sampling-of-alternatives (McFadden 1978)
- Sign reparametrisation via softplus for β_t, β_c, γ
- 150 epochs, AdamW, separate lr for θ vs RUM head

#### 3.3 Scenario Forward (Counterfactual)
- Frozen θ̂, β̂_t, ψ̂, ...
- do(X_S = x*): clamp subset of node features
- StructuralGNN.forward_under_intervention with **frozen baseline norm stats** (key methodological point — without this, normalisation washes intervention)
- BPR fixed-point UE solver for car congestion (forward only)

#### 3.4 Agent-level Heterogeneity Layer
- For each agent n: β_n(j) = β_base · (1 + φ·z_o(n) + ψ·z_w(j) + ε_n)
- ε_n ~ N(0, σ²(income_tier)) with σ from WebTAG / Wardman 2014 (CoV ≈ 0.4)
- Per-agent softmax → aggregate by mean
- Quantifies Jensen bias = aggregate prediction - agent prediction

### 4. Data (1 page)

| Source | Use | Year |
|---|---|---|
| NOMIS WU03EW (2011) | retrodict baseline OD | 2011 |
| NOMIS ODWP01EW (2021) | training OD (full Census 2021) | 2021 |
| ONS Census KS101EW | 2011 population | 2011 |
| BRES (NOMIS) | 2024 employment + sector mix | 2024 |
| ASHE workplace earnings | wage_j proxy | 2024 |
| WebTAG TAG A1.3 | VOT central + spread | 2024 |
| OSM POIs | accessibility features | 2024 |
| TfL stations | transit network | 2024 (filtered to 2011 for retrodict) |
| MSOA 2011→2021 lookup (ONS) | spatial harmonisation | — |

1725 grids × 22 static features × ~30K-1.2M OD pairs (2011 / 2021 full).

### 5. Results

#### 5.1 Headline performance
- Spatial holdout (4 boroughs: Westminster, Hackney, Brent, Bromley)
- CPC ensemble = 0.344 unconstrained, 0.290 constrained
- 95% bootstrap CI [0.325, 0.363]
- Paired bootstrap p < 0.0001 vs gravity & radiation
- Multi-metric: MAE 0.022, Spearman 0.219, KL/origin 0.741
- Per-borough: Westminster 0.48, Bromley 0.35, Brent 0.29, Hackney 0.20

#### 5.2 Baseline comparison (D8)
| Model | CPC |
|---|---|
| Gravity (Wilson) | 0.170 |
| Radiation (Simini12) | 0.174 |
| Deep Gravity (our minimal MLP) | 0.016 |
| GAT (4-head, our impl) | 0.246 |
| **Ours (SAGE unconstrained ensemble)** | **0.344** |

Caveat: minimal Deep Gravity does not match Simini 2021 published CPC; our scale (1725 1km grids, 1478 train origins) is much finer than their county-scale setting.

#### 5.3 Recovered parameters
- β_base = -0.13 (unconstrained), -0.058 (constrained)
- ψ = -0.15 (unconstrained, suggests inverse income gradient on aggregate), +0.39 (constrained mainstream)
- VOT (constrained): low-wage dest £4.49/h, mid £6.59, high £8.23
- Calibrated to Wardman RP central £6.5/h commute VOT

#### 5.4 Multiverse robustness (D9)
- 100 hyperparameter configurations (hidden, depth, K, lr, ...)
- All > gravity baseline 0.17 (X% of configs)
- Median val_CPC distribution
- (To fill in after D9 finishes)

### 6. Retrodiction validation (Section 4 in some structures)

#### 6.1 City-wide forward retrodict
- 2011-trained model + 2024 features → predicted 2021 OD vs observed full Census 2021
- CPC = 0.56 (vs in-sample CPC 0.71)
- Spearman across all OD pairs = 0.52
- Per-borough CPC 0.52-0.62

#### 6.2 Bromley negative control
- Suburban borough, no major intervention 2011-2021
- Bromley emp grew 1.55× (modest)
- Predicted Δshare +1.46pp / Observed +1.26pp = **1.15× ratio** (model well-calibrated)

#### 6.3 Stratford intervention case study
- Olympics + Westfield 2011-2021, employment +5×
- Predicted Δshare +1.33pp / Observed +0.23pp = 5.7× over-prediction
- Attributed to (i) MNL IIA + isolated single-cluster intervention; (ii) COVID 2021 commute disruption damping observed change at CBD-adjacent areas
- Per-destination Pearson = 0.99 (correct shape, magnitude inflated)

### 7. Policy Scenarios

#### 7.1 Scenario A: spatial intervention (+65k jobs at OOC 9-grid cluster)
- ΔInflow into cluster: +126 (free-flow) → +105 (BPR equilibrium, 17% pushback)
- Accessibility +1.34% → +0.45% (BPR ate 2/3 of gain)
- Inequality WORSE: Gini +1.26%, Palma +1.51%, Atkinson +2.53%
- **Regressive policy** in accessibility terms

#### 7.2 Scenario B: temporal intervention (flexible work, peak ×0.5, off ×1.5)
- Peak (h7-9) flow -46% (mass conserved per origin)
- Peak congestion: h08 mean t_ij -3.7% (33.99 → 32.73 min)
- Accessibility at h08: +10.2% (1.09M → 1.20M jobs in reach)
- Inequality BETTER: Gini -0.011, Palma drops
- **Progressive policy** — under-connected areas benefit most

#### 7.3 Agent-level heterogeneity (Section 6 alt)
- 10K agents with WebTAG-anchored σ (CoV 0.4)
- Stratford intervention: Jensen bias +2.1% (small)
- Random outer cluster: Jensen bias -16.7% (aggregate UNDER-predicts greenfield response)
- CoV sensitivity scan: |bias| < 4% across σ ∈ [0, 1] for Stratford
- Per-destination Pearson 0.99 (aggregate gets shape right, agent gets magnitude right)

### 8. Discussion (1 page)

- Methodological insight: aggregate inverse + agent-level heterogeneity bridge
- Two scenario types reveal opposite inequality effects (spatial regressive vs temporal progressive)
- COVID effect on 2021 Census limits Stratford magnitude validation
- Single-cluster intervention has IIA-induced over-prediction; multi-grid clusters needed for realistic policy modelling
- StructuralGNN normalisation invalidates intervention without baseline-frozen norm stats — methodological warning for future work
- Greenfield vs mature interventions: Jensen bias direction flips, aggregate model under-predicts greenfield

### 9. Limitations

1. Aggregate OD identification: ψ direction sensitive (variant 4 inverse, variant 6 mainstream); we report both
2. Free-flow t_ij in inverse training (BPR only in forward)
3. COVID 2021 Census effect on observed commute patterns
4. Per-borough Hackney CPC 0.20 (model weakest at inner non-CBD)
5. Single intervention case study (Stratford) for retrodict; multi-case study deferred
6. No within-sample mode choice (commuters' mode is observed, not modelled)
7. Deep Gravity baseline implementation minimal; full Simini 2021 implementation may close some gap

### 10. Beijing Application (Paper B preview)

- Same deep choice + GNN framework applies directly
- Beijing data inputs:
  - Amap-scraped travel times (replaces OSMnx free-flow)
  - Statistical yearbook employment (replaces BRES)
  - POI from public Amap dataset
  - VOT spread from Chinese transport literature (Bao et al., Wang et al.)
- Validation surrogates only (no Beijing OD retrodict possible due to data restrictions)
- Cross-city β recovery comparison left for future (cross-city invariance penalty, V-REx style)

### 11. Reproducibility

- Code: `experiments/paper_a/` + `models_lib/inverse_rum/`
- Trained ckpt: `phase_b_v6_seed{0..4}.pt`, `phase_b_v6_2011_seed0.pt`, `phase_b_v6_hourly_*.pt`
- Data: NOMIS open data (links in Section 4)
- Multi-seed ensemble = 5 seeds × ~30s training each
- Single ckpt training: ~30 sec on CPU
- Multiverse 100 configs: ~3 hours CPU

---

## Figures plan

| # | Caption | Source |
|---|---|---|
| F1 | Architecture diagram (GNN → V_j → softmax → flow) | Whiteboard / Tikz |
| F2 | London 1725-grid map with 4 holdout boroughs highlighted | matplotlib + geopandas |
| F3 | CPC bar chart: gravity / radiation / Deep Gravity / GAT / ours | matplotlib |
| F4 | Per-borough CPC bar chart with bootstrap CI | matplotlib |
| F5 | Retrodict scatterplot: predicted vs observed Δinflow per dest | matplotlib |
| F6 | Scenario A: ΔInflow heatmap on London map (free-flow vs BPR) | geopandas |
| F7 | Scenario B: per-hour flow distribution baseline vs scenario | matplotlib |
| F8 | Agent vs aggregate prediction scatter for Scenario A | matplotlib |
| F9 | Jensen bias vs CoV sensitivity scan | matplotlib |

## Tables plan

| # | Caption | Source |
|---|---|---|
| T1 | Headline leaderboard: model × CPC × CI × p-value | results 5.1-5.2 |
| T2 | Recovered parameters with 95% CI | results 5.3 |
| T3 | Per-borough CPC | results 5.1 |
| T4 | Multiverse summary (mean / std / p10/p50/p90) | results 5.4 |
| T5 | Retrodict 4-metric scoreboard | results 6.1-6.3 |
| T6 | Scenario A inequality table (Gini / Palma / Atkinson, before/after BPR) | results 7.1 |
| T7 | Scenario B per-hour congestion + accessibility | results 7.2 |
| T8 | Jensen bias by intervention type (Stratford / random / etc.) | results 7.3 |

---

## Estimated paper length

10-12 pages including figures + tables. Workshop format = 4-6 pages (compress methodology + scenarios into 2 pages). TMLR / journal = 15-20 pages with appendix.

## Next steps

1. **Wait for D9 multiverse to finish** (~50 min) → fill in 5.4 numbers
2. **Train hourly seeds 1-4** for Scenario B robustness (8 min × 4 = 32 min)
3. **Generate figures** (~half day)
4. **Write prose** (~2-3 days for first complete draft)
5. **Get supervisor review** (~1 week turnaround)
6. **Revise** (~1 week)
7. **Submit** (target NeurIPS 2026 workshop deadline TBD)
