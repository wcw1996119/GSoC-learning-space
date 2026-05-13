# Historical Retrodiction (London 2011 → 2021): Scope + Methodology

_2026-05-06 — committed after user re-opened the credibility question._

## 1 · Why retrodiction at all

Per the 5-2 / 5-4 stated priority chain (`memory/project_v2_full_model.md`):
> Credibility ranking: historical retrodiction > multi-method triangulation
> > sensitivity > theory > OOS predictive > external benchmarks > in-sample.

Every other validation route on the Phase B+ list (D9 multiverse, D10 plausibility,
D11 triangulation) is a **surrogate**. Retrodiction is the only one with
ground truth.

## 2 · Two interpretations of "validates the model"

| | Tight scope | Loose scope |
|---|---|---|
| Claim | "model recovers a known intervention's effect within X% error" | "model's spatial pattern of predicted ΔF matches observed ΔF in direction" |
| Cost | High — need clean train/test split, comparable units, no contamination | Low — can compare correlation maps |
| Defensibility | Reviewer accepts as evidence model is "correct" | Reviewer treats as supportive but not decisive |

We do **both**, report **tight scope** as the headline.

## 3 · Why Beijing transferability is *not* threatened

Method validation ≠ application validation. Paper A frames the deep-choice
GNN as **a method**, retrodicted on London. Paper B applies that **already-
validated method** to Beijing using surrogate validation (multi-baseline +
elasticity bracketing + plausibility scan). Same pattern as Yao 2021 (method
on Beijing imputation, transferable elsewhere) and Simini 2021 (Deep Gravity
on global cities, validated in a few).

If London retrodiction reveals the method is broken → both papers need
reframing, but you would **want to know** that before publishing.

## 4 · Concrete experimental design

### 4.1 Two snapshots
- **t = 2011**: Census 2011 workplace OD (NOMIS WU03BEW), BRES 2011 employment, derived 2011 sector mix.
- **t = 2021**: Census 2021 workplace OD (we have), BRES 2024 ≈ 2021 employment (3-year drift, document caveat).

### 4.2 Treatment area: Stratford regeneration
- 2011: Stratford (Newham, postcode E20 + immediate environs) was relatively low-employment, primarily residential.
- 2011-2021: Westfield Stratford City (Sept 2011), 2012 Olympics, Olympic Park redevelopment (Here East, BBC Sport, Stratford International commercial blocks). Daytime employment grew an order of magnitude.
- **Treatment cluster**: 5-10 grids covering Stratford + Olympic Park.

### 4.3 Train / test split protocol

```
Inputs at t=2011 →  Trained model M_2011  →  Predicted F_2011_pred  →  fit error
                                          |
                                          ├── Apply do(X_Stratford = X_2021)
                                          ↓
                                          Predicted F_pred_2021_under_intervention
                                          |
Observed F_2021                           ↓ compare
                          ←──────────────  
```

**Steps**:
1. Train Phase B v6 on t=2011 OD + 2011 employment + 2011 t_ij (free-flow, assumed time-invariant for car).
2. Verify fit on 2011 (sanity: CPC > 0.25).
3. Construct intervention: `do(X_Stratford ← 2021 employment levels)` while keeping all other grids at 2011 values.
4. Forward through model (with BPR equilibrium for car).
5. Compare predicted ΔF (Stratford 2021 vs Stratford 2011-counterfactual) to observed ΔF (Stratford 2021 vs Stratford 2011 actual).

### 4.4 Headline metrics

| Metric | Definition | Threshold for "model retrodicts well" |
|---|---|---|
| Stratford **inflow ratio** error | (predicted Δin / observed Δin) | within [0.5, 2.0] |
| **Sign agreement** on top-20 destinations affected | % of top-20 dests where predicted sign == observed sign | ≥ 60% |
| **Spearman ρ** on per-grid Δinflow vector | rank correlation between predicted and observed full Δ | ≥ 0.4 |
| **Cluster-bound flow Spearman** | among origins commuting to Stratford in 2021, rank correlation of predicted vs observed | ≥ 0.4 |

Any single threshold failing → write up as "model captures direction but
not magnitude" or similar honest framing. Multiple failing → "method
needs refinement before publication".

### 4.5 Confounding to flag and isolate

| Confound | Mitigation |
|---|---|
| 10-year time gap → mode shares shift, accessibility patterns change for non-Stratford reasons | Limit comparison to **destination-side** ΔF, not origin-side; isolate Stratford with control boroughs |
| COVID 2020-21 reduced commuting volumes by ~20-30% in some sectors | Aggregate to annual flows; report multipliers in % terms not absolute counts |
| Crossrail (Elizabeth line) opened 2022, after 2021 Census, but Crossrail-area expectations may have shifted commuting patterns earlier | Drop Crossrail-served origins from validation sample (or run as ablation) |
| MSOA boundary changes between 2011 and 2021 | Use ONS 2011→2021 MSOA correspondence table |
| 2011 OD covers usual residents 16-74; 2021 OD covers 16+ | Renormalise to age-comparable subsets |

### 4.6 Negative control

Pick a borough with **no major intervention** between 2011-2021 (e.g. Bromley, mostly stable). Predict 2021 OD using 2011-trained model with 2011 features held constant (no intervention). Predicted ΔF should be ≈ 0; observed ΔF should also be ≈ 0 modulo noise. Large predicted ΔF → model is unstable (false alarm).

## 5 · Data acquisition checklist (estimated 1-2 hours)

- [ ] **NOMIS WU03BEW 2011 OD at MSOA level**: dataset NM_1228_1, download CSV for all London MSOAs (~983 → 967 origin-MSOA after boundary changes), origin × destination, with mode breakdown.
  - URL pattern: `https://www.nomisweb.co.uk/api/v01/dataset/NM_1228_1.data.csv?...`
- [ ] **NOMIS BRES 2011 employment by MSOA + by SIC** (~18 sectors): dataset NM_142_1, download with 2011 year filter.
- [ ] **ONS 2011→2021 MSOA correspondence table**: from ONS Open Geography Portal.
- [ ] Build adapter `data/scripts/build_2011_grid_od.py`: MSOA→grid disaggregation using the existing `grid_msoa_primary.csv` mapping (potentially with population weights for accuracy).
- [ ] Build adapter `data/scripts/build_2011_grid_static.py`: BRES MSOA → 1km grid, log+zscore using same μ, σ as 2024 (so model sees same scale).

## 6 · Trainer changes needed (~30 min)

The Phase B v6 trainer takes (static, F_ij, t_ij, log_d, occ_match, income_score, wage_score). For 2011:
- static_2011: rebuild from BRES 2011 → grid (same 22 columns)
- F_ij_2011: from WU03 + grid mapping
- t_ij_2011: assume = 2024 free-flow (network similar; minor, document caveat)
- log_d, occ_match, income_score, wage_score: not strictly time-varying;
  reuse 2024 values as approximation, document caveat.

Train script: `experiments/paper_a/train_phase_b_v6_2011.py` — straight clone
of `train_phase_b_v6_ckpt.py` with 2011 data inputs.

Output: `phase_b_v6_2011_seed{0,1,2}.pt`.

## 7 · Retrodiction script (~30 min)

`experiments/paper_a/retrodict_stratford.py`:
1. Load 2011-trained ckpt.
2. Build 2011-baseline X.
3. Build do(X) = 2011 X with Stratford grids replaced by 2021 X for emp block.
4. Forward → F_pred_under_intervention.
5. Compute ΔF_pred = F_pred_under_intervention − F_2011_pred.
6. Compare to observed ΔF = F_2021_obs − F_2011_obs at Stratford bound.
7. Report 4 metrics from §4.4.

## 8 · Time budget (incremental)

| Step | Estimated hours |
|---|---|
| Data acquisition (NOMIS WU03 + BRES + correspondence) | 1.5 |
| Build adapters + diagnostics | 1.5 |
| Train 2011 model (3 seeds × ~30s) | 0.05 |
| Retrodict + metrics + write up | 1.5 |
| **Total** | **~4.5 hours** |

If acquisition takes longer (NOMIS API quirks), buffer to 6 hours.

## 9 · Outputs

- `methodology/retrodiction_scope.md` (this file)
- `data/processed/grid_od_2011.csv`, `grid_static_features_2011.csv`
- `evaluation_outputs/paper_a/retrodict_stratford.json` (4 metrics + per-grid maps)
- `evaluation_outputs/paper_a/retrodict_negative_control.json` (Bromley control)

## 10 · Failure response plan

If headline metrics fail (e.g. Stratford inflow ratio outside [0.5, 2.0]
across all 3 seeds):
1. Diagnose: feature drift, model bias on extreme growth, BPR misspecification.
2. Reframe Paper A: title shift from "method validated" to "method + identified
   failure modes for high-growth interventions". Still publishable in TMLR / JTG.
3. Re-baseline Paper B target: keep deep-choice frame but downweight
   policy-prediction claims; emphasise inverse choice recovery as the
   contribution.
