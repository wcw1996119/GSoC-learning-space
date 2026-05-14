# 路线 3b' — Destination→Mode Nested Logit + Borough-level λ + Non-linear V_dest

**Status**: Phase 3b' (revised after smoke negative result on partial M→D nested)
**Date**: 2026-05-15
**Authors**: Claude + user (D→M direction is user's insight)

---

## 1. Motivation: why we redesigned from M→D to D→M

### Smoke result that triggered redesign

Partial M→D nested logit (200ep × 1 seed AutoDL) gave:

| Metric | v2 baseline | partial nested smoke | Δ |
|---|---:|---:|---:|
| CPC | 0.485 | 0.486 | noise |
| δ_free (GNN blend) | 0.446 | 0.446 | **zero** |
| λ_m (per mode) | n/a | {0.92, 0.94, 0.99} | nearly 1 |

λ_m did not move materially; δ did not move at all. Hypothesis "per-mode destination softmax temperature reduces GNN burden" is **falsified**.

### Why M→D is wrong for commuting

In M→D ("mode first, destination second"), the upper-nest mode-choice would imply that commuters cognitively select mode before destination. This conflicts with commuting reality:

- **Short-run (daily)**: workplace is fixed; destination is not an active choice.
- **Long-run (residential + workplace)**: destination is determined by housing market, career, school district — a multi-year equilibrium, not a daily logit decision.
- **Mode** is the only daily active choice (given fixed origin and destination).

The user's insight (2026-05-15): "通勤 destination 是相对固定的." This points to D→M as the structurally correct nesting for commuting OD modeling.

### D→M matches classical trip-distribution + mode-split

D→M nested logit is mathematically equivalent (Ortúzar & Willumsen 2011, ch.6) to the classical 4-step planning sequence:
1. **Trip distribution** — predict F_{ij,t} (where do trips go)
2. **Mode split** — given (i,j), what fraction by each mode

We treat step 1 as an equilibrium model (utility-based gravity with GNN correction) and step 2 as a real choice model (mode-level utility with KL constraint to Census).

---

## 2. Mathematical structure

### 2.1 Two-level decision tree

```
                    commute flow F_{ij,t}
                          |
              [Upper nest] choose destination j ∈ {1, ..., 1725}
                          |
        ┌─────────────────┼─────────────────┐
       j₁                j₂               j₁₇₂₅
        |                 |                  |
    [Lower nest]      [Lower nest]      [Lower nest]
    choose m | (i,j₁)   m | (i,j₂)        m | (i,j₁₇₂₅)
        ↓                 ↓                  ↓
   car / transit /    car / transit /    car / transit /
       walk               walk                walk
```

### 2.2 Lower nest: mode choice given (i, j)

For each (origin i, destination j, hour t, mode m):

```
V_mode(i, j, t, m) = β_m(d_ij) · t_{ij,m}        ← mode-time disutility (non-linear in d)
                   + c_m · cost_{ij,m}            ← if cost available, else dropped
                   + θ_m^T · X_{i, m}             ← mode-level origin features
                   + α_m                           ← mode bias
                   + κ_k · interaction_terms      ← tier × mode (optional)
```

Borough-level nest correlation parameter:

```
λ_b(j) ∈ [ε_min, 1]                              ← b(j) = borough of destination j
                                                   33 distinct λ values total
```

Mode-choice probability:

```
P_M(m | i, j, t) = softmax_m [ V_mode(i, j, t, m) / λ_{b(j)} ]

                    exp( V_mode(i, j, t, m) / λ_{b(j)} )
                = ────────────────────────────────────────
                    Σ_{m'} exp( V_mode(i, j, t, m') / λ_{b(j)} )
```

### 2.3 Inclusive value (mode-side log-sum)

```
IV_mode(i, j, t) = log Σ_m exp( V_mode(i, j, t, m) / λ_{b(j)} )
```

This is the "scaled total utility of all modes available for trip (i,j) at time t" — a one-dimensional summary of mode-level attractiveness.

### 2.4 Upper nest: destination choice

```
V_dest(i, j, t) = γ · log_d_ij                   ← gravity log-distance
                + δ · OccMatch_ij                 ← labour-match (income-tier interaction)
                + α · V_GNN_jt                    ← GNN destination embedding
                + λ_{b(j)} · IV_mode(i, j, t)     ← mode-side accessibility, coupled
                + (optional non-linear interactions, see §3)
```

Destination-choice probability:

```
P_D(j | i, t) = softmax_j [ V_dest(i, j, t) ]
```

### 2.5 Joint and observed-flow likelihood

Joint:
```
P(j, m | i, t) = P_D(j | i, t) · P_M(m | i, j, t)
```

But we observe **aggregate** F_{ij,t} (no mode-tag in OD data). So we marginalise modes:

```
P(j | i, t) = Σ_m P(j, m | i, t)
            = P_D(j | i, t) · Σ_m P_M(m | i, j, t)
            = P_D(j | i, t)             ← because Σ_m P_M = 1
```

**Crucial observation**: the destination NLL only depends on P_D directly. But P_D **depends on λ_b(j) through IV_mode**, which in turn depends on V_mode. So mode-side parameters (β_m, θ_m, α_m) influence destination prediction via the inclusive value coupling. **This is the structural channel through which V_mode informs F_{ij,t}.**

### 2.6 KL constraint loss

Without F_{ij,m,t} (mode-specific OD), we use Census observed pair-level mode share π_m(i, j) as supervisory signal on P_M:

```
L_KL = Σ_{i,j} F_{ij,t}-weighted-KL( π_m(i, j) || P_M(m | i, j, t) )

     = -Σ_{i,j,t} F_{ij,t} · Σ_m π_m(i, j) · log P_M(m | i, j, t)
                                                + (entropy of π_m, constant w.r.t. params)
```

We use the flow-weighted version so high-flow (i,j) pairs dominate (small/negligible flows get less weight, avoiding noise from rare OD pairs).

### 2.7 Total loss

```
L_total = L_NLL_destination  +  λ_kl · L_KL_mode

       = -Σ_{i,j,t} F_{ij,t} · log P_D(j | i, t)
         + λ_kl · L_KL
```

**λ_kl schedule** (hyperparameter, not λ_b):
- Start at 0, ramp up linearly to target value over first 10-20 epochs ("warmup")
- Target value: tune from {0.1, 0.5, 1.0, 2.0}; start with 1.0
- Without warmup, KL can dominate early and freeze V_mode at boring uniform mode shares before V_dest learns destination structure

---

## 3. Non-linear V components

To attack the **second** root cause (linear V is too restrictive), we add non-linearities:

### 3.1 Distance-dependent β_m(d)

```
β_m(d_ij) = β_m,0  +  β_m,1 · log(d_ij)
```

(monotone in log d, simple 2-parameter spline). Captures: time disutility is **steeper at longer distances** (long commute is disproportionately bad). v2's flat β_m can't express this.

Identifiability: β_m,0 and β_m,1 are jointly identifiable as long as t_{ij,m} and d_{ij} aren't collinear — they aren't (mode-specific times vary differently with distance).

### 3.2 OccMatch × distance interaction

```
δ_eff(d_ij) = δ_0  +  δ_1 · log(d_ij)
```

Captures: occupational matching matters more for far-away destinations (you don't commute 30 km unless the job really fits you).

### 3.3 (Optional, Phase 3b'.C) Mode × tier interaction

```
β_{m, k} = β_m · κ_k  + δ_mk
                       ↑
                  small interaction tensor (3 × 3 = 9 free params)
                  captures e.g. high-income drive farther by car
```

This is a richer version of v2's κ_k scaling. v2 has β_m · κ_k (multiplicative), we add δ_mk (additive correction).

---

## 4. Mode-level features X_{i, m}

Derived from existing v2 data — **no new data acquisition**. Per origin i and mode m:

| Feature | Formula | Intuition |
|---|---|---|
| `acc_m(i)` | log Σ_j jobs_j · exp(−β_ref · t_{ij,m}) | Hansen-style accessibility by mode |
| `mean_t_m(i)` | Σ_j t_{ij,m} · jobs_j / Σ jobs_j | mean travel time by mode, jobs-weighted |
| `min_t_m(i)` | min_j t_{ij,m} subject to jobs_j > threshold | best-case travel time |
| `mean_t_ratio_m(i)` | mean_t_walk / mean_t_m | how much faster than walking |
| `n_jobs_30min_m(i)` | # destinations with t_{ij,m} ≤ 30 and jobs_j > 0 | 30-min job catchment |

We start with 3 features (acc_m, mean_t_m, n_jobs_30min_m) per mode to keep θ_m small (3 × 3 = 9 params). Expand if smoke is promising.

**Identifiability sanity**: 3 modes × ~3 origin features = 9 θ params. Plus 3 α_m. Plus 33 λ_b. Plus 6 (β_m,0, β_m,1). Plus 2 δ (δ_0, δ_1). Plus 2 κ. Total ~55 RUM params. With 1725 × 3 = 5175 mode-share constraints + 1725 × 24 × 1725 ≈ 71M destination cells, strongly over-identified.

---

## 5. Architecture: where the GNN sits

```
GNN  →  V_GNN_jt  →  feeds into V_dest (upper nest, destination accessibility correction)

V_dest = γ log_d + δ OccMatch + α V_GNN + λ IV_mode + (non-linear)
                                ↑
                              GNN here, anchored on destination
```

V_GNN is **purely destination-level** (1 scalar per (j, t)). It corrects v2's gravity baseline at the destination softmax level — exactly what GraphSAGE was producing before. **No architecture change to GNN**; we change only the RUM head.

Blend / TB-ResNet (Wang style) optional layer:
```
V_dest = (1 − δ_blend) · V_dest_RUM + δ_blend · V_GNN
                                       ↑
                              maintains v2's blend_max parameter
                              (default 1.0 free, can constrain to 0.3 for Wang strict)
```

---

## 6. Numerical stability

Three known instabilities:

### 6.1 λ_b near 0
λ_b → 0 means `V_mode / λ_b → ∞`, gradient explodes.

**Bound**: λ_b = ε_min + (1 − ε_min) · sigmoid(raw_lambda_b), with ε_min = 0.05.
**Init**: raw_lambda_b set so λ_b ≈ 0.95 at start (close to v2 flat MNL baseline). Let model push down if useful.

### 6.2 λ_b > 1 (inconsistent with random utility max)
Mathematically, λ_b > 1 violates Daly-Zachary regularity → choice probabilities not from any valid utility-max model.

**Bound**: λ_b ∈ (0, 1] strictly enforced by sigmoid scaling.

### 6.3 Log-sum-exp overflow
IV_mode requires log-sum-exp over modes. With only 3 modes this is well-behaved (3 terms), but combined with KL constraint can blow up if V_mode magnitudes diverge.

**Mitigation**: clip raw V_mode logits to [-50, +50] before exp. Standard PyTorch `log_softmax` handles overflow internally; we additionally clip pre-softmax for safety.

### 6.4 KL warmup
KL term competing with NLL at initialization can freeze V_mode parameters at trivial uniform values.

**Schedule**: λ_kl(epoch) = target · min(1, epoch / warmup_epochs), warmup_epochs = 15.

---

## 7. Diagnostic plan (Phase 3b'.F)

After 200ep × 3 seeds on AutoDL, check:

| Quantity | Target / interpretation |
|---|---|
| `CPC` | ≥ v2 baseline 0.485 (richer model should fit at least as well) |
| `δ_free` (GNN blend) | **Hypothesis**: drops from 0.446 to 0.30-0.40 |
| `λ_b` distribution across 33 boroughs | Variance > 0.1 → spatial heterogeneity present; uniform → simpler model fine |
| `λ_b` map | Plot on London → paper Figure (mode-substitution geography) |
| `β_m,0, β_m,1` | β_m,1 < 0 (steeper disutility at long distance) → consistent with theory |
| `θ_m` | Sign should match intuition (high acc_car → high P(car), etc.) |
| KL loss final | Should converge near 0 (model matches Census mode share) |
| Mode share prediction | P_M(m\|i,j) at sample (i,j) ≈ observed π_m(i,j) |

### Three outcome paths (all paper-worthy)

| δ_free after 3b' | Verdict | Paper claim |
|---|---|---|
| < 0.30 | **Hypothesis confirmed**: nested + non-linear V displaces GNN | Path A: "structural model adequately captures spatial heterogeneity" |
| 0.30-0.40 | Partial improvement | Path B: "TB-ResNet extension to spatial commuting; spatial structure intrinsically requires NN supplement" |
| > 0.40 | Marginal | Path C: "data-driven balanced model; ruled out destination-IIA and linear-V as bottlenecks" |

---

## 8. Implementation file layout (Phase 3b'.B onwards)

```
04_london_nested_logit_v3/
├── models_lib/inverse_rum/
│   ├── nested_logit_head.py        ← obsolete (partial M→D, kept for ablation)
│   └── dm_nested_logit_head.py     ← NEW: DM nested + borough λ + non-linear V
├── experiments/paper_a/
│   ├── train_nested_smoke.py       ← obsolete (partial nested smoke)
│   ├── train_dm_nested.py          ← NEW: D→M training entry
│   ├── build_mode_level_features.py ← NEW: derive X_{i,m} from v2 data
│   └── pack_autodl_bundle.py       ← updated to bundle new modules
├── data/processed/
│   └── mode_level_features.npz     ← built by build_mode_level_features.py
└── methodology/
    ├── 01_nested_logit_minimal.md  ← partial nested design (now retired)
    └── 02_dm_nested_design.md      ← THIS DOCUMENT
```

---

## 9. Phase 3b' roadmap

| Phase | Task | Time | Owner |
|---|---|---|---|
| 3b'.A | This design doc | 0.5-1 day | Claude (done) |
| 3b'.B | Implement DM_NestedLogitHead | 2-3 days | Claude |
| 3b'.C | Non-linear V_dest (β_m(d), δ(d), interactions) | 2 days | Claude |
| 3b'.D | Mode-level features X_{i,m} engineering | 1 day | Claude |
| 3b'.E | Integration + numerical stability + AutoDL train | 2-3 days | Claude + AutoDL |
| 3b'.F | Evaluation + v2 comparison + paper draft v3 skeleton | 1-2 days | Joint |
| **Total** | | **~1.5-2 weeks** | |

Parallel (user, no blocking): Phase 1 attribute lit review (Schwanen 2003 / Crane 2007 / Susilo 2015 + Beijing data availability). Output feeds into θ_m feature decisions in 3b'.D refinement (not needed for initial 3b'.B-E).

---

## 10. Open design questions (need user input before Phase 3b'.B starts)

None blocking — start Phase 3b'.B with defaults below. Revisit if smoke unexpected:

- **λ_kl target**: default 1.0, sweep {0.1, 0.5, 1.0, 2.0} if hyperparameter matters
- **β_m(d) form**: default linear in log_d (2 params/mode); upgrade to monotone spline if non-linear is desired
- **Non-linear δ vs constant δ**: default add δ_1 · log_d; can disable if identifiability concerns
- **Mode-level features in θ_m**: start with 3 (acc_m, mean_t_m, n_jobs_30min_m); expand to 5+ after Phase 1 lit review if user adds household composition / car ownership

---

## 11. Honest framing for paper

This design is **not** a Wang-grade theory-dominant model. Honest claims:

✅ Nested logit (D→M) more behaviorally appropriate for commuting than flat MNL
✅ Borough-level λ_b is a substantive finding (33 spatial values, mappable)
✅ Non-linear V_dest tests the "linear utility too restrictive" hypothesis from negative smoke
✅ KL-constrained mode choice uses observed Census data without requiring (unavailable) mode-specific OD
⚠️ Without full mode-specific OD, upper-nest mode utility identification is partial; we acknowledge this in limitations
⚠️ If δ_free stays > 0.4, model is honestly "structural skeleton + spatial NN correction", not Wang theory-dominant

These honest caveats are paper-publishable as **transport methodology paper** (TR-B / EPB / IJGIS), not as ML methodology paper.

---

## 12. References

- McFadden 1978: "Modelling the choice of residential location" — original nested logit (D→M direction implicit)
- Ben-Akiva & Lerman 1985 ch.10: nested logit math + both directions formalized
- Ortúzar & Willumsen 2011 ch.6: trip distribution + mode split equivalence
- Daly & Zachary 1978: λ ∈ (0, 1] regularity condition
- Wang Mo Zhao 2020 TR-B: TB-ResNet, V_RUM + V_GNN blend (used here as architecture inspiration)
- Train 2009 ch.4: nested logit identifiability + IV interpretation
- v2 smoke (this folder, 2026-05-15): partial M→D nested failed → motivated this design
