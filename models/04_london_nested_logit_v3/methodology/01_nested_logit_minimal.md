# Minimal Nested Logit Design (Phase 2, Smoke Version)

**Status**: Phase 2 minimal — full design pending Phase 1 lit review.
**Goal**: Test whether adding nested structure to V_RUM reduces GNN's residual share.

## Constraint discovered during Phase 0 survey

v2 uses **observed** mode share π_m(i,j) (Census) and **observed** tier share π_k(i)
as mixture priors. Model learns only destination choice within each (m, k).

Joint (mode, destination) observations are **not available** — only aggregate
F_{ij,t}. So classical nested logit with **learned** upper-nest mode choice is
infeasible without mode-specific OD data.

**Workaround for smoke validation**: add λ_m as per-mode dispersion (temperature)
parameter on the destination softmax. This is a partial nested logit (no learned
upper nest), but it tests whether nest-like structure adds RUM expressiveness.

## Mathematical form (minimal)

For each (origin i, hour t, mode m, tier k):

```
V_RUM_{t,i,j,m,k}  =  β_{t,m} · κ_k · t_{ij,m}  +  γ_m · log_d_ij  +  δ_k · OccMatch_ij
V_GNN_{t,j}        =  α · V_jt                                    (destination-only, GNN output)

V_total            =  (1 - δ_blend) · V_RUM + δ_blend · V_GNN     (Wang TB-ResNet blend)

# NEW vs v2: divide by λ_m before softmax
log P(j | i,t,m,k) =  log_softmax_j ( V_total / λ_m )

# Mixture (unchanged from v2):
log P(j | i,t)     =  logsumexp_{m,k} [ log π_m(i,j) + log π_k(i) + log P(j|i,t,m,k) ]
```

### λ_m parameterization

λ_m ∈ (0, 1] via:
```
λ_m  =  sigmoid(raw_lambda_m)   +   ε_min      where ε_min = 0.05 to avoid λ → 0
```
Actually use: `λ_m = ε_min + (1 - ε_min) * sigmoid(raw_lambda_m)` to keep λ ∈ [0.05, 1].

Initialization: raw_lambda_m = 10 → λ_m ≈ 1 (start as v2 baseline). Let model learn
to lower λ_m if nested structure helps.

### What λ_m means

- λ_m = 1: destination softmax temperature is 1 (v2 behavior, no nest correlation)
- λ_m < 1: softmax sharpens **for mode m** — destinations within mode m become more
  concentrated. Captures "within mode m, alternatives are tightly substitutable"
- λ_m → 0: extreme sharpening (degenerate). Numerical instability — hence ε_min.

This is **not** the full GEV nested logit (no upper nest IV), but the destination
dispersion parameter has the same mathematical role as nested logit's λ.

## Diagnostic test (smoke train)

Run 50ep CPU, single seed, blend_max=1.0 (free), all else == v2 baseline.

Compare to v2 baseline (3 seeds × 200ep, blend_max=1.0):
- v2 CPC = 0.485 ± 0.001
- v2 δ_free ≈ 0.446 (learned, no cap)
- v2 has no λ_m (effectively λ_m = 1)

| Outcome | Interpretation |
|---|---|
| λ_m all ≈ 1.0 (within ±0.05) | Nested structure doesn't help; IIA was fine in v2 data; δ_free should stay ~0.45 |
| Some λ_m < 0.8 + δ_free drops to < 0.35 | **Hypothesis confirmed** — nested adds expressiveness, GNN burden reduced |
| λ_m diverge (some → ε_min) + CPC drops | Numerical instability or overfitting; need regularization |
| λ_m mixed (e.g., car ≈ 1, walk ≈ 0.5) + CPC stable | Asymmetric mode-specific nest correlation — interesting paper finding regardless of δ direction |

## What this smoke does NOT test

- Full nested logit with learned mode choice (needs F_{ij,m,t})
- Richer attribute portfolio in upper nest (needs Phase 1 lit review)
- Welfare analysis under nested structure (Wang Block 6)
- Identifiability of λ_m formally (Hessian test, Wang Block 2)

These are next steps **if smoke succeeds**.

## Failure modes to watch

- **λ_m → ε_min**: collapse to point-mass destination prediction; gradient explosion
- **CPC drops significantly** (e.g., < 0.45): λ_m is hurting fit; bug or over-flexibility
- **β_t,m blows up**: when λ_m < 1, the effective β is β / λ_m which can grow; check β scaling
- **Numerical underflow in log_softmax**: ensure subtract-max trick is in place (PyTorch's log_softmax handles it)

## Implementation files

- `models_lib/inverse_rum/nested_logit_head.py` — copy of v2 MixtureRUMHead +
  `raw_lambda_per_mode` param + λ_m property
- `models_lib/inverse_rum/dual_branch_nested_trainer.py` — copy of v2 trainer with
  nested head plugged in, `_per_hour_log_p` divides logits by λ_m
- `experiments/paper_a/train_nested_smoke.py` — minimal entry point, loads v2 data
  via relative path, 50ep CPU, single seed

## Expected runtime

50ep CPU on London 1725-grid × 24h: ~30-45 min (per v2 benchmark for 50ep).
Single seed. No GPU needed.
