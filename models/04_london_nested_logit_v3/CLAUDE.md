# CLAUDE.md (v3)

Guidance for Claude Code working in **04_london_nested_logit_v3/**.

## Critical context

- **This is v3.** v2 (`../03_london_full_model_v2/`) is frozen at Wang Sprint Block 1 / paper draft v2 (commit `19eaad7`). Paper A v2 may submit workshop on Path C framing. **Do NOT modify v2 from this folder.**
- **v3 reason for being**: v2 hit a wall — linear MNL utility (V_RUM) cannot express London's multi-centric spatial structure, so GNN was forced to carry ~50% of the signal (δ_free ≈ 0.45). Wang TB-ResNet framework demands theory-dominant (δ ≤ 0.3) for identifiability + counterfactual reliability. v3 redesigns RUM head as **nested logit** to break the IIA limitation, intending to let theory carry more weight by enriching it.
- **Beijing is the real target**, London is sandbox. v3 architecture decisions must be Beijing-friendly. (Same red line as v2.)
- **Paper B (Beijing) primary, paper A workshop submission optional.** No rush on paper A.

## Architecture decisions (do not re-discuss)

### V_RUM: Nested logit on graph
- Two-level choice: upper nest = **mode** (car / transit / walk), lower nest = **destination | mode**
- Math:
  ```
  V_{ij,m}^lower  = -β_m·t_{ij,m} + γ·log(M_j) + δ_m,tier·income_term + (attribute interactions)
  IV_m(i)         = log Σ_j exp(V_{ij,m}^lower / λ_m)
  V_m^upper(i)    = α_m + θ_m·X_{i,m}^mode + λ_m·IV_m(i)
  P(m|i)          = exp(V_m^upper) / Σ_{m'} exp(V_{m'}^upper)
  P(j|i,m)        = exp(V_{ij,m}^lower/λ_m) / Σ_{j'} exp(V_{ij',m}^lower/λ_m)
  P(m,j|i)        = P(m|i) × P(j|i,m)
  ```
- λ_m ∈ (0, 1] per mode = 3 new parameters; estimated, **paper finding in itself**
- McFadden 1978 GEV family — identifiability theory exists
- Welfare analysis (consumer surplus) still works → Wang Block 6 unlocked

### V_GNN attachment point
- GNN modifies **lower nest** (destination utility) only. Upper nest stays pure RUM.
- This **structurally protects** mode choice from NN influence — answers Wang's "theory must dominate causal interpretation" concern at the mode level.
- GNN itself: **keep GraphSAGE** in Phase 3. Phase 4 considers GAT / Graph Transformer / hierarchical pooling **only if** evaluation in Phase 5 shows real need.

### Attribute portfolio (Beijing-friendly red line)

**To include** (depends on Phase 1 lit review — Schwanen 2003 / Crane 2007 / Susilo 2015 family):
- `income_tier` (lower nest δ; carryover from v2)
- `car_ownership_rate` (upper nest, mode availability discriminator)
- `household_composition` (upper nest, family/children → mode preference; Schwanen)
- `occupational_class` (upper nest or lower interaction; knowledge / service / manual)
- `jobs/housing balance` (lower nest, destination)
- `transit_accessibility` (upper nest, available in London as PTAL, in Beijing as 公交线网密度)

**Do NOT use** (London-only, no Beijing equivalent):
- ASHE income absolute value
- Census 2011 mode shares
- TS063 SOC9 occupational detail
- JTS (Journey Time Statistics)

## ⚠⚠⚠ Plan E baseline TRUE config (from .pt args, 2026-05-22 recovery)

The README in `data/autodl_bundle/` mixes 5/19 sdt_v1 flags with 5/20 Plan E flags. **The .pt file `evaluation_outputs/paper_a/v3l_planE_full_s0.pt` is the only source of truth** for Plan E's actual args. Critical differences from README:

| Flag | README says | .pt actual |
|---|---|---|
| `--use-frozen-match-mask` | required | **NOT USED** |
| `--match-mean-mult` | 1.0 | N/A (frozen mask off) |
| `--match-mask-log-penalty` | -50 | N/A |
| `--k-match-min-per-soc` | 20 | **0.1** (default) |
| `--tau-match-floor-mult` | 0.5 | **0.0** (default) |
| `--gnn-mode` | (unspecified) | **residual** |
| `--gnn-residual-scale-init` | (unspecified) | **0.5** |

**True Plan E command** (= CPC 0.557 baseline):
```
python experiments/paper_a/train_cervero_shen.py \
    --epochs 500 --device cuda --seed 0 --patience 100 \
    --use-dual-pair-encoder \
    --use-tier-mixture --use-soc-mixture \
    --use-consideration-filter --use-time-gate --use-self-loop-boost \
    --commute-hours 7,8,9,17,18 --use-nts-weighted-loss \
    --gnn-mode residual --gnn-residual-scale-init 0.5
```

**Hard rule**: when reproducing or extending a baseline, ALWAYS load the original .pt's args dict (`torch.load(pt).args`), NEVER trust README/memory recollection.

## ⚠⚠ Critical: `--gnn-mode residual` is REQUIRED for good baseline (added 2026-05-22)

CLI default is `--gnn-mode convex` (legacy), BUT all historical best results (sdt_v1 0.589, Plan E 0.564) used `--gnn-mode residual --gnn-residual-scale-init 0.1`. Without `residual`:
- Default convex mode: V_dest = (1 - blend) · V_RUM + blend · V_NN, blend init ≈ 0.5
- At ep 0, V_NN is random noise (~0.5 RMS), so 50% of V_dest is noise
- Training trajectory degraded: ep 0 vnll 6.71 (vs 4.89 with residual), CPC plateaus around 0.487
- **-0.075 CPC vs residual mode**

**Hard rule**: Every experiment must include `--gnn-mode residual --gnn-residual-scale-init 0.1` (or relevant config). 5/22 R6-R9 fiasco was caused by missing this flag.

## ⚠ Baseline comparison hygiene (added 2026-05-22 after R6-R9 fiasco)

**MUST follow** before reporting any "experiment X vs baseline" CPC/NLL comparison:

1. **NEVER use historical JSON as baseline** if you've changed `train_cervero_shen.py` or `cervero_shen_head.py` (or any code in their dependency tree) since the JSON was produced. Code changes have non-trivial unconditional effects (clamp logic, property formulas, init paths) even when new features are "default off".

2. **Always re-train a CURRENT-CODE baseline** with the exact same CLI args as the comparison experiment, minus the new flag(s). This is the only fair benchmark.

3. **At ep 0**, check `tnll`/`vnll`/`rum_rms` between baseline and experiment. If they differ by > 5% with identical seed/CLI, **the code paths differ** — investigate before continuing the experiment.

4. **Cross-check param init values** at ep 0 for shared modules (α_wage, γ_M, λ_b, k_match etc.). Different init values with same CLI = something changed in head/trainer that needs documenting.

5. Karpathy 4 principles (the link user always sends): Think Before Coding / Simplicity First / Surgical Changes / Goal-Driven Execution. The R6-R9 sequence violated #1 (assumed JSON baseline valid) and #4 (compared against wrong reference for goal of breaking ceiling).

## Key constraints

- Don't add features that have no Beijing equivalent. Beijing-friendly is a **hard constraint**, same as v2.
- Don't implement nested logit code before Phase 1 lit review is done. Attribute portfolio must be settled first, otherwise feature engineering gets thrown away.
- Don't claim Wang-grade reliability if Phase 5 evaluation shows δ_free > 0.4. Honest framing: Path C ("data-driven, characterized reliability"), not Path A ("theory dominate").
- BPR for t_ij in London; Amap API for Beijing — must remain swap-able. Same as v2.
- λ_m near 0 causes gradient explosion (log-sum-exp). Phase 4 must include numerical stability tests.

## v3 ↔ v2 relationship

- v3 shares **raw data** with v2: `../03_london_full_model_v2/data/raw/` (GEODS hourly OD 2019, BPR-derived t_per_mode, 1725 grids)
- v3 builds **new** processed data: `data/processed/` (attribute portfolio features, mode-level X_{i,m})
- v3 evaluation outputs live in v3 folder, not v2
- v2 results (`../03_london_full_model_v2/evaluation_outputs/paper_a/`) serve as **baseline comparison** in v3 paper

## Phase roadmap (2026-05-15 → ~6 weeks)

| Phase | Work | Time | Owner |
|---|---|---|---|
| 0 | v3 folder setup, CLAUDE.md (this) | 0.5 day | Claude (done) |
| 1 | Attribute lit review + portfolio decision | 2-3 days | **User** (parallel) |
| 2 | Nested logit math design doc | 0.5-1 day | Claude + user review |
| 3 | Implement nested_logit_head + adapt mixture_head + trainer loss | 3-5 days | Claude |
| 4 | Debug numerical stability (λ near 0, log-sum-exp nesting) + 200ep × 3 seed AutoDL | 2-3 days | Claude + AutoDL |
| 5 | Evaluate δ_free, λ_m, CPC, scenario A/B re-run | 1-2 days | Claude + user |
| 6 | Paper A v3 rewrite **or** Beijing paper B kickoff | 1-2 weeks+ | Joint |

## Commands

(To be filled when Phase 3 implementation lands.) Expected:
- `python train_nested.py --config baseline` — train nested logit + GraphSAGE
- `python evaluate.py --scenario A` — counterfactual eval
- `python diagnose_lambda.py` — λ_m identifiability + boundary check

## Files (current state, will grow)

- `models_lib/inverse_rum/` — to add: `nested_logit_head.py`, adapted `mixture_head.py`
- `experiments/paper_a/` — to add: `train_nested.py`, `run_v3_ablation.py`
- `methodology/` — to add: `01_nested_logit_design.md` (Phase 2 output)
- `data/processed/` — to add: attribute portfolio features (Phase 1 → Phase 3)
- `report/` — to add: `paper_a_v3_draft_{en,zh}.md` (Phase 6)

## User preferences

(Inherited from v2 / global memory)

- 中文交流，文档可中英混合
- 不要堆学术词 (避免 RUM 缩写堆砌、credibility 这种 buzzword)
- commit 不加 Co-Authored-By
- 不主动写 markdown 总结文档（除非用户要求）—— `methodology/` 下的文档是用户明确要求的设计文档，不算总结
- Phase 1 lit review 是**用户做**, 不是 Claude 做 (这是文献判断 + 北京可获取性判断, 必须用户自己看)

## 反事实预期 (诚实记录)

Phase 5 评估时 δ_free 落在哪个区间, 直接决定 v3 paper claim 强度:

| δ_free 区间 | 解读 | Paper claim 路径 |
|---|---|---|
| **0.20-0.30** | nested logit + attribute 真的解了 IIA, V_RUM 够强 | Path A: Wang theory-dominant 站得住, 强 claim |
| 0.30-0.40 | 部分改善, V_RUM 表达力还不够 | Path B: "spatial extension of TB-ResNet", 折中 claim |
| **0.40+** | London 空间本质太重, 架构救不了 | Path C: data finding, "London commuting requires balanced model", 弱但 honest |

任何一个 outcome 都是 paper-worthy, 不要在 Phase 5 时为了 push δ 低而调参作弊。
