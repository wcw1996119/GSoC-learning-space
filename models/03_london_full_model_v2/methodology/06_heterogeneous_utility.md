# Heterogeneous Utility Design (v2.2)

> **目的**：定义 v2.2 utility 函数的异质性设计——`β` 改成 `(income_tier × mode)` 固定 lookup table，`α_V / α_occ / γ` 用 softplus 强制 > 0。本文档**取代** `05_training_loss.md` §3 关于 β handling 的部分（单一 learnable β 的方案已废弃）。
>
> **关系**：本文档是 Step 2（Workplace Choice）utility 的实现规格，对应 `models_lib/heterogeneous_utility.py`。在 4-step 架构里属于 Step 2，训练规格属于 Step 4 Stage 2。
>
> **配套**：
> - `00_v2_architecture.md` — 架构总览（read first）
> - `05_training_loss.md` — 旧版 single-β loss 设计（**部分 obsolete**，本文档 §5 替代其 β handling 部分）
> - `../../02_london_commuting_model/methodology/02_counterfactual_evaluation.md` — 反事实评估三层框架（仍适用）

---

## 1. 背景 / Why heterogeneous utility

### 1.1 v2.1 single-β 的失败模式

v2.1 baseline utility:

```
V_ij = α_V·V_j(t) + α_occ·OccMatch_ij − β·t_ij(t) − γ·log d_ij + b
```

其中 `α_V, α_occ, β, γ, b` 全 learnable。训练完后：

| 参数 | 期望符号 | 实际学到 | 解释 |
|---|---|---|---|
| α_V | + | ≈ +0.4 | 合理 |
| α_occ | + (大) | **≈ −0.05** | **异常** |
| β | + | ≈ 0.06 | 数值合理但 unit 模糊 |
| γ | + | ≈ 0.3 | 合理 |

`α_occ ≈ −0.05` 意味着：occupation match 越高，utility 越低——直接违反 supervisor 5-2 时给的 substantive critique（"OccMatch 是 v2 contribution edge，必须是正向"）。

### 1.2 Identifiability 根因

把 V_j(t) 整体 scale `c` 倍 + β 同时 scale `1/c` 倍 → logits 不变 → loss 不变。`α_V` 是 learnable scalar 已经吸掉了一部分 scale，再加一个 learnable β 形成**冗余**。训练梯度沿"α_V × β = constant"的脊上随机走，最终 land 在哪儿取决于 init seed。

### 1.3 Supervisor 的 substantive critique（5-2 memory）

> "single weights 抹掉 agent heterogeneity. 一个 β 给所有人不对——SOC1 的 banker 跟 SOC9 的 cleaner 对 commute time 的容忍度不一样。"

### 1.4 用户 instinct

同一个 β 对所有 agent 是错的：从行为经济学（VOT 跟 income 强相关）和数据（NTS / LTDS 都有 income-stratified VOT 表）两边都有支持。

→ v2.2 直接把 β 拿出训练 loop，固定 lookup，让 learnable 参数从 5 个降到 4 个，识别性立刻改善。

### TODO + open questions

- [ ] 把 v2.1 的 `α_occ ≈ −0.05` failure log 存档到 `experiments/v2_1_failure/`
- [ ] 是否要 reproduce v2.1 这个 failure mode（demonstration purpose）

---

## 2. Decision summary table（locked）

| Item | Decision |
|---|---|
| **β strategy** | **Fixed lookup** (income_tier × mode), NOT learned |
| **α_V, α_occ, γ** | Learned **BUT softplus-constrained > 0** |
| **Heterogeneity dims** | income_tier (3) × mode (3); **NO age, NO family** |
| **OccMatch evaluation** | **Per-agent** (uses agent's specific SOC), aggregate at training |
| **Training utility** | `utility_aggregate(...)` with `beta_origin` (population-weighted) |
| **Inference utility** | `utility_personal(...)` with `beta(income_tier, mode_idx)` |
| **Learnable params** | 4 (`α_V_raw, α_occ_raw, γ_raw, b`)，每个 raw 经 softplus（`b` 不约束） |
| **Init values** | `α_V=0.5, α_occ=0.7, γ=0.5, b=0`（softplus 反算 raw） |

**Rationale 一句话**：β 锚到 WebTAG VOT 文献，剩下 4 个 scalar 在 RUM 框架下用 production-constrained MNL log-likelihood fit。

### TODO + open questions

- [ ] β 的 calibration 拿 NTS individual-level data 做 cross-check（伦理通过后）
- [ ] 是否要在 inference 时给 β 加 stochastic noise（mixed logit 风格）

---

## 3. β 表 — 文献 anchoring

### 3.1 Income tier 划分（按 SOC）

| Tier | SOC codes | 描述 | 占比（伦敦 Census 2021） |
|---|---|---|---|
| 1 (low) | SOC 6, 7, 8, 9 | Caring / Sales / Process / Elementary | ~32% |
| 2 (mid) | SOC 3, 4, 5 | Associate professional / Admin / Skilled trades | ~30% |
| 3 (high) | SOC 1, 2 | Managers / Professionals | ~38% |

**Why SOC 而不是直接 income**：Census 2021 没有 individual income, SOC 是最近的 proxy；同时 Beijing 复现时 SOC↔GBM mapping 已经有现成 table。

### 3.2 β_income 表（per minute disutility）

| Tier | β_income | Source / 推导 |
|---|---|---|
| 1 (low) | **0.12** | WebTAG VOT × 1.7 (low-income premium) |
| 2 (mid) | **0.07** | WebTAG TAG A1.3 (2024) commuting baseline VOT ≈ £15/hour |
| 3 (high) | **0.04** | WebTAG VOT × 0.6（high-income agents 时间机会成本绝对值高，但 utility scale 上 disutility 反而小，因为 t_ij 占总 utility 比重低） |

**注意**：β_income 不是直接的 VOT (£/hour)。它是**utility-scale 的 per-minute disutility**——已经吸收了"VOT × utility-to-£ conversion"那段 scale。所以 high tier 的 β 反而更小。

**WebTAG 原文**：TAG Unit A1.3 (DfT 2024) 给 commuting VOT distribution by income quintile：
- Q1 (lowest): £8.50/hour
- Q3 (median): £15.10/hour
- Q5 (highest): £28.40/hour

→ 把 VOT 转成 utility-scale per-minute disutility 时（base utility scale 设为 mid-tier baseline `β_mid = 0.07`），ratio 大致：

```
β_low / β_mid  ≈ (1 / 8.5) / (1 / 15.1) × scale_correction ≈ 1.7
β_high / β_mid ≈ (1 / 28.4) / (1 / 15.1) × scale_correction ≈ 0.6
```

得到 `[0.12, 0.07, 0.04]`。

### 3.3 β_mode 表（multiplicative penalty）

| Mode | β_mode | Source / 推导 |
|---|---|---|
| Car | **1.0** | Reference |
| PT | **1.2** | Wardman 2004：peak crowding penalty 增加感知 travel time ~20% |
| Active (walk/bike) | **1.5** | Wardman 2014：active mode 因 weather/effort 在 utility scale 上更重 |

**Why multiplicative not additive**：β_mode 表达"同样 1 分钟在不同 mode 上的 disutility 不同"，这是 multiplicative 的 natural form。Additive 形式（β_mode 当 fixed cost）会让"短程 active vs 短程 car"的边际效应失真。

### 3.4 联合 β table（9 个 cell）

| | Car | PT | Active |
|---|---|---|---|
| Low | 0.120 | 0.144 | 0.180 |
| Mid | 0.070 | 0.084 | 0.105 |
| High | 0.040 | 0.048 | 0.060 |

**Sanity check**：
- Low income × active = 0.18，最敏感 → 长程 walking 对 low-income agent 最 punishing → 跟 LTDS 观测的"low-income 集中在 short active commute"一致。
- High income × car = 0.04，最不敏感 → high-income agent 可以承受长程开车通勤 → 跟 outer-borough 高收入区高 mean commute time 一致。

### TODO + open questions

- [ ] β_income 的 1.7 / 0.6 ratio 用 NTS UK 2018-2023 income-stratified VOT 重新拟（目前是 webtag table 简化）
- [ ] β_mode 的 1.0 / 1.2 / 1.5 是否需要 split peak vs off-peak（peak PT 拥堵 penalty 应更大）
- [ ] Beijing 复现时这张表怎么 transfer：北京 VOT 文献（Yang 2017 等）的 income-stratified 数值可用，但 SOC↔GBM mapping 决定 tier 边界

---

## 4. Softplus enforcement of α_V, α_occ, γ

### 4.1 为什么需要

v2.1 的 `α_occ ≈ −0.05` failure mode 来自两个原因：

1. **数据 identifiability 弱**：OccMatch 跟 V_j(t) 在 spatial pattern 上高度相关（high-employment grid 多半也是 high-OccMatch grid）→ loss 对 α_occ 的 gradient 噪声大。
2. **没有先验约束**：如果训练初期 `α_occ` 偶然走到负值，loss 不会 strongly penalize（因为 α_V 可以补偿）。

→ 需要用 **prior 强制 α_occ > 0**。同样的逻辑用到 α_V（destination attractiveness 必须正向促进选择）和 γ（distance 必须负向，但 utility 公式里 `−γ·log d`，所以 γ 本身要 > 0）。

### 4.2 Softplus reparameterization

```
α_V    = softplus(α_V_raw)    = log(1 + exp(α_V_raw))
α_occ  = softplus(α_occ_raw)  = log(1 + exp(α_occ_raw))
γ      = softplus(γ_raw)      = log(1 + exp(γ_raw))
b      = b_raw                ← 不约束（bias 可正可负）
```

**Property**:
- `softplus: ℝ → ℝ⁺`，永远严格大于 0
- 无穷处近似 identity（`x >> 0` 时 `softplus(x) ≈ x`）
- 0 附近平滑（梯度永远存在，不像 ReLU 在 0 处梯度 ill-defined）

**Init**: 给目标 init 值 `v`，反算 raw：

```python
def _inv_softplus(v):
    return math.log(math.exp(v) - 1.0)

α_V_raw_init    = _inv_softplus(0.5)   ≈ -0.43
α_occ_raw_init  = _inv_softplus(0.7)   ≈ -0.10
γ_raw_init      = _inv_softplus(0.5)   ≈ -0.43
```

### 4.3 为什么不用 `exp(·)` 或 `(·)²`

| 方案 | Pro | Con |
|---|---|---|
| `softplus` | 平滑 + 在 0 附近 well-conditioned | 略多一次 exp 调用 |
| `exp(raw)` | 简单 | scale 不稳；raw drift 时 alpha 爆炸 |
| `(raw)²` | 简单 | raw=0 时梯度=0（dead point） + 双 valued (raw 和 -raw 同效果) |
| `clamp(raw, min=eps)` | 直观 | 在 boundary 处梯度=0 → optimizer stuck |

**结论**：softplus 在 PyTorch / NumPy 都是 stable kernel，identifiability 性质好，是标准选择。

### 4.4 跟 v2.1 对比的失效曲线

预期实验：用 `experiment_alpha_failure.py`（待写）跑 100 个 random seeds：

| Setting | α_occ < 0 的频率 | mean α_occ |
|---|---|---|
| v2.1 (unconstrained) | ~35% | +0.05（高方差） |
| v2.2 (softplus) | 0% (by construction) | +0.55（低方差） |

→ Softplus 把 failure mode 从概率事件变成结构上不可能。

### TODO + open questions

- [ ] 跑 ablation：unconstrained vs softplus，看 fit quality（CPC, KL）是否有 trade-off
- [ ] 是否要给 raw 参数加 weight decay（L2）防 raw 漂得太远（导致 softplus 进 saturated regime）

---

## 5. Training (Stage 2 update)

> **取代** `05_training_loss.md` §3 关于 β 的部分。本节描述 v2.2 的 Stage 2 NLL training。

### 5.1 Aggregate β per origin

由于训练 loss 是 aggregate 形式（按 origin × hour 的 OD distribution），β 需要从 per-agent 聚合到 per-origin：

```
β_origin(i) = Σ_(tier, mode) P(tier, mode | i) × β_income[tier] × β_mode[mode]
```

**计算方法**（见 `compute_beta_per_origin` in `heterogeneous_utility.py`）：

```python
for each agent in agents_df:
    i = agent.home_grid_idx
    tier = agent.income_tier
    mode = agent.mode_initial
    beta_origin[i] += beta_for(tier, mode)
    counts[i] += 1

beta_origin /= counts  # population-weighted average per origin
```

**Edge case**：grid `i` 没有 agent（synthetic OD 的 sparse origin）→ fallback 到 `beta_for(2, "car") = 0.07`（mid×car，neutral）。

### 5.2 Aggregate utility for training

```
V_ij^t = α_V · V_j^t
       + α_occ · OccMatch_ij^aggregate
       − β_origin(i) · t_ij^t
       − γ · log d_ij
       + b
```

注意：`OccMatch_ij^aggregate` 是 grid-level cosine（home grid 的 SOC 边际分布 vs work grid 的 industry mix），不是 per-agent。

### 5.3 Loss

跟 `05_training_loss.md` §2 完全一致——production-constrained MNL NLL：

```
L = − Σ_(i, t) Σ_j  [F_ij^t / Σ_k F_ik^t] · log_softmax(V_ij^t)_j
```

但**只有 4 个参数 learnable**：`α_V_raw, α_occ_raw, γ_raw, b`。比 v2.1 少 1 个（β）。

### 5.4 Optimizer

跟 v2.1 一致（`05_training_loss.md` §6）：
- Adam, lr=1e-3
- Cosine decay over 100 epochs
- Early stopping on val NLL, patience=10
- L2 on raw 参数 weight decay 1e-4

### 5.5 跟 Stage 1 / Stage 3 的接口

- **Stage 1 (encoder)**: 给 `V_j^t` (decoder bilinear)。Stage 2 只 fine-tune utility，encoder frozen。
- **Stage 3 (scalar calibration)**: BPR α/β、Gumbel scale、agent population scale 跟 utility 参数**正交**——它们改 t_ij^t 和噪声 scale，不改 utility coefficient。

### TODO + open questions

- [ ] β_origin(i) 应该是**初始** mode shares 的加权，还是 equilibrium 后 mode shares 的加权？ → 第一轮用初始（Census 2011 mode share），后续 outer iteration 重算
- [ ] 是否要 cache `β_origin` 在 disk（`agents_df` 不变情况下 deterministic）

---

## 6. Inference (ABM agents)

### 6.1 Per-agent utility

每个 agent `n` 在 Step 2 sample workplace 时：

```python
V_ij = utility.utility_personal(
    V_j_t        = h_j^t,                    # from Step 1 STGNN
    occ_match    = OccMatch(SOC_n, IndMix_j),  # per-agent! uses agent's specific SOC
    t_ij_t       = travel_time[n][j][t],     # mode-specific
    log_d_ij     = log_distance[i_n][j],
    income_tier  = agents_df.loc[n, 'income_tier'],
    mode_idx     = MODE_TO_IDX[agents_df.loc[n, 'mode']]
)
```

→ β 通过 `(income_tier, mode_idx)` 查表，不进 backprop。

### 6.2 OccMatch — per-agent vs aggregate

| | Training | Inference |
|---|---|---|
| OccMatch 形式 | Aggregate（home grid 的 SOC 边际 × IndMix_j） | Per-agent（agent_n 的 SOC × IndMix_j） |
| 原因 | 训练数据是 grid-level OD（没有 individual SOC label） | 每个 agent 有具体 SOC，应该用上 |
| 一致性 | 在期望意义上一致（aggregate = expectation of per-agent） | — |

### 6.3 Choice probability

```
P(j | n, t) = exp(V_nj^t) / Σ_k exp(V_nk^t)
```

每个 agent 的 choice set `C_n` 当前是 all grids（distance pruning 见 `00_v2_architecture.md` §3.8 open question）。

### 6.4 Mode 是否会变

当前设计：
- Mode 在 agent 生成时 sampled from Census 2011 mode shares，**后续不变**
- 即 β_(income, mode) 在 agent 整个生命周期固定

**Future v2.3** 考虑：mode choice 作 nested logit upper level，先 sample mode → 再 sample workplace。需要重新设计 β 的角色（不再是 fixed lookup，而是 nested utility 的 lower nest 参数）。

### TODO + open questions

- [ ] mode-switching dynamics（v2.3 nested logit）
- [ ] OccMatch 的 IndMix_j 是否要 hour-specific（业务 sector 时段差异）
- [ ] Choice set distance pruning（30 km cap 是否合理）

---

## 7. Identifiability discussion

### 7.1 v2.2 的 identifiability 改善

| 来源 | v2.1 | v2.2 |
|---|---|---|
| α_V × V_j scale 冗余 | β learnable → 冗余存在 | β fixed → α_V 唯一 controls scale |
| α_occ 符号 | 数据弱 → 实际负值 | softplus → 强制正 |
| 训练参数总数 | 5 (α_V, α_occ, β, γ, b) | 4 (α_V, α_occ, γ, b) |
| Empirical 收敛性 | seed-dependent，~20% 训练失败 | （expected）seed-stable |

### 7.2 Heterogeneity 是 prior，不是 data-identified

**重要 caveat**：β 的 (3 income × 3 mode) 异质性结构是**外生施加的 prior**，不是从训练数据 identify 出来的。

理由：
- Aggregate OD F_ij^t 不带 individual covariate
- 即使打开"每个 (tier, mode) cell 一个 free β"，identifiability 还是有问题（9 个 β 跟 V_j scale 集体冗余）
- 真正的 individual-level identification 需要 NTS / LTDS individual-level travel data，这是 v2.3 ethics-cleared 后的事情

→ v2.2 是**hybrid**：β 锚 literature（structural prior），α_V/α_occ/γ fit aggregate data。这是 transport modeling 的标准做法（Ben-Akiva 1985 §5.3）。

### 7.3 跟 mixed logit 的对比

| | v2.2 (fixed-β heterogeneity) | Mixed logit |
|---|---|---|
| Heterogeneity source | Discrete tier × mode lookup | Continuous distribution N(μ_β, σ_β²) |
| Identifiability | Imposed by literature prior | Identified from panel data |
| Compute cost | Cheap | Expensive (Monte Carlo integration) |
| Aggregate data 友好 | Yes | No (需要 panel) |

→ v2.2 是 aggregate-data setting 下的合理 compromise。

### TODO + open questions

- [ ] Counterfactual robustness：β 锁死 → 反事实情景下 β 是否需要重 calibrate？（e.g. 副中心扩容 → 新就业中心带来 income mix 变化）
- [ ] Sensitivity analysis：β table 各 cell ± 30% 对 main outcome (Gini, accessibility gap) 的影响

---

## 8. Why NOT include age / family

### 8.1 Age

**Theoretical case**：年轻 agent 可能更愿意 long commute（更多职业 search、less family constraint）；老年 agent 反之。

**Why not in v2.2**:
- VOT vs age 的关系**不是 monotone**（中年 agent peak income & peak family burden 同时存在）
- WebTAG / NTS UK 没有 stable age-stratified VOT table（CI overlap 大）
- 加 age 会变成 3 (income) × 3 (mode) × K (age bins) → 至少 27 cells，data sparsity 立即出现

**Future v2.3**：等 NTS individual-level data 跑出来再考虑。

### 8.2 Family / household structure

**Theoretical case**：有小孩的 agent 通勤范围会受 school / childcare constraint。

**Why not in v2.2**:
- Census 2021 microdata 在伦敦层面 individual-level family composition 不公开
- 需要 LSOA-level household type marginal + IPF 模拟 family composition → 整条 pipeline 没建
- v1 demo 和 v2 baseline 的目标是"income × mode × geography"三轴，先 lock 这个

**Future v2.4**：household-level agent generation（pair home worker），跟 schools / childcare provider 联合 simulate。

### TODO + open questions

- [ ] 评估"加 age 是否值得"的 prerequisite：先看 age effect 在 LTDS aggregate 上是否 detectable
- [ ] Family structure 的 modeling 复杂度估算（v2.4 scope）

---

## 9. TODO / Open questions

### 9.1 β 表 calibration

- [ ] **NTS individual-level VOT regression**（伦理通过后）
  - Target: 用 NTS UK 2018-2023 跑 `ln(commute_time) ~ income_quintile + mode + controls`
  - 把 implied β 跟 v2.2 table 对比
- [ ] **Cross-validation against external benchmarks**
  - WebTAG 2025 update（有的话）
  - DfT Mode Choice Model β values（如果可获）
  - Sensitivity ±30% on each cell

### 9.2 Mode choice as nested logit (v2.3)

- [ ] Upper nest: mode（car / PT / active）
- [ ] Lower nest: workplace given mode
- [ ] β 角色重新设计：lower nest 用 (mode-specific) β，upper nest 用 IIA-relaxed scale parameter
- [ ] Identifiability: nested logit with aggregate data—需要重新检查是否可学

### 9.3 Heterogeneity 扩展

- [ ] Spatial heterogeneity：β 是否随 location 变（中心 vs 外围 commuter 不同）
  - Pro: 行为合理
  - Con: 跟 V_j(t) confound
- [ ] Mode × time-of-day：peak PT 比 off-peak PT 应有更高 β_mode（拥堵）

### 9.4 Code / Engineering

- [ ] `tests/test_heterogeneous_utility.py`：单元测试 softplus/β lookup/aggregate matching
- [ ] `train.py` 集成：替换 `models_lib/utility.py` 旧 utility 调用
- [ ] `experiments/v2_2_softplus_vs_v2_1/`：A/B test，跑 5 seeds

---

## 10. Cross-references

### 10.1 内部文档

| 文档 | 关系 |
|---|---|
| `00_v2_architecture.md` | **Master**。本文档实现 §3 (Step 2) 的 utility 部分 |
| `05_training_loss.md` | **Older spec, partially superseded**：本文档 §5 取代其 §3（β handling）；其余 (Stage 1 reconstruction loss, F_ij^t 合成, spatial holdout) 仍 valid |
| `03_node_features.md` | h_j^t 来源（V_j^t = h_j^t 或 decoder 输出） |
| `04_edge_features.md` | log d_ij 计算 |
| `../../02_london_commuting_model/methodology/02_counterfactual_evaluation.md` | 反事实评估三层框架，β 异质性对 counterfactual robustness（Layer 3）有直接影响 |

### 10.2 与 `00_v2_architecture.md` 的潜在冲突

`00_v2_architecture.md` §3.4 当前的 utility 公式写的是：

```
V_ij^t = MLP([h_i^t ; h_j^t ; OccMatch_ij ; t_ij^t ; c_ij ; SES_i])
```

→ MLP-based black-box utility。

**v2.2 改用**：linear utility with softplus-positive coefficients + heterogeneous β。即把 MLP 替换成可解释 linear form。

**Reconcile action**：建议在 `00_v2_architecture.md` §3 加一个 note：

> v2.2 update: utility specification 从 NN-based MLP 简化为 linear with softplus-positive coefficients。详见 `06_heterogeneous_utility.md`。MLP 形式作为 v2.3 ablation 保留。

→ 待 v2.2 实验完成后回补 `00` 文档。

### 10.3 代码引用

- `models_lib/heterogeneous_utility.py` — 本文档对应实现
- `models_lib/agents.py` — agent generation（写入 `income_tier, mode_initial`）
- `train.py` Stage 2 — 调用 `utility_aggregate` 训练
- `model.py` Step 2 — 调用 `utility_personal` 推理

### 10.4 文献

- McFadden 1974 — Conditional logit (RUM closure)
- Ben-Akiva & Lerman 1985 — Discrete choice handbook §5.3 (literature-anchored prior in aggregate data setting)
- Wardman 2004 — VOT meta-analysis (PT crowding penalty)
- Wardman 2014 — Active mode VOT
- WebTAG TAG Unit A1.3 (UK DfT 2024) — VOT distribution by income
- Train 2009 — Discrete choice with simulation (mixed logit reference)

---

## 11. Changelog

| Version | Date | Change |
|---|---|---|
| v2.1 → v2.2 | 2026-05-04 | β 从 single learnable scalar 改为 (3 × 3) fixed lookup; α_V/α_occ/γ 加 softplus 约束 |
| v2.2 (this doc) | 2026-05-04 | Initial draft, locks decision table |
