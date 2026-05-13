> ⚠️ **OBSOLETE / SUPERSEDED — 部分内容需更新**
>
> 本文档写于 v2 早期，假设是**单 stage end-to-end** 的 GNN + RUM 联合训练（forward pass = STGNN → V_j(t) → softmax over j → cross-entropy on synthesized hourly OD）。
>
> 5-4 之后架构定型为 **4-step 架构 + Step 4 multi-stage calibration**（详见 `00_v2_architecture.md` §5）：
>
> 1. **Stage 1** — Encoder reconstruction loss（OD MSE，本文档 §2 描述的 NLL 不直接用）
> 2. **Stage 2** — Choice model NLL（encoder frozen，MNL on observed Census choices——更接近本文档 §2 但 V_ij^t 来自 utility NN 而不是直接 GNN 输出）
> 3. **Stage 3** — Scalar calibration（BPR α/β、Gumbel scale、agent population scale；grid search / Bayesian opt，**本文档完全没覆盖**）
>
> **本文档当前 valid 的部分**：
> - §3 β handling（仍适用于 Stage 2 / Stage 3）
> - §4 时间维度 + F_ij^t 合成（仍是 Stage 1 的 supervision source）
> - §5 spatial holdout 切分（适用全 pipeline）
> - §6 optimizer details（适用 Stage 1 + 2）
> - §7 evaluation metric（适用全 pipeline）
>
> **需要更新的部分**：
> - §1 决策汇总表 — "loss 类型" 项需说明是分 stage 的，不是单一 NLL
> - §2 Loss specification — 需拆成 Stage 1 reconstruction loss + Stage 2 choice loss 两段
> - 新增 Stage 3 scalar calibration 章节（grid search、multi-objective、weight tuning）
>
> **TODO**: 重写本文档为 `05_stage1_2_loss.md`（reconstruction + choice）+ `06_stage3_calibration.md`（scalar calibration），或 inline 更新本文档分章节描述 3 个 stage。
>
> **请先读 `00_v2_architecture.md` §5** 拿到 Step 4 multi-stage calibration 的最新设计，本文档作为 Stage 1/2 loss 的 detailed reference。

---

# Training Loss & Optimization（v2）

> **目的**：定义 v2 STGNN + RUM 联合训练的 loss 函数、优化器、train/val/test 切分。
>
> **关系**：本文档**取代** `../../02_london_commuting_model/methodology/01_step1_gnn_decisions.md` 里 Decision 1 / 5 / 6 / 7 中跟 loss 和训练相关的部分。原决策基于 4-30 的 bilinear decoder + GNN-RUM 分离架构，5-4 改为 GNN 直接输出 V_j(t) + RUM closure 后已 obsolete。
>
> **配套**：`00_v2_architecture.md`（架构总览，read first）、`03_node_features.md`、`04_edge_features.md`、`02_counterfactual_evaluation.md`。

---

## 1. 决策汇总（locked）

| 项目 | 决定 | 理由 |
|---|---|---|
| Loss 类型 | Production-constrained MNL log-likelihood | RUM closure 自然形式 |
| β（成本敏感度） | Fixed from WebTAG VOT（baseline）+ learned（ablation） | Identifiability 保证 + 数据驱动验证 |
| 时间维度 | Per-hour P(j\|i,t) 训练 | STGNN 时间层必须有 hourly signal |
| F_ij^t 来源 | 合成（Census 2021 × LTDS/NTS 时间分布） | 真 hourly OD 不可得 |
| Spatial split | 75 / 10 / 15 (train/val/test) | 反事实预测核心 challenge 是 spatial generalization |
| Temporal split | 无（只有 2021 一年） | 多年 OD 数据不可得 |
| Negative sampling | 不用，全 softmax | N ≈ 900-1500，全 softmax 计算可行 |
| Optimizer | Adam, lr=1e-3, cosine decay | 标准选择 |
| Regularization | L2 on GNN weights | 防过拟合，最简单 baseline |

---

## 2. Loss specification

### 2.1 Forward pass per (origin i, hour t)

```
1. STGNN forward:
   V(t) = STGNN(x_seq, edge_index, edge_attr)  → tensor shape (N,)，每个 grid 的 V_j(t)

2. Travel cost from provider:
   t_costs = TravelTimeProvider.get(i, all_j, t, mode='generalized')  → (N,)

3. RUM logits:
   logits(j) = V_j(t) − β · t_costs(j)

4. Log probabilities:
   log_probs = log_softmax(logits)  → (N,) ，Σ_j P(j|i,t) = 1

5. Target distribution (synthesized hourly OD):
   target(j) = F_ij^t / Σ_k F_ik^t  ← 归一化
   
6. NLL:
   NLL_{i,t} = − Σ_j target(j) · log_probs(j)
```

### 2.2 Total loss (over batch)

```
L_total = Σ_{(i,t) ∈ batch} N_i^t · NLL_{i,t}  +  λ · ||θ_GNN||²

其中：
  N_i^t = Σ_j F_ij^t  ← origin i 在 hour t 的总通勤量（loss weight）
  λ ≈ 1e-4（L2 强度）
```

**Weights N_i^t 的作用**：让"流量大的 origin × hour"对 loss 贡献更大，等价于在所有 trip-level data 上做 NLL（不是在 (i,t) pair 级 NLL）。

### 2.3 等价于 trip-level MNL maximum likelihood

数学上：

```
L_total = − Σ_{trips} log P(j_trip | i_trip, t_trip)
```

这就是 McFadden 1974 经典 MNL 估计——v2 用 GNN 参数化 V_j(t)，但训练 objective 跟传统 logit 模型完全一致。

### 2.4 为什么 production-constrained 而不是 doubly-constrained

| | Production-constrained | Doubly-constrained |
|---|---|---|
| 强制 | Σ_j P(j\|i,t) = 1（softmax 自带） | 同 + Σ_i F_ij = D_j |
| 训练 | 直接 cross-entropy | 需要 Sinkhorn iteration |
| 反事实操作 | D_j 自由变化 → **副中心扩容场景友好** | D_j 被锁死 → 反事实不灵活 |

→ **副中心扩容场景**（D_j 增加 50 万）要求 destination marginal 是 scenario variable，不能是训练硬约束。

### 2.5 为什么 V_j 里**不**加 distance term

01 Decision 6 提到过 V_j 里加 γ·log d_ij。在 v2 不加，理由：

- V_j(t) 应该是**纯 destination attractiveness**，跟 origin i 无关
- 距离信息已经通过两个渠道进入：
  1. Edge features（Gaussian decay weight + free-flow time）→ 影响 GNN message passing
  2. RUM 的 β·t_ij(t)（generalized travel cost）
- 加进 V_j 会让 V_j(t,i) 变成 origin-dependent，破坏 factorization

---

## 3. β（成本敏感度）handling

### 3.1 Baseline：固定值（WebTAG VOT 转换）

WebTAG TAG Unit A1.3 (2024) 通勤时间价值约 **£15/hour**。在 RUM utility 单位下：

```
generalized_cost = monetary_cost + VOT × time
                 = 0 + £15/hr × t/60     （假设无显式 monetary cost）
                 = £0.25 × t (minutes)

如果 utility 单位是"等价 £ "：
  U_ij(t) = V_j − 0.25 × t_ij(t)
  → β = 0.25

如果 utility 单位是"OD trip log-odds"，需要根据样本量级 scale。
约定 baseline β = 0.07 per minute（中等 commute scale）。
```

### 3.2 Identifiability 担忧

如果 V_j 和 β 都 learnable：
- V_j 整体 scale × c → β scale × 1/c → 同 logits → 同 loss
- 训练会陷入"V_j 任意大、β 任意小"或反过来的退化解

**解决**：固定 β，让 GNN 学的 V_j 跟 β 的 unit 保持一致。

### 3.3 Ablation：learned β

主实验跑完，对比 ablation：
1. 初始化 β = 0.07
2. 给 β 一个独立 lr（1e-4，比 GNN 慢一个量级）
3. 训练完看 β_learned 跟 β_fixed 偏差
4. 解读：
   - |β_learned − β_fixed| / β_fixed < 20% → 固定 β 合理
   - 偏差大 → 模型 specification 有问题或 identifiability 没解决

### 3.4 Heterogeneous β（v2.1+，不在 baseline）

5-2 memory 提到 A1（income-stratified β = f(income)）。这是 v2.1 扩展，不在 baseline loss 设计中。届时：
- agent 按 income tier 分组
- 每组独立的 β
- Loss 改为 sum over groups

---

## 4. 时间维度 in training

### 4.1 Per-hour 训练样本

Training sample = (origin i, hour t, F_ij^t distribution over j)

24 hours × N grids ≈ 24 × 1500 ≈ 36,000 samples / "epoch-equivalent"

### 4.2 F_ij^t 合成（详见 01 doc §合成温度 OD 方案）

```
F_ij^t = F_ij × P(mode | i, j) × π(t | mode, OD)

伦敦：
  F_ij ← Census 2021 WU03UK
  P(mode | i, j) ← Census 2011 mode shares (with fallback to London-mean)
  π(t | mode, OD) ← LTDS / NTS departure-time histogram

北京（伦理通过后）：
  F_ij ← Beijing 2021 OD
  P(mode | ...) ← 北京交通发展年报 mode shares
  π(t | ...) ← 北京通勤调查 departure-time
```

### 4.3 Limitation 明示

**F_ij^t 是合成的，不是观测**。STGNN 时间层学到的是**合成的时间模式**，不是真 hourly dynamics。论文 framing 必须 honest：
- ✅ "我们用 STGNN 学习 hourly OD 分布的 location-specific 模式"
- ❌ "我们的 STGNN 捕捉真实 hourly 通勤动态"

→ Aggregate validation：用 TomTom 整体 hourly traffic counts 跟 STGNN 预测的 hourly aggregate flow 对照，至少看 aggregate 层面合理。

### 4.4 是否在 loss 里 reweight peak vs off-peak

考虑过：peak hours 合成更可靠（mass 集中、 NTS sample size 充足），off-peak 合成噪声大 → 给 off-peak 较低权重。

**当前决策**：**不 reweight**（保持 KISS）。Ablation 项可以试。

---

## 5. Train / Val / Test split

### 5.1 Spatial holdout（核心）

| Set | 比例 | 大小（伦敦 ~1500 grids） |
|---|---|---|
| Train | 75% | ~1125 grids |
| Val | 10% | ~150 grids |
| Test | 15% | ~225 grids |

**抽样方式**：分层随机（按 borough/district 分层），保证 train/val/test 都覆盖城市梯度（中心 + 外围）。

**禁忌**：训练时 test grid 的 node features 和 edges **完全不可见**，包括邻居关系——必须用 induced subgraph + masked aggregation。PyG 的 `RandomNodeSplit` 或 `subgraph` 工具可处理。

### 5.2 Temporal split

**无**——只有 2021 一年数据。

如果未来有多年（v2 可扩展）：
- 2021 训练
- 2022 / 2023 val / test
- 这种 setup 更接近反事实评估场景（OOD time）

### 5.3 为什么 spatial only

- 反事实预测的核心 OOD 维度是 **spatial**（新就业中心 = 没见过的 grid 配置）
- Spatial holdout 直接 stress-test 模型的 spatial generalization
- Yao 2021 / Simini 2021 / 大多数 OD 模型都用 spatial holdout

---

## 6. Optimization details

### 6.1 Hyperparameters (defaults)

| Param | Default | Range（消融） |
|---|---|---|
| GNN hidden dim | 128 | {64, 128, 256} |
| GAT heads | 4 | {1, 2, 4, 8} |
| GRU hidden dim | 64 | {32, 64, 128} |
| K (kNN) | 10 | {5, 10, 15} |
| Learning rate | 1e-3 | {1e-4, 3e-4, 1e-3, 3e-3} |
| L2 weight λ | 1e-4 | {1e-5, 1e-4, 1e-3} |
| Batch size (i,t pairs) | 32 | {16, 32, 64} |
| Epochs | 100 (with early stop) | — |

### 6.2 Optimizer schedule

- Adam, β₁=0.9, β₂=0.999
- LR warmup：5 epochs linear from 0 → 1e-3
- LR decay：cosine over remaining 95 epochs
- Early stopping：val NLL，patience=10

### 6.3 Random seed

固定 3 个 seeds（42, 123, 2024）跑同一配置 → 报告 mean ± std 反映训练稳定性。

---

## 7. Training metrics（per epoch logging）

| Metric | 公式 / 解释 | 用途 |
|---|---|---|
| Train NLL / trip | 每 trip 平均 NLL | 收敛检查 |
| Val NLL / trip | 同上但 val set | Early stopping & model selection |
| **CPC** | Common Part of Commuters = Σ_{ij} min(F_pred, F_obs) / Σ F_obs | OD 预测金标准 metric (Lenormand 2016) |
| **Spearman ρ** | rank correlation on flow | rank-based, 容噪 |
| **KL divergence** | KL(P_pred(j\|i,t) \|\| P_obs(j\|i,t)) | distribution divergence |
| Mean β（learned ablation） | 训练中 β 轨迹 | sanity check |

### 7.1 Baseline comparison metrics

跟 gravity baseline 同时报告，看 GNN improvement：

| Model | CPC | Spearman ρ | KL |
|---|---|---|---|
| Gravity (Wilson 1971) | x | x | x |
| Radiation (Simini 2012) | x | x | x |
| MLP (no graph structure) | x | x | x |
| **v2 STGNN + RUM** | x | x | x |

---

## 8. Ablation experiments（按重要性排序）

### 8.1 必跑（论文 main table）
- [ ] Fixed β vs learned β
- [ ] With / without GRU 时间层（替换为 mean over time）
- [ ] With / without Gaussian decay edge feature
- [ ] K = 5 / 10 / 15

### 8.2 选做（supplementary）
- [ ] Hidden dim 64 / 128 / 256
- [ ] GAT heads 1 / 4 / 8
- [ ] Off-peak loss reweight on / off
- [ ] Negative sampling vs full softmax（伦敦 N=1500 看 epoch time 是否需要）

---

## 9. Implementation TODO

### Phase 1（伦敦试点）
- [ ] `models_lib/loss.py`：production_constrained_mnl_nll(...)
- [ ] `models_lib/stgnn.py`：T-GCN forward (GNN spatial + GRU temporal)
- [ ] `train.py`：spatial holdout split + training loop
- [ ] Synthesize F_ij^t pipeline（在 v1 已部分有，迁移到 v2）
- [ ] Logging：wandb 或 tensorboard
- [ ] Sanity check：在小 N (e.g. 50 grids) 上 overfit，确认 loss 在合成 data 上能推到 0

### Phase 2（北京）
- [ ] BeijingSyntheticOD pipeline
- [ ] β 初始值是否需要 calibrate（北京 VOT 不同）
- [ ] 重训完整模型

---

## 10. Open questions（待确认 / 未来 sprint）

- [ ] β 单位是否要在 utility scale 上 calibrate（避免 V_j 和 β 单位错位）
- [ ] 是否需要 reconstruction auxiliary loss（除 RUM NLL 外，加一个 V_j 重构 employment 的 aux loss）
- [ ] 是否在 loss 里 explicit penalize V_j 的时间 smoothness（||V_j(t) − V_j(t-1)||²）
- [ ] Off-peak hours 的 loss 权重 scheme
- [ ] 是否在 inference 阶段用 Sinkhorn 强制 doubly-constrained（避免预测的总通勤量违反就业 capacity）

---

## 11. 跟 Yao 2021 的 loss 对比

| | Yao 2021 SI-GCN | v2 |
|---|---|---|
| Loss | MSE on log(1+flow) | Production-constrained MNL log-likelihood |
| 概率解释 | 无 | 有（直接对应 RUM） |
| Negative sampling | 用 | 不用 |
| 与 RUM 兼容 | 否 | 是（自然 RUM closure） |
| Identifiability | scale 不约束 | β fixed → 有约束 |

**Contribution edge**：v2 把 SI-GCN 的回归形式升级成 discrete choice MLE，**直接接驳 50 年的行为决策文献**（McFadden, Ben-Akiva, Train），同时**让反事实情景预测有理论基础**（RUM 是结构因果模型，MSE 不是）。

→ 这是 v2 跟 Yao 2021 在方法学层面的**实质区分**之一，跟 5-2 memory 中"behavioral framework embedded" contribution edge #2 直接对应。

---

## 12. 参考

- McFadden 1974 — 多项 logit 模型
- Ben-Akiva & Lerman 1985 — discrete choice handbook
- Train 2009 — discrete choice with simulation
- Wilson 1971 — gravity model（baseline）
- Simini 2012 *Nature* — radiation model（baseline）
- Lenormand 2016 — CPC metric
- WebTAG TAG Unit A1.3 (UK DfT, 2024) — VOT 标准值
- Yao 2021 — SI-GCN（loss 对比）
- `01_step1_gnn_decisions.md`（v1，部分 obsolete）
- `02_counterfactual_evaluation.md`、`03_node_features.md`、`04_edge_features.md`
