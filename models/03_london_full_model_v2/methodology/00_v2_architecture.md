# v2 Master Architecture — 4-Step Pipeline

> **目的**：定义 v2 London Commuting ABM 的整体 4-step 架构，作为所有其他 methodology 文档的"地图"。先看本文档，再看 03/04/05 这些 step-specific 详细文档。
>
> **关系**：本文档是 v2 的**架构总览**。03/04/05 是**Step 1 内部**的细节（节点特征、边特征、训练 loss），本文档定义它们在整体里的位置 + 描述 Step 2/3/4 的设计。
>
> **配套文档**：
> - `03_node_features.md` — Step 1 GNN 节点特征 inventory
> - `04_edge_features.md` — Step 1 GNN 图拓扑 + 边特征
> - `05_training_loss.md` — Step 1 forward pass loss（**部分 obsolete**：Step 4 多阶段 calibration 还没补全）
> - `../../02_london_commuting_model/methodology/02_counterfactual_evaluation.md` — 反事实评估三层证据链（**仍然适用**，与具体模型架构无关）

---

## 0. TL;DR

整个 v2 模型 = **Encoder（ML）+ Workplace Choice（ABM）+ Congestion（ABM）+ Calibration**，分 4 个 step 串起来：

```
[Step 1] STGNN Encoder      →  embeddings h_i^t, h_j^t
[Step 2] Workplace Choice   →  agent (home, work*, mode, departure)
[Step 3] Congestion ABM     →  t_ij^t, A_i^t, Gini/Palma
[Step 4] Calibration        →  fit θ against observed stats
```

**两个 feedback loop**：
- 红色：Step 3 算出的拥堵时间反喂回 Step 2，agent 重新决策直到均衡（congestion equilibrium）
- 紫色：Step 4 calibration 全局更新所有 component 参数

**关键设计原则**：
1. ML 只在 Step 1 用（学 embedding）+ Step 2 用（utility NN），其他都是行为/物理 ABM
2. 时间维度贯穿全 pipeline（hourly slices, t = 6am … 9pm）
3. Beijing-friendly：每个组件都能在伦敦 + 北京两边复现

---

## 1. 架构总图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA INPUTS                               │
│  Spatial network · Location attrs · OD flows · Mode/POI/Subway      │
│       (详见 03_node_features.md, 04_edge_features.md)               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — STGNN ENCODER (ML)                                        │
│                                                                     │
│   Input embedding ──► GraphSAGE × 2 ──► GRU (temporal)              │
│                                                                     │
│   Outputs:                                                          │
│     • h_i^t, h_j^t  ∈ ℝ^64    (node embeddings, fed to Step 2)     │
│     • Ŝ_ij^t = h_i^t M h_j^t − γ log d_ij  (aux OD reconstruction) │
└─────────────────────────────────────────────────────────────────────┘
                                │  h_i^t, h_j^t
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — WORKPLACE CHOICE (ABM)                                    │
│                                                                     │
│   Generate N agents per home grid                                   │
│      ├─ home_i, SOC, SES (income tier, age), initial mode           │
│      ▼                                                              │
│   Compute OccMatch_ij = cos(SOC_i, industry_mix_j)                  │
│      ▼                                                              │
│   Utility NN:                                                       │
│      V_ij^t = NN_θ([h_i^t ; h_j^t ; OccMatch_ij ; t_ij^t ; …])     │
│      ▼                                                              │
│   Workplace selection (softmax / MNL):                              │
│      P(j | i, t) = exp(V_ij^t) / Σ_k exp(V_ik^t)                   │
│      ▼                                                              │
│   Each agent samples j* ~ P(· | i, t)                              │
│                                                                     │
│   Outputs: Commuter agents with (home, work*, mode, departure)      │
└─────────────────────────────────────────────────────────────────────┘
                                │   agent flows
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — CONGESTION ABM                                            │
│                                                                     │
│   Mode-specific path & cost:                                        │
│      Car   → BPR(volume, capacity)                                  │
│      PT    → free-flow + crowding penalty                           │
│      Active→ free-flow                                              │
│                                                                     │
│   Aggregate flows:  V_e^t = Σ_n 𝟙[agent n on edge e at t]          │
│   Update t_ij^t = t_0 (1 + α (V_e^t · m_h / C_eff)^β)              │
│                                                                     │
│   Outputs: t_ij^t, A_i^t = Σ_j E_j exp(−β t_ij^t),                 │
│            Gini, Palma, SOC accessibility gap                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — CALIBRATION (Multi-stage)                                 │
│                                                                     │
│   Stage 1 — Encoder training (OD reconstruction, MSE)               │
│   Stage 2 — Choice model training (MNL, encoder frozen)             │
│   Stage 3 — Scalar calibration (BPR α/β, Gumbel scale, agent scale) │
│                                                                     │
│   θ = {GraphSAGE, GRU, NN_θ, decoder M, BPR α/β, Gumbel scale}      │
└─────────────────────────────────────────────────────────────────────┘

   Feedback loops:
     ── (red)  Step 3 t_ij^t  ─►  Step 2 (re-decision until equilibrium)
     ── (purple) Step 4 ─►  Step 1 / 2 / 3 全局更新
```

---

## 2. Step 1 — STGNN Encoder

> **任务**：从地理 + 时间特征学 location embedding，让下游 utility 函数能用上"网络结构 + 时间动态"信息。

### 2.1 Inputs

| Input | 来源 | 详细文档 |
|---|---|---|
| Node features (静态 + 时变) | 21 + 4 维 / grid / hour | `03_node_features.md` |
| Edge index + edge features | kNN(K=10), 7 维 | `04_edge_features.md` |
| Auxiliary supervision | 合成的 hourly OD F_ij^t | `05_training_loss.md` §4.2 |

### 2.2 Processing

```
1. Input embedding:
   x_i^t = StandardScaler(raw_features_i^t)  ∈ ℝ^25

2. Spatial encoder (GraphSAGE × 2, hidden = 64):
   h_i^(l+1) = σ(W · MEAN({h_j^(l) : j ∈ N(i)}))

3. Temporal encoder (GRU × 1, hidden = 64):
   h_i^t = GRU(h_i^(l=2,t-1), spatial_encoded_i^t)
```

### 2.3 Outputs

| Output | Shape | 用途 |
|---|---|---|
| h_i^t (node embedding) | (T, N, 64) | 喂给 Step 2 utility NN |
| Ŝ_ij^t = h_i^t M h_j^t − γ log d_ij | edge-level | Step 4 Stage 1 OD reconstruction loss |

### 2.4 Key equations

```
GraphSAGE message passing:
  h_i^(l+1) = ReLU( W^(l) · CONCAT[ h_i^(l), MEAN_{j∈N(i)} h_j^(l) ] )

GRU update:
  z_t = σ(W_z · [h_{t-1}, x_t])         ← update gate
  r_t = σ(W_r · [h_{t-1}, x_t])         ← reset gate
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
  h_t = z_t ⊙ h_{t-1} + (1 − z_t) ⊙ h̃_t

Bilinear decoder (auxiliary):
  Ŝ_ij^t = (h_i^t)^⊤ M h_j^t − γ log d_ij
```

### 2.5 Why this design

- **GraphSAGE 而不是 GCN**：sample-based aggregation，scale 到大图（伦敦 ~1500 grid + 北京 ~900 grid）更稳；适合 inductive setting（test grid 没见过）
- **GRU 而不是 attention**：通勤 hourly 模式比较 stationary（早高峰、晚高峰 pattern 重复），GRU 足够；attention 增加复杂度但 marginal gain 不明显
- **Bilinear decoder 作 auxiliary loss**：保证 embedding 不退化成 trivial solution，同时 distance term `γ log d_ij` 给强 prior

### 2.6 Implementation TODO

- [ ] `models_lib/stgnn.py`：GraphSAGE × 2 + GRU × 1
- [ ] `models_lib/decoder.py`：bilinear decoder + γ scalar
- [ ] 单元测试：在 50 grid toy 数据上 overfit，确认 reconstruction 能拟合

### 2.7 Open questions

- [ ] GraphSAGE 用 mean / max / LSTM 哪个 aggregator
- [ ] 是否要 attention 替换 GRU（v2.1 ablation）
- [ ] hidden dim 32 / 64 / 128 sweep

---

## 3. Step 2 — Workplace Choice

> **任务**：每个 agent 用 RUM 框架选 workplace，把 ML embedding 翻译成行为决策。

### 3.1 Inputs

| Input | 来源 |
|---|---|
| h_i^t, h_j^t | Step 1 输出 |
| Agent attributes (SOC, SES, mode, age) | Census 2021 + LTDS 联合采样 |
| Industry mix per grid j | `03_node_features.md` §3.2 (8 类 sector) |
| t_ij^t (travel time per OD × hour) | Step 3（first iteration: free-flow） |
| c_ij (monetary cost) | Mode-dependent fare table |

### 3.2 Processing

```
1. Agent generation:
   For each home grid i:
     N_i = round(population_i × agent_scale)
     For each n in 1..N_i:
       sample SOC_n ~ Census 2021 occupation marginal at i
       sample SES_n ~ LTDS conditional on SOC_n
       sample initial_mode_n ~ Census 2011 mode shares at i

2. Occupation match:
   OccMatch_ij = cosine(SOC_affinity_n, industry_mix_j)
   where SOC_affinity_n is a learned/lookup vector mapping SOC → industry weight

3. Utility evaluation (2-layer MLP, hidden = 64, ReLU):
   V_ij^t = NN_θ([ h_i^t ; h_j^t ; OccMatch_ij ; t_ij^t ; c_ij ; SES_n ])

4. Workplace selection (multinomial logit):
   P(j | i, n, t) = exp(V_ij^t) / Σ_k∈C_i exp(V_ik^t)
   sample j* ~ P(· | i, n, t)
```

### 3.3 Outputs

每个 agent 有一个 tuple：

```
agent_n = {
  home: i,
  work*: j*,
  mode: m,
  departure_time: t_dep
}
```

### 3.4 Key equations

```
Cosine occupation match:
  OccMatch_ij = (SOC_i · IndMix_j) / (||SOC_i|| · ||IndMix_j||)

Utility:
  V_ij^t = MLP([h_i^t ; h_j^t ; OccMatch_ij ; t_ij^t ; c_ij ; SES_i])

Choice probability:
  P(j | i, t) = exp(V_ij^t) / Σ_k exp(V_ik^t)

  ← McFadden 1974 conditional logit; provides RUM closure
```

### 3.5 Why this design

- **NN 的 utility 而不是 linear utility**：捕捉 SES × travel time 这种交互（low-income agent 对长通勤更敏感）；linear utility 假设太强
- **SOC × industry mix 而不是 raw occupation**：让 occupation 信息进 utility 但不爆维度；同时 Beijing-friendly（SOC↔GBM mapping 时只需对齐 8 类 industry）
- **从 agent-level sample 而不是 distribution-level fit**：给 Step 3 的 emergent congestion 留接口（每个 agent 有独立 path）

### 3.6 Trade-offs

| Choice | Pro | Con |
|---|---|---|
| Linear utility (传统 logit) | Identifiability 强，参数可解释 | 难捕捉交互项 |
| **NN utility (v2 选)** | 捕捉非线性 + ML embedding 自然进入 | 解释性低，需要更多正则 |
| Mixed logit | 异质性 explicit | 训练慢 + identifiability 复杂 |

### 3.7 Implementation TODO

- [ ] `models_lib/agents.py`：CommuterAgent + 采样 pipeline
- [ ] `models_lib/utility.py`：utility NN
- [ ] `providers/soc_industry.py`：SOC affinity vector + industry mix
- [ ] 单元测试：N=100 agents 在 toy 5×5 grid 上能正确 sample

### 3.8 Open questions

- [ ] SOC affinity 是 lookup table（专家给）还是 learnable embedding
- [ ] 是否要 nested logit（先 sector → 再 specific grid）
- [ ] Choice set C_i 是 all grids 还是 distance-pruned (e.g. d_ij < 30 km)

---

## 4. Step 3 — Emergent Congestion ABM

> **任务**：把每个 agent 的选择聚合成 grid-level flow，用 BPR 算拥堵时间，反喂回 Step 2 直到均衡。

### 4.1 Inputs

| Input | 来源 |
|---|---|
| Agent (home, work, mode, departure) | Step 2 输出 |
| Road / PT network | OSM + TfL routes（伦敦）/ OSM + 高德（北京） |
| Free-flow time t_0 per edge | `04_edge_features.md` §4.2 |
| Capacity C_eff per edge | OSM lane count + 工程 default |

### 4.2 Processing

```
1. Mode-specific shortest path:
   For each agent_n with (home_i, work_j, mode_m):
     path_n = shortest_path(network_m, i, j)

2. Segment flow aggregation per hour:
   V_e^t = Σ_n 𝟙[agent n on segment e at hour t]

3. Cost update:
   Car:    t_e^t = t_0_e × (1 + α × (V_e^t × m_h / C_eff_e)^β)
   PT:     t_e^t = t_0_e × (1 + crowd_penalty(V_e^t / C_PT_e))
   Active: t_e^t = t_0_e

4. OD-pair travel time:
   t_ij^t = Σ_{e ∈ path_ij} t_e^t

5. Equilibrium loop:
   Iterate Step 2 ↔ Step 3 until ‖t_ij^t(k) − t_ij^t(k−1)‖ < ε
   (typically 5-10 iterations; Method of Successive Averages)

6. Final outputs:
   Accessibility:  A_i^t = Σ_j E_j × exp(−β × t_ij^t)
   Gini, Palma, SOC accessibility gap on {A_i^t}
```

### 4.3 Outputs

| Output | Shape | 用途 |
|---|---|---|
| t_ij^t (congested travel time) | (T, OD pairs) | 反馈 Step 2; 评估 metric |
| A_i^t (accessibility) | (T, N) | 不平等评估 |
| Gini / Palma / SOC gap | scalar per t | 论文 main outcome |

### 4.4 Key equations

```
BPR (Bureau of Public Roads, 1964):
  t_e = t_0 × (1 + α (V/C)^β)
  典型 α = 0.15, β = 4

PT crowding penalty (Wardman 2004 风格):
  crowd_penalty(load) = max(0, k × (load − threshold))
  load = V_PT / seat_capacity

Accessibility (gravity / Hansen 1959):
  A_i^t = Σ_j E_j × exp(−β_acc × t_ij^t)

Method of Successive Averages (Sheffi 1985):
  t_ij^(k+1) = (1 − 1/k) × t_ij^(k) + (1/k) × t_ij^new
  保证收敛到 user equilibrium
```

### 4.5 Why this design

- **BPR 而不是 microscopic traffic sim**：训练 / inference 速度差几个数量级；BPR 是 transport planning 标准 functional form，有 50 年文献支持
- **Equilibrium loop 而不是单次 forward**：通勤决策 + 拥堵是相互影响的——不迭代会 underestimate 长程通勤的实际不方便
- **MSA averaging**：保证 fixed-point 收敛，避免 oscillation

### 4.6 Implementation TODO

- [ ] `models_lib/congestion.py`：BPR + crowding + MSA loop
- [ ] `providers/network.py`：road / PT graph loader
- [ ] `models_lib/accessibility.py`：A_i^t + Gini/Palma
- [ ] 收敛性单元测试：在小 N 上 MSA 应在 < 20 iterations 达到 ε=1e-3

### 4.7 Open questions

- [ ] Equilibrium 用 user equilibrium (Wardrop 1) 还是 stochastic UE
- [ ] PT crowding 的 threshold 怎么校 (从 LTDS observation?)
- [ ] BPR α/β 是 city-wide 还是 borough-specific

---

## 5. Step 4 — Multi-stage Calibration

> **任务**：把整个 4-step pipeline 的所有可学/可调参数 fit 到观测数据。**关键设计**：分 3 stage 训练，避免 joint optimization 的 identifiability 灾难。

### 5.1 Why multi-stage（不是 end-to-end）

End-to-end 联合训练 GNN + utility NN + BPR scalar 会遇到：

1. **Identifiability**：BPR β 和 utility β 都控制 travel-time sensitivity，joint train 互相吞掉 signal
2. **Loss scale 不平衡**：MSE on flow vs. NLL on choice vs. RMSE on travel time——三个 loss 量级不同
3. **Convergence 困难**：拥堵 equilibrium loop 嵌进梯度里，反向传播会 explode

→ Multi-stage 把每个 stage 独立 fit，每 stage 有清晰的 supervision signal。

### 5.2 Stage 1 — Encoder training

**Goal**：让 STGNN learn 合理的 location embedding。

**Loss**:
```
L_recon = Σ_{(i,j,t)} [ f_ij^t − Ŝ_ij^t ]²

其中：
  f_ij^t = 合成的 hourly OD flow（详见 05_training_loss.md §4.2）
  Ŝ_ij^t = (h_i^t)^⊤ M h_j^t − γ log d_ij
```

**Supervision**:
- 合成 hourly OD：`Census 2021 OD × LTDS departure-time × ATC hourly calibration`
- 详见 `05_training_loss.md` §4.2

**Spatial holdout**:
- 4 boroughs reserved as test set（详见 `05_training_loss.md` §5）
- 75% train / 10% val / 15% test, 分层按 borough 抽样

**Output**: trained GraphSAGE + GRU + decoder weights (frozen 进入 Stage 2)

### 5.3 Stage 2 — Choice model training

**Goal**：让 utility NN 学 workplace choice 行为。

**Loss**:
```
L_choice = − Σ_n log P(j_n* | i_n, t_n; θ_NN)

其中：
  j_n* = observed workplace from Census 2021 individual-level data
  P(j | i, t) = exp(V_ij^t) / Σ_k exp(V_ik^t)
  V_ij^t = NN_θ([h_i^t (frozen) ; h_j^t (frozen) ; OccMatch_ij ; t_ij^t ; c_ij ; SES])
```

**Encoder frozen**：Stage 1 的 GraphSAGE + GRU 不再更新，只 fine-tune utility NN。
- 避免 utility loss 把 embedding 拉偏
- Identifiability 简化：embedding 是 fixed feature，utility NN 是唯一可学

**Travel time t_ij^t**：用 Stage 3 calibrated BPR 算的 baseline congestion time（first iteration: free-flow）

**Output**: trained NN_θ utility weights

### 5.4 Stage 3 — Scalar calibration

**Goal**: 校 BPR α/β、Gumbel scale、agent population scale 这种 "scalar knob"。

**Method**: Bayesian optimization or grid search on:

| Param | Range | Target metric |
|---|---|---|
| BPR α | [0.10, 0.30] | JTS PT travel time RMSE |
| BPR β | [3, 6] | TomTom hourly congestion 对比 |
| Gumbel scale (RUM noise) | [0.5, 2.0] | Census mode share KL |
| Agent population scale | [0.05, 0.20] | Total trip count vs. observed |

**Loss** (multi-objective, weighted sum):
```
L_scalar = w1 × RMSE(t_ij^pred, t_ij^JTS)
         + w2 × KL(mode_pred || mode_Census)
         + w3 × |trip_count_pred − trip_count_obs| / trip_count_obs
```

**Search method**:
- Grid search 4D（每个 axis 5-7 个值，total 600-2400 evaluations）
- 每次 evaluation 跑一遍 Step 2 + Step 3 equilibrium
- 算 cost 上 GPU + cache embedding (Stage 1 已 frozen)

**Output**: optimal {α, β, Gumbel σ, agent scale}

### 5.5 Why this 3-stage decomposition works

| Property | Joint training | Multi-stage |
|---|---|---|
| Identifiability | 差（多个 β 互相 entangle） | 好（每 stage 唯一 free param 集合） |
| Loss scale 平衡 | 需要手动 weight tuning | 每 stage 独立 supervision |
| Debug | 难（一个 stage 坏全坏） | 易（每 stage 独立 metric） |
| Theoretical fit | end-to-end better in theory | 多 stage 在 transport modeling 是标准 (Ben-Akiva 1985) |
| 计算成本 | 高（每次 fwd 全 pipeline） | 低（Stage 1 一次跑完 cache embedding） |

### 5.6 Implementation TODO

- [ ] `train.py` Stage 1：encoder + decoder reconstruction loss
- [ ] `train.py` Stage 2：utility NN fit on Census choice，encoder frozen
- [ ] `train.py` Stage 3：scalar grid search + multi-metric eval
- [ ] `evaluate.py`：spatial holdout test set 上的 OD CPC、accessibility correlation、mode share KL

### 5.7 Open questions

- [ ] Stage 1 是否要加 contrastive auxiliary（除了 OD reconstruction，加一个 same-borough vs. cross-borough discrimination）
- [ ] Stage 3 的 weights w1/w2/w3 怎么决定（Pareto frontier 还是 lex order）
- [ ] 是否在 Stage 3 之后做一个 mini joint fine-tune（unfreeze utility NN 上层 1 epoch）

---

## 6. Feedback loops 详细说明

### 6.1 Congestion equilibrium loop（红色）

```
iteration k = 0:
  agent_n: choose j* using free-flow t_ij^0
  → Step 3 aggregates flows → t_ij^1 (with congestion)

iteration k = 1, 2, …:
  agent_n: re-choose j* using t_ij^k
  → Step 3 re-aggregates → t_ij^(k+1)

  Apply MSA averaging:
    t_ij^(k+1) = (1 − 1/k) × t_ij^k + (1/k) × t_ij^new

stop when:
  max_{i,j,t} | t_ij^(k+1) − t_ij^k | < ε  (typically ε = 1 minute)
  or k > k_max  (typically 20)
```

**Why MSA**：保证 monotone convergence；纯 best-response 会 oscillate（拥堵 → 大家避开 → 那条路又空了 → 大家又涌过去）。

**Computational cost**：每次 iteration 重跑 N agents 的 path + softmax sample。在 1500 grid + 100 K agents 上预计 ~30s / iter on GPU → 全 equilibrium 5-10 min。

### 6.2 Calibration loop（紫色）

```
Stage 1 → freeze encoder
Stage 2 → freeze utility NN
Stage 3 → grid search scalars

→ optionally repeat:
   re-train Stage 2 utility with NEW scalars from Stage 3
   (1-2 outer iterations max)
```

**为什么 outer iteration 通常 1-2 次**：Stage 3 改 scalar 会改 t_ij^t 的尺度，原 utility NN 的 weight 可能偏；但 outer iteration 太多会回到 joint training 的 identifiability 问题。

---

## 7. 数据 cross-reference

每个 step 用的数据 source：

| Step | 主要 input | 文档 |
|---|---|---|
| Step 1 | Node features (21+4 dim), kNN edges, hourly OD | `03_node_features.md`, `04_edge_features.md`, `05_training_loss.md` §4.2 |
| Step 2 | h_i^t (from Step 1), SOC×industry, SES, fares | `03_node_features.md` §3.2 (industry mix); 单独需要 SOC affinity provider |
| Step 3 | Road / PT network, BPR params, capacity | OSM + TfL（伦敦），后续 Beijing |
| Step 4 | Census 2021 OD, JTS travel time, mode share | 与 v1 overlap，详见 `05_training_loss.md` §5 |

---

## 8. Counterfactual evaluation cross-reference

反事实评估（"通州副中心扩容"、"伦敦新就业中心"等情景）的三层证据链 framework 跟具体模型架构无关，**直接参考** `../../02_london_commuting_model/methodology/02_counterfactual_evaluation.md`。

简要回顾三层：

1. **Layer 1 (in-sample fit)**：模型在训练 / val / test set 上的 OD CPC、accessibility correlation
2. **Layer 2 (retrodiction / 历史回放)**：用历史已知干预（e.g. Crossrail 开通）做 holdout，检查模型预测 vs. 真实变化
3. **Layer 3 (counterfactual robustness)**：反事实情景下，结果对 BPR β、agent scale、utility NN seed 的敏感性

→ Layer 2 + 3 是反事实可信度的关键，不是 Layer 1 的高 CPC。

---

## 9. 关于 03 / 04 / 05 docs 的 scope

| 文档 | Scope | 状态 |
|---|---|---|
| `03_node_features.md` | **仅 Step 1 GNN 节点特征** | OK，clarification 待加（标明是 Step 1） |
| `04_edge_features.md` | **仅 Step 1 GNN 图拓扑 + 边特征** | OK，clarification 待加（标明是 Step 1） |
| `05_training_loss.md` | **仅 Step 1 forward pass + 单 stage loss** | **PARTIAL**：缺 Step 4 多 stage calibration 描述（已加 OBSOLETE header） |

**读者建议**：先读本文档（00）建立全局框架，再按需读 03/04/05 拿 Step 1 细节。Step 2/3 的实现细节预留 `06_workplace_choice.md` / `07_congestion_abm.md`（待写）。

---

## 10. Implementation TODO 总览

### Phase 1（伦敦 baseline，2026 Q3 目标）

**Step 1**:
- [ ] `models_lib/stgnn.py` (GraphSAGE + GRU)
- [ ] `models_lib/decoder.py` (bilinear + γ log d)

**Step 2**:
- [ ] `models_lib/agents.py` (agent generation pipeline)
- [ ] `models_lib/utility.py` (utility NN)
- [ ] `providers/soc_industry.py`

**Step 3**:
- [ ] `models_lib/congestion.py` (BPR + crowding + MSA)
- [ ] `providers/network.py` (road / PT graph)
- [ ] `models_lib/accessibility.py`

**Step 4**:
- [ ] `train.py` Stage 1 / 2 / 3
- [ ] `evaluate.py` 三层 metric reporting

**Equilibrium**:
- [ ] `models_lib/equilibrium.py`：MSA outer loop wrapping Step 2 ↔ Step 3

### Phase 2（北京阶段，伦理通过后）

- [ ] BeijingFeatureProvider / BeijingNetworkProvider
- [ ] 北京 OD + 通勤时间分布合成
- [ ] BPR α/β 在北京数据上重新 calibrate

---

## 11. Open questions（架构层面的）

- [ ] **Equilibrium 是否每 step 4 outer iteration 都重跑**：训练 Stage 2 时拿到的 t_ij^t 是 free-flow 还是带 congestion 的？如果是后者，t_ij^t 来自哪一轮 equilibrium？
- [ ] **Multi-objective Stage 3 的权重** w1/w2/w3 由 user 拍 vs. Pareto frontier 报告
- [ ] **Step 3 PT crowding penalty** 的具体 functional form 是否需要文献支持（vs. 工程经验值）
- [ ] **Counterfactual 时是否重跑 Stage 1**：副中心扩容会改变 employment by sector → 改变 Step 1 input → 是否要重训 encoder？还是 freeze encoder 只重跑 Step 2/3 forward？

---

## 12. 参考

### 架构层文献
- McFadden 1974 — Conditional logit (Step 2 RUM closure)
- Ben-Akiva & Lerman 1985 — Discrete choice handbook (multi-stage estimation)
- Sheffi 1985 — Urban Transportation Networks (BPR + MSA equilibrium)
- Hansen 1959 — Accessibility gravity (Step 3)
- Wardrop 1952 — User equilibrium (Step 3 fixed point)

### ML 文献
- Hamilton et al. 2017 — GraphSAGE (Step 1 spatial encoder)
- Cho et al. 2014 — GRU (Step 1 temporal encoder)
- Kipf & Welling 2017 — GCN baseline (Step 1 ablation)
- Yao et al. 2021 — SI-GCN (closest baseline; v2 contribution edges over Yao)

### 数据 / 校准文献
- Wilson 1971 — Gravity model (Step 4 Stage 1 baseline)
- Simini et al. 2012 — Radiation model (Step 4 Stage 1 baseline)
- WebTAG TAG Unit A1.3 (UK DfT 2024) — VOT for Step 2 cost weight
- Lenormand et al. 2016 — CPC metric (Step 4 evaluation)

### v2 项目内部文档
- `03_node_features.md`, `04_edge_features.md`, `05_training_loss.md` — Step 1 detailed
- `../../02_london_commuting_model/methodology/02_counterfactual_evaluation.md` — 反事实评估三层框架
- `../README.md` — v2 项目说明
- memory `project_v2_full_model.md` — v2 方向 + 暂停点
- memory `project_beijing_constraint.md` — Beijing-friendly 硬约束
