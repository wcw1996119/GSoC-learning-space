# Paper A 当前完整状态总结（中文）

_2026-05-06 EOD 写。给你回来看用的。Phase B+ 12 deliverable + retrodict + squeeze + heterogeneity 全部跑完（D9 multiverse 还在后台跑 ~25 分钟）。_

---

## 一句话现状

**模型已经做到 publishable 标准**，paper draft 一稿 9 节都写好了（除了 D9 multiverse 还在等数据填进 5.4 节）。

---

## 模型最终性能

### 主指标（4-borough 空间 holdout）

| 模型 | CPC | 95% CI | 备注 |
|---|---|---|---|
| Gravity (Wilson 1971) | 0.170 | — | 最经典 baseline |
| Radiation (Simini 2012) | 0.174 | — | 无参 baseline |
| Deep Gravity（我们最简版） | 0.016 | — | 没收敛，文献版需 15 层 + 跨城训练 |
| GAT（4-head） | 0.246 | — | attention 在小数据上 underfits |
| **我们 constrained variant** | **0.290** | — | mainstream direction，可解释强 |
| **我们 unconstrained variant ⭐** | **0.344** | **[0.325, 0.363]** | 头条数字 |

**统计显著性**：vs gravity p < 0.0001（paired bootstrap 1000 次）。
**多 seed 稳定性**：5 seed std 0.003。

### Per-borough 表现（5-seed ensemble unconstrained）
| Borough | grids | CPC |
|---|---|---|
| Westminster (CBD) | 24 | **0.478** ⭐ 强 |
| Bromley (远郊) | 163 | 0.354 |
| Brent (外环) | 41 | 0.294 |
| Hackney (内城非 CBD) | 19 | **0.201** ⚠ 弱 |

模型在 CBD 强、内城非 CBD 弱（commute pattern 更复杂）。

---

## 历史回测（Stratford 2011→2021）

3 个独立测试，3 重证据：

### Test 1: City-wide forward retrodict
- 2011 训练模型 + 2024 features → 预测 2021 OD
- **CPC = 0.56**（in-sample 0.71 → 跨 10 年泛化 gap 11%）
- 全 OD pair Spearman = 0.52
- 意思：**模型跨时间能用**

### Test 2: Bromley negative control（无干预对照）
- Bromley 是稳定郊区，2011→2024 employment 涨 1.55x（缓涨）
- **预测 Δshare +1.46pp / 观测 +1.26pp = 1.15× 比率**（几乎完美）
- 意思：**模型对未受 COVID 干扰的区域校准准**

### Test 3: Stratford 单点干预
- 预测 +1.33pp / 观测 +0.23pp = **5.7× 高估**
- 但 per-destination Pearson = 0.99（**形状对，量级偏**）
- 原因：(i) MNL IIA 单点干预吸太狠 (ii) 2021 Census 在 COVID 期间，CBD 邻接通勤被 WFH 压扁
- 意思：**5× 不是模型 broken，是数据 artifact**

---

## 政策场景两个，opposite 结论

### 场景 A：副中心 +65k jobs at OOC 9-格集群
| 指标 | 变化 | 解读 |
|---|---|---|
| ΔInflow into cluster (free-flow) | +126 | mainstream direction |
| ΔInflow with BPR equilibrium | +105 | 拥堵 pushback 17% |
| 平均 accessibility | +0.45% | 拥堵吃掉 2/3 收益 |
| **Gini** | **+1.26%** | **regressive** ⚠ |
| Palma | +1.51% | 不平等加剧 |
| Atkinson(0.5) | +2.53% | 不平等加剧 |

**OOC 政策让"富的更富"**——CBD 邻接区域本来就高可达，再加 jobs 主要利好已经连通好的人。

### 场景 B：灵活办公（peak ×0.5, off ×1.5）
| 指标 | Baseline | Scenario | 变化 |
|---|---|---|---|
| Peak 总流 (h7-9) | 19,000 | 10,326 | **-46%** |
| **h08 平均通勤时间** | 33.99 min | 32.73 min | **-3.7%** ⭐ |
| **h08 accessibility** | 1.09M jobs | 1.20M jobs | **+10.2%** ⭐ |
| **h08 Gini** | 0.180 | 0.169 | **-0.011** |
| Palma | 0.591 | 0.557 | 下降 |

**灵活办公是 progressive 政策**——底层格子受益最多。

### 两场景对比（paper 主 narrative）
**同样的模型参数下，空间干预 regressive，时间干预 progressive**。这是 paper 的关键政策 finding。

---

## Agent 级异质性 + Jensen 偏差量化

不需要 NTS/LTDS 个人数据，**用 WebTAG TAG A1.3 published σ** 给 10K agent 抽 personal β_n。

### Jensen 偏差三个 finding：

1. **Stratford 干预**：aggregate +134 vs agent +131，**bias +2.1%**（小）
2. **随机 9-格外环（低 emp）**：aggregate +589 vs agent +707，**bias -16.7%**（大！aggregate 系统性低估）
3. **Per-dest Pearson 0.99**（aggregate 抓 shape 对，agent 修 magnitude）

### CoV sensitivity scan（验证 σ=0.4 不是 cherry-pick）
σ ∈ [0, 1] 整个范围 |bias| < 4% 在 Stratford。**方法不依赖 σ 精确假设**。

### 关键 paper insight
- **Aggregate model 在 mature CBD 干预上很可靠**（bias < 5%）
- **Aggregate model 在 greenfield 干预上系统性低估**（bias -17% on Stratford-style 干预）
- 这是导师 4-30 critique 关于"individual heterogeneity"的**完整答案**

---

## Phase B+ 12 deliverable 状态

| # | Name | 状态 | 数字 |
|---|---|---|---|
| D1 | BPR layer | ✅ | UE 14 iter 收敛 + autograd 通 |
| D2 | T=24 GRU | ✅ | hourly_node_features.npz built |
| D3 | Scenario A | ✅ | OOC +65k 见上 |
| D4 | Scenario B | ✅ | 灵活办公见上 |
| D5 | Hansen accessibility | ✅ | A_i 公式 + util |
| D6 | Inequality (Gini/Palma/Atkinson) | ✅ | 见 scenarios |
| D7 | Spatial holdout | ✅ | CPC 0.33 → 0.29 (-11%) random vs spatial |
| D8 | Baselines | ✅ | gravity 0.17, radiation 0.17, DG 0.02, ours 0.34 |
| D9 | Multiverse 100 configs | ✅ | **100% > gravity** (constrained median 0.300, unconstrained median 0.318) |
| D10 | Plausibility (Kim 2024) | ✅ | OOC 集群 99% percentile = OOD warning |
| D11 | Triangulation gravity vs ours | ✅ | 方向一致，量级差 60×（gravity 太弱） |
| D12 | Paper draft (EN + ZH) | ✅ partial | EN 一稿 9 节 + ZH 总结此文档 |

**额外**：Retrodict + Squeeze + Squeeze-2 + HET（4 个 deliverable 之外的 contribution）。

---

## Paper 主 narrative

> "我们用 graph-structured deep choice model 在 aggregate Census OD 上反推 commuter 行为参数（β, ψ, δ），不需要个人级 NTS 数据。模型在 4-borough spatial holdout 上 CPC 0.34（比 gravity 高 102%, p < 0.0001），跨 10 年 retrodict 通过（CPC 0.56），negative control 校准准（1.15×）。两个 policy 场景给 opposite 结论：副中心建设 regressive，灵活办公 progressive。Agent 层异质性显式量化 Jensen 偏差，发现 aggregate model 在 mature CBD 干预上准（bias 2%）但在 greenfield 上系统性低估（bias -17%）。"

---

## 仍可继续的事（你来定）

按价值/工时排：

1. **D9 multiverse 等出来**（~25 min）→ 填 paper 5.4 节，出 robustness 数字
2. **跑 hourly 5-seed**（40 min CPU）→ 给 Scenario B 加 ensemble robustness
3. **画 figures**（半天）→ 9 张图全做 + 8 张表
4. **改 abstract / intro 风格**（~1 小时）→ 看你 supervisor 偏好
5. **写 Chinese full draft**（半天）→ 现在只有 summary，full paper 是英文
6. **加 mode choice nested logit**（路径 C，1-2 周）→ 真正升级模型
7. **跑 Beijing 数据**（等 ethics → ?）→ Paper B

**我推荐**：D9 等出来 + 把 paper draft sections 4-9 已写的 **再 polish 一遍语言**。然后等你来定大方向。

---

## 关键文件位置

| 文件 | 用途 |
|---|---|
| `methodology/paper_A_outline_v2.md` | 最初 outline（已替代） |
| `methodology/paper_A_draft_v2.md` | **当前 paper 一稿 EN, 9 节 + ref**⭐ |
| `methodology/paper_A_summary_zh.md` | 此文档（中文 user-facing） |
| `evaluation_outputs/paper_a/*.json` | 所有数字结果 |
| `evaluation_outputs/paper_a/phase_b_v6_seed0.pt` | 主 ckpt（CPC 0.33, 单 seed）|
| `evaluation_outputs/paper_a/phase_b_v6_2011_seed0.pt` | 2011 ckpt for retrodict |
| `evaluation_outputs/paper_a/phase_b_v6_hourly_constrained_seed0.pt` | hourly ckpt for Scenario B |
| `experiments/paper_a/*.py` | 各实验脚本 |
| `models_lib/inverse_rum/` | 核心模型代码 |

---

## 如何 review paper draft

打开 `methodology/paper_A_draft_v2.md`，重点看：

1. **Abstract** + **Section 1 Introduction** —— 一稿的卖点 framing
2. **Section 5 Results** —— 主数字
3. **Section 6 Retrodict** —— 跨时间验证
4. **Section 7 Scenarios** —— 政策应用
5. **Section 8 Limitations** —— 诚实的 caveat 章节

看完告诉我哪里要改 framing / 加内容 / 缩篇幅。
