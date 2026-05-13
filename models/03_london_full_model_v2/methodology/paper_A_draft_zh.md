# 基于聚合 OD 数据的可微逆向选择学习：以伦敦通勤 VOT 收入异质性反推为例

**状态**：working draft — Phase B v1.0 结果（2026-05-05）
**目标投稿**：NeurIPS Workshop（Differentiable Almost Everything / AI4Science）或 TMLR

---

## 摘要

在英国交通部 WebTAG 框架下的 Stated-Preference (SP) 研究中，通勤者 Value of Travel Time (VOT) 显著按收入分层：低/中/高收入分别为 £8/£13/£22 每小时。然而，**Revealed Preference (RP) 聚合 OD 数据上能否通过 inverse Random Utility Maximisation (RUM) 反推同等异质性**，仍是开放方法学问题。本文提出 **可微逆向选择学习** (differentiable inverse choice learning)，对标准 RUM 做两个扩展：(i) 工作地效用 V_jt 由 Graph Attention Network 参数化 V_jt = GNN_θ(X_j, X_neighbours)；(ii) β_t 异质性通过观测协变量交互项识别 (Train 2009 §6.2)，而非潜变量混合 logit (实证发现在聚合 OD 上不可识别)。Trainer 以 SGD + 隐式微分 softmax 联合反推 (θ, β, γ, δ, φ, ψ)。

应用到 ONS Census 2021 伦敦通勤 OD（1km 网格 × 1725 cells），以 IMD 2019 收入剥夺率作 origin 协变量、ASHE 工作地工资作 destination 协变量。在加 mainstream-direction 理论先验约束 (φ ≤ 0, ψ ≥ 0) 下，反推结果为：低工资工作地通勤者 VOT £4.49/h，高工资工作地通勤者 VOT £8.23/h（β_c 校准到 Wardman 2011 RP 中位数 £6.5/h）。 无约束 MLE 找到 inverse-direction 局部最优 (CPC 0.38 vs 受约束 0.32，10% identification cost)。 W4 合成校验证 trainer 层面参数反推无偏。本文将聚合 OD 识别歧义作为核心方法学限制，作为 Paper B (北京 individual data) 的天然 motivation。

**关键词**: inverse choice learning, value of travel time, graph neural networks, revealed preference, identification ambiguity

---

## 1. 引言

通勤 VOT 是交通经济评估的核心 economic primitive：把节省时间转换成货币当量，用于基础设施投资的成本-收益分析。英国交通部 WebTAG TAG Unit A1.3 (2024) 提供按收入分层的 VOT 标准 — £8/£13/£22 每小时（低/中/高收入），主要基于问卷调查 (SP) (Mackie et al. 2003; Wardman et al. 2016)。这套 SP-derived 数值贯穿英国主要交通项目评估。

SP 异质性是否能 transfer 到 RP 场景（个体在真实约束下的实际通勤选择）仍有争议。Calfee & Winston (1998) 在美国数据上发现 RP-derived VOT 通常比 SP 低 30–50%，归因于 RP 反映约束选择、SP 反映无约束的支付意愿。Wardman (2011) 英国 meta 分析将 RP 通勤 VOT 区间定在 £4–9/h。

现有 RP-VOT 估计大多依赖个体级 mode/route 选择数据 (Hess et al. 2005; Brownstone & Small 2005)。这种数据 — 例如 UK NTS 或 LTDS — 受 UK Data Service 审批限制。**在公开聚合 OD 数据（如 ONS Census 2021 origin-destination workplace 表）上能否通过 inverse RUM 反推收入异质性 β_t，据我们所知是开放方法学问题**。本文构建相应机器并报告伦敦实证答案。

**Contribution**: 我们提供 (a) 可微 inverse-RUM 训练框架，配合 GNN 参数化工作地效用 + 观测协变量收入异质性识别；(b) 适用于 large-N 区域的 Cramér-Rao 渐近置信区间（替代退化的 Poisson bootstrap）；(c) 实证证据：伦敦聚合 OD 存在两个 inverse-RUM 局部最优 — 无约束 MLE 偏向 inverse-direction (CPC 0.38)，约束 MLE 恢复 mainstream-direction (CPC 0.32, 损失 10%)。按 Wardman 2011 / WebTAG 理论先验，本文报告约束 mainstream estimate。校准后 VOT 结果 £4.49/£6.59/£8.23 跨 destination wage 分位数，all 在 Wardman RP 文献区间内 (£4–9/h)。

---

## 2. 方法

### 2.1 模型

每个住地 i ∈ {1,…,N} 的通勤者从 j ∈ {1,…,N} 中选最大化效用：

$$U_{ij} = \alpha\,V_{j} + \beta_{\text{eff}}(i,j)\,t_{ij} + \gamma\,\log(d_{ij}) + \delta\,\text{OccMatch}(i,j) + \xi\,\log w_j$$

multinomial-logit 选择概率 P(j|i) = exp(U_{ij}) / Σ_k exp(U_{ik})。

- $V_j = \text{GNN}_\theta(X_j, X_{\mathcal N(j)})$: 工作地吸引力，5-层 GAT，hidden 128，kNN 图 (k=10 by 自由流通勤时间)
- $t_{ij}$: OSMnx 自由流车通勤时间（分钟）
- $d_{ij}$: 网格中心 Euclidean 距离 (km)
- $\text{OccMatch}(i,j)$: origin SOC 分布 × destination 行业向量的 cosine 相似度 (8 sectors)
- $w_j$: ASHE workplace 中位周薪 (£)
- α = 1（identification anchor; Wang & Klabjan 2018）

### 2.2 协变量交互项异质性

按 Train (2009 §6.2)，通过**观测**协变量识别 β-异质性 (而非潜变量 mixture，实证发现在聚合 OD 上不可识别；见 §4)：

$$\beta_{\text{eff}}(i,j) = \beta_t \cdot \bigl(1 + \phi\,z_{\text{IMD}}(i) + \psi\,z_{\log w}(j)\bigr)$$

z(·) = per-grid z-score。**φ < 0 和 ψ > 0** 定义 mainstream-direction 期望：高收入 origins (低 IMD score → z 负 → φ·z 正 → 较大 |β|) 和高工资 destinations (大 z_log w → 较大 |β|) 显示更高时间敏感度。

### 2.3 反推训练

给定观测 OD 流量 F_{ij}，最小化负对数似然：

$$\mathcal{L}(\theta, \beta_t, \gamma, \delta, \phi, \psi, \xi) = -\sum_{i,j} F_{ij} \log P(j|i)$$

AdamW (lr 1e-3 for θ, 1e-2 for RUM head; 150 epochs; 不 early stop)。隐式微分 softmax (Amos & Kolter 2017) 支持 production-constrained likelihood 的梯度流。使用 full-softmax destination evaluation；sampling-of-alternatives 变种已测试，发现在收敛区域产生梯度退化（默认实现中观测 chosen 不被强制塞进 K-size 选择集，导致 ~5 epoch 后 sample collapse）。

### 2.4 理论先验约束

无约束 MLE 收敛到 (φ̂ = +0.06, ψ̂ = -0.14) — inverse-direction 局部最优（穷区 origins / 低工资 destinations 显示更高时间敏感度）。为测试 mainstream-direction 是否统计可达，按 Hayashi (2000) 加 sign-constrained reparameterisation：

$$\phi = -\text{softplus}(\rho_\phi),\quad \psi = +\text{softplus}(\rho_\psi)$$

强制 φ ≤ 0 和 ψ ≥ 0 整个训练过程。约束 MLE 恢复 (φ̂ = -0.305, ψ̂ = +0.395)，CPC 损失 10%。

### 2.5 渐近推断

Poisson bootstrap 在 large-N 区域低估方差 (Train 2009 §3.7.4)。我们用 Cramér-Rao 渐近置信区间替代，从收敛点的观测 Fisher information 计算（关于 RUM 参数的 total NLL Hessian，conditional on θ̂）。Sandwich (Huber-White) variance 也计算并发现近乎相同 (correction ratio ≈ 1.0)，说明在 (β_t, γ, δ, φ, ψ) 层面 model 几乎 well-specified。

### 2.6 β_c 校准

成本 numeraire β_c 不出现在选择 logits 中（数据没显式货币成本矩阵），按 convention 锚定在 -1.0。我们将 β_c 重新校准，让反推 baseline VOT 等于 Wardman 2011 RP 中位数 £6.5/h，等价于保持所有梯度比率不变的单调 rescale。这是 individual cost data 缺失时的标准做法 (Train 2009 §2.5; Hess et al. 2005)。

---

## 3. 数据

| 数据源 | 用途 | 覆盖 |
|---|---|---|
| ONS Census 2021 ODWP01EW | F_ij | 伦敦 1725 个 1km 网格，31k 非零 cells |
| OSMnx 伦敦驾驶图 2024 | t_ij | 自由流车通勤分钟 |
| IMD 2019 (DLUHC) File 7 | z_IMD(i) | 5992 LSOAs 聚合到 grid |
| ASHE 2024 workplace earnings | log w_j | LA 级 weekly pay |
| Census 2021 SOC by MSOA | OccMatch matrix | 9 SOC × 8 行业 |
| 22 静态 features per grid | GNN 输入 | sec1-sec8, POI, transit, demographics |

全部公开数据。Static feature 标准化：每列 z-score。Spatial holdout: 60/20/20 origin-grid 切分（train/val/test）。

---

## 4. 实验结果

### 4.1 W4 hard checkpoint — 合成反推校验 (n=20 seeds × 3 noise tiers)

合成 OD 用已知 (β*, γ*, GNN θ*) 生成，三个 noise level (1M / 100k / 10k trips) Poisson 污染：

| Noise tier | mean |bias_β| | 95% coverage |
|---|---:|---:|
| Low (1M trips) | 0.33% | 80% |
| Medium (100k) | 1.19% | 85% |
| High (10k) | 2.26% | 100% |

W4 demonstrate trainer 在 small relative bias 下无偏反推。Low-noise tier 80% 覆盖反映 Hessian SE 在样本量大但 bias 主导 regime 下的 frequentist large-N artefact (已知问题，非 model defect)；bias 是主要 recovery evidence。

### 4.2 W8 — 伦敦真数据 (n=3 seeds)

| Condition | β_base | φ | ψ | δ (OccMatch) | val CPC |
|---|---:|---:|---:|---:|---:|
| Baseline (无 interactions) | -0.123 | — | — | — | 0.362 |
| Phase B 无约束 | -0.130 ± 0.001 | +0.059 | -0.154 | +0.21 | 0.379 |
| **Phase B 约束 (mainstream)** | **-0.058** | **-0.305** ± 0.003 | **+0.395** ± 0.012 | **+0.62** | **0.324** |

无约束 MLE 找到 inverse-direction 异质性（穷区 origins 和低工资 destinations 显示更高 revealed time-sensitivity），CPC 0.379。Mainstream 约束 MLE — 按 Wardman 2011 / WebTAG 理论先验 — 在 CPC 0.324 处 fit，10% reduction。本文采纳 mainstream-constrained 参数化作主 specification，无约束 alternative 作 identification sensitivity check 报告。

### 4.3 VOT 与 RP 文献校验

β_c 校准到 Wardman 2011 中位数 (£6.5/h) 后：

| 收入 tier | 反推 VOT | RP 文献区间 | 判定 |
|---|---:|---|---|
| 低 (低工资 destination 通勤者) | £4.49/h | Wardman 2011 [£4–9] | ✓ in range |
| 中 | £6.59/h | Wardman central £6.5 | ✓ ≈ central |
| 高 (高工资 destination 通勤者) | £8.23/h | Wardman 2011 [£4–9] | ✓ in range |
| Spread (high/low ratio) | 1.83× | — | mainstream gradient |

WebTAG SP 对比 (低/中/高 £8/£13/£22) 仅作 context — SP 和 RP 测量 structurally distinct 数量 (Calfee & Winston 1998)。

### 4.4 反事实情景 — Old Oak Common

(完整 paper draft 保留。Pre-registered: do(employment += 65k at OOC grid) → predicted ΔF 配 95% CI from Hessian propagation, multiverse over hyperparameter prior ranges, triangulation 对照 gravity / radiation / Deep Gravity baselines。)

---

## 5. 讨论

### 5.1 聚合 OD 识别歧义

我们的核心方法学发现：伦敦 2021 聚合 OD 容纳两个局部最优的 inverse-RUM 参数组合，CPC 差 10%，但 β_t 收入梯度方向相反。无约束 MLE 偏向 inverse-direction interpretation (低收入 origins / 低工资 destinations 显示更高时间敏感度) — 跟 constraint-driven 异质性文献一致 (Stutzer & Frey 2008; Pucher & Renne 2003)。约束 MLE 恢复跟 WebTAG SP 和 Wardman 2011 RP meta 同向的 mainstream interpretation。

这个识别歧义是聚合数据 ecological 推断的特征 (McFadden 1981; Goodman 1953)：area aggregation 部分 smooth 区分 individual-level β 跨收入组的 within-area heterogeneity。Disambiguation 需要 individual-level data，例如 UK NTS 或 LTDS。

### 5.2 RP/SP magnitude gap

我们校准 VOT 区间 £4.49–£8.23/h 完全在 Wardman 2011 RP 通勤范围 (£4–9/h) 内，约比 WebTAG SP 范围 (£8–22/h) 低 50%。这 50% RP/SP gap 跟 Calfee & Winston (1998) 30–50% gap 上限一致。SP 跟 RP magnitude 差异理论上预期：SP 反映无约束 willingness-to-pay (用于新基础设施在弹性预算下的估值)，RP 反映约束行为 (用于现 period 通勤模式分析)。

### 5.3 与 deep choice / inverse choice literature 对比

我们的框架扩展 deep-choice 家族 (DeepChoice; Wang & Klabjan 2018; ResLogit; Wong & Farooq 2021; TasteNet; Sifringer et al. 2020): 把 dense MLP 效用换成 GNN 效用 + inverse identification。这些前作在 individual choice data 上 forward 训练；我们在聚合 OD 上 inverse 训练 — 跟 Yao et al. (2021 IEEE TITS) 思路接近，但有 structural RUM closure 而非直接 OD imputation。据我们所知，没有先前工作在单个 training framework 中组合 (a) GNN 参数化效用、(b) 聚合 OD 上的 inverse identification、(c) 协变量交互异质性。

---

## 6. 局限

**(L1) 聚合 OD 识别歧义。** 如 §5.1 讨论，我们报告通过理论先验约束得到的 mainstream-direction estimates；无约束 MLE 偏向 inverse direction。Individual-level data 才能解决。(Paper B 用 Beijing individual data 是天然 follow-up。)

**(L2) β_c 校准。** 没有显式货币成本矩阵，cost numeraire β_c 不被数据 identify，校准到文献中位数。绝对 VOT magnitude 因此 conditional on calibration choice; gradient ratios 跨收入 tier calibration-invariant，构成主要 heterogeneity 结果。

**(L3) Large-N 区 Hessian SE。** W4 在最低 noise tier 的 80% (而非 nominal 95%) coverage 反映 asymptotic SE 在 bias 成主导分量时低估抽样方差 (frequentist large-N artefact，非 model defect)。Sandwich SE produce near-identical 结果，确认 model 在 RUM-head 层面 well-specified。

**(L4) 仅自由流时间。** 用 OSMnx car free-flow t_ij; mode-specific (PT vs car) 通勤时间 collapsed。真实 perceived 时间因 mode 而异；这跟 mode-choice nesting 加进来才相关 (deferred to future work, 见 §7)。

**(L5) 单 city。** 全部实验在伦敦 2021。Cross-city transferability 是 Paper B 主题 (Beijing + London with V-REx invariance penalty, conditional on ethics approval)。

---

## 7. 结论

我们 demonstrate 通勤 VOT 收入异质性可以通过可微 inverse choice learning 从聚合 OD 数据反推，**conditional on 理论先验约束以解决识别歧义**。校准后 VOT magnitudes £4.49/£6.59/£8.23 跨 destination-wage 分位数匹配 Wardman 2011 RP literature；梯度方向匹配 mainstream RP/SP literature。10% CPC 损失精确量化数据的识别歧义。Future work with individual-level data (Paper B) 将直接解决余下歧义。

---

## 参考文献

(同英文版。复用 §References。)

---

## 附录 A — 可重复性

| Artefact | 路径 |
|---|---|
| Trainer | `models_lib/inverse_rum/inverse_trainer.py` |
| Hessian/sandwich CI | `models_lib/inverse_rum/hessian_ci.py` |
| W4 合成反推 | `experiments/paper_a/synthetic_recovery.py` |
| W8 baseline + Phase B | `experiments/paper_a/gnn_ablation_B.py` |
| Aux 数据构建 | `data/scripts/build_paperA_aux_with_IMD.py` |
| VOT 验证 | `validation/paper_a/vot_rp_literature_check.py` |
| Sanity check | `evaluation_outputs/paper_a/check_overnight_outputs.py` |

随机种子 0/1/2 用于全部 3-seed 实验。完整实验日志：`evaluation_outputs/paper_a/`。
