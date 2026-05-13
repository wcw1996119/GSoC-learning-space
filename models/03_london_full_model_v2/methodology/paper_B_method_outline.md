# Paper B — Method Outline

**Working title:** *Cross-City Inverse Choice Learning: Recovering Transferable Behavioral Parameters from Aggregate Commuting OD via Invariance Penalties*

**One-sentence pitch:** 从两座城市（北京 + 伦敦）的聚合 OD 流量中，反推一组在两边都成立的出行行为参数 (β_t, γ, GNN θ̂)，使同一个学到的 destination utility 模型能够跨城市迁移。

---

## 1. Introduction (~200 words)

Paper A 已经证明：在伦敦单城内，给定 2021 census MSOA-level OD，可以通过 differentiable inverse RUM 恢复出可解释的行为参数 (β_t, γ, δ) 以及 GNN 参数化的 destination value V_j（McFadden 1974; Train 2009; Wang & Klabjan 2018; Sifringer et al. 2020）。**Research question of Paper B**：这套学到的 β̂、γ̂、θ̂，能不能直接用到北京？或者更一般地，能不能学出一个 *cross-city invariant* 的 θ̂，让同一个 destination value function 在两座结构差别极大的城市（伦敦：放射 + 多中心；北京：环路 + 单中心 + 副中心）都成立？

**Gap**：当前 deep choice models（包括 Yao et al. 2021 IEEE TITS 的北京 GCN 模型）几乎都是 single-city 训练 + single-city evaluate。Krueger et al. 2021、Arjovsky et al. 2019 已经在 vision / NLP 上反复显示：单环境训练的 deep model 会过拟合 spurious correlations（在城市语境下：地铁线路布局、特定通勤走廊、就业中心绝对位置），换一个城市就崩。

**Contribution**：把 V-REx (Krueger et al. 2021) 和 HSIC (Gretton et al. 2008) 两个 invariance penalty 嵌入 Paper A 的 inverse RUM 训练 loop，使得 (a) 两城 NLL loss 大致相等，(b) 学到的 V_j 表示在统计上独立于 city ID。这把 Schölkopf et al. 2021 提出的 "single-component invariance" 思想第一次落到 urban choice modeling 上。

---

## 2. Method (~400 words)

### 2.1 Inverse RUM recap (from Paper A)

For commuter living in origin i choosing work destination j ∈ J_i:

$$U_{ij} = \alpha V_j + \beta_t\, t_{ij} + \gamma \log d_{ij} + \delta\, \mathrm{OccMatch}_{ij} + \varepsilon_{ij}$$

ε ~ Gumbel ⇒ choice probability 是 softmax over j（McFadden 1974）。给定观测的 aggregate OD flow F_ij，损失是 weighted NLL：

$$\mathcal{L}(\theta, \beta, \gamma, \delta) = -\sum_{i,j} F_{ij} \log P(j \mid i; \theta, \beta, \gamma, \delta)$$

Paper A 已经证明这个 loss 在 synthetic recovery 上 ~95% 恢复 ground-truth 参数（W4 状态）。

### 2.2 GNN parameterization of destination value

$$V_j = \mathrm{GNN}_\theta(X_j, \mathcal{N}(j))$$

具体是 2 层 GAT (Veličković et al. 2018) + 一个 GRU 聚合 historical features。X_j 包括 employment density、occupation mix、log floor area。

### 2.3 Cross-city loss (Paper B 的核心新增)

把伦敦和北京视作两个 environments e ∈ {London, Beijing}。每个 environment 有自己的 NLL loss L_e。Total loss：

$$\mathcal{L}_{\mathrm{total}} = \sum_e \mathcal{L}_e + \lambda_{\mathrm{VREx}} \cdot \mathrm{Var}_e[\mathcal{L}_e] + \lambda_{\mathrm{HSIC}} \cdot \mathrm{HSIC}(V_j, c_j)$$

其中 c_j ∈ {0, 1} 是 destination j 所属城市的 ID。

**为什么用 V-REx**：Krueger et al. 2021 证明 risk extrapolation（最小化跨环境 loss 方差）比 IRM (Arjovsky et al. 2019) 在 covariate shift 大的场景下更稳。北京 vs 伦敦的 covariate shift 极大（density、通勤距离分布、收入），V-REx 比 IRM 更合适。直觉：penalty 强迫模型不能在伦敦特别准、在北京特别烂。

**为什么用 HSIC**：Gretton et al. 2008 的 kernel-based independence test。这里把 city ID 当作 nuisance variable，强迫学到的 V_j representation 与 city ID 统计独立。这正好对应 Schölkopf et al. 2021 "Toward Causal Representation Learning" 里讲的 "single-component invariance"——某些行为机制（人对通勤时间的负效用 β_t）应该是跨文化稳定的，应该把它从 city-specific 的 nuisance 里 disentangle 出来。

**两个 penalty 是互补的**：V-REx 作用在 *loss* 层面（outcome-level invariance），HSIC 作用在 *representation* 层面（feature-level invariance）。Paper B 主张两个一起用。

---

## 3. Experiments (~250 words)

### 3.1 W3–W4: Multi-borough dry run on London (NOW, no Beijing needed)

把伦敦 33 个 borough 切成 K=4 组（比如 Inner West / Inner East / Outer North / Outer South），每组当作一个 "fake city"。在这个 setup 上跑完整的 V-REx + HSIC pipeline：

- 验证 invariance penalty 实现正确（gradient flow、λ schedule）
- 测试 single-city baseline 和 multi-environment 训练之间的 generalization gap
- 选 λ_VREx, λ_HSIC 的初步范围
- **这一步现在就能做**，不依赖北京 ethics 批复

### 3.2 W5–W8: Cross-city on Beijing + London (post-ethics)

- 数据：北京 2020 census OD（街道级，~300+ zones）+ 伦敦 2021 OD（MSOA, ~983 zones）
- Train: leave-one-city-out + V-REx + HSIC
- Test: held-out OD pairs in each city

### 3.3 Validation

1. **VOT sanity check**：从恢复的 (β_t, β_c) 算出 Value of Travel Time = β_t / β_c × 60，要求两城都落在 WebTAG 范围 £8–22/h。这是行为参数 plausibility 的硬约束。
2. **Counterfactual extrapolation**：用 shared θ̂ 跑两个反事实——北京通州副中心新增就业 200k；伦敦 Old Oak Common HS2 站 catalysis。看预测的 OD 重分布是否符合规划文献的 ex-ante 估计。
3. **Ablations**：去掉 V-REx / 去掉 HSIC / 都去掉 / 换成 IRM。预期 V-REx + HSIC 在 cross-city test NLL 上严格优于其余 setting。

---

## 4. Relationship to Paper A (~100 words)

Paper B 的代码 = Paper A 的 trainer + 两行额外 loss term（V-REx variance penalty + HSIC kernel penalty）+ environment indexing。**不是重写**，是 incremental extension。这意味着：

1. Paper A 必须先投出去（NeurIPS workshop / TMLR）。在 Paper A 还没定稿之前不动 Paper B 的代码。同时开两个项目是上次（2026-05-05 教训笔记里）已经记录过的反模式。
2. 北京 ethics 批复期间，可以做 §3.1 的 multi-borough dry run，把 invariance machinery 完全调通。等北京数据一到，理论上一周内能跑出第一版 cross-city result。
3. Paper B 最终是 NeurIPS / ICLR main track 体量，因为加上了 causal representation 的 angle 和真实 cross-city benchmark。

---

## 5. Risks & Limitations (~150 words)

1. **Ethics timeline**：北京数据走的是机构 ethics 委员会流程，不可控。Mitigation：§3.1 的 London multi-borough dry run 是不依赖北京数据的 fallback paper，万一 ethics 一直批不下来，可以单独投成 "intra-city domain generalization" 的 short paper。
2. **β identifiability across cities**：如果 β_t 在两城 *本来就* 不同（比如北京通勤者对时间的边际负效用确实更高，因为绝对通勤时间更长），那 V-REx 强行拉平 loss 可能是 misspecified。需要做 sensitivity：分别 fit 单城 β_t，看差异是否落在合理 range 内，再决定是 share β 还是 share θ-only。
3. **Functional form**：北京可能需要把 mode choice（地铁 vs. 自驾 vs. 共享单车）显式建模，伦敦的 PT/car/active 三分法不一定平移得过去。Paper B v1 先用 destination choice only，mode 留给 v2。
4. **Aggregation bias**：MSOA-level / 街道-level 都是 aggregate OD，存在 ecological fallacy 风险（Train 2009, Ch. 2）。结论应限定为 zone-level 行为，不外推到个体。

---

## References (key papers, inline-cited above)

- Arjovsky et al. 2019 — Invariant Risk Minimization
- Gretton et al. 2008 — A Kernel Statistical Test of Independence (HSIC)
- Krueger et al. 2021 ICML — Out-of-Distribution Generalization via Risk Extrapolation (V-REx)
- McFadden 1974 — Conditional logit analysis of qualitative choice behavior (RUM)
- Schölkopf et al. 2021 — Toward Causal Representation Learning
- Sifringer et al. 2020 — Enhancing discrete choice models with representation learning (TasteNet)
- Train 2009 — *Discrete Choice Methods with Simulation* (2nd ed.)
- Veličković et al. 2018 — Graph Attention Networks
- Wang & Klabjan 2018 — Deep choice model
- Yao et al. 2021 IEEE TITS — Beijing commuting GCN reference
