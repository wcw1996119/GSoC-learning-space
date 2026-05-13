# Paper B — Literature Review: Invariance Penalties for Cross-Environment Learning

本文档为 Paper B 的 Related Work 部分提供文献基础。Paper B 的核心 setup 是：复用 Paper A 的 inverse RUM + GNN 机制，在 loss 中加入 invariance penalty，使得恢复出的偏好参数 (β, γ) 和 GNN 表示 θ 在 London 与 Beijing 两个 city 上同时成立。本综述聚焦于两类候选 penalty——**V-REx** 与 **HSIC**——的理论起源、相关 baseline，以及在 cross-city / spatial transfer 场景中的相关工作。

每篇文献的讨论结构为：(a) 论文提出了什么；(b) 与 Paper B 的相关性；(c) 我们 adopt / adapt / not use 的部分。对于不确定的细节（venue、页码、具体年份），我会标注 "(verify before paper submission)"。

---

## Section 1 — V-REx 与 IRM 谱系

V-REx 是 Paper B 计划主用的 invariance penalty 之一，其概念直接来源于 IRM。这一节追溯两篇核心文献。

### 1.1 Arjovsky, Bottou, Gulrajani, Lopez-Paz (2019) — "Invariant Risk Minimization"

(a) IRM 论文是 invariance penalty 思路在现代深度学习中的奠基之作。该 arXiv preprint (2019, arXiv:1907.02893, verify before paper submission whether it appeared in a formal venue) 把 environment-wise data 视为来自不同 SCM 的样本，提出寻找一个 representation Φ(x)，使得 optimal classifier on top of Φ 在所有 environment 上同时是 optimal 的。形式上转化为一个 bi-level optimization，在实践中近似为 IRMv1 penalty——对每个 environment 的 risk 关于 dummy classifier 求梯度的平方范数。

(b) 对 Paper B 的相关性：IRM 提供了 "在多个 environment 上要求条件 invariance" 这一概念框架。London 与 Beijing 在 IRM 语言里就是两个 environment，我们想要 recover 的 (β, γ) 在 IRM 视角下扮演 "invariant predictor" 的角色。

(c) Adopt: 多 environment 形式化、environment-conditional invariance 的概念框架。Adapt: IRMv1 penalty 形式过于敏感（gradient-based，optimization 不稳定，参见 Rosenfeld et al. 2021 critique, verify before paper submission），因此我们不直接用 IRMv1。Not use: bi-level optimization 形式；我们改用 V-REx 的 risk-variance 形式，因其优化更稳定。

### 1.2 Krueger, Caballero, Jacobsen, Zhang, Binas, Le Priol, Courville (ICML 2021) — "Out-of-Distribution Generalization via Risk Extrapolation (REx)"

(a) 这篇 ICML 2021 论文提出了 V-REx penalty：在 ERM loss 之上加一项 environment-wise risk 的方差 Var_e(R_e(θ))。直觉是：若模型在所有 environment 上 risk 都相近（low variance），则在 risk 空间中外推到 unseen environment 时会更鲁棒。论文同时提出 MM-REx（minimax 形式），并从理论上论证 V-REx 在 affine combinations of training risks 上比 ERM 更鲁棒。

(b) 与 Paper B 高度相关。Paper B 只有两个 environment（London, Beijing），V-REx 的 risk-variance 形式在 |E|=2 时退化为 (R_London − R_Beijing)² 的尺度——非常干净，optimization 稳定。这正契合我们 "用最小代价跨城市迁移" 的目标。

(c) Adopt: V-REx 的 risk-variance penalty 形式，作为 Paper B 主 penalty 候选之一。Adapt: 原 V-REx 用在 supervised classification（CIFAR、ColoredMNIST），我们要把它套到 RUM negative log-likelihood 上——R_e(θ) 改成 city-specific NLL，即 R_London = −Σ log P_London(j | i; β, γ, θ_GNN)。Not use: MM-REx 变体，因为我们 environment 数太少，minimax 退化无意义。

---

## Section 2 — HSIC 与 information-theoretic dependence measures

HSIC 是 Paper B 的第二条 penalty 候选路线，思路与 V-REx 不同：不是约束 risk 在 environment 间相近，而是直接约束某层 representation 与 environment label 之间的统计独立性。

### 2.1 Gretton, Bousquet, Smola, Schölkopf (ALT 2005) — "Measuring Statistical Dependence with Hilbert-Schmidt Norms"

(a) 这篇 ALT 2005 paper（后续 JMLR 系列扩展，verify exact venue/year for the version you cite）定义了 HSIC：两个随机变量 X, Y 的 cross-covariance operator 在 RKHS 上的 Hilbert-Schmidt 范数平方。HSIC = 0 当且仅当 X ⊥ Y（在 universal kernel 假设下）。它有 O(n²) 的 biased empirical estimator，基于两个 kernel matrix 的中心化乘积的迹。

(b) 与 Paper B 的相关性：HSIC 给我们提供了 "强制 GNN representation 与 city label 独立" 的可微 penalty。形式上 penalty = HSIC(GNN_embedding, city_indicator)。这与 V-REx 的 risk-level 约束互补——HSIC 是 representation-level 约束。

(c) Adopt: HSIC 的 biased empirical estimator (HSIC_b)，在 mini-batch 上计算。Adapt: kernel 选择需要根据 GNN embedding 的 dimensionality 调，我们倾向用 RBF kernel + median heuristic for bandwidth。Not use: HSIC 的 unbiased estimator 形式（O(n²) 计算更复杂，且在小 batch 下方差大）。

### 2.2 Gretton et al. — 后续 HSIC 扩展工作 (2008 NeurIPS, "A Kernel Statistical Test of Independence", verify before paper submission)

(a) 这一系列工作把 HSIC 发展成 hypothesis test，给出 null distribution 近似（Gamma approximation 或 permutation test），并讨论了 kernel choice 对 power 的影响。

(b) 对 Paper B：我们不做 hypothesis test，但这些 paper 提供了 kernel choice 和 bandwidth 选择的实践指导。

(c) Adopt: median heuristic for RBF bandwidth。Not use: hypothesis testing pipeline、null distribution 估计，因为 Paper B 要的是可微 penalty，不是 p-value。

### 2.3 Long, Cao, Wang, Jordan (ICML 2018) — "Conditional Adversarial Domain Adaptation" (CDAN)

(a) CDAN 是 adversarial domain adaptation 的代表作之一。它在 DANN (Ganin & Lempitsky 2015) 基础上，把 adversarial discriminator 的输入从 feature 单独，改成 feature 与 classifier output 的外积——使 alignment 同时考虑 feature distribution 和 conditional label distribution。Loss 包含 source classification loss + adversarial loss with gradient reversal。 (Verify whether the lead authors are exactly Long-Cao-Wang-Jordan and whether it is ICML 2018 or NeurIPS 2018 — I recall ICML 2018 but please double-check before submission.)

(b) 对 Paper B：CDAN 是 HSIC 路线的 adversarial alternative。HSIC 用 kernel 度量独立性，CDAN 用一个对抗 discriminator 度量。两条路线都试图 "让某个 representation 在两个 domain 上 indistinguishable"。Paper B 选 HSIC 而非 adversarial 的主要原因是：adversarial training 在 RUM likelihood 这种结构化目标上 optimization 不稳定，且 GNN 已经够慢，不想再加 inner-loop discriminator。

(c) Adopt: 把 CDAN 作为 baseline 写入实验对比章节（"V-REx vs HSIC vs CDAN"）。Not use: 作为主方法。Not adapt: conditioning on classifier output 的 trick——在 RUM 里，"classifier output" 是 choice probability，已经被 likelihood 涵盖。

---

## Section 3 — Causal representation learning 视角

V-REx 和 HSIC 都可以从 causal invariance 视角统一理解。这一节梳理两篇相关 paper，为 Paper B 的 motivation 提供理论 framing。

### 3.1 Peters, Bühlmann, Meinshausen (2016) — "Causal inference by using invariant prediction"

(a) 这篇 JRSS-B (Series B) 2016 paper 提出 ICP (Invariant Causal Prediction)：寻找一个变量子集 S，使得 conditional distribution P(Y | X_S) 在所有 environment 上都不变；该 S 在适当假设下对应 Y 的 causal parents。方法基于多 environment 残差检验。

(b) 对 Paper B：Peters et al. 是 "invariance ⇒ causality" 这一 motif 在统计学界的奠基。Paper B 的论证逻辑——"在 London + Beijing 都成立的 (β, γ) 更接近真实的 commuter preference"——本质上是 ICP 思路的 mode choice 版。

(c) Adopt: invariance-as-causality 的 motivation framing，写在 Paper B intro 的 "why invariance" 段。Not use: ICP 的具体 testing procedure（残差独立性检验），因为我们要的是可微 penalty 而非 hypothesis test。Adapt: ICP 假定离散 environment + linear SCM；我们的 setting 是 nonlinear (GNN) + 两个 environment，所以借用 motivation 而不借用机制。

### 3.2 Schölkopf, Locatello, Bauer, Ke, Kalchbrenner, Goyal, Bengio (Proc. IEEE 2021) — "Toward Causal Representation Learning"

(a) 这篇 Proceedings of the IEEE 2021 综述把 ML 中的 representation learning 与 SCM 视角结合，提出 "independent causal mechanisms (ICM)" 和 "sparse mechanism shift" 等概念，论证 distribution shift 在 causal 层面通常是 sparse / single-component 的。文中明确把 IRM、V-REx、disentanglement 等放到统一框架下解释。

(b) 对 Paper B：Schölkopf et al. 2021 给我们提供了 "为什么 invariance penalty 应该在 GNN 中间层而非 final layer" 的论证——只有那些反映 invariant causal mechanism 的 component 应该被强制 invariant，与 city 相关的 idiosyncratic component（房价水平、行政边界）应保留。这指导我们 penalty 的 placement。

(c) Adopt: ICM 概念用作 Paper B 的 motivation，特别是 "preference parameters (β, γ) 是 invariant mechanism，spatial network 是 environment-specific" 这一 split。Adapt: sparse mechanism shift 假设在我们 setup 下表现为 "GNN θ 中只有 spatial-structure 相关的 component invariant"。Not use: 该综述本身是 conceptual paper，不提供具体 algorithm，所以 "not use" 的是空集。

---

## Section 4 — Cross-city / spatial transfer applications

这一节梳理 transportation / urban computing 中已有的 cross-city / spatial transfer 工作，作为 Paper B 的 application baseline。

### 4.1 Wang, Liu, Zhang (2022) 或类似 cross-city transportation deep learning paper

(a) 我对 Wang/Liu/Zhang 2022 这篇具体论文的细节不确定（**verify before paper submission**：可能是 KDD 2022 或 IEEE TITS 2022 上的一篇 cross-city traffic prediction paper，常见做法是用 meta-learning 或 fine-tuning 把 source city 的 GCN 模型迁移到 target city）。一般这类 paper 提出 source-target adaptation framework，使用 MAML-style meta learning 或 adversarial domain adaptation 缓解少 target data 问题。

(b) 对 Paper B：cross-city transportation deep learning 文献为 Paper B 提供了 application context——证明 "cross-city transfer" 是一个被 community 关注的实际问题。但已有工作普遍是 prediction-oriented (next-day flow, traffic speed)，而 Paper B 是 inverse-oriented (recover preferences)，所以 contribution 角度不同。

(c) Adopt: 把这条 line 作为 application motivation 写进 intro。Adapt: 我们的 transfer 不是为了在 target city 做 better prediction，而是为了 recover 一组 city-invariant preference parameters——这个 framing 与已有 cross-city work 不同，需要在 related work 里清楚区分。Not use: meta-learning / fine-tuning 范式，因为我们恰恰要避免 "用 Beijing data fine-tune London model" 这种破坏 invariance 的做法。

### 4.2 Yao, Gao, Zhu, Manley, Wang, Liu (IEEE TITS 2021) — Beijing 30×30 GCN imputation

(a) 这篇 IEEE TITS 2021 paper（**verify exact title and author order before paper submission**——我记得是关于用 GCN 在 Beijing 30×30 grid 上做 OD flow imputation 的工作）使用 graph convolutional network 在稀疏 OD 矩阵上做补全，以 Beijing 出租车 / 公交数据为 testbed。

(b) 对 Paper B：这是已经在 user 文献库里的 Beijing 侧 baseline（参见 user MEMORY: project_beijing_constraint.md）。Paper B 的 Beijing 数据处理 pipeline 部分可以借用 Yao et al. 的 30×30 grid 划分。

(c) Adopt: 30×30 grid 作为 Beijing 侧 spatial unit；Yao et al. 的 imputation 结果可作为 Beijing OD missing data 的预处理工具。Adapt: 我们的 task 不是 imputation，是 inverse choice，所以 GCN 架构需要重新设计。Not use: Yao et al. 的 supervised loss——我们用 RUM likelihood。

---

## Section 5 — Synthesis: 哪些组合最适合 Paper B

综合以上文献，Paper B 的 setup 是 **inverse RUM + GNN + aggregate OD on two environments (London, Beijing)**。在这个 specific configuration 下：

**主推方案：V-REx + HSIC 双 penalty 组合。** V-REx 在 risk-level 强制 NLL 在两 city 上方差小，保证 (β, γ) recoverability 的 city-invariance；HSIC 在 GNN embedding 层强制 representation 与 city label 独立，保证 GNN θ 学到的是 city-invariant spatial mechanism 而非 city-specific shortcuts。两者互补：V-REx 是 outcome-level invariance，HSIC 是 representation-level invariance。这一组合对应 Schölkopf et al. 2021 中 "invariant mechanism + environment-specific noise" 的分解 motif，且回避了 IRMv1 的 optimization 不稳定（Rosenfeld 2021）和 CDAN 的 adversarial inner loop 开销。从 Peters et al. 2016 的 ICP motivation 出发，Paper B 的 contribution 可以 framed 为 "把 ICP 思路扩展到 nonlinear RUM + GNN + 仅两 environment 的 setting"。在 application 层面，相比 Wang 2022 / Yao 2021 这类 prediction-oriented cross-city work，Paper B 的 inverse-oriented framing 是新的贡献点；Beijing-side data pipeline 借用 Yao et al. 的 30×30 grid 但 task 完全不同。**风险点**：两 environment (|E|=2) 时 V-REx 退化为 squared difference，可能过 weak；建议在 ablation 中也 report HSIC-only、V-REx-only、joint 三种 variant，用以确认双 penalty 的必要性。

---

## Notes on citation accuracy

以下 citations 我对细节不完全确定，**请在投稿前 verify**：

- **Arjovsky et al. 2019 (IRM)**：arXiv preprint，未确认是否最终 publish 在 conference。
- **Gretton et al. HSIC**：原始 ALT 2005 vs 后续 NeurIPS 2007/2008 vs JMLR 2008——具体哪篇是最 canonical citation 取决于上下文，建议引用最早的 ALT 2005 + JMLR 2008 expanded version。
- **Long et al. CDAN**：ICML 2018 还是 NeurIPS 2018 我不确定，author order 也建议 verify。
- **Wang/Liu/Zhang 2022**：这是一类 paper 的代表，具体哪一篇最贴 Paper B 需要 user 自己 confirm，可能需要替换为 Wang et al. 2019 (KDD) "Cross-City Transfer Learning for Deep Spatio-Temporal Prediction" 之类更明确的 citation。
- **Yao et al. 2021 IEEE TITS**：title 和 author order 来自 user MEMORY，但具体细节请以 user bibliography 为准。
- **Peters Bühlmann Meinshausen 2016**：JRSS-B 应该正确，但 page numbers 未给出。
- **Schölkopf et al. 2021**：Proc. IEEE，volume/issue 未给出。

未引用但 Paper B 写作中可能需要补充的文献：
- Pearl & Bareinboim 2014 on **causal transportability**（user 的可选第 10 篇）——若 Paper B 需要更强的 theoretical framing，建议读 Bareinboim & Pearl 2016 PNAS "Causal inference and the data-fusion problem"，明确 "transport formula" 概念。
- Rosenfeld, Ravikumar, Risteski 2021 ICLR "The Risks of Invariant Risk Minimization"——IRMv1 critique，Paper B 需要在解释 "为什么不用 IRMv1" 时 cite。
- Ganin & Lempitsky 2015 ICML "Unsupervised Domain Adaptation by Backpropagation" (DANN)——adversarial DA 的 baseline，CDAN 的前身。
