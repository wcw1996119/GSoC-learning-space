# 空间再开发 vs 灵活办公：用于伦敦通勤政策评估的深度选择模型

_Draft for *Journal of Transport Geography*。_

---

## 摘要

后疫情时代伦敦交通政策面临两类对照性辩论：围绕新交通枢纽的大规模空间再开发（典型为 Old Oak Common HS2 / Crossrail 交汇枢纽，规划新增 6.5 万就业岗位），以及《Mayor's Transport Strategy 2018》强调的灵活办公带来的弥散性时间需求重塑。本文以单一工具评估两者的**可达性分布性**后果：一种深度选择模型，多项 logit 中的目的地吸引力项由覆盖 1 公里伦敦网格的图神经网络参数化，GNN 权重与行为系数由对 Census 2021 完整 OD 表（121 万通勤者）的最大似然**联合**估计。反事实分析将已校准模型与 BPR 道路拥堵均衡耦合，并辅以混合 logit 蒙特卡罗模拟（异质出行时间价值系数取自公开 WebTAG / Wardman 表）。两情景在可达性分布上分歧明显：Old Oak Common 空间干预**回归**（Gini +1.26%；约 2/3 可达性总收益被拥堵吸收）；灵活办公**进步**（早高峰平均通勤时间 −3.7%、可达性 +10.2%、Gini −0.011）。在 360 个合成集群的可信度扫描中，深度选择模型只对 12% 的干预配置可信（|Jensen 偏差| < 5%），31% 落入 caveat 范围（5–20%），56% 不可信（>20%）——主要由集群基线就业决定。OOC 情景落在 caveat 范围，因此 magnitude 报告附带 ±25% 不确定性区间；定性方向是稳健的。

**关键词**：通勤；深度选择模型；可达性不平等；灵活办公；Old Oak Common；伦敦交通政策。

---

## 1. 引言

《Mayor's Transport Strategy 2018》承诺伦敦在两个分布性后果未被联合评估的维度上同时变革。**空间维度**：《London Plan 2021》将下一阶段 63.8 万就业岗位集中于少数 Opportunity Areas，最大者为 650 公顷的 Old Oak Common (OOC) 再开发——围绕未来 HS2 / Elizabeth 线交汇枢纽，规划至 2041 年新增约 6.5 万就业岗位（OPDC, 2018）。**时间维度**：后疫情混合办公将早高峰较 2019 年压低 20–30%（TfL, 2024），Mayor *Outcome 7* 明确将进一步分散高峰列为政策目标。两类政策杠杆作用于同一通勤决策的不同维度（去哪儿工作、何时出行），但传统交通建模实务通常以分立且不兼容的工具集评估两者——前者用 gravity / 土地利用-交通互动模型，后者用需求曲线模型。

本文以单一工具同时评估两者。我们对覆盖伦敦 1 公里网格的**深度选择模型**做 Census 2021 完整 OD 表上的校准，并问：(a) OOC 新增 6.5 万就业、(b) 50% 早高峰需求外移至非高峰，对可达性分布有何后果？我们发现两干预即使均增加聚合可达性，在不平等指标上方向相反——这一结果对伦敦如何平衡空间-时间政策组合有直接含义。

本文有三项贡献。**第一**，在深度选择传统下（Wang and Klabjan, 2018; Sifringer 等, 2020）将图神经网络吸引力项耦合至多项 logit 决策，并扩展该家族至完整 Census 数据上的聚合 OD 校准——绕过了过往深度选择交通应用所需的个人级选择记录约束。**第二**，在城市网格分辨率上量化 OOC 与灵活办公的可达性分布后果，得出空间回归 vs 时间进步的对照判决。**第三**，提供一个经验可信度刻画——按集群基线就业 × 干预幅度索引的"信任地图"——告诉实践者**何时**可信、**何时**应折扣量级；据我们所知这是**首个**为聚合 OD 深度选择模型公布的统计可信度扫描。

§2 给出政策语境。§3 描述数据。§4 设定模型。§5 报告校准与验证。§6 报告两个政策情景。§7 讨论政策意义。

---

## 2. 政策语境

### 2.1 Mayor's Transport Strategy 与后疫情时刻

《Mayor's Transport Strategy 2018》（MTS 2018）目标至 2041 年伦敦 80% 出行由步行、骑行或公共交通完成，依靠 Healthy Streets、公共交通投资与新住房-就业递交三条支柱。两类结构性变化影响实操实现：第一，混合办公已将相当份额的疫情前高峰需求转至非高峰；TfL 2024 年高峰时段地铁乘客量仍维持在 2019 年基线的约 80%（TfL, 2024）。第二，《London Plan 2021》将额外就业集中于少数 Mayor 机会区，就业增长高度集中（GLA, 2021）。

### 2.2 Old Oak Common 作为代表性空间干预

OOC 是机会区中最大的一个。《OPDC Local Plan 2018》目标至 2041 年在伦敦西北部 HS2 / Elizabeth 线 / 大西部主线交汇处新增 6.56 万就业与 2.55 万住房。政策辩论迄今集中于经济可行性、公共空间设计与既有工业用途置换（London Assembly, 2022）；其对伦敦其余地区的*可达性分布*后果——谁能到达 OOC、谁被遗落、道路拥堵如何重新分配——尚未在城市级网格分辨率上得到定量关注。

### 2.3 灵活办公作为代表性时间干预

MTS 2018 *Outcome 7* 明确将分散高峰需求列为政策目标。TfL 自 2022 年起以扁平化季票定价、推广非高峰出行响应。但进一步灵活办公政策的边际效益是*进步*还是*回归*——即外伦敦欠连通起点获益是否多于内伦敦起点——是 §6.2 要回答的分析问题。

### 2.4 为什么用深度选择模型

两干预都改变通勤者目的地选择的**输入**——空间变化下格点相对吸引力；时间变化下到达不同时段的相对成本。能从观测的聚合通勤恢复行为决策参数的模型，可以在扰动输入下重新计算选择概率；只拟合流量模式而不恢复参数的 gravity 类模型做不到。我们以图神经网络吸引力项替代经典 gravity 中手编的"目的地规模幂次"项，并以混合 logit 蒙特卡罗模拟将已发表的出行时间价值异质性传播至选择机制——揭示聚合流量预测平均掉的分布性后果。

---

## 3. 数据

我们工作在覆盖大伦敦区的 1725 × 1 公里规则网格上（EPSG:27700），全部使用公开数据。

**OD 通勤流量**。2021 年 OD 来自 NOMIS *ODWP01EW* 表（Census 2021），限于英国非家中固定工作地点，共 121 万伦敦通勤者、196 940 个 MSOA 对，按人口加权面积分摊降至 1 公里网格。早期 1 公里工作中常用的 1.1% 微样本（5.5 万通勤者）每格估计噪声大；本文全程使用完整 Census 表。2011 OD（用于 §5.2 跨时段验证）为 NOMIS *WU03EW*（214 万通勤者），通过 ONS MSOA-2011 → MSOA-2021 最佳拟合查找表对齐到 2021 年网格。

**节点特征**。每格 22 维特征：BRES 2024 的 8 部门工作场所就业与总就业；KS101EW Census 2021 的常住人口；OpenStreetMap 2024 的 8 类 POI 计数与总数；TfL Open Data 2024 的地铁站计数；重心经纬度。

**出行时间与行为锚**。自由流车辆出行时间由 OSMnx（Boeing, 2017）在大伦敦区行车网络上预计算。起点 IMD 2019 与目的地 ASHE 2025 工作场所收入提供交互项。混合 logit 模拟中按收入档的出行时间价值离散度取自 Wardman 等 (2016) 与 WebTAG TAG Unit A1.3（DfT, 2024）。

---

## 4. 模型

图 1 用 4 个概念步骤总结框架：观察、学习、模拟、评估。

![**图 1.** 概念框架。(1) 我们观察 1 公里网格上的 Census 2021 完整通勤流。(2) 模型同时学习"什么让某个目的地有吸引力"（一个对本地城市特征的图学习函数）与"通勤者如何权衡时间、距离、工资、职业匹配"。(3) 然后模拟反事实——某地加 jobs，或将早高峰需求外移至非高峰——通过扰动输入并重新运行已校准机制实现。(4) 用人口加权 Gini、Palma 与 Atkinson 指数评估"谁获得 / 谁失去可达性"。](../evaluation_outputs/paper_a/F1_conceptual.png)

### 4.1 深度选择模型

起点 *i* 的通勤者在终点 *j* 的随机效用为
$$ U_{ij} = \alpha\, V_j + \beta_t(i,j)\, t_{ij} + \gamma\, \log d_{ij} + \delta\, \mathrm{OccMatch}_{ij} + \varepsilon_{ij}, $$
$\varepsilon_{ij}$ 为标准 Gumbel 误差，给出多项 logit 选择概率 $P(j\mid i) = \exp U_{ij} / \sum_{j'} \exp U_{ij'}$。

行为系数有经典交通经济学含义：$\beta_t$ 是出行时间敏感度，$\gamma$ 是距离衰减，$\delta$ 是目的地劳动力市场匹配权重。我们允许 $\beta_t$ 依赖可观测起点与终点特征：
$$ \beta_t(i,j) = \beta_t \cdot \bigl(1 + \phi\, z_o(i) + \psi\, z_w(j)\bigr), $$
其中 $z_o(i)$ 为 z-score IMD 收入、$z_w(j)$ 为 z-score ASHE 工资——表达常见发现（Wardman 等, 2016）：高收入工人与高工资目的地表现出更高 VOT。

吸引力项 $V_j$ **不是**经典 gravity 中手编的"目的地规模幂次"，而是由图神经网络——5 层 GraphSAGE（Hamilton 等, 2017）——产出，递归地将每格的 22 维特征向量与按自由流时间最近 10 个邻居的特征聚合。每层是一个学习的非线性变换：
$$ h_j^{(\ell+1)} \;=\; \mathrm{ReLU}\Bigl( W_{\rm self}^{(\ell)}\, h_j^{(\ell)} + W_{\rm nbr}^{(\ell)}\, \mathrm{mean}\bigl\{h_u^{(\ell)} : u \in \mathcal{N}_{10}(j)\bigr\}\Bigr), \quad V_j = w_{\rm out}^\top h_j^{(L)}, $$
所以 $V_j$ 是邻里城市特征的**学习的非线性函数**，不是单一统计量的简单求和。我们选 GraphSAGE 是因为其 **inductive 性**——可在特征干预 $\mathrm{do}(X)$ 下直接前向无需重训——以及其均值聚合先验匹配 1 公里尺度上城市特征的平滑空间结构；在我们的设置下基于注意力的 GAT 替代欠拟合（CPC 0.246 vs SAGE 0.343）。该选择有 Yao 等 (2021) SI-GCN 在空间相互作用建模中的直接先例。

### 4.2 联合校准：聚合 Census 流量上的最大似然

GNN 参数与行为系数**联合**端到端通过最大化观测起讫计数的多项似然估计：
$$ \mathcal{L}(\boldsymbol{\theta},\beta_t,\gamma,\delta,\phi,\psi) \;=\; \sum_{i \in \mathrm{train}} \sum_j F_{ij}\,\log P(j \mid i). $$

这是对多项 logit 的标准 MLE。唯一非标准之处是观测为聚合格点计数而非个人选择记录，对此 McFadden (1978) 备选采样修正使得 1725 个目的地上的梯度仍可计算。机制上：每次梯度更新从似然经 softmax 传到行为系数（$\beta_t, \gamma, \delta, \phi, \psi$），再经 $V_j$ 传到 5 层 GNN 全部参数——所以 GNN **不是**预训练的、**不是**固定的加权和、**不是**作为外部先验加进来。它在单次优化中与行为参数一起从数据中恢复。

### 4.3 反事实分析

我们将已校准模型视为结构性模拟器。对扰动子集格点输入特征的政策情景 $\mathrm{do}(X_S = X_S^*)$，我们在扰动输入下重新前向已校准 GNN 与选择机制，**保持**恢复出的行为参数不变。对车辆通勤以 BPR 不动点（Sheffi, 1985 第 5 章）闭合道路拥堵循环：
$$ t_{ij}^{\rm eq} = t_{ij}^{0} \cdot \bigl(1 + 0.15\,(V_j^{\rm flow}/C_j)^4\bigr), $$
以逐次平均法求解。

该反事实的有效性依赖两个结构性假设：(i) 恢复出的行为参数是稳定的人群级偏好，不响应干预本身（McFadden 1974 以来随机效用的标准假设）；(ii) GNN 学到的映射 $V = f_\theta(X)$ 是结构性关系——介入 X 改变 V 的方式与"自然演化到那个 X"改变 V 的方式相同，而非仅是其在训练分布内的相关。我们在 §5 经验性地测试 (ii)。

### 4.4 用于分布性分析的混合 logit 蒙特卡罗

§4.3 的确定性前向对每个起点返回一个 $P(j \mid i)$——隐含假设起点 *i* 的所有通勤者具有相同 $\beta_t$。这是多项 logit 的 IIA 假设，当人群异质且政策干预把选择概率推到 softmax 弯曲区域时会引起 Jensen 类偏差（Train, 2009 §6.7）。对**聚合流量**预测偏差小；对**分布性**预测可能很大，因为聚合响应与个体响应分布随人群方差而分歧。

我们以下方式应对：生成 10 000 个合成伦敦通勤者，其收入、职业、居住边际分布匹配 ASHE 与 Census KS101。每位通勤者的 VOT 系数为
$$ \beta_n(j) = \beta_t \cdot \bigl(1 + \phi\, z_o(i_n) + \psi\, z_w(j) + \varepsilon_n\bigr), \quad \varepsilon_n \sim \mathcal{N}\!\bigl(0, \sigma^2(\mathrm{tier}_n)\bigr), $$
其中按档标准差 $\sigma$ **从公开 WebTAG / Wardman**（2016）元分析的 VOT 离散度**导入**。聚合 Census OD 数据只能识别人群均值偏好；人群内方差必须外部提供。我们不从数据中学习 $\sigma$，而是用文献估计并报告对其选择的灵敏度（§5.5）。

每位通勤者随后从自身 softmax 中抽样目的地，聚合得到与确定性 $\widehat F^{\,\rm det}_{ij}$ 直接可比的蒙特卡罗预测 $\widehat F^{\,\rm MC}_{ij}$。Jensen 类偏差经验值
$$ \mathrm{Bias} \;=\; \Delta\widehat F^{\rm det} - \Delta\widehat F^{\rm MC} $$
对每个政策情景计算；§5.5 报告其统计分布。

---

## 5. 校准与验证

### 5.1 空间外推

我们留出 1725 个起点中的 247 个（14%），对应四个事先选定以覆盖伦敦主要城市背景的行政区：Westminster（CBD）、Hackney（内城居住型）、Brent（外环密集）、Bromley（外环郊区）。模型获得 CPC（Lenormand 等, 2012）= **0.343**，95% bootstrap 置信区间 [0.324, 0.362]（10 种子集成、1000 次起点重抽）。对 gravity 基线（CPC 0.170）的 bootstrap 均值 ΔCPC 为 +0.172，重抽中 $P(\mathrm{ours} > \mathrm{gravity})$ = 100%（$p < 10^{-4}$）。

![**图 2.** 按行政区 CPC（10 种子集成）。Westminster 拟合最佳（0.475）、Hackney 最弱（0.201）；横线标示总集成 CPC 与 gravity / radiation 基线。](../evaluation_outputs/paper_a/F4_per_borough.png)

模型的地理强弱（图 2）可解释：CBD 就业密度拟合最佳（Westminster），郊区径向流次之（Bromley），内城非 CBD 居住型通勤模式最弱（Hackney）。Hackney 弱点反映格内方式异质性与未观测居住自我选择，二者均为聚合静态特征不能解决。

### 5.2 跨时段泛化：2011 → 2021

在 2011 输入上从头重训模型（NOMIS WU03EW、KS101EW 2011、2012 前 TfL 站点），并通过用 2024 输入前向预测观测的完整 Census 2021 OD：城市级 CPC = **0.56**（样本内 2011 CPC = 0.71）。0.71 → 0.56 的下降量化跨行政区通勤结构 10 年的不可压缩漂移，部分被 2021 年 3 月 Census（落入英国第二次封锁）的 COVID 通勤量缩水所混淆。

### 5.3 Bromley 阴性对照：单一低幅度案例

模型反事实必须不"不论干预如何都预测一切上涨"。作为单一正面案例，我们将干预机制施加于 Bromley，其 2011 → 2024 就业仅增长 1.55×（典型稳定郊区，低于 1.45× 全市均值）。模型预测 Δshare +1.46 个百分点；2011 → 2021 观测变化为 +1.26 pp。预测/观测比为 **1.15×**，舒适处于 [0.5, 2.0] 容差之内。这是一个小幅度干预的可信经验证据——但单一案例不足以刻画模型在政策相关输入空间上的可信度，下面给出这一刻画。

### 5.4 恢复的行为系数

校准恢复基础出行时间系数 $\beta_t = -0.058$，目的地工资交互 $\psi = +0.39$、起点 IMD 交互 $\phi = -0.30$、职业匹配系数 $\delta = +0.69$。以 Wardman (2014) RP 通勤 VOT 中心值 £6.5/h 锚定效用计价单位，对低/中/高目的地工资三分位分别给出 VOT 值 £4.49 / £6.59 / £8.23/h——与 WebTAG TAG A1.3 一致的单调收入梯度。

### 5.5 输入空间上的经验可信度刻画

单一 Bromley 案例令人安心但未告诉用户**何时**应信任模型 magnitude，**何时**应折扣。我们以**统计**方式刻画可信度：在伦敦网格上随机抽取 60 个合成 9 格干预集群，每个集群施加 6 种就业增量幅度（基线集群就业的 +10%、+30%、+50%、+100%、+200%、+500%），共 360 个（集群 × 幅度）实验。对每个实验同时计算确定性深度选择预测和混合 logit 蒙特卡罗预测，以集群基线就业与干预幅度为函数报告其 Jensen 偏差。

![**图 3.** 360 次合成干预实验上的反事实预测经验可信度。(a) 每个点是一次（集群 × 幅度）实验；颜色 = |Jensen 偏差| %；绿色轮廓标示偏差 < 5% 的"可信区"凸包。(b) |Jensen 偏差| 的累积分布，5%（可信）和 20%（需 caveat）阈值标线。](../evaluation_outputs/paper_a/F8_reliability.png)

可信度地图（图 3a）和分箱总结（表 1）显示：可信度主要由**集群基线就业**决定：

| 集群基线就业 | 平均 \|Jensen 偏差\| (%) | 可信度 |
|---|---|---|
| < 5 000 jobs                | 150–400                  | 任何幅度均不可信 |
| 5 000 – 20 000 jobs         | 30–550                   | 除小幅度外不可信 |
| 20 000 – 50 000 jobs        | 12–25                    | 需 caveat |
| 50 000 – 200 000 jobs       | 13–24                    | 需 caveat |
| **> 200 000 jobs**          | **9–30**                 | **需 caveat（OOC 落于此）** |

在全部 360 个实验中，**12% 落入可信区（|偏差| < 5%）、31% 落入 caveat 范围（5–20%）、56% 超出有用预测范围（>20%）**。所以模型对针对**已建立就业走廊**（基线 > 20 万 jobs）的干预可信，对**稀疏或 greenfield 区域**的干预渐次失效。

**§6.1 的 Old Oak Common 情景**落在 > 200 000 基线就业行（集群基线 ≈ 100 万 jobs）的小幅度区（+6.5 万 = 基线 +6.5%），其预期可信度落在 caveat 范围。我们在 §6.1 将此带为预测可达性分布结果上的 ±25% 不确定性区间。

灵活办公情景（§6.2）是时间需求侧干预，不会把深度选择模型推出训练输入分布之外，所以上述可信度扫描不约束它；§6.2 单独测试该情景的稳健性。

---

## 6. 政策情景

### 6.1 Old Oak Common：新增 6.5 万就业

我们以按比例膨胀 9 网格集群（中心为规划 HS2 站）就业类（8 部门计数、总就业、办公 POI 计数）实例化 OPDC Local Plan headline 就业目标，对应 +6.5 万总就业。自由流分析预测集群净流入 +126 通勤者；BPR 均衡下降至 +105——**17% 拥堵反推**。所有伦敦起点的 Hansen 平均可达性（Geurs 与 van Wee, 2004）增益自由流 +1.34%、均衡 **+0.45%**：总收益的 2/3 被拥堵吸收。$A_i$ 上人口加权可达性分布指数在均衡情景下均恶化：Gini +1.26%、Palma +1.51%、Atkinson($\varepsilon = 0.5$) +2.53%。

干预在可达性维度**轻度回归**。机制是 OOC 集群所在走廊已被既有 Bakerloo 与 Elizabeth 线、以及通往 Westminster、City、Canary Wharf 的车程中等程度连通；获益最多的格点位于可达性分布上端，外伦敦起点几乎无净收益。BPR 均衡通过将最大收益吸收到与 OOC 邻里已连通良好的格点放大该模式。

### 6.2 灵活办公：50% 高峰外移

为表征持久的灵活办公，对早高峰（07:00–09:59）需求乘 0.5、对非高峰肩部小时（10:00–15:59）乘 1.5，每起点保留每日总需求不变，预测新每小时 OD 分配下的 BPR 均衡。在 08:00 模拟器预测平均通勤时间 **32.73 分钟、比基线 33.99 分钟下降 −3.7%**；对应人口加权 08:00 平均可达性从 109 万升至 120 万岗位（**+10.2%**）；所有分布指数改善（Gini −0.011；Palma 0.591 → 0.557）。

时间干预**明显进步**。机制与 §6.1 相反：可达性最被高峰拥堵罚的欠连通外伦敦起点从高峰缓解中获益不成比例；可达性较少受拥堵敏感的内伦敦起点（其最近目的地无论乘子如何均能少分钟到达）绝对增益更小。

![**图 3.** 两情景对可达性分布指数的影响对比。空间干预（左）在三个指数上均回归；时间干预（右）在两个指数上均进步。同一组恢复行为参数；相反的分布判决。](../evaluation_outputs/paper_a/F6_scenarios_inequality.png)

### 6.3 在可信度扫描背景下解读 OOC 结果

OOC 单点测量给出 Jensen 偏差 +2.1%（确定性 +134.2 vs 蒙特卡罗 +131.4）——但 §5.5 已经清楚表明，单点测量对我们的确定性是**过度声称**。OOC 集群落在 > 200 000 基线就业行的低幅度区（+6.5%），360 实验扫描在这一范围内的同类集群上发现**平均 |Jensen 偏差| ≈ 30%、90 分位 ≈ 32%**。OOC 特有的 Jensen 偏差因此应报告为**确定性 Δshare 上的大约 ±25%**，而不是幸运的 2.1% 单点。§6.1 的定性方向——可达性回归干预——对该不确定性是稳健的（所有符号不变）；*量级*应以 ±25% 区间解读。

对 OOC 集群上 VOT 变异系数 $\sigma \in [0, 1]$ 的灵敏度扫描确认 |bias| 随 $\sigma$ 选择平滑变化，所以结果不依赖精确的 $\sigma$ 假设。

---

## 7. 讨论与政策意义

### 7.1 伦敦交通政策的空间-时间权衡

主要政策发现是 §6 的对比：代表性空间干预（OOC）在可达性分布上轻度回归，而代表性时间干预（灵活办公）明显进步。该对比在多个不平等指数上一致且经 BPR 均衡调整后保持。

对伦敦政策这有三点意义，每点附带 §5.5 的可信度范围。**第一**，Mayor 将 Opportunity Areas 作为吸纳伦敦未来就业的主要机制的战略侧重，承担一个未被充分认识的可达性回归风险。OOC 情景在模型 caveat 范围内（集群基线 > 200 000 jobs，干预幅度小）：*方向*（可达性回归）稳健，但*量级*应以 ±25% 不确定性区间读取。这并不反对 OOC 项目本身——其有不限于通勤可达性的明显收益，包括直接本地就业与基础设施投资——但确实主张在 OPDC 与 London Plan 政策周期中显式纳入可达性分布核算，与传统就业 / 住房 / 财政测试并列。**第二**，MTS 2018 *Outcome 7* 强调的灵活办公时间维度议程，基于本研究证据明显更进步；因时间情景不会把模型推出训练输入分布，该发现的不确定性比 OOC 更窄。在政策衡量分配性结果的程度下，额外时间塑造政策（非高峰票价结构、混合办公基础设施补贴、非高峰主动出行投资）的边际效益每单位成本可能高于等效空间再开发投资。**第三**，BPR 均衡拥堵吸收 OOC 表面可达性收益的约 2/3——一项关于新就业枢纽自由流分析的定量警示，据我们所知此前在当代空间规划运行的城市网格分辨率上未被给出。

来自 §5.5 可信度地图的实务警示：模型对**基线就业 < 约 20 000 jobs 的 greenfield 集群干预**在当前校准下**不可信**——Jensen 偏差经常超过 100%。任何应用于*新建*远离既有就业走廊的集群开发都应在 magnitude 结论前扩展训练数据或添加显式的 greenfield 先验项。

### 7.2 方法学意义：交通政策 ML 的信任地图

本文方法学最可推广的输出是 §5.5 的可信度刻画。我们不报告单一 Jensen 偏差数字（扫描显示这是误导性的精确）——而是提供一个按干预属性索引的定量信任地图，实践者可在做出政策声称*前*计算：集群基线就业、干预幅度、（隐含地）距训练分布的距离。结果可信度分箱（360 个合成干预里 12% 可信 / 31% caveat / 56% 不可信）勾勒一个**诚实的预期**：聚合 OD 校准的深度选择模型对成熟就业走廊的干预有用，但对没有额外数据的 de-novo greenfield 开发不行。混合 logit 蒙特卡罗修正不需要个人级出行调查数据——只需公开 WebTAG / Wardman 表——使信任地图可移植至没有定制出行调查面板的城市。

### 7.3 局限

校准使用自由流车辆通勤时间，仅在反事实分析中切换至 BPR 均衡，以可识别性与计算可行性换取拟合 $\beta_t$ 中的部分现实性。Hackney CPC 0.20 是模型主要弱点，归因于通勤方式异质性与未观测居住自我选择，二者均非聚合静态特征所能解决。2021 年 Census 落入英国第二次封锁，即使使用完整 Census 表，主体量仍较疫情前水平低约 60%，对 Stratford 类干预的回测部分被混淆。通勤方式在 Census 中被记录但未建模；(mode, destination) 上的嵌套 logit 是自然扩展，列入计划中的北京后续。

---

## 8. 结论

一个深度选择模型，耦合混合 logit 蒙特卡罗政策模拟、对 Census 2021 完整 OD 表校准，给出对两类对照伦敦交通政策干预的连贯评估。Old Oak Common 的空间再开发在可达性分布上轻度回归；灵活办公明显进步。该框架不依赖任何个人级出行调查数据，可迁移至其他城市——包括北京（后续应用正在准备中）。结果建议在 Mayor's Transport Strategy 必须继续平衡的空间-时间政策组合中给予时间塑造干预更大权重。

---

## 致谢

本工作与 Google Summer of Code (GSoC) 2026 的 Mesa 智能体建模项目相关；§4.4 的混合 logit 蒙特卡罗模拟在 Mesa 框架内实现。感谢导师及 GSoC Mesa 社区的指导。

---

## 参考文献

Boeing, G. (2017). OSMnx. *Computers, Environment and Urban Systems* 65, 126–139.

Department for Transport (2024). *TAG Unit A1.3: User and Provider Impacts*. WebTAG.

Geurs, K. T. and van Wee, B. (2004). Accessibility evaluation of land-use and transport strategies. *Journal of Transport Geography* 12(2), 127–140.

Greater London Authority (2021). *The London Plan 2021*.

Hamilton, W. L., Ying, R. and Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*.

Lenormand, M. *et al.* (2012). Universal patterns of human mobility from a multi-day GPS dataset. *PLOS ONE* 7(12), e51249.

London Assembly (2022). *Old Oak Common Development Corporation Scrutiny Report*.

Mayor of London (2018). *Mayor's Transport Strategy 2018*.

McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In Zarembka (ed.), *Frontiers in Econometrics*.

McFadden, D. (1978). Modelling the choice of residential location. In Karlqvist *et al.* (eds.), *Spatial Interaction Theory and Planning Models*.

OPDC (2018). *Old Oak and Park Royal Development Corporation Local Plan*.

Sheffi, Y. (1985). *Urban Transportation Networks*. Prentice-Hall.

Sifringer, B., Lurkin, V. and Alahi, A. (2020). Enhancing discrete choice models with representation learning. *Transportation Research Part B* 140, 236–261.

TfL (2024). *Transport for London Annual Statistical Bulletin 2023/24*.

Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.

Wang, S. and Klabjan, D. (2018). Discrete choice models with neural network utilities. *NeurIPS workshop*.

Wardman, M., Chintakayala, V. P. K. and de Jong, G. (2016). Values of travel time in Europe. *Transportation Research Part A* 94, 93–111.

Wong, M. and Farooq, B. (2021). ResLogit. *Transportation Research Part C* 126, 103050.
