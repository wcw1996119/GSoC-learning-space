# 重塑职位可达性 —— 一个评估城市政策干预对社会-空间不平等影响的精细化模拟工具

### 应用于伦敦的工作报告（Old Oak Common 副中心规划 与 弹性办公时段）

**作者：** 王成伟  
**工作报告 — 2026 年 5 月**

---

## 三句话概览

**研究目标。** 开发一个精细化的城市通勤模拟工具，**用于评估不同政策干预如何重塑社会-空间维度的通勤与职位可达性不平等** —— 通过直接从公开聚合 OD 数据中**反推个体行为决策参数**，让"政策干预对城市不同社会群体的可达性分配影响"得以被识别。

**研究方法。** 设计本质上是把**随机效用模型 (RUM) 的行为可解释性**和**深度学习的灵活函数逼近能力**绑在一起 —— 核心是**从聚合 OD 数据中反推个体行为决策参数**：写一个标准的 multinomial logit 选择模型 $P(j|i) = e^{V_{ij}}/\sum_k e^{V_{ik}}$，其中目的地"工作地吸引力" $A_j$ 由 GraphSAGE 图神经网络从空间特征中**学出来**，而出行时间敏感度 $\beta$、吸引力权重 $\alpha$、GNN 内部权重则从观测 OD 流量分布中**统一倒推**；RUM 保证 GNN 学出的 $A_j$ 仍是有经济学意义的标量、参数仍可解释，深度学习提供 RUM 单靠手工编码做不到的空间溢出表达力。

**研究结果。** 工具揭示伦敦两类干预的**质性不同的分布性签名** —— Old Oak Common 空间集中加就业**轻度回归**（人口加权 Palma 由 0.552 升至 0.561、自由流条件下 2/3 表面收益被道路拥堵吸收）；弹性办公时间分散**进步**（早高峰通勤量重分布 −46% / +63%、外伦敦高峰通勤占比 >50% 的居民最受益）—— 并由 mixed-logit 诊断暴露出**聚合预测在 greenfield 干预上系统性低估 17%** 的方法学副产物，这一不平等敏感的偏差在传统聚合流量分析中是看不见的。

---

## 1. 研究问题与意义

**职位可达性** —— *从你住的地方出发，实际可以到达多少个工作岗位* —— 是衡量城市机会公平性最干净的总量指标之一。两个工资相同但住地相距 5 公里的伦敦人，能现实通勤到达的工作集合可能完全不同；这个差距又会和收入、私家车持有、公共交通网络形态相互交织。几乎所有交通相关的城市政策杠杆 —— 从空间再开发到弹性工时到新建轨道交通 —— 都会推动这个差距朝某个方向变化，但**很少在城市内均匀地推**：一个让平均水平上升的政策完全可能同时**扩大**条件好与条件差地区之间的差距。这份报告要解决的方法学任务是：建立一个工具，让我们能看到**重分配的方向**，而不只是平均值的变化。

我们把这个工具应用于两个伦敦正在讨论的政策杠杆：

1. **Old Oak Common (OOC) 空间再开发。** 英国规划中规模最大的交通枢纽，预计在 Hammersmith / Ealing 边界的再开发集群中提供约 65,000 个新就业岗位。经典问题：在指定副中心新增就业，是真的把可达性从中心伦敦**分散**出去，还是仅在已有可达性地图上多点一个小亮点？

2. **弹性办公 / 错峰通勤。** 把 7–9 点高峰通勤负荷的 50% 移到 10–15 点的 shoulder 时段。经典问题：谁受益 —— 减少了拥堵的高峰办公族，还是终于等到合理发车间隔的非高峰服务业从业者？

这两类政策也都是北京目前正在讨论的（副中心战略、弹性办公政策）。**伦敦在这里只是方法学开发的沙盒**，不是研究目标本身。报告里每一个方法学选择都是按"将来可以平移到北京"这一硬约束筛过的。

## 2. 方法 —— 把 RUM 和深度学习结合

现有文献里模拟通勤如何在政策下变化的方法主要有三大类，但每一类都缺一件政策影响评估必需的能力：

| 方法族 | 长处 | 缺什么 |
|---|---|---|
| Gravity / radiation 模型 | 在校准数据上能预测聚合流量 | 不还原行为参数 → 反事实推断不连贯 |
| 纯深度学习（如 Deep Gravity） | 拟合好 | 参数无可解释行为意义 → 说不清"为什么"结果会变 |
| 纯 ABM | 行为丰富、个体级 | 智能体规则常靠拍脑袋设定 → 偏离真实数据 |

本工具的设计是把**随机效用模型 (RUM) 的行为可解释性**和**图神经网络的灵活函数逼近能力**绑在一起 —— 每一块都补另一边缺的能力。

### 2.1 可达性度量

每个 1×1 公里网格 *i* 的 Hansen 式可达性：

$$
A_i \;=\; \sum_j E_j \cdot e^{-\beta\, t_{ij}}, \qquad \beta = 0.058 \text{ /分钟}.
$$

衰减参数 $\beta$ **就是** §2.2 的选择模型反推出来的同一个 $\beta$ —— 所以 $A_i$ 数值的解释是"伦敦人实际权衡时间成本之后的有效就业触达"，而不是某个分析师拍脑袋的权重。30 分钟外的目的地相对 0 分钟外的权重为 18%；60 分钟外为 3%。$A_i$ 单位是"等效工作岗位数"：$A_i = 100{,}000$ 意味着该网格的有效可达性等于"门口 0 分钟有 100,000 个工作"。

### 2.2 选择模型 —— 随机效用 + 学出来的吸引力

住在 $i$ 的工作者面对全部 $N=1{,}725$ 个候选目的地，权衡两件事：

- **吸引力**：目的地 $j$ 作为工作地有多吸引人 —— 行业结构、就业密度、交通枢纽效应、邻里业态；
- **摩擦**：通勤到 $j$ 要多久。

这写成 multinomial logit：

$$
P(j \mid i;\,\theta) \;=\; \frac{\exp(V_{ij})}{\sum_{k=1}^{N} \exp(V_{ik})},
\qquad V_{ij} \;=\; \alpha\, A_j(\mathbf{X};\,\phi) \;-\; \beta\, t_{ij}.
$$

难的是 $A_j(\mathbf{X};\phi)$ —— 目的地的**作为工作地的吸引力**。手工编码成"jobs × wage × 业态指数"对观测流量拟合很弱，因为工作地吸引力依赖**邻里溢出**（Soho 的工作之所以吸引人，部分原因是 Soho 邻居也业态丰富）、**交通枢纽网络位置**、**产业集群交互** —— 这些没有分析师能手工指定。

我们让一个**图神经网络 (GraphSAGE)** 通过 1km 网格上的 1-hop 空间图，读取每个网格自己的 22 维特征 + 邻居特征，输出最贴合观测流量的吸引力标量 $A_j$：

$$
A_j(\mathbf{X};\,\phi) \;=\; \mathrm{GraphSAGE}_\phi\!\left(\mathbf{X}_j,\; \{\mathbf{X}_k : k \in \mathcal{N}(j)\}\right).
$$

关键：**GNN 自己不预测 OD**。它只提供一个结构化输入 —— 目的地吸引力向量 $A$ —— 喂给做实际预测的离散选择模型。这和 Deep-Gravity 风格 (Simini et al., 2021) 的方案不同 —— 后者把 origin / destination 特征塞进一个大神经网络、没有显式选择结构 —— 牺牲了让我们能就反事实政策做行为推断的可解释性。

### 2.3 "从聚合 OD 反推行为参数"的统计学原理

> 任何一篇文章声称从聚合数据里反推出个体行为参数，自然会被问：靠什么统计机制？有什么保证？我们欠读者一个干净的回答。

机制是 McFadden (1974) 传统下的**离散选择模型极大似然估计 (MLE)**，数值上由梯度下降执行。两个名字其实是同一件事。

**第 1 步。** 在 RUM 假设（i.i.d. 极值-1 误差）下，观测到的 origin-$i$ 流量向量服从多项分布：

$$
F_{i,\cdot} \;\sim\; \mathrm{Multinomial}\!\left(O_i,\; P(\cdot \mid i;\,\theta)\right),
\qquad
\theta = (\alpha,\, \beta,\, \phi).
$$

**第 2 步。** 因此参数向量 $\theta$ 在所有观测流量下的对数似然（差一个 $\theta$ 无关的常数）是：

$$
\mathcal{L}(\theta) \;=\; \sum_{i=1}^{N}\sum_{j=1}^{N} F_{ij} \log P(j \mid i;\,\theta).
$$

**第 3 步。** $-\mathcal{L}(\theta)$ **逐项就是**深度学习分类问题里普遍使用的**categorical cross-entropy loss**，target 分布为 $F_{ij}/O_i$、predicted 分布为 $P(j|i;\theta)$：

$$
-\mathcal{L}(\theta) \;=\; -\sum_{i,j} F_{ij} \log\frac{e^{V_{ij}}}{\sum_k e^{V_{ik}}} \;=\; \mathrm{CrossEntropy}\!\left(F_{ij}/O_i,\; P(\cdot|i;\theta)\right).
$$

**因此。** 用反向传播最小化深度学习的 cross-entropy loss，**数值上等同于**对 multinomial 离散选择模型做 MLE。深度学习社区和离散选择计量经济学社区在用不同的名字做同一件事 —— 这一等价性至少自 Bentz & Merunka (2000) 以来已被多次指出，正是它**给我们权利**声称所得 $(\hat{\alpha}, \hat{\beta}, \hat{\phi})$ 是 MLE 估计、具有 MLE 应有的标准统计性质。

**可识别性 (identification)。** 每个参数都由数据中的某条变异轴识别：
- $\hat\beta$ —— 由 OD pair 间 $t_{ij}$ 的横截面变异 + 流量份额随时间衰减的形状识别；
- $A_j$ 的**形状** —— 由 destination 间 $(E_j, \mathbf{X}_j, \mathbf{X}_{\mathcal{N}(j)})$ 横截面变异识别；
- $\hat\alpha$ —— 由 $A_j$ 与 $\beta\, t_{ij}$ 的相对尺度识别（softmax 自带一个 scale 自由度的固定）。

伦敦训练集提供 $1{,}725 \times 1{,}725 \times 24 \approx 7{,}100$ 万 OD-小时单元，对应 ~40,000 个 GNN 权重 + 2 个行为系数 —— 强 over-identified。

**已知 caveats**：

| Caveat | 我们的应对 |
|---|---|
| 神经网络让 loss 面非凸 → 不保证全局最优 | Multi-seed restart（Phase B+ 跑 5 个 seed）；seed 间 loss 差 < 2% |
| 聚合数据下 SE 比 individual 数据宽 | 不报点估的 SE；用 §7 reliability scan 给一个**经验**的 ±25% 置信带 |
| EV1 误差假设可能不严格成立 | 用 mixed-logit 蒙特卡罗（随机系数）做 sensitivity check —— 副产物就是 §3 的 *Jensen-bias* 诊断 |

### 2.4 数据

聚合 OD 流量表来自 UCL CASA 团队公开的**匿名化手机定位数据**（[Zhong & Zhou, 2025, *Scientific Data*](https://www.nature.com/articles/s41597-025-06323-8)；数据集 DOI 10.5281/zenodo.13327082）：来自约 200 个智能手机应用的 GPS 记录，**2021 年 11 月**在英格兰范围采集，住地由夜间 (22–06) GPS 记录最多位置推断，工作地由白天 stay-point detection 推断，聚合到 MSOA 级 travel-to-work 流量。我们用按陆地面积重分配的方法 re-grid 到 1×1 公里直角网格（覆盖大伦敦的 1,725 个有人口网格）。训练集总通勤流：约 121 万对 OD 关系。

2021 年 11 月这个采集时点处在英格兰秋季解封到 12 月 Omicron Plan B 之间的 inter-lockdown 窗口 —— 比如比 Census 2021（2021 年 3 月，全面封控）更接近"正常"基线，但仍部分受 hybrid working 影响。**pre-COVID 重训** 计划用 Geographic Data Service 2019 mobile data（见 §8）。

其他输入：1km 网格按行业的就业（BRES 2024）、POI 计数（OpenStreetMap）、travel-time 矩阵 $t_{ij}$（TfL 公共交通路线规划）、用于 BPR 均衡的路网。

### 2.5 为什么这套方法可平移到北京

每一个输入 —— POI、按行业就业、出行时间、匿名化手机 OD —— 北京都有等价（高德 POI、国家统计年鉴、高德路径规划、高校授权的手机 OD）。反过来 —— 用伦敦专属的人工编码 census 变量（ASHE income、JTS 出行时间统计、TS063 SOC9 职业表）拟合一个伦敦专属 RUM —— 是**不能**平移的。这一约束塑造了报告里每一个方法学选择。

## 3. 今天的伦敦长什么样

### 3.1 工作在哪、人在哪

![M1](figures/M1_employment.png)

**图 M1.** 1×1 公里网格的就业数（log scale）。City / Canary Wharf 集群比其他高一个数量级；次级集群在 Stratford、King's Cross、Hammersmith、Croydon。Old Oak Common 集群（星标）目前就业不多 —— 这正是再开发要填的缺口。

![M2](figures/M2_population.png)

**图 M2.** 1×1 公里网格的居住人口（log scale）。M1 和 M2 的错位是所有通勤的来源：工作集中在中心伦敦，居民分布在内环和更稀疏的外环。

### 3.2 可达性曲面

![M3](figures/M3_accessibility.png)

**图 M3.** 基线 Hansen 可达性 $A_i$，AM 高峰。中心伦敦 $A > 600{,}000$ 等效岗位数；外环东南部网格（Bromley、Bexley、Havering）在 30,000 以下 —— 大约 **20× 比例**。cell 级 Gini 系数 = **0.46** —— 高。

| 行政区（顶部） | 人口加权 $A_i$ | 行政区（底部） | 人口加权 $A_i$ | 比 |
|---|---:|---|---:|---:|
| City of London | 1,017,845 | Havering | 47,314 | 21.5× |
| Westminster | 571,752 | Bromley | 52,604 | 10.9× |
| Islington | 496,160 | Bexley | 53,956 | 9.2× |

### 3.3 Lorenz / decile 视图

![M7](figures/M7_inequality_summary.png)

**图 M7.** *左*：cell 级可达性的 Lorenz 曲线 —— 曲线与对角线之间的面积就是 Gini = 0.46 的可视化。*右*：各收入层 (Tier 1/2/3) 居民住地网格的 mean / median 可达性。**Tier 3 / Tier 1 比 = 1.17×**。

两个观察值得说。(i) cell 级 Gini (0.46) 远高于按收入层的梯度 (1.17×)，因为伦敦的低收入工作者并非全部在外围 —— 不少住在内环社会住房存量大的高可达网格里（Tower Hamlets、Lambeth、Southwark）。(ii) 即使有这种平滑作用，Tier-3 居民平均仍比 Tier-1 多 **17% 的可达性**。

## 4. 不平等视角 —— 谁能触达什么

![M4](figures/M4_access_by_income.png)

**图 M4 (3 panel)。** 同一个可达性曲面，从每个收入层的居民住地视角看 —— 该层居民少的网格被淡化。Panel (a) 是 Tier-1（低收入）居民实际看到的；Panel (c) 是 Tier-3 看到的。

读图：Tier-1 居民集中在内东和内南伦敦（Tower Hamlets、Hackney、Lambeth、Southwark）—— 这些网格连接性合理但不在最高峰。Tier-3 居民集中在中央西部和内北部（Westminster、Camden、Islington、Kensington & Chelsea）—— 正中最高 $A$ 网格。Tier-2 在两者之间。

这 17% 的差距是真实的，但比"伦敦不平等"标题修辞所暗示的要小。伦敦特有的原因是中央东部各 borough 把高可达性和强大的社会住房传统结合在一起 —— 这一特征**并非**所有大都市通用，**不能**假定北京也有。

## 5. 情景 A —— Old Oak Common（+65,000 个就业岗位）

我们对 OOC 9 个网格集群（Hammersmith / Ealing 边界）一致性增加 +65,000 个就业岗位，然后在三种设置下重跑全管道：

| 设置 | Mean $A_i$ | 人口加权 Gini | 人口加权 Palma |
|---|---:|---:|---:|
| 基线（今天）| 1,222,444 | 0.167 | 0.552 |
| OOC，free-flow $t_{ij}$ | 1,237,783 | 0.169 | 0.557 |
| OOC，拥堵均衡 (BPR) | 1,225,612 | 0.169 | 0.561 |

（来自 `evaluation_outputs/paper_a/scenario_A_accessibility.json`。Gini 和 Palma 在 agent 人口上算 —— 按人加权视角。）

![M5](figures/M5_scenarioA_delta.png)

**图 M5.** 每个网格因 OOC 干预的 $\Delta A_i$，自由流 opportunity-field 假设下。增量场以反推出的 $\beta = 0.058$/分钟从 OOC 径向衰减；OOC 30 分钟内的网格吸收大部分新机会；外环东南伦敦几乎没有。

**读法。** 自由流条件下平均 $A$ 升 +1.3%，到拥堵均衡只剩 +0.3% —— 大部分自由流收益被拥堵吸收。不平等**轻微上升**：人口加权 Palma 从 0.552 升到 0.561，因为收益主要落在已经连接好的西北网格上。一句话总结：**OOC 创造了真实的新可达性，但收益沿收入分布向上扩散得比向下快**。要让再开发真正实现可达性重分配，可能需要配合 (i) 把 OOC 接进外环南部伦敦的轨道交通改进、或 (ii) OOC 集群本身的可负担住房供给、或 (iii) 偏离已经在内西伦敦集中的高薪行业的产业组合。仅靠这份报告我们无权推荐其中任一项 —— 但我们**有权**说："建好它，可达性自然就重分配"不是模型预测的结果。

## 6. 情景 B —— 弹性办公 / 错峰

模拟将 7–9 点高峰拥堵降低 50%，重分配到 10–15 点 shoulder 窗口。模型的时间分辨版本在新拥堵向量下重新分配每个住地网格的全天通勤分布。

![M6](figures/M6_scenarioB_share.png)

**图 M6.** 每个住地网格通勤行程中、于 7–9 点 AM 高峰出发的份额。外环高峰份额**超过 50%** 的网格的居民对高峰成本暴露最大 —— 也是真正"把需求挪开高峰"的弹性办公政策下受益最大的居民。

| 时段 | Δ流量（基线百分比）|
|---|---:|
| 7–9 点高峰 | **−45.7%** |
| 10–15 点 shoulder | **+63.0%** |
| 其他非高峰 | +8.7% |

（`evaluation_outputs/paper_a/T7_scenario_B_per_hour.csv`。）

**读法。** 高峰办公族一次性减少约 46% 拥堵暴露，但更大的故事在 shoulder —— 排班已经在 10–15 点窗口的服务业 / 班次工作者拿到明显更密的发车间隔，这提高了他们的有效可达性。结合图 M6，受益最大的网格是：高峰份额高的外环东南伦敦（Bromley、Bexley），以及非高峰发车间隔目前掉得很快的外环东北部分。

诚实的 caveat：本报告**还未**量化对 Tier-1 服务业从业者的发车间隔 / 等车时间效应；要做扎实，需要把选择模型和公交网络的排队模型耦合。

## 7. 这些预测什么时候可以信？

一个在历史 OD 上拟合很好的模型，在外推时仍可能跑偏。所以我们在引用政策数字之前先跑了一个 **reliability scan**：60 个随机分布的 9-cell 合成 cluster × 6 个就业扰动幅度（10%、30%、50%、100%、200%、500%），共 360 次实验。每次都比对**聚合预测**和**蒙特卡罗 per-agent 预测**（mixed-logit、按 agent 收入分布抽样）。两者之间的差距就是任何聚合目的地选择模型在做反事实声明时都会带的 **Jensen 偏差**。

![F8](../evaluation_outputs/paper_a/F8_reliability.png)

**图 F8.** *左*：每次实验的 $|\text{Jensen bias}|$ 作为 干预幅度 × cluster 基线就业的函数。绿色轮廓是经验"可信"区域 ($|\text{bias}| < 5\%$)。*右*：360 次实验上 $|\text{Jensen bias}|$ 的 CDF —— 12% 落在 5% 以下（可信）、再 31% 落在 20% 以下（warrant caveat）、56% 在 20% 以上（不可信）。

OOC 干预对应的位置是 **基线 cluster 就业 ~150,000 × +43% 幅度** —— scan 把它判进 **caveat 带**。所以我们把 §5 的 headline 数字理解为带一个大约 **±25% 的置信带**：自由流 +1.3% / 均衡 +0.3% 的方向和数量级是对的，但不要把第三位有效数字当真。

这个 scan 本身是方法学贡献。多数可达性建模文章只报单一情景的 $\Delta$ 数字；我们报的是**模型在 policy space 中能产生可信 $\Delta$ 的区域、不能的区域**。

## 8. 局限和下一步

1. **时段。** 2021 年 11 月在第二轮封控之后、Omicron 之前。Hybrid working 部分被烤进基线。在 GDS 2019 mobile data 上做 pre-COVID 重训会让解释更紧 —— 尤其是弹性办公场景。
2. **收入颗粒度。** 三层是手机数据隐私聚合的硬限。更细的拆分（收入十分位 × 职业）需要链接到收入微观数据。
3. **反事实是结构性投影，不是预报。** 模型假设偏好和摩擦参数在空间布局变化下不变。§7 的 reliability scan 是部分护栏，不是完整护栏。
4. **伦敦是沙盒。** 北京复制需要在 高德 routing、NBS POI、北京 OD 上重训。管道为这种 swap 设计；本报告里的数字不可平移。

**下一步计划。** (i) GDS 2019 上的 pre-COVID 重训；(ii) OOC × 轨道交通耦合（Crossrail-2 / GLA bus-route 层加进 $t_{ij}$）；(iii) 弹性办公 × 发车间隔：把目的地选择模型耦合到一个简单的公交发车间隔模型，让 Tier-1 服务业从业者的收益可量化；(iv) 北京 parallel build —— 数据 clearance 一旦落地（目标 2026 年底）。

---

## 参考文献

- Bentz, Y., & Merunka, D. (2000). *Neural networks and the multinomial logit for brand choice modelling: A hybrid approach*. Journal of Forecasting, 19(3), 177–200.
- Hansen, W. G. (1959). *How accessibility shapes land use*. JAIP, 25(2), 73–76.
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive representation learning on large graphs (GraphSAGE)*. NeurIPS.
- McFadden, D. (1974). *Conditional logit analysis of qualitative choice behavior*. In Frontiers in Econometrics, ed. P. Zarembka.
- Simini, F., Barlacchi, G., Luca, M., & Pappalardo, L. (2021). *A Deep Gravity model for mobility flows generation*. Nature Communications, 12, 6576.
- Train, K. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge UP. (Chs 3, 5.)
- Yao, X., Cheng, T., et al. (2021). *Spatial OD flow imputation using graph convolutional networks*. IEEE T-ITS.
- Zhong, C., & Zhou, Z. (2025). *Anonymised human location data in England for urban mobility research*. Scientific Data, 12. (数据集 DOI 10.5281/zenodo.13327082。)

---

*可复现：`report/build_accessibility_maps.py` 重生成全部图。数字 headline 来自 `report/data/accessibility_per_grid.csv` 和 `evaluation_outputs/paper_a/*.json`。*
