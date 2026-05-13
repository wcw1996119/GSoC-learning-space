# Paper A — Revealed-Preference VOT Literature Review

**目的**：为 Paper A (differentiable inverse choice learning) 提供 RP-VOT 文献基线，验证两件事：
1. 我们从 London 2021 OD 反推出的 baseline VOT ≈ £7/h 是否落在已发表的 RP-VOT 区间内
2. 我们发现的 negative income gradient（低收入区 → 更高 VOT）是否与 RP 文献一致 —— 注意这与 WebTAG (主要基于 SP) 的传统 income gradient 方向相反

> ⚠ **Honesty caveat**：本综述的具体数值（£/h ranges、年份、page numbers）部分为我从训练数据中回忆得来。所有标注 "(verify before paper submission)" 的数字必须在投稿前核对原文。Citation strings 中我对作者拼写和年份相对有把握，对期刊卷期号把握较低。

---

## Section 1: RP vs SP framework

**Revealed Preference (RP)** 数据来自实际选择行为 —— 比如 Census journey-to-work、smartcard 数据、road-pricing 实验中的真实路线选择。**Stated Preference (SP)** 数据来自假想情境问卷，让受访者在 hypothetical 的 (time, cost) trade-off 中表态。

两者各有 bias。RP 的优势是 ecological validity（人真的做了这个选择，没有 hypothetical bias），但缺点是 attribute correlation 严重（实际选项中 time 和 cost 往往高度共线，identifiability 差）和选择集模糊（你不知道 commuter 实际考虑了哪些备选）。SP 的优势是 experimental control —— 设计者可以人为去耦 time 和 cost —— 但有 hypothetical bias、scale bias 和 strategic response 等问题。

**Calfee & Winston (1998)** 是经典的 RP/SP gap 文献，他们用 US 数据发现 SP 估出的 VOT 系统性高于 RP。**Wardman (2008)** 的 UK meta-analysis 进一步定量化这个差距：commute 场景下 RP-VOT 比 SP-VOT 低约 30–50% (verify before paper submission，具体数字可能是 30% 或 40%)。这个 gap 至今仍是 transport econometrics 的标准 stylised fact。

对 Paper A 而言，这意味着：如果我们的 inverse-choice 模型从 OD 流量中反推出的 VOT 显著低于 WebTAG 官方值（WebTAG 大量依赖 SP），这本身**不**是 model bug，而是 expected RP/SP gap 的体现。

---

## Section 2: UK-specific RP-VOT estimates for commute

下面按论文逐条整理 UK / European RP-VOT 数值。**所有 £/h 数字都尽量统一到 2010–2020 年价（部分论文原文是 pence/min 或更早年份价，我会标注）**。

### 2.1 Wardman (2003 / 2011 update)
Wardman 是 UK 交通经济学界 VOT meta-analysis 的核心人物。他 2003 年 Transportation Research Part E 上的 meta-analysis (Wardman 2003, "Public transport values of time", verify before paper submission) 综合了 1960s–2000s 的 UK 研究。后续 2011 年前后他做过 update。**Commute RP-VOT 区间大致 £4–8/h (2010 prices)** (verify before paper submission)，这是综合大量研究的中位数感觉，而非单篇论文的点估计。

### 2.2 Wardman, Chintakayala & de Jong (2016)
"Values of travel time in Europe: Review and meta-analysis", *Transportation Research Part A: Policy and Practice*, Vol. 94, pp. 93–111 (verify before paper submission). 这是欧洲范围最权威的 VOT meta-analysis。他们整合了 100+ 篇研究，区分 RP 和 SP，区分 commute / business / leisure。**Commute RP-VOT 在欧洲国家间集中在 €5–12/h (2010 €)**，UK 子样本大致 **£5–9/h (commute, 2010 prices)** (verify before paper submission)。他们也明确 confirm RP < SP gap，commute 场景 gap 在 30% 左右。

### 2.3 Mackie, Wardman, Fowkes, Whelan, Nellthorp & Bates (2003)
"Values of travel time savings UK", ITS Leeds report to DfT (verify exact title and series). 这是 UK WebTAG 官方 VOT 数值的源头之一。它**主要基于 SP**（大型 stated choice survey），但也校准了 RP 数据。Commute VTTS 当年报告值约 £4.46/h (2002 prices) (verify before paper submission)，inflate 到 2020 prices 大致 £6.5–7.5/h。注意这个数字接近 SP 端，不是纯 RP。

### 2.4 DfT WebTAG TAG A1.3 (current)
WebTAG 是 UK DfT 的官方 appraisal guidance。**TAG Unit A1.3 "User and provider impacts"** 给出 commute VTTS（最近版本 2022/2023 update，verify before paper submission）大约 £11–13/h（commute, 2010 market prices, perceived），business 显著更高（£20+/h）。**关键澄清**：WebTAG 的 commute VTTS **主要来源是 Mackie et al. 2003 的 SP work + 后续 inflation/GDP scaling**，并非 pure RP。WebTAG 本身**不**用 income gradient 区分 commute VOT —— 它对所有 commuter 用同一个值（"equity assumption"），这点对我们的 inverse 比较很关键。

### 2.5 Hess, Bierlaire & Polak (2005)
"Estimation of value of travel-time savings using mixed logit models", *Transportation Research Part A*, Vol. 39 (2–3), pp. 221–236 (verify before paper submission). 用 mixed logit 估 UK / Swiss / Dutch 数据中的 VOT 分布异质性。**Mean commute VOT 约 £6–8/h (UK subset)** (verify before paper submission)，并且发现 substantial heterogeneity —— VOT 在人群中的分布有 long right tail。这篇论文的 methodological 贡献大于点估计，但 mean 值落在 £6–8 范围内。

### 2.6 Börjesson & Eliasson (2014)
"Experiences from the Swedish Value of Time study", *Transportation Research Part A*, Vol. 59, pp. 144–158 (verify before paper submission). Stockholm 数据，瑞典官方 VOT 研究。Commute VOT 约 **SEK 70–90/h (2008 prices)，约 €7–10/h，约 £6–8/h** (verify before paper submission)。这篇论文的价值在于它结合了 RP（实际通勤选择）和 SP（问卷），并 explicitly 讨论了 income segmentation。

### 2.7 TfL / DfT internal RP-VOT studies
TfL 内部用 Oyster smartcard 数据做过 mode-choice 模型 calibration（"LTS" London Transport Studies model）。这些研究的 VOT 估计**通常未公开发表**，但 TfL 的 business case 文档中偶尔引用。我**不**记得具体公开数字 (verify before paper submission)。LATS (London Area Transport Survey) 数据也被多次用于 RP 估计，但论文级 publication 有限。

---

## Section 3: Income gradient — does RP show "low-income → higher VOT"?

这是 Paper A 的关键 validation 问题。WebTAG **不**按 income 分层（equity assumption），但传统 SP-based literature 普遍报告 **income elasticity of VOT ≈ 0.5–0.8** —— 即收入越高 VOT 越高（time is money）。我们的 inverse-choice 结果反过来：**低收入 MSOA 显示更高 β_t**。下面看 RP 文献是否支持这个反向关系。

### 3.1 Stutzer & Frey (2008)
"Stress that doesn't pay: The commuting paradox", *Scandinavian Journal of Economics*, Vol. 110 (2), pp. 339–366. 这篇论文不是 transport econometrics 而是 happiness economics，但 highly relevant：他们发现 commuters **systematically underestimate** 长通勤的 disutility，且 **low-income / forced 长通勤者** 受影响最大（无法用住房选择 compensate）。这暗示：低收入人群的真实（revealed）time disutility 实际上**更高**，但他们因为住房 constraint 无法 reveal 在 mode/route choice 上 —— 反而可能 reveal 在其他边际 (e.g. departure time, willingness to skip trip)。**支持我们 finding 的 indirect 证据**。

### 3.2 Pucher & Renne (2003)
"Socioeconomics of urban travel: Evidence from the 2001 NHTS", *Transportation Quarterly*, Vol. 57 (3), pp. 49–77 (verify before paper submission). US 数据，但 stylised facts 普适：低收入家庭 disproportionately 依赖 transit、长 commute time、少 mode flexibility。这意味着 RP 中观察到的低收入 commuter 选择 = constrained choice，他们的 "revealed" β_t 可能反映**约束**而非**偏好** —— 这是解读我们 finding 的重要 caveat。

### 3.3 Wardman & Whelan (2011)
UK 文献中我对 Wardman & Whelan 关于 income/VOT 关系有 vague 印象，但**不能确定具体论文 title** (verify before paper submission)。Wardman 在多篇论文中报告 income elasticity of SP-VOT 约 0.5，对 RP-VOT 的 income gradient 报告较少。

### 3.4 Small, Winston & Yan (2005)
"Uncovering the Distribution of Motorists' Preferences for Travel Time and Reliability", *Econometrica*, Vol. 73 (4), pp. 1367–1382. 经典 RP 研究，使用 California SR-91 express lane（付费快车道） vs 免费拥堵车道的真实选择数据。他们用 mixed logit 估计 VOT 异质性分布。**关键 finding**：median VOT ≈ \$22/h (2000 USD)，**且 VOT 与 income 有 positive 相关**（低收入 driver VOT 较低）。这是经典的 SP-style income gradient 在 RP 数据中的体现 —— **与我们 finding 方向相反**。

但要注意：Small et al. 的 sample 是 already-driving 的 motorists，已经 self-selected 出能负担 car 的人群。Mode choice 的 selection bias 可能 mask 了不同方向的 preference。

### 3.5 Karlström & Franklin (2008)
"Behavioral adjustments and equity effects of congestion pricing: Analysis of morning commutes during the Stockholm Trial", *Transportation Research Part A*, Vol. 43 (3), pp. 283–296 (verify volume/year). Stockholm 拥堵收费 RP 研究。他们发现 **low-income commuter 的 departure time shift 反应更强** —— 即对相同收费 disutility 更大，意味着对 generalized cost 更敏感。但他们也发现 mode shift 反应较弱（受 accessibility 约束）。**这部分支持低收入 → 高 time/cost sensitivity 的方向**，但要小心：高 cost sensitivity 不直接等于高 time sensitivity（VOT 是比值 β_t / β_c）。

### 3.6 Brownstone & Small (2005)
"Valuing time and reliability: assessing the evidence from road pricing demonstrations", *Transportation Research Part A*, Vol. 39 (4), pp. 279–293 (verify). 综述 US/UK road pricing demonstrations 的 RP-VOT 估计。他们的总结：**RP-VOT typically 高于 SP-VOT 在这种特定 setting 下**（这与 Wardman 2008 的 commute meta 方向相反，因为 road-pricing 场景下 cost 不再是 hypothetical）。Income gradient 大致 positive 但 noisy。

### 3.7 综合判断
RP 文献关于 income/VOT 关系的证据**混合 (mixed)**，但**多数偏向 positive gradient** (高收入 → 高 VOT)。与我们 finding 相反方向。

**我们 finding 的可能解释**（需要在 Paper A 讨论）：
1. **Constrained choice artefact**: 低收入 commuter 的 mode/route choice set 被 housing 和 affordability 约束，他们对 time 看似更敏感是因为 cost 这一边被压缩
2. **Aggregation bias**: 我们的 β_t 是 MSOA 级别 aggregate，低收入 MSOA 的 β_t 可能反映 sample composition 而非个体 preference
3. **真实信号**: 低收入人群单位时间机会成本（chores、多份工作、childcare）反而比 wage rate 暗示的更高 —— Stutzer & Frey 的 happiness angle 支持这个

---

## Section 4: Synthesis for Paper A

**Magnitude check**: 我们 baseline VOT ≈ £7/h 落在 UK RP-VOT 公开区间 **£4–9/h** 的中位偏低位置，这是 healthy 的 sanity check。具体对照：
- Wardman 2003/meta median ≈ £4–8/h ✓
- Wardman et al. 2016 European meta UK subset ≈ £5–9/h ✓
- Hess et al. 2005 mixed logit mean ≈ £6–8/h ✓
- WebTAG (mostly SP) ≈ £11–13/h —— 我们低 ~40%，与 Wardman 2008 报告的 RP/SP gap (~30–50%) 一致

所以 magnitude 上 Paper A 的结果是 **defensible** 的，**对照基准应该是 RP literature 而不是 WebTAG**。在 paper 中要明确说："our recovered VOT lies within published RP-VOT ranges; the gap with WebTAG reflects the well-documented RP/SP discrepancy (Wardman 2008; Calfee & Winston 1998)."

**Direction (income gradient) check**: 我们 finding 的方向 (低收入 → 高 VOT) 与**多数 RP 文献相反**。这是 Paper A 必须正面面对的 anomaly：
- 不能 oversell 为 "we discovered the true gradient"
- 应该 frame 为 "constrained-choice / aggregation lens 下 MSOA-level inverse-recovered β_t 的 interpretation"
- 引用 Stutzer & Frey 2008 和 Pucher & Renne 2003 作为 indirect support
- 把 Karlström & Franklin 2008 的 departure-time evidence 作为 narrowest support
- 老实承认：没有强 RP-VOT income-gradient evidence 直接支持我们 finding

**对 Paper A 的实际 framing 建议**：把 income gradient 结果放在 "Findings" 中实事求是地报告，但**主结论**应该是 "differentiable inverse choice 这个方法本身能从 OD 中恢复 VOT 到 RP-literature-consistent magnitude"，而不是 "我们发现了反向 income gradient"。后者是 secondary finding，需要更强的 robustness checks（Phase B 处理）。

---

## Section 5: Bibliography & honest uncertainty

> ⚠ All entries below: 我对作者名 + 年份 + 大致主题相对有把握；对**期刊卷期号、page numbers、具体 £/h 数字**把握较低。所有 (verify) 项必须投稿前核对。

1. **Calfee, J. & Winston, C. (1998)**. The value of automobile travel time: implications for congestion policy. *Journal of Public Economics*, Vol. 69 (1), pp. 83–102 (verify before paper submission). DOI: 不记得具体 DOI。

2. **Wardman, M. (2003)**. Public transport values of time. *Transport Policy*, Vol. 11 (4), pp. 363–377 (verify volume/page). DOI: 不记得。

3. **Wardman, M. (2008)**. Twenty years of meta-analysis on value of travel time savings: an update and extension. ITS Leeds working paper or *Transportation Research* article (verify exact venue and year — possibly 2004 or 2008).

4. **Wardman, M., Chintakayala, V.P.K. & de Jong, G. (2016)**. Values of travel time in Europe: Review and meta-analysis. *Transportation Research Part A: Policy and Practice*, Vol. 94, pp. 93–111 (verify pages). DOI: 10.1016/j.tra.2016.08.019 (verify DOI).

5. **Mackie, P., Wardman, M., Fowkes, A.S., Whelan, G., Nellthorp, J. & Bates, J. (2003)**. Values of travel time savings UK. ITS Leeds Working Paper / Report to DfT (verify exact title).

6. **DfT (UK Department for Transport)**. *TAG Unit A1.3: User and Provider Impacts*. Latest version 2022/2023 (verify current version date). Available via gov.uk/transport-analysis-guidance-webtag.

7. **Hess, S., Bierlaire, M. & Polak, J.W. (2005)**. Estimation of value of travel-time savings using mixed logit models. *Transportation Research Part A: Policy and Practice*, Vol. 39 (2–3), pp. 221–236 (verify pages). DOI: 10.1016/j.tra.2004.09.007 (verify DOI).

8. **Börjesson, M. & Eliasson, J. (2014)**. Experiences from the Swedish Value of Time study. *Transportation Research Part A: Policy and Practice*, Vol. 59, pp. 144–158 (verify pages).

9. **Stutzer, A. & Frey, B.S. (2008)**. Stress that doesn't pay: The commuting paradox. *Scandinavian Journal of Economics*, Vol. 110 (2), pp. 339–366. DOI: 10.1111/j.1467-9442.2008.00542.x (verify DOI).

10. **Pucher, J. & Renne, J.L. (2003)**. Socioeconomics of urban travel: Evidence from the 2001 NHTS. *Transportation Quarterly*, Vol. 57 (3), pp. 49–77 (verify pages and exact venue — may have been published in *Transportation Research Record* instead).

11. **Small, K.A., Winston, C. & Yan, J. (2005)**. Uncovering the distribution of motorists' preferences for travel time and reliability. *Econometrica*, Vol. 73 (4), pp. 1367–1382. DOI: 10.1111/j.1468-0262.2005.00619.x (verify DOI).

12. **Karlström, A. & Franklin, J.P. (2008/2009)**. Behavioral adjustments and equity effects of congestion pricing: Analysis of morning commutes during the Stockholm Trial. *Transportation Research Part A*, Vol. 43 (3), pp. 283–296 (verify year — could be 2009).

13. **Brownstone, D. & Small, K.A. (2005)**. Valuing time and reliability: assessing the evidence from road pricing demonstrations. *Transportation Research Part A: Policy and Practice*, Vol. 39 (4), pp. 279–293 (verify pages). DOI: 10.1016/j.tra.2004.11.001 (verify DOI).

14. **Wardman, M. & Whelan, G. (2011)** — UK income/VOT relationship paper, **exact title and venue I cannot recall reliably** (verify before paper submission). Likely in *Transport Reviews* or *Transportation*.

---

**Action items before paper submission**:
- (a) 用 Google Scholar 核对每条 citation 的 exact title / journal / volume / pages / DOI
- (b) 用 ITS Leeds working paper repository 核对 Wardman 系列论文的 exact value ranges
- (c) 用 DfT WebTAG live document 核对当前 TAG A1.3 commute VTTS 数字（2024 update 可能存在）
- (d) 把所有 £/h 数字统一到一个 base year（建议 2020 prices, GDP deflator scaling）
