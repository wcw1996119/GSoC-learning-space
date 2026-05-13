# Node Features 清单（v2）

> **目的**：定义 v2 STGNN 的节点特征 inventory，要求"伦敦 + 北京"双城对齐。
>
> **关系**：本文档**取代** `../../02_london_commuting_model/methodology/01_step1_gnn_decisions.md` 里的 Decision 4。原 Decision 4 写于 4-30，当时未考虑 Beijing-friendly 约束 + 1km 网格 + STGNN 时间维度，已 obsolete。

---

## 1. 决策约束

来自 5-4 架构定型（参见 `02_counterfactual_evaluation.md` §4 + memory `project_beijing_constraint.md`）：

1. **Beijing-friendly 硬约束**：每个特征必须在伦敦和北京都有等价数据源
2. **1km × 1km 网格**：特征单元是网格，不是 MSOA / 街道
3. **STGNN 架构需要**：必须有时间维度特征喂给 GRU 时间层
4. **不依赖伦敦专属数据**：PTAL / ASHE / IMD / SOC entropy 全部 retire

---

## 2. 决策汇总表（locked）

### 2.1 Baseline 静态特征（10 项 → 21 维）

| # | 特征 | 维度 | 伦敦数据源 | 北京数据源 |
|---|---|---|---|---|
| 1 | Total employment E_j | 1 | BRES → 1km disaggregation | 年鉴行业级 → POI 加权 disaggregation |
| 2 | Employment by sector（8 类） | 8 | BRES 18 产业 → 8 类聚合 | 年鉴 → 8 类对齐 |
| 3 | Total population（居住） | 1 | Census 2021 OA → 1km grid | WorldPop 1km 或 年鉴 + dasymetric |
| 4 | POI total count | 1 | OSM POI | 高德 POI |
| 5 | POI by category（8 类） | 8 | OSM tags → 8 类映射 | 高德 POI 分类 → 8 类映射 |
| 6 | Subway station count within 500m | 1 | TfL stations | 北京地铁 |
| 7 | Lat | 1 | grid centroid | grid centroid |
| 8 | Lon | 1 | grid centroid | grid centroid |

**合计静态维度**：21

### 2.2 Baseline 时间特征（4 项 → 4 维 per hour）

| # | 特征 | 维度 | 伦敦数据源 | 北京数据源 |
|---|---|---|---|---|
| 9 | Hourly congestion at grid | 1 | TomTom borough → grid（同 borough 共享） | 高德拥堵指数 → 路段 → grid 平均 |
| 10 | Hour-of-day sin | 1 | derived | derived |
| 11 | Hour-of-day cos | 1 | derived | derived |
| 12 | Workplace population estimate | 1 | 从 OD × NTS/LTDS 时间分布 disaggregate | 同方法（北京 OD + 通勤时间分布） |

**合计时间维度**：4 per hour

### 2.3 总输入维度

每个 grid 每小时 = 21 + 4 = **25 维**

---

## 3. 静态特征详细说明

### 3.1 Total employment E_j

**伦敦**：BRES 2024 提供 18 产业 × MSOA 就业数。1km 网格化方法：
- MSOA 边界 + grid 相交 → 按面积权重分配（**area-weighted**）
- 或：用 BRES address-level（如可获得）→ 直接 grid 聚合

**北京**：年鉴只到区（district）级行业就业。Disaggregation:
- 每个 district 的 sector j 总就业 → 按该 district 内 grid 的 POI(sector j) count 加权分配到 grid
- 这是 **POI-weighted dasymetric** 方法

**Normalization**：log(1+x) → z-score

### 3.2 Employment by sector（8 类）

为什么 8 类：
- 太少（如 3 类，跟 Yao 2021 一样）→ 信息损失
- 太多（如 BRES 18 类）→ 北京年鉴对不齐 + 维度膨胀
- 8 类是"行业意义清晰 + 跨城市对齐可行"的折中

**8 类定义**（伦敦 BRES → 北京年鉴 mapping）：

| v2 Sector | BRES (UK SIC) | 北京年鉴对应 |
|---|---|---|
| 1. Primary | A, B（农林渔业、采矿） | 农林牧渔、采矿业 |
| 2. Manufacturing | C（制造业） | 制造业 |
| 3. Construction | F（建筑） | 建筑业 |
| 4. Retail & Wholesale | G（批零） | 批发零售业 |
| 5. F&B & Hospitality | I（住宿餐饮） | 住宿餐饮业 |
| 6. Information & Finance | J, K（信息、金融） | 信息传输/软件、金融业 |
| 7. Public services（教育/医疗/政府） | O, P, Q（公共行政、教育、卫生） | 公共管理、教育、卫生 |
| 8. Other services（专业、商务、艺术） | M, N, R, S（专业、商务、文体） | 租赁商务、文化体育 |

**Why**：通州副中心反事实需要"加 +N 万 **什么类型** 岗位"——必须能区分 sector。

**Normalization**：每个 grid 的 sector 比例（sum=1）+ log scale total

### 3.3 Total population（居住）

**伦敦**：Census 2021 OA-level → grid（OA 比 1km 粒度更细，直接面积权重聚合）
**北京**：WorldPop 1km grid 直接用，或年鉴 + dasymetric mapping

**Normalization**：log(1+x) → z-score

### 3.4 POI total count + POI by category（8 类）

**8 个 POI 类别**：

| v2 POI category | OSM tag examples | 高德分类 examples |
|---|---|---|
| 1. Commercial | shop=*, office | 商务住宅 / 商务大厦 |
| 2. F&B | amenity=restaurant, cafe, bar, food_court | 餐饮服务 |
| 3. Retail | shop=supermarket, mall | 购物服务 |
| 4. Office | office=* | 公司企业 |
| 5. Education | amenity=school, university | 科教文化（学校部分） |
| 6. Healthcare | amenity=hospital, clinic | 医疗保健 |
| 7. Transport | public_transport=*, amenity=bus_station | 交通设施 |
| 8. Public | amenity=townhall, government | 政府机构 |

**OSM ↔ Amap 映射表**作为独立工程，需要：
- 写 `providers/poi_category_mapping.py`，定义 8 类的 OSM 和 Amap 关键字 list
- 单元测试：抽样验证映射准确度

**Normalization**：每个 grid 的 category count → log(1+x) → z-score

### 3.5 Subway station count within 500m

**伦敦**：TfL Open Data 提供 station 经纬度。每个 grid centroid 计算 500m 半径内的 station 数量。
**北京**：从北京地铁官方公开数据 / OSM railway=subway。

**Why 500m**：步行 5-7 分钟可达距离。可消融 300m / 800m。

**为什么不用 PTAL**：PTAL 是 TfL 综合指标（含频率、距离、走行时间）——北京无等价综合指标。"地铁站点数"是**双方都能算的简化版**。

**Normalization**：raw count（小整数，无需变换）

### 3.6 Lat/Lon

**直接用 grid centroid 经纬度**，z-score normalize。

**Why 不用 cyclic encoding**：1km 网格在伦敦 ~50×30 范围内（北京 ~30×30），远不到地球曲率引起的非欧几何问题。直接坐标够用。

**Why 保留**：GNN 在 message passing 时不天然知道全局位置。lat/lon 给"绝对位置 anchor"，对捕捉单中心 / 多中心 monocentric structure 有帮助。

---

## 4. 时间特征详细说明

### 4.1 Hourly congestion at grid

**伦敦**：TomTom 是 **borough 级**，所有 grid in same borough 共享同一 ratio。粒度粗，但**等价数据北京有更细的**（高德拥堵指数到路段级），所以**接口要支持任意空间粒度**。

```python
# Provider 接口设计
class CongestionProvider:
    def get_grid_congestion(self, grid_id, hour) -> float:
        # London: 查 grid 所属 borough 的 ratio
        # Beijing: 查 grid 内所有路段的平均 ratio
        ...
```

**Normalization**：congestion ratio 已经是相对值（free-flow / actual），直接用或 log。

### 4.2 Hour-of-day sin/cos

经典 cyclic encoding：
- `hour_sin = sin(2π · h / 24)`
- `hour_cos = cos(2π · h / 24)`

让 23:00 和 00:00 在 feature space 距离近（避免数值跳变）。

### 4.3 Workplace population estimate

**核心思路**：从静态 OD + 通勤时间分布合成 hourly workplace population。

```
workplace_pop[j][hour] = Σ_i OD[i, j] × P(at_workplace_at_hour | mode, t_ij)
```

**伦敦**：
- OD：Census 2021 WU03UK
- P(at_workplace_at_hour) ← LTDS / NTS departure-time × t_ij → arrival-time → on-site duration

**北京**：
- OD：2021 数据（伦理通过后获得）
- P(at_workplace_at_hour) ← 北京通勤时间分布（年鉴 / 公开调查文献）

**Why 这个特征**：让 V_j(t) 真的有时间变化。不然 GNN 时间层基本闲置——通勤场景下 V_j 大部分静态，需要这种"在地人口"反映"现在 j 这个地方有多少人"的活跃度。

**Limitation**：合成质量依赖辅助分布。北京的 LTDS 等价物覆盖不全，可能要用更粗的"全市平均通勤时间分布"。

---

## 5. 已 retire 的特征（明确说明）

| 特征 | 为什么 retire |
|---|---|
| **PTAL**（公交可达性等级） | TfL 专属合成指标；北京无综合等价 |
| **ASHE workplace income** | 英国 zone 级；北京年鉴只到行业平均工资，无地理粒度 |
| **IMD**（剥夺指数） | 英国专属；中国无单一综合指标 |
| **SOC entropy** | 基于 SOC9 分类；中国 GBM 分类不同，难直接对齐 |
| **Census 2011 mode shares** | v1 用过；mode 在 v2 改为 RUM 决策，不作输入特征 |
| **JTS 通勤时间基准** | DfT 专属；北京无等价。改为 cross-check 时用 baseline 模型 |
| **Population density** | 跟 population + area 共线，trivial 派生，不进 baseline |
| **Distance to city center** | 强 monocentric prior，但 lat/lon 已隐含；进 ablation |

---

## 6. Ablation 候选（不进 baseline，但准备好）

- Land use（dominant 类别 one-hot，OSM landuse / Globeland30）
- Distance to city center
- Population density
- POI density（POI count / area）
- Subway station distance（最近一个站的距离，而非 500m count）

→ 跑完 baseline 后，用 ablation 看哪些 feature 实质提升 metric（CPC / Spearman）。

---

## 7. Normalization summary

| 特征类型 | 处理 |
|---|---|
| 偏态分布（employment, population, POI count） | log(1+x) → z-score |
| 比例（sector proportion, POI category proportion） | 已在 [0,1]，直接用 |
| 计数（subway station count） | 小整数，直接用或 z-score |
| 坐标（lat, lon） | z-score |
| Congestion ratio | 已是相对值，z-score 或 log |
| Sin/cos encoding | 已在 [-1,1]，直接用 |

**关键 rule**：normalization 参数（mean / std）**只在训练集上 fit**，绝对不能 fit on full data → 防止 spatial / temporal leakage。

---

## 8. Implementation TODO

### Phase 1（伦敦试点）
- [ ] 写 `providers/features.py`：NodeFeatureProvider 接口 + LondonFeatureProvider 实现
- [ ] 1km 网格化伦敦 BRES（area-weighted MSOA → grid）
- [ ] 1km 网格化伦敦 Census 2021 population
- [ ] 1km 网格化伦敦 OSM POI（用 8 类映射）
- [ ] 1km 网格化伦敦 TfL stations
- [ ] OSM ↔ POI 8-类映射表（`providers/poi_category_mapping.py`）
- [ ] 合成 hourly workplace population（用 NTS / LTDS）
- [ ] Normalization fit 在伦敦训练集上

### Phase 2（北京阶段，伦理通过后）
- [ ] BeijingFeatureProvider 实现
- [ ] 年鉴 → 1km grid disaggregation pipeline（POI-weighted dasymetric）
- [ ] 高德 POI 8 类映射
- [ ] 北京 WorldPop 加载
- [ ] 北京 OD × 通勤时间分布合成 workplace population

---

## 9. Open questions（待小导师 / 后续 sprint 决定）

- [ ] 8 sector 的具体边界是否合理（特别是"information & finance"合并是否过粗）
- [ ] POI 8-类映射的精度 acceptable 标准（人工抽样验证多少 POI 可信）
- [ ] Workplace population 用 OD 合成的 limitation 是否要在论文标 limitation
- [ ] Land use 进 baseline 的争议（是否有跨城市对齐的快速方案推翻当前决策）

---

## 10. 参考

- `01_step1_gnn_decisions.md`（v1 文件夹，原 Decision 4 已 obsolete）
- `02_counterfactual_evaluation.md`（v1 文件夹，反事实评估三层证据链）
- `../README.md`（v2 项目说明）
- Yao et al. 2021 SI-GCN — 用 3 个属性（POI、population、housing），v2 加入 employment by sector 是 contribution edge
- WorldPop — `https://hub.worldpop.org/` 中国 + 英国 1km gridded population 公开
- BRES（UK NOMIS）— `https://www.nomisweb.co.uk/`
