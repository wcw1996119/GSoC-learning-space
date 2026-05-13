# London Full Model v2

> Mesa ABM + STGNN + RUM 的混合模型。**伦敦只是开发沙盒**——目标是把架构搭出来，等北京数据伦理审批通过后迁移过去（北京才是主数据集）。

**状态**：搭建中（2026-05-04 启动）

---

## 与 v1 的关系

| | v1 | v2 |
|---|---|---|
| 路径 | `../02_london_commuting_model/` | 本文件夹 |
| 状态 | HF 已部署的简化通勤模型，**不动** | 开发中 |
| GNN | 无 | TSI-GCN（T-GCN 起步） |
| Mode | 固定采样属性 | RUM 决策 |
| 拥堵 | TomTom borough 级 | 同（伦敦），北京会换成高德爬虫 |
| 空间单元 | MSOA（983 个） | 1km × 1km 网格 |
| 角色 | 公开 demo | 论文方法学开发 |

**v1 不删不改**——继续作为 HF 公开 demo 跑。

---

## 架构

完整设计文档（在 v1 文件夹下，未来可能迁移过来）：
- `../02_london_commuting_model/methodology/01_step1_gnn_decisions.md` — STEP 1 GNN 7 个 methodology 决策
- `../02_london_commuting_model/methodology/02_counterfactual_evaluation.md` — 反事实评估三层证据链

### 核心选择（2026-05-04 定型）

- **单 GNN**：TSI-GCN，T-GCN 起步（GCN spatial + GRU temporal）
- **空间单元**：1km × 1km 网格（对标 Yao 2021）
- **图设计**：做法 B = 地理邻接 + 边特征向量 [通勤时间, 班次数, 地铁直达 flag, ...]
- **RUM 闭合**：P(j|i, t) = exp(V_j(t) − β·t_ij(t)) / Σ_k exp(...)
- **t_ij 来源**：伦敦 BPR / 北京高德 API（接口抽象，可换 implementation）
- **反事实场景**：A 副中心扩容（空间）+ B 灵活办公（时间）
- **反事实评估**：plausibility 距离 + magnitude scan + multi-model cross-check

### 关键 Beijing-friendly 约束

伦敦框架开发时**不能依赖伦敦专属数据**——否则迁移到北京无等价数据。

| v1 用的 | 北京无等价 → v2 怎么处理 |
|---|---|
| Census 2011 mode shares (MSOA11CD) | ❌ 北京年鉴只到 aggregate → mode 改为 RUM 决策 |
| ASHE workplace income | ❌ 北京只到行业级平均工资 → 简化或 POI proxy |
| TomTom borough 拥堵 | ✅ 北京有道路级高德爬虫，更细 → 接口抽象 |
| BRES 18 产业就业 | ⚠️ 年鉴有但粒度粗 → POI 替代或补充 |
| JTS 通勤时间基准 | ❌ 北京没有 → 用 baseline 模型 cross-check 替代 |
| TS063 SOC 9 类职业 | ⚠️ 中国 GBM 分类不同 → 接口抽象 |

---

## 文件结构（计划）

```
03_london_full_model_v2/
├── data/
│   ├── raw/             # 共享 v1 raw 数据（symlink 或 copy）
│   └── processed/       # v2 自己处理的（1km 网格化后）
├── providers/           # 数据源抽象层（伦敦/北京可换）
│   ├── travel_time.py   # TravelTimeProvider（BPR / Amap）
│   ├── attractiveness.py # AttractivenessProvider
│   └── features.py      # NodeFeatureProvider
├── models_lib/          # STGNN + RUM（不用 models/ 避免和 mesa 命名冲突）
│   ├── stgnn.py         # T-GCN 架构
│   └── rum.py           # RUM closure
├── methodology/         # v2 自己的方法学文档（暂时空，看是否从 v1 迁移）
├── tests/
├── train.py             # STGNN 训练入口
├── evaluate.py          # 反事实评估（实施 methodology/02 三层证据）
├── model.py             # Mesa 模型主入口（agent + scheduler）
├── agents.py            # CommuterAgent
├── README.md
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

---

## 环境

新建 `mesa-gnn` conda 环境（避免污染 v1 的 `mesa-demo` HF 部署 env）：

```bash
conda create -n mesa-gnn python=3.11 -c conda-forge
conda activate mesa-gnn
conda install pytorch torch-geometric -c pyg -c pytorch
pip install mesa pandas geopandas scikit-learn matplotlib shapely pyproj pyogrio
```

---

## 当前 TODO（按优先级）

### Phase 1: Design 收尾
- [ ] 节点特征清单（伦敦 + 北京 align）
- [ ] 边特征清单
- [ ] STGNN 训练 loss 具体形式（log-likelihood vs MSE）
- [ ] Provider 接口签名定义

### Phase 2: 数据准备
- [ ] 1km 网格化伦敦数据（v1 是 MSOA）
- [ ] 网格化后的 OD / employment / population pipeline
- [ ] 高德 API 调用 wrapper（伦敦不用，但接口要先定）

### Phase 3: 模型实现
- [ ] Provider 抽象层
- [ ] STGNN（T-GCN）实现
- [ ] RUM 闭合
- [ ] Mesa agent 包装

### Phase 4: 训练 + 评估
- [ ] 训练 pipeline + spatial holdout
- [ ] Plausibility 实现
- [ ] Magnitude scan 实现
- [ ] Cross-check baseline（gravity, pure RUM）

### Phase 5: 反事实情景（伦敦试点）
- [ ] 模拟"副中心"类场景（伦敦无对应，用某个 plausible 候选地块）
- [ ] 模拟"灵活办公"场景

---

## 不在本 repo 范围内（北京阶段）

- 北京 OD 数据接入（伦理通过后）
- 高德路径规划批量爬虫
- 北京统计年鉴数据 ingestion
- 北京 POI 数据接入

→ 这些都是**北京阶段**才做的事。本 v2 只做"伦敦框架"，但**接口必须 Beijing-friendly**。
