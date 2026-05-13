# Edge Features 与图拓扑（v2）

> **目的**：定义 v2 STGNN 的图拓扑（哪些 grid 相连）+ 边特征 inventory。
>
> **关系**：本文档**取代** `../../02_london_commuting_model/methodology/01_step1_gnn_decisions.md` 里的 Decision 3（边构建）。原 Decision 3 是 MSOA + Gaussian-decay scalar weight，5-4 改为 1km grid + 多维 edge feature vector 后已 obsolete。
>
> **配套文档**：`03_node_features.md`（节点特征）、`02_counterfactual_evaluation.md`（反事实评估）。

---

## 1. 决策约束

- **Beijing-friendly**：每个边特征必须在伦敦和北京都有等价数据
- **1km × 1km 网格**：边在 grid 之间，不是 MSOA / 路口
- **Static edge features**：边特征不随时间变（时变信息走节点 GRU + RUM 的 β·t_ij(t)）
- **有向图**：(i→j) 和 (j→i) 是两条不同的边

---

## 2. 决策汇总表（locked）

### 2.1 图拓扑

| 决策 | 锁定 |
|---|---|
| 拓扑构建方式 | **kNN(K=10) by Euclidean distance** |
| 度（avg） | 10 (近似常数，所有 grid 都连到最近 10 邻居) |
| 自环 | 包含（GCN 标准做法） |
| 方向性 | **有向**：edge_index 包含 (i,j) 和 (j,i) 两条 |
| K 消融 | K ∈ {5, 10, 15} |

### 2.2 Baseline 边特征（5 项 → 7 维）

| # | 特征 | 维度 | 类型 | 伦敦来源 | 北京来源 |
|---|---|---|---|---|---|
| 1 | Euclidean distance d_ij | 1 | 连续 | grid centroid | grid centroid |
| 2 | Free-flow travel time t⁰_ij | 1 | 连续 | BPR(congestion=0) | 高德路径规划"理想时间" |
| 3 | Gaussian decay weight | 1 | 连续 | exp(−d²/σ²) | 同 |
| 4 | Subway directly connected flag | 1 | 二值 | TfL 线路 + station 1km buffer | 北京地铁 + 站点 1km buffer |
| 5 | Mode availability mask（walk / bus / subway） | 3 | 二值 | TfL routes + OSM | 高德公交 + 北京地铁 |

**总边特征维度：7**

### 2.3 Ablation 边特征（不进 baseline）

- 公交线路数（continuous count）
- 同行政区 flag（borough/district）
- 跨水/跨山 flag

---

## 3. 拓扑设计详细说明

### 3.1 为什么 kNN(K=10)

**对比候选方案**：

| 方案 | 含义 | 1km 网格下的特性 |
|---|---|---|
| Queen contiguity | 共享边界（1-shell ≈ 8 邻居） | 太稀疏；长程通勤（30 km 跨城）GNN 学不到 |
| **kNN(K=10)** | 距离最近 10 个 grid | ✅ 局部 + 适度长程；度数恒定 |
| Distance threshold (d ≤ 5km) | 半径阈值 | 中心区度数爆炸（100+），外围孤立 |
| Hybrid Queen ∪ kNN(8) | 局部 + 全局 | 实现复杂；degree 不均匀 |
| Road network adjacency | 路网拓扑 | 跟方案 3（路网级 GNN）耦合，已 reject |

**Why K=10**：
- 在 30×30 grid 城市里，K=10 ≈ 1-2 圈邻居 + 几个稍远 grid，覆盖近距离通勤主体
- 文献常用范围 K ∈ [5, 20]，K=10 是中位
- Yao 2021 用 1-shell 邻接，其实跟 Queen 类似——v2 的 K=10 比他们更全局

**消融**：训练时跑 K ∈ {5, 10, 15}，看 OOS metric（CPC、Spearman）的敏感性。

### 3.2 自环 vs 不自环

包含自环（每个 grid 连自己）。GCN 数学上 `Â = A + I` 是标准操作，让节点能"保留自己的 feature"。PyG `add_self_loops=True` 默认开启。

### 3.3 有向 vs 无向

**有向**——通勤本身有方向：
- 早高峰 i→j（住宅→工作）的拥堵 ≠ j→i 拥堵
- 实现：edge_index 包含 (i,j) 和 (j,i) 两列，对应两份 edge_attr
- 上下行 free-flow time 通常相同，但**带 congestion 的 t_ij(t)**（不在 edge feature 里，在 RUM）会不对称

**Note**：当前 baseline 边特征都是对称的（d_ij = d_ji，subway flag 对称，etc.）。**有向性**主要体现在未来动态扩展（dynamic congestion-aware edge features）。

---

## 4. 边特征详细说明

### 4.1 Euclidean distance d_ij

**计算**：grid centroid 之间直线距离（米或公里）。

**Why**：
- 距离衰减是通勤选择最强的 prior
- GNN 不天然知道"远近"，必须显式喂
- 配合 Gaussian decay 形成 distance-aware 注意力

**Normalization**：log(1+x) → z-score（fit on training set）

### 4.2 Free-flow travel time t⁰_ij

**伦敦实现（BPR free-flow）**：
```
t⁰_ij = d_ij × circuity_factor / avg_speed
       = d_ij × 1.3 / 20 km/h    （v1 的参数）
```

**北京实现（高德）**：
- 调用高德路径规划 API，参数设 `strategy=0`（速度优先）+ 在凌晨/低峰时段查询
- 或直接拿"理想时间"字段（如 API 提供）

**Why 不用动态 t_ij(t)**：
- 动态 t_ij(t) 走 RUM 的 β·t_ij(t) 项
- Edge feature 是静态结构特征，"自由流时间"反映"如果没拥堵"的网络几何
- 拥堵效应通过节点 hourly congestion + RUM closure 捕捉

**Normalization**：log(1+x) → z-score

### 4.3 Gaussian decay weight

**公式**：`w_ij = exp(−d_ij² / σ²)`

**σ 选择**：grid-pair 距离的中位数（fit on training set）。

**Why**：
- 强 distance prior，让 GCN message passing 自动按距离衰减加权
- LeSage & Pace 2009 spatial econometrics 标准做法
- 即使 K=10 把"远邻居"也连上，Gaussian decay 让远邻居 message 自动变弱

**Normalization**：已在 (0, 1] 范围，直接用。

### 4.4 Subway directly connected flag

**定义**：grid i 1km buffer 内的地铁站点 + grid j 1km buffer 内的地铁站点 → 是否在**同一条线路**上（不需换乘）。

**伦敦**：TfL Open Data 提供 line × station 表 + station 经纬度。
**北京**：北京地铁公开线路图 + station 经纬度。

**Why**：
- 直达地铁是通勤偏好的强信号（少换乘）
- 跟 mode availability mask 互补——availability 只说"有 vs 没有"，直达 flag 说"方便程度"

**Normalization**：二值，无需变换。

### 4.5 Mode availability mask（3-bit）

**定义**：每条边记录 [walk, bus, subway] 三个 bit：
- walk：d_ij ≤ 1 km（步行可达）
- bus：i 和 j 各自 500m 内有公交站，且属于至少一条共同线路
- subway：i 和 j 各自 1km 内有地铁站（无论是否同线）

**Why**：
- 给 GNN 显式信息"这条 OD 对能用哪些 mode"
- 影响后续 RUM 的 mode choice（v2.2 计划）
- 北京 / 伦敦都能生成

**Normalization**：3 个二值，no normalization。

---

## 5. PyG 实现 sketch

```python
# providers/graph_builder.py

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

def build_grid_graph(
    grid_centroids,  # (N, 2) lat/lon
    K=10,
    sigma=None,
    static_metadata=None,  # subway_lines, bus_routes, etc.
):
    # 1. kNN by Euclidean
    edge_index = knn_graph(grid_centroids, k=K, loop=False)
    # edge_index shape: (2, N*K)

    # 2. Compute edge features
    src, dst = edge_index
    d_ij = torch.norm(grid_centroids[src] - grid_centroids[dst], dim=1)

    if sigma is None:
        sigma = d_ij.median()
    decay = torch.exp(-(d_ij ** 2) / (sigma ** 2))

    t0_ij = compute_freeflow_time(d_ij)  # BPR or Amap

    subway_flag = compute_subway_directly_connected(src, dst, static_metadata)
    mode_mask = compute_mode_availability(src, dst, static_metadata)  # (N*K, 3)

    # 3. Stack edge features
    edge_attr = torch.cat([
        d_ij.unsqueeze(1).log1p(),
        t0_ij.unsqueeze(1).log1p(),
        decay.unsqueeze(1),
        subway_flag.unsqueeze(1).float(),
        mode_mask.float(),  # (N*K, 3)
    ], dim=1)
    # edge_attr shape: (N*K, 7)

    # 4. Add self-loops
    edge_index, edge_attr = add_self_loops(
        edge_index, edge_attr, fill_value=0., num_nodes=N
    )

    return edge_index, edge_attr
```

边特征喂给 GNN 的方式（以 GAT 为例）：

```python
import torch_geometric.nn as gnn

class V2_STGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.gat1 = gnn.GATv2Conv(
            node_dim, hidden_dim, edge_dim=edge_dim, heads=4
        )
        self.gat2 = gnn.GATv2Conv(
            hidden_dim * 4, hidden_dim, edge_dim=edge_dim, heads=1
        )
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)  # → V_j

    def forward(self, x_seq, edge_index, edge_attr):
        # x_seq: (T, N, node_dim) — T 时间步
        h_seq = []
        for t in range(x_seq.size(0)):
            h = self.gat1(x_seq[t], edge_index, edge_attr).relu()
            h = self.gat2(h, edge_index, edge_attr)
            h_seq.append(h)
        h_seq = torch.stack(h_seq)  # (T, N, hidden_dim)

        # GRU 在时间维度
        h_temporal, _ = self.gru(h_seq)  # (T, N, hidden_dim)

        V_jt = self.out(h_temporal)  # (T, N, 1) → V_j(t)
        return V_jt
```

---

## 6. Normalization summary（边特征）

| 特征 | 处理 |
|---|---|
| d_ij | log(1+x) → z-score |
| t⁰_ij | log(1+x) → z-score |
| Gaussian decay | already (0, 1]，no normalization |
| Subway flag | binary, as-is |
| Mode availability | binary 3-bit, as-is |

**关键 rule**：log + z-score 的参数（mean / std）只在训练集 fit。

---

## 7. Implementation TODO

### Phase 1（伦敦试点）
- [ ] 写 `providers/graph_builder.py`（PyG sketch 见 §5）
- [ ] kNN(K=10) 构建（torch_geometric.nn.knn_graph）
- [ ] Free-flow time 计算（BPR with congestion=0）
- [ ] Subway directly connected flag（TfL line × station 数据 + 1km buffer 判断）
- [ ] Mode availability mask（OSM walk / TfL bus / TfL subway 综合）
- [ ] Edge feature normalization fit on training set
- [ ] 单元测试：可视化几个 OD pair 的 edge features，sanity check

### Phase 2（北京阶段）
- [ ] BeijingGraphBuilder 实现
- [ ] 高德路径规划"理想时间"批量爬取（用作 t⁰_ij ground truth，可选）
- [ ] 北京地铁线路 + station 数据 ingestion
- [ ] 高德公交数据 ingestion
- [ ] OSM 北京 walk / road network 加载

---

## 8. Open questions（待小导师 / 后续 sprint 决定）

- [ ] K=10 是否经得起消融（vs K=5/15）
- [ ] Subway directly connected 的 1km buffer 半径（vs 500m / 1.5km）
- [ ] Mode availability 是否需要 PT 频率维度（不仅仅是 yes/no）
- [ ] 自由流时间是否要 mode-specific（car/walk/subway 各一份），还是通用 mean

---

## 9. 参考

- `01_step1_gnn_decisions.md`（v1 文件夹，原 Decision 3 已 obsolete）
- `03_node_features.md`（节点特征清单）
- `02_counterfactual_evaluation.md`（反事实评估）
- LeSage & Pace 2009 *Introduction to Spatial Econometrics* — Gaussian decay 标准做法
- PyG 官方文档 `knn_graph`、`GATv2Conv` 接口
- Yao et al. 2021 SI-GCN — 1-shell 邻接 baseline，v2 用 K=10 是扩展
