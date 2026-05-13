# Dual-Branch ST-GNN + RUM 架构 API Contract

**目的**：派 agent 实现各模块前先把 API 钉死，避免 4 个 agent 写出 4 套不兼容接口（参考 `feedback_agent_team_lessons.md` 错 1）。

**版本**：2026-05-08

---

## 1. 数据约定

### 1.1 张量 shape 规范

| 名称 | shape | 含义 | 单位 |
|---|---|---|---|
| `X_static` | (N, F_static=22) | 不变特征：POI、就业、人口 | z-score |
| `X_dynamic` | (T=24, N, F_dyn) | 分小时特征：拥堵 + 入流 + 排队残留 + sin/cos | mixed (z-score) |
| `edge_index` | (2, E) | 1km 网格地理邻接（固定）| - |
| `edge_weight` | (E,) optional | 边权重 (e.g. exp(-t/τ)) | - |
| `t_ij` | (T, N, N) | 通勤时间（free-flow during training）| 分钟 |
| `log_d_ij` | (N, N) | log 距离 | log km |
| `F_ij_t` | (T, N, N) | 观测 hourly OD 流量 | 人次 |

**N**=1725（伦敦 1km 网格）/ TBD（北京）；**T**=24（一天 24 小时）

### 1.2 数据来源对应

| 张量 | 伦敦来源 | 北京来源 |
|---|---|---|
| `X_static` | demo_cache.npz `static_features` (1725, 22) | 待构 (基于 ss_city_grid_attr) |
| `X_dynamic[..., 0]` 拥堵 | demo_cache `borough_hourly_congestion` 映射到 grid | 高德路况手动截 |
| `X_dynamic[..., 1]` 每小时入流 | `data/processed/london_hourly_destination_inflow.csv` (MSOA 级) → 映射 grid | 分时到访人数 |
| `X_dynamic[..., 2]` 上小时排队残留 | BPR 训练时不用（free-flow），反事实时用 | 同左 |
| `X_dynamic[..., 3:5]` sin/cos hour | 直接生成 | 直接生成 |
| `F_ij_t` | GEODS 2019 hourly OD 映射到 grid (新构) | `o_d_beijing/*.csv` |

**单位 sanity check 强制规则**（参考 `feedback_data_sanity_check.md`）：
- 任何输入 tensor 进 trainer 前打印 shape + range + median
- `t_ij` 必须是分钟（中位数预期 30-50，最大 < 100），**不是** demo_cache 那个 cost composite
- 用 `car_freeflow_t_ij.npy`（真分钟）broadcast 到 (T, N, N)，**不是** demo_cache 的 t_ij_t

---

## 2. 模块接口

### 2.1 `StaticBranch`（不变特征那一路）

```python
class StaticBranch(nn.Module):
    """22 维不变特征 → 静态目的地表示

    输入: X_static (N, F_static)
    输出: A_static (N,) 或 (N, hidden_dim)
    """
    def __init__(self, node_dim: int, hidden_dim: int = 32, n_layers: int = 2,
                 layer_type: str = "sage"):
        ...

    def forward(self, X_static: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # returns: (N, hidden_dim)
```

实现：复用现有 `_SAGELayer`（structural_gnn.py 已有）。**不需要** GRU 或时间机制。

### 2.2 `DynamicBranch`（分小时特征那一路）

```python
class DynamicBranch(nn.Module):
    """分小时特征 → 动态目的地表示，含 GRU + multi-scale TCN

    输入: X_dyn (T, N, F_dyn)
    输出: A_dyn (T, N, hidden_dim)
    """
    def __init__(self, node_dim: int, hidden_dim: int = 32, n_sage_layers: int = 2,
                 gru_hidden: int = 32, tcn_kernels: tuple = (3, 5, 7)):
        ...

    def forward(self, X_dyn: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 内部:
        # 1. for t in range(T): h_t = SAGE(X_dyn[t]), 得 (T, N, hidden)
        # 2. h_seq = h_stack 通过 GRU 跨 hour，得 (T, N, gru_hidden)
        # 3. h_seq 同时也喂三个 dilated TCN (kernel 3/5/7)，concat
        # 4. fuse GRU 输出和 TCN 输出 → (T, N, hidden)
        # returns: (T, N, hidden_dim)
```

实现：
- SAGE 部分复用 `_SAGELayer`
- GRU: `nn.GRU(hidden, gru_hidden, batch_first=False)` —— 在每个 grid 上跨 T 跑（输入 shape (T, N, hidden)）
- TCN: 1D conv along time axis，dilation=1
- Fuse: 简单 concat + linear

### 2.3 `GatedFusion`（合并）

```python
class GatedFusion(nn.Module):
    """合并 static + dynamic → 最终 A_j(t)

    输入: A_static (N, hidden), A_dyn (T, N, hidden)
    输出: V_jt (T, N) — 每个 (hour, grid) 一个标量，给 RUM head
    """
    def __init__(self, hidden_dim: int):
        ...

    def forward(self, A_static: torch.Tensor, A_dyn: torch.Tensor) -> torch.Tensor:
        # gate: σ(W [A_static | A_dyn])，scalar per (t, j) 或 per-feature
        # combined = gate * A_static + (1 - gate) * A_dyn
        # head: linear → (T, N)
        # 内部 zero-mean unit-std 归一化（保持跟现有 StructuralGNN 一致）
        # returns: (T, N)
```

实现：先做 scalar gate（一个标量），简单。如果效果不好再改 per-feature gate。

### 2.4 `DualBranchEncoder`（顶层 encoder）

```python
class DualBranchEncoder(nn.Module):
    """组合 StaticBranch + DynamicBranch + GatedFusion，输出 V_jt
    
    输入: X_static (N, F_static), X_dyn (T, N, F_dyn), edge_index (2, E)
    输出: V_jt (T, N)
    """
    def __init__(self, static_dim: int, dyn_dim: int, hidden_dim: int = 32,
                 gru_hidden: int = 32):
        self.static_branch = StaticBranch(static_dim, hidden_dim)
        self.dyn_branch = DynamicBranch(dyn_dim, hidden_dim, gru_hidden=gru_hidden)
        self.fusion = GatedFusion(hidden_dim)

    def forward(self, X_static, X_dyn, edge_index, norm_stats=None):
        A_s = self.static_branch(X_static, edge_index)   # (N, hidden)
        A_d = self.dyn_branch(X_dyn, edge_index)          # (T, N, hidden)
        V_jt = self.fusion(A_s, A_d)                      # (T, N)
        # zero-mean unit-std 归一化（同现有 StructuralGNN）
        if norm_stats is None:
            mean_v, std_v = V_jt.mean(), V_jt.std() + 1e-6
        else:
            mean_v, std_v = norm_stats
            std_v = std_v + 1e-6
        return (V_jt - mean_v) / std_v
```

### 2.5 `DualBranchInverseTrainer`（trainer）

复用现有 `_RUMHead`（不动）。修改 `_forward_V` 适配双输入。新加 trainer 类：

```python
class DualBranchInverseTrainer:
    """双路 encoder + RUM head 的 inverse trainer。

    跟 InverseRUMTrainer 区别：
    - 接受 X_static (N, F) + X_dynamic (T, N, F') 两个 input
    - 内部用 DualBranchEncoder 替代 StructuralGNN
    - 其余逻辑（loss / CPC / early stop / RUM head 各种 interactions）完全复用
    """
    def __init__(self, X_static, X_dynamic, edge_index,
                 observed_OD, t_ij_t, log_d_ij, ...):
        ...
```

---

## 3. Forward 流（端到端）

```
X_static (N, 22) ─→ StaticBranch (GraphSAGE) ─→ A_static (N, hidden)
                                                                    │
X_dynamic (T, N, F_dyn) ─→ DynamicBranch ─→ A_dyn (T, N, hidden) ──┤
       (per hour SAGE → GRU → multi-scale TCN → fuse)               │
                                                                    │
                                                          GatedFusion
                                                                ↓
                                                        V_jt (T, N)
                                                                ↓
                                                       _RUMHead (not changed)
                                                                ↓
                                            logits[t,i,j] = α V_jt - β t_ij - γ log_d
                                                                ↓
                                              softmax over j → log P(j|i,t)
                                                                ↓
                                          cross-entropy with F_ij_t (hourly OD)
```

---

## 4. 何时 commit 这份 contract

写代码前 **不 commit**。代码跑通 + smoke test 过后再 commit（防止下次 review 时这份 markdown 已经过时）。
