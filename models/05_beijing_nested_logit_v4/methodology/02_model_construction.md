# 北京 v4:模型是怎么建起来的(对应代码逻辑)

(2026-06-06 整理。本文逐层对应 `data/scripts/*.py`(数据)、`experiments/beijing_model.py`(模型)、
`experiments/train_beijing.py`(训练),不讲"为什么可信"(那在 [01_identifiability_reliability.md])
而讲"具体怎么搭的、每个符号从哪来"。读这份能照着代码对上每一行。)

## 0. 一句话核心

把北京 16907 个格子之间"谁往哪通勤"建成一个**按出发地分段的 softmax 分类器**:
每个出发地是一道选择题(去哪个格子上班),目的地效用 = 嵌套 logit 的方式打包值(怎么去)
+ 引力/工资/竞争(目的地多好)+ 一层神经网络残差 + 一层词典筛选;参数用文献符号约束,
两个聚合数据识别不出的行为参数(方式、职业)靠外部数据锚进来。

整条流水线:

```
原始数据(只读, 不出库)
  PaperC_local/{cell_index, t_matrix, od_cell_flow, occ_share, P_occ_given_ind, xj_industry}
  congestion_project/{刷卡, 小区房价, 公交线网}
        │  builders (data/scripts/build_beijing_*.py)
        ▼
处理后 npz (data/processed/, gitignore)
  beijing_grid.npz   格子层 (district/xy/jobs/residents)
  beijing_edges.npz  稀疏候选集 30.4M 边 + 4 时段观测流量
  beijing_modes.npz  per-mode 时间 (步行/公交) + 方式分担先验
  beijing_aux.npz    引力/工资/竞争/收入档/职业 (v3 同构 aux)
  beijing_transit_od.npz  刷卡公交 OD (mode 去向锚)
        │  build_data() 装进一个 data dict
        ▼
模型 (beijing_model.py)
  BeijingPairEncoder  双路 NN: GraphSAGE-lite -> 候选 edge bilinear V_NN
  BeijingNestedHead   嵌套 logit head: 上 mode 下 destination
        │  train_beijing.py: 4 时段加权 CE + 双锚(Ben-Akiva-Morikawa) + CPC
        ▼
evaluation_outputs/v4_full_anchored_s{0,1,2}.pt  (CPC 0.6926±0.0083)
```

---

## 1. 数据层:5 个 npz 怎么造出来的

每个 builder = 读只读原始数据 → 算 → 存一个 npz。绝对路径只读引用,不复制进库(governance 红线)。

### 1.1 `beijing_grid.npz` ← `build_beijing_grid.py`
- **输入**:`PaperC_local/cell_index.parquet`(16907 格)。
- **算**:经纬度 → 投影米坐标 `xy_m`;格子归属 16 区 `district_idx`;`jobs`/`residents` 总量。
- **输出字段**:`coords_latlon, xy_m, district_idx, district_names, jobs, residents`。
- **用途**:NN 的节点特征底座 + self-loop 的 district 索引 + 引力的岗位总量。

### 1.2 `beijing_edges.npz` ← `build_beijing_choiceset.py`(稀疏候选集,地基)
- **输入**:`t_matrix.parquet`(30.4M 行:`o_idx,d_idx,hav_km,t_obs_min,t_ff_min`)+
  `od_cell_flow.parquet`(7.06M 行:`f_ampeak/f_pmpeak/f_midday/f_night`)。
- **算**:把候选对(谁能到谁)和观测流量对齐;边**按出发地 `o` 排序**(CSR 关键,后面分段全靠这个)。
- **输出字段**:`o_idx, d_idx, N, t_obs, t_ff, log_d, flow`(`flow` 是 `(E,4)`,4 时段)。
- **量级**:~30.4M 边,均值 2674 候选/出发地,覆盖 97.2% 观测流。
- **为什么这是地基**:16907² 稠密放不下 → 只存"有候选的边",所有 softmax 改成**按出发地分段**(见 §3)。

### 1.3 `beijing_modes.npz` ← `build_beijing_modes.py`(公交时间长杆)
- **car**:不在这里存,在 `train_beijing.py:83` 按时段从 `t_obs/t_ff` 拼:
  `ampeak/pmpeak → t_obs`(拥堵)、`night → t_ff`(自由流)、`midday → 0.5(t_obs+t_ff)`(插值)。
- **walk**:`hav_km / 步速`,封顶。
- **transit**:刷卡实测标定的公交速度 `V_BUS_KMH = 13.4`(含地铁的代理)→ 距离档公交时间 + 等待。
- **输出字段**:`t_walk, t_transit, mode_share, v_bus_kmh`。
- **mode_share**:北京公开方式分担先验(按距离档),给方式选择当软先验。

### 1.4 `beijing_aux.npz` ← `build_beijing_aux.py`(v3 同构 aux,目的地效用全在这)
汇总成跟伦敦 v3 一模一样的 aux 结构,模型 head 直接吃:
- **引力 `log_M_z`**:`jobs` 取 log 后 z-score(typed-mass 时进一步按本职业岗位 `M_j^o`)。
- **工资 `log_W_z`**:房价代理收入 → 目的地工资水平。
- **竞争 `log_D_z`**(Shen accessibility):residents/jobs 推竞争强度。
- **收入 `income_z` + `income_tier_props`**:出发地居民收入档(低/中/高)软占比。
- **职业 `soc_props`(出发地 7 职业构成)+ `demand_share`(目的地 7 职业岗位需求占比)**。
- **`match`**:非 soc 混合时的标量职业匹配信号。

### 1.5 `beijing_transit_od.npz` ← `build_beijing_transit_od.py`(mode 去向锚)
- **输入**:`Public Transit Smart Card Data/20190506.csv`(10.6M 刷卡行)。
- **算**:刷卡 → 格子级公交(含地铁)OD。
- **输出字段**:`to_o, to_d, to_n`。
- **用途**:训练时的 `--anchor-transit`,把模型预测的公交流量**空间分布**对齐刷卡真实分布。

> 职业的 finer-geo(2008 经普街道行业 → cell 级 + 2023 AOI 校准)由
> `build_beijing_street_industry / cell_industry(_cal).py` 生成,喂进 `demand_share`,
> 细节见 [01_identifiability_reliability.md] §4.2,这里不重复。

---

## 2. 数据契约:`build_data()` 把 npz 装成一个 dict

`train_beijing.py:34-111`。关键动作:

1. **CSR 分段**(`:78-82`):边已按 `o` 排序 → `np.unique` 出每个出发地的首边 `seg_starts`,
   `seg_id` 给每条边标"属于第几个出发地"。**这是稀疏 softmax 的命脉**。
2. **per-period car 时间**(`:83`):`t_car = {0:t_obs, 1:t_obs, 2:0.5(t_obs+t_ff), 3:t_ff}`。
3. **NN 节点特征 `Xnode`**(`:85-86`,8 维):
   `[log_M, log_D, income, log_W, zlog(jobs), zlog(residents), xy_x/1e5, xy_y/1e5]`。
4. **NN 图 `edge_index`**(`:88-93`,仅 `--use-nn`):空间 kNN(k=8)对称图,给 GraphSAGE 传播用。
5. **验证集划分**(`:108-110`):**按出发地整段划**(`val_frac=0.15`),
   `edge_is_val = val_seg[seg_id]` —— 一个出发地要么全训练要么全验证,**不在边级泄漏**。

`make_batch()`(`:114-130`)按 chunk + 时段切出一批边,把目的地侧特征 gather 到边上
(`log_M_d = log_M[d]` 等),`t_min` = 三方式最短时间(给词典筛选的成本/时间门用)。

---

## 3. 稀疏机制:dense softmax 怎么变成 segment 版

这是整个北京迁移的技术核心,三件套:

### 3.1 `segment_logsumexp`(`beijing_model.py:35-41`)
伦敦 `log_softmax(dim=-1)` 是在稠密 `(N,N)` 的目的地维上做。北京没有这个维,改成
**按 `seg_id` 分组**做 logsumexp:先 `scatter_reduce(amax)` 求每段最大值稳数值,再 shift-exp-scatter_add。
`logP_c = V_dest - segment_logsumexp(V_dest)[seg_id]` —— 等价于"在每个出发地的候选集里 softmax"。

### 3.2 origin-chunking(`:133-141` + 训练循环)
`--origin-chunks G` 把出发地切 G 块,每块**单独前向/backward**。因为边按 `o` 连续(CSR),
分块在出发地边界上切是**精确无近似**的(不会把一个出发地的候选集切两半)。显存降到 1/G,
soc 混合(21 类)+ 大网格必须用(anchored 全量用 `G=16`)。

### 3.3 逐类增量 logaddexp(`:215-242`)
tier(×soc)混合本来要堆 `(E, n_classes)` 的张量,显存炸。改成**逐类累加** `torch.logaddexp(acc, term)`,
永远只持有 `(E,)`。代价是 Python 循环 n_classes 次,换显存。

---

## 4. 模型 forward:目的地效用一项一项怎么搭

`BeijingNestedHead.forward`(`:188-243`)。对每个 class `c`(收入档 tier,soc 混合时 = tier×soc):

**架构一眼总览(三块 + 结合方式)** —— 整个模型 = 给每个出发地在候选目的地上做一次 softmax,每条边效用:
```
V_dest = V_M + V_other + λ·IV(mode)  +  self_loop  +  w_nn·V_NN  +  词典筛选 log-mask
         └──── RUM 结构(4.1-4.2,看得懂)────┘   └同格┘  └GNN(4.5)┘  └consideration(4.3)┘
logP   = V_dest − segment_logsumexp(同出发地所有目的地)     # 目的地 softmax,再按 tier×soc 混合(4.4)
```
- **RUM(§4.1-4.2)** = 两层嵌套:下层方式 IV → 上层引力/工资/竞争;参数 per 教育档×职业、全 sign-constrained。职业经 typed-mass 进引力(`V_M`),也经 L1 进词典筛选。
- **GNN(§4.5)** = 双路 ST-GNN,出每条边一个标量 `V_NN`;**决策变量(时间/岗位/工资)不喂它,留给 RUM**(doc 04 边界)。
- **结合方式 = 加性残差**:`w_nn·V_NN` 直接加到 RUM 效用上,`w_nn` 是**单个标量闸门**。**不是**凸组合 `(1−δ)V_RUM+δV_NN`(Wang TB-ResNet)、**不是**乘性。无锚时 w_nn≈0.09,强锚时塌到 ≈0(诚实归零,见 [01] §8 / memory)。

### 4.1 下层:mode 嵌套 inclusive value(怎么去)
对车/公交/步行三方式各算下层效用,再打包成一个 IV:
```
V_lower_m = (β_t0_m + β_t1_m·log_d)·t_m + asc_m + θ_inc_m·income_o      # 距离相关的时间敏感
log_iv    = logΣ_m exp(V_lower_m / λ)                                   # (:199) 嵌套 inclusive value
```
- `β_t0, β_t1 < 0`(时间越长越不愿;`β_t1·log_d` 让长途每分钟更值钱)。
- `λ ∈ [0.05, 1]`,→1 趋近独立选择,<1 方式内部相关。北京学到 ≈0.56(嵌套真起作用)。
- `t_m`:车按时段、公交/步行固定。

### 4.2 上层:目的地效用(那地方多好)
```
# 引力项, 两种 formulation:
typed-mass(默认全量):  V_M = γ_M_tier · (log_M_d + log match_sig)       # (:231) 引力作用在本职业岗位 M_j^o
否则:                  V_M = γ_M_tier · log_M_d + δ_c · match_sig       # (:233) 总引力 + 独立 δ 修正

V_other = α_W_tier · log_W_d + ν_D_tier · log_D_d                       # 工资吸引 + 竞争排斥
V_dest  = V_M + V_other + λ · log_iv                                    # (:235) 上层 = 引力 + 打包的方式值
```
- `γ_M > 0`(岗位多更吸引)、`α_W ≥ 0`(工资高更吸引)、`ν_D < 0`(竞争者多更不吸引)。
- 三个系数都 **per 收入档**(低/中/高各一套)。

### 4.3 三个可选加项
```
+ self_loop[district_o, period]    # (:206-208) 仅同格通勤(is_self), 按区×时段的同城自雇 boost
+ w_nn · V_NN                       # (:210-212) NN 残差, w_nn>0, gnn-mode residual
+ _consideration_mask(...)          # (:238) 词典筛选三层 log-mask
```

**词典筛选三层**(`_consideration_mask`, `:157-177`),每层一个软 sigmoid 的 log-mask 相加:
- **L1 匹配门**:`logsigmoid(k_match·(职业需求信号 - τ))` —— 职业不匹配的目的地被软压。
  - ⚠ 自由阈值会塌陷到不筛(冗余, 被 typed-mass 引力吃掉)。**主 spec 用 soft-lex**(`--soft-lex-match`: τ_s≥floor=0.5×mean + k_s≥20),
    3-seed 学出差异化筛除(蓝领制造 68-82% / 白领~0%, CPC 中性) → 职业筛选才真工作。见 [01_identifiability §7.5]。
  - 职业**双机制**: 排序(typed-mass 引力 §4.2, 全职业) + 筛选(本门 soft-lex, 仅蓝领), 不重复。
- **L2 成本门**:`cost = θ_t·t_min + θ_d·log_d`,`logsigmoid(k_cost·(阈 - cost/budget_tier))` —— 太贵的软压。
- **L3 时间门**:`logsigmoid(k_time·(T_max_tier - t_min))` —— 超过通勤忍受度 `T_max`(Bhat 三档 ~40/60/90 起)软压。

### 4.4 混合:tier(×soc)加权
```
log_pi_c = log tier_props_o[tier]  (+ log soc_props_o[soc])             # 出发地的档/职业软占比
term     = log_pi_c + (V_dest - segment_logsumexp(V_dest)[seg_id])      # (:241) 该 class 的 log 概率
logP     = logaddexp 累加所有 class                                     # (:242) 混合
```
返回 `(E,)` 的 `logP` —— 每条边"从这个出发地选到这个目的地"的对数概率。

### 4.5 NN 编码器:双路 ST-GNN(`BeijingDualBranchEncoder`)/ 静态版(`BeijingPairEncoder`)(`beijing_model.py:58-160`)
**双路 ST-GNN(`--use-dynamic`,当前 edu/full 在用)**:
- **静态腿**:静态空间特征 → `s_in` → 2 层 GraphSAGE-lite(`SageLayer` = self 线性 + 邻居均值聚合线性)。
- **动态腿**:逐小时 pop/拥堵序列 → 每小时过 GraphSAGE → **GRU**(沿 24h 时间)+ **多尺度 TCN**(kernel 3/5/7,circular padding 让 24h 首尾环绕)→ `dfuse` 融合 → 24 小时**池化到 4 时段**。
- 静态 + 动态拼接 → 每时段 origin/dest 嵌入(16 维);`edge_vnn` = `(e_o[period][o] · e_d[period][d]).sum()` = 每条边一个标量 `V_NN[period]`。
**静态版(不开 `--use-dynamic`)**:只有静态腿,`V_NN` 时段无关。
- 进 head 当**加性残差** `w_nn·V_NN`(见上总览)。NN first-class(ablation 关掉掉 CPC),但**决策变量不喂 GNN,留给 RUM**——反事实可信靠这条边界(doc 04;CV 236%→4%)。

### 4.6 sign 约束怎么实现的(`:138-155`)
不直接 clamp,而是**重参数化**:`β_t = -softplus(raw)`(恒负)、`λ = 0.05+0.95·sigmoid(raw)`(恒 (0,1])、
`γ_M/α_W/δ/T_max = softplus(raw)`(恒正)、`ν_D = -softplus(raw)`(恒负)。
梯度光滑,符号永不越界 —— 这就是文献符号约束的落地方式。

---

## 5. 训练:loss + 双锚 + CPC

`train_beijing.py:219-294`。

### 5.1 主损失:4 时段加权交叉熵(`:227-236`)
```
loss_p = -(flow[:,p] · train_mask · logP).sum() / flow_train_sum       # 每时段 CE, 观测流量当权重
```
逐 chunk 逐时段 backward(省显存),只在**训练边**上算。本质 = 用真实 OD 流量当目标的多类 softmax 分类。

### 5.2 双锚:Ben-Akiva-Morikawa 联合估计(`:237-250`)
聚合 OD 识别不出 mode → 从外部带标签数据补,在 ~2M 边的**固定采样**上算(脱离全局归一,兼容 chunking):
- **去向锚**(`--anchor-transit`):预测公交流量分布 vs 刷卡 transit OD 分布,CE。定"公交去哪"。
- **份额锚**(`--anchor-share`):流量加权方式份额 vs 目标 `[0.21,0.35,0.44]`,CE。定"方式份额水平"。
- 锚走 `head.mode_logits(asb)` 独立 forward(不依赖 NN/segment),独立 backward。

### 5.3 CPC(`:254-276`)
每 10 epoch 在**验证边**上算 edge 级 Sørensen:
```
pred  = exp(logP) · 出发地观测总量[seg_id]          # 概率 × 该出发地总流 = 预测流
CPC_p = 2·Σ min(pred, obs) / (Σ pred + Σ obs)        # 每时段
CPC   = 4 时段按验证流量加权平均
```
存 `best_cpc`。注意 pred 用"出发地观测总量"还原绝对流量 —— 模型只管**条件分布**(往哪分),
总量是给定的,这是 trip-distribution 的标准做法。

### 5.4 落盘(`:289-294`)
`.pt` 存 `head_state + enc_state + args(vars) + param_report + best_cpc`。
`args` 是 single source of truth —— 复现/分析全靠 `torch.load(pt)["args"]`(见 baseline 卫生教训)。

---

## 6. flag → 模型形态 映射(一眼看懂每个开关变什么)

| flag | 默认 | 打开后 | 代码位置 |
|---|---|---|---|
| `--use-nn` | 关 | 加双路 NN 残差 `w_nn·V_NN` + 建 kNN 图 | `:88-93, 209-211` |
| `--use-consideration` | 关 | 加词典筛选三层 log-mask(匹配/成本/时间门) | `:157-177` |
| `--use-soc-mixture` | 关 | class 从 3 档变 3×7=21(tier×soc),per-soc δ/demand | `:97, 216-222` |
| `--typed-mass-occ` | 关 | 引力作用在本职业岗位 `M_j^o`(需 soc-mixture) | `:229-231` |
| `--anchor-transit` | 关 | 加刷卡公交去向锚 | `:242-245` |
| `--anchor-share` | 关 | 加方式份额水平锚 | `:246-248` |
| `--gnn-mode residual` | residual | NN 当残差(伦敦硬规则,convex 会掉 CPC) | `:210-212` |
| `--no-self-loop` | 关(即默认开) | 去掉同格通勤 boost | `:114-115, 206-208` |
| `--origin-chunks G` | 1 | 出发地切 G 块省显存(精确无近似) | `:133-141` |

**anchored 全量命令**(= 论文主规格,`data/scripts/autodl_v4_run.py`):
```
python experiments/train_beijing.py --epochs 300 --device cuda --seed $s \
  --use-nn --use-consideration --use-soc-mixture --typed-mass-occ \
  --anchor-transit --anchor-share --gnn-mode residual \
  --origin-chunks 16 --lr 0.03 --out evaluation_outputs/v4_full_anchored_s$s.pt
```
即:NN + 词典筛选 + soc 混合 + typed-mass 职业 + mode 双锚,全开。3-seed CPC 0.6926±0.0083。

---

## 7. 一行对照表(符号 → 文献 → 代码)

| 符号 | 含义 | 文献锚 | 代码 |
|---|---|---|---|
| λ | mode 嵌套 inclusive value 系数 | nested logit (McFadden) | `lam`, `:143` |
| β_t0, β_t1 | 时间负效用(距离相关 VOT) | Bhat distance-dependent VOT | `beta_t0/t1`, `:139-141` |
| γ_M | 岗位引力 | gravity / Cervero | `gamma_M`, `:147` |
| α_W | 工资吸引 | Wachs wage | `alpha_W`, `:145` |
| ν_D | 竞争排斥 | Shen accessibility | `nu_D`, `:149` |
| δ / typed-mass | 职业匹配 | Stoll-Houston / 重参数化 | `delta` / `:229-231` |
| T_max | 通勤忍受度三档 | Bhat 1995 | `T_max`, `:155` |
| w_nn | NN 残差权重 | Wang residual(升级版) | `w_nn`, `:153` |
| self_loop | 同格通勤 boost | 自雇/就近 | `self_loop`, `:115` |

(完。配套 [01_identifiability_reliability.md] = 这套模型建出来"能信到哪"。)
