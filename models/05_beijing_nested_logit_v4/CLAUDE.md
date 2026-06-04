# CLAUDE.md (v4 — 北京)

Guidance for Claude Code working in **05_beijing_nested_logit_v4/**.

## 这是什么

把伦敦 v3（`../04_london_nested_logit_v3/`，Paper A 模型线，3-seed CPC ≈ 0.56）**整条模型线迁移到北京**。同一套架构：嵌套 logit（上 mode、下 destination）+ dual-branch NN + Cervero/Hansen 引力 + 词典筛选。北京是 CLAUDE.md 一直说的"真正目标"，伦敦是 sandbox。

迁移不是换数据路径——两城有两处硬结构错配，但 **mode 嵌套必须保留**（用户明确要求）：

1. **规模**：1,725 格 → **16,907 格**。稠密 `(T,N,N)` 放不下 → 整个模型改成**稀疏 choice-set (edge/segment)**。`log_softmax(dim=-1)` → 按 origin 分段的 **segment-logsumexp**。
2. **方式数据形态**：伦敦 `F_ij_t` 本身也是方式合并流量，mode 嵌套靠 **per-mode 时间 + 外生 mode_share 先验** 识别，不是分方式 OD label。北京同理：车=高德 `t_matrix`、步行=距离、**公交=用北京公交数据自造**（长杆任务，见下）。
3. **职业**：北京只有区级（16 区 ×7），聚合 OD 下 `delta_match` **点估不出**（已撞墙三次，见伦敦 memory）。全移植保留这套机器，**预期它平 = paper finding，不调参压它**。

## ⚠⚠⚠ GOVERNANCE 红线

- **北京手机信令及其一切派生数据，2026-07 前不出库、不 commit。**
- 代码 (`*.py` / `*.md` / methodology 文档) 可 commit；数据走 `.gitignore`（已配：`data/`、`*.npz/*.parquet/*.pt` 等）。
- 北京原始数据**绝对路径只读引用**，不复制进库：
  - OD / 时间 / 职业：`D:\UoM\PaperC_local\{cell_index,od_cell_flow,t_matrix,occ_share_by_district,P_occ_given_ind,xj_industry_2023}.parquet`
  - 公交 / 房价：`D:\UoM\congestion_project\{01北京市公交,bus_crawl,202501北京小区}`

## 数据契约（北京 ↔ 伦敦 v3）

| 维度 | 伦敦 v3 | 北京 |
|---|---|---|
| 格子 N | 1,725 | 16,907 (`cell_index.parquet`) |
| 出发地 | 1,725 | 11,371 (有候选集) |
| OD 目标 | `F_ij_t` (24,N,N) 稠密 | `od_cell_flow.parquet` 7.06M 行, **f_ampeak/f_pmpeak/f_midday/f_night** → T=4 时段 |
| 候选集/距离 | — | `t_matrix.parquet` 30.4M 行 (o_idx,d_idx,hav_km,t_obs_min,t_ff_min), 均值 2674/出发地 |
| 车时间 | BPR | `t_matrix` t_obs(拥堵)/t_ff(自由流), ampeak/pmpeak→t_obs, night→t_ff, midday→插值 |
| 步行时间 | BPR | hav_km / 步速, 封顶 |
| 公交时间 | BPR | **自造**: `01北京市公交` 线网 shp + `bus_crawl/timetable`(逐站到站时刻+发车间隔) + `bus_crawl/bus_status`(227 快照实测速度/拥挤) |
| mode_share 先验 | 距离感知 | 北京公开方式分担率(年鉴/普查 按距离档) |
| 引力 M_j | log_M_z | `cell_index.jobs` → log z |
| 竞争 D_j (Shen) | log_D_z | residents/jobs 推 |
| 收入 | ASHE | `202501北京小区` 房价/㎡ 空间 join |
| 职业 | soc_props(N,9)+ε(9,8)+grid_industry(N,8) | `occ_share_by_district`(16×7)+`P_occ_given_ind`(19×7)+`xj_industry_2023`(19×16) 区级广播 |
| 行政区 | 33 borough | 16 区 |

## 公交时间 builder（Phase 1 长杆，单列）

- 线网几何/站点: `01北京市公交/北京市公交线路.shp` + `北京市公交_点.shp`
- 站间时间/发车间隔: `bus_crawl/timetable/timetable.xlsx`(每车逐站到站时刻 字段2..30) + `busline.xlsx`(bus_id→时刻表)
- 实测校准: `bus_crawl/bus_status/*.jsonl`(227 快照/天, 车辆经纬度+到站状态+拥挤度)
- **第一版** t_transit ≈ 步行接驳最近站 + 车内时间(时刻表站间) + 等待(间隔/2) + 下车步行。太重则退化为"距离档公交速度模型 + 等待"。
- **Caveats(诚实写)**: ① 只有公交无地铁 → 纯公交时间系统性偏高, 地铁后续补; ② 单日快照; ③ 全多段换乘路由是后续精修。

## 架构决定（沿用伦敦, 勿重议）

- 稀疏 edge 表示, segment-logsumexp 替代 dense softmax。
- mode 嵌套完整: per-mode V_lower → IV=logΣ_m exp(V_lower_m/λ_m) → V_dest=V_M+V_other+λ·IV。λ_m∈(0,1] (近 0 梯度爆炸, 见伦敦 CLAUDE.md, 需数值稳定测试)。
- `--gnn-mode residual`(伦敦硬规则, convex 默认会掉 0.075 CPC)。
- T=4 时段各自 softmax + CPC; T_max/time-gate/λ_m 阈值按"段"重标, **历史伦敦 CPC 数字不可直接比**。

## baseline 卫生（沿用伦敦教训）

- 改了 head/trainer 后, **重训 current-code baseline**, 不用历史 JSON。
- ep0 数值健康检查 (tnll/vnll/系数 init)。
- 不用手工阈值、不为 cosmetic 降 CPC（behavioral fidelity 可 ≤0.010 trade-off）。

## 落地顺序

Phase 0 脚手架(done) → Phase 1 builders(grid/choiceset/modes/income/occupation/aux) → Phase 2 稀疏 forward(保留 mode) → Phase 3 稀疏 trainer+CPC → Phase 4 smoke(top-2000 稠密验语义) → 全量(AutoDL 32GB) → 评估。

本地 Windows CPU 稀疏 GNN 易 segfault → 全量上 AutoDL, 本地只做 builders + smoke。

## 用户偏好（继承全局）

- 中文交流, 文档可中英混合, 不堆学术词/缩写
- commit 不加 Co-Authored-By
- 不主动写总结 markdown（除非要求; methodology/ 下设计文档是明确要求的不算）
- 解释先锚到"一句话核心想法"再讲技术细节
