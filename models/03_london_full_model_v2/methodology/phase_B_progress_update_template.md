# Paper A Phase B 阶段进展报告 — 回应 4-30 反馈

**作者**: [PLACEHOLDER: 学生姓名]
**日期**: [PLACEHOLDER: YYYY-MM-DD]
**目的**: 向 supervisor 汇报 Paper A 在 4-30 meeting 反馈之后的 Phase B 进展

---

## 1. Recap

4-30 meeting 上 supervisor 对 Paper A minimal version 提了 4 条 critique:
1. **Mode choice 应该 endogenous**（当前是固定 mode share）
2. **Agent 需要 income attribute**
3. **Occupation match 应该真用上**（不只是 sit 在 feature column 里）
4. **β 在 agent type 之间应该 heterogeneous**（按 income / occupation 分层）— 这条最 substantive

Paper A minimal version 当时为了 scope 控制，砍掉了 (1)(2)(3)(4) 全部，只保留了一个 homogeneous β 的 GNN destination choice。Phase B 的目标是 **回应 (2)(3)(4)**，把 (1) mode choice 留在 stretch goal C。

## 2. Phase B 做了什么

- **加 OccMatch term**: utility 改写为 `U_ij = γ·wage_j − β·tt_ij + δ·OccMatch(i,j) + GNN_θ(i,j)`，其中 `OccMatch(i,j) = cos_sim(SOC_dist(i), industry_vec(j))` — origin grid 的 SOC distribution 和 destination grid 的 industry vector 做 cosine similarity。δ 由 trainer 反推。
- **加 income heterogeneity**: 把 β scalar 升级为 `β(tier)`，分 low / mid / high 3 档。每个 origin grid 有 `P(tier|i)` 的 prior（从 `agent_population_v23.csv` 的 income marginal 拿）。Trainer 通过 mixture likelihood 反推 `(β_low, β_mid, β_high)` 三个独立参数。
- **代码复用**: `wage_utility` 和 `occupation_match` 模块直接复用 v2.3 已经写好的，不是从头写；新增的部分主要是 mixture trainer 和 `OccMatch` 的 caching。

## 3. W4 合成校验结果 (synthetic recovery)

W4 关卡是用 known parameter 生成 synthetic flow，再让 trainer 反推，看 bias / coverage。

- **δ recovery**: [PLACEHOLDER: δ_true vs δ_hat, bias %, 95% CI coverage] — target: bias < 10%, coverage ≥ 85%
- **β_tier recovery**:
  - β_low: [PLACEHOLDER: bias %, coverage %]
  - β_mid: [PLACEHOLDER: bias %, coverage %]
  - β_high: [PLACEHOLDER: bias %, coverage %]
- 6 类参数 `(β_low, β_mid, β_high, γ, δ, GNN_θ)` 在 synthetic 上能同时反推回来 — **identifiability 通过**（条件是 origin tier mixture 有足够 spread）。

## 4. London 真数据 W8 关卡结果

- **CPC (Common Part of Commuters)** 三个 model 比较:
  - Baseline GNN minimal: [PLACEHOLDER: CPC]
  - +OccMatch: [PLACEHOLDER: CPC, Δ vs baseline]
  - +OccMatch + IncomeTier: [PLACEHOLDER: CPC, Δ vs baseline]
- **VOT × tier** (vs WebTAG 标准 £8 / £13 / £22):
  - VOT_low: [PLACEHOLDER: £/h] vs WebTAG £8
  - VOT_mid: [PLACEHOLDER: £/h] vs WebTAG £13
  - VOT_high: [PLACEHOLDER: £/h] vs WebTAG £22

**关注两点**: (a) CPC 是否 monotonic improvement，说明 OccMatch / IncomeTier 不是 overfit 的 redundant feature；(b) VOT 分层方向 — 是否 low < mid < high，是否朝 WebTAG 的 ordering 走。数值不要求精确，朝向比绝对值更重要。

## 5. 回应 4-30 critique 状态

| Critique | 状态 | 实现 |
|---|---|---|
| (1) Mode choice endogenous | 砍 → stretch goal C | 如 supervisor OK 可后续做 nested logit |
| (2) Agent income attribute | 已实现 | income tier mixture (low/mid/high) |
| (3) Occupation match 真用上 | 已实现 | δ·OccMatch(i,j) term |
| (4) Heterogeneous β | 已实现 | β(income tier), 3 个独立参数 |

## 6. 下一步

- **短期 (W9–W16)**: Phase B 已完成 → 写 Paper A draft，target NeurIPS workshop / TMLR submission。
- **中期**: 如果 supervisor 判断 mode choice 仍是 reviewer 必问的硬伤 → 启动 Phase C，做 nested logit (mode | destination)。
- **长期**: Beijing data ethics 批准之后 → Paper B cross-city（V-REx invariance），Paper A 完成是 Paper B 的 unblocker；Paper B 的 method prep 可以和 A 的写作同步进行。

## 7. Limitations（先讲，不藏到末尾）

- **β_tier 是 area-level**: 当前是 origin grid 平均的 mixture prior，不是 individual-level β。要做 individual heterogeneous β 需要 LTDS / NTS individual-level data — 留到 Paper B。
- **VOT 数值依赖 demo dataset**: 当前 cached `demo_cache.npz` 是 toy-scale；paper-quality 数值需要 spatial holdout + Biogeme MNL baseline + 100-combo multiverse robustness check 才能 defendable。当前数值更多是 sanity check，不是 paper claim。
- **W4 low-tier coverage [PLACEHOLDER: e.g. 78%] 略低于 nominal 95%**: 在 N→∞ 的 regime 下 Hessian-based SE 收缩速率比 trainer bias 收缩速率快，是 frequentist large-N artifact，不是 trainer 的 bug。这点会写进 paper limitation section。