"""北京 v4 稀疏嵌套 logit 模型 (Phase 2 + 2b 完整版)

伦敦 forward_cs (稠密 T,N,N) -> 北京 edge/segment 稀疏版, 完整保留 mode 嵌套。
destination-primary 结构 (= v3l):
   V_lower_m = (β_t0_m + β_t1_m·log_d)·t_m + asc_m + θ_inc_m·income_o
   IV(e)     = logΣ_m exp(V_lower_m / λ)                       # mode 嵌套 inclusive value
   V_M_c     = γ_M_c·log_M_d + δ_c·match_signal_c
   V_other_c = α_W_c·log_W_d + ν_D_c·log_D_d
   V_dest_c  = V_M_c + V_other_c + λ·IV + self_loop + w_nn·V_NN + 词典筛选 log-mask
   logP_c    = V_dest_c - segment_logsumexp(V_dest_c, origin)  # 按出发地 softmax
   logP      = logΣ_c log(π_c(o)) + logP_c                     # tier (×soc) 混合

完整功能 (flag 开关):
  - NN dual-branch residual (GraphSAGE-lite + 候选 edge bilinear), gnn-mode residual
  - 词典筛选 (consideration filter): match / cost / time-gate 软 sigmoid log-mask
  - soc 混合: 类 = tier×soc, per-soc demand_share + per-soc δ/match filter
sign 约束 (lit-anchored): γ>0 / ν<0 / α≥0 / β_t<0 / δ≥0 / λ∈(ε,1] / T_max>0.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    v = max(v, 1e-6)
    return math.log(math.expm1(v)) if v < 20 else v


def _inv_sigmoid(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return math.log(s / (1 - s))


def segment_logsumexp(values, seg_id, num_seg):
    m = torch.full((num_seg,), float("-inf"), device=values.device, dtype=values.dtype)
    m = m.scatter_reduce(0, seg_id, values, reduce="amax", include_self=False)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = (values - m_safe[seg_id]).exp()
    se = torch.zeros(num_seg, device=values.device, dtype=values.dtype).scatter_add(0, seg_id, shifted)
    return m_safe + se.clamp_min(1e-38).log()


def scatter_mean(src, index, dim_size, weight=None):
    """src (E,F), index (E,) -> (dim_size,F) 邻居聚合。weight (E,) 给则加权 mean(路网边权)。"""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    if weight is None:
        out.index_add_(0, index, src)
        cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        cnt.index_add_(0, index, torch.ones(index.shape[0], device=src.device, dtype=src.dtype))
        return out / cnt.clamp_min(1.0).unsqueeze(1)
    out.index_add_(0, index, src * weight.unsqueeze(1))
    wsum = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    wsum.index_add_(0, index, weight)
    return out / wsum.clamp_min(1e-6).unsqueeze(1)


class SageLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h, edge_index, N, edge_weight=None):
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_mean(h[src], dst, N, weight=edge_weight)
        return self.w_self(h) + self.w_neigh(agg)


class BeijingPairEncoder(nn.Module):
    """GraphSAGE-lite -> origin/dest 嵌入 -> 候选 edge bilinear V_NN。静态(时段无关)。"""
    def __init__(self, in_dim, hid=32, emb=16, n_layers=2):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hid)
        self.sage = nn.ModuleList([SageLayer(hid, hid) for _ in range(n_layers)])
        self.origin_head = nn.Linear(hid, emb)
        self.dest_head = nn.Linear(hid, emb)

    def node_embed(self, x, edge_index, N, edge_weight=None):
        h = F.relu(self.lin_in(x))
        for layer in self.sage:
            h = F.relu(layer(h, edge_index, N, edge_weight))
        return self.origin_head(h), self.dest_head(h)   # (N,emb), (N,emb)

    def edge_vnn(self, e_o, e_d, o_idx, d_idx):
        return (e_o[o_idx] * e_d[d_idx]).sum(-1)        # (E,)


class _MultiScaleTCN(nn.Module):
    """沿时间轴的多尺度 1D 卷积 (kernels 3/5/7, circular padding 让 24h 环绕)。
    输入 (Th,N,C) -> (N,C,Th) conv -> (Th,N,C)。"""
    def __init__(self, hid, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(hid, hid, k, padding=k // 2,
                                              padding_mode="circular") for k in kernels])
        self.proj = nn.Linear(hid * len(kernels), hid)

    def forward(self, h_seq):                            # (Th,N,C)
        x = h_seq.permute(1, 2, 0)                       # (N,C,Th)
        outs = [F.relu(c(x)) for c in self.convs]
        cat = torch.cat(outs, dim=1).permute(2, 0, 1)    # (Th,N,C*K)
        return self.proj(cat)                            # (Th,N,C)


class BeijingDualBranchEncoder(nn.Module):
    """双路 ST-GNN: 静态空间 GraphSAGE + 动态(per-hour SAGE + GRU + 多尺度TCN)。
    细特征逐小时进 -> 输出端池化到 4 时段 -> 每时段 origin/dest 嵌入 -> 候选 edge bilinear V_NN[period]。
    分工: 静态腿吃空间结构特征, 动态腿吃 pop/拥堵逐小时序列, RUM 头吃决策变量(外部, 不在这)。"""
    def __init__(self, static_dim, dyn_dim, hid=32, emb=16, gru_hidden=32,
                 n_layers=2, tcn_kernels=(3, 5, 7), n_periods=4):
        super().__init__()
        self.n_periods = n_periods
        # 静态腿
        self.s_in = nn.Linear(static_dim, hid)
        self.s_sage = nn.ModuleList([SageLayer(hid, hid) for _ in range(n_layers)])
        # 动态腿
        self.d_in = nn.Linear(dyn_dim, hid)
        self.d_sage = nn.ModuleList([SageLayer(hid, hid) for _ in range(n_layers)])
        self.gru = nn.GRU(hid, gru_hidden, batch_first=False)
        self.tcn = _MultiScaleTCN(hid, tcn_kernels)
        self.dfuse = nn.Linear(gru_hidden + hid, hid)
        # 静态+动态 -> 逐时段 origin/dest 嵌入
        self.origin_head = nn.Linear(hid * 2, emb)
        self.dest_head = nn.Linear(hid * 2, emb)

    def node_embed(self, Xstatic, Xdyn, edge_index, N, hour2period, edge_weight=None):
        """Xstatic (N,Fs); Xdyn (Th,N,Fd); hour2period (Th,) long; edge_weight (E,) 路网边权。
        返回 e_o, e_d 各 (P,N,emb)。"""
        # 静态
        hs = F.relu(self.s_in(Xstatic))
        for l in self.s_sage:
            hs = F.relu(l(hs, edge_index, N, edge_weight))   # (N,hid)
        # 动态 per hour SAGE
        Th = Xdyn.shape[0]
        hl = []
        for t in range(Th):
            h = F.relu(self.d_in(Xdyn[t]))
            for l in self.d_sage:
                h = F.relu(l(h, edge_index, N, edge_weight))
            hl.append(h)
        hstack = torch.stack(hl, 0)                       # (Th,N,hid)
        gru_out, _ = self.gru(hstack)                     # (Th,N,gru_hidden)
        tcn_out = self.tcn(hstack)                        # (Th,N,hid)
        hd = F.relu(self.dfuse(torch.cat([gru_out, tcn_out], -1)))   # (Th,N,hid)
        # 小时 -> 时段 池化 (均值)
        P = self.n_periods
        hd_p = torch.zeros(P, hd.shape[1], hd.shape[2], device=hd.device, dtype=hd.dtype)
        cnt = torch.zeros(P, device=hd.device, dtype=hd.dtype)
        hd_p.index_add_(0, hour2period, hd)
        cnt.index_add_(0, hour2period, torch.ones(Th, device=hd.device, dtype=hd.dtype))
        hd_p = hd_p / cnt.clamp_min(1).view(P, 1, 1)
        # 逐时段 origin/dest 嵌入
        e_o, e_d = [], []
        for p in range(P):
            comb = torch.cat([hs, hd_p[p]], -1)           # (N, hid*2)
            e_o.append(self.origin_head(comb)); e_d.append(self.dest_head(comb))
        return torch.stack(e_o, 0), torch.stack(e_d, 0)   # (P,N,emb), (P,N,emb)

    def edge_vnn(self, e_o, e_d, period, o_idx, d_idx):
        return (e_o[period][o_idx] * e_d[period][d_idx]).sum(-1)     # (E,)


class BeijingNestedHead(nn.Module):
    def __init__(self, n_modes=3, n_tiers=3, n_soc=7, n_districts=16, n_periods=4,
                 use_self_loop=True, use_consideration=False, use_soc_mixture=False,
                 gnn_mode="residual", residual_scale_init=0.1, use_typed_mass=False,
                 use_match_filter=True, use_frozen_occ_mask=False,
                 soft_lex_match=False, k_match_min=20.0):
        super().__init__()
        self.M, self.K, self.S = n_modes, n_tiers, n_soc
        self.n_districts, self.n_periods = n_districts, n_periods
        self.use_self_loop = use_self_loop
        self.use_consideration = use_consideration
        self.use_soc_mixture = use_soc_mixture
        self.use_typed_mass = use_typed_mass
        self.use_match_filter = use_match_filter   # L1 词典筛选职业匹配门 (ablation 可关, 默认 on)
        self.use_frozen_occ_mask = use_frozen_occ_mask  # L1b 外生硬职业 mask (非补偿考虑集筛选, 跟 typed-mass 引力分工)
        self.soft_lex_match = soft_lex_match   # soft-lexicographic: τ_s≥floor + k_s≥k_min (防塌陷防糊, 伦敦 winning 设计)
        self.k_match_min = float(k_match_min)
        self.tau_match_floor = None            # (S,) floor = mult×mean(demand_share[:,s]); trainer 经 set_tau_match_floor 填
        self.gnn_mode = gnn_mode
        self.lam_eps = 0.05
        self.n_classes = (n_tiers * n_soc) if use_soc_mixture else n_tiers

        # mode 下层
        self.raw_beta_t0 = nn.Parameter(torch.full((n_modes,), _inv_softplus(0.07)))
        self.raw_beta_t1 = nn.Parameter(torch.full((n_modes,), _inv_softplus(0.005)))
        self.asc = nn.Parameter(torch.zeros(n_modes))
        self.theta_inc = nn.Parameter(torch.zeros(n_modes))
        self.raw_lambda = nn.Parameter(torch.tensor(_inv_sigmoid(0.8)))

        # destination 上层 (per tier)
        self.raw_alpha_W = nn.Parameter(torch.full((n_tiers,), _inv_softplus(0.10)))
        self.raw_gamma_M = nn.Parameter(torch.full((n_tiers,), _inv_softplus(0.50)))
        self.raw_nu_D = nn.Parameter(torch.full((n_tiers,), _inv_softplus(0.10)))
        # δ: soc 混合时 per-soc, 否则 per-tier
        n_delta = n_soc if use_soc_mixture else n_tiers
        self.raw_delta = nn.Parameter(torch.full((n_delta,), _inv_softplus(0.05)))

        if use_self_loop:
            self.self_loop = nn.Parameter(torch.zeros(n_districts, n_periods))

        # NN residual scale
        if gnn_mode == "residual":
            self.raw_w_nn = nn.Parameter(torch.tensor(_inv_softplus(residual_scale_init)))

        # 词典筛选
        if use_consideration:
            # L1 match filter (per soc 若 soc 混合, 否则全局)
            n_mf = n_soc if use_soc_mixture else 1
            self.raw_k_match = nn.Parameter(torch.full((n_mf,), _inv_softplus(10.0)))
            # soft-lex 时 τ=floor+softplus(raw), raw 初值取 softplus≈0.01 让 τ≈floor 起步; 否则旧版 τ=raw(init 0)
            self.match_thresh = nn.Parameter(torch.full((n_mf,),
                _inv_softplus(0.01) if soft_lex_match else 0.0))
            # L2 cost filter (per tier)
            self.raw_theta_t_cost = nn.Parameter(torch.tensor(_inv_softplus(0.05)))
            self.raw_theta_d_cost = nn.Parameter(torch.tensor(_inv_softplus(0.10)))
            self.raw_cost_budget = nn.Parameter(torch.full((n_tiers,), _inv_softplus(1.0)))
            self.raw_cost_thresh = nn.Parameter(torch.full((n_tiers,), _inv_softplus(1.0)))
            self.raw_k_cost = nn.Parameter(torch.tensor(_inv_softplus(3.0)))
            # L3 time gate (per tier) — Bhat 通勤忍受度 T_max
            self.raw_T_max = nn.Parameter(torch.tensor([_inv_softplus(v) for v in (40., 60., 90.)]))
            self.raw_k_time = nn.Parameter(torch.tensor(_inv_softplus(0.2)))

    def set_tau_match_floor(self, floor):   # soft-lex: 由 trainer 用 mult×mean(demand_share) 填
        self.tau_match_floor = floor

    # sign-constrained 读出
    @property
    def beta_t0(self):  return -F.softplus(self.raw_beta_t0)
    @property
    def beta_t1(self):  return -F.softplus(self.raw_beta_t1)
    @property
    def lam(self):      return self.lam_eps + (1 - self.lam_eps) * torch.sigmoid(self.raw_lambda)
    @property
    def alpha_W(self):  return F.softplus(self.raw_alpha_W)
    @property
    def gamma_M(self):  return F.softplus(self.raw_gamma_M)
    @property
    def nu_D(self):     return -F.softplus(self.raw_nu_D)
    @property
    def delta(self):    return F.softplus(self.raw_delta)
    @property
    def w_nn(self):     return F.softplus(self.raw_w_nn) if self.gnn_mode == "residual" else None
    @property
    def T_max(self):    return F.softplus(self.raw_T_max)

    def _consideration_mask(self, k_tier, soc_idx, batch):
        """返回该 class 的 log-mask (E,) (词典筛选三层之和)。"""
        log_mask = 0.0
        # L1 match (职业匹配筛选门; ablation --no-match-filter 可关, 保留 L2/L3)
        if self.use_match_filter:
            mi = soc_idx if self.use_soc_mixture else 0
            sig = batch["demand_share_d"][:, soc_idx] if self.use_soc_mixture else batch["match"]
            km = F.softplus(self.raw_k_match[mi])
            if self.soft_lex_match:                                  # 防糊: k 强制锐利
                km = km.clamp(min=self.k_match_min)
                tau = F.softplus(self.match_thresh[mi])              # 防塌陷: τ = floor + softplus(raw) ≥ floor
                if self.tau_match_floor is not None:
                    tau = tau + self.tau_match_floor[mi]
            else:
                tau = self.match_thresh[mi]                          # 旧版自由阈值(会塌陷到不筛)
            log_mask = log_mask + F.logsigmoid(km * (sig - tau))
        # L1b 外生硬职业 mask: 占本职业岗位太少的目的地直接踢出考虑集(非补偿, typed-mass 连续 log 做不到)
        if self.use_frozen_occ_mask and self.use_soc_mixture and "occ_mask_d" in batch:
            elig = batch["occ_mask_d"][:, soc_idx]            # (E,) 1 可去 / 0 不可去
            log_mask = log_mask + (elig - 1.0) * 30.0          # 可去 +0, 不可去 -30 (硬; 全筛段 softmax 自愈)
        # L2 cost
        t_min = batch["t_min"]; log_d = batch["log_d"]
        cost = F.softplus(self.raw_theta_t_cost) * t_min + F.softplus(self.raw_theta_d_cost) * log_d
        budget = F.softplus(self.raw_cost_budget[k_tier]) * 50.0   # 尺度 ~分钟
        ratio = cost / budget.clamp_min(1e-3)
        thr = F.softplus(self.raw_cost_thresh[k_tier])
        log_mask = log_mask + F.logsigmoid(F.softplus(self.raw_k_cost) * (thr - ratio))
        # L3 time gate
        log_mask = log_mask + F.logsigmoid(
            F.softplus(self.raw_k_time) * (self.T_max[k_tier] - t_min))
        return log_mask

    def mode_logits(self, batch):
        """每 edge 的 per-mode scaled 效用 (3,E), softmax 即 P(m|i,j)。"""
        log_d = batch["log_d"]; income_o = batch["income_o"]; lam = self.lam
        tms = [batch["t_car"], batch["t_transit"], batch["t_walk"]]
        b0, b1 = self.beta_t0, self.beta_t1
        V = [(b0[m] + b1[m] * log_d) * tms[m] + self.asc[m] + self.theta_inc[m] * income_o
             for m in range(self.M)]
        return torch.stack(V, 0) / lam

    def forward(self, batch, v_nn=None):
        seg_id, num_seg = batch["seg_id"], batch["num_seg"]
        log_d = batch["log_d"]; income_o = batch["income_o"]; lam = self.lam

        # mode IV (per edge)
        t_modes = [batch["t_car"], batch["t_transit"], batch["t_walk"]]
        b0, b1 = self.beta_t0, self.beta_t1
        scaled = []
        for m in range(self.M):
            V_lower_m = (b0[m] + b1[m] * log_d) * t_modes[m] + self.asc[m] + self.theta_inc[m] * income_o
            scaled.append(V_lower_m / lam)
        log_iv = torch.logsumexp(torch.stack(scaled, 0), 0)        # (E,)

        log_M_d, log_W_d, log_D_d = batch["log_M_d"], batch["log_W_d"], batch["log_D_d"]
        tier_props_o = batch["tier_props_o"]
        aW, gM, nD, dl = self.alpha_W, self.gamma_M, self.nu_D, self.delta

        boost = None
        if self.use_self_loop:
            sl = self.self_loop[batch["district_o"], batch["period"]]
            boost = torch.where(batch["is_self"], sl, torch.zeros_like(sl))

        nn_term = None
        if v_nn is not None and self.gnn_mode == "residual":
            nn_term = self.w_nn * v_nn

        # 逐类增量 logaddexp (不堆 (E,C) stack, 省显存)
        acc = None
        for c in range(self.n_classes):
            if self.use_soc_mixture:
                k_tier, soc = c // self.S, c % self.S
                match_sig = batch["demand_share_d"][:, soc]
                delta_c = dl[soc]
                log_pi_c = (torch.log(tier_props_o[:, k_tier].clamp_min(1e-9))
                            + torch.log(batch["soc_props_o"][:, soc].clamp_min(1e-9)))
            else:
                k_tier, soc = c, 0
                match_sig = batch["match"]
                delta_c = dl[c]
                log_pi_c = torch.log(tier_props_o[:, k_tier].clamp_min(1e-9))

            if self.use_typed_mass and self.use_soc_mixture:
                # 引力作用在"本职业岗位"M_j^o = 总岗位×职业需求 (不混总数, 微观验证 δ≈γ)
                V_M = gM[k_tier] * (log_M_d + torch.log(match_sig.clamp_min(1e-6)))
            else:
                V_M = gM[k_tier] * log_M_d + delta_c * match_sig
            V_other = aW[k_tier] * log_W_d + nD[k_tier] * log_D_d
            V_dest = V_M + V_other + lam * log_iv
            if boost is not None: V_dest = V_dest + boost
            if nn_term is not None: V_dest = V_dest + nn_term
            if self.use_consideration:
                V_dest = V_dest + self._consideration_mask(k_tier, soc, batch)
            lse = segment_logsumexp(V_dest, seg_id, num_seg)
            term = log_pi_c + (V_dest - lse[seg_id])     # (E,)
            acc = term if acc is None else torch.logaddexp(acc, term)
        return acc                                        # (E,)

    def param_report(self):
        with torch.no_grad():
            r = {
                "lambda": float(self.lam),
                "beta_t0": [round(x, 3) for x in self.beta_t0.tolist()],
                "alpha_W": [round(x, 3) for x in self.alpha_W.tolist()],
                "gamma_M": [round(x, 3) for x in self.gamma_M.tolist()],
                "nu_D": [round(x, 3) for x in self.nu_D.tolist()],
                "delta": [round(x, 4) for x in self.delta.tolist()],
            }
            if self.gnn_mode == "residual":
                r["w_nn"] = round(float(self.w_nn), 3)
            if self.use_consideration:
                r["T_max"] = [round(x, 1) for x in self.T_max.tolist()]
            return r
