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


def scatter_mean(src, index, dim_size):
    """src (E,F), index (E,) -> (dim_size,F) 邻居均值聚合。"""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones(index.shape[0], device=src.device, dtype=src.dtype))
    return out / cnt.clamp_min(1.0).unsqueeze(1)


class SageLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h, edge_index, N):
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_mean(h[src], dst, N)
        return self.w_self(h) + self.w_neigh(agg)


class BeijingPairEncoder(nn.Module):
    """GraphSAGE-lite -> origin/dest 嵌入 -> 候选 edge bilinear V_NN。静态(时段无关)。"""
    def __init__(self, in_dim, hid=32, emb=16, n_layers=2):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hid)
        self.sage = nn.ModuleList([SageLayer(hid, hid) for _ in range(n_layers)])
        self.origin_head = nn.Linear(hid, emb)
        self.dest_head = nn.Linear(hid, emb)

    def node_embed(self, x, edge_index, N):
        h = F.relu(self.lin_in(x))
        for layer in self.sage:
            h = F.relu(layer(h, edge_index, N))
        return self.origin_head(h), self.dest_head(h)   # (N,emb), (N,emb)

    def edge_vnn(self, e_o, e_d, o_idx, d_idx):
        return (e_o[o_idx] * e_d[d_idx]).sum(-1)        # (E,)


class BeijingNestedHead(nn.Module):
    def __init__(self, n_modes=3, n_tiers=3, n_soc=7, n_districts=16, n_periods=4,
                 use_self_loop=True, use_consideration=False, use_soc_mixture=False,
                 gnn_mode="residual", residual_scale_init=0.1):
        super().__init__()
        self.M, self.K, self.S = n_modes, n_tiers, n_soc
        self.n_districts, self.n_periods = n_districts, n_periods
        self.use_self_loop = use_self_loop
        self.use_consideration = use_consideration
        self.use_soc_mixture = use_soc_mixture
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
            self.match_thresh = nn.Parameter(torch.full((n_mf,), 0.0))
            # L2 cost filter (per tier)
            self.raw_theta_t_cost = nn.Parameter(torch.tensor(_inv_softplus(0.05)))
            self.raw_theta_d_cost = nn.Parameter(torch.tensor(_inv_softplus(0.10)))
            self.raw_cost_budget = nn.Parameter(torch.full((n_tiers,), _inv_softplus(1.0)))
            self.raw_cost_thresh = nn.Parameter(torch.full((n_tiers,), _inv_softplus(1.0)))
            self.raw_k_cost = nn.Parameter(torch.tensor(_inv_softplus(3.0)))
            # L3 time gate (per tier) — Bhat 通勤忍受度 T_max
            self.raw_T_max = nn.Parameter(torch.tensor([_inv_softplus(v) for v in (40., 60., 90.)]))
            self.raw_k_time = nn.Parameter(torch.tensor(_inv_softplus(0.2)))

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
        # L1 match
        if self.use_soc_mixture:
            sig = batch["demand_share_d"][:, soc_idx]            # (E,)
            km = F.softplus(self.raw_k_match[soc_idx]); tau = self.match_thresh[soc_idx]
        else:
            sig = batch["match"]; km = F.softplus(self.raw_k_match[0]); tau = self.match_thresh[0]
        log_mask = log_mask + F.logsigmoid(km * (sig - tau))
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
