"""架构 B(独立版本, 不动 production beijing_model.py):
   GNN 学 "workplace attractiveness" A_j + RUM 异质偏好 α_k·A_j。

V_M = α_k · A_j   (A_j = GNN 目的地吸引力标量, α_k≥0 分收入档 valuation)  —— 取代引力 γ_M·log岗位
其余 RUM(竞争 ν_D、嵌套 IV 即成本 β_t·t、词典筛选、self-loop)全保留; 无 bilinear 残差(GNN 的活=产 A_j)。
子类化 production 的 encoder/head, 只覆写吸引力相关部分。
"""
import torch, torch.nn as nn, torch.nn.functional as F
from beijing_model import (BeijingDualBranchEncoder, BeijingNestedHead,
                           segment_logsumexp, _inv_softplus)


class BeijingAttractEncoder(BeijingDualBranchEncoder):
    """双路 ST-GNN + 目的地吸引力读出 A_j。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        emb = self.dest_head.out_features
        self.attract_head = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, 1))

    def dest_attract(self, e_d, period, d_idx):
        """每 edge 的目的地吸引力 A_j (E,)。"""
        return self.attract_head(e_d[period][d_idx]).squeeze(-1)


class BeijingAttractHead(BeijingNestedHead):
    """V_M = α_k·A_j (异质 valuation), 其余继承。forward 的 v_nn 形参 = a_dest(吸引力)。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_beta_dist = nn.Parameter(torch.full((self.K,), _inv_softplus(0.1)))   # per-收入档 距离厌恶
        # α 固定=1 (治 α·A 尺度不定); 吸引力纳入职业匹配: V_M = A_j + log(职业岗位份额)

    @property
    def beta_dist(self):
        return F.softplus(self.raw_beta_dist)        # ≥0: 越远效用越低(分收入档)

    def forward(self, batch, v_nn=None):
        a_dest = v_nn                                # B: 传进来的是吸引力 A_j, 不是 bilinear
        seg_id, num_seg = batch["seg_id"], batch["num_seg"]
        log_d = batch["log_d"]; income_o = batch["income_o"]; lam = self.lam

        # mode IV (成本 β_t·t, 同 production)
        t_modes = [batch["t_car"], batch["t_transit"], batch["t_walk"]]
        b0, b1 = self.beta_t0, self.beta_t1
        scaled = []
        for m in range(self.M):
            V_lower_m = (b0[m] + b1[m] * log_d) * t_modes[m] + self.asc[m] + self.theta_inc[m] * income_o
            scaled.append(V_lower_m / lam)
        log_iv = torch.logsumexp(torch.stack(scaled, 0), 0)

        log_W_d, log_D_d = batch["log_W_d"], batch["log_D_d"]
        tier_props_o = batch["tier_props_o"]
        aW, nD = self.alpha_W, self.nu_D; bdist = self.beta_dist

        boost = None
        if self.use_self_loop:
            sl = self.self_loop[batch["district_o"], batch["period"]]
            boost = torch.where(batch["is_self"], sl, torch.zeros_like(sl))

        acc = None
        for c in range(self.n_classes):
            if self.use_soc_mixture:
                k_tier, soc = c // self.S, c % self.S
                match_sig = batch["demand_share_d"][:, soc]      # 该职业在目的地的岗位份额
                log_pi_c = (torch.log(tier_props_o[:, k_tier].clamp_min(1e-9))
                            + torch.log(batch["soc_props_o"][:, soc].clamp_min(1e-9)))
            else:
                k_tier, soc = c, 0
                match_sig = batch["match"]
                log_pi_c = torch.log(tier_props_o[:, k_tier].clamp_min(1e-9))

            # ⭐ B: 职业特定吸引力 = GNN通用吸引力 + 职业岗位匹配 (α固定=1, 治尺度)
            V_M = a_dest + torch.log(match_sig.clamp_min(1e-6))
            V_other = aW[k_tier] * log_W_d + nD[k_tier] * log_D_d
            V_dest = V_M - bdist[k_tier] * log_d + V_other + lam * log_iv   # − 分收入档距离厌恶
            if boost is not None: V_dest = V_dest + boost
            if self.use_consideration:
                V_dest = V_dest + self._consideration_mask(k_tier, soc, batch)
            lse = segment_logsumexp(V_dest, seg_id, num_seg)
            term = log_pi_c + (V_dest - lse[seg_id])
            acc = term if acc is None else torch.logaddexp(acc, term)
        return acc

    def param_report(self):
        with torch.no_grad():
            return {"lambda": round(float(self.lam), 3),
                    "beta_dist": [round(x, 3) for x in self.beta_dist.tolist()],
                    "beta_t0": [round(x, 3) for x in self.beta_t0.tolist()],
                    "nu_D": [round(x, 3) for x in self.nu_D.tolist()],
                    "alpha_W": [round(x, 3) for x in self.alpha_W.tolist()],
                    "T_max": [round(x, 1) for x in self.T_max.tolist()] if self.use_consideration else None}
