# -*- coding: utf-8 -*-
"""Empirical Fisher information of the flows-loss w.r.t. the interpretable head
parameters: FIM = sum_i g_i g_i^T, g_i = grad of origin i's commute-hour
destination log-likelihood. PSD by construction (avoids the indefinite raw
Hessian). At the trained model, cached encoder. Saves empirical_fisher_s0.npz."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import importlib.util as ilu, torch, numpy as np
from pathlib import Path

V3 = Path("/root/autodl-tmp/v3_nested_smoke/04_london_nested_logit_v3")
spec = ilu.spec_from_file_location("scenA", V3 / "experiments" / "paper_a" / "scenario_v3l_A.py")
m = ilu.module_from_spec(spec); spec.loader.exec_module(m); fcs = m.forward_cs
dev = torch.device("cuda")
ck = torch.load(f"{V3}/evaluation_outputs/paper_a/v3l_R7a_clean_s0.pt", map_location=dev, weights_only=False)
enc, rum, ta = m.reconstruct_model(ck, dev); ta.aux_path = "data/processed/paperA_v3_aux_cervero.npz"
import os; os.chdir(V3)
data = m.load_scenario_inputs(ta, dev)
with torch.no_grad():
    Vraw = enc(data["X_static"], data["X_dynamic"], data["edge_index"]).contiguous()
class Cached:
    def __init__(s, v): s.v = v
    def __call__(s, *a, **k): return s.v
    def eval(s): return s
cenc = Cached(Vraw)

GROUPS = [
    ("Commute-time tolerance","raw_T_max_per_tier"),("Distance decay","raw_gamma_decay"),
    ("Time sensitivity","raw_beta_t_per_mode"),("Time x distance","raw_beta_t_slope_per_mode"),
    ("Job-mass pull (gravity)","raw_gamma_M"),("Crowding avoidance","raw_nu_D"),
    ("Cost budget","raw_cost_budget_per_tier"),("Cost threshold","raw_cost_thresh_per_tier"),
    ("Mode constants","asc_per_mode"),("Mode x income","theta_inc_per_mode"),
    ("Mode x kids","theta_kids_per_mode"),("Mode x cars","theta_cars_per_mode"),
    ("Wage pull (class)","raw_alpha_wage"),("Occupation sorting","raw_delta_match"),
]
names, groups, params, slices = [], [], [], []
i0 = 0
for gname, attr in GROUPS:
    t = getattr(rum, attr); t.requires_grad_(True); params.append(t)
    n = t.numel(); slices.append((i0, i0+n)); names += [f"{gname}[{k}]" for k in range(n)]
    groups += [gname]*n; i0 += n
N = i0
theta0 = torch.cat([p.detach().reshape(-1) for p in params]).cpu().numpy()
scale = np.clip(np.abs(theta0), 1e-3, None)
print(f"N={N} params", flush=True)

CH = [7,8,9,17,18]
obs = data["observed_OD"]                                   # (T,N,N)
Norig = obs.shape[1]
# forward once, keep graph for repeated per-origin backward
logP = fcs(cenc, rum, data["X_static"], data["X_dynamic"], data["edge_index"], data["t_per_mode"],
           data["mode_names"], data["log_d_ij"], data["match_prob"], data["log_M_z"], data["log_W_z"],
           data["log_D_z"], data["income_score"], data["pct_kids"], data["mean_cars"],
           data["income_tier_props"], data["pi_m_pair"], data["grid_borough_idx"],
           data["observed_OD"], data["val_mask"], soc_props_per_origin=data.get("soc_props_per_origin"),
           per_soc_demand_share_j=data.get("per_soc_demand_share_j"))["log_P_D"]               # (T,N,N)
obs_ch = obs[CH]; logP_ch = logP[CH]                        # (5,N,N)
rowtot = obs_ch.sum(dim=(0,2))                              # (N,) commuters per origin
active = torch.where(rowtot > 0)[0].tolist()
print(f"active origins: {len(active)}", flush=True)

FIM = np.zeros((N, N))
t0 = time.time()
for c, i in enumerate(active):
    li = -(obs_ch[:, i, :] * logP_ch[:, i, :]).sum()       # origin i commute log-lik (neg)
    gi = torch.autograd.grad(li, params, retain_graph=True)
    g = torch.cat([x.reshape(-1) for x in gi]).detach().cpu().numpy()
    FIM += np.outer(g, g)
    if (c+1) % 200 == 0:
        print(f"  origin {c+1}/{len(active)} ({time.time()-t0:.0f}s)", flush=True)
FIM /= len(active)
# non-dimensionalise by parameter magnitude
S = np.diag(scale); FIM_s = S @ FIM @ S
np.savez(f"{V3}/evaluation_outputs/paper_a/empirical_fisher_s0.npz",
         FIM=FIM, FIM_s=FIM_s, names=np.array(names), groups=np.array(groups),
         scale=scale, theta0=theta0, slices=np.array(slices))
ev = np.linalg.eigvalsh(FIM_s)
print(f"\nFIM eig: all >=0 ? min={ev.min():.2e} max={ev.max():.2e}  (PSD check)")
print("saved empirical_fisher_s0.npz")
