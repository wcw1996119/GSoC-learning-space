# -*- coding: utf-8 -*-
"""Mode-share moment information: expected/GN Fisher of the mode-choice softmax
P(m|i,j) w.r.t. all head params, weighted by commuting flow. This is the
identifying information the observed mode shares (WU03, an injected external
moment) add about the mode parameters. PSD by construction. Add to the
destination Fisher to show mode params contracting under external data."""
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

GROUPS = [("Commute-time tolerance","raw_T_max_per_tier"),("Distance decay","raw_gamma_decay"),
    ("Time sensitivity","raw_beta_t_per_mode"),("Time x distance","raw_beta_t_slope_per_mode"),
    ("Job-mass pull (gravity)","raw_gamma_M"),("Crowding avoidance","raw_nu_D"),
    ("Cost budget","raw_cost_budget_per_tier"),("Cost threshold","raw_cost_thresh_per_tier"),
    ("Mode constants","asc_per_mode"),("Mode x income","theta_inc_per_mode"),
    ("Mode x kids","theta_kids_per_mode"),("Mode x cars","theta_cars_per_mode"),
    ("Wage pull (class)","raw_alpha_wage"),("Occupation sorting","raw_delta_match")]
names, groups, tensors, slices = [], [], [], []
i0 = 0
for gname, attr in GROUPS:
    t = getattr(rum, attr); tensors.append(t); slices.append((i0, i0+t.numel()))
    names += [f"{gname}[{k}]" for k in range(t.numel())]; groups += [gname]*t.numel(); i0 += t.numel()
N = i0
theta0 = torch.cat([t.detach().reshape(-1) for t in tensors]).clone()
scale = theta0.abs().clamp(min=1e-3)
print(f"N={N}", flush=True)

def set_flat(theta):
    with torch.no_grad():
        for t,(a,b) in zip(tensors,slices): t.copy_(theta[a:b].view_as(t))

CH = [7,8,9,17,18]
def fwd_logPm(grad):
    for t in tensors:
        t.requires_grad_(grad)
        if t.grad is not None: t.grad=None
    o = fcs(cenc, rum, data["X_static"], data["X_dynamic"], data["edge_index"], data["t_per_mode"],
            data["mode_names"], data["log_d_ij"], data["match_prob"], data["log_M_z"], data["log_W_z"],
            data["log_D_z"], data["income_score"], data["pct_kids"], data["mean_cars"],
            data["income_tier_props"], data["pi_m_pair"], data["grid_borough_idx"],
            data["observed_OD"], data["val_mask"], soc_props_per_origin=data.get("soc_props_per_origin"),
            per_soc_demand_share_j=data.get("per_soc_demand_share_j"))
    return o["log_P_m"][CH]                                            # (5,N,N,M)

w = data["observed_OD"][CH]                                            # (5,N,N) flow weight per (t,i,j)
with torch.no_grad():
    base = fwd_logPm(False); logPm_det = base.clone(); pm = base.exp(); del base
torch.cuda.empty_cache()

FIM = torch.zeros(N, N, device=dev); t0 = time.time()
for k in range(N):
    h = 0.02 * float(scale[k])
    tp = theta0.clone(); tp[k] += h; set_flat(tp)
    with torch.no_grad():
        u = (fwd_logPm(False) - logPm_det) / h                        # (5,N,N,M) = J@e_k
    set_flat(theta0)
    pu = (pm*u).sum(-1, keepdim=True)
    Mu = (w.unsqueeze(-1) * (pm*u - pm*pu)).detach()                  # softmax curvature over modes, flow-weighted
    del u, pu; torch.cuda.empty_cache()
    logPm_g = fwd_logPm(True)
    col = torch.autograd.grad((logPm_g * Mu).sum(), tensors, allow_unused=True)
    FIM[:,k] = torch.cat([(c if c is not None else torch.zeros_like(t)).reshape(-1)
                          for c,t in zip(col, tensors)]).detach()
    del logPm_g, Mu, col; torch.cuda.empty_cache()
    if (k+1)%10==0: print(f"  col {k+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
set_flat(theta0)
FIM = 0.5*(FIM+FIM.t())
S = torch.diag(scale); FIM_s = (S@FIM@S).cpu().double().numpy()
np.savez(f"{V3}/evaluation_outputs/paper_a/gn_mode_fisher_s0.npz",
         FIM_s=FIM_s, names=np.array(names), groups=np.array(groups), scale=scale.cpu().numpy())
ev=np.linalg.eigvalsh(FIM_s)
print(f"\nmode GN eig: min={ev.min():.2e} max={ev.max():.2e}")
# which groups get the most mode-share info
diag=np.diag(FIM_s)
for g in dict.fromkeys(groups):
    idx=[i for i in range(N) if groups[i]==g]; print(f"  {g:26s} mode-info(diag mean)={np.mean(diag[idx]):.2e}")
print("saved gn_mode_fisher_s0.npz")
