# -*- coding: utf-8 -*-
"""Profile likelihood along the STRUCTURE axis (avoids the scale-compensation
confound). For each parameter group, morph it theta(a)=mean + a*(trained-mean):
a=0 scrambled (no structure), a=1 trained, a=2 overshoot. Fix it, RE-FIT all other
params on the complete objective, record the loss. U-shape (min at a=1) = the data
identifies the structure (FINDING); flat = it doesn't (PRIOR). Saves after EACH
group so a premature shutdown keeps partial results."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import importlib.util as ilu, torch, numpy as np
import torch.nn.functional as F
from pathlib import Path
V3 = Path("/root/autodl-tmp/v3_nested_smoke/04_london_nested_logit_v3")
spec = ilu.spec_from_file_location("scenA", V3/"experiments"/"paper_a"/"scenario_v3l_A.py")
m = ilu.module_from_spec(spec); spec.loader.exec_module(m); fcs = m.forward_cs
dev = torch.device("cuda")
ck = torch.load(f"{V3}/evaluation_outputs/paper_a/v3l_R7a_clean_s0.pt", map_location=dev, weights_only=False)
enc, rum, ta = m.reconstruct_model(ck, dev); ta.aux_path="data/processed/paperA_v3_aux_cervero.npz"
import os; os.chdir(V3)
data = m.load_scenario_inputs(ta, dev)
with torch.no_grad():
    Vraw = enc(data["X_static"], data["X_dynamic"], data["edge_index"]).contiguous()
class C:
    def __init__(s,v):s.v=v
    def __call__(s,*a,**k):return s.v
    def eval(s):return s
cenc=C(Vraw)
KL=float(ck["args"].get("lambda_kl",1.0)); LO=100.0; LC=100.0
occ_ext=torch.tensor([0.033,0.130,0.077,-0.019,0.252,0.244,0.435,0.337,0.382],device=dev)
def fwd():
    return fcs(cenc,rum,data["X_static"],data["X_dynamic"],data["edge_index"],data["t_per_mode"],data["mode_names"],
        data["log_d_ij"],data["match_prob"],data["log_M_z"],data["log_W_z"],data["log_D_z"],data["income_score"],
        data["pct_kids"],data["mean_cars"],data["income_tier_props"],data["pi_m_pair"],data["grid_borough_idx"],
        data["observed_OD"],data["val_mask"],soc_props_per_origin=data.get("soc_props_per_origin"),
        per_soc_demand_share_j=data.get("per_soc_demand_share_j"))
def full_loss(o):
    L=o["nll_dest"]+KL*o["ce_mode"]
    dm=rum.delta_match; dmz=(dm-dm.mean())/(dm.std()+1e-6); ez=(occ_ext-occ_ext.mean())/(occ_ext.std()+1e-6)
    L=L+LO*((dmz-ez)**2).mean()
    aw=F.softplus(rum.raw_alpha_wage); awz=(aw-aw.mean())/(aw.std()+1e-6)
    ct=torch.linspace(0,1,aw.numel(),device=dev); ctz=(ct-ct.mean())/(ct.std()+1e-6)
    return L+LC*((awz-ctz)**2).mean()

GROUPS=[("Pull toward job-dense areas","raw_gamma_M","flows"),("Distance decay","raw_gamma_decay","flows"),
    ("Sensitivity to travel time","raw_beta_t_per_mode","flows"),
    ("Occupation - matching workplaces","raw_delta_match","external"),
    ("Pull to high-pay areas (class)","raw_alpha_wage","external"),
    ("Mode choice x car access","theta_cars_per_mode","external"),
    ("Travel-cost budget","raw_cost_budget_per_tier","prior"),
    ("Commute-time tolerance","raw_T_max_per_tier","prior")]
trained={n:p.detach().clone() for n,p in rum.named_parameters()}
all_p=dict(rum.named_parameters())
def restore_all():
    with torch.no_grad():
        for n,p in all_p.items(): p.copy_(trained[n])
ALPHAS=[0.0,0.5,1.0,1.5,2.0]
names=[g[0] for g in GROUPS]; srcs=[g[2] for g in GROUPS]; curves=[]
for gname,attr,src in GROUPS:
    tr=trained[attr]; mean=tr.mean()
    others=[p for n,p in all_p.items() if n!=attr]
    row=[]; t0=time.time()
    for a in ALPHAS:
        restore_all()
        with torch.no_grad(): getattr(rum,attr).copy_(mean + a*(tr-mean))
        getattr(rum,attr).requires_grad_(False)
        for p in others: p.requires_grad_(True)
        opt=torch.optim.Adam(others,lr=0.01)
        for step in range(60):
            opt.zero_grad(); full_loss(fwd()).backward(); opt.step()
        with torch.no_grad(): row.append(float(full_loss(fwd())))
        getattr(rum,attr).requires_grad_(True)
    curves.append(row)
    twod=2*(np.array(row)-min(row))
    print(f"  {gname:30s} 2dLoss@a[0,.5,1,1.5,2]: "+" ".join(f"{v:.3f}" for v in twod)+f"  ({time.time()-t0:.0f}s)",flush=True)
    np.savez(f"{V3}/evaluation_outputs/paper_a/profile_structure_axis.npz",   # incremental save
             alphas=np.array(ALPHAS), names=np.array(names[:len(curves)]),
             srcs=np.array(srcs[:len(curves)]), curves=np.array(curves))
restore_all()
print("\nsaved profile_structure_axis.npz (all groups)")
