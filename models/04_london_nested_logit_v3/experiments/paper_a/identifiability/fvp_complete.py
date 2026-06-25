# -*- coding: utf-8 -*-
"""Finding-vs-prior under the COMPLETE model loss (flows + ALL external moments:
mode shares + occupation census + class census). For each group: flatten its
within-group structure, fine-tune on the FULL training objective, see if the
structure is RESTORED (by flows OR by the external moment that targets it)."""
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
def full_loss():
    o=fcs(cenc,rum,data["X_static"],data["X_dynamic"],data["edge_index"],data["t_per_mode"],data["mode_names"],
        data["log_d_ij"],data["match_prob"],data["log_M_z"],data["log_W_z"],data["log_D_z"],data["income_score"],
        data["pct_kids"],data["mean_cars"],data["income_tier_props"],data["pi_m_pair"],data["grid_borough_idx"],
        data["observed_OD"],data["val_mask"],soc_props_per_origin=data.get("soc_props_per_origin"),
        per_soc_demand_share_j=data.get("per_soc_demand_share_j"))
    L=o["nll_dest"]+KL*o["ce_mode"]
    dm=rum.delta_match; dmz=(dm-dm.mean())/(dm.std()+1e-6); ez=(occ_ext-occ_ext.mean())/(occ_ext.std()+1e-6)
    L=L+LO*((dmz-ez)**2).mean()
    aw=F.softplus(rum.raw_alpha_wage); awz=(aw-aw.mean())/(aw.std()+1e-6)
    ct=torch.linspace(0,1,aw.numel(),device=dev); ctz=(ct-ct.mean())/(ct.std()+1e-6)
    L=L+LC*((awz-ctz)**2).mean()
    return L

GROUPS=[("commute tolerance (T_max)","raw_T_max_per_tier"),("distance decay","raw_gamma_decay"),
    ("time sensitivity","raw_beta_t_per_mode"),("gravity (job-mass)","raw_gamma_M"),
    ("competition (crowding)","raw_nu_D"),("cost budget","raw_cost_budget_per_tier"),
    ("cost threshold","raw_cost_thresh_per_tier"),("mode constants","asc_per_mode"),
    ("mode x cars","theta_cars_per_mode"),("wage pull (class)","raw_alpha_wage"),
    ("occupation sorting","raw_delta_match")]
trained={a:getattr(rum,a).detach().clone() for _,a in GROUPS}
all_p=list(rum.parameters())
def restore():
    with torch.no_grad():
        for _,a in GROUPS: getattr(rum,a).copy_(trained[a])
results=[]
for gname,attr in GROUPS:
    restore(); p=getattr(rum,attr); tr=trained[attr]
    with torch.no_grad(): p.copy_(tr.mean().expand_as(p))
    for q in all_p: q.requires_grad_(True)
    opt=torch.optim.Adam(all_p,lr=0.01)
    for step in range(150):
        opt.zero_grad(); full_loss().backward(); opt.step()
    final=p.detach().clone()
    td=(tr-tr.mean()).cpu().numpy(); fd=(final-final.mean()).cpu().numpy()
    sr=float(np.std(fd)/(np.std(td)+1e-9)); cr=float(np.corrcoef(td,fd)[0,1]) if tr.numel()>1 and np.std(fd)>1e-9 else 0.0
    v="FINDING" if (sr>0.4 and cr>0.5) else ("partial" if sr>0.2 else "PRIOR")
    results.append((gname,sr,cr,v)); print(f"  {gname:26s} spread={sr:5.2f} corr={cr:+.2f} -> {v}",flush=True)
restore()
np.savez(f"{V3}/evaluation_outputs/paper_a/fvp_complete.npz",
         names=np.array([r[0] for r in results]),spread=np.array([r[1] for r in results]),
         corr=np.array([r[2] for r in results]),verdict=np.array([r[3] for r in results]))
print("\n=== COMPLETE-MODEL finding vs prior ===")
for g,s,c,v in results: print(f"  {v:8s} | {g} (spread {s:.2f}, corr {c:+.2f})")
print("saved fvp_complete.npz")
