# -*- coding: utf-8 -*-
"""Batch 'finding vs prior' test. For each structural parameter group: flatten its
within-group structure (set every component to the group mean), fine-tune ALL
params on the flows, and see whether the data RESTORES the structure.
- structure returns (spread recovers, correlates with trained) -> data identifies it (FINDING)
- stays flat                                                    -> data doesn't (PRIOR / lit-anchored)."""
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
def loss():
    o=fcs(cenc,rum,data["X_static"],data["X_dynamic"],data["edge_index"],data["t_per_mode"],data["mode_names"],
        data["log_d_ij"],data["match_prob"],data["log_M_z"],data["log_W_z"],data["log_D_z"],data["income_score"],
        data["pct_kids"],data["mean_cars"],data["income_tier_props"],data["pi_m_pair"],data["grid_borough_idx"],
        data["observed_OD"],data["val_mask"],soc_props_per_origin=data.get("soc_props_per_origin"),
        per_soc_demand_share_j=data.get("per_soc_demand_share_j"))
    return o["nll_dest"]

GROUPS=[("commute tolerance (T_max)","raw_T_max_per_tier"),("distance decay","raw_gamma_decay"),
    ("time sensitivity","raw_beta_t_per_mode"),("gravity (job-mass)","raw_gamma_M"),
    ("competition (crowding)","raw_nu_D"),("cost budget","raw_cost_budget_per_tier"),
    ("cost threshold","raw_cost_thresh_per_tier"),("mode constants","asc_per_mode")]
trained={a:getattr(rum,a).detach().clone() for _,a in GROUPS}
all_p=list(rum.parameters())

def restore():
    with torch.no_grad():
        for _,a in GROUPS: getattr(rum,a).copy_(trained[a])

results=[]
for gname,attr in GROUPS:
    restore()
    p=getattr(rum,attr); tr=trained[attr]
    with torch.no_grad(): p.copy_(tr.mean().expand_as(p))      # FLATTEN within-group structure
    for q in all_p: q.requires_grad_(True)
    opt=torch.optim.Adam(all_p,lr=0.01)
    t0=time.time()
    for step in range(150):
        opt.zero_grad(); L=loss(); L.backward(); opt.step()
    final=p.detach().clone()
    tr_dev=(tr-tr.mean()).cpu().numpy(); fi_dev=(final-final.mean()).cpu().numpy()
    spread_ratio=float(np.std(fi_dev)/(np.std(tr_dev)+1e-9))
    corr=float(np.corrcoef(tr_dev,fi_dev)[0,1]) if tr.numel()>1 and np.std(fi_dev)>1e-9 else 0.0
    verdict="FINDING" if (spread_ratio>0.4 and corr>0.5) else ("partial" if spread_ratio>0.2 else "PRIOR")
    results.append((gname,spread_ratio,corr,verdict))
    print(f"  {gname:26s} spread_recovered={spread_ratio:5.2f}  corr={corr:+.2f}  -> {verdict}", flush=True)
restore()
np.savez(f"{V3}/evaluation_outputs/paper_a/finding_vs_prior.npz",
         names=np.array([r[0] for r in results]), spread=np.array([r[1] for r in results]),
         corr=np.array([r[2] for r in results]), verdict=np.array([r[3] for r in results]))
print("\n=== SUMMARY: which structural params the flows actually identify ===")
for g,s,c,v in results: print(f"  {v:8s} | {g} (spread {s:.2f}, corr {c:+.2f})")
print("saved finding_vs_prior.npz")
