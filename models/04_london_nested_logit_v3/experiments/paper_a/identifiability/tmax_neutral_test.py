# -*- coding: utf-8 -*-
"""Credibility test for the income-graded commute tolerance. Reset T_max to a
NEUTRAL init (90/90/90, all equal), fine-tune the head on the flows, and track
whether T_max spontaneously diverges into the low<mid<high income gradient.
- diverges to ~59/88/123  -> the flows DO inform it (data-driven finding)
- stays near 90/90/90      -> the flows DON'T (it was a literature prior)."""
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

orig = F.softplus(rum.raw_T_max_per_tier).detach().cpu().numpy()
print("original learned T_max:", np.round(orig,1))
with torch.no_grad():
    rum.raw_T_max_per_tier.copy_(torch.tensor([90.,90.,90.],device=dev))   # NEUTRAL
print("reset to neutral:", np.round(F.softplus(rum.raw_T_max_per_tier).detach().cpu().numpy(),1))

# fine-tune ALL head params from this neutral T_max (same lr as training: lr_rum=0.01)
for p in rum.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam(rum.parameters(), lr=0.01)
traj=[]
for step in range(400):
    opt.zero_grad(); L=loss(); L.backward(); opt.step()
    if step%20==0 or step==399:
        T=F.softplus(rum.raw_T_max_per_tier).detach().cpu().numpy()
        traj.append((step,float(L.detach()),*T))
        print(f"  step {step:3d} nll {float(L.detach()):.5f}  T_max {np.round(T,1)}  ratio 1:{T[1]/T[0]:.2f}:{T[2]/T[0]:.2f}", flush=True)
Tf=F.softplus(rum.raw_T_max_per_tier).detach().cpu().numpy()
np.savez(f"{V3}/evaluation_outputs/paper_a/tmax_neutral_test.npz", traj=np.array(traj), orig=orig, final=Tf)
print(f"\noriginal: {np.round(orig,1)}  neutral-start: [90,90,90]  fine-tuned: {np.round(Tf,1)}")
mono = Tf[0]<Tf[1]<Tf[2]
print(f"recovered income gradient (low<mid<high)? {mono}   spread={Tf.max()-Tf.min():.1f} min")
print("saved tmax_neutral_test.npz")
