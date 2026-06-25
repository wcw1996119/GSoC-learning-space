# -*- coding: utf-8 -*-
"""Complete main figure: posterior of each behaviour, from commuting flows alone
(grey) vs after adding ALL injected external data (green) = occupation census +
class census + observed mode shares. Forces pinned by flows; occupation, class
wage-pull, and travel-mode preferences all contracted by their external moments."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = r"D:\GIT\mesa Gsoc\GSoC-learning-space\models\04_london_nested_logit_v3\report\figures"
d = np.load(R+r"\empirical_fisher_s0.npz", allow_pickle=True)   # F_dest (flows)
Fd = d["FIM_s"]; groups = d["groups"]; scale = d["scale"]; theta0 = d["theta0"]; N = len(groups)
md = np.load(R+r"\gn_mode_fisher_s0.npz", allow_pickle=True)     # F_mode (mode shares)
Fm = md["FIM_s"]; Fm = 0.5*(Fm+Fm.T)
ev=np.linalg.eigvalsh(Fm); Fm = Fm + max(0,-ev.min()+1e-9)*np.eye(N)   # PSD-clip
DM = np.where(groups=="Occupation sorting")[0]; AW = np.where(groups=="Wage pull (class)")[0]
occ_ext = torch.tensor([0.033,0.130,0.077,-0.019,0.252,0.244,0.435,0.337,0.382],dtype=torch.double)
def pen(th):
    dm=torch.nn.functional.softplus(th[DM[0]:DM[-1]+1]); dmz=(dm-dm.mean())/(dm.std()+1e-6)
    ez=(occ_ext-occ_ext.mean())/(occ_ext.std()+1e-6); Lo=100*((dmz-ez)**2).mean()
    aw=torch.nn.functional.softplus(th[AW[0]:AW[-1]+1]); awz=(aw-aw.mean())/(aw.std()+1e-6)
    ct=torch.linspace(0,1,len(AW),dtype=torch.double); ctz=(ct-ct.mean())/(ct.std()+1e-6)
    return Lo+100*((awz-ctz)**2).mean()
th=torch.tensor(theta0,dtype=torch.double,requires_grad=True)
Hc=torch.autograd.functional.hessian(pen,th).numpy(); S=np.diag(scale); Hc_s=S@Hc@S
lam=np.median(np.diag(Fd))*0.05
Cb=np.linalg.inv(Fd + lam*np.eye(N))                      # flows alone
Ca=np.linalg.inv(Fd + Hc_s + Fm + lam*np.eye(N))         # + occ census + class census + mode shares
def agg(g,C):
    idx=np.where(groups==g)[0]; u=np.zeros(N); u[idx]=1/len(idx); return float(np.sqrt(u@C@u))
def contrast(g,C):  # SD in the within-group contrast subspace (gauge-fixed: remove mean)
    idx=np.where(groups==g)[0]; n=len(idx)
    if n<2: return float(np.sqrt(np.diag(C)[idx][0]))
    Cg=C[np.ix_(idx,idx)]; P=np.eye(n)-np.ones((n,n))/n; Cp=P@Cg@P
    wv=np.linalg.eigvalsh(Cp); wv=wv[wv>1e-12]; return float(np.sqrt(np.mean(wv)))

PRETTY={"Job-mass pull (gravity)":"Pull toward job-dense areas","Time sensitivity":"Sensitivity to travel time",
 "Cost budget":"Travel-cost budget","Time x distance":"Time cost rises with distance",
 "Crowding avoidance":"Avoiding crowded workplaces","Distance decay":"Distance decay",
 "Cost threshold":"Cost tolerance","Commute-time tolerance":"Commute-time tolerance",
 "Mode constants":"Baseline mode preference","Mode x income":"Mode choice x income",
 "Mode x kids":"Mode choice x children","Mode x cars":"Mode choice x car access",
 "Wage pull (class)":"Pull to high-pay areas (class)","Occupation sorting":"Occupation - matching workplaces"}
# MAIN figure: only the parameters with a clear identifiability story —
# 6 forces pinned by flows + 3 group preferences restored to near-point by external data.
# (The partially-identified ones — filter thresholds, gauge/weak mode contrasts — go to a supplement.)
ext_groups={"Wage pull (class)","Occupation sorting","Mode x cars"}
order=["Job-mass pull (gravity)","Cost budget","Time sensitivity","Time x distance",
 "Crowding avoidance","Distance decay",
 "Mode x cars","Wage pull (class)","Occupation sorting"]
# forces: report overall strength (agg), grey only (pinned by flows, green=grey).
# group preferences: report breakdown (brk), grey->green (external data contracts).
rows=[]
for g in order:
    if g in ext_groups: rows.append((PRETTY[g], contrast(g,Cb), contrast(g,Ca), True))
    else: rows.append((PRETTY[g], agg(g,Cb), agg(g,Cb), False))
rows=rows[::-1]
xx=np.linspace(-4,4,500)
def gauss(s): s=max(s,1e-3); y=np.exp(-0.5*(xx/s)**2); return y/y.max()

fig,ax=plt.subplots(figsize=(11,9.5))
for i,(lab,sb,sa,ext) in enumerate(rows):
    b=i*1.0
    ax.fill_between(xx,b,b+0.92*gauss(sb),color="#c7c7c7",alpha=0.6,zorder=2)
    ax.plot(xx,b+0.92*gauss(sb),color="#777",lw=1,zorder=3)
    if ext:
        ax.fill_between(xx,b,b+0.92*gauss(sa),color="#1a9850",alpha=0.8,zorder=4)
        ax.plot(xx,b+0.92*gauss(sa),color="#0b5d2e",lw=1.5,zorder=5)
    ax.text(-4.15,b+0.12,lab,ha="right",va="bottom",fontsize=9.5,
            color="#0b5d2e" if ext else "#333",fontweight="bold" if ext else "normal")
ax.axvline(0,color="#bbb",ls=":",lw=1)
ax.set_xlim(-4,5.4); ax.set_ylim(-0.2,len(rows)+0.2); ax.set_yticks([])
ax.set_xlabel("preference value  (standardised; width = posterior uncertainty)",fontsize=10.5)
ax.set_title("What commuting flows pin down, and what the injected external data restores",
             fontsize=13,fontweight="bold",loc="left")
ax.text(4.25,len(rows)-1.0,"UNIVERSAL FORCES\n→ pinned by the\ncommuting flows",fontsize=9.5,color="#444",ha="left",va="top")
ax.text(4.25,2.4,"GROUP PREFERENCES\nflows blind (grey)\n→ external data\nrestores to near-point\n(green):\n• occupation census\n• class census\n• mode shares",
        fontsize=9.5,color="#0b5d2e",ha="left",va="top",fontweight="bold")
ax.legend(handles=[Patch(facecolor="#c7c7c7",alpha=.6,label="from commuting flows alone"),
                   Patch(facecolor="#1a9850",alpha=.8,label="after adding all external data")],
          fontsize=9.5,loc="upper left",frameon=True)
fig.tight_layout()
out=R+r"\posterior_ridgeline.png"
fig.savefig(out,dpi=150,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
for lab,sb,sa,ext in rows[::-1]: print(f"  {lab:34s} {sb:.3f}->{sa:.3f} {'EXT' if ext else ''}")
print("saved",out)
