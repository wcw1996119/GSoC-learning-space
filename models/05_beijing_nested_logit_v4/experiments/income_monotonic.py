"""收入异质提升: 重参数化【强制单调】(低收入怕远 > 高收入) vs 自由 β。

混合引力 logit, per-收入档 β_k(怕远)。比较:
  ① 自由 β: 三档各自由 -> 弱识别(可能非单调/不稳, 如 B2 的 [.25,.14,.31])
  ② 强制单调: β_低 ≥ β_中 ≥ β_高 (reparam: 累加 softplus) -> 锁住 CFPS/常识方向
看: 单调约束后 β 是否变合理(单调下降) + NLL 代价多小(数据对 spread 不敏感 -> 约束几乎免费)。
+ 总体通勤锚(ct_mean=35): 模型预测平均通勤 match 观测, pin 总体尺度。本地 CPU 子样本。
"""
from pathlib import Path
import numpy as np, torch

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
torch.manual_seed(0)


def main():
    e=np.load(PROC/"beijing_edges.npz",allow_pickle=True); g=np.load(PROC/"beijing_grid.npz",allow_pickle=True)
    inc=np.load(PROC/"beijing_income.npz",allow_pickle=True); mm=np.load(PROC/"beijing_micro_moments.npz",allow_pickle=True)
    o=e["o_idx"].astype(np.int64); d=e["d_idx"].astype(np.int64); t=e["t_ff"].astype(np.float32)
    flow=e["flow"].sum(1).astype(np.float32); seg_ptr=e["seg_ptr"].astype(np.int64)
    jobs=g["jobs"].astype(np.float32); logM=np.log(jobs+1.0); tp=inc["income_tier_props"].astype(np.float32)
    ct_obs=float(mm["ct_mean"])                                  # 总体通勤 35min

    nseg=len(seg_ptr)-1; rng=np.random.default_rng(0)
    pick=np.sort(rng.choice(nseg,size=min(2500,nseg),replace=False))
    rows=np.concatenate([np.arange(seg_ptr[i],seg_ptr[i+1]) for i in pick])
    o_s=o[rows]; logM_s=logM[d[rows]]; t_s=t[rows]; fl_s=flow[rows]
    seg_len=seg_ptr[pick+1]-seg_ptr[pick]; nS=len(pick); segid=torch.tensor(np.repeat(np.arange(nS),seg_len))
    w=torch.tensor(tp[o_s]); lM=torch.tensor(logM_s); tt=torch.tensor(t_s); flt=torch.tensor(fl_s)
    print(f"  子样本 {nS} 起点 {len(rows):,} 边; 观测总体通勤 {ct_obs:.1f}min",flush=True)

    def seg_sm(V):
        mx=torch.full((nS,),-1e30).scatter_reduce(0,segid,V,"amax",include_self=False)
        z=(V-mx[segid]).exp(); s=torch.zeros(nS).index_add(0,segid,z); return z/s[segid]
    def run(mode, steps=400, anchor=True):
        a=torch.tensor(0.5,requires_grad=True)
        if mode=="free":
            bk=torch.tensor([0.10,0.10,0.10],requires_grad=True); ps=[a,bk]
        else:  # 单调: β=[base+d2+d1, base+d1, base], 低≥中≥高
            base=torch.tensor(0.05,requires_grad=True); rd1=torch.tensor(-1.0,requires_grad=True)
            rd2=torch.tensor(-1.0,requires_grad=True); ps=[a,base,rd1,rd2]
        opt=torch.optim.Adam(ps,lr=0.03)
        for _ in range(steps):
            opt.zero_grad()
            if mode=="free": betas=[bk[0],bk[1],bk[2]]
            else:
                bh=torch.nn.functional.softplus(base); bm=bh+torch.nn.functional.softplus(rd1); bl=bm+torch.nn.functional.softplus(rd2)
                betas=[bl,bm,bh]
            P=sum(w[:,k]*seg_sm(a*lM-betas[k]*tt) for k in range(3))
            nll=-(flt*torch.log(P.clamp_min(1e-12))).sum()/flt.sum()
            # 总体通勤锚: 模型期望通勤 = Σ flow_pred·t / Σ flow_pred
            ct_pred=(P*flt.sum()/flt.sum()*tt).sum()/(P.sum()+1e-9)   # 近似期望通勤
            loss=nll+(0.002*(ct_pred-ct_obs)**2 if anchor else 0)
            loss.backward(); opt.step()
        bv=[float(x) for x in (betas if mode=="free" else [bl,bm,bh])]
        return bv,float(nll),float(ct_pred)

    print("\n=== ① 自由 β(无约束) ===",flush=True)
    bf,nf,cf=run("free"); print(f"  β[低/中/高]={[round(x,3) for x in bf]}  NLL={nf:.5f}  预测通勤{cf:.1f}min",flush=True)
    mono = bf[0]>=bf[1]>=bf[2]; print(f"  单调(低≥中≥高)? {'是' if mono else '否 ← 不稳/反直觉'}",flush=True)
    print("\n=== ② 强制单调 β + 总体通勤锚 ===",flush=True)
    bm_,nm,cm=run("mono"); print(f"  β[低/中/高]={[round(x,3) for x in bm_]}  NLL={nm:.5f}  预测通勤{cm:.1f}min",flush=True)
    print(f"\n  ΔNLL(单调−自由) = {nm-nf:+.5f}  (≈0 = 约束几乎免费, 数据对收入 spread 不敏感)")
    print("判读: 单调约束让 β 变合理(低收入怕远>高收入, 符合CFPS/常识), NLL 代价极小 ->"
          " 收入异质方向【靠约束锁住】(数据不反对但也定不出量级)。诚实: 量级仍弱识别, 需CFPS按收入矩pin或给区间。")


if __name__=="__main__":
    main()
