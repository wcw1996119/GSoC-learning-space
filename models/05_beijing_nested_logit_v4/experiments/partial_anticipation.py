"""Paper B 部分预判: 数据直接估 agent 多大程度预判拥堵(φ) + 去confound β。

agent 响应 τ=t_ff+φ·delay (delay=t_obs−t_ff). 效用 V=α·logM − β·t_ff − βφ·delay。
拟合 V=α·logM − b1·t_ff − b2·delay (两个时间回归量):
  b1=β(对自由流时间的敏感度=去confound β), b2=βφ, φ=b2/b1=预判程度(0不预判/1完全)。
对比: t_ff单拟合β=0.177(φ=0假设), t_obs单拟合β=0.107(φ=1假设)。
诚实caveat: delay内生(=拥堵), b2有偏(偏低)→ φ是下界; 均衡约束能纠偏(后续). 本地CPU全量.
"""
from pathlib import Path
import numpy as np, torch
PROC=Path(__file__).resolve().parents[1]/"data"/"processed"


def seg_nll(V,seg_id,nseg,flow):
    mx=torch.full((nseg,),-1e30).scatter_reduce(0,seg_id,V,"amax",include_self=False)
    z=V-mx[seg_id]; ssum=torch.zeros(nseg).index_add(0,seg_id,z.exp())
    logP=z-ssum[seg_id].clamp_min(1e-30).log()
    return -(flow*logP).sum()/flow.sum()


def main():
    e=np.load(PROC/"beijing_edges.npz",allow_pickle=True); g=np.load(PROC/"beijing_grid.npz",allow_pickle=True)
    d=e["d_idx"].astype(np.int64); t_obs=e["t_obs"].astype(np.float32); t_ff=e["t_ff"].astype(np.float32)
    flow=e["flow"].sum(1).astype(np.float32); seg_ptr=e["seg_ptr"].astype(np.int64)
    jobs=g["jobs"].astype(np.float32); logM=np.log(jobs+1.0)
    delay=np.maximum(t_obs-t_ff,0.0)
    nseg=len(seg_ptr)-1; seg_id=torch.from_numpy(np.repeat(np.arange(nseg),np.diff(seg_ptr)))
    lM=torch.from_numpy(logM[d]); tff=torch.from_numpy(t_ff); dly=torch.from_numpy(delay); fl=torch.from_numpy(flow)
    print(f"  北京全量 {len(d):,}边; delay 中位{np.median(delay):.1f}min, >0占{(delay>0).mean()*100:.0f}%",flush=True)

    a=torch.tensor(0.5,requires_grad=True); b1=torch.tensor(0.15,requires_grad=True); b2=torch.tensor(0.05,requires_grad=True)
    opt=torch.optim.Adam([a,b1,b2],lr=0.02)
    for it in range(500):
        opt.zero_grad()
        V=a*lM - b1*tff - b2*dly
        loss=seg_nll(V,seg_id,nseg,fl); loss.backward(); opt.step()
        if it%100==0: print(f"    it{it} b1={float(b1):.4f} b2={float(b2):.4f}",flush=True)
    B1,B2=float(b1),float(b2); phi=B2/B1 if B1>1e-6 else float('nan')
    print(f"\n  ⭐ 两回归量拟合: β(=b1,对自由流)={B1:.4f}  b2(=βφ)={B2:.4f}  → φ(预判程度)={phi:.2f}")
    print(f"     对比: t_ff单拟合β=0.177(假设φ=0), t_obs单拟合β=0.107(假设φ=1)")
    print(f"  判读: φ={phi:.2f} = agent 把拥堵延误算进 {phi*100:.0f}%; 去confound β={B1:.3f}。"
          " φ内生有偏(偏低,delay内生)→ 真φ可能更高, 均衡约束纠偏是后续。"
          " 但这已从数据直接给出'北京通勤者部分预判拥堵'的量级。")


if __name__=="__main__":
    main()
