"""Profile likelihood 诊断: 收入异质(怕远 β 按收入档分)在聚合北京 OD 上识别得出吗?

混合引力 logit: P(j|i)=Σ_k w_ik·softmax(α·logM − β_k·t), w_ik=该cell收入档构成(tier_props)。
β_k = β_mean + spread·(k−1)  (低/中/高 = β_mean−spread / β_mean / β_mean+spread)。
扫 spread, 每点重优化(α,β_mean), 记 OD-NLL:
  曲线【陡】(有明显最小)= 收入异质识别得出; 【平】= 识别不出(= 聚合OD收入盲)。
对比: 也扫 β_mean(已知识别得出方向)看"陡"长啥样。本地 CPU, 子样本提速。
"""
from pathlib import Path
import numpy as np, torch

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
torch.manual_seed(0)


def main():
    e = np.load(PROC/"beijing_edges.npz", allow_pickle=True); g = np.load(PROC/"beijing_grid.npz", allow_pickle=True)
    inc = np.load(PROC/"beijing_income.npz", allow_pickle=True)
    o=e["o_idx"].astype(np.int64); d=e["d_idx"].astype(np.int64); t=e["t_ff"].astype(np.float32)
    flow=e["flow"].sum(1).astype(np.float32); seg_ptr=e["seg_ptr"].astype(np.int64)
    jobs=g["jobs"].astype(np.float32); logM=np.log(jobs+1.0)
    tp=inc["income_tier_props"].astype(np.float32)            # (Ncell,3) 收入档构成
    print(f"  tier_props 形状 {tp.shape}; 各档全市均值 {tp.mean(0).round(2)}; "
          f"低档跨cell std {tp[:,0].std():.3f}(有变异才可能识别)")

    # 子样本: 随机取 segments 提速
    nseg=len(seg_ptr)-1; rng=np.random.default_rng(0)
    pick=np.sort(rng.choice(nseg, size=min(1500,nseg), replace=False))
    rows=np.concatenate([np.arange(seg_ptr[i],seg_ptr[i+1]) for i in pick])
    o_s=o[rows]; logM_s=logM[d[rows]]; t_s=t[rows]; fl_s=flow[rows]
    seg_len=seg_ptr[pick+1]-seg_ptr[pick]; segid=torch.tensor(np.repeat(np.arange(len(pick)),seg_len))
    nS=len(pick); w=torch.tensor(tp[o_s])                    # (E,3) 该边出发地收入构成
    lM=torch.tensor(logM_s); tt=torch.tensor(t_s); flt=torch.tensor(fl_s)
    print(f"  子样本 {nS} 起点, {len(rows):,} 边")

    def seg_sm(V):
        mx=torch.full((nS,),-1e30).scatter_reduce(0,segid,V,"amax",include_self=False)
        z=(V-mx[segid]).exp(); s=torch.zeros(nS).index_add(0,segid,z); return z/s[segid]

    def nll_at(spread, free_mean=True, fixed_mean=0.10, steps=120):
        a=torch.tensor(0.5,requires_grad=True)
        bm=torch.tensor(fixed_mean,requires_grad=free_mean)
        params=[a]+([bm] if free_mean else [])
        opt=torch.optim.Adam(params,lr=0.03)
        for _ in range(steps):
            opt.zero_grad()
            betas=[bm-spread, bm, bm+spread]                  # 低/中/高
            P=sum(w[:,k]*seg_sm(a*lM-betas[k]*tt) for k in range(3))
            loss=-(flt*torch.log(P.clamp_min(1e-12))).sum()/flt.sum()
            loss.backward(); opt.step()
        return float(loss)

    print("\n=== ① 扫 收入档间 spread (怕远的收入异质) ===", flush=True)
    base=None
    for sp in [0.0, 0.05, 0.10, 0.20]:
        nl=nll_at(sp); base=nl if base is None else base
        print(f"  spread={sp:.2f}: NLL={nl:.5f}  ΔNLL={nl-base:+.5f}", flush=True)

    print("\n=== ② 对比: 扫 总体 β_mean (已知识别得出方向, spread=0) ===", flush=True)
    vals=[(bm, nll_at(0.0, free_mean=False, fixed_mean=bm)) for bm in [0.05,0.10,0.15]]
    mn=min(v for _,v in vals)
    for bm,nl in vals: print(f"  β_mean={bm:.2f}: NLL={nl:.5f}  Δ={nl-mn:+.5f}", flush=True)

    print("\n判读: ① spread 曲线若【平】(ΔNLL~1e-3 量级跟噪声) = 收入异质识别不出(聚合OD收入盲);"
          " ② β_mean 曲线【陡】(Δ 大) = 这才是识别得出的样子。两者对比量级 = 你的可识别性证据。")


if __name__=="__main__":
    main()
