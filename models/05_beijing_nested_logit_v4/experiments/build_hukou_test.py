"""验"户口当控制变量 de-confound": 放户口进去, 别的偏好(距离厌恶 β)变干净吗?

① 1% 普查: 本地/外地(京外户口 M39) + 区级外地占比 + 各自通勤(锚)。
② 区级广播到 cell。
③ 模型对比:
   M1 单一 β(不控户口)= pooled, 混了本地+外地;
   M2 户口混合 β_本地/β_外地 + 通勤锚 = 分开。
看: pooled β(M1) vs 本地 β(M2) 差多少 = 户口对"真实居民偏好"的污染量级 = de-confound 价值。
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.manual_seed(0)
PROC=Path(__file__).resolve().parents[1]/"data"/"processed"
CSV=Path(r"D:\UoM\congestion_project\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式"
         r"\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式.csv")
CODE2NAME={"110101":"东城","110102":"西城","110105":"朝阳","110106":"丰台","110107":"石景山",
 "110108":"海淀","110109":"门头沟","110111":"房山","110112":"通州","110113":"顺义",
 "110114":"昌平","110115":"大兴","110116":"怀柔","110117":"平谷","110118":"密云","110119":"延庆"}


def main():
    n=lambda s:pd.to_numeric(s,errors="coerce")
    df=pd.read_csv(CSV,dtype=str,usecols=lambda c:c in ["M2","M39","M59","M62"],low_memory=False)
    df=df[df["M2"].astype(str).str.startswith("11")].copy()
    df=df[n(df["M59"]).isin([1,2])&(n(df["M62"])>0)&(n(df["M62"])<240)].copy()
    prov=df["M39"].fillna("").str[:2]; df["mig"]=(prov.str.isdigit()&(prov!="11")).astype(int)
    df["ct"]=n(df["M62"]); df["dc"]=df["M2"].astype(str).str[:6]
    anch=df.groupby("mig")["ct"].mean().values                        # [本地, 外地] 通勤锚
    print(f"  通勤锚[本地/外地]={anch.round(1)}  外地总占比{df['mig'].mean()*100:.0f}%")

    g=np.load(PROC/"beijing_grid.npz",allow_pickle=True)
    dnames=[str(x) for x in g["district_names"]]; didx=g["district_idx"].astype(np.int64)
    name2gi={nm:i for i,nm in enumerate(dnames)}
    migshare=np.full(len(dnames),df["mig"].mean())                    # 默认全市均值
    for dc,sub in df.groupby("dc"):
        nm=CODE2NAME.get(dc); gi=name2gi.get(nm) if nm else None
        if gi is None and nm:
            for k,v in name2gi.items():
                if nm in k or k in nm: gi=v; break
        if gi is not None: migshare[gi]=sub["mig"].mean()
    print(f"  区级外地占比: 朝阳{migshare[name2gi.get('朝阳',0)]:.2f} 海淀{migshare[name2gi.get('海淀',0)]:.2f} "
          f"东城{migshare[name2gi.get('东城',0)]:.2f}")
    cell_mig=migshare[didx]                                           # 每cell外地占比
    hk=np.stack([1-cell_mig, cell_mig],1)                            # (Ncell,2) [本地,外地]构成

    e=np.load(PROC/"beijing_edges.npz",allow_pickle=True)
    o=e["o_idx"].astype(np.int64); d=e["d_idx"].astype(np.int64); t=e["t_ff"].astype(np.float32)
    flow=e["flow"].sum(1).astype(np.float32); seg_ptr=e["seg_ptr"].astype(np.int64)
    jobs=g["jobs"].astype(np.float32); logM=np.log(jobs+1.0)
    rng=np.random.default_rng(0); nseg=len(seg_ptr)-1
    pick=np.sort(rng.choice(nseg,size=2500,replace=False))
    rows=np.concatenate([np.arange(seg_ptr[i],seg_ptr[i+1]) for i in pick])
    o_s=o[rows]; lM=torch.tensor(logM[d[rows]]); tt=torch.tensor(t[rows]); flt=torch.tensor(flow[rows])
    seg_len=seg_ptr[pick+1]-seg_ptr[pick]; nS=len(pick); segid=torch.tensor(np.repeat(np.arange(nS),seg_len))
    w=torch.tensor(hk[o_s]); ancht=torch.tensor(anch,dtype=torch.float32)
    print(f"  子样本 {nS} 起点 {len(rows):,} 边",flush=True)

    def seg_sm(V):
        mx=torch.full((nS,),-1e30).scatter_reduce(0,segid,V,"amax",include_self=False)
        z=(V-mx[segid]).exp(); s=torch.zeros(nS).index_add(0,segid,z); return z/s[segid]
    def m1(steps=400):  # 单一 β
        a=torch.tensor(0.5,requires_grad=True); b=torch.tensor(0.1,requires_grad=True)
        opt=torch.optim.Adam([a,b],lr=0.03)
        for _ in range(steps):
            opt.zero_grad(); P=seg_sm(a*lM-b*tt)
            (-(flt*torch.log(P.clamp_min(1e-12))).sum()/flt.sum()).backward(); opt.step()
        return float(b)
    def m2(steps=500):  # 户口混合 + 锚
        a=torch.tensor(0.5,requires_grad=True); bk=torch.tensor([0.1,0.13],requires_grad=True)
        opt=torch.optim.Adam([a,bk],lr=0.03)
        for _ in range(steps):
            opt.zero_grad(); Ps=[seg_sm(a*lM-bk[h]*tt) for h in range(2)]
            P=sum(w[:,h]*Ps[h] for h in range(2))
            nll=-(flt*torch.log(P.clamp_min(1e-12))).sum()/flt.sum()
            ctk=torch.stack([(w[:,h]*Ps[h]*tt).sum()/((w[:,h]*Ps[h]).sum()+1e-9) for h in range(2)])
            (nll+0.01*((ctk-ancht)**2).mean()).backward(); opt.step()
        return [round(float(x),3) for x in bk]

    print("\n=== M1 单一 β(不控户口, pooled) ===",flush=True)
    bp=m1(); print(f"  pooled β = {bp:.3f}",flush=True)
    print("=== M2 户口混合 β + 通勤锚 ===",flush=True)
    bk=m2(); print(f"  β[本地/外地] = {bk}",flush=True)
    print(f"\n  本地 β(M2)={bk[0]} vs pooled β(M1)={bp:.3f}  差 {bk[0]-bp:+.3f}")
    print("判读: 外地 β 应>本地(外地选近=高距离厌恶, 因预选住所); 本地 β(去confound)若 != pooled "
          "=> 户口确实在污染'真实居民'偏好, 控制有价值。差越大, de-confound 越重要。")


if __name__=="__main__":
    main()
