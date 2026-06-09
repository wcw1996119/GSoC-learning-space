"""验"教育替换收入": 教育异质(怕远 β 按教育档)识别得出吗? (对比收入的塌平)

① 1% 普查: 教育3档(M51) + 区级教育构成 + 各档通勤(干净锚)。
② 区级构成广播到 cell(grid.district_idx)。
③ 混合引力 logit, per-教育档 β: (a)自由 (b)+通勤锚。
看: 锚后 β 是否【单调下降+稳】(高教育怕远低=通勤长), 区别于收入的塌平/打架。
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.manual_seed(0)
PROC=Path(__file__).resolve().parents[1]/"data"/"processed"
CSV=Path(r"D:\UoM\congestion_project\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式"
         r"\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式.csv")
# 北京区码->名(标准)
CODE2NAME={"110101":"东城","110102":"西城","110105":"朝阳","110106":"丰台","110107":"石景山",
 "110108":"海淀","110109":"门头沟","110111":"房山","110112":"通州","110113":"顺义",
 "110114":"昌平","110115":"大兴","110116":"怀柔","110117":"平谷","110118":"密云","110119":"延庆"}


def edu_tier(e):  # M51 1-8 -> 0低/1中/2高
    return np.where(e<=3,0,np.where(e<=5,1,2))


def main():
    n=lambda s:pd.to_numeric(s,errors="coerce")
    df=pd.read_csv(CSV,dtype=str,usecols=lambda c:c in ["M2","M51","M59","M62"],low_memory=False)
    df=df[df["M2"].astype(str).str.startswith("11")].copy()
    df=df[n(df["M59"]).isin([1,2])&(n(df["M62"])>0)&(n(df["M62"])<240)].copy()
    df["et"]=edu_tier(n(df["M51"])); df["ct"]=n(df["M62"]); df["dc"]=df["M2"].astype(str).str[:6]
    anch=df.groupby("et")["ct"].mean().values                       # 各教育档通勤锚
    print(f"  教育档通勤锚[低/中/高] = {anch.round(1)} (n={df.groupby('et').size().values})")

    # 区级教育构成
    g=np.load(PROC/"beijing_grid.npz",allow_pickle=True)
    dnames=[str(x) for x in g["district_names"]]; didx=g["district_idx"].astype(np.int64)
    name2gi={nm:i for i,nm in enumerate(dnames)}
    comp=np.full((len(dnames),3),1/3)                               # 默认均匀
    cov=0
    for dc,sub in df.groupby("dc"):
        nm=CODE2NAME.get(dc); gi=name2gi.get(nm) if nm else None
        if gi is None:
            for k,v in name2gi.items():
                if nm and (nm in k or k in nm): gi=v; break
        if gi is not None:
            sh=sub["et"].value_counts(normalize=True); comp[gi]=[sh.get(0,0),sh.get(1,0),sh.get(2,0)]; cov+=1
    print(f"  区映射覆盖 {cov}/{len(dnames)}; 海淀教育构成={comp[name2gi.get('海淀',0)].round(2)} "
          f"(高教育区应偏高档)")
    edu_props=comp[didx]                                            # (Ncell,3)

    # edges + 混合模型
    e=np.load(PROC/"beijing_edges.npz",allow_pickle=True)
    o=e["o_idx"].astype(np.int64); d=e["d_idx"].astype(np.int64); t=e["t_ff"].astype(np.float32)
    flow=e["flow"].sum(1).astype(np.float32); seg_ptr=e["seg_ptr"].astype(np.int64)
    jobs=g["jobs"].astype(np.float32); logM=np.log(jobs+1.0)
    rng=np.random.default_rng(0); nseg=len(seg_ptr)-1
    pick=np.sort(rng.choice(nseg,size=2500,replace=False))
    rows=np.concatenate([np.arange(seg_ptr[i],seg_ptr[i+1]) for i in pick])
    o_s=o[rows]; lM=torch.tensor(logM[d[rows]]); tt=torch.tensor(t[rows]); flt=torch.tensor(flow[rows])
    seg_len=seg_ptr[pick+1]-seg_ptr[pick]; nS=len(pick); segid=torch.tensor(np.repeat(np.arange(nS),seg_len))
    w=torch.tensor(edu_props[o_s]); ancht=torch.tensor(anch,dtype=torch.float32)
    print(f"  子样本 {nS} 起点 {len(rows):,} 边",flush=True)

    def seg_sm(V):
        mx=torch.full((nS,),-1e30).scatter_reduce(0,segid,V,"amax",include_self=False)
        z=(V-mx[segid]).exp(); s=torch.zeros(nS).index_add(0,segid,z); return z/s[segid]
    def run(anchor, steps=400):
        a=torch.tensor(0.5,requires_grad=True); bk=torch.tensor([0.13,0.12,0.10],requires_grad=True)
        opt=torch.optim.Adam([a,bk],lr=0.03)
        for _ in range(steps):
            opt.zero_grad(); Ps=[seg_sm(a*lM-bk[k]*tt) for k in range(3)]
            P=sum(w[:,k]*Ps[k] for k in range(3))
            nll=-(flt*torch.log(P.clamp_min(1e-12))).sum()/flt.sum()
            loss=nll
            if anchor:  # 各档预测通勤 -> 匹配锚
                ctk=torch.stack([ (w[:,k]*Ps[k]*tt).sum()/((w[:,k]*Ps[k]).sum()+1e-9) for k in range(3)])
                loss=nll+0.01*((ctk-ancht)**2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            ctk=[float((w[:,k]*seg_sm(a*lM-bk[k]*tt)*tt).sum()/((w[:,k]*seg_sm(a*lM-bk[k]*tt)).sum()+1e-9)) for k in range(3)]
        return [round(float(x),3) for x in bk], float(nll), [round(x,1) for x in ctk]

    print("\n=== ① 自由 β(教育, 无锚) ===",flush=True)
    bf,nf,cf=run(False); mono=bf[0]>=bf[1]>=bf[2]
    print(f"  β[低/中/高]={bf}  NLL={nf:.5f}  预测各档通勤{cf}  单调?{'是' if mono else '否'}",flush=True)
    print("\n=== ② 教育 β + 通勤锚 ===",flush=True)
    ba,na,ca=run(True); monoa=ba[0]>=ba[1]>=ba[2]
    print(f"  β[低/中/高]={ba}  NLL={na:.5f}  预测各档通勤{ca}(锚{anch.round(1)})  单调?{'是' if monoa else '否'}",flush=True)
    print(f"\n  ΔNLL(锚−自由)={na-nf:+.5f}; β spread(低−高)= 自由{bf[0]-bf[2]:.3f} / 锚{ba[0]-ba[2]:.3f}")
    print("判读: 锚后 β 若【单调下降且 spread 明显】(高教育怕远低=通勤长) + 预测通勤对上锚 -> "
          "教育异质【识别得出/可锚】, 区别于收入塌平。则建议 Paper A 用教育替收入。")


if __name__=="__main__":
    main()
