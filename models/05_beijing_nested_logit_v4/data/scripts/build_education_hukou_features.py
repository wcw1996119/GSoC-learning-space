"""Paper A 重训第1步: 建【教育档构成 + 户口构成 + 锚】特征 (替收入 + 加户口控制)。

从 2015 1% 普查(北京就业通勤者):
  - 教育3档(M51): 区级构成广播到 cell -> education_tier_props (N,3); 各档通勤锚 ct_by_edu (3,)
  - 户口(M39京外=外地): 区级外地占比广播 -> hukou_migshare (N,); 各档通勤锚 ct_by_hukou (2,)
存 beijing_edu_hukou.npz, 给改后的 trainer 用。governance: 数据不出库不commit。
"""
from pathlib import Path
import numpy as np, pandas as pd
PROC=Path(__file__).resolve().parents[1]/"processed"
CSV=Path(r"D:\UoM\congestion_project\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式"
         r"\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式.csv")
CODE2NAME={"110101":"东城","110102":"西城","110105":"朝阳","110106":"丰台","110107":"石景山",
 "110108":"海淀","110109":"门头沟","110111":"房山","110112":"通州","110113":"顺义",
 "110114":"昌平","110115":"大兴","110116":"怀柔","110117":"平谷","110118":"密云","110119":"延庆"}


def main():
    n=lambda s:pd.to_numeric(s,errors="coerce")
    df=pd.read_csv(CSV,dtype=str,usecols=lambda c:c in ["M2","M39","M51","M59","M62"],low_memory=False)
    df=df[df["M2"].astype(str).str.startswith("11")].copy()
    df=df[n(df["M59"]).isin([1,2])&(n(df["M62"])>0)&(n(df["M62"])<240)].copy()
    df["ct"]=n(df["M62"]); df["dc"]=df["M2"].astype(str).str[:6]
    df["et"]=np.where(n(df["M51"])<=3,0,np.where(n(df["M51"])<=5,1,2))     # 教育3档
    prov=df["M39"].fillna("").str[:2]; df["mig"]=(prov.str.isdigit()&(prov!="11")).astype(int)

    ct_edu=df.groupby("et")["ct"].mean().values
    ct_hukou=df.groupby("mig")["ct"].mean().values
    print(f"  教育档通勤锚 {ct_edu.round(1)}; 户口通勤锚[本地/外地] {ct_hukou.round(1)}; 外地占比 {df['mig'].mean()*100:.0f}%")

    g=np.load(PROC/"beijing_grid.npz",allow_pickle=True)
    dnames=[str(x) for x in g["district_names"]]; didx=g["district_idx"].astype(np.int64)
    name2gi={nm:i for i,nm in enumerate(dnames)}
    def gi_of(dc):
        nm=CODE2NAME.get(dc)
        if not nm: return None
        if nm in name2gi: return name2gi[nm]
        for k,v in name2gi.items():
            if nm in k or k in nm: return v
        return None
    edu_comp=np.full((len(dnames),3),1/3); mig_sh=np.full(len(dnames),df["mig"].mean()); cov=0
    for dc,sub in df.groupby("dc"):
        gi=gi_of(dc)
        if gi is not None:
            sh=sub["et"].value_counts(normalize=True); edu_comp[gi]=[sh.get(0,0),sh.get(1,0),sh.get(2,0)]
            mig_sh[gi]=sub["mig"].mean(); cov+=1
    print(f"  区映射覆盖 {cov}/{len(dnames)}; 海淀教育{edu_comp[name2gi.get('海淀',0)].round(2)} 朝阳外地{mig_sh[name2gi.get('朝阳',0)]:.2f}")

    N=len(didx)
    out=dict(education_tier_props=edu_comp[didx].astype(np.float32),       # (N,3)
             hukou_migshare=mig_sh[didx].astype(np.float32),               # (N,)
             ct_by_edu=ct_edu.astype(np.float32),                          # (3,) 教育通勤锚
             ct_by_hukou=ct_hukou.astype(np.float32),                      # (2,) 户口通勤锚
             district_edu_comp=edu_comp.astype(np.float32),
             district_migshare=mig_sh.astype(np.float32))
    np.savez(PROC/"beijing_edu_hukou.npz",**out)
    print(f"  [OK] 存 {PROC/'beijing_edu_hukou.npz'}  (N={N})")
    print("  下一步: 改 trainer 用 education_tier_props 替 income_tier_props + hukou_migshare 当控制 + 两个锚。")


if __name__=="__main__":
    main()
