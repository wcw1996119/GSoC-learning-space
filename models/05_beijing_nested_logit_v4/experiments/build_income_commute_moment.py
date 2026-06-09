"""从 2015 1% 普查凑出【按收入分的通勤时间矩】(pin 收入异质量级的外部锚)。

北京就业通勤者: 收入代理 = 住房面积(M9) + 拥有住房数(M16) + 教育(M51) z-score 合成 → 三档。
各档平均通勤时间(M62) = Paper A 收入异质的 Ben-Akiva-Morikawa 锚。
诚实: 代理非真收入(普查不问收入), 但住房/教育是标准 wealth proxy; 跟模型的房价收入档同源(都靠住房)。
"""
from pathlib import Path
import numpy as np, pandas as pd

CSV = Path(r"D:\UoM\congestion_project\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式"
           r"\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式.csv")
def num(s): return pd.to_numeric(s, errors="coerce")


def main():
    cols=["M2","M9","M16","M35","M51","M58","M59","M62"]
    df=pd.read_csv(CSV,dtype=str,usecols=lambda c:c in cols,low_memory=False)
    df=df[df["M2"].astype(str).str.startswith("11")].copy()             # 北京
    df["M59n"]=num(df["M59"]); df["M62n"]=num(df["M62"])
    df=df[df["M59n"].isin([1,2])&(df["M62n"]>0)&(df["M62n"]<240)].copy() # 就业通勤者
    df["age"]=2015-num(df["M35"]); df=df[(df["age"]>=16)&(df["age"]<=70)]
    area=num(df["M9"]); homes=num(df["M16"]); edu=num(df["M51"])
    print(f"北京就业通勤者 {len(df)}; 住房面积有效 {area.notna().mean()*100:.0f}%, "
          f"拥房数 {homes.notna().mean()*100:.0f}%, 教育 {edu.notna().mean()*100:.0f}%")
    print(f"  M9面积 中位{area.median():.0f}  M16拥房 分布{homes.value_counts().sort_index().head(5).to_dict()}  "
          f"M51教育 中位{edu.median():.0f}")

    def z(x): m=x.median(); s=x.std(); return ((x.fillna(m)-x.mean())/(s+1e-9))
    score = z(area)+z(homes)+z(edu)                                    # 合成收入代理(越高越富)
    df["score"]=score
    q=score.quantile([1/3,2/3])
    tier=np.where(score<=q.iloc[0],0,np.where(score<=q.iloc[1],1,2))   # 0低 1中 2高
    df["tier"]=tier

    print("\n=== ⭐ 按收入代理分档的通勤时间(Paper A 收入锚) ===")
    g=df.groupby("tier")["M62n"].agg(["mean","median","size"])
    names=["低收入","中收入","高收入"]
    for t in range(3):
        print(f"  {names[t]}: 平均通勤 {g.loc[t,'mean']:.1f}min  中位 {g.loc[t,'median']:.0f}  (n={int(g.loc[t,'size'])})")
    grad=g.loc[2,"mean"]-g.loc[0,"mean"]
    print(f"\n  梯度(高−低) = {grad:+.1f}min  总体 {df['M62n'].mean():.1f}min")
    # 单代理稳健性
    print("\n  稳健性(各单代理 高档−低档 通勤差):")
    for nm,x in [("面积",area),("拥房数",homes),("教育",edu)]:
        qq=x.quantile([1/3,2/3]); tt=np.where(x<=qq.iloc[0],0,np.where(x<=qq.iloc[1],1,2))
        gg=df.groupby(tt)["M62n"].mean()
        if 0 in gg and 2 in gg: print(f"    {nm}: {gg[2]-gg[0]:+.1f}min")
    out={"ct_by_income_mean":g["mean"].values.astype(float),
         "ct_by_income_median":g["median"].values.astype(float),
         "n_by_income":g["size"].values.astype(int)}
    np.savez(Path(__file__).resolve().parents[1]/"data"/"processed"/"beijing_income_commute_moment.npz",**out)
    print(f"\n  [OK] 存 data/processed/beijing_income_commute_moment.npz")
    print("判读: 梯度>0 = 高收入通勤更长(像 CFPS/伦敦); <0 反向。这个矩接进 income_monotonic 当锚 -> pin 收入量级。")


if __name__=="__main__":
    main()
