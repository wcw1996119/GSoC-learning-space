"""task13(finer geo) 底座: 2008经普 -> 街道级 行业就业 (737街道 × 19门类)

源: D:\\UoM\\congestion_project\\2008经济普查\\08经济普查-北京市.dta (24万企业)
出: data/processed/beijing_street_industry_2008.npz (gitignored)

行政区划代码 12位: [:6]=区县码 [:9]=街道码 [10:]=居委会
行业代码 4位: [:2]=中类 -> GB门类 A-S -> 19行业idx
⚠ 2008数据: 街道空间格局含中关村/金融街等(pre-2008稳), 缺亦庄/望京/未科(post-2008)。
   下一步需 街道->格子 geocode + 2023(网格职住/AOI)校准 currency。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, re

DTA = Path(r"D:/UoM/congestion_project/2008经济普查/08经济普查-北京市.dta")
PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
OUT = PROC / "beijing_street_industry_2008.npz"

# 2008 区县码 -> 区名 (含已撤并: 110103崇文->东城, 110104宣武->西城)
CODE2NAME = {"110101":"东城","110102":"西城","110103":"东城","110104":"西城","110105":"朝阳",
             "110106":"丰台","110107":"石景山","110108":"海淀","110109":"门头沟","110111":"房山",
             "110112":"通州","110113":"顺义","110114":"昌平","110115":"大兴","110116":"怀柔",
             "110117":"平谷","110228":"密云","110229":"延庆"}
IND_LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S']

def cz2letter(cz):
    # GB/T 4754-2002 (2008经普用此版), 区间经业务文字反推校准:
    #   60电信/62软件=G, 63批发/65零售=H, 66住宿/67餐饮=I, 72房地产=K, 73-74租赁商务=L, 75科研=M, 84教育=P
    try: c = int(cz)
    except: return None
    R=[(1,5,'A'),(6,12,'B'),(13,43,'C'),(44,46,'D'),(47,50,'E'),(51,59,'F'),(60,62,'G'),
       (63,65,'H'),(66,67,'I'),(68,71,'J'),(72,72,'K'),(73,74,'L'),(75,78,'M'),(79,81,'N'),
       (82,83,'O'),(84,84,'P'),(85,87,'Q'),(88,92,'R'),(93,97,'S')]
    for a,b,L in R:
        if a<=c<=b: return L
    return None

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = np.load(GRID, allow_pickle=True)
    dn = [str(x) for x in g["district_names"]]; name2idx = {n:i for i,n in enumerate(dn)}
    let2idx = {L:i for i,L in enumerate(IND_LETTERS)}

    d = pd.read_stata(DTA, columns=["国民经济行业代码","行政区划代码","街道办事处","年末从业人员合计_总计"])
    d["adm"] = d["行政区划代码"].astype(str).str.replace(r"\.0$","",regex=True)
    d = d[d["adm"].str.len() >= 9].copy()
    d["dcode"] = d["adm"].str[:6]
    d["street"] = d["adm"].str[:9]
    d["didx"] = d["dcode"].map(lambda c: name2idx.get(CODE2NAME.get(c)))
    d = d[d["didx"].notna()].copy(); d["didx"] = d["didx"].astype(int)
    d["indL"] = d["国民经济行业代码"].astype(str).str.replace(r"\.0$","",regex=True).str[:2].map(cz2letter)
    d["iidx"] = d["indL"].map(let2idx)
    d = d[d["iidx"].notna()].copy(); d["iidx"] = d["iidx"].astype(int)
    d["emp"] = pd.to_numeric(d["年末从业人员合计_总计"], errors="coerce").fillna(0).clip(lower=0)

    streets = sorted(d["street"].unique())
    s2i = {s:i for i,s in enumerate(streets)}; S = len(streets)
    M = np.zeros((S, 19))
    for st, ii, e in zip(d["street"], d["iidx"], d["emp"]):
        M[s2i[st], ii] += e
    street_didx = np.array([int(d[d["street"]==s]["didx"].iloc[0]) for s in streets])
    # 街道名 (geocode 用)
    street_name = np.array([str(d[d["street"]==s]["街道办事处"].iloc[0]) for s in streets], dtype="U40")

    np.savez_compressed(OUT,
        street_code=np.array(streets, dtype="U12"),
        street_didx=street_didx, street_name=street_name,
        street_industry_emp=M.astype(np.float32), ind_letters=np.array(IND_LETTERS, dtype="U2"))

    print(f"[OK] {OUT}")
    print(f"  街道 {S}, 总从业 {M.sum():,.0f}")
    print(f"  各区街道数: {np.bincount(street_didx, minlength=16).tolist()}")
    # 合理性: 中关村街道 行业 top3 (应 信息/科研/制造)
    for kw in ["中关村","金融街","朝阳门"]:
        idx=[i for i,n in enumerate(street_name) if kw in n]
        for i in idx[:1]:
            v=M[i]; top=np.argsort(-v)[:3]
            print(f"  {street_name[i]}({IND_LETTERS}): top行业 "+", ".join(f'{IND_LETTERS[t]}{v[t]/v.sum()*100:.0f}%' for t in top))
    # 区级聚合 vs 街道内方差 (有没有 sub-district 信号)
    print(f"  街道行业构成 vs 区级: 同区内街道行业占比的标准差 mean "
          f"{np.mean([np.std((M[street_didx==k]/M[street_didx==k].sum(1,keepdims=True).clip(1)),0).mean() for k in range(16) if (street_didx==k).sum()>3]):.3f} (>0=有街道级变化)")

if __name__ == "__main__":
    main()
