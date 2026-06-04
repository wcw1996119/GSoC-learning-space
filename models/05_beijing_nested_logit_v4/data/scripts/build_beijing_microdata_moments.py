"""task13: 2015 1% 微观 -> district-level moment (职业×居住区×工作区 OD + 个体mode + 通勤时间)

源: D:\\UoM\\congestion_project\\...census2015_origin CSV格式\\*.csv (137万人, 北京21833)
依赖: beijing_grid.npz (district_names 对齐 16 区, 算区中心算距离)
出: data/processed/beijing_micro_moments.npz (gitignored)

变量: M2居住省地县码 / M58职业(首位=GB6565大类1-8) / M59工作地点(1本街道2本市其他3外市)
      / M60工作区码(M59=2时) / M61交通工具 / M62通勤时间
产出:
  occ_wp_od (7,16,16)   各职业 居住区->工作区 计数 (δ_match 识别)
  mode_share_overall(3) 真实 [车,公交,步] 分担; mode_by_dist (nbands,3)
  commute_time_dist     通勤时间分布 (T_max)
  + 诊断: 控制居住区后 职业->工作区 信号 (corr)
⚠ 2015 vs 模型2023; 区级(16); 每记录约代表64人但算share/分布等权即可。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

CSV = Path(r"D:/UoM/congestion_project/2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式/2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式.csv")
PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
OUT = PROC / "beijing_micro_moments.npz"

# 2015 北京区码 -> 区名 (110228密云县/110229延庆县 老码)
CODE2NAME = {"110101":"东城","110102":"西城","110105":"朝阳","110106":"丰台",
             "110107":"石景山","110108":"海淀","110109":"门头沟","110111":"房山",
             "110112":"通州","110113":"顺义","110114":"昌平","110115":"大兴",
             "110116":"怀柔","110117":"平谷","110228":"密云","110229":"延庆"}
# M61 -> 3 mode (1走2自行车3电动->步; 4摩托5私车->车; 6公交7地铁->公交; 8其他->drop)
MODE3 = {"1":0,"2":0,"3":0,"4":1,"5":1,"6":2,"7":2}  # 0=walk(慢行) wait 重排下
# 统一顺序 [车,公交,步] = [car,transit,walk] 跟模型一致
M61_TO = {"4":"car","5":"car","6":"transit","7":"transit","1":"walk","2":"walk","3":"walk"}

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = np.load(GRID, allow_pickle=True)
    dn = [str(x) for x in g["district_names"]]
    name2idx = {n: i for i, n in enumerate(dn)}
    code2idx = {c: name2idx[n] for c, n in CODE2NAME.items() if n in name2idx}
    assert len(code2idx) == 16, f"区映射缺: {set(CODE2NAME.values())-set(dn)}"
    # 区中心 (算距离用)
    xy = g["xy_m"].astype(np.float64); didx = g["district_idx"]
    cent = np.stack([xy[didx == k].mean(0) for k in range(16)])  # (16,2) m

    d = pd.read_csv(CSV, usecols=["M2","M58","M59","M60","M61","M62"], dtype=str)
    bj = d[d["M2"].str.startswith("11")].copy()
    emp = bj[bj["M59"].notna()].copy()
    # 居住/工作区 idx
    emp["rc"] = emp["M2"].str[:6]
    emp["wc"] = np.where(emp["M59"] == "2", emp["M60"].str[:6], emp["M2"].str[:6])
    emp["ri"] = emp["rc"].map(code2idx); emp["wi"] = emp["wc"].map(code2idx)
    emp = emp[emp["ri"].notna() & emp["wi"].notna()].copy()   # 留市内(M59=3外市/无效区码 drop)
    emp["ri"] = emp["ri"].astype(int); emp["wi"] = emp["wi"].astype(int)
    # 职业 (M58 首位 1-7 -> occ 0-6; 8 drop)
    emp["occ"] = pd.to_numeric(emp["M58"].str[0], errors="coerce")
    emp = emp[emp["occ"].between(1, 7)].copy(); emp["occ"] = emp["occ"].astype(int) - 1
    # mode
    emp["mode"] = emp["M61"].map(M61_TO)
    # 通勤时间
    emp["ct"] = pd.to_numeric(emp["M62"], errors="coerce")
    n = len(emp)

    # ---- occ × 居住区 × 工作区 OD ----
    occ_wp_od = np.zeros((7, 16, 16))
    for o, r, w in zip(emp["occ"], emp["ri"], emp["wi"]):
        occ_wp_od[o, r, w] += 1

    # ---- 真实 mode 分担 ----
    md = emp[emp["mode"].notna()]
    order = ["car","transit","walk"]
    mode_overall = np.array([(md["mode"] == m).mean() for m in order])
    # 按距离档 (区中心距)
    md = md.copy()
    md["km"] = [np.linalg.norm(cent[r]-cent[w])/1000 for r, w in zip(md["ri"], md["wi"])]
    bands = [(0,1),(1,5),(5,15),(15,100)]  # 同区~0
    mode_by_dist = np.zeros((len(bands), 3))
    for bi,(lo,hi) in enumerate(bands):
        sub = md[(md["km"]>=lo)&(md["km"]<hi)]
        if len(sub): mode_by_dist[bi] = [ (sub["mode"]==m).mean() for m in order]

    # ---- 通勤时间分布 ----
    ct = emp["ct"].dropna(); ct = ct[(ct>0)&(ct<240)]
    ct_bins = [0,15,30,60,90,120,180,9999]
    ct_hist = np.histogram(ct, bins=ct_bins)[0] / len(ct)

    np.savez_compressed(OUT,
        occ_wp_od=occ_wp_od.astype(np.float32),
        mode_overall=mode_overall.astype(np.float32),
        mode_by_dist=mode_by_dist.astype(np.float32),
        dist_bands=np.array([b[1] for b in bands]),
        ct_hist=ct_hist.astype(np.float32), ct_mean=np.float32(ct.mean()),
        district_centroids=cent.astype(np.float32))

    # ---- 诊断 ----
    print(f"[OK] {OUT}  北京就业市内 {n}")
    print(f"  真实 mode [车,公交,步] = {mode_overall.round(3).tolist()}")
    print(f"  通勤时间 mean {ct.mean():.1f}min; 分布(0-15/15-30/30-60/60-90/90-120/120-180/180+) = {ct_hist.round(3).tolist()}")
    print(f"  各职业样本: {np.bincount(emp['occ'],minlength=7).tolist()}")
    # ⭐ 控制居住区后 职业->工作区 信号: 大区内 各职业工作区分布相关
    import itertools
    within_cors=[]
    for r in range(16):
        sub=emp[emp["ri"]==r]
        if len(sub)<300: continue
        piv=pd.crosstab(sub["wi"], sub["occ"], normalize="columns")
        big=[o for o in piv.columns if (sub["occ"]==o).sum()>=40]
        if len(big)<3: continue
        cs=[np.corrcoef(piv[a],piv[b])[0,1] for a,b in itertools.combinations(big,2)]
        within_cors.append(np.nanmean(cs))
    print(f"  ⭐控制居住区后 职业-工作区分布相关 mean {np.nanmean(within_cors):.3f} "
          f"({len(within_cors)}个大区) (低=职业真预测去向, δ_match可识别)")
    # 整体(不控居住)
    piv=pd.crosstab(emp["wi"], emp["occ"], normalize="columns")
    cs=[np.corrcoef(piv[a],piv[b])[0,1] for a,b in itertools.combinations(list(piv.columns),2)]
    print(f"  (不控居住 整体相关 mean {np.nanmean(cs):.3f})")

if __name__ == "__main__":
    main()
