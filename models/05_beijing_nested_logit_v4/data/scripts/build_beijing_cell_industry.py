"""task13 finer-geo step1: 街道->格子 -> cell级行业构成 (16907×19)

输入:
  beijing_grid.npz (cells: xy_m, district_idx, district_names)
  beijing_street_industry_2008.npz (319街道×19行业, 街道名+didx)
  D:/UoM/PaperC_local/bj_streets/*.json (高德当前街道中心点, 16区)
出: data/processed/beijing_cell_industry.npz (gitignored)

法: 每格 -> 同区最近高德街道; 高德街道按名字配2008街道行业; 配不上回退区级2008行业。
得 cell_industry_prop(16907,19)。⚠ 2008行业格局(中关村/金融街准, 缺post-2008新区)。
下一步: 2023网格职住rescale量 + AOI校准。
"""
import sys, json, glob, re
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
ST = PROC / "beijing_street_industry_2008.npz"
GAODE = Path(r"D:/UoM/PaperC_local/bj_streets")
OUT = PROC / "beijing_cell_industry.npz"

# 高德当前 adcode(6位) -> 区名 (注: 高德用 110118密云区/110119延庆区 新码)
ADC2NAME = {"110101":"东城","110102":"西城","110105":"朝阳","110106":"丰台","110107":"石景山",
            "110108":"海淀","110109":"门头沟","110111":"房山","110112":"通州","110113":"顺义",
            "110114":"昌平","110115":"大兴","110116":"怀柔","110117":"平谷","110118":"密云","110119":"延庆"}

def clean(s):
    s = str(s)
    return re.sub(r"(街道办事处|地区办事处|街道|地区|镇人民政府|乡人民政府|镇|乡|办事处)$", "", s)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = np.load(GRID, allow_pickle=True)
    dn = [str(x) for x in g["district_names"]]; name2idx = {n:i for i,n in enumerate(dn)}
    coords = g["coords_latlon"].astype(np.float64); didx = g["district_idx"]; N = len(didx)
    lat0, lon0 = float(np.median(coords[:,0])), float(np.median(coords[:,1]))
    def proj(lat, lon):
        return np.stack([(lon-lon0)*np.cos(np.deg2rad(lat0))*111320.0, (lat-lat0)*110540.0], axis=1)
    cell_xy = proj(coords[:,0], coords[:,1])

    st = np.load(ST, allow_pickle=True)
    st_didx = st["street_didx"]; st_name = [clean(x) for x in st["street_name"]]
    st_ind = st["street_industry_emp"]  # (319,19)
    # 2008 街道行业 lookup: (didx, cleanname) -> 19vec
    st_lk = {}
    for i in range(len(st_didx)):
        st_lk[(int(st_didx[i]), st_name[i])] = st_ind[i]
    # 区级2008行业(回退)
    dist_ind = np.array([st_ind[st_didx==k].sum(0) for k in range(16)])  # (16,19)

    # 高德街道
    gd = []  # (didx, cleanname, x, y)
    for f in glob.glob(str(GAODE/"*.json")):
        j = json.load(open(f, encoding="utf-8"))
        for d in j.get("districts", []):
            adc = d.get("adcode"); dname = ADC2NAME.get(adc)
            if dname not in name2idx: continue
            di = name2idx[dname]
            for s in d.get("districts", []):
                if s.get("level") != "street": continue
                c = s.get("center","").split(",")
                if len(c)!=2: continue
                lng, lat = float(c[0]), float(c[1])
                gd.append((di, clean(s["name"]), lng, lat))
    print(f"高德街道 {len(gd)} 个")
    gd_xy = proj(np.array([x[3] for x in gd]), np.array([x[2] for x in gd]))
    gd_didx = np.array([x[0] for x in gd]); gd_name = [x[1] for x in gd]
    # 每高德街道 -> 2008行业(配名), 配不上回退区级
    gd_ind = np.zeros((len(gd),19)); nmatch=0
    for i in range(len(gd)):
        key=(gd_didx[i], gd_name[i])
        if key in st_lk: gd_ind[i]=st_lk[key]; nmatch+=1
        else: gd_ind[i]=dist_ind[gd_didx[i]]
    print(f"高德街道配上2008 {nmatch}/{len(gd)}")

    # 每格 -> 同区最近高德街道
    from scipy.spatial import cKDTree
    cell_ind = np.zeros((N,19))
    for k in range(16):
        cm = didx==k; gm = gd_didx==k
        if gm.sum()==0:
            cell_ind[cm]=dist_ind[k]; continue
        tree=cKDTree(gd_xy[gm]); _,nn=tree.query(cell_xy[cm],k=1)
        gidx=np.where(gm)[0][nn]
        cell_ind[cm]=gd_ind[gidx]
    cell_prop = cell_ind/cell_ind.sum(1,keepdims=True).clip(1e-9)

    np.savez_compressed(OUT, cell_industry_prop=cell_prop.astype(np.float32),
                        ind_letters=st["ind_letters"])
    print(f"[OK] {OUT}  cell_industry_prop {cell_prop.shape}")
    # 验证: 海淀中关村一带的格子 G(信息)占比 vs 全市
    IL=[str(x) for x in st["ind_letters"]]; gi=IL.index("G")
    print(f"  G(信息)占比: 全市cell均值 {cell_prop[:,gi].mean():.3f}, "
          f"海淀cell均值 {cell_prop[didx==name2idx['海淀'],gi].mean():.3f}")
    # 唯一行数 (>16 说明真到街道级了)
    print(f"  cell行业构成唯一行数 {len(np.unique(cell_prop.round(5),axis=0))} (>16=突破区级广播)")

if __name__ == "__main__":
    main()
