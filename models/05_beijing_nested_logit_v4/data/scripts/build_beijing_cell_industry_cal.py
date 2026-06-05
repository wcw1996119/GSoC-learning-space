"""task13 ②: 用 2023 AOI 校准 cell 行业构成 (补 2008 缺的 post-2008 集聚: 望京/亦庄/未科)

输入: beijing_cell_industry.npz (2008街道, 已修对齐) + 202302北京AOI (2023, 类型)
出: beijing_cell_industry.npz (覆盖, 加 2023 校准)

法: AOI 按功能分组 -> 19行业(规范idx); 分配到最近格子 -> cell AOI 行业强度;
    blend: 有AOI岗位信号的格子 把 2008构成 往 AOI(2023) 拉。
⚠ AOI 49%住宅/风景, 岗位类(写字楼/产业园)稀疏 -> 校准影响预期有限, 如实报。
"""
import sys
from pathlib import Path
import numpy as np, pyogrio

PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
CELL = PROC / "beijing_cell_industry.npz"
AOI = Path(r"D:/UoM/congestion_project/202302北京AOI信息/北京AOI_WGS84.shp")
BLEND = 0.5  # AOI信号格子里 2023占比上限

# AOI 小类/中类/大类 关键词 -> 19规范行业idx (5批零6交通7住宿8信息9金融10房产11商务12科研15教育16卫生18公管, 2制造)
def aoi_to_ind(da, zh, xiao):
    s = f"{da}|{zh}|{xiao}"
    if any(k in s for k in ["写字楼","公司","楼宇","公司企业"]): return [8,9,11,12]   # 办公->信息/金融/商务/科研
    if "产业园" in s: return [2,8]            # 产业园->制造/信息
    if any(k in s for k in ["市场","购物","建材","家居"]): return [5]   # 批零
    if "医" in s: return [16]                  # 卫生
    if any(k in s for k in ["学校","高等院校","科研机构","幼儿园","小学","中学"]): return [15,12]  # 教育/科研
    if any(k in s for k in ["宾馆","酒店","餐饮"]): return [7]  # 住宿餐饮
    if any(k in s for k in ["政府","社会团体"]): return [18]    # 公管
    return None  # 住宅/风景名胜等 -> 非岗位, skip

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = np.load(GRID, allow_pickle=True)
    coords = g["coords_latlon"].astype(np.float64); didx = g["district_idx"]; N = len(didx)
    dn = [str(x) for x in g["district_names"]]
    lat0, lon0 = float(np.median(coords[:,0])), float(np.median(coords[:,1]))
    def proj(lat, lon): return np.stack([(lon-lon0)*np.cos(np.deg2rad(lat0))*111320,(lat-lat0)*110540],1)
    cell_xy = proj(coords[:,0], coords[:,1])
    cell_2008 = np.load(CELL)["cell_industry_prop"].astype(np.float64)

    df = pyogrio.read_dataframe(str(AOI), read_geometry=False)
    cols = list(df.columns); da,zh,xiao = cols[-3],cols[-2],cols[-1]
    lng = df["wgs_lng"].astype(float).values; lat = df["wgs_lat"].astype(float).values
    from scipy.spatial import cKDTree
    tree = cKDTree(cell_xy)
    aoi_xy = proj(lat, lng); _, ac = tree.query(aoi_xy, k=1)
    aoi_ind = np.zeros((N,19)); njob=0
    for i in range(len(df)):
        idxs = aoi_to_ind(str(df[da].iloc[i]), str(df[zh].iloc[i]), str(df[xiao].iloc[i]))
        if idxs is None: continue
        njob+=1
        for j in idxs: aoi_ind[ac[i], j] += 1.0/len(idxs)
    cells_with_aoi = (aoi_ind.sum(1)>0).sum()
    print(f"AOI {len(df)} 个, 岗位类 {njob} 个, 落在 {cells_with_aoi} 格")

    # blend: 有AOI岗位信号的格子, w = BLEND*min(1, aoi_count/3)
    aoi_tot = aoi_ind.sum(1, keepdims=True)
    aoi_prop = np.where(aoi_tot>0, aoi_ind/aoi_tot.clip(1e-9), 0)
    w = (BLEND*np.minimum(1.0, aoi_tot[:,0]/3.0)).reshape(-1,1)
    cell_cal = (1-w)*cell_2008 + w*aoi_prop
    cell_cal = cell_cal/cell_cal.sum(1,keepdims=True).clip(1e-9)

    np.savez_compressed(CELL, cell_industry_prop=cell_cal.astype(np.float32),
                        ind_letters=np.load(CELL)["ind_letters"])
    print(f"[OK] {CELL} (2023 AOI 校准)")
    # 影响: 信息业(8) 在 望京(朝阳)/亦庄(大兴) 校准前后
    def darea(nm): return didx==dn.index(nm)
    for nm in ["朝阳","大兴","海淀"]:
        m=darea(nm)
        print(f"  {nm}: 信息I 2008={cell_2008[m,8].mean():.3f}->cal={cell_cal[m,8].mean():.3f}; "
              f"制造C 2008={cell_2008[m,2].mean():.3f}->cal={cell_cal[m,2].mean():.3f}")
    print(f"  全市变动幅度(L1) mean {np.abs(cell_cal-cell_2008).sum(1).mean():.3f} (小=AOI影响有限)")

if __name__ == "__main__":
    main()
