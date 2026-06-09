"""可达性不平等 (Hansen, 含职业偏好) —— 1a 搬岗位前后, 各职业/收入群体可达性怎么变。

可达性(结构量, 不碰NN -> 可外推):
  A_i^职业 = Σ_j (岗位_j × demand_share[j,职业]) × exp(-0.0947·t_ij)   # 职业偏好=你这行的岗在哪
  A_i^总   = Σ_j 岗位_j × exp(-0.0947·t_ij)                            # 收入用总岗位(差异在住哪+忍受度)
群体体验 = 该群体居民所在 origin 的可达性, 按"群体住哪"(soc_props/income_tier_props×residents)加权。
不平等: 群体内 CV(住得散→机会差异大) + 群体间 ratio(谁高谁低)。
1a: 通州+30万(按现岗位比例) / 核心区(东城+西城)×0.8。travel time 固定(partial-eq, 拥堵归B模型)。
"""
import argparse
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
THETA = 0.0947           # Hansen 衰减 /min (项目DCM标定)
TONGZHOU = 13; CORE = [0, 12]    # 通州 / 东城+西城
OCC = ["负责人", "专技", "办事", "商服", "农林", "生产", "其他"]   # GB7大类(待用户校准)


def wavg(x, w): return float((x * w).sum() / max(w.sum(), 1e-9))
def wcv(x, w):
    m = wavg(x, w); v = wavg((x - m) ** 2, w); return float(v ** 0.5 / max(abs(m), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-tongzhou", type=float, default=300000.0)
    ap.add_argument("--core-cut", type=float, default=0.2)
    a = ap.parse_args()

    g = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    jobs = g["jobs"].astype(np.float64); residents = g["residents"].astype(np.float64)
    district = g["district_idx"]; N = len(jobs)
    e = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    o = e["o_idx"].astype(np.int64); d = e["d_idx"].astype(np.int64); t = e["t_obs"].astype(np.float64)
    occ = np.load(PROC / "beijing_occupation.npz", allow_pickle=True)
    dshare = occ["demand_share"].astype(np.float64)      # (N,7) 职业岗位需求面(哪有这行的岗)
    soc = occ["soc_props"].astype(np.float64)             # (N,7) 居民职业构成(谁住哪)
    inc = np.load(PROC / "beijing_income.npz", allow_pickle=True)["income_tier_props"].astype(np.float64)  # (N,3)

    decay = np.exp(-THETA * t)

    def hansen(jb):
        A = np.zeros((N, 7))
        for k in range(7):
            Mk = jb * dshare[:, k]
            A[:, k] = np.bincount(o, weights=Mk[d] * decay, minlength=N)
        Atot = np.bincount(o, weights=jb[d] * decay, minlength=N)
        return A, Atot

    A0, T0 = hansen(jobs)
    jobs1 = jobs.copy()
    tc = np.where(district == TONGZHOU)[0]; cc = np.where(np.isin(district, CORE))[0]
    w = jobs[tc] + 1.0; jobs1[tc] += a.add_tongzhou * w / w.sum()
    jobs1[cc] *= (1 - a.core_cut)
    A1, T1 = hansen(jobs1)
    print(f"1a: 通州{len(tc)}格 +{a.add_tongzhou:.0f}岗位; 核心(东城+西城){len(cc)}格 ×{1-a.core_cut:.1f}\n")

    # ---- 职业可达性不平等 ----
    print("=== 各职业可达性 (该职业居民体验到的'本行岗位可达性') ===")
    print(f"{'职业':<6} {'前':>10} {'后':>10} {'变化%':>7} {'群体内CV前→后':>14}")
    means0, means1, names = [], [], []
    for k in range(7):
        wk = soc[:, k] * residents
        m0 = wavg(A0[:, k], wk); m1 = wavg(A1[:, k], wk)
        if m0 < 1e-6:
            print(f"{OCC[k]:<6} {'(无岗位面, 跳过)'}"); continue
        cv0 = wcv(A0[:, k], wk); cv1 = wcv(A1[:, k], wk)
        means0.append(m0); means1.append(m1); names.append(OCC[k])
        print(f"{OCC[k]:<6} {m0:>10.1f} {m1:>10.1f} {(m1/m0-1)*100:>+6.1f}% {cv0:>6.2f}→{cv1:<6.2f}")
    means0, means1 = np.array(means0), np.array(means1); chg = means1 / means0
    r0 = means0.max() / means0.min(); r1 = means1.max() / means1.min()
    print(f"\n  职业间不平等(max/min): {r0:.2f} → {r1:.2f}  ({'缩小✓' if r1<r0 else '扩大✗'})")
    print(f"  最受益: {names[int(np.argmax(chg))]} (+{chg.max()*100-100:.1f}%); "
          f"最不受益: {names[int(np.argmin(chg))]} (+{chg.min()*100-100:.1f}%)")

    # ---- 收入可达性不平等 ----
    print("\n=== 各收入档可达性 (总岗位可达性, 差异来自住哪) ===")
    print(f"{'收入档':<6} {'前':>10} {'后':>10} {'变化%':>7} {'群体内CV前→后':>14}")
    im0, im1 = [], []
    for ti, nm in enumerate(["低", "中", "高"]):
        wt = inc[:, ti] * residents
        m0 = wavg(T0, wt); m1 = wavg(T1, wt); cv0 = wcv(T0, wt); cv1 = wcv(T1, wt)
        im0.append(m0); im1.append(m1)
        print(f"{nm:<6} {m0:>10.1f} {m1:>10.1f} {(m1/m0-1)*100:>+6.1f}% {cv0:>6.2f}→{cv1:<6.2f}")
    im0, im1 = np.array(im0), np.array(im1)
    print(f"\n  收入间不平等(高/低): {im0[2]/im0[0]:.2f} → {im1[2]/im1[0]:.2f}  "
          f"({'缩小✓' if im1[2]/im1[0]<im0[2]/im0[0] else '扩大✗'})")

    # ---- 构成 vs 区位 拆解 (职业) ----
    print("\n=== 拆解: 职业可达性差异是'构成'还是'区位'? ===")
    comp = dshare.sum(0); comp = comp / comp.sum()        # 全城各职业岗位构成比
    print("  各职业全城岗位构成比(构成): " + " ".join(f"{OCC[k]}{comp[k]*100:.0f}%" for k in range(7)))
    print(f"  构成不平等(max/min构成比): {comp.max()/comp.min():.2f}  (政策搬岗位难动这个)")
    print(f"  区位不平等(可达性 max/min, 已扣构成?): 见上 {r0:.2f}→{r1:.2f}")


if __name__ == "__main__":
    main()
