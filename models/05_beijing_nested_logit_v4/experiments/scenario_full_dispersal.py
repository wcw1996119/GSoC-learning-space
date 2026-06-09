"""B5 收口: 整合两个政策输出 —— 疏解 -> 拥堵反馈 -> GE 可达性不平等。

三套可达性 A_i^g = Σ_j 岗位_j^g · exp(-θ t_ij):
  baseline    : 岗位0, t_obs
  partial-eq  : 岗位1(疏解), t_obs        (只换岗位, 不管拥堵)
  GE          : 岗位1, t'(=t_obs×拥堵变化比)  (岗位+拥堵反馈)
拥堵反馈: t'_ij = t_obs × sqrt( (d1(o)/d0(o))·(d1(d)/d0(d)) ), d=1+α·min(V/C,cap)^β(来自B5)。
问: 拥堵反馈让 equity 结论变多少? 用法: python experiments/scenario_full_dispersal.py
"""
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
THETA = 0.0947; ALPHA, BETA, DELAY_CAP = 0.15, 4.0, 2.9   # 延误封顶=实测 t_obs/t_ff 上限
TONGZHOU = 13; CORE = [0, 12]; ADD = 300000.0; CUT = 0.2
OCC = ["负责人", "专技", "办事", "商服", "农林", "生产", "其他"]


def wavg(x, w): return float((x * w).sum() / max(w.sum(), 1e-9))


def main():
    g = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    jobs = g["jobs"].astype(np.float64); residents = g["residents"].astype(np.float64); district = g["district_idx"]
    N = len(jobs)
    e = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    o = e["o_idx"].astype(np.int64); d = e["d_idx"].astype(np.int64); t = e["t_obs"].astype(np.float64)
    occ = np.load(PROC / "beijing_occupation.npz", allow_pickle=True)
    dshare = occ["demand_share"].astype(np.float64); soc = occ["soc_props"].astype(np.float64)
    inc = np.load(PROC / "beijing_income.npz", allow_pickle=True)["income_tier_props"].astype(np.float64)
    cong = np.load(PROC / "beijing_congest_scenario_p0.npz", allow_pickle=True)
    vc0 = cong["cell_vc0"].astype(np.float64); vc1 = cong["cell_vc1"].astype(np.float64)

    def hansen(jb, tt):
        dec = np.exp(-THETA * tt)
        A = np.zeros((N, 7))
        for k in range(7):
            A[:, k] = np.bincount(o, weights=(jb * dshare[:, k])[d] * dec, minlength=N)
        return A

    jobs1 = jobs.copy(); tc = np.where(district == TONGZHOU)[0]; cc = np.where(np.isin(district, CORE))[0]
    w = jobs[tc] + 1.0; jobs1[tc] += ADD * w / w.sum(); jobs1[cc] *= (1 - CUT)
    # 拥堵反馈 -> t'
    d0 = np.minimum(1 + ALPHA * vc0 ** BETA, DELAY_CAP); d1 = np.minimum(1 + ALPHA * vc1 ** BETA, DELAY_CAP)
    ratio = np.sqrt((d1[o] / d0[o]) * (d1[d] / d0[d]))
    tge = t * ratio
    print(f"拥堵反馈: OD 时间比中位 {np.median(ratio):.3f}, 通州终点OD时间 ×{np.median(ratio[np.isin(d,tc)]):.3f}")

    A_base = hansen(jobs, t); A_part = hansen(jobs1, t); A_ge = hansen(jobs1, tge)

    print("\n=== 各职业可达性: 疏解效应 (partial=只换岗位 / GE=+拥堵反馈) ===")
    print(f"{'职业':<6}{'partial%':>9}{'GE%':>8}{'拥堵反馈差':>10}")
    for k in range(7):
        wk = soc[:, k] * residents
        b = wavg(A_base[:, k], wk)
        if b < 1e-6: continue
        p = wavg(A_part[:, k], wk) / b - 1; ge = wavg(A_ge[:, k], wk) / b - 1
        print(f"{OCC[k]:<6}{p*100:>+8.1f}%{ge*100:>+7.1f}%{(ge-p)*100:>+9.1f}pp")

    print("\n=== 各收入档可达性 (总岗位) ===")
    At_base = A_base.sum(1)  # 近似总(各职业和)... 用总岗位更准:
    Atot_b = np.bincount(o, weights=jobs[d] * np.exp(-THETA * t), minlength=N)
    Atot_p = np.bincount(o, weights=jobs1[d] * np.exp(-THETA * t), minlength=N)
    Atot_g = np.bincount(o, weights=jobs1[d] * np.exp(-THETA * tge), minlength=N)
    for ti, nm in enumerate(["低", "中", "高"]):
        wt = inc[:, ti] * residents
        b = wavg(Atot_b, wt); p = wavg(Atot_p, wt) / b - 1; ge = wavg(Atot_g, wt) / b - 1
        print(f"{nm:<6}{p*100:>+8.1f}%{ge*100:>+7.1f}%{(ge-p)*100:>+9.1f}pp")

    print("\n判读: '拥堵反馈差'(pp)小 -> 拥堵反馈对 equity 是二阶, 可达性结论稳;"
          " 大 -> 必须 GE 才对。全市拥堵几乎不变(B5)预期二阶。")


if __name__ == "__main__":
    main()
