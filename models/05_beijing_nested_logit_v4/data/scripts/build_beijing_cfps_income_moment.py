"""CFPS 2020 北京 真·收入×通勤时间矩 -> beijing_cfps_income_moment.npz

源: data/_cfps_tmp/.../cfps2020person_202306.dta (CFPS 2020 个人级, 公开微观)
    qg11=每月税后工资(元/月), qg3011=上下班单程时间(分钟), provcd20=省码(北京11)
出: data/processed/beijing_cfps_income_moment.npz (gitignored)

产出: 收入三分位档(对齐模型收入档 低/中/高) × 平均通勤时间 [target_ct (3,)]
  钉收入档拆分(γ_M/T_max per tier, 现最软那类)。
⚠ 北京样本 110(小); CFPS 收入档(月薪)≈模型收入档(房价)假设; 2020 vs OD 2023。
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd

PROC = Path(__file__).resolve().parents[1] / "processed"
OUT = PROC / "beijing_cfps_income_moment.npz"

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    fp = glob.glob(str(Path(__file__).resolve().parents[1] / "_cfps_tmp" / "**" / "cfps2020person*.dta"), recursive=True)
    assert fp, "CFPS person.dta 未找到 (先解压 rar 到 data/_cfps_tmp)"
    df = pd.read_stata(fp[0], columns=["provcd20", "qg11", "qg3011"], convert_categoricals=False)
    bj = df[df["provcd20"] == 11].copy()
    bj["inc"] = pd.to_numeric(bj["qg11"], errors="coerce")
    bj["ct"] = pd.to_numeric(bj["qg3011"], errors="coerce")
    v = bj[(bj["inc"] > 0) & (bj["ct"].between(1, 180))].copy()
    q33, q67 = v["inc"].quantile([1/3, 2/3]).values
    v["tier"] = np.where(v["inc"] <= q33, 0, np.where(v["inc"] <= q67, 1, 2))   # 0低1中2高
    g = v.groupby("tier")["ct"].agg(["mean", "size"])
    target_ct = g["mean"].reindex([0, 1, 2]).values.astype(np.float32)
    n_tier = g["size"].reindex([0, 1, 2]).fillna(0).values.astype(np.int64)

    np.savez_compressed(OUT,
        target_ct=target_ct,                          # (3,) 各收入档平均通勤时间(min)
        n_tier=n_tier,
        inc_cut=np.array([q33, q67], np.float32),
        n_total=np.int64(len(v)))

    print(f"[OK] {OUT}")
    print(f"  北京有效样本 {len(v)} (各档 {n_tier.tolist()})")
    print(f"  月薪分位 33%={q33:.0f} 67%={q67:.0f} 元")
    print(f"  目标 收入档[低,中,高] 平均通勤时间 = {[round(float(x),1) for x in target_ct]} min")
    print(f"  -> 高收入 {target_ct[2]:.0f}min vs 低 {target_ct[0]:.0f}min = {target_ct[2]/target_ct[0]:.1f}x")

if __name__ == "__main__":
    main()
