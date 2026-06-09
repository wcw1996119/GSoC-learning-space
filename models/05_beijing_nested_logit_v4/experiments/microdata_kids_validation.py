"""微观验证(2015 1%普查, 北京就业通勤者): 有娃 vs 无娃 -> 通勤时间 + 开车率, 控混杂。
回应导师"有孩子更就近/开车"假设。控: 年龄/性别/职业/婚姻/居住区。不碰模型。
M2 居住码 / M34 性别 / M35 出生年 / M58 职业 / M59 工作地类型 / M61 方式 / M62 通勤时间 /
M68 婚姻 / M73+M74 本户内存活子女数。
"""
from pathlib import Path
import numpy as np, pandas as pd

CSV = Path(r"D:\UoM\congestion_project\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式"
           r"\2015年全国1%普查抽样调查个人微观数据census2015_origin CSV格式.csv")

def num(s): return pd.to_numeric(s, errors="coerce")

def main():
    cols = ["M2", "M34", "M35", "M58", "M59", "M61", "M62", "M68", "M73", "M74"]
    df = pd.read_csv(CSV, dtype=str, usecols=lambda c: c in cols, low_memory=False)
    print("列:", df.columns.tolist(), " 总行", len(df))
    # 北京居住 (M2 以 11 开头)
    df = df[df["M2"].astype(str).str.startswith("11")].copy()
    # 就业通勤者: 有工作地(M59 in 1,2 市内) + 有通勤时间
    df["M59n"] = num(df["M59"]); df["M62n"] = num(df["M62"])
    df = df[df["M59n"].isin([1, 2]) & (df["M62n"] > 0) & (df["M62n"] < 240)].copy()
    print("北京就业通勤者:", len(df))

    df["age"] = 2015 - num(df["M35"]); df = df[(df["age"] >= 16) & (df["age"] <= 70)]
    df["male"] = (num(df["M34"]) == 1).astype(int)
    df["occ"] = df["M58"].astype(str).str[0]
    df["marital"] = df["M68"].astype(str)
    df["resid"] = df["M2"].astype(str).str[:6]
    df["nkids"] = num(df["M73"]).fillna(0) + num(df["M74"]).fillna(0)
    df["has_kids"] = (df["nkids"] > 0).astype(int)
    m61 = num(df["M61"])
    print("\nM61 方式分布:\n", m61.value_counts().sort_index().to_string())
    df["drive"] = m61.isin([5]).astype(int)        # 5=私家车
    df["drive_moto"] = m61.isin([4, 5]).astype(int)  # +摩托

    print(f"\n样本 {len(df)}; 有娃 {df['has_kids'].mean()*100:.0f}%  平均年龄 {df['age'].mean():.0f}")
    # ---- 原始(不控) ----
    print("\n=== 原始(不控混杂) ===")
    g = df.groupby("has_kids").agg(通勤=("M62n", "mean"), 开车率=("drive", "mean"),
                                   开车摩托=("drive_moto", "mean"), n=("M62n", "size"))
    print(g.to_string())

    # ---- 控混杂回归 ----
    try:
        import statsmodels.formula.api as smf
        print("\n=== 控混杂(年龄+年龄²+性别+职业+婚姻+居住区) ===")
        d = df.copy(); d["age2"] = d["age"]**2
        f = "has_kids + age + age2 + male + C(occ) + C(marital) + C(resid)"
        r1 = smf.ols("M62n ~ " + f, data=d).fit()
        b, p = r1.params["has_kids"], r1.pvalues["has_kids"]
        base = d["M62n"].mean()
        print(f"通勤时间: 有娃效应 = {b:+.2f}分 (基准{base:.1f}分, {b/base*100:+.1f}%) p={p:.3f}")
        r2 = smf.logit("drive ~ " + f, data=d).fit(disp=0)
        bo, po = r2.params["has_kids"], r2.pvalues["has_kids"]
        print(f"开车(私家车): 有娃 log-odds = {bo:+.3f} (OR={np.exp(bo):.2f}) p={po:.3f}")
        r3 = smf.logit("drive_moto ~ " + f, data=d).fit(disp=0)
        print(f"开车+摩托:   有娃 log-odds = {r3.params['has_kids']:+.3f} "
              f"(OR={np.exp(r3.params['has_kids']):.2f}) p={r3.pvalues['has_kids']:.3f}")
        print("\n判读: 控混杂后 有娃效应 显著(p<0.05)且量级可观 -> 导师假设成立, 可考虑进模型;"
              " 不显著/微小 -> 是混杂(如住得近), 别加。")
    except Exception as e:
        print("statsmodels 不可用, 仅原始:", e)


if __name__ == "__main__":
    main()
