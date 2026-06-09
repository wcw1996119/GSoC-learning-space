"""Paper B 评估第三块: forced-choice 的空间分布 + equity 含义。

用去confound 的【真实偏好】(β_ff, 从自由流拟合), 算每个出发地的【可达性】两次:
  A_ff  = logsumexp_j(α·logM_j − β·t_ff)    自由流(无拥堵)能达到的
  A_cong= logsumexp_j(α·logM_j − β·t_obs)   拥堵下实际能达到的
拥堵造成的可达性损失 loss_i = A_ff − A_cong (越大=被拥堵 forced 越狠)。
按收入分档看: 谁损失最多 = equity 含义。
注: 用真实偏好评估两种拥堵情形, 是去confound 的直接政策产物。本地 CPU。
"""
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
ALPHA, BETA_FF = 0.535, 0.177          # 去confound 拟合值(自由流 -> 真实偏好), 见 beijing_deconfound_simple


def seg_logsumexp(V, seg_ptr):
    out = np.empty(len(seg_ptr) - 1, dtype=np.float64)
    for i in range(len(seg_ptr) - 1):
        s, e = seg_ptr[i], seg_ptr[i + 1]
        if e > s:
            v = V[s:e]; m = v.max(); out[i] = m + np.log(np.exp(v - m).sum())
        else:
            out[i] = -np.inf
    return out


def main():
    e = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    g = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    inc = np.load(PROC / "beijing_income.npz", allow_pickle=True)
    print("income npz keys:", list(inc.keys()))

    o = e["o_idx"].astype(np.int64); d = e["d_idx"].astype(np.int64)
    t_obs = e["t_obs"].astype(np.float64); t_ff = e["t_ff"].astype(np.float64)
    seg_ptr = e["seg_ptr"].astype(np.int64)
    jobs = g["jobs"].astype(np.float64); logM = np.log(jobs + 1.0)
    logM_d = logM[d]

    A_ff = seg_logsumexp(ALPHA * logM_d - BETA_FF * t_ff, seg_ptr)
    A_cong = seg_logsumexp(ALPHA * logM_d - BETA_FF * t_obs, seg_ptr)
    loss = A_ff - A_cong                      # 效用单位
    loss_min = loss / BETA_FF                  # 折成"分钟当量"可达性损失

    # 出发地 cell id (每段第一条边的 o)
    orig_cell = o[seg_ptr[:-1]]
    nseg = len(seg_ptr) - 1

    # 收入: 找连续收入代理 (income_z 或类似)
    inc_key = next((k for k in ["income_z", "income", "wage_z", "price_z"] if k in inc), None)
    income_cell = inc[inc_key].astype(np.float64) if inc_key else None
    print(f"\n出发地 {nseg:,}; 拥堵可达性损失(分钟当量) 中位 {np.median(loss_min):.2f}, "
          f"均值 {loss_min.mean():.2f}\n")

    if income_cell is not None:
        income_o = income_cell[orig_cell]
        msk = np.isfinite(income_o) & np.isfinite(loss_min)
        io_, lo_ = income_o[msk], loss_min[msk]
        r = np.corrcoef(io_, lo_)[0, 1]
        print(f"收入代理 = '{inc_key}'; 收入 vs 可达性损失 相关 r = {r:+.3f}")
        # 按收入三档
        q = np.quantile(io_, [1/3, 2/3])
        for name, sel in [("低收入", io_ <= q[0]), ("中收入", (io_ > q[0]) & (io_ <= q[1])), ("高收入", io_ > q[1])]:
            print(f"  {name}: 可达性损失中位 {np.median(lo_[sel]):.2f} 分钟当量  (n={sel.sum():,})")
        print(f"\n判读: r<0 = 低收入区损失更大(拥堵 equity 恶化); r>0 = 高收入区损失更大。"
              " 这是去confound 真实偏好下的 forced-choice equity, Paper B §评估第三块。")
    else:
        print("没找到收入代理键, 仅出可达性损失分布。")
    np.savez(PROC.parent / "paperB_forced_choice_loss.npz",
             orig_cell=orig_cell, loss_util=loss, loss_min=loss_min)
    print("\n[OK] 每出发地损失存 data/paperB_forced_choice_loss.npz (供画图)")


if __name__ == "__main__":
    main()
