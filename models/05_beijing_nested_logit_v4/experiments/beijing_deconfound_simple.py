"""Paper B 上北京真实数据·第一步(去confound 核心对比):
真实北京 OD 上拟合引力 logit P(j|i)∝exp(α·logM_j − β·t_ij), 比较
  ① t=t_obs(堵后时间, 老办法/naive)  vs  ② t=t_ff(自由流时间, 去confound)
β 差多少 = 拥堵把"时间敏感度"估偏了多少(forced-choice 混淆在真实数据上的量级)。
注: 完整可微均衡是后续; 这步用现成 t_obs/t_ff, 立刻出真实数字。本地 CPU。
"""
from pathlib import Path
import numpy as np, torch

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def seg_softmax_nll(V, seg_id, nseg, flow):
    smax = torch.full((nseg,), -1e30).scatter_reduce(0, seg_id, V, "amax", include_self=False)
    z = V - smax[seg_id]
    ssum = torch.zeros(nseg).index_add(0, seg_id, z.exp())
    logP = z - ssum[seg_id].clamp_min(1e-30).log()
    return -(flow * logP).sum() / flow.sum(), logP


def fit(logM_d, t, seg_id, nseg, flow, steps=400):
    a = torch.tensor(0.5, requires_grad=True); b = torch.tensor(0.05, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        V = a * logM_d - b * t
        loss, _ = seg_softmax_nll(V, seg_id, nseg, flow)
        loss.backward(); opt.step()
    return float(a), float(b)


def main():
    e = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    g = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    o = e["o_idx"].astype(np.int64); d = e["d_idx"].astype(np.int64)
    t_obs = e["t_obs"].astype(np.float32); t_ff = e["t_ff"].astype(np.float32)
    flow = e["flow"].sum(1).astype(np.float32); seg_ptr = e["seg_ptr"].astype(np.int64)
    jobs = g["jobs"].astype(np.float32); logM = np.log(jobs + 1.0)

    nseg = len(seg_ptr) - 1
    seg_id = torch.from_numpy(np.repeat(np.arange(nseg), np.diff(seg_ptr)))
    logM_d = torch.from_numpy(logM[d]); fl = torch.from_numpy(flow)
    to = torch.from_numpy(t_obs); tf = torch.from_numpy(t_ff)
    print(f"北京真实 OD: {len(o):,} 边, {nseg:,} 出发地")
    print(f"  t_obs 中位 {np.median(t_obs):.1f}min, t_ff 中位 {np.median(t_ff):.1f}min, "
          f"拥堵延误 t_obs/t_ff 中位 {np.median(t_obs/np.maximum(t_ff,1)):.2f}\n")

    a_o, b_o = fit(logM_d, to, seg_id, nseg, fl)        # naive: 堵后时间
    a_f, b_f = fit(logM_d, tf, seg_id, nseg, fl)        # 去confound: 自由流时间
    print(f"① naive(堵后时间 t_obs):  α̂={a_o:.3f}  β̂(怕堵)={b_o:.4f}")
    print(f"② 去confound(自由流 t_ff): α̂={a_f:.3f}  β̂(怕堵)={b_f:.4f}")
    print(f"\n→ 用堵后时间 vs 自由流时间, 时间敏感度 β 差 {(b_f/b_o-1)*100:+.0f}%")
    print("判读: 若 β_ff > β_obs, 说明老办法(用堵后时间)【低估】了人的怕堵程度"
          "(forced-choice 混淆), 真实北京也有这个偏差; 完整均衡法会把真值钉在两者之间。")


if __name__ == "__main__":
    main()
