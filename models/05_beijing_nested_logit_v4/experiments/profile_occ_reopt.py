"""Occupation 可识别性 — 严谨 profile (固定 delta_match[tier], 重训其他参数)

对照 mode: mode 有 IV->可达性结构通道可识别; occupation 靠 δ·match 直接进 V_M,
无结构通道, 且 match 来自区级(出发地+目的地都区级) -> 近常数(mean0.98 std0.03)。
测: 固定 δ 扫值重训其他, OD-NLL 是否有曲率。预期【平】= 不可识别。
+ 对比: 若有街道级目的地行业(细化 match) 能否救 -> 本脚本先测区级基线。

用法: python experiments/profile_occ_reopt.py [--cells 1500 --steps 40 --tier 1]
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, _inv_softplus
from train_beijing import build_data, make_batch
import torch.nn.functional as F


def fit_fixed_delta(data, tier, delta_val, steps, seed=0):
    torch.manual_seed(seed)
    head = BeijingNestedHead(n_districts=16, use_self_loop=True,
                             use_consideration=True, use_soc_mixture=False, gnn_mode="convex")
    # 固定 delta[tier] = delta_val (通过 raw 反解 + 冻结整个 raw_delta, 其他 tier 用 init)
    with torch.no_grad():
        head.raw_delta.data[tier] = _inv_softplus(max(delta_val, 1e-4))
    head.raw_delta.requires_grad_(False)
    params = [p for n, p in head.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.05)
    nseg = data["num_seg"]; E = len(data["o"])
    flow = data["flow"]; fs = flow.sum().clamp_min(1.0)
    for it in range(steps):
        opt.zero_grad(); loss = 0.0
        for p in range(4):
            b = make_batch(data, p, False, 0, E, 0, nseg)
            loss = loss - (flow[:, p] * head(b)).sum() / fs
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    with torch.no_grad():
        nll = 0.0
        for p in range(4):
            b = make_batch(data, p, False, 0, E, 0, nseg)
            nll = nll - (flow[:, p] * head(b)).sum() / fs
    return float(nll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--tier", type=int, default=1)
    args = ap.parse_args()
    da = SimpleNamespace(smoke_cells=args.cells, use_nn=False, seed=0, val_frac=0.0, device="cpu")
    data = build_data(da)
    grid = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    print(f"  子集 {args.cells}格 {len(data['o']):,}edges, steps={args.steps}, 固定 delta[tier{args.tier}] 扫值")
    print(f"  match(区级cosine) std={float(data['match'].std()):.4f} mean={float(data['match'].mean()):.4f}")
    print(f"\n  {'delta':>6} {'OD-NLL(重训其他)':>16}")
    nlls = []
    for g in grid:
        n = fit_fixed_delta(data, args.tier, g, args.steps); nlls.append(n)
        print(f"  {g:6.1f} {n:16.5f}")
    nl = np.array(nlls)
    print(f"\n  OD-NLL span over delta = {nl.max()-nl.min():.5f}")
    print(f"  -> 小(~0)=occupation 不可识别(区级match无信息); 对比 mode asc span 看差距")

if __name__ == "__main__":
    main()
