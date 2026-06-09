"""通用可识别性 profile — 固定一个结构参数, 重训其余, 量 OD-NLL span。

推广 profile_occ_reopt.py: 不只 delta_match, 还能测 gamma_M / alpha_W / nu_D / T_max
(per tier) 与 self_loop(全局常数 boost)。判据同 §01:
  OD-NLL span 大 = 聚合 OD 能识别该参数(固定到错值, 重训别的补不回来);
  span ~0       = 不可识别(随便取值都拟合一样)。

比 occ 版更严: 只钉住【目标那一档】, 其余档每步重训后只重置目标项(不冻整张量)。

用法: python experiments/profile_param_reopt.py --param gamma_M --tier 1 [--cells 1500 --steps 40]
      python experiments/profile_param_reopt.py --param self_loop          # 全局 boost 扫描
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, _inv_softplus
from train_beijing import build_data, make_batch

# param -> (raw 属性名, 是否 per-tier, 默认扫描格点[有效值], 说明)
SPEC = {
    "gamma_M":   ("raw_gamma_M", True,  [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],   "引力(岗位)"),
    "alpha_W":   ("raw_alpha_W", True,  [0.0, 0.05, 0.1, 0.2, 0.4, 0.8],   "工资吸引"),
    "nu_D":      ("raw_nu_D",    True,  [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],    "竞争(取|值|)"),
    "T_max":     ("raw_T_max",   True,  [20., 40., 60., 90., 120., 180.],  "通勤忍受度(min)"),
    "self_loop": ("self_loop",   False, [-2., 0., 1., 2., 4., 6.],         "同格 boost(全局常数)"),
}


def fit_fixed(data, param, tier, val, steps, seed=0):
    torch.manual_seed(seed)
    head = BeijingNestedHead(n_districts=16, use_self_loop=True,
                             use_consideration=True, use_soc_mixture=False, gnn_mode="convex")
    raw_name, per_tier, _, _ = SPEC[param]
    raw = getattr(head, raw_name)
    if param == "self_loop":
        with torch.no_grad():
            raw.data[:] = val                  # 全部 self-edge 一个常数 boost
        raw.requires_grad_(False)
        pin = None
    else:
        raw_val = _inv_softplus(max(abs(val), 1e-4))   # 有效值 -> raw (softplus 反解)
        with torch.no_grad():
            raw.data[tier] = raw_val
        pin = (raw, tier, raw_val)             # 每步后重置这一档, 其余档照训

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
        if pin is not None:
            with torch.no_grad():
                pin[0].data[pin[1]] = pin[2]   # 重钉目标档
    with torch.no_grad():
        nll = 0.0
        for p in range(4):
            b = make_batch(data, p, False, 0, E, 0, nseg)
            nll = nll - (flow[:, p] * head(b)).sum() / fs
    return float(nll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, choices=list(SPEC.keys()))
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--cells", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--grid", default="", help="逗号分隔覆盖默认格点")
    args = ap.parse_args()

    da = SimpleNamespace(smoke_cells=args.cells, use_nn=False, seed=0, val_frac=0.0, device="cpu")
    data = build_data(da)
    raw_name, per_tier, grid_def, desc = SPEC[args.param]
    grid = [float(x) for x in args.grid.split(",")] if args.grid else grid_def
    tier_str = f"[tier{args.tier}]" if per_tier else "(全局)"
    print(f"  param={args.param}{tier_str} {desc}  子集{args.cells}格 {len(data['o']):,}edges steps={args.steps}")
    print(f"\n  {'值':>8} {'OD-NLL(重训其余)':>18}")
    nlls = []
    for g in grid:
        n = fit_fixed(data, args.param, args.tier, g, args.steps); nlls.append(n)
        print(f"  {g:8.2f} {n:18.5f}")
    nl = np.array(nlls)
    span = nl.max() - nl.min()
    print(f"\n  OD-NLL span = {span:.5f}")
    print(f"  -> 对照: delta_match 0.00004(不可识别) / mode asc 0.035(也弱) / 大=聚合OD识别")


if __name__ == "__main__":
    main()
