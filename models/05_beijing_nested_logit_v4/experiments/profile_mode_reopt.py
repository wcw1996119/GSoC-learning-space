"""Mode 可识别性 — 严谨 profile (固定 asc_transit, 重新优化其他参数)

vs profile_mode_beijing.py (其他参数固定): 那个会高估识别性。
真识别性: 每个 asc 值下重训其他参数, 看 OD-NLL 是否仍有曲率。
  - 仍凸 -> mode 被 OD 真识别 (其他参数补偿不掉)
  - 被压平 -> OD 不可识别 (其他参数补偿) -> 再看 +刷卡锚 能否恢复

用法: python experiments/profile_mode_reopt.py [--cells 1500 --steps 40]
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead
from train_beijing import build_data, make_batch

PROC = ROOT / "data" / "processed"


def fit_fixed_asc(data, obs_tr_t, asc_val, steps, anchor_w, seed=0):
    """固定 asc_transit=asc_val, 优化其他 head 参数 (OD [+anchor]). 返回最终 OD-NLL。"""
    torch.manual_seed(seed)
    head = BeijingNestedHead(n_districts=16, use_self_loop=True,
                             use_consideration=True, use_soc_mixture=False, gnn_mode="convex")
    head.asc.data[1] = asc_val
    # 冻结 asc (只它固定, 其他全训)
    head.asc.requires_grad_(False)
    params = [p for n, p in head.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.05)
    nseg = data["num_seg"]; E = len(data["o"])
    flow = data["flow"]; flow_sum = flow.sum().clamp_min(1.0)
    flow_tot = flow.sum(1)
    obs_dist = obs_tr_t / obs_tr_t.sum()
    for it in range(steps):
        opt.zero_grad(); head.asc.data[1] = asc_val
        loss = 0.0
        for p in range(4):
            b = make_batch(data, p, False, 0, E, 0, nseg)
            logP = head(b)
            loss = loss - (flow[:, p] * logP).sum() / flow_sum
            if p == 0 and anchor_w > 0:
                p_tr = torch.softmax(head.mode_logits(b), 0)[1]
                pred = (p_tr * flow_tot).clamp_min(1e-12); pred = pred / pred.sum()
                loss = loss + anchor_w * (-(obs_dist * torch.log(pred)).sum())
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    # 最终纯 OD-NLL (不含 anchor)
    with torch.no_grad():
        head.asc.data[1] = asc_val; nll = 0.0
        for p in range(4):
            b = make_batch(data, p, False, 0, E, 0, nseg)
            nll = nll - (flow[:, p] * head(b)).sum() / flow_sum
        p_tr = torch.softmax(head.mode_logits(make_batch(data,0,False,0,E,0,nseg)),0)[1]
    return float(nll), float(p_tr.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    da = SimpleNamespace(smoke_cells=args.cells, use_nn=False, seed=0, val_frac=0.0, device="cpu")
    data = build_data(da)
    N = data["N"]; o = data["o"].numpy(); d = data["d"].numpy()
    to = np.load(PROC / "beijing_transit_od.npz")
    mp = {int(k): float(v) for k, v in zip(to["to_o"].astype(np.int64)*N + to["to_d"].astype(np.int64), to["to_n"])}
    obs_tr_t = torch.from_numpy(np.array([mp.get(int(k),0.0) for k in (o.astype(np.int64)*N+d.astype(np.int64))])).float()

    grid = [-3,-2,-1,0,1,2,3]
    print(f"  子集 {args.cells}格 {len(o):,}edges, steps={args.steps}")
    print(f"\n  {'asc_tr':>6} | OD-only重训: {'OD-NLL':>9} {'P(tr)':>6} | OD+刷卡锚重训: {'OD-NLL':>9} {'P(tr)':>6}")
    od_nll=[]; an_nll=[]
    for g in grid:
        n0,p0 = fit_fixed_asc(data, obs_tr_t, float(g), args.steps, anchor_w=0.0)
        n1,p1 = fit_fixed_asc(data, obs_tr_t, float(g), args.steps, anchor_w=1.0)
        od_nll.append(n0); an_nll.append(n1)
        print(f"  {g:6.1f} |              {n0:9.5f} {p0:6.3f} |               {n1:9.5f} {p1:6.3f}")
    od=np.array(od_nll); an=np.array(an_nll)
    print(f"\n  === 严谨识别性 (重训其他参数后 OD-NLL 曲率) ===")
    print(f"  OD-only 重训   OD-NLL span: {od.max()-od.min():.5f}  最优asc={grid[od.argmin()]}")
    print(f"    -> 小=mode 被其他参数补偿掉, OD 不可识别; 大=OD 真识别 mode")
    print(f"  OD+刷卡锚 重训 OD-NLL span: {an.max()-an.min():.5f}  最优asc={grid[an.argmin()]}")
    print(f"    -> 刷卡把 P(transit) 钉在 {[round(p,2) for p in [an_nll]][0] if False else '见上'} 附近")

if __name__ == "__main__":
    main()
