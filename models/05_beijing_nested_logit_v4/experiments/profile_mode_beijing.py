"""Mode 可识别性 profile likelihood (本地 CPU, 不需 GPU)

核心问题: 聚合信令 OD 无方式标签 -> mode 参数识别得出吗?
做法: 扫 asc_transit (公交 ASC) 一格格, 看两条 loss:
  (A) 目的地 CPC-NLL (只用信令 OD) -> 预期【平】= OD 识别不出 mode
  (B) 刷卡公交锚 NLL (Ben-Akiva-Morikawa) -> 预期【有曲率】= 刷卡能识别
若 A 平 B 凸, 即证: 外部 mode-labeled 数据(刷卡)解了 mode 可识别性。

用法: python experiments/profile_mode_beijing.py [--cells 3000]
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, BeijingPairEncoder, segment_logsumexp
from train_beijing import build_data, make_batch, segment_sum

PROC = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "evaluation_outputs" / "v4_full_s0.pt"))
    ap.add_argument("--cells", type=int, default=3000)
    args = ap.parse_args()
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    a = ck["args"]

    da = SimpleNamespace(smoke_cells=args.cells, use_nn=False, seed=a["seed"],
                         val_frac=0.0, device="cpu")
    data = build_data(da)
    N = data["N"]; o = data["o"].numpy(); d = data["d"].numpy()

    head = BeijingNestedHead(n_districts=16, use_self_loop=not a.get("no_self_loop", False),
                             use_consideration=a.get("use_consideration", False),
                             use_soc_mixture=False, gnn_mode="convex")  # 不带NN, 纯RUM profile
    # 用训练好的参数填充(共享名), 缺失的(NN/soc)忽略
    sd = {k: v for k, v in ck["head_state"].items() if k in head.state_dict()
          and v.shape == head.state_dict()[k].shape}
    head.load_state_dict(sd, strict=False); head.eval()

    # 刷卡公交 OD -> 本子集 edge 的 obs_transit
    to = np.load(PROC / "beijing_transit_od.npz")
    key = to["to_o"].astype(np.int64) * N + to["to_d"].astype(np.int64)
    val = to["to_n"].astype(np.float64)
    m = {int(k): float(v) for k, v in zip(key, val)}
    ekey = o.astype(np.int64) * N + d.astype(np.int64)
    obs_tr = np.array([m.get(int(k), 0.0) for k in ekey], dtype=np.float64)
    obs_tr_t = torch.from_numpy(obs_tr).float()
    print(f"  子集 {args.cells} 格: edges {len(o):,}, 命中刷卡公交的 edge {int((obs_tr>0).sum()):,} "
          f"(占公交出行 {obs_tr.sum()/val.sum()*100:.0f}%)")

    flow_tot = data["flow"].sum(1)            # (E,) 信令总流
    seg = data["seg_id"]; nseg = data["num_seg"]

    def dest_nll_and_anchor(asc_tr):
        with torch.no_grad():
            head.asc.data[1] = asc_tr
            # 目的地 NLL (全时段加权, full forward chunks=1)
            nll = 0.0
            ptr_acc = torch.zeros(len(o))
            for p in range(4):
                b = make_batch(data, p, False, 0, len(o), 0, nseg)
                logP = head(b)
                nll = nll - (data["flow"][:, p] * logP).sum() / flow_tot.sum()
                if p == 0:
                    ml = head.mode_logits(b)                  # (3,E) 时段无关近似(用ampeak)
                    p_tr = torch.softmax(ml, 0)[1]
            # 刷卡锚: 预测公交分布 vs 观测公交分布 的 CE
            pred_tr = (p_tr * flow_tot).clamp_min(1e-12)
            pred_tr_dist = pred_tr / pred_tr.sum()
            obs_dist = obs_tr_t / obs_tr_t.sum()
            anchor = -(obs_dist * torch.log(pred_tr_dist)).sum()
            return float(nll), float(anchor), float(p_tr.mean())

    base_asc = float(head.asc.data[1].clone())
    grid = np.linspace(-4, 4, 17)
    print(f"\n  asc_transit profile (训练值={base_asc:.2f}):")
    print(f"  {'asc':>6} {'目的地NLL':>12} {'刷卡锚NLL':>12} {'P(transit)均值':>14}")
    rows = []
    for g in grid:
        nll, anc, ptm = dest_nll_and_anchor(float(g))
        rows.append((g, nll, anc, ptm))
        print(f"  {g:6.1f} {nll:12.5f} {anc:12.5f} {ptm:14.3f}")

    nlls = np.array([r[1] for r in rows]); ancs = np.array([r[2] for r in rows])
    print(f"\n  === 识别性判据 (曲率 = max-min span) ===")
    print(f"  目的地 NLL span: {nlls.max()-nlls.min():.5f}  (小=OD 识别不出 mode)")
    print(f"  刷卡锚 NLL span: {ancs.max()-ancs.min():.5f}  (大=刷卡能识别 mode)")
    print(f"  比值 刷卡/OD = {(ancs.max()-ancs.min())/max(nlls.max()-nlls.min(),1e-9):.1f}×")
    print(f"  刷卡锚最优 asc = {grid[ancs.argmin()]:.1f}")

if __name__ == "__main__":
    main()
