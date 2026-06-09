"""地基验证: 各 RUM 行为成分"跨区一致"吗? —— 给"一致性=可信"这个思路验地基。

做法(便宜, 不重训): 把北京按城市梯度分 4 片(核心/近郊/远郊/远), NN 冻住,
在【全局训练好的参数】处, 算每一片的 OD-NLL 对每个 RUM 系数的【梯度(拉力)】。
  - 某系数: 各片拉力都小/同向 -> 各片"满意"全局值 -> 跨区一致 -> 可信(可外推);
  - 各片拉力大且打架 -> 各片想要不同值 -> 跨区不一致 -> 伪相关(别外推)。
判据: 我们已知引力(γ_M)给"假如"稳(CV 4%) -> 它应跨区一致 -> 若是, 地基稳。
用法: python experiments/validate_consistency.py --pt evaluation_outputs/v4_stgnn_s0.pt
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "experiments"))
from train_beijing import make_batch, chunk_ranges
from scenario_beijing_A import load_model

# 城市梯度 4 片 (区 idx): 核心/近郊/远郊/远
REGIONS = {
    "核心": [0, 12, 9, 10],      # 东城/西城/朝阳/海淀
    "近郊": [1, 11, 8, 13],      # 丰台/石景山/昌平/通州
    "远郊": [2, 15, 7, 14],      # 大兴/顺义/房山/门头沟
    "远":  [3, 4, 5, 6],         # 密云/平谷/延庆/怀柔
}
# 看哪些系数(名字 -> head 属性)
COEFS = [("γ_M", "raw_gamma_M"), ("β_t0", "raw_beta_t0"), ("β_t1", "raw_beta_t1"),
         ("α_W", "raw_alpha_W"), ("ν_D", "raw_nu_D"), ("λ", "raw_lambda")]


def region_grad(data, head, enc, dyn, use_soc, district, region_cells, chunks):
    """该片 OD-NLL 对各 RUM 系数 + 对 GNN(encoder) 的梯度 (片内 flow 归一, 各片等权)。
    GNN 用 in-graph v_nn -> 能拿到 encoder 梯度范数 = 该片把 GNN 往多用力地拉。"""
    in_reg = torch.zeros(data["N"], dtype=torch.bool); in_reg[region_cells] = True
    head.zero_grad()
    if enc is not None: enc.zero_grad()
    flow = data["flow"]; fs = 0.0
    for (e0, e1, sb, ns) in chunks:
        o = data["o"][e0:e1]; m = in_reg[o]
        if m.any(): fs += float(flow[e0:e1][m].sum())
    fs = max(fs, 1.0)
    # encoder in-graph (每片重算, 拿 encoder 梯度)
    if enc is not None:
        if dyn:
            e_o, e_d = enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"], data["N"],
                                      data["hour2period"], edge_weight=data.get("edge_weight"))
        else:
            e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"], edge_weight=data.get("edge_weight"))
    for (e0, e1, sb, ns) in chunks:
        o = data["o"][e0:e1]; d = data["d"][e0:e1]; m = in_reg[o]
        if not bool(m.any()): continue
        for p in range(4):
            b = make_batch(data, p, use_soc, e0, e1, sb, ns)
            vnn = (enc.edge_vnn(e_o, e_d, p, o, d) if dyn else enc.edge_vnn(e_o, e_d, o, d)) if enc is not None else None
            logP = head(b, v_nn=vnn)
            loss = -(flow[e0:e1, p] * m * logP).sum() / fs
            loss.backward(retain_graph=True)
    g = {}
    for nm, attr in COEFS:
        t = getattr(head, attr, None)
        g[nm] = t.grad.detach().abs().mean().item() if (t is not None and t.grad is not None) else float("nan")
    # GNN: encoder 所有参数梯度的总范数 / 参数量 (跟系数可比的"平均拉力")
    if enc is not None:
        gs = [pp.grad.detach() for pp in enc.parameters() if pp.grad is not None]
        npar = sum(x.numel() for x in gs)
        g["GNN"] = float(torch.sqrt(sum((x**2).sum() for x in gs)) / max(npar**0.5, 1.0))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "evaluation_outputs" / "v4_stgnn_s0.pt"))
    ap.add_argument("--chunks", type=int, default=16)
    args = ap.parse_args()
    ck, a, data, head, enc, use_soc, dyn = load_model(args.pt)
    for prm in head.parameters(): prm.requires_grad_(True)
    if enc is not None:
        for prm in enc.parameters(): prm.requires_grad_(True)
    O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, args.chunks)
    district = data["district"]

    print(f"=== 地基验证: {Path(args.pt).name} (CPC {ck['best_cpc']:.4f}) ===")
    print("各片对 RUM 系数 + GNN 的'拉力'(梯度|均|, 小=该片满意=一致):\n")
    grads = {}
    for rname, rdists in REGIONS.items():
        cells = np.where(np.isin(district.numpy(), rdists))[0]
        g = region_grad(data, head, enc, dyn, use_soc, district, torch.from_numpy(cells), chunks)
        grads[rname] = g
        print(f"  {rname}: " + "  ".join(f"{k}={v:.2e}" for k, v in g.items()))

    names = [c[0] for c in COEFS] + (["GNN"] if enc is not None else [])
    print("\n跨区一致性(CV=std/mean; 小=一致/可信, 大=打架/伪相关):")
    rows = []
    for nm in names:
        vals = np.array([grads[r][nm] for r in REGIONS])
        cv = vals.std() / max(abs(vals.mean()), 1e-12)
        rows.append((nm, cv, vals.mean()))
    rows.sort(key=lambda x: x[1])
    for nm, cv, mn in rows:
        tag = "一致✓" if cv < 0.6 else ("中" if cv < 1.2 else "打架✗")
        print(f"  {nm:5s}: CV={cv:5.2f}  ({tag})  平均拉力 {mn:.2e}")
    cv = {nm: c for nm, c, _ in rows}
    print(f"\n判据(关键): GNN 的 CV({cv.get('GNN',float('nan')):.2f}) 要 >> 引力 γ_M 的 CV({cv.get('γ_M',float('nan')):.2f})")
    print("  GNN >> γ_M -> '不一致=不可信'成立, 一致性能区分好坏 -> 地基稳")
    print("  GNN ≈ γ_M  -> 一致性分不出好坏 -> 地基不稳, 思路要改")


if __name__ == "__main__":
    main()
