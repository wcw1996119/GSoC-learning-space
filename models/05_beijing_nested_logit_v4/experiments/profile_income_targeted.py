"""识别性流程 第④步: 收入分档忍受度 T_max 的【定点】CFPS 矩注入 + 验证。

冻住全部参数(尤其 γ_M、NN), 只放 head.raw_T_max(分收入档通勤忍受度),
用 CFPS 收入×通勤矩(目标[22,23,47]min)优化它。验证:
  ① T_max 能否被钉住, 预测分档通勤时间 → [22,23,47]?
  ② γ_M 是否不动(冻住 -> 应原样, 无之前全局注入的副作用)?
  ③ CPC 掉不掉(只动 T_max)?
结论: 钉得住+γ_M稳+CPC稳 -> 收入识别性可【定点】解决(流程第④步通过)。
用法: python experiments/profile_income_targeted.py --pt evaluation_outputs/v4_stgnn_s0.pt
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from train_beijing import make_batch, make_batch_idx, chunk_ranges, segment_sum
from scenario_beijing_A import load_model
PROC = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "evaluation_outputs" / "v4_stgnn_s0.pt"))
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--ko", type=int, default=2500)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--param", default="tmax", choices=["tmax", "betat"],
                    help="收入档杠杆: tmax=忍受度门 / betat=分档时间敏感度")
    a = ap.parse_args()
    ck, ar, data, head, enc, use_soc, dyn = load_model(a.pt)
    O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, a.chunks)

    # 冻全部, 只放 raw_T_max
    for p in head.parameters(): p.requires_grad_(False)
    if enc is not None:
        for p in enc.parameters(): p.requires_grad_(False)
    free = [head.raw_T_max] if a.param == "tmax" else [head.raw_beta_t0, head.raw_beta_t1]
    for p in free: p.requires_grad_(True)

    # NN 嵌入(冻, 一次)
    with torch.no_grad():
        if enc is None: e_o = e_d = None
        elif dyn: e_o, e_d = enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"], data["N"],
                                            data["hour2period"], edge_weight=data.get("edge_weight"))
        else: e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"], edge_weight=data.get("edge_weight"))
    def vnn(o, d, p=0):
        if enc is None: return None
        return (enc.edge_vnn(e_o, e_d, p, o, d) if dyn else enc.edge_vnn(e_o, e_d, o, d)).detach()

    # 收入矩采样
    ss = data["seg_starts"]; rng = np.random.RandomState(7); oids = rng.choice(O, min(a.ko, O), replace=False)
    idx_l = [np.arange(int(ss[oi]), int(ss[oi + 1])) for oi in oids]
    seg_l = [np.full(len(x), k, np.int64) for k, x in enumerate(idx_l)]
    inc_idx = torch.from_numpy(np.concatenate(idx_l)); inc_seg = torch.from_numpy(np.concatenate(seg_l)); nseg = len(oids)
    mm = np.load(PROC / "beijing_cfps_income_moment.npz"); target = torch.tensor(mm["target_ct"])
    oi_e = data["o"][inc_idx]; di_e = data["d"][inc_idx]

    def pred_ct():
        bi = make_batch_idx(data, 0, use_soc, inc_idx, inc_seg, nseg)
        logP = head(bi, v_nn=vnn(oi_e, di_e))
        otot = segment_sum(data["flow"][inc_idx, 0], bi["seg_id"], nseg)
        pred = logP.exp() * otot[bi["seg_id"]]; tmin = bi["t_min"]; tpr = bi["tier_props_o"]
        out = []
        for T in range(3):
            wT = pred * tpr[:, T]; out.append((wT * tmin).sum() / wT.sum().clamp_min(1e-6))
        return torch.stack(out)

    def cpc():
        num = den = 0.0
        with torch.no_grad():
            for (e0, e1, sb, ns) in chunks:
                o = data["o"][e0:e1]; d = data["d"][e0:e1]
                for p in range(4):
                    b = make_batch(data, p, use_soc, e0, e1, sb, ns)
                    logP = head(b, v_nn=vnn(o, d, p))
                    fp = data["flow"][e0:e1, p]; otot = segment_sum(fp, b["seg_id"], ns)
                    pred = logP.exp() * otot[b["seg_id"]]
                    num += float(torch.minimum(pred, fp).sum()); den += float(fp.sum())
        return num / den

    gM0 = head.gamma_M.detach().clone()
    print(f"=== 第④步: 收入档 T_max 定点注入 ({Path(a.pt).name}) ===")
    print(f"目标(CFPS)分档通勤[低,中,高] = {[round(float(x),1) for x in target]}min")
    print(f"注入前: T_max={[round(x,1) for x in head.T_max.tolist()]}  "
          f"预测通勤={[round(float(x),1) for x in pred_ct()]}  γ_M={[round(x,3) for x in head.gamma_M.tolist()]}")
    cpc0 = cpc(); print(f"  注入前 CPC={cpc0:.4f}")

    opt = torch.optim.Adam(free, lr=0.05)
    for s in range(a.steps):
        opt.zero_grad(); ct = pred_ct()
        loss = (((ct - target) / target) ** 2).sum(); loss.backward(); opt.step()
        if (s + 1) % 100 == 0:
            print(f"  step {s+1}: loss={float(loss):.4f} T_max={[round(x,1) for x in head.T_max.tolist()]}")

    print(f"\n注入后: T_max={[round(x,1) for x in head.T_max.tolist()]}  "
          f"预测通勤={[round(float(x),1) for x in pred_ct()]}  γ_M={[round(x,3) for x in head.gamma_M.tolist()]}")
    cpc1 = cpc()
    dG = float((head.gamma_M.detach() - gM0).abs().max())
    print(f"  注入后 CPC={cpc1:.4f} (Δ{cpc1-cpc0:+.4f})  γ_M 最大变动={dG:.2e}(冻住应≈0)")
    print(f"\n判读: 预测通勤逼近[22,23,47] + γ_M≈0变动 + CPC≈不掉 -> 收入忍受度可【定点】钉住, "
          f"第④步通过(无全局注入的 γ_M 副作用)。")


if __name__ == "__main__":
    main()
