"""架构 B 训练器(独立版本): GNN 学吸引力 A_j + RUM 异质偏好。

复用 train_beijing 的 build_data/make_batch/chunk_ranges/segment_sum + beijing_model_B 的 B 模型。
第一版聚焦: OD-NLL + CPC(验全量 B 能否训/拟合, 对比 production 0.69); 锚/收入矩后续加。
B 要双路 ST-GNN: 必须 --use-nn --use-dynamic。dest_attract 替 edge_vnn。
本地小验: --smoke-cells 1500 --epochs 30 (子集); 全量: AutoDL GPU。
用法: python experiments/train_beijing_B.py --use-nn --use-dynamic --road-graph --use-consideration \
        --use-soc-mixture --origin-chunks 16 --epochs 200 --device cuda --out evaluation_outputs/v4_B_s0.pt
"""
import argparse, time
from pathlib import Path
import numpy as np, torch
from train_beijing import build_data, make_batch, chunk_ranges, segment_sum
from beijing_model_B import BeijingAttractEncoder, BeijingAttractHead

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--smoke-cells", type=int, default=0)
    ap.add_argument("--no-self-loop", action="store_true")
    ap.add_argument("--use-nn", action="store_true"); ap.add_argument("--use-dynamic", action="store_true")
    ap.add_argument("--road-graph", action="store_true"); ap.add_argument("--nn-exclude-jobs", action="store_true")
    ap.add_argument("--use-consideration", action="store_true"); ap.add_argument("--use-soc-mixture", action="store_true")
    ap.add_argument("--typed-mass-occ", action="store_true"); ap.add_argument("--gnn-mode", default="residual")
    ap.add_argument("--residual-scale-init", type=float, default=0.1)
    ap.add_argument("--origin-chunks", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "evaluation_outputs" / "v4_B_run.pt"))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not (args.use_nn and args.use_dynamic):
        raise SystemExit("架构 B 需要 --use-nn --use-dynamic(GNN 出吸引力)")

    data = build_data(args)
    O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, args.origin_chunks)
    head = BeijingAttractHead(n_districts=16, use_self_loop=not args.no_self_loop,
                              use_consideration=args.use_consideration,
                              use_soc_mixture=args.use_soc_mixture).to(args.device)
    enc = BeijingAttractEncoder(static_dim=data["Xnode"].shape[1],
                                dyn_dim=data["Xdyn"].shape[2]).to(args.device)
    params = list(head.parameters()) + list(enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    ew = data.get("edge_weight")

    def embed():
        return enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"], data["N"],
                              data["hour2period"], edge_weight=ew)

    flow_all = data["flow"]; train_m = data["edge_is_train"]
    flow_train_sum = (flow_all * train_m.unsqueeze(1)).sum().clamp_min(1.0)
    n_back = len(chunks) * 4; best_cpc = -1.0; t0 = time.time()

    for ep in range(args.epochs):
        head.train(); enc.train(); opt.zero_grad()
        e_o, e_d = embed(); loss_val = 0.0; bi = 0
        for (e0, e1, sb, ns) in chunks:
            d_c = data["d"][e0:e1]
            for p in range(4):
                batch = make_batch(data, p, args.use_soc_mixture, e0, e1, sb, ns)
                a_dest = enc.dest_attract(e_d, p, d_c)
                logP = head(batch, v_nn=a_dest)
                loss_p = -(flow_all[e0:e1, p] * train_m[e0:e1] * logP).sum() / flow_train_sum
                bi += 1
                loss_p.backward(retain_graph=(bi < n_back))
                loss_val += loss_p.item()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            head.eval(); enc.eval(); num = [0.0]*4; den = [0.0]*4
            with torch.no_grad():
                e_o2, e_d2 = embed()
                for (e0, e1, sb, ns) in chunks:
                    vm = data["edge_is_val"][e0:e1]; d_c = data["d"][e0:e1]
                    for p in range(4):
                        batch = make_batch(data, p, args.use_soc_mixture, e0, e1, sb, ns)
                        logP = head(batch, v_nn=enc.dest_attract(e_d2, p, d_c))
                        fp = flow_all[e0:e1, p]; otot = segment_sum(fp, batch["seg_id"], ns)
                        pred = logP.exp() * otot[batch["seg_id"]]
                        num[p] += float(torch.minimum(pred[vm], fp[vm]).sum())
                        den[p] += float(pred[vm].sum() + fp[vm].sum())
            cpcs = [2*num[p]/max(den[p], 1.0) for p in range(4)]
            w = [float((flow_all[:, p] * data["edge_is_val"]).sum()) for p in range(4)]
            cpc = sum(c*wi for c, wi in zip(cpcs, w)) / max(sum(w), 1); best_cpc = max(best_cpc, cpc)
            pr = head.param_report()
            print(f"ep {ep:3d} loss {loss_val:.4f} CPC {cpc:.4f} [{','.join(f'{c:.3f}' for c in cpcs)}] "
                  f"β_dist={pr['beta_dist']} αW={pr['alpha_W']} ν={pr['nu_D']} "
                  f"{'Tmax='+str(pr['T_max']) if pr.get('T_max') else ''}")

    torch.save({"head_state": head.state_dict(), "enc_state": enc.state_dict(),
                "args": vars(args), "param_report": head.param_report(), "best_cpc": best_cpc}, args.out)
    print(f"[OK] saved {args.out}  best CPC {best_cpc:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
