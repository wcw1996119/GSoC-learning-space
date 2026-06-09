"""架构 B 地基验证(自包含, 小稠密子集, 本地CPU): GNN算吸引力 A_j + RUM异质偏好。

B:   A_j = GNN(空间/岗位特征) ;  V_ij^k = softplus(α_k)·A_j − softplus(β_k)·t_ij  (k=收入档)
对照 G(引力): A_j = log(岗位_j)  其余同结构。
看: ① B 能训+CPC合理 ② α_k/β_k 异质出来没 ③ 学出吸引力 vs 引力 CPC 谁高。
子集 = top-N 岗位格(稠密 N×N), 避开全量稀疏 segfault。
用法: python experiments/smoke_B_attractiveness.py --N 1200 --epochs 150
"""
import argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def haversine(a, b):  # (.,2) latlon, km
    R = 6371.0; la1, lo1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    la2, lo2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(d))


class MiniSAGE(nn.Module):
    """2 层 GraphSAGE 算 per-cell 吸引力标量。"""
    def __init__(self, fin, hid=24):
        super().__init__()
        self.l1 = nn.Linear(fin * 2, hid); self.l2 = nn.Linear(hid * 2, hid)
        self.out = nn.Linear(hid, 1)
    def conv(self, lin, h, nbr_idx, nbr_ptr):
        # 邻居均值池化
        deg = np.diff(nbr_ptr); agg = torch.zeros_like(h)
        seg = torch.repeat_interleave(torch.arange(len(deg)), torch.tensor(deg))
        agg.index_add_(0, seg, h[nbr_idx]); agg = agg / torch.tensor(deg).clamp_min(1).unsqueeze(1)
        return F.relu(lin(torch.cat([h, agg], -1)))
    def forward(self, x, nbr_idx, nbr_ptr):
        h = self.conv(self.l1, x, nbr_idx, nbr_ptr)
        h = self.conv(self.l2, h, nbr_idx, nbr_ptr)
        return self.out(h).squeeze(-1)   # (N,) 吸引力


class ChoiceB(nn.Module):
    def __init__(self, fin, n_tier=3, gravity=False):
        super().__init__()
        self.gravity = gravity
        if not gravity: self.gnn = MiniSAGE(fin)
        self.raw_alpha = nn.Parameter(torch.zeros(n_tier))   # softplus -> α_k≥0 (看重吸引力)
        self.raw_beta = nn.Parameter(torch.zeros(n_tier))    # softplus -> β_k≥0 (怕远)
    def attract(self, x, nbr_idx, nbr_ptr, logjobs):
        return logjobs if self.gravity else self.gnn(x, nbr_idx, nbr_ptr)
    def forward(self, A, t, oi, di, tier_props, nseg):
        a = F.softplus(self.raw_alpha); b = F.softplus(self.raw_beta)
        logP_mix = None
        for k in range(3):
            V = a[k] * A[di] - b[k] * t                       # (E,)
            # segment softmax by origin
            m = torch.zeros(nseg).index_reduce_(0, oi, V, "amax", include_self=False)
            ex = torch.exp(V - m[oi]); s = torch.zeros(nseg).index_add_(0, oi, ex)
            logp = V - m[oi] - torch.log(s[oi])
            w = tier_props[oi, k]
            logP_mix = w * logp.exp() if logP_mix is None else logP_mix + w * logp.exp()
        return torch.log(logP_mix.clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1200); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--knn", type=int, default=8); a = ap.parse_args()
    g = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    jobs = g["jobs"].astype(np.float64); res = g["residents"].astype(np.float64); ll = g["coords_latlon"].astype(np.float64)
    inc = np.load(PROC / "beijing_income.npz", allow_pickle=True)["income_tier_props"].astype(np.float64)
    top = np.argsort(-jobs)[:a.N]                       # top-N 岗位格(当 origin+dest)
    from scipy.spatial import cKDTree
    xy = g["xy_m"][top]; tree = cKDTree(xy)
    _, nb = tree.query(xy, k=a.knn + 1); nb = nb[:, 1:]            # kNN 邻居(去自己)
    nbr_idx = torch.tensor(nb.reshape(-1)); nbr_ptr = np.arange(0, a.N * a.knn + 1, a.knn)
    # 稠密 choice set: 每 origin -> 所有 N dest
    oi = torch.repeat_interleave(torch.arange(a.N), a.N); di = torch.arange(a.N).repeat(a.N)
    t = torch.tensor(haversine(ll[top][oi.numpy()], ll[top][di.numpy()]) / 25.0 * 60.0, dtype=torch.float32)  # min @25km/h
    # 观测流: 从 edges 取子集对
    ed = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    pos = {c: i for i, c in enumerate(top)}
    o_e = ed["o_idx"]; d_e = ed["d_idx"]; fl = ed["flow"].sum(1)
    mask = np.isin(o_e, top) & np.isin(d_e, top)
    om = np.array([pos[c] for c in o_e[mask]]); dm = np.array([pos[c] for c in d_e[mask]])
    flow = torch.zeros(a.N, a.N); flow[om, dm] = torch.tensor(fl[mask], dtype=torch.float32)
    flow = flow.reshape(-1)
    feat = torch.tensor(np.stack([np.log(jobs[top] + 1), np.log(res[top] + 1),
                                  (xy[:, 0] - xy[:, 0].mean()) / 1e4, (xy[:, 1] - xy[:, 1].mean()) / 1e4], 1), dtype=torch.float32)
    logjobs = torch.tensor(np.log(jobs[top] + 1), dtype=torch.float32)
    tier_props = torch.tensor(inc[top], dtype=torch.float32)
    otot = torch.zeros(a.N).index_add_(0, oi, flow)
    print(f"子集 N={a.N}, 观测对 {int((flow>0).sum())}, 总流 {float(flow.sum()):.0f}")

    def run(gravity):
        torch.manual_seed(0); m = ChoiceB(feat.shape[1], gravity=gravity)
        opt = torch.optim.Adam(m.parameters(), lr=0.03)
        for e in range(a.epochs):
            opt.zero_grad()
            A = m.attract(feat, nbr_idx, nbr_ptr, logjobs)
            logP = m.forward(A, t, oi, di, tier_props, a.N)
            loss = -(flow * logP).sum() / flow.sum(); loss.backward(); opt.step()
        with torch.no_grad():
            A = m.attract(feat, nbr_idx, nbr_ptr, logjobs)
            pred = m.forward(A, t, oi, di, tier_props, a.N).exp() * otot[oi]
            cpc = float(torch.minimum(pred, flow).sum() / flow.sum())
        return cpc, F.softplus(m.raw_alpha).tolist(), F.softplus(m.raw_beta).tolist()

    cg, ag, bg = run(True)
    print(f"\n对照 G(引力 log岗位): CPC={cg:.4f}  α[低中高]={[round(x,2) for x in ag]}  β={[round(x,2) for x in bg]}")
    cb, ab, bb = run(False)
    print(f"B(GNN学吸引力):       CPC={cb:.4f}  α[低中高]={[round(x,2) for x in ab]}  β={[round(x,2) for x in bb]}")
    print(f"\n判读: B 能训+CPC≈或>G -> 地基成立; α/β 分档不同 -> 异质偏好出来了; "
          f"ΔCPC(B-G)={cb-cg:+.4f}")


if __name__ == "__main__":
    main()
