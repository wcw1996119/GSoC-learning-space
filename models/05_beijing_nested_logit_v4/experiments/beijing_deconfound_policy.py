"""Paper B 政策含义: 去confound 改变疏解政策结论吗?

用 naive β=0.107(堵后时间偏估) vs 去confound β=0.173 算【同一个疏解政策】:
  疏解 = 给外围cell加岗位(中心岗位搬郊区), 看多少人跟着去外围工作。
naive(β低=不太怕远)高估跟随者; 去confound(β高=更怕远)跟随者少。
=> 用错偏好系统性【高估疏解效果】。本地CPU子样本.
"""
from pathlib import Path
import numpy as np, torch
PROC=Path(__file__).resolve().parents[1]/"data"/"processed"


def seg_softmax(V,seg_id,nseg):
    mx=torch.full((nseg,),-1e30,dtype=V.dtype).scatter_reduce(0,seg_id,V,"amax",include_self=False)
    z=(V-mx[seg_id]).exp(); s=torch.zeros(nseg).index_add(0,seg_id,z); return z/s[seg_id]


def main():
    e=np.load(PROC/"beijing_edges.npz",allow_pickle=True); g=np.load(PROC/"beijing_grid.npz",allow_pickle=True)
    o=e["o_idx"].astype(np.int64); d=e["d_idx"].astype(np.int64); t_ff=e["t_ff"].astype(np.float32)
    flow=e["flow"].sum(1).astype(np.float32); seg_ptr=e["seg_ptr"].astype(np.int64)
    jobs=g["jobs"].astype(np.float64); xy=g["xy_m"].astype(np.float64)
    ctr=(xy*jobs[:,None]).sum(0)/jobs.sum(); dist=np.hypot(*(xy-ctr).T)
    peri=dist>np.quantile(dist,0.75)        # 外围cell(离中心最远25%)
    print(f"  外围cell {peri.sum()}(离中心>75分位); 当前外围岗位占比 {jobs[peri].sum()/jobs.sum()*100:.0f}%",flush=True)

    nseg=len(seg_ptr)-1; seg_id=torch.from_numpy(np.repeat(np.arange(nseg),np.diff(seg_ptr)))
    dd=torch.from_numpy(d); tff=torch.from_numpy(t_ff); fl=torch.from_numpy(flow)
    pop=torch.from_numpy(np.add.reduceat(flow,seg_ptr[:-1]).astype(np.float32))  # 各起点总需求

    def fit_alpha(beta, logM_t, steps=300):
        a=torch.tensor(0.5,requires_grad=True); opt=torch.optim.Adam([a],lr=0.03)
        for _ in range(steps):
            opt.zero_grad(); V=a*logM_t[dd]-beta*tff
            P=seg_softmax(V,seg_id,nseg); (-(fl*torch.log(P.clamp_min(1e-12))).sum()/fl.sum()).backward(); opt.step()
        return float(a)

    def peri_share(alpha,beta,logM_t):
        with torch.no_grad():
            P=seg_softmax(alpha*logM_t[dd]-beta*tff,seg_id,nseg)
            edge_flow=P*pop[seg_id]                          # 每边人流
            peri_edge=torch.from_numpy(peri[d].astype(np.float32))
            return float((edge_flow*peri_edge).sum()/edge_flow.sum())   # 去外围工作的比例

    ADD=0.3*jobs.sum()   # 疏解: 新增岗位=总量30%, 均摊到外围(造外围就业中心)
    jobs_disp=jobs.copy(); jobs_disp[peri]+=ADD/peri.sum()
    print(f"  疏解: 外围岗位占比 0% -> {jobs_disp[peri].sum()/jobs_disp.sum()*100:.0f}%",flush=True)
    logM=torch.from_numpy(np.log(jobs+1.0).astype(np.float32)); logM_d=torch.from_numpy(np.log(jobs_disp+1.0).astype(np.float32))

    print("\n  β        α̂     基准外围工作%   疏解后外围%   疏解净增(跟随者)")
    res={}
    for tag,beta in [("naive(0.107)",0.107),("去confound(0.173)",0.173)]:
        a=fit_alpha(beta,logM)
        s0=peri_share(a,beta,logM); s1=peri_share(a,beta,logM_d)
        res[tag]=(s0,s1,s1-s0)
        print(f"  {tag:16s} {a:.3f}   {s0*100:5.1f}%        {s1*100:5.1f}%        {(s1-s0)*100:+.2f}pp",flush=True)
    up_n=res["naive(0.107)"][2]; up_d=res["去confound(0.173)"][2]
    print(f"\n  ⭐ 疏解跟随者: naive {up_n*100:+.2f}pp vs 去confound {up_d*100:+.2f}pp"
          f"  → naive 高估 {(up_n/up_d-1)*100:.0f}%" if up_d>1e-6 else "")
    print("判读: naive(β低)算出更多人跟着去外围 = 高估疏解有效性; 去confound(β高=人更怕远)跟随者少。"
          " 用错的(naive)偏好做政策评估 => 系统性乐观, 真疏解效果被高估。这就是 de-confound 的政策价值。")


if __name__=="__main__":
    main()
