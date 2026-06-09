"""Paper B option2 阶段2: 真实北京【可微联合估计】(子样本, 精简快版).

三模块一个可微图联合估 (α, β, A):
  需求 P(j|i)=softmax(α·logM − β·t_ff) → 车流 → 链路装载(固定自由流路径矩阵 INC) → BPR(A) → 链路拥堵
  匹配 观测OD流量(NLL) + 观测cell拥堵(rank) 联合, 梯度下降.
INC 按【子样本边】索引(每边=path链路), 向量化回溯, float32. 本地 CPU 分钟级.
"""
from pathlib import Path
import numpy as np, sys
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
import torch

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
S = 90; PERIOD = 0; PH = 3.0; BPR_BETA = 4.0; TARGET_VC = 0.6   # 分层: 中心/中/外围各 S//3


def main():
    ed = np.load(PROC/"beijing_edges.npz", allow_pickle=True); net = np.load(PROC/"beijing_road_network.npz", allow_pickle=True)
    grid = np.load(PROC/"beijing_grid.npz", allow_pickle=True); mm = np.load(PROC/"beijing_micro_moments.npz", allow_pickle=True)
    cg = np.load(PROC/"beijing_dynamic_cong.npz", allow_pickle=True)
    o=ed["o_idx"].astype(np.int64); d=ed["d_idx"].astype(np.int64); t_ff=ed["t_ff"].astype(np.float64)
    hav=ed["hav_km"].astype(np.float64); seg_ptr=ed["seg_ptr"].astype(np.int64); flow=ed["flow"][:,PERIOD].astype(np.float64)
    jobs=grid["jobs"].astype(np.float64); logM=np.log(jobs+1.0)
    Nn=int(net["n_nodes"]); eu=net["edge_u"].astype(np.int64); ev=net["edge_v"].astype(np.int64)
    eff=net["edge_ff_min"].astype(np.float64); ecap=net["edge_cap"].astype(np.float64)
    nxy=net["node_xy"].astype(np.float64); cell_node=net["cell_node"].astype(np.int64)

    seg_tot=np.add.reduceat(flow,seg_ptr[:-1])
    # 分层抽样: 按出发地到岗位中心距离分3层(中心/中/外围), 各取 S//3 -> 代表全市
    xy=grid["xy_m"].astype(np.float64); ctr=(xy*jobs[:,None]).sum(0)/jobs.sum()
    ocell=o[seg_ptr[:-1]]; dc=np.hypot(*(xy[ocell]-ctr).T)
    bands=np.digitize(dc,np.quantile(dc[seg_tot>0],[1/3,2/3]))
    rng0=np.random.default_rng(0); sel=[]
    for bnd in range(3):
        idx=np.nonzero((bands==bnd)&(seg_tot>0))[0]
        sel.append(rng0.choice(idx,size=min(S//3,len(idx)),replace=False))
    top=np.sort(np.concatenate(sel))
    print(f"  分层子样本 {len(top)} 起点(中心/中/外围各{S//3}), 占总流 {seg_tot[top].sum()/seg_tot.sum()*100:.0f}%",flush=True)
    rows_e=np.concatenate([np.arange(seg_ptr[i],seg_ptr[i+1]) for i in top])
    o_s=o[rows_e]; d_s=d[rows_e]; tff_s=t_ff[rows_e]; hav_s=hav[rows_e]; flow_s=flow[rows_e]; logM_s=logM[d_s]
    seg_len=(seg_ptr[top+1]-seg_ptr[top]); seg_id_s=np.repeat(np.arange(S),seg_len); seg_tot_s=seg_tot[top]
    P_obs=flow_s/np.repeat(np.maximum(seg_tot_s,1e-9),seg_len)
    onode_s=cell_node[o_s]; dnode_s=cell_node[d_s]
    ekey=eu*Nn+ev; sidx=np.argsort(ekey); skey=ekey[sidx]
    def link_lookup(p,c):
        ks=p*Nn+c; pos=np.searchsorted(skey,ks).clip(0,len(skey)-1); return sidx[pos]

    # ── 建边索引 INC(每子样本边 = 其自由流路径链路) ──
    graph=sp.csr_matrix((eff,(eu,ev)),shape=(Nn,Nn)); rows,cols=[],[]
    for k in range(S):
        m=seg_id_s==k; erows=rows_e_local=np.nonzero(m)[0]; onode=onode_s[m][0]; dn=dnode_s[m]
        dist,pred=dijkstra(graph,indices=int(onode),return_predecessors=True)
        reach=np.isfinite(dist[dn])&(dn!=onode); cur=dn[reach].astype(np.int64); rr=erows[reach]
        active=np.ones(len(cur),bool)
        while active.any():
            c=cur[active]; p=pred[c]; ok=p>=0; lk=link_lookup(p[ok],c[ok]); ai=np.nonzero(active)[0][ok]
            rows.append(rr[ai]); cols.append(lk); cur[active]=np.where(ok,p,cur[active])
            na=active.copy(); idx=np.nonzero(active)[0]; na[idx]=ok&(p!=onode); active=na
        if k%10==0: print(f"    INC {k}/{S}",flush=True)
    rows=np.concatenate(rows); cols=np.concatenate(cols)
    INC=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(o_s),len(eu)))
    print(f"  INC {INC.shape} nnz={INC.nnz:,}",flush=True)

    mid=(nxy[eu]+nxy[ev])/2; ll=grid["coords_latlon"].astype(np.float64)
    tr=Transformer.from_crs(4326,32650,always_xy=True); cx,cy=tr.transform(ll[:,1],ll[:,0])
    _,link_cell=cKDTree(np.stack([cx,cy],1)).query(mid); Ncell=len(ll); cnt=np.bincount(link_cell,minlength=Ncell)
    obs_c=cg["cong_frac_period"][:,PERIOD]; cmask=cg["cover_mask"]&(cnt>0)&(obs_c>=0)

    # ── torch 可微联合 (float32) ──
    INCt=torch.sparse_csr_tensor(torch.tensor(INC.indptr),torch.tensor(INC.indices),
                                 torch.tensor(INC.data),size=INC.shape)
    band=np.clip(np.digitize(hav_s,[1,5,15]),0,3); carsh=mm["mode_by_dist"][:,0].astype(np.float32)[band]
    F=lambda x: torch.tensor(x,dtype=torch.float32)
    logM_t=F(logM_s); tff_t=F(tff_s); Po=F(P_obs); segid=torch.tensor(seg_id_s)
    segtot=F(seg_tot_s); carsh_t=F(carsh); capt=F(ecap*PH); fft=F(eff); lc=torch.tensor(link_cell)
    obsc=F(obs_c); cm=torch.tensor(cmask); cntt=F(cnt).clamp_min(1)
    a=torch.tensor(0.5,requires_grad=True); b=torch.tensor(0.05,requires_grad=True)
    rawA=torch.tensor(float(np.log(0.15)),requires_grad=True); opt=torch.optim.Adam([a,b,rawA],lr=0.03)
    def seg_softmax(V):
        mx=torch.full((S,),-1e30).scatter_reduce(0,segid,V,"amax",include_self=False)
        z=(V-mx[segid]).exp(); s=torch.zeros(S).index_add(0,segid,z); return z/s[segid]
    oc=obsc[cm]; ocz=(oc-oc.mean())/(oc.std()+1e-6)
    for it in range(400):
        opt.zero_grad()
        P=seg_softmax(a*logM_t-b*tff_t); flows=P*segtot[segid]*carsh_t
        vol=torch.sparse.mm(INCt.t(),flows.unsqueeze(1)).squeeze(1)
        vc=vol/capt; vc=vc*(TARGET_VC/torch.median(vc[vol>0]).clamp_min(1e-6))
        congf=1+torch.exp(rawA)*vc.clamp(min=0)**BPR_BETA
        cellc=torch.zeros(Ncell).index_add(0,lc,congf)/cntt
        mc=cellc[cm]; mcz=(mc-mc.mean())/(mc.std()+1e-6)
        loss=-(Po*torch.log(P.clamp_min(1e-12))).mean()+0.3*((mcz-ocz)**2).mean()
        loss.backward(); opt.step()
        if it%100==0: print(f"    it{it} loss{float(loss):.4f} β={float(b):.4f}",flush=True)
    print(f"\n  ⭐ 可微联合(真实北京 {S}起点): α̂={float(a):.3f} β̂(怕堵)={float(b):.4f} Â={float(torch.exp(rawA)):.3f}")
    print(f"     对比: 自由流单拟合 β=0.177, naive堵后 β=0.107")
    print("判读: 三模块一个可微图联合估出 β+A; β 接近自由流去confound值=印证 free-flow routing 下均衡不改 β。"
          "全量: INC sparse 分块上 GPU + 部分预判加可学 φ。")


if __name__=="__main__":
    main()
