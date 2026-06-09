"""Paper B 个体 ABM (用户设计): ML→个体agent→自由流最短路→MBPR拥堵→真实出行时间。

- 从选择模型(去confound β=0.173)采样 N 个 agent: 各有 居住cell + 目的地(按选择概率) + 教育档 + 方式。
- 车 agent 在路网走自由流最短路(按OD-node聚合路由提速, 但agent个体保留属性/时间)。
- 所有车流压链路 → MBPR 拥堵 → 每个 agent 的【真实】出行时间。
- 个体级输出: 通勤时间分布(对~35min)+ 按教育档/拥堵延误。可加反事实(疏解)。
本地 CPU. 这是"个体粒度"(每个agent有属性/被堵的真实时间), 区别于纯流量级.
"""
from pathlib import Path
import numpy as np
from traffic_assignment import aon, PROC, PERIOD_HOURS, ALPHA, BETA, VC_CAP, TARGET_VC

BETA_PREF=0.173; ALPHA_PREF=0.535; PERIOD=0; N_AGENTS=200_000


def seg_softmax(V,seg_ptr):
    P=np.empty_like(V)
    for i in range(len(seg_ptr)-1):
        s,e=seg_ptr[i],seg_ptr[i+1]
        if e>s:
            v=V[s:e]-V[s:e].max(); ex=np.exp(v); P[s:e]=ex/ex.sum()
    return P


def main():
    ed=np.load(PROC/"beijing_edges.npz",allow_pickle=True); net=np.load(PROC/"beijing_road_network.npz",allow_pickle=True)
    grid=np.load(PROC/"beijing_grid.npz",allow_pickle=True); mm=np.load(PROC/"beijing_micro_moments.npz",allow_pickle=True)
    eh=np.load(PROC/"beijing_edu_hukou.npz",allow_pickle=True)
    o=ed["o_idx"].astype(np.int64); d=ed["d_idx"].astype(np.int64); t_ff=ed["t_ff"].astype(np.float64)
    hav=ed["hav_km"].astype(np.float64); seg_ptr=ed["seg_ptr"].astype(np.int64); flow=ed["flow"][:,PERIOD].astype(np.float64)
    jobs=grid["jobs"].astype(np.float64); logM=np.log(jobs+1.0); edu=eh["education_tier_props"]

    # ── ① 每条边的选择概率(去confound β) -> 采样 agent 的 (o,d) ──
    P=seg_softmax(ALPHA_PREF*logM[d]-BETA_PREF*t_ff, seg_ptr)
    seg_tot=np.add.reduceat(flow,seg_ptr[:-1])
    edge_w=P*np.repeat(seg_tot,np.diff(seg_ptr))                 # 每边期望人流
    edge_w=edge_w/edge_w.sum()
    rng=np.random.default_rng(0)
    aidx=rng.choice(len(o),size=N_AGENTS,p=edge_w)              # 每个 agent 选一条边(o->d)
    a_o=o[aidx]; a_d=d[aidx]; a_hav=hav[aidx]
    # agent 属性: 教育档(按居住cell构成抽) + 方式(按距离档)
    a_edu=np.array([rng.choice(3,p=edu[c]/edu[c].sum()) for c in a_o[:2000]])  # 抽样2000算分布(全量太慢)
    band=np.clip(np.digitize(a_hav,[1,5,15]),0,3); carp=mm["mode_by_dist"][:,0][band]
    a_car=rng.random(N_AGENTS)<carp                             # 是否开车
    print(f"  {N_AGENTS:,} agent; 车 {a_car.mean()*100:.0f}%; 居住cell {len(np.unique(a_o))}",flush=True)

    # ── ② 车 agent 压路网(按 OD-node 聚合路由) -> ③ MBPR 拥堵 ──
    Nn=int(net["n_nodes"]); eu=net["edge_u"].astype(np.int64); ev=net["edge_v"].astype(np.int64)
    eff=net["edge_ff_min"].astype(np.float64); ecap=net["edge_cap"].astype(np.float64); cell_node=net["cell_node"].astype(np.int64)
    car_o=cell_node[a_o[a_car]]; car_d=cell_node[a_d[a_car]]
    key=car_o*Nn+car_d; uk,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    o_of=uk//Nn; d_of=uk%Nn
    oo=np.unique(o_of); odem=np.bincount(np.searchsorted(oo,o_of),weights=cnt)
    order=np.argsort(-odem); cum=np.cumsum(odem[order])/odem.sum(); keep=oo[order[:np.searchsorted(cum,0.97)+1]]
    dmap={oi:(d_of[o_of==oi].astype(np.int64),cnt[o_of==oi].astype(np.float64)) for oi in keep}
    ekey=eu*Nn+ev; sidx=np.argsort(ekey); skey=ekey[sidx]
    print(f"  车 agent 分配中({len(keep)} 起点node)...",flush=True)
    lv=aon(eu,ev,eff.copy(),Nn,keep,dmap,np.arange(Nn),skey,sidx)
    vc=lv/(ecap*PERIOD_HOURS[PERIOD]+1e-9); vc*=TARGET_VC/max(np.median(vc[vc>0]),1e-6)
    cong=1+ALPHA*np.minimum(vc,VC_CAP)**BETA                    # 每链路 t_cong/t_ff

    # ── 每个 agent 的真实出行时间(自由流时间 × 其目的地周边拥堵) ──
    from scipy.spatial import cKDTree
    from pyproj import Transformer
    mid=(net["node_xy"][eu]+net["node_xy"][ev])/2; ll=grid["coords_latlon"]
    tr=Transformer.from_crs(4326,32650,always_xy=True); cx,cy=tr.transform(ll[:,1],ll[:,0])
    _,lc=cKDTree(np.stack([cx,cy],1)).query(mid); Nc=len(ll); ck=np.bincount(lc,minlength=Nc)
    cong_cell=np.bincount(lc,weights=cong,minlength=Nc)/np.maximum(ck,1); cong_cell[ck==0]=np.median(cong)
    a_tff=t_ff[aidx]; a_treal=a_tff.copy()
    a_treal[a_car]=a_tff[a_car]*cong_cell[a_d[a_car]]          # 车 agent 挨堵; 非车按自由流近似
    print(f"\n  ⭐ 个体 ABM 输出:")
    print(f"    自由流通勤 中位 {np.median(a_tff):.1f}min; 真实(挨堵)中位 {np.median(a_treal):.1f}min (观测~35)")
    print(f"    车 agent 拥堵延误 中位 {np.median((a_treal/a_tff)[a_car]):.2f}x")
    print(f"    教育档(2000抽样)通勤: 低{a_tff[:2000][a_edu==0].mean():.0f} 中{a_tff[:2000][a_edu==1].mean():.0f} 高{a_tff[:2000][a_edu==2].mean():.0f}min")
    np.savez(PROC.parent/"paperB_abm_agents.npz",o=a_o,d=a_d,car=a_car,t_ff=a_tff,t_real=a_treal)
    print(f"  [OK] 存 data/paperB_abm_agents.npz ({N_AGENTS} agent)")
    print("  这是个体 ABM 骨架: 每 agent 有 居住/去向/方式/真实挨堵时间. 下一步可加反事实(疏解重采目的地)+ 教育/户口属性全量.")


if __name__=="__main__":
    main()
