# -*- coding: utf-8 -*-
"""Stage B v1: differentiable fixed-route congestion equilibrium (skeleton).

Design (see chapter 5.5):
  * Routes are FIXED at the free-flow shortest path -> a sparse path-link incidence
    Delta [OD x L]. Only travel TIMES respond to congestion, so the whole loop is
    smooth and the argmin (shortest path) is OUT of the differentiable region.
  * Fixed point is solved in LINK space (dim ~|L|, thousands), not OD space (millions):
        t = BPR( Delta^T @ demand(Delta @ t ; beta) ; ff, cap )
  * Gradient through the fixed point: 'unroll' (truncated backprop, simple/correct) or
    'implicit' (DEQ-style adjoint via a short Neumann series, memory-O(1)).
  * demand(tt_od; beta) is the HOOK where the Stage-A behavioural core plugs in
    (travel times -> car flows per OD). A smooth surrogate is used for the self-test.

Run the self-test (no data needed):
    python experiments/diff_equilibrium.py --selftest
"""
import argparse
import numpy as np
import torch

ALPHA, BETA_POW, VC_CAP = 0.15, 4.0, 3.0   # BPR (matches traffic_assignment.py)


# ----------------------------------------------------------------------------- BPR
def bpr(x, ff, cap, hours, alpha=ALPHA, beta_pow=BETA_POW, vc_cap=VC_CAP, scale=1.0):
    """Congested link time. Differentiable; the cap is a (sub-differentiable) clamp."""
    vc = (x * scale) / (cap * hours + 1e-9)
    vc = torch.clamp(vc, max=vc_cap)
    return ff * (1.0 + alpha * vc.pow(beta_pow))


# --------------------------------------------------------------- one equilibrium step
def make_step(Delta, ff, cap, hours, demand_fn, scale=1.0):
    """Return F: link_times -> link_times, the map whose fixed point is the equilibrium."""
    DeltaT = Delta.t().coalesce()

    def F(t):
        tt_od = torch.sparse.mm(Delta, t.unsqueeze(1)).squeeze(1)   # OD travel times
        f_od = demand_fn(tt_od)                                     # car flow per OD (Stage A hook)
        x = torch.sparse.mm(DeltaT, f_od.unsqueeze(1)).squeeze(1)   # link volumes
        return bpr(x, ff, cap, hours, scale=scale)
    return F


def solve_fixed_point(F, t0, n_iter=800, eta=0.1, tol=1e-6, verbose=False, method="damped"):
    """Fixed-point solve. method='damped' (constant step, fine for gentle maps) or
    'msa' (Method of Successive Averages, step 1/(k+1) — provably convergent for the
    stiff power-BPR maps where constant damping oscillates)."""
    t = t0
    for k in range(n_iter):
        step = (1.0 / (k + 2)) if method == "msa" else eta
        t_new = (1 - step) * t + step * F(t)
        rel = (t_new - t).norm() / (t.norm() + 1e-9)
        t = t_new
        if verbose and k % 50 == 0:
            print(f"    iter {k:4d}  rel={rel.item():.2e}")
        if rel < tol:
            break
    return t, k + 1


def equilibrium_times(F, t_init, grad_mode="unroll", n_iter=800, eta=0.1,
                      tol=1e-6, unroll_last=120, neumann=40, method="damped", adj_eta=None):
    """Solve t* = F(t*) and return t* with a usable gradient w.r.t. F's parameters."""
    if grad_mode == "unroll":
        # Phase 1: converge without grad; Phase 2: a few iters WITH grad (truncated backprop).
        with torch.no_grad():
            t, _ = solve_fixed_point(F, t_init, n_iter=max(0, n_iter - unroll_last), eta=eta, tol=tol, method=method)
        for _ in range(unroll_last):
            t = (1 - eta) * t + eta * F(t)
        return t

    elif grad_mode == "implicit":
        # DEQ-style: detached fixed point t*, then implicit-function gradient on t* = F(t*).
        # Adjoint solves (I - J^T) u = dL/dt* by GMRES (robust to non-contractive J, unlike
        # a Neumann series), where J = dF/dt* and J^T v is a vector-Jacobian product.
        with torch.no_grad():
            t_star, _ = solve_fixed_point(F, t_init, n_iter=n_iter, eta=eta, tol=tol, method=method)
        t_star = t_star.detach().requires_grad_(True)
        return _implicit_gmres(F, t_star, maxiter=neumann, tol=tol)

    else:
        raise ValueError(grad_mode)


def _implicit_gmres(F, t_star, maxiter=200, tol=1e-6):
    """Return F(t*) but with the implicit-function gradient: backward solves
    (I - J^T) u = grad_out via GMRES (J = dF/dt*)."""
    import scipy.sparse.linalg as spla
    f1 = F(t_star)                                       # differentiable graph for J^T vjp
    n = t_star.numel()

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f1_in):
            return f1_in.detach()

        @staticmethod
        def backward(ctx, g):
            def matvec(v):
                vt = torch.as_tensor(v, dtype=t_star.dtype, device=t_star.device)
                Jtv, = torch.autograd.grad(f1, t_star, vt, retain_graph=True)
                return v - Jtv.detach().cpu().numpy()     # (I - J^T) v
            A = spla.LinearOperator((n, n), matvec=matvec)
            g_np = g.detach().cpu().numpy().astype(np.float64)
            try:
                u, info = spla.gmres(A, g_np, rtol=tol, atol=0.0, maxiter=maxiter, restart=50)
            except TypeError:                             # older scipy uses 'tol'
                u, info = spla.gmres(A, g_np, tol=tol, atol=0.0, maxiter=maxiter, restart=50)
            return torch.as_tensor(u, dtype=g.dtype, device=g.device)
    return _F.apply(f1)


# ----------------------------------------------------------------------------- Delta builder
def build_delta_from_tree(n_nodes, eu, ev, ff_links, origins, dests_per_origin):
    """Free-flow shortest-path tree per origin -> sparse path-link incidence Delta [OD x L].
    Pure precompute (no grad). For full Beijing run this on the kept (97% flow) origins.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    G = sp.csr_matrix((ff_links, (eu, ev)), shape=(n_nodes, n_nodes))
    link_id = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(eu, ev))}
    rows, cols, od_pairs = [], [], []
    r = 0
    for oi in origins:
        dist, pred = dijkstra(G, indices=int(oi), return_predecessors=True)
        for dj in dests_per_origin[oi]:
            if dj == oi or not np.isfinite(dist[dj]):
                continue
            nd = int(dj)
            while pred[nd] >= 0:
                p = int(pred[nd])
                lid = link_id.get((p, nd))
                if lid is not None:
                    rows.append(r); cols.append(lid)
                nd = p
            od_pairs.append((int(oi), int(dj)))
            r += 1
    L = len(eu)
    Delta = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, L))
    return Delta, od_pairs


def scipy_to_torch_sparse(M, dtype=torch.float64):
    Mc = M.tocoo()
    idx = torch.tensor(np.vstack([Mc.row, Mc.col]), dtype=torch.long)
    val = torch.tensor(Mc.data, dtype=dtype)
    return torch.sparse_coo_tensor(idx, val, M.shape).coalesce()


# ----------------------------------------------------------------------------- self-test
def selftest():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    print("[selftest] building a synthetic geometric network ...")
    n_nodes = 60
    xy = rng.random((n_nodes, 2))
    from scipy.spatial import cKDTree
    _, nbr = cKDTree(xy).query(xy, k=5)
    eu, ev, ff = [], [], []
    for u in range(n_nodes):
        for v in nbr[u, 1:]:
            d = float(np.linalg.norm(xy[u] - xy[v]))
            eu += [u, v]; ev += [v, u]; ff += [d * 10, d * 10]   # symmetric, ff in 'minutes'
    eu = np.array(eu); ev = np.array(ev); ff = np.array(ff)
    L = len(eu)
    cap = np.full(L, 1.0)
    TRIPS = 4.0                                          # trips per OD -> V/C ~1.2 (moderate congestion, contractive)

    origins = list(range(0, n_nodes, 6))                 # ~10 origins
    dests = {o: list(rng.choice(n_nodes, 8, replace=False)) for o in origins}
    Delta_sp, od = build_delta_from_tree(n_nodes, eu, ev, ff, origins, dests)
    print(f"[selftest] OD pairs={len(od)}  links={L}  Delta nnz={Delta_sp.nnz}")

    Delta = scipy_to_torch_sparse(Delta_sp)
    ffT = torch.tensor(ff); capT = torch.tensor(cap); hours = 3.0
    N_o = torch.tensor([200.0] * len(od))                # per-OD origin scale (toy)
    od_origin = torch.tensor([o for (o, d) in od])
    uniq_o = torch.unique(od_origin)

    def make_demand(beta_raw):
        """Toy Stage-A surrogate: per-origin softmax over -|beta|*tt -> car flow.
        beta_raw is the unconstrained param; true coefficient = -softplus(beta_raw) (time disutility)."""
        beta = -torch.nn.functional.softplus(beta_raw)
        def demand(tt_od):
            v = beta * tt_od
            f = torch.zeros_like(tt_od)
            for o in uniq_o:
                m = od_origin == o
                w = torch.softmax(v[m], dim=0)
                f = f.index_add(0, torch.nonzero(m).squeeze(1), w * TRIPS)
            return f
        return demand

    # ---- generate a 'true' equilibrium as the estimation target ----
    beta_true = torch.tensor(0.5, dtype=torch.float64)
    F_true = make_step(Delta, ffT, capT, hours, make_demand(beta_true))
    with torch.no_grad():
        t_star_true, it = solve_fixed_point(F_true, ffT.clone(), verbose=True)
    tt_target = torch.sparse.mm(Delta, t_star_true.unsqueeze(1)).squeeze(1).detach()
    print(f"[selftest] true equilibrium reached in {it} iters; mean delay x{(t_star_true/ffT).mean():.3f}")

    # ---- loss as a function of an estimated beta_raw ----
    def loss_of(beta_raw, grad_mode):
        F = make_step(Delta, ffT, capT, hours, make_demand(beta_raw))
        t_star = equilibrium_times(F, ffT.clone(), grad_mode=grad_mode)
        tt = torch.sparse.mm(Delta, t_star.unsqueeze(1)).squeeze(1)
        return ((tt - tt_target) ** 2).mean()

    for mode in ("unroll", "implicit"):
        b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        L_ = loss_of(b, mode); L_.backward()
        g_auto = b.grad.item()
        eps = 1e-4
        with torch.no_grad():
            gp = loss_of(torch.tensor(0.1 + eps, dtype=torch.float64), "unroll").item()
            gm = loss_of(torch.tensor(0.1 - eps, dtype=torch.float64), "unroll").item()
        g_fd = (gp - gm) / (2 * eps)
        rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-12)
        print(f"[selftest] grad_mode={mode:8s}  autograd={g_auto:+.5e}  finite-diff={g_fd:+.5e}  rel.err={rel:.2e}  "
              f"{'PASS' if rel < 1e-2 else 'CHECK'}")

    # ---- tiny recovery demo: can we move beta_raw toward beta_true by gradient? ----
    b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([b], lr=0.1)
    for step in range(40):
        opt.zero_grad(); L_ = loss_of(b, "unroll"); L_.backward(); opt.step()
    print(f"[selftest] recovered -softplus(beta)={-torch.nn.functional.softplus(b).item():+.3f} "
          f"(target {-torch.nn.functional.softplus(beta_true).item():+.3f}), final loss={L_.item():.3e}")
    print("[selftest] done.")


PERIOD_HOURS = [3.0, 3.0, 6.0, 8.0]
TARGET_VC = 0.45


def run_real(period=0, n_origins=40, dests_per_origin=120, grad_mode="unroll", estimate=False):
    """End-to-end on the REAL Beijing OSM network at smoke scale, with a real-data
    congestion-responsive demand. (The full Stage-A demand replaces make_demand below.)"""
    import pathlib
    P = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
    net = np.load(P / "beijing_road_network.npz", allow_pickle=True)
    Nn = int(net["n_nodes"]); eu = net["edge_u"].astype(np.int64); ev = net["edge_v"].astype(np.int64)
    ff = net["edge_ff_min"].astype(np.float64); cap = net["edge_cap"].astype(np.float64)
    cell_node = net["cell_node"].astype(np.int64); hours = PERIOD_HOURS[period]

    ed = np.load(P / "beijing_edges.npz", allow_pickle=True)
    seg = ed["seg_ptr"].astype(np.int64); oids = ed["origin_ids"].astype(np.int64)
    d_idx = ed["d_idx"]; hav = ed["hav_km"]; t_obs = ed["t_obs"].astype(np.float64)
    flow = ed["flow"][:, period].astype(np.float64)
    mm = np.load(P / "beijing_micro_moments.npz", allow_pickle=True)
    car_share = mm["mode_by_dist"][:, 0].astype(np.float64); bands = mm["dist_bands"]
    car_all = flow * car_share[np.clip(np.digitize(hav, bands), 0, 3)]

    otot = np.add.reduceat(car_all, seg[:-1])
    keep = np.argsort(-otot)[:n_origins]
    base, ttw, ttd, dests = {}, {}, {}, {}
    for i in keep:
        s, e = int(seg[i]), int(seg[i + 1]); onode = int(cell_node[oids[i]])
        fcar = car_all[s:e]; dcell = d_idx[s:e]; tob = t_obs[s:e]
        top = np.argsort(-fcar)[:dests_per_origin]
        dn = cell_node[dcell[top]]
        for k, dd in zip(top, dn):
            key = (onode, int(dd))
            base[key] = base.get(key, 0.0) + float(fcar[k])
            ttw[key] = ttw.get(key, 0.0) + float(fcar[k] * tob[k]); ttd[key] = ttd.get(key, 0.0) + float(fcar[k])
        dests.setdefault(onode, set()).update(int(x) for x in dn)
    origins = list(dests); dests = {o: list(dests[o]) for o in origins}
    print(f"[real] period {period}: kept {len(origins)} origin-nodes; building Delta on OSM net ...")
    Delta_sp, od = build_delta_from_tree(Nn, eu, ev, ff, origins, dests)
    print(f"[real] OD node-pairs={len(od)}  links={len(eu)}  Delta nnz={Delta_sp.nnz}")

    base_arr = np.array([base.get(p, 1e-9) for p in od])
    ttobs_arr = np.array([ttw[p] / ttd[p] if ttd.get(p, 0) > 0 else 0.0 for p in od])
    onode_of = np.array([p[0] for p in od])
    uniq, grp = np.unique(onode_of, return_inverse=True)
    No = np.zeros(len(uniq)); np.add.at(No, grp, base_arr)             # origin totals -> baseline reproduces base

    Delta = scipy_to_torch_sparse(Delta_sp); DeltaT = Delta.t().coalesce()
    ffT = torch.tensor(ff); capT = torch.tensor(cap)
    a_od = torch.tensor(np.log(base_arr + 1e-9)); ttobs = torch.tensor(ttobs_arr)
    grpT = torch.tensor(grp); NoT = torch.tensor(No)

    def make_demand(braw):
        bcoef = torch.nn.functional.softplus(braw)                    # >0 time-response
        def demand(tt_od):
            v = a_od + bcoef * (ttobs - tt_od)                        # tt=ttobs -> reproduces observed split
            m = v.max()
            ex = torch.exp(v - m); denom = torch.zeros(len(NoT), dtype=ex.dtype).index_add(0, grpT, ex)
            return NoT[grpT] * ex / denom[grpT]
        return demand

    # calibrate a fixed congestion scale (free-flow assignment -> median V/C = TARGET_VC), like traffic_assignment.py
    with torch.no_grad():
        x0 = torch.sparse.mm(DeltaT, torch.tensor(base_arr).unsqueeze(1)).squeeze(1)
        vc0 = (x0 / (capT * hours + 1e-9)).numpy()
        scale = float(TARGET_VC / max(np.median(vc0[vc0 > 0]), 1e-6))
    print(f"[real] congestion scale = {scale:.2f} (free-flow median V/C -> {TARGET_VC})")

    if estimate:
        import torch.nn.functional as Fnn
        print("[real] === estimation loop: recover a known coefficient through the real-network equilibrium ===")
        NIT = 2000
        with torch.no_grad():                            # generate a target equilibrium from a 'true' coef
            Ftru = make_step(Delta, ffT, capT, hours, make_demand(torch.tensor(0.6, dtype=torch.float64)), scale=scale)
            t_tru, _ = solve_fixed_point(Ftru, ffT.clone(), n_iter=NIT, tol=1e-6, method="msa")
            tt_target = torch.sparse.mm(Delta, t_tru.unsqueeze(1)).squeeze(1).detach()
        b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([b], lr=0.05)             # lower lr -> monotone (lr=0.15 overshoots)
        best = (1e9, None)
        for s in range(16):
            opt.zero_grad()
            F = make_step(Delta, ffT, capT, hours, make_demand(b), scale=scale)
            t_g = equilibrium_times(F, ffT.clone(), grad_mode="implicit", method="msa",
                                    n_iter=NIT, eta=0.1, neumann=150, tol=1e-6)
            tt = torch.sparse.mm(Delta, t_g.unsqueeze(1)).squeeze(1)
            loss = ((tt - tt_target) ** 2).mean()
            cur = Fnn.softplus(b).item()
            if loss.item() < best[0]:
                best = (loss.item(), cur)
            loss.backward(); opt.step()
            print(f"    step {s:2d}  coef={cur:.3f}  loss={loss.item():.3e}")
        print(f"[real] estimation done: best-loss coef={best[1]:.3f} (loss {best[0]:.2e}); "
              f"true {Fnn.softplus(torch.tensor(0.6)).item():.3f}")
        return

    NIT = 4000

    def loss_val(braw):                                   # no-grad scalar loss via MSA forward
        with torch.no_grad():
            F = make_step(Delta, ffT, capT, hours, make_demand(braw), scale=scale)
            t_s, _ = solve_fixed_point(F, ffT.clone(), n_iter=NIT, tol=1e-6, method="msa")
            tt = torch.sparse.mm(Delta, t_s.unsqueeze(1)).squeeze(1)
            return ((tt - ttobs) ** 2).mean().item(), t_s

    # forward equilibrium sanity (MSA -- provably convergent for stiff BPR)
    L0, t_star = loss_val(torch.tensor(0.3, dtype=torch.float64))
    with torch.no_grad():
        F0 = make_step(Delta, ffT, capT, hours, make_demand(torch.tensor(0.3, dtype=torch.float64)), scale=scale)
        rel = (F0(t_star) - t_star).norm().item() / (t_star.norm().item() + 1e-9)
    used = (t_star > ffT + 1e-9)
    print(f"[real] MSA equilibrium: residual={rel:.1e} {'(converged)' if rel < 1e-4 else '(NOT converged)'}; "
          f"congested links {int(used.sum())}; mean delay x{(t_star[used]/ffT[used]).mean().item():.3f}  "
          f"max x{(t_star/ffT).max().item():.2f}")

    # differentiable gradient through the real-network equilibrium (implicit adjoint)
    b = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    F = make_step(Delta, ffT, capT, hours, make_demand(b), scale=scale)
    t_g = equilibrium_times(F, ffT.clone(), grad_mode="implicit", method="msa",
                            n_iter=NIT, eta=0.1, adj_eta=0.02, neumann=200, tol=1e-6)
    L_ = ((torch.sparse.mm(Delta, t_g.unsqueeze(1)).squeeze(1) - ttobs) ** 2).mean()
    L_.backward(); g_auto = b.grad.item()
    eps = 1e-3
    gp, _ = loss_val(torch.tensor(0.3 + eps, dtype=torch.float64))
    gm_, _ = loss_val(torch.tensor(0.3 - eps, dtype=torch.float64))
    g_fd = (gp - gm_) / (2 * eps); rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-12)
    ok = (rel < 1e-1) and (abs(g_auto) < 1e6)
    print(f"[real] implicit grad (GMRES adjoint): autograd={g_auto:+.4e}  finite-diff={g_fd:+.4e}  rel.err={rel:.2e}  "
          f"{'PASS' if ok else 'CHECK'}  (residual {rel < 5e-2 and 'tight' or 'limited by forward tol ~1e-3'})")
    print("[real] done. (swap make_demand for the Stage-A forward; freeze GNN; add flow-CE loss.)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--period", type=int, default=0)
    ap.add_argument("--n-origins", type=int, default=40)
    ap.add_argument("--estimate", action="store_true", help="run the Stage-B estimation loop on the real network")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.real:
        run_real(period=a.period, n_origins=a.n_origins, estimate=a.estimate)
    else:
        print("Use --selftest (synthetic) or --real (real OSM network smoke).")


if __name__ == "__main__":
    main()
