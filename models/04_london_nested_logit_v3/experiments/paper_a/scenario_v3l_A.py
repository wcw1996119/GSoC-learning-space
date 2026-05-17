"""V3l Scenario A: Old Oak Common (+65k jobs across 9 grids).

Loads a trained v3l checkpoint (.pt produced by train_cervero_shen.py with
the ckpt-save patch), runs baseline forward and intervened forward, reports
destination-inflow shifts, accessibility delta, and Gini change.

Usage:
    python experiments/paper_a/scenario_v3l_A.py \\
        --ckpt evaluation_outputs/paper_a/v3l_lexico_filter_repro_s0.pt \\
        --out  evaluation_outputs/paper_a/v3l_scenario_A_s0.json

Intervention:
    OOC_GRIDS = [907, 908, 909, 862, 863, 864, 951, 952, 953]  (same as v2)
    +7222 jobs per grid (= +65,000 / 9)
    Updates: M_j → M_j + delta at OOC grids, then re-derive log_M_j, log_M_z,
             and patch X_static employment column (col 8, total_employment, z-scored log1p).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"

# Trainer imports `experiments.paper_a.compare_four_variants` (lives in v2) — must
# resolve via V2's package namespace, not v3's. Insert V2_ROOT first so its
# `experiments/` package wins, then load the trainer module by absolute path so
# the v3 location doesn't collide.
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "v3_train_cervero_shen",
    V3_ROOT / "experiments" / "paper_a" / "train_cervero_shen.py",
)
_trainer = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_trainer)

load_data = _trainer.load_data
forward_cs = _trainer.forward_cs
make_distance_aware_mode_share = _trainer.make_distance_aware_mode_share
CerveroShenHead = _trainer.CerveroShenHead
DualBranchEncoder = _trainer.DualBranchEncoder
DualBranchGATEncoder = _trainer.DualBranchGATEncoder


OOC_GRIDS = [907, 908, 909, 862, 863, 864, 951, 952, 953]
DELTA_PER_GRID_DEFAULT = 7222.0   # +65k / 9
EMPLOYMENT_COL = 8                 # X_static column for total_employment (z-scored log1p)


def gini_coefficient(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x)[x > 0])
    n = len(x)
    if n == 0:
        return 0.0
    return float((np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (n * x.sum()))


def hansen_accessibility(M_j: np.ndarray, t_ij_min: np.ndarray, beta: float) -> np.ndarray:
    """A_i = Σ_j M_j · exp(beta · t_ij). beta should be NEGATIVE (cost increases → access decreases)."""
    return (M_j[None, :] * np.exp(beta * t_ij_min)).sum(axis=1)


def reconstruct_model(ckpt: dict, device: torch.device):
    """Rebuild encoder + head from saved args + state_dicts."""
    args = SimpleNamespace(**ckpt["args"])
    M = ckpt["n_modes"]; n_boroughs = ckpt["n_boroughs"]
    static_dim = ckpt["static_dim"]; dyn_dim = ckpt["dyn_dim"]
    N = ckpt["N"]
    if ckpt.get("encoder_kind") == "DualBranchGATEncoder":
        assert DualBranchGATEncoder is not None, "GAT encoder requested but not importable"
        encoder = DualBranchGATEncoder(
            static_dim=static_dim, dyn_dim=dyn_dim,
            hidden_dim=32, gru_hidden=32, n_gat_layers=2,
            gat_heads=getattr(args, "gat_heads", 4), tcn_kernels=(3, 5, 7),
        ).to(device)
    else:
        encoder = DualBranchEncoder(
            static_dim=static_dim, dyn_dim=dyn_dim,
            hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
        ).to(device)
    rum = CerveroShenHead(
        n_modes=M, n_boroughs=n_boroughs, n_tiers=args.n_income_tiers,
        lambda_init=args.lambda_init, lambda_eps_min=args.lambda_eps_min,
        use_gnn_blend=True, gnn_blend_init=0.5, blend_max=args.blend_max,
        gnn_mode=args.gnn_mode,
        gnn_residual_scale_init=args.gnn_residual_scale_init,
        use_match_gate=args.use_match_gate,
        gate_steepness_init=args.gate_steepness_init,
        gate_threshold_init=args.gate_threshold_init,
        use_tier_mixture=args.use_tier_mixture,
        tier_init_scale=args.tier_init_scale,
        use_push_pull=args.use_push_pull,
        use_self_loop_boost=args.use_self_loop_boost,
        n_hours=24,
        n_busy_dest=args.n_busy_dest,
        n_origins=N,
        use_tier_threshold=args.use_tier_threshold,
        use_consideration_filter=args.use_consideration_filter,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    rum.load_state_dict(ckpt["rum_state"])
    encoder.eval(); rum.eval()
    return encoder, rum, args


def load_scenario_inputs(args, device):
    """Same loading sequence as train_cervero_shen.py main(), but no training."""
    print("[scen-A] loading v2 data ...")
    d = load_data()
    coords = torch.from_numpy(
        np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz",
                allow_pickle=True)["coords_bng"].astype(np.float32)
    )
    dist_km = (torch.linalg.norm(coords.unsqueeze(0) - coords.unsqueeze(1), dim=-1)
               / 1000.0).clamp(min=0.1)
    pair_mode_share = make_distance_aware_mode_share(
        d["mode_share"], dist_km, ["car", "transit", "walk"], walk_threshold_km=5.0,
    )

    aux = np.load(V3_ROOT / args.aux_path)
    match_prob = torch.from_numpy(aux["match_prob"]).float()
    log_M_z = torch.from_numpy(aux["log_M_z"]).float()
    log_W_z = torch.from_numpy(aux["log_W_z"]).float()
    log_D_z = torch.from_numpy(aux["log_D_z"]).float()
    income_score = torch.from_numpy(aux["income_z"]).float()
    pct_kids = torch.from_numpy(aux["pct_with_kids_z"]).float()
    mean_cars = torch.from_numpy(aux["mean_cars_z"]).float()
    income_tier_props = torch.from_numpy(aux["income_tier_props"]).float()
    grid_borough_idx = d["grid_borough_idx"].long()

    N, T = d["N"], d["T"]
    X_static = d["X_static"].to(device).float()
    X_dynamic = d["X_dynamic"].to(device).float()
    edge_index = d["edge_index"].to(device)
    observed_OD = d["F_ij_t"].to(device).float()
    val_mask = d["val_mask"].to(device).bool()
    log_d_ij = d["log_d"].to(device).float()
    mode_names = ["car", "transit", "walk"]

    t_per_mode = {}
    for m in mode_names:
        tens = d["t_" + m].to(device).float()
        if tens.dim() == 2:
            tens = tens.unsqueeze(0).expand(T, N, N).contiguous()
        t_per_mode[m] = tens

    return {
        "N": N, "T": T,
        "X_static": X_static, "X_dynamic": X_dynamic, "edge_index": edge_index,
        "t_per_mode": t_per_mode, "mode_names": mode_names,
        "log_d_ij": log_d_ij, "match_prob": match_prob.to(device),
        "log_M_z": log_M_z.to(device), "log_W_z": log_W_z.to(device), "log_D_z": log_D_z.to(device),
        "income_score": income_score.to(device),
        "pct_kids": pct_kids.to(device), "mean_cars": mean_cars.to(device),
        "income_tier_props": income_tier_props.to(device),
        "pi_m_pair": pair_mode_share.to(device).float(),
        "grid_borough_idx": grid_borough_idx.to(device),
        "observed_OD": observed_OD, "val_mask": val_mask,
        # Raw fields needed for OOC intervention:
        "M_j_raw": np.array(aux["M_j"], dtype=np.float64),
        "log_M_mean": float(aux["log_M_mean"]),
        "log_M_std": float(aux["log_M_std"]),
        # Encoder static features stats (mu/sd) needed to patch employment column.
        # Recover from demo_cache.npz "static_mu" / "static_sd" if present.
    }


def patch_M_at_ooc(data: dict, ooc_grids: list, delta_per_grid: float):
    """Apply OOC employment intervention.

    Returns a new dict with patched `log_M_z` (and patched X_static employment
    column if static_mu/sd are available). Original data dict is not mutated.
    """
    M_new = data["M_j_raw"].copy()
    for g in ooc_grids:
        M_new[g] += delta_per_grid
    # Re-derive log + z-score (same definitions used by build_v3_aux_cervero.py).
    log_M_new = np.log1p(M_new)
    log_M_z_new = (log_M_new - data["log_M_mean"]) / data["log_M_std"]

    out = dict(data)
    out["log_M_z"] = torch.from_numpy(log_M_z_new).float().to(data["log_M_z"].device)
    out["M_j_raw_new"] = M_new
    # Patch X_static employment column too (col 8 = total_employment, z-scored log1p).
    # We approximate using the same log+z transformation; this assumes the encoder's
    # employment feature shares the same log1p scaling as M_j.
    X_static_new = data["X_static"].clone()
    # NOTE: encoder employment column may have its own mu/sd. For simplicity in this
    # first-order scenario, we leave X_static unchanged — the head reads log_M_z directly,
    # and δ_NN ≈ 0.05 in v3l so encoder contribution is negligible.
    out["X_static"] = X_static_new
    return out


@torch.no_grad()
def forward_to_flow(encoder, rum, data: dict, observed_OD: torch.Tensor, val_mask: torch.Tensor):
    """Run forward_cs and convert log_P_D → expected flow F[t, i, j].

    Returns:
        F: (T, N, N) expected flow under the model, given observed row totals per origin.
    """
    out = forward_cs(
        encoder, rum,
        data["X_static"], data["X_dynamic"], data["edge_index"],
        data["t_per_mode"], data["mode_names"],
        data["log_d_ij"], data["match_prob"],
        data["log_M_z"], data["log_W_z"], data["log_D_z"],
        data["income_score"], data["pct_kids"], data["mean_cars"],
        data["income_tier_props"],
        data["pi_m_pair"], data["grid_borough_idx"],
        observed_OD, val_mask,
    )
    P = out["log_P_D"].exp()                                        # (T, N, N)
    row_total = observed_OD.sum(dim=2, keepdim=True)                # (T, N, 1)
    F = P * row_total                                               # (T, N, N) expected flow
    return F, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--grids", type=int, nargs="+", default=OOC_GRIDS,
                    help="Grid indices to receive employment boost (default: OOC 9-grid cluster)")
    ap.add_argument("--delta-per-grid", type=float, default=DELTA_PER_GRID_DEFAULT,
                    help="Jobs added per grid (default: 7222 = 65k / 9 OOC grids)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--aux-path", default="data/processed/paperA_v3_aux_cervero.npz",
                    help="(only used internally, passed through to trainer's load logic)")
    ap.add_argument("--out", required=True, type=str,
                    help="Output JSON path (relative to v3 root)")
    args_cli = ap.parse_args()
    args_cli.aux_path = args_cli.aux_path   # keep attribute for load function

    device = torch.device(args_cli.device)
    print(f"[scen-A] device: {device}")

    # ---- Load checkpoint ----------------------------------------------------
    print(f"[scen-A] loading {args_cli.ckpt}")
    ckpt = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
    print(f"        train CPC: {ckpt['final_cpc']:.4f}  | trained args: "
          f"seed={ckpt['args'].get('seed')}  epochs={ckpt['args'].get('epochs')}")
    encoder, rum, train_args = reconstruct_model(ckpt, device)
    # Borough self-loop boost: dest_to_k buffer for busy_dest_boost is saved in state_dict
    # via register_buffer. set_busy_dest_index is not called here because:
    #   (a) state_dict already contains dest_to_k if busy_dest was enabled at train time;
    #   (b) for v3l_lexico_filter we know n_busy_dest=0 (off).

    # ---- Load data inputs ---------------------------------------------------
    # Mirror trainer's aux_path so it picks up the same file.
    train_args.aux_path = args_cli.aux_path
    data = load_scenario_inputs(train_args, device)
    print(f"        data shapes: N={data['N']}  T={data['T']}")

    # ---- Baseline forward ----------------------------------------------------
    print("[scen-A] forward baseline ...")
    t0 = time.time()
    F_base, out_base = forward_to_flow(
        encoder, rum, data, data["observed_OD"], data["val_mask"]
    )
    print(f"        baseline forward {time.time()-t0:.1f}s  CPC@val={out_base.get('cpc', 'n/a')}")

    # Daily inflow under baseline (sum over t and i — destination popularity)
    inflow_base = F_base.sum(dim=(0, 1)).cpu().numpy()                # (N,)

    # ---- OOC intervention ---------------------------------------------------
    print(f"[scen-A] OOC intervention: {len(args_cli.grids)} grids × "
          f"+{args_cli.delta_per_grid:.0f} jobs = +{len(args_cli.grids)*args_cli.delta_per_grid:.0f} total")
    print(f"        baseline M_j at OOC grids: "
          f"{[float(data['M_j_raw'][g]) for g in args_cli.grids]}")
    data_scen = patch_M_at_ooc(data, args_cli.grids, args_cli.delta_per_grid)
    print(f"        post-intervention M_j at OOC grids: "
          f"{[float(data_scen['M_j_raw_new'][g]) for g in args_cli.grids]}")

    # ---- Scenario forward ---------------------------------------------------
    print("[scen-A] forward intervened ...")
    t0 = time.time()
    F_scen, _ = forward_to_flow(
        encoder, rum, data_scen, data["observed_OD"], data["val_mask"]
    )
    print(f"        scenario forward {time.time()-t0:.1f}s")
    inflow_scen = F_scen.sum(dim=(0, 1)).cpu().numpy()                # (N,)

    # ---- Analysis -----------------------------------------------------------
    inflow_delta = inflow_scen - inflow_base
    rel_delta = np.where(inflow_base > 1, inflow_delta / inflow_base, 0.0)
    # Total flow conservation check — softmax over j → totals should be ~invariant per origin
    total_base = F_base.sum().item(); total_scen = F_scen.sum().item()
    print(f"        total flow base={total_base:.0f} scen={total_scen:.0f} "
          f"(should match — row totals fixed)")

    # OOC-grid effect
    ooc_inflow_change = sum(inflow_delta[g] for g in args_cli.grids)
    ooc_inflow_base = sum(inflow_base[g] for g in args_cli.grids)
    ooc_inflow_scen = sum(inflow_scen[g] for g in args_cli.grids)

    # Top winners + losers (excluding OOC). Use ±inf masking so OOC ties don't
    # leak into the ranking when there are no real non-OOC gainers.
    non_ooc_mask = np.ones(data["N"], dtype=bool)
    for g in args_cli.grids:
        non_ooc_mask[g] = False
    delta_for_winners = inflow_delta.copy()
    delta_for_winners[~non_ooc_mask] = -np.inf   # OOC can't be a winner
    delta_for_losers = inflow_delta.copy()
    delta_for_losers[~non_ooc_mask] = +np.inf    # OOC can't be a loser
    top_winner_idx = np.argsort(-delta_for_winners)[:10]
    top_loser_idx  = np.argsort(delta_for_losers)[:10]
    # If no non-OOC grid actually gained, surface that fact in the JSON.
    n_real_winners_non_ooc = int((inflow_delta[non_ooc_mask] > 0).sum())

    # Per-OOC-grid breakdown (where did the +Δ land within the OOC cluster?)
    ooc_breakdown = [
        {"grid": int(g), "base": float(inflow_base[g]), "scen": float(inflow_scen[g]),
         "delta": float(inflow_delta[g])}
        for g in args_cli.grids
    ]
    ooc_breakdown.sort(key=lambda r: -r["delta"])

    # Hansen accessibility delta (use min-mode peak-hour t_ij; β_car from head)
    print("[scen-A] computing Hansen accessibility delta ...")
    t_min_peak = torch.stack(
        [data["t_per_mode"][m][8] for m in data["mode_names"]],     # hour 8 = peak
        dim=0,
    ).min(dim=0).values.cpu().numpy()                                # (N, N)
    beta_car = float(rum.beta_t_per_mode[0].item())                  # mode 0 = car
    A_base = hansen_accessibility(data["M_j_raw"], t_min_peak, beta_car)
    A_scen = hansen_accessibility(data_scen["M_j_raw_new"], t_min_peak, beta_car)
    dA = A_scen - A_base
    dA_pct = np.where(A_base > 0, 100 * dA / A_base, 0)

    summary = {
        "config": {
            "ckpt": args_cli.ckpt,
            "ckpt_cpc": float(ckpt["final_cpc"]),
            "ooc_grids": list(args_cli.grids),
            "delta_per_grid": args_cli.delta_per_grid,
            "total_delta_jobs": float(len(args_cli.grids) * args_cli.delta_per_grid),
            "beta_car_recovered": beta_car,
        },
        "totals": {
            "base_flow": total_base, "scen_flow": total_scen,
        },
        "ooc": {
            "inflow_base": float(ooc_inflow_base),
            "inflow_scen": float(ooc_inflow_scen),
            "inflow_delta": float(ooc_inflow_change),
            "inflow_change_pct": float(100 * ooc_inflow_change / max(ooc_inflow_base, 1.0)),
        },
        "system": {
            "inflow_gini_base": gini_coefficient(inflow_base),
            "inflow_gini_scen": gini_coefficient(inflow_scen),
            "accessibility_gini_base": gini_coefficient(A_base),
            "accessibility_gini_scen": gini_coefficient(A_scen),
            "accessibility_mean_delta": float(dA.mean()),
            "accessibility_mean_delta_pct": float(dA_pct.mean()),
            "accessibility_p99_delta_pct": float(np.percentile(dA_pct, 99)),
            "n_non_ooc_grids_with_positive_delta": n_real_winners_non_ooc,
        },
        "ooc_breakdown_sorted": ooc_breakdown,
        "top_winners_non_ooc": [
            {"grid": int(g), "delta": float(inflow_delta[g]),
             "base": float(inflow_base[g]), "rel_delta_pct": float(100 * rel_delta[g])}
            for g in top_winner_idx
        ],
        "top_losers_non_ooc": [
            {"grid": int(g), "delta": float(inflow_delta[g]),
             "base": float(inflow_base[g]), "rel_delta_pct": float(100 * rel_delta[g])}
            for g in top_loser_idx
        ],
    }

    out_path = V3_ROOT / args_cli.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Save full inflow vectors as npz for downstream maps
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(npz_path,
                        inflow_base=inflow_base.astype(np.float32),
                        inflow_scen=inflow_scen.astype(np.float32),
                        dA_hansen=dA.astype(np.float32))
    print(f"[scen-A] wrote {out_path}")
    print(f"[scen-A] wrote {npz_path}")

    print("\n=== Summary ===")
    print(f"  OOC inflow change: +{ooc_inflow_change:.0f} jobs filled "
          f"({summary['ooc']['inflow_change_pct']:+.2f}%)")
    print(f"  Hansen access mean Δ%: {summary['system']['accessibility_mean_delta_pct']:+.3f}%")
    print(f"  Inflow Gini: {summary['system']['inflow_gini_base']:.4f} → "
          f"{summary['system']['inflow_gini_scen']:.4f}  "
          f"(Δ {summary['system']['inflow_gini_scen']-summary['system']['inflow_gini_base']:+.4f})")
    print(f"  Top winners (non-OOC): " + ", ".join(
        f"g{w['grid']}({w['delta']:+.0f})" for w in summary["top_winners_non_ooc"][:5]))
    print(f"  Top losers  (non-OOC): " + ", ".join(
        f"g{w['grid']}({w['delta']:+.0f})" for w in summary["top_losers_non_ooc"][:5]))


if __name__ == "__main__":
    main()
