"""Multi-metric comparison: full Deep Gravity vs DUAL_HET.

Reads:
  evaluation_outputs/paper_a/dg_predictions.npz   (downloaded from Colab)
  evaluation_outputs/paper_a/dual_het_seed0.pt    (local DUAL_HET ckpt)

Computes a multi-metric leaderboard on the val_mask spatial holdout:
  - overall CPC (daily aggregate + hourly)
  - peak (7-9 + 17-19) vs off-peak CPC (hourly only)
  - short (<5 km) vs long (>20 km) trip CPC
  - JS divergence
  - Top-5 destination accuracy
  - RMSE
  - calibration ECE (binned)

Outputs:
  evaluation_outputs/paper_a/dg_vs_dual_het_table.md
  evaluation_outputs/paper_a/dg_vs_dual_het_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_a.compare_four_variants import load_data
from experiments.paper_a.dual_het_scenario_helpers import predict_flow

OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"


# ────────────────────────────────────────────────────────────────────────────
#  Metric helpers
# ────────────────────────────────────────────────────────────────────────────

def cpc(pred: np.ndarray, obs: np.ndarray) -> float:
    """Common Part of Commuters."""
    num = 2.0 * np.minimum(pred, obs).sum()
    den = pred.sum() + obs.sum() + 1e-9
    return float(num / den)


def js_divergence(pred: np.ndarray, obs: np.ndarray, axis: int = -1) -> float:
    """Mean per-row JS divergence (after row-normalising both)."""
    pred = pred.copy(); obs = obs.copy()
    p_sum = pred.sum(axis=axis, keepdims=True); p_sum[p_sum == 0] = 1
    o_sum = obs.sum(axis=axis, keepdims=True); o_sum[o_sum == 0] = 1
    p = pred / p_sum + 1e-12; o = obs / o_sum + 1e-12
    m = 0.5 * (p + o)
    js = 0.5 * (p * np.log(p / m)).sum(axis=axis) + 0.5 * (o * np.log(o / m)).sum(axis=axis)
    return float(js.mean())


def topk_accuracy(pred: np.ndarray, obs: np.ndarray, k: int = 5) -> float:
    """Per-row: is the actual top-1 dest within top-K predicted dests?"""
    obs_top = obs.argmax(axis=-1)
    pred_topk = np.argsort(-pred, axis=-1)[..., :k]
    hit = (pred_topk == obs_top[..., None]).any(axis=-1)
    return float(hit.mean())


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(((pred - obs) ** 2).mean()))


def calibration_ece(pred_p: np.ndarray, obs_p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.
    pred_p, obs_p: row-normalised probabilities of same shape."""
    p = pred_p.flatten(); o = obs_p.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; total = 0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0: continue
        bin_pred = p[mask].mean(); bin_obs = o[mask].mean()
        ece += abs(bin_pred - bin_obs) * mask.sum()
        total += mask.sum()
    return float(ece / max(total, 1))


# ────────────────────────────────────────────────────────────────────────────
#  Load DUAL_HET predictions from ckpt
# ────────────────────────────────────────────────────────────────────────────

def load_dual_het_predictions(d: dict) -> np.ndarray:
    """Run DUAL_HET inference on full data, return F̂ (T, N, N)."""
    from models_lib.inverse_rum.dual_branch_encoder import DualBranchEncoder
    from models_lib.inverse_rum.mixture_head import MixtureRUMHead

    ckpt_path = OUT_DIR / "dual_het_seed0.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DUAL_HET ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    encoder = DualBranchEncoder(
        static_dim=cfg["F_static"], dyn_dim=cfg["F_dynamic"],
        hidden_dim=cfg["hidden_dim"], gru_hidden=cfg["gru_hidden"],
        n_sage_layers=cfg["n_sage_layers"], tcn_kernels=tuple(cfg["tcn_kernels"]),
    )
    encoder.load_state_dict(ckpt["encoder_state"]); encoder.eval()

    head = MixtureRUMHead(
        n_hours=cfg["n_hours"], n_modes=cfg["n_modes"], n_tiers=cfg["n_tiers"],
        tier_specific_delta=cfg.get("tier_specific_delta", True),
    )
    head.load_state_dict(ckpt["head_state"], strict=False); head.eval()

    # Build pair_mode_share like training
    from models_lib.inverse_rum.dual_branch_mixture_trainer import make_distance_aware_mode_share
    coords = torch.from_numpy(
        np.load(ROOT / "data" / "processed" / "demo_cache.npz",
                allow_pickle=True)["coords_bng"].astype(np.float32))
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_km = (torch.linalg.norm(diff, dim=-1) / 1000.0).clamp(min=0.1)
    mode_names = ["car", "transit", "walk"]
    pair_mode_share = make_distance_aware_mode_share(
        d["mode_share"], dist_km, mode_names,
        walk_threshold_km=cfg.get("walk_threshold_km", 5.0),
    )
    log_pi_mk_pair = torch.log(
        pair_mode_share.unsqueeze(-1) * d["income_tier_props"][:, None, None, :].clamp(min=1e-9)
    ).clamp(min=-30.0)

    t_per_mode = {"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]}
    norm_stats = ckpt.get("norm_stats", None)

    F_pred = predict_flow(
        encoder, head,
        d["X_static"], d["X_dynamic"], d["edge_index"],
        t_per_mode, d["log_d"], log_pi_mk_pair, d["F_ij_t"],
        occ_match=d["occ_match"], norm_stats=norm_stats,
    )
    return F_pred.detach().numpy()                                # (T, N, N)


# ────────────────────────────────────────────────────────────────────────────
#  Slicers
# ────────────────────────────────────────────────────────────────────────────

PEAK_HOURS = (7, 8, 9, 17, 18, 19)
OFFPEAK_HOURS = tuple(h for h in range(24) if h not in PEAK_HOURS)


def metric_dict(F_hat: np.ndarray, F_obs: np.ndarray, val_idx: np.ndarray,
                 dist_km: np.ndarray, hourly: bool) -> dict:
    """Compute all metrics restricted to val origins."""
    out = {}
    if hourly:
        F_hat_v = F_hat[:, val_idx, :]
        F_obs_v = F_obs[:, val_idx, :]
    else:
        F_hat_v = F_hat[val_idx, :]
        F_obs_v = F_obs[val_idx, :]
    out["overall_cpc"] = cpc(F_hat_v, F_obs_v)
    out["rmse"] = rmse(F_hat_v, F_obs_v)
    if hourly:
        peak_pred = F_hat[list(PEAK_HOURS)][:, val_idx, :]
        peak_obs = F_obs[list(PEAK_HOURS)][:, val_idx, :]
        offpeak_pred = F_hat[list(OFFPEAK_HOURS)][:, val_idx, :]
        offpeak_obs = F_obs[list(OFFPEAK_HOURS)][:, val_idx, :]
        out["peak_cpc"] = cpc(peak_pred, peak_obs)
        out["offpeak_cpc"] = cpc(offpeak_pred, offpeak_obs)
    # Short / long trips
    short_mask = dist_km < 5.0
    long_mask = dist_km > 20.0
    if hourly:
        F_hat_short = F_hat_v.copy(); F_hat_short[:, ~short_mask[val_idx]] = 0
        F_obs_short = F_obs_v.copy(); F_obs_short[:, ~short_mask[val_idx]] = 0
        F_hat_long = F_hat_v.copy(); F_hat_long[:, ~long_mask[val_idx]] = 0
        F_obs_long = F_obs_v.copy(); F_obs_long[:, ~long_mask[val_idx]] = 0
    else:
        F_hat_short = F_hat_v.copy(); F_hat_short[~short_mask[val_idx]] = 0
        F_obs_short = F_obs_v.copy(); F_obs_short[~short_mask[val_idx]] = 0
        F_hat_long = F_hat_v.copy(); F_hat_long[~long_mask[val_idx]] = 0
        F_obs_long = F_obs_v.copy(); F_obs_long[~long_mask[val_idx]] = 0
    out["short_trip_cpc"] = cpc(F_hat_short, F_obs_short)
    out["long_trip_cpc"] = cpc(F_hat_long, F_obs_long)
    # JS, Top-K, ECE: row-normalise to probabilities
    if hourly:
        T_it = F_obs_v.sum(axis=2, keepdims=True); T_it[T_it == 0] = 1
        p_obs = F_obs_v / T_it
        T_it_h = F_hat_v.sum(axis=2, keepdims=True); T_it_h[T_it_h == 0] = 1
        p_hat = F_hat_v / T_it_h
    else:
        T_i = F_obs_v.sum(axis=1, keepdims=True); T_i[T_i == 0] = 1
        p_obs = F_obs_v / T_i
        T_i_h = F_hat_v.sum(axis=1, keepdims=True); T_i_h[T_i_h == 0] = 1
        p_hat = F_hat_v / T_i_h
    out["js_div"] = js_divergence(p_hat, p_obs)
    out["top5_acc"] = topk_accuracy(p_hat, p_obs, k=5)
    out["ece"] = calibration_ece(p_hat, p_obs)
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    d = load_data()
    F_obs = d["F_ij_t"].numpy()                                   # (T, N, N)
    F_obs_daily = F_obs.sum(axis=0)                               # (N, N)
    val_mask = d["val_mask"].numpy()
    val_idx = np.where(val_mask)[0]

    coords = np.load(ROOT / "data" / "processed" / "demo_cache.npz",
                      allow_pickle=True)["coords_bng"].astype(np.float32)
    diff = coords[None, :, :] - coords[:, None, :]
    dist_km = np.maximum(np.linalg.norm(diff, axis=-1) / 1000.0, 0.1)

    # ── Load DUAL_HET predictions
    print("\nRunning DUAL_HET inference ...")
    F_dual = load_dual_het_predictions(d)
    print(f"  F_dual shape: {F_dual.shape}, sum: {F_dual.sum():.0f}")

    # ── Load Deep Gravity predictions (optional; print DUAL_HET-only if missing)
    dg_path = OUT_DIR / "dg_predictions.npz"
    F_dg_daily = None; F_dg_hourly = None
    if not dg_path.exists():
        print(f"\nWARNING: {dg_path} not found — printing DUAL_HET-only metrics.")
        print("To compare against Deep Gravity, run on Colab T4 first:")
        print("  1. Upload `evaluation_outputs/paper_a/dg_colab_bundle.npz`")
        print("  2. Upload `experiments/paper_a/train_deep_gravity_colab.py`")
        print("  3. !python train_deep_gravity_colab.py")
        print("  4. Download `dg_predictions.npz` back to evaluation_outputs/paper_a/")
        print("  5. Re-run this script.\n")
    else:
        print(f"\nLoading Deep Gravity predictions from {dg_path} ...")
        dg = np.load(dg_path, allow_pickle=True)
        print(f"  keys: {list(dg.files)}")
        F_dg_daily = dg["pred_daily"] if "pred_daily" in dg.files else None
        F_dg_hourly = dg["pred_hourly"] if "pred_hourly" in dg.files else None

    # ── Build comparison
    rows = []

    print("\nComputing metrics ...")
    F_dual_daily = F_dual.sum(axis=0)
    m_dual_daily = metric_dict(F_dual_daily, F_obs_daily, val_idx, dist_km, hourly=False)
    m_dual_hourly = metric_dict(F_dual, F_obs, val_idx, dist_km, hourly=True)
    rows.append(("DUAL_HET (daily agg)", m_dual_daily))
    rows.append(("DUAL_HET (hourly)", m_dual_hourly))

    if F_dg_daily is not None:
        m_dg_daily = metric_dict(F_dg_daily, F_obs_daily, val_idx, dist_km, hourly=False)
        rows.append(("Deep Gravity full (daily)", m_dg_daily))
    if F_dg_hourly is not None:
        m_dg_hourly = metric_dict(F_dg_hourly, F_obs, val_idx, dist_km, hourly=True)
        rows.append(("Deep Gravity full (hourly)", m_dg_hourly))

    # ── Print + save markdown table
    md_lines = []
    md_lines.append("# Deep Gravity (full paper-spec) vs DUAL_HET — multi-metric leaderboard\n")
    md_lines.append("Spatial holdout (val_mask, 172 origins).\n")
    md_lines.append("\n## Daily aggregate metrics\n")
    md_lines.append("| Model | overall CPC | short<5km | long>20km | JS div ↓ | Top-5 acc | ECE ↓ | RMSE ↓ |")
    md_lines.append("|-------|-------------|-----------|-----------|----------|-----------|-------|--------|")
    for name, m in rows:
        if "hourly" in name.lower():
            continue
        md_lines.append(
            f"| {name} | {m['overall_cpc']:.4f} | {m['short_trip_cpc']:.4f} | "
            f"{m['long_trip_cpc']:.4f} | {m['js_div']:.4f} | {m['top5_acc']:.4f} | "
            f"{m['ece']:.4f} | {m['rmse']:.2f} |"
        )

    md_lines.append("\n## Hourly metrics\n")
    md_lines.append("| Model | overall CPC | peak (7-9,17-19) | off-peak | short<5km | long>20km | JS div ↓ | Top-5 acc | ECE ↓ |")
    md_lines.append("|-------|-------------|------------------|----------|-----------|-----------|----------|-----------|-------|")
    for name, m in rows:
        if "hourly" not in name.lower():
            continue
        md_lines.append(
            f"| {name} | {m['overall_cpc']:.4f} | {m['peak_cpc']:.4f} | {m['offpeak_cpc']:.4f} | "
            f"{m['short_trip_cpc']:.4f} | {m['long_trip_cpc']:.4f} | "
            f"{m['js_div']:.4f} | {m['top5_acc']:.4f} | {m['ece']:.4f} |"
        )

    md = "\n".join(md_lines) + "\n"
    print("\n" + md)
    out_md = OUT_DIR / "dg_vs_dual_het_table.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\nWrote {out_md}")

    # ── Save JSON
    json_out = {name: m for name, m in rows}
    out_json = OUT_DIR / "dg_vs_dual_het_metrics.json"
    out_json.write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
