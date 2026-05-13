"""D8: baselines on the same 4-borough spatial holdout as D7.

Compares to Phase B v6 (CPC ~0.29 spatial holdout). Three reference models:

  Gravity (Wilson power form, production-constrained):
      U_ij = alpha · log(D_j) - beta · t_ij + gamma · log_d_ij
      P(j|i) = softmax_j(U_ij)
      Fit (alpha, beta, gamma) by MLE on training origins.

  Radiation (Simini 2012, parameter-free):
      F_ij ∝ T_i · m_i · n_j / [(m_i + s_ij)(m_i + n_j + s_ij)]
      with s_ij = sum of n_k for grids closer to i than j (excluding i and j).
      m_i = population, n_j = total_employment.

  Deep Gravity (Simini 2021, single-MLP):
      h_ij = MLP([z_i, z_j, log_d_ij])
      P(j|i) = softmax_j(h_ij)
      Fit MLP weights by NLL on training origins.

All three are evaluated on the SAME val_mask (4 held-out boroughs) and the
SAME CPC formula used for Phase B v6 in spatial_holdout.py.

Run:
    python experiments/paper_a/baselines_spatial_holdout.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, cpc, load_data
from experiments.paper_a.spatial_holdout import HELDOUT_BOROUGHS, make_spatial_masks


# ---------------------------------------------------------------- gravity
def fit_gravity_softmax(
    F_ij: torch.Tensor,
    D_j: torch.Tensor,
    t_ij: torch.Tensor,
    log_d_ij: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = 400,
    lr: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Production-constrained Wilson gravity. Fit (alpha, beta, gamma) by MLE."""
    N = F_ij.shape[0]
    log_D = torch.log(D_j.clamp(min=1.0))                              # (N,)
    log_D_b = log_D.unsqueeze(0).expand(N, N)                          # broadcast over origins

    raw_alpha = torch.tensor(0.0, requires_grad=True)
    raw_beta = torch.tensor(np.log(np.exp(0.05) - 1.0), requires_grad=True)   # softplus^-1(0.05)
    raw_gamma = torch.tensor(np.log(np.exp(1.0) - 1.0), requires_grad=True)
    opt = torch.optim.Adam([raw_alpha, raw_beta, raw_gamma], lr=lr)

    O_i = F_ij.sum(dim=1, keepdim=True)                                # (N, 1)
    train_idx = train_mask.bool()
    F_train = F_ij[train_idx]                                           # (n_train, N)
    last = {}
    for ep in range(epochs):
        opt.zero_grad()
        alpha = F.softplus(raw_alpha)        # >0
        beta = F.softplus(raw_beta)          # >0 (we use -beta in utility)
        gamma = F.softplus(raw_gamma)        # >0 (we use -gamma in utility for distance)
        # Utility V_ij; signs: alpha*log_D positive, time/distance penalties negative.
        V = alpha * log_D_b - beta * t_ij - gamma * log_d_ij           # (N, N)
        V = V.clone()
        V.fill_diagonal_(-1e9)
        log_p = F.log_softmax(V, dim=1)
        # NLL on training origins
        loss = -(F_train * log_p[train_idx]).sum() / max(F_train.sum().item(), 1.0)
        loss.backward()
        opt.step()
        if ep == epochs - 1 or ep % 100 == 0:
            last = {"epoch": ep, "loss": float(loss.item()),
                    "alpha": float(alpha.item()), "beta": float(beta.item()),
                    "gamma": float(gamma.item())}
    # Final prediction
    with torch.no_grad():
        alpha = F.softplus(raw_alpha); beta = F.softplus(raw_beta); gamma = F.softplus(raw_gamma)
        V = alpha * log_D_b - beta * t_ij - gamma * log_d_ij
        V = V.clone(); V.fill_diagonal_(-1e9)
        P = F.softmax(V, dim=1)
        F_pred = O_i * P                                                # (N, N)
    return F_pred, last


# ---------------------------------------------------------------- radiation
def fit_radiation(
    F_ij: torch.Tensor,
    m_i: np.ndarray,
    n_j: np.ndarray,
    coords: np.ndarray,
) -> torch.Tensor:
    """Simini 2012 parameter-free. Returns (N, N) predicted flows."""
    N = len(m_i)
    # Pairwise distance (km)
    diff = coords[:, None, :] - coords[None, :, :]
    d_ij = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0                   # (N, N) km
    # s_ij = sum of n_k for k closer to i than j, excluding i and j
    order = np.argsort(d_ij, axis=1)
    n_sorted = n_j[order]
    cum = np.cumsum(n_sorted, axis=1)
    s_in_order = cum - n_sorted
    inv_order = np.argsort(order, axis=1)
    s = np.take_along_axis(s_in_order, inv_order, axis=1)
    s = s - n_j[None, :] * 0  # already excluded n_j by subtracting n_sorted
    s = s - m_i[:, None] * 0  # m_i not in n_j set so no extra subtraction
    s = np.maximum(s, 0.0)
    np.fill_diagonal(s, 0.0)
    # Probability prob_ij = (m_i * n_j) / [(m_i + s_ij) * (m_i + n_j + s_ij)]
    M = m_i[:, None]                                                    # (N, 1)
    Nj = n_j[None, :]                                                   # (1, N)
    denom = (M + s) * (M + Nj + s)
    denom = np.where(denom <= 0, np.nan, denom)
    prob = (M * Nj) / denom
    prob = np.nan_to_num(prob, nan=0.0)
    np.fill_diagonal(prob, 0.0)
    # Normalise per origin
    row_sum = prob.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    prob = prob / row_sum
    # Scale by observed origin total
    O_i = F_ij.sum(dim=1).cpu().numpy()                                 # (N,)
    F_pred = O_i[:, None] * prob
    return torch.tensor(F_pred, dtype=torch.float32)


# ---------------------------------------------------------------- deep gravity
class _DeepGravityMLP(nn.Module):
    """Per-OD-pair MLP. Forward processes one origin block at a time so the
    pairwise (N, N, 2F+1) tensor is never materialised in full."""
    def __init__(self, feat_dim: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []
        d_in = 2 * feat_dim + 1
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.ReLU()]
            d_in = hidden
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward_block(self, X_i_block: torch.Tensor, X_j: torch.Tensor,
                      log_d_block: torch.Tensor) -> torch.Tensor:
        """X_i_block: (B, F)  X_j: (N, F)  log_d_block: (B, N) -> (B, N) logits."""
        B = X_i_block.shape[0]; N = X_j.shape[0]
        Xi_b = X_i_block.unsqueeze(1).expand(B, N, -1)
        Xj_b = X_j.unsqueeze(0).expand(B, N, -1)
        d_b = log_d_block.unsqueeze(-1)
        feat = torch.cat([Xi_b, Xj_b, d_b], dim=-1)                      # (B, N, 2F+1)
        return self.net(feat).squeeze(-1)                                # (B, N)


def fit_deep_gravity(
    F_ij: torch.Tensor,
    static: torch.Tensor,
    log_d_ij: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-3,
    hidden: int = 128,
    n_layers: int = 5,
    seed: int = 0,
    batch_size: int = 128,
    val_mask: torch.Tensor = None,
    early_stop_patience: int = 40,
    lr_schedule: bool = True,
    device: str = "cpu",
    verbose_every: int = 50,
) -> tuple[torch.Tensor, dict]:
    """Memory-aware Deep Gravity training, GPU-ready.

    Defaults follow Simini 2021 spirit: 5 layers × 128 units, 1000 epoch cap with
    early stopping on val NLL (patience=40), cosine lr schedule. Pairwise
    activation per batch is (B, N, 2F+1), so a single batch fits comfortably
    on a 16GB GPU even at hidden=128.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    F_ij = F_ij.to(dev); static = static.to(dev); log_d_ij = log_d_ij.to(dev)
    train_mask = train_mask.to(dev)
    if val_mask is not None:
        val_mask = val_mask.to(dev)

    N, Fdim = static.shape
    model = _DeepGravityMLP(feat_dim=Fdim, hidden=hidden, n_layers=n_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if lr_schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_idx = torch.where(train_mask.bool())[0]
    F_train_total = F_ij[train_idx].sum().item()
    O_i = F_ij.sum(dim=1, keepdim=True)

    val_idx = torch.where(val_mask.bool())[0] if val_mask is not None else None
    F_val_total = F_ij[val_idx].sum().item() if val_idx is not None else 0.0

    best_val = float("inf"); best_state = None; no_improve = 0
    last = {}
    history = []

    for ep in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=dev)]
        ep_loss = 0.0; n_pairs = 0
        for b0 in range(0, len(perm), batch_size):
            batch_oids = perm[b0:b0 + batch_size]
            opt.zero_grad()
            h = model.forward_block(static[batch_oids], static, log_d_ij[batch_oids])
            # Mask self-loop where origin == column
            mask_diag = torch.zeros_like(h)
            for k, oid in enumerate(batch_oids):
                mask_diag[k, oid] = -1e9
            h = h + mask_diag
            log_p = F.log_softmax(h, dim=1)
            F_b = F_ij[batch_oids]
            loss = -(F_b * log_p).sum() / max(F_train_total, 1.0)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()); n_pairs += 1
        if lr_schedule:
            sched.step()
        avg_train = ep_loss / max(n_pairs, 1)

        # Val NLL
        avg_val = float("nan")
        if val_idx is not None and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                v_loss = 0.0; v_n = 0
                for b0 in range(0, len(val_idx), batch_size):
                    bo = val_idx[b0:b0 + batch_size]
                    h = model.forward_block(static[bo], static, log_d_ij[bo])
                    mask_diag = torch.zeros_like(h)
                    for k, oid in enumerate(bo):
                        mask_diag[k, oid] = -1e9
                    h = h + mask_diag
                    log_p = F.log_softmax(h, dim=1)
                    F_b = F_ij[bo]
                    v_loss += float(-(F_b * log_p).sum().item()) / max(F_val_total, 1.0)
                    v_n += 1
                avg_val = v_loss

        history.append({"epoch": ep, "train_nll": avg_train, "val_nll": avg_val,
                        "lr": opt.param_groups[0]["lr"]})
        if val_idx is not None and avg_val < best_val - 1e-6:
            best_val = avg_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if verbose_every and (ep == 0 or ep % verbose_every == 0 or ep == epochs - 1):
            print(f"  [DG s{seed}] ep {ep:4d}  train_nll {avg_train:.4f}  "
                  f"val_nll {avg_val:.4f}  lr {opt.param_groups[0]['lr']:.5f}  "
                  f"no_improve {no_improve}/{early_stop_patience}")
        if val_idx is not None and no_improve >= early_stop_patience:
            print(f"  [DG s{seed}] early stop at epoch {ep} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    last = {"epoch": ep, "loss": best_val if val_idx is not None else avg_train,
            "best_val": best_val, "history": history[-5:]}

    # Inference
    model.eval()
    with torch.no_grad():
        F_pred = torch.zeros_like(F_ij)
        for b0 in range(0, N, batch_size):
            bo = torch.arange(b0, min(b0 + batch_size, N), device=dev)
            h = model.forward_block(static[bo], static, log_d_ij[bo])
            mask_diag = torch.zeros_like(h)
            for k, oid in enumerate(bo):
                mask_diag[k, oid] = -1e9
            h = h + mask_diag
            P = F.softmax(h, dim=1)
            F_pred[bo] = O_i[bo] * P
    return F_pred.cpu(), last


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds for Deep Gravity (gravity & radiation are deterministic)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda (only Deep Gravity benefits)")
    parser.add_argument("--dg_epochs", type=int, default=1000)
    parser.add_argument("--dg_hidden", type=int, default=128)
    parser.add_argument("--dg_layers", type=int, default=5)
    parser.add_argument("--dg_lr", type=float, default=1e-3)
    parser.add_argument("--dg_patience", type=int, default=40)
    args = parser.parse_args()

    print("[D8] loading data ...")
    data = load_data()
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)
    print(f"[D8] holdout: {mask_info}")

    # Common tensors
    F_ij = data["F_ij"]                                                 # (N, N)
    t_ij = data["t_ij"]                                                 # (N, N)
    log_d = data["log_d"]                                               # (N, N)
    static = data["static"]                                             # (N, F)
    F_obs_np = F_ij.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)

    # Raw D_j (total_employment) and m_i (population)
    df_grid = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    grid_ids = cache["grid_ids"].tolist()
    df_grid = df_grid.set_index("grid_id").reindex(grid_ids)
    D_j = torch.tensor(df_grid["total_employment"].fillna(0.0).to_numpy(),
                       dtype=torch.float32)
    m_i = df_grid["population"].fillna(0.0).to_numpy().astype(np.float64)
    n_j = df_grid["total_employment"].fillna(0.0).to_numpy().astype(np.float64)
    coords = cache["coords_bng"].astype(np.float64)                     # (N, 2) metres

    rows = []

    # --- Gravity (deterministic given fit, run once)
    print("\n[D8] fit Gravity (Wilson production-constrained, softmax MLE) ...")
    t0 = time.time()
    F_pred, info = fit_gravity_softmax(F_ij, D_j, t_ij, log_d, train_mask, epochs=400, lr=0.05)
    grav_cpc = cpc(F_obs_np, F_pred.cpu().numpy(), val_mask_np)
    print(f"  Gravity: val_CPC(spatial) = {grav_cpc:.4f}  "
          f"(alpha={info['alpha']:.3f} beta={info['beta']:.4f} gamma={info['gamma']:.3f}) "
          f"t={time.time()-t0:.1f}s")
    rows.append({"model": "gravity", "seed": "n/a", "val_cpc_spatial": grav_cpc, **info})

    # --- Radiation (parameter-free, deterministic)
    print("\n[D8] fit Radiation (Simini 2012, parameter-free) ...")
    t0 = time.time()
    F_pred = fit_radiation(F_ij, m_i, n_j, coords)
    rad_cpc = cpc(F_obs_np, F_pred.cpu().numpy(), val_mask_np)
    print(f"  Radiation: val_CPC(spatial) = {rad_cpc:.4f}  t={time.time()-t0:.1f}s")
    rows.append({"model": "radiation", "seed": "n/a", "val_cpc_spatial": rad_cpc})

    # --- Deep Gravity (multiple seeds for noise band)
    print(f"\n[D8] fit Deep Gravity (Simini 2021 spirit; "
          f"layers={args.dg_layers} hidden={args.dg_hidden} epochs={args.dg_epochs} "
          f"early_stop_patience={args.dg_patience} device={args.device}) ...")
    dg_cpcs = []
    for seed in range(args.seeds):
        t0 = time.time()
        F_pred, info = fit_deep_gravity(
            F_ij, static, log_d, train_mask,
            epochs=args.dg_epochs, lr=args.dg_lr,
            hidden=args.dg_hidden, n_layers=args.dg_layers,
            seed=seed, val_mask=val_mask,
            early_stop_patience=args.dg_patience,
            device=args.device,
        )
        dg_cpc = cpc(F_obs_np, F_pred.cpu().numpy(), val_mask_np)
        dg_cpcs.append(dg_cpc)
        print(f"  Deep Gravity seed={seed}: val_CPC(spatial) = {dg_cpc:.4f}  "
              f"best_val_NLL={info['best_val']:.4f}  t={time.time()-t0:.1f}s")
        # info contains 'history' (last 5 epochs) — strip for CSV row simplicity
        info_row = {k: v for k, v in info.items() if k != 'history'}
        rows.append({"model": "deep_gravity", "seed": seed, "val_cpc_spatial": dg_cpc, **info_row})

    df_out = pd.DataFrame(rows)
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "baselines_spatial_holdout.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[D8] wrote {out_csv}")

    # --- Comparison vs Phase B v6 spatial holdout
    pbv6_path = ROOT / "evaluation_outputs" / "paper_a" / "spatial_holdout.json"
    if pbv6_path.exists():
        pbv6 = json.load(open(pbv6_path))
        pbv6_cpc = pbv6["spatial_holdout_cpc_mean"]
        pbv6_std = pbv6["spatial_holdout_cpc_std"]
    else:
        pbv6_cpc, pbv6_std = float("nan"), float("nan")

    dg_mean = float(np.mean(dg_cpcs)); dg_std = float(np.std(dg_cpcs))
    print(f"\n[D8] === leaderboard (4-borough spatial holdout) ===")
    print(f"  {'Model':<25} {'val_CPC':>10} {'± std':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Gravity (Wilson)':<25} {grav_cpc:>10.4f} {0:>10.4f}")
    print(f"  {'Radiation (Simini12)':<25} {rad_cpc:>10.4f} {0:>10.4f}")
    print(f"  {'Deep Gravity (MLP)':<25} {dg_mean:>10.4f} {dg_std:>10.4f}  (n={args.seeds})")
    print(f"  {'Phase B v6 (ours)':<25} {pbv6_cpc:>10.4f} {pbv6_std:>10.4f}")
    print()
    if not np.isnan(pbv6_cpc):
        for name, c in [("Gravity", grav_cpc), ("Radiation", rad_cpc), ("Deep Gravity", dg_mean)]:
            uplift = (pbv6_cpc - c) / c * 100
            print(f"  ours vs {name:<15} : Δ={pbv6_cpc - c:+.4f}  ({uplift:+.1f}%)")

    summary = {
        "heldout_boroughs": HELDOUT_BOROUGHS,
        "mask_info": mask_info,
        "rows": rows,
        "leaderboard": {
            "gravity": grav_cpc,
            "radiation": rad_cpc,
            "deep_gravity_mean": dg_mean,
            "deep_gravity_std": dg_std,
            "phase_b_v6_mean": pbv6_cpc,
            "phase_b_v6_std": pbv6_std,
        },
    }
    out_json = ROOT / "evaluation_outputs" / "paper_a" / "baselines_spatial_holdout.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[D8] wrote {out_json}")


if __name__ == "__main__":
    main()
