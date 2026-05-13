"""Full paper-spec Deep Gravity training — to run on Colab T4 GPU.

Architecture: 15 hidden layers × 256 LeakyReLU units (Simini et al. 2021 NComm).
Trains TWO variants on the same spatial-holdout split as DUAL_HET:
  (1) Daily aggregate Deep Gravity   — paper-native, P(j | i)
  (2) Hourly Deep Gravity (extended) — adds sin h / cos h, P(j | i, t)

Outputs predictions + metrics to dg_predictions.npz, ready for local
multi-metric comparison vs DUAL_HET.

Usage on Colab (after `Runtime → Change runtime type → T4 GPU`):
    !python train_deep_gravity_colab.py

Required upload (1 file): `dg_colab_bundle.npz` produced by
`prep_dg_colab_bundle.py` locally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
#  Full paper-spec Deep Gravity: 15 hidden × 256 LeakyReLU
# ────────────────────────────────────────────────────────────────────────────

class DeepGravityFull(nn.Module):
    """15 hidden layers × 256 units LeakyReLU MLP, per Simini 2021.

    Input per (origin, dest) pair: [X_o, X_d, log d_ij, optional time embed].
    Output: 1 score per pair, softmax over destinations within an origin.
    """
    def __init__(self, feat_dim: int, n_time_features: int = 0,
                 hidden: int = 256, n_layers: int = 15):
        super().__init__()
        in_dim = 2 * feat_dim + 1 + n_time_features
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2, inplace=True)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.feat_dim = feat_dim
        self.n_time_features = n_time_features

    def forward(self, X_pair: torch.Tensor) -> torch.Tensor:
        """X_pair: (..., in_dim) → (...,) score."""
        return self.net(X_pair).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Daily DG training (P(j|i), no time)
# ────────────────────────────────────────────────────────────────────────────

def train_daily_dg(
    X_static: torch.Tensor,        # (N, F)
    F_ij: torch.Tensor,            # (N, N), aggregate over hours
    log_d: torch.Tensor,           # (N, N)
    train_mask: torch.Tensor,      # (N,)
    val_mask: torch.Tensor,        # (N,)
    device: str,
    epochs: int = 1000,
    lr: float = 1e-3,
    batch_origins: int = 32,
    patience: int = 60,
    seed: int = 0,
    verbose_every: int = 20,
) -> tuple[DeepGravityFull, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    N, Fdim = X_static.shape
    model = DeepGravityFull(feat_dim=Fdim, n_time_features=0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[daily] params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_static = X_static.to(device)
    F_ij = F_ij.to(device)
    log_d = log_d.to(device)
    train_idx = torch.where(train_mask)[0].to(device)
    val_idx = torch.where(val_mask)[0].to(device)

    # Origin totals
    T_i = F_ij.sum(dim=1)                                       # (N,)

    # Pre-compute static-pair feature scaffold per origin batch.
    def build_pair_features(origins: torch.Tensor) -> torch.Tensor:
        """origins: (B,) → (B, N, 2F+1) pairwise features."""
        Xo = X_static[origins].unsqueeze(1).expand(-1, N, -1)    # (B, N, F)
        Xd = X_static.unsqueeze(0).expand(origins.size(0), -1, -1)  # (B, N, F)
        ld = log_d[origins].unsqueeze(-1)                        # (B, N, 1)
        return torch.cat([Xo, Xd, ld], dim=-1)                   # (B, N, 2F+1)

    @torch.no_grad()
    def eval_nll_cpc(idx: torch.Tensor) -> tuple[float, float]:
        model.eval()
        nll_total = 0.0
        Ti_total = 0.0
        F_pred_rows = []
        for s in range(0, idx.size(0), batch_origins):
            o_b = idx[s : s + batch_origins]
            X_pair = build_pair_features(o_b)
            scores = model(X_pair)                               # (B, N)
            # Mask self-loop
            self_mask = torch.zeros_like(scores)
            for k, oi in enumerate(o_b):
                self_mask[k, oi] = -1e9
            scores = scores + self_mask
            logp = F.log_softmax(scores, dim=-1)
            f_obs = F_ij[o_b]                                    # (B, N)
            t_b = T_i[o_b]                                       # (B,)
            valid = t_b > 0
            if valid.sum() == 0:
                continue
            p_obs = f_obs[valid] / t_b[valid].unsqueeze(-1).clamp(min=1e-6)
            nll_b = -(t_b[valid].unsqueeze(-1) * p_obs * logp[valid]).sum().item()
            nll_total += nll_b
            Ti_total += t_b[valid].sum().item()
            F_pred_rows.append((o_b[valid], (logp[valid].exp() * t_b[valid].unsqueeze(-1)).cpu()))
        # CPC on the validation rows
        pred = torch.zeros_like(F_ij).cpu()
        obs = F_ij.cpu()
        for o_v, p_v in F_pred_rows:
            for k, oi in enumerate(o_v.cpu().tolist()):
                pred[oi] = p_v[k]
        rows = idx.cpu().numpy()
        cpc = 2 * np.minimum(pred[rows].numpy(), obs[rows].numpy()).sum() / \
              (pred[rows].numpy().sum() + obs[rows].numpy().sum() + 1e-9)
        return nll_total / max(Ti_total, 1.0), float(cpc)

    history = {"epoch": [], "train_nll": [], "val_nll": [], "val_cpc": [], "lr": []}
    best_val = float("inf"); best_state = None; best_epoch = 0; bad_count = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(train_idx.size(0), device=device)]
        nll_train_total = 0.0; Ti_train = 0.0
        for s in range(0, perm.size(0), batch_origins):
            o_b = perm[s : s + batch_origins]
            X_pair = build_pair_features(o_b)
            scores = model(X_pair)
            self_mask = torch.zeros_like(scores)
            for k, oi in enumerate(o_b):
                self_mask[k, oi] = -1e9
            scores = scores + self_mask
            logp = F.log_softmax(scores, dim=-1)
            f_obs = F_ij[o_b]
            t_b = T_i[o_b]
            valid = t_b > 0
            if valid.sum() == 0: continue
            p_obs = f_obs[valid] / t_b[valid].unsqueeze(-1).clamp(min=1e-6)
            loss = -(t_b[valid].unsqueeze(-1) * p_obs * logp[valid]).sum() / t_b[valid].sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            nll_train_total += loss.item() * t_b[valid].sum().item()
            Ti_train += t_b[valid].sum().item()
        sched.step()
        train_nll = nll_train_total / max(Ti_train, 1.0)

        # eval every k epochs (cheap on T4 GPU)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            val_nll, val_cpc = eval_nll_cpc(val_idx)
            history["epoch"].append(ep); history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll); history["val_cpc"].append(val_cpc)
            history["lr"].append(opt.param_groups[0]["lr"])
            if ep % verbose_every == 0 or ep == 1 or ep == epochs:
                print(f"[daily] ep {ep:4d}/{epochs}  train_nll={train_nll:.4f}  "
                      f"val_nll={val_nll:.4f}  val_cpc={val_cpc:.4f}  lr={opt.param_groups[0]['lr']:.1e}")
            if val_nll < best_val - 1e-5:
                best_val = val_nll; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = ep; bad_count = 0
            else:
                bad_count += 1
                if bad_count >= patience // 5:
                    print(f"[daily] early stop at epoch {ep} (best epoch {best_epoch})")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_nll, final_val_cpc = eval_nll_cpc(val_idx)
    print(f"[daily] FINAL: best_epoch={best_epoch}, val_nll={final_val_nll:.4f}, val_cpc={final_val_cpc:.4f}")
    return model, {"history": history, "best_epoch": best_epoch,
                    "final_val_nll": final_val_nll, "final_val_cpc": final_val_cpc,
                    "n_params": n_params}


# ────────────────────────────────────────────────────────────────────────────
#  Hourly DG training (P(j|i,t) with sin h cos h embedding)
# ────────────────────────────────────────────────────────────────────────────

def train_hourly_dg(
    X_static: torch.Tensor,        # (N, F)
    F_ij_t: torch.Tensor,          # (T, N, N)
    log_d: torch.Tensor,           # (N, N)
    train_mask: torch.Tensor,      # (N,)
    val_mask: torch.Tensor,
    device: str,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_pairs: int = 16,         # (origin, hour) pairs per batch
    patience: int = 30,
    seed: int = 0,
    verbose_every: int = 5,
) -> tuple[DeepGravityFull, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    N, Fdim = X_static.shape
    T = F_ij_t.shape[0]
    model = DeepGravityFull(feat_dim=Fdim, n_time_features=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hourly] params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_static = X_static.to(device)
    F_ij_t = F_ij_t.to(device)
    log_d = log_d.to(device)
    train_idx = torch.where(train_mask)[0].to(device)
    val_idx = torch.where(val_mask)[0].to(device)

    # Time embedding (sin/cos)
    hours = torch.arange(T, device=device).float()
    sin_h = torch.sin(2 * math.pi * hours / 24)
    cos_h = torch.cos(2 * math.pi * hours / 24)

    # T_{i,t}
    T_it = F_ij_t.sum(dim=2)                                     # (T, N)

    def build_pair_features(o_b: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
        """o_b: (B,) origins, t_b: (B,) hours → (B, N, 2F+3)."""
        B = o_b.size(0)
        Xo = X_static[o_b].unsqueeze(1).expand(-1, N, -1)        # (B, N, F)
        Xd = X_static.unsqueeze(0).expand(B, -1, -1)             # (B, N, F)
        ld = log_d[o_b].unsqueeze(-1)                            # (B, N, 1)
        sh = sin_h[t_b].view(B, 1, 1).expand(-1, N, -1)          # (B, N, 1)
        ch = cos_h[t_b].view(B, 1, 1).expand(-1, N, -1)          # (B, N, 1)
        return torch.cat([Xo, Xd, ld, sh, ch], dim=-1)           # (B, N, 2F+3)

    @torch.no_grad()
    def eval_nll_cpc(idx: torch.Tensor) -> tuple[float, float]:
        model.eval()
        nll_total = 0.0; Tit_total = 0.0
        # Sum predicted F[t, i, j] for each (t, i) over j and compare CPC at val origins
        pred_t_i_j = torch.zeros((T, idx.size(0), N), device=device)
        for ti in range(T):
            for s in range(0, idx.size(0), batch_pairs):
                o_b = idx[s : s + batch_pairs]
                t_b = torch.full_like(o_b, ti)
                X_pair = build_pair_features(o_b, t_b)
                scores = model(X_pair)
                self_mask = torch.zeros_like(scores)
                for k, oi in enumerate(o_b):
                    self_mask[k, oi] = -1e9
                scores = scores + self_mask
                logp = F.log_softmax(scores, dim=-1)
                f_obs = F_ij_t[ti, o_b]
                t_it = T_it[ti, o_b]
                valid = t_it > 0
                if valid.sum() > 0:
                    p_obs = f_obs[valid] / t_it[valid].unsqueeze(-1).clamp(min=1e-6)
                    nll_b = -(t_it[valid].unsqueeze(-1) * p_obs * logp[valid]).sum().item()
                    nll_total += nll_b; Tit_total += t_it[valid].sum().item()
                # store predicted flow
                p_pred = logp.exp()
                F_pred_b = p_pred * t_it.unsqueeze(-1)
                pred_t_i_j[ti, s : s + o_b.size(0), :] = F_pred_b
        F_obs_sub = F_ij_t[:, idx, :]
        cpc = 2 * torch.minimum(pred_t_i_j, F_obs_sub).sum().item() / \
              (pred_t_i_j.sum().item() + F_obs_sub.sum().item() + 1e-9)
        return nll_total / max(Tit_total, 1.0), float(cpc)

    history = {"epoch": [], "train_nll": [], "val_nll": [], "val_cpc": [], "lr": []}
    best_val = float("inf"); best_state = None; best_epoch = 0; bad_count = 0

    # All (origin, hour) pairs in train
    train_pairs = torch.stack(torch.meshgrid(
        train_idx, torch.arange(T, device=device), indexing="ij"), dim=-1).reshape(-1, 2)

    for ep in range(1, epochs + 1):
        model.train()
        perm = train_pairs[torch.randperm(train_pairs.size(0), device=device)]
        nll_train_total = 0.0; Tit_train = 0.0
        for s in range(0, perm.size(0), batch_pairs):
            ot_b = perm[s : s + batch_pairs]
            o_b = ot_b[:, 0]; t_b = ot_b[:, 1]
            X_pair = build_pair_features(o_b, t_b)
            scores = model(X_pair)
            self_mask = torch.zeros_like(scores)
            for k, oi in enumerate(o_b):
                self_mask[k, oi] = -1e9
            scores = scores + self_mask
            logp = F.log_softmax(scores, dim=-1)
            f_obs = F_ij_t[t_b, o_b]
            t_it = T_it[t_b, o_b]
            valid = t_it > 0
            if valid.sum() == 0: continue
            p_obs = f_obs[valid] / t_it[valid].unsqueeze(-1).clamp(min=1e-6)
            loss = -(t_it[valid].unsqueeze(-1) * p_obs * logp[valid]).sum() / t_it[valid].sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            nll_train_total += loss.item() * t_it[valid].sum().item()
            Tit_train += t_it[valid].sum().item()
        sched.step()
        train_nll = nll_train_total / max(Tit_train, 1.0)

        if ep % 2 == 0 or ep == 1 or ep == epochs:
            val_nll, val_cpc = eval_nll_cpc(val_idx)
            history["epoch"].append(ep); history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll); history["val_cpc"].append(val_cpc)
            history["lr"].append(opt.param_groups[0]["lr"])
            if ep % verbose_every == 0 or ep == 1 or ep == epochs:
                print(f"[hourly] ep {ep:4d}/{epochs}  train_nll={train_nll:.4f}  "
                      f"val_nll={val_nll:.4f}  val_cpc={val_cpc:.4f}  lr={opt.param_groups[0]['lr']:.1e}")
            if val_nll < best_val - 1e-5:
                best_val = val_nll; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = ep; bad_count = 0
            else:
                bad_count += 1
                if bad_count >= patience // 2:
                    print(f"[hourly] early stop at epoch {ep} (best epoch {best_epoch})")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_nll, final_val_cpc = eval_nll_cpc(val_idx)
    print(f"[hourly] FINAL: best_epoch={best_epoch}, val_nll={final_val_nll:.4f}, val_cpc={final_val_cpc:.4f}")
    return model, {"history": history, "best_epoch": best_epoch,
                    "final_val_nll": final_val_nll, "final_val_cpc": final_val_cpc,
                    "n_params": n_params}


# ────────────────────────────────────────────────────────────────────────────
#  Predict full F̂ tensors for downstream eval
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_daily(model: DeepGravityFull, X_static, log_d, F_ij, device,
                   batch: int = 32) -> np.ndarray:
    model.eval()
    N = X_static.size(0)
    X_static = X_static.to(device); log_d = log_d.to(device); F_ij = F_ij.to(device)
    T_i = F_ij.sum(dim=1)
    pred = torch.zeros((N, N), device=device)
    for s in range(0, N, batch):
        o_b = torch.arange(s, min(s + batch, N), device=device)
        Xo = X_static[o_b].unsqueeze(1).expand(-1, N, -1)
        Xd = X_static.unsqueeze(0).expand(o_b.size(0), -1, -1)
        ld = log_d[o_b].unsqueeze(-1)
        X_pair = torch.cat([Xo, Xd, ld], dim=-1)
        scores = model(X_pair)
        for k, oi in enumerate(o_b):
            scores[k, oi] = -1e9
        p = F.softmax(scores, dim=-1)
        pred[o_b] = p * T_i[o_b].unsqueeze(-1)
    return pred.cpu().numpy()


@torch.no_grad()
def predict_hourly(model: DeepGravityFull, X_static, log_d, F_ij_t, device,
                    batch: int = 16) -> np.ndarray:
    model.eval()
    N = X_static.size(0); T = F_ij_t.size(0)
    X_static = X_static.to(device); log_d = log_d.to(device); F_ij_t = F_ij_t.to(device)
    T_it = F_ij_t.sum(dim=2)
    pred = torch.zeros((T, N, N), device=device)
    hours = torch.arange(T, device=device).float()
    sin_h = torch.sin(2 * math.pi * hours / 24)
    cos_h = torch.cos(2 * math.pi * hours / 24)
    for ti in range(T):
        for s in range(0, N, batch):
            o_b = torch.arange(s, min(s + batch, N), device=device)
            Xo = X_static[o_b].unsqueeze(1).expand(-1, N, -1)
            Xd = X_static.unsqueeze(0).expand(o_b.size(0), -1, -1)
            ld = log_d[o_b].unsqueeze(-1)
            sh = sin_h[ti].view(1, 1, 1).expand(o_b.size(0), N, 1)
            ch = cos_h[ti].view(1, 1, 1).expand(o_b.size(0), N, 1)
            X_pair = torch.cat([Xo, Xd, ld, sh, ch], dim=-1)
            scores = model(X_pair)
            for k, oi in enumerate(o_b):
                scores[k, oi] = -1e9
            p = F.softmax(scores, dim=-1)
            pred[ti, o_b] = p * T_it[ti, o_b].unsqueeze(-1)
    return pred.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=str, default="dg_colab_bundle.npz")
    parser.add_argument("--out", type=str, default="dg_predictions.npz")
    parser.add_argument("--epochs_daily", type=int, default=1000)
    parser.add_argument("--epochs_hourly", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_daily", action="store_true")
    parser.add_argument("--skip_hourly", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA available, training will be VERY slow.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading bundle from {args.bundle} ...")
    data = np.load(args.bundle)
    X_static = torch.from_numpy(data["X_static"]).float()
    F_ij_t = torch.from_numpy(data["F_ij_t"]).float()
    log_d = torch.from_numpy(data["log_d"]).float()
    train_mask = torch.from_numpy(data["train_mask"]).bool()
    val_mask = torch.from_numpy(data["val_mask"]).bool()
    test_mask = torch.from_numpy(data["test_mask"]).bool()

    N, Fdim = X_static.shape
    T = F_ij_t.shape[0]
    print(f"  N={N}, T={T}, F_static={Fdim}")
    print(f"  train origins={int(train_mask.sum())}/{N}")
    print(f"  val origins={int(val_mask.sum())}/{N}")
    print(f"  test origins={int(test_mask.sum())}/{N}")

    F_ij_daily = F_ij_t.sum(dim=0)                                # (N, N)

    out: dict[str, np.ndarray] = {}

    if not args.skip_daily:
        print("\n" + "=" * 70)
        print("Training DAILY Deep Gravity (15 hidden × 256, paper-spec)")
        print("=" * 70)
        t0 = time.time()
        model_daily, log_daily = train_daily_dg(
            X_static, F_ij_daily, log_d, train_mask, val_mask, device,
            epochs=args.epochs_daily, seed=args.seed,
        )
        wall = time.time() - t0
        print(f"[daily] wall time: {wall/60:.1f} min")
        pred_daily = predict_daily(model_daily, X_static, log_d, F_ij_daily, device)
        out["pred_daily"] = pred_daily.astype(np.float32)
        out["meta_daily"] = np.array(json.dumps({
            "n_params": log_daily["n_params"],
            "best_epoch": log_daily["best_epoch"],
            "final_val_nll": log_daily["final_val_nll"],
            "final_val_cpc": log_daily["final_val_cpc"],
            "wall_min": wall / 60,
            "history": log_daily["history"],
        }))

    if not args.skip_hourly:
        print("\n" + "=" * 70)
        print("Training HOURLY Deep Gravity (15 hidden × 256 + sin/cos h)")
        print("=" * 70)
        t0 = time.time()
        model_hourly, log_hourly = train_hourly_dg(
            X_static, F_ij_t, log_d, train_mask, val_mask, device,
            epochs=args.epochs_hourly, seed=args.seed,
        )
        wall = time.time() - t0
        print(f"[hourly] wall time: {wall/60:.1f} min")
        pred_hourly = predict_hourly(model_hourly, X_static, log_d, F_ij_t, device)
        out["pred_hourly"] = pred_hourly.astype(np.float32)
        out["meta_hourly"] = np.array(json.dumps({
            "n_params": log_hourly["n_params"],
            "best_epoch": log_hourly["best_epoch"],
            "final_val_nll": log_hourly["final_val_nll"],
            "final_val_cpc": log_hourly["final_val_cpc"],
            "wall_min": wall / 60,
            "history": log_hourly["history"],
        }))

    out["train_mask"] = train_mask.numpy()
    out["val_mask"] = val_mask.numpy()
    out["test_mask"] = test_mask.numpy()

    np.savez_compressed(args.out, **out)
    print(f"\nWrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print("Download this file back to your local machine, then run:")
    print("  python experiments/paper_a/compare_dg_vs_dual_het.py")


if __name__ == "__main__":
    main()
