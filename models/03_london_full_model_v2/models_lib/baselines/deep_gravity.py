"""Simini et al. 2021 (Nature Comms) Deep Gravity model.

For each origin i, an MLP scores every candidate destination j as a
utility V_ij from concat(feat_i, feat_j, log(t_ij+1), log(d_ij+1)). The
prediction is T_i * softmax_j(V_ij), i.e. an origin-conditional
multinomial. We train with cross-entropy on the empirical destination
distribution, weighted by observed origin total T_i.

Architecture per paper: 2 hidden layers of 256 then 128 ReLU units.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


class _DGNet(nn.Module):
    def __init__(self, in_dim: int, h1: int = 256, h2: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepGravity:
    def __init__(
        self,
        h1: int = 256,
        h2: int = 128,
        epochs: int = 100,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.h1 = h1
        self.h2 = h2
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.net_: Optional[_DGNet] = None
        self.feat_mean_: Optional[np.ndarray] = None
        self.feat_std_: Optional[np.ndarray] = None

    def _pair_X(
        self,
        oi: np.ndarray,
        dj: np.ndarray,
        node_feat: np.ndarray,
        t_ij: np.ndarray,
        d_ij: np.ndarray,
    ) -> np.ndarray:
        f_o = node_feat[oi]
        f_d = node_feat[dj]
        log_t = np.log(np.asarray(t_ij, float) + 1.0).reshape(-1, 1)
        log_d = np.log(np.asarray(d_ij, float) + 1.0).reshape(-1, 1)
        return np.concatenate([f_o, f_d, log_t, log_d], axis=1).astype(np.float32)

    def fit(
        self,
        origins_idx: np.ndarray,
        dests_idx: np.ndarray,
        node_feat: np.ndarray,
        t_ij: np.ndarray,
        d_ij: np.ndarray,
        flow: np.ndarray,
    ) -> "DeepGravity":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build per-origin destination groups for softmax.
        df = pd.DataFrame(
            {"o": origins_idx, "d": dests_idx, "t": t_ij, "dist": d_ij, "f": flow}
        )
        groups: Dict[int, pd.DataFrame] = {o: g for o, g in df.groupby("o")}

        X_full = self._pair_X(
            df["o"].to_numpy(), df["d"].to_numpy(), node_feat,
            df["t"].to_numpy(), df["dist"].to_numpy(),
        )
        self.feat_mean_ = X_full.mean(axis=0)
        self.feat_std_ = X_full.std(axis=0) + 1e-6
        in_dim = X_full.shape[1]
        self.net_ = _DGNet(in_dim, self.h1, self.h2).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        # Pre-build per-origin tensors.
        origin_batches: List[Dict[str, torch.Tensor]] = []
        for o, g in groups.items():
            Xg = self._pair_X(
                g["o"].to_numpy(), g["d"].to_numpy(), node_feat,
                g["t"].to_numpy(), g["dist"].to_numpy(),
            )
            Xg = (Xg - self.feat_mean_) / self.feat_std_
            f = g["f"].to_numpy(dtype=np.float32)
            T_i = f.sum()
            if T_i <= 0:
                continue
            origin_batches.append({
                "X": torch.from_numpy(Xg).to(self.device),
                "p": torch.from_numpy(f / T_i).to(self.device),
                "T": float(T_i),
            })

        for _ in range(self.epochs):
            perm = np.random.permutation(len(origin_batches))
            for k in perm:
                b = origin_batches[k]
                opt.zero_grad()
                v = self.net_(b["X"])
                logp = torch.log_softmax(v, dim=0)
                # Weighted cross-entropy: -T_i * sum_j p_obs * logp_pred.
                loss = -(b["T"] * (b["p"] * logp).sum())
                loss.backward()
                opt.step()
        return self

    def predict(
        self,
        origins_idx: np.ndarray,
        dests_idx: np.ndarray,
        node_feat: np.ndarray,
        t_ij: np.ndarray,
        d_ij: np.ndarray,
        T_i: np.ndarray,
    ) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("DeepGravity.fit must be called first")
        X = self._pair_X(origins_idx, dests_idx, node_feat, t_ij, d_ij)
        Xn = (X - self.feat_mean_) / self.feat_std_
        self.net_.eval()
        with torch.no_grad():
            v = self.net_(torch.from_numpy(Xn).to(self.device)).cpu().numpy()
        df = pd.DataFrame({"o": origins_idx, "v": v, "T": T_i})
        # softmax per origin
        df["v"] = df.groupby("o")["v"].transform(lambda s: s - s.max())
        df["e"] = np.exp(df["v"].to_numpy())
        s = df.groupby("o")["e"].transform("sum").replace(0, np.nan)
        df["p"] = (df["e"] / s).fillna(0.0)
        return (df["p"] * df["T"]).to_numpy()
