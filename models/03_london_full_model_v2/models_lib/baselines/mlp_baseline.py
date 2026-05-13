"""Plain feed-forward MLP baseline (no graph structure).

Input per OD pair: concat(node_feat[i], node_feat[j], log(t_ij+1),
log(d_ij+1)). Two hidden layers of 64 ReLU units. Output is log(1+flow).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class _MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPBaseline:
    def __init__(
        self,
        hidden: int = 64,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 4096,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.net_: Optional[_MLPNet] = None
        self.feat_mean_: Optional[np.ndarray] = None
        self.feat_std_: Optional[np.ndarray] = None

    def _build_X(
        self,
        origins_idx: np.ndarray,
        dests_idx: np.ndarray,
        node_feat: np.ndarray,
        t_ij: np.ndarray,
        d_ij: np.ndarray,
    ) -> np.ndarray:
        f_o = node_feat[origins_idx]
        f_d = node_feat[dests_idx]
        log_t = np.log(np.asarray(t_ij, dtype=float) + 1.0).reshape(-1, 1)
        log_d = np.log(np.asarray(d_ij, dtype=float) + 1.0).reshape(-1, 1)
        return np.concatenate([f_o, f_d, log_t, log_d], axis=1).astype(np.float32)

    def fit(
        self,
        origins_idx: np.ndarray,
        dests_idx: np.ndarray,
        node_feat: np.ndarray,
        t_ij: np.ndarray,
        d_ij: np.ndarray,
        flow: np.ndarray,
    ) -> "MLPBaseline":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = self._build_X(origins_idx, dests_idx, node_feat, t_ij, d_ij)
        y = np.log1p(np.asarray(flow, dtype=float)).astype(np.float32)

        self.feat_mean_ = X.mean(axis=0)
        self.feat_std_ = X.std(axis=0) + 1e-6
        Xn = (X - self.feat_mean_) / self.feat_std_

        Xt = torch.from_numpy(Xn).to(self.device)
        yt = torch.from_numpy(y).to(self.device)

        self.net_ = _MLPNet(in_dim=Xn.shape[1], hidden=self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        n = Xn.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                opt.zero_grad()
                pred = self.net_(Xt[idx])
                loss = loss_fn(pred, yt[idx])
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
    ) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("MLPBaseline.fit must be called first")
        X = self._build_X(origins_idx, dests_idx, node_feat, t_ij, d_ij)
        Xn = (X - self.feat_mean_) / self.feat_std_
        self.net_.eval()
        with torch.no_grad():
            pred = self.net_(torch.from_numpy(Xn).to(self.device)).cpu().numpy()
        return np.expm1(np.maximum(pred, 0.0))
