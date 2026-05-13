"""Reload phase_b_v6_seed0.pt and verify it predicts the same val_CPC.

Round-trip check before D3 uses the checkpoint as a frozen scenario backbone.

Run:
    python experiments/paper_a/smoke_reload_phase_b_ckpt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN

from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, cpc, load_data


def main() -> None:
    ckpt_path = ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"
    print(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    saved_metrics = ckpt["metrics"]
    print(f"  config: {cfg}")
    print(f"  saved metrics: val_cpc={saved_metrics['val_cpc']:.4f} "
          f"beta_base={saved_metrics['beta_base']:+.4f} "
          f"phi={saved_metrics['phi']:+.4f} psi={saved_metrics['psi']:+.4f}")

    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)

    util_net = StructuralGNN(
        in_features=cfg["in_features"], hidden=cfg["hidden"],
        out=cfg["out"], depth=cfg["depth"],
    )
    # Build a fresh trainer just to host the rum head + predict_OD path.
    # Important: do NOT call .fit() — we only want the inference pipeline.
    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=data["F_ij"],
        t_ij_t=data["t_ij"],
        log_d_ij=data["log_d"],
        K=cfg["K"],
        utility_net=util_net,
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        device="cpu",
        seed=0,
        epochs=1, patience=1,         # never used; we skip fit
        residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=cfg["enable_wage_attraction"],
        enforce_mainstream_direction=cfg["enforce_mainstream_direction"],
    )
    trainer.gnn.load_state_dict(ckpt["gnn_state"])
    trainer.rum.load_state_dict(ckpt["rum_state"])
    trainer.gnn.eval(); trainer.rum.eval()

    # Reload-side scalars
    phi = float(trainer.rum.phi.item())
    psi = float(trainer.rum.psi.item())
    beta_base = float(trainer.rum.beta_t.mean().item())
    beta_c = float(trainer.rum.beta_c.item())
    print(f"  reloaded   : beta_base={beta_base:+.4f} phi={phi:+.4f} psi={psi:+.4f} "
          f"beta_c={beta_c:+.4f}")

    # CPC round-trip
    pred = trainer.predict_OD().detach().cpu().numpy()
    pred_F = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else (pred.sum(0) if pred.ndim == 3 else pred)
    val_mask_np = data["val_mask"].cpu().numpy().astype(bool)
    val_cpc_reload = cpc(data["F_ij"].cpu().numpy(), pred_F, val_mask_np)
    print(f"  val_CPC reload={val_cpc_reload:.4f}  saved={saved_metrics['val_cpc']:.4f}  "
          f"|Δ|={abs(val_cpc_reload - saved_metrics['val_cpc']):.2e}")

    assert abs(val_cpc_reload - saved_metrics['val_cpc']) < 1e-3, \
        "Reload CPC drift > 1e-3 — ckpt round-trip broken"
    assert abs(phi - saved_metrics['phi']) < 1e-5
    assert abs(psi - saved_metrics['psi']) < 1e-5
    print("== ALL OK ==")


if __name__ == "__main__":
    main()
