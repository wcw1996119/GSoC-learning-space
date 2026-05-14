"""Pack v3 nested-logit smoke for AutoDL.

Creates `data/autodl_bundle/v3_nested_smoke_<timestamp>.tar.gz` containing:
- Slim v2 data (~35 MB): demo_cache_minimal.npz (drops huge prediction tensors)
  + grid_hourly_od_2019.npz + grid_dynamic_features.npz + hourly_node_features.npz
  + car_freeflow_t_ij.npy + paperA_v23_aux.npz + mode_inputs.npz
- v2 code dependencies: dual_branch_encoder, structural_gnn,
  dual_branch_mixture_trainer (for make_distance_aware_mode_share),
  compare_four_variants (for load_data), inverse_trainer + flow_metrics
- v3 code: nested_logit_head + train_nested_smoke
- README_AUTODL.md with run instructions

Run locally:
    python experiments/paper_a/pack_autodl_bundle.py

Result: data/autodl_bundle/v3_nested_smoke_<ts>.tar.gz (~35-40 MB)
"""
from __future__ import annotations

import datetime
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"


def make_demo_cache_minimal(src: Path, dst: Path):
    """Strip huge prediction tensors from demo_cache.npz, keep what load_data needs."""
    print(f"  slimming demo_cache.npz: {src} -> {dst}")
    d = np.load(src, allow_pickle=True)
    keep = [
        "static_features", "static_mean", "static_std",
        "train_mask", "val_mask", "test_mask",
        "coords_bng", "grid_ids", "boroughs", "grid_borough_idx",
        "borough_hourly_congestion", "pi_t",
    ]
    out = {k: d[k] for k in keep if k in d.files}
    np.savez_compressed(dst, **out)
    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    print(f"  {src_mb:.1f} MB -> {dst_mb:.2f} MB")


def main():
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = V3_ROOT / "data" / "autodl_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Build bundle/ tree mirroring v2 + v3 layout for transparent run on AutoDL.
        bundle = td / "v3_nested_smoke"
        v2_dir = bundle / "03_london_full_model_v2"
        v3_dir = bundle / "04_london_nested_logit_v3"

        # ---- v2 data (slim) ----
        v2_data = v2_dir / "data" / "processed"
        v2_data.mkdir(parents=True, exist_ok=True)
        print("[data]")
        make_demo_cache_minimal(
            V2_ROOT / "data" / "processed" / "demo_cache.npz",
            v2_data / "demo_cache.npz",
        )
        for fname in (
            "grid_hourly_od_2019.npz",
            "grid_dynamic_features.npz",
            "hourly_node_features.npz",
            "car_freeflow_t_ij.npy",
            "paperA_v23_aux.npz",
            "mode_inputs.npz",
        ):
            src = V2_ROOT / "data" / "processed" / fname
            shutil.copy2(src, v2_data / fname)
            print(f"  copied {fname} ({src.stat().st_size/1e6:.2f} MB)")

        # ---- v2 code (necessary deps) ----
        v2_code_pairs = [
            ("models_lib/__init__.py", None),  # may not exist
            ("models_lib/inverse_rum/__init__.py", None),  # replaced with empty below
            ("models_lib/inverse_rum/dual_branch_encoder.py", None),
            ("models_lib/inverse_rum/structural_gnn.py", None),
            ("models_lib/inverse_rum/dual_branch_trainer.py", None),  # compare_four_variants imports this
            ("models_lib/inverse_rum/dual_branch_mixture_trainer.py", None),
            ("models_lib/inverse_rum/inverse_trainer.py", None),  # imports TrainingLog
            ("models_lib/inverse_rum/mixture_head.py", None),  # imported by trainer
            ("models_lib/inverse_rum/irm_penalty.py", None),  # imported by trainer
            ("models_lib/inverse_rum/flow_metrics.py", None),  # may not be needed but cheap
            # inverse_trainer top-level imports these — must bundle or import fails:
            ("models_lib/inverse_rum/implicit_softmax.py", None),
            ("models_lib/inverse_rum/topk_choice.py", None),
            ("experiments/__init__.py", None),
            ("experiments/paper_a/__init__.py", None),
            ("experiments/paper_a/compare_four_variants.py", None),
        ]
        print("[v2 code]")
        for rel, _ in v2_code_pairs:
            src = V2_ROOT / rel
            dst = v2_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith("inverse_rum/__init__.py"):
                # v2's real __init__.py eagerly imports many submodules
                # (implicit_softmax, topk_choice, bpr_layer, accessibility, …)
                # that this smoke doesn't need. Ship an empty __init__.py so
                # Python treats the directory as a package without triggering
                # those imports.
                dst.touch()
                print(f"  wrote empty {rel} (avoid eager imports we don't bundle)")
            elif src.exists():
                shutil.copy2(src, dst)
                print(f"  copied {rel}")
            else:
                if rel.endswith("__init__.py"):
                    dst.touch()
                    print(f"  created empty {rel}")
                else:
                    print(f"  WARNING: missing {rel}")

        # ---- v3 code ----
        v3_code_pairs = [
            ("models_lib/inverse_rum/nested_logit_head.py", None),     # partial nested (smoke v1, retained for ablation)
            ("models_lib/inverse_rum/dm_nested_logit_head.py", None),  # D→M nested + borough λ
            ("experiments/paper_a/train_nested_smoke.py", None),
            ("experiments/paper_a/train_dm_nested.py", None),
            ("experiments/paper_a/build_mode_level_features.py", None),
        ]
        v3_data_pairs = [
            ("data/processed/mode_level_features.npz", None),          # built locally, ship to AutoDL
        ]
        # __init__.py files for v3
        for rel in ("models_lib/__init__.py", "models_lib/inverse_rum/__init__.py",
                    "experiments/__init__.py", "experiments/paper_a/__init__.py"):
            (v3_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (v3_dir / rel).touch()

        print("[v3 code]")
        for rel, _ in v3_code_pairs:
            src = V3_ROOT / rel
            dst = v3_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {rel}")

        print("[v3 data]")
        for rel, _ in v3_data_pairs:
            src = V3_ROOT / rel
            dst = v3_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  copied {rel} ({src.stat().st_size/1024:.1f} KB)")
            else:
                print(f"  WARNING: missing {rel} — run build_mode_level_features.py first")

        # ---- README + run script ----
        readme = bundle / "README_AUTODL.md"
        readme.write_text("""# v3 Nested Logit Smoke on AutoDL

Extract this bundle on the AutoDL machine, then run the training script.

## Setup

```bash
# 1. Extract on AutoDL (assumes you scp'd this bundle to /root/autodl-tmp/)
cd /root/autodl-tmp
tar xzf v3_nested_smoke_<timestamp>.tar.gz
cd v3_nested_smoke

# 2. (Optional) Make sure deps are present in your AutoDL env:
#    python -c "import torch, numpy"
```

## Run

```bash
cd /root/autodl-tmp/v3_nested_smoke/04_london_nested_logit_v3
python experiments/paper_a/train_nested_smoke.py \\
    --epochs 200 --patience 30 --seed 0 \\
    --blend-max 1.0 \\
    --lambda-init 0.99 --lambda-eps-min 0.05 \\
    --device cuda --verbose
```

Expected GPU time: ~5-10 min on RTX 3090 for 200 epochs.

Output: `04_london_nested_logit_v3/evaluation_outputs/paper_a/nested_logit_smoke.json`

## Bring result back

```bash
# From your local machine:
scp -P <port> root@<autodl-host>:/root/autodl-tmp/v3_nested_smoke/04_london_nested_logit_v3/evaluation_outputs/paper_a/nested_logit_smoke.json ./
```

## What this tests

Hypothesis: nested logit's λ_m (per-mode destination softmax temperature) increases
V_RUM expressiveness and reduces GNN's learned blend share δ.

Verdict written to JSON's `verdict` field + printed at end of run.

If hypothesis confirmed (some λ_m < 0.8 AND δ < 0.40), commit to nested logit
direction. Otherwise reconsider.
""")
        print("[meta]")
        print(f"  README_AUTODL.md created")

        # ---- pack tar.gz ----
        out_path = out_dir / f"v3_nested_smoke_{ts}.tar.gz"
        print(f"[pack] -> {out_path}")
        with tarfile.open(out_path, "w:gz") as tf:
            tf.add(bundle, arcname="v3_nested_smoke")
        size_mb = out_path.stat().st_size / 1e6
        print(f"\nDone: {out_path}  ({size_mb:.1f} MB)")
        print(f"\nUpload to AutoDL:")
        print(f"  scp -P <port> {out_path.as_posix()} root@<host>:/root/autodl-tmp/")


if __name__ == "__main__":
    main()
