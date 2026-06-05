"""打包 AutoDL bundle: 代码 + 处理后数据 -> v4_beijing_bundle.tar.gz

⚠ GOVERNANCE: bundle 含北京 OD 派生数据 (data/processed/*.npz)。上传 AutoDL = 出库,
   需用户明确授权 (2026-07 红线)。本脚本只打包不上传。

用法: python data/scripts/pack_autodl_bundle.py
解包后在 AutoDL: 见 AUTODL_RUN.md
"""
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # 项目根 05_beijing_nested_logit_v4/
OUT = ROOT / "data" / "v4_beijing_bundle.tar.gz"

INCLUDE = [
    "experiments/beijing_model.py",
    "experiments/train_beijing.py",
    "data/processed/beijing_grid.npz",
    "data/processed/beijing_edges.npz",
    "data/processed/beijing_modes.npz",
    "data/processed/beijing_aux.npz",
    "data/processed/beijing_income.npz",
    "data/processed/beijing_occupation.npz",
    "data/processed/beijing_transit_od.npz",   # 刷卡锚需要
    "AUTODL_RUN.md",
]

def main():
    with tarfile.open(OUT, "w:gz") as tar:
        for rel in INCLUDE:
            p = ROOT / rel
            if not p.exists():
                print(f"  [skip] {rel} 不存在"); continue
            tar.add(p, arcname=f"v4_beijing/{rel}")
            print(f"  + {rel}  ({p.stat().st_size/1e6:.1f}MB)")
    print(f"[OK] {OUT}  ({OUT.stat().st_size/1e6:.1f}MB)")

if __name__ == "__main__":
    main()
