#!/usr/bin/env bash
# Wang TB-ResNet δ_max sweep — 6 configs × 3 seeds × 200 epochs
# Maps theory-dominance vs fit-quality trade-off.
#
# Usage from /root/autodl-tmp/wang_sprint:
#   bash experiments/paper_a/wang_sweep.sh
#
# Expected runtime on RTX 3090: ~12-15 min per config × 6 = ~75-90 min total.
#
# Outputs:
#   evaluation_outputs/paper_a/wang_sweep/blend_max_0p0/...
#   evaluation_outputs/paper_a/wang_sweep/blend_max_0p1/...
#   ...
#   evaluation_outputs/paper_a/wang_sweep/blend_max_1p0/...
#
# After all 6 finish:
#   python experiments/paper_a/wang_sweep_evaluate.py
# → produces multi-proxy trade-off table.

set -e

SWEEP_ROOT="evaluation_outputs/paper_a/wang_sweep"
mkdir -p "$SWEEP_ROOT"

run_one() {
    local bm="$1"
    local tag="$2"
    local out_dir="$SWEEP_ROOT/blend_max_$tag"

    echo
    echo "================================================================"
    echo "= blend_max = $bm  (tag $tag)"
    echo "================================================================"
    python experiments/paper_a/run_dual_het_ablation.py \
        --config baseline_tb_resnet \
        --seeds 0 1 2 \
        --epochs 200 \
        --patience 30 \
        --cache_name demo_cache.npz \
        --device cuda \
        --blend_max "$bm" \
        --out_dir "$out_dir" \
        2>&1 | tee "$out_dir.log"
}

run_one 0.0 0p0
run_one 0.1 0p1
run_one 0.2 0p2
run_one 0.3 0p3
run_one 0.5 0p5
run_one 1.0 1p0

echo
echo "================================================================"
echo "= sweep complete. running evaluation..."
echo "================================================================"
python experiments/paper_a/wang_sweep_evaluate.py
