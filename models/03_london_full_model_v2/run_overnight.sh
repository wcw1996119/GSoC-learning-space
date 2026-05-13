#!/usr/bin/env bash
# Overnight pipeline for Paper A.
# Sequential execution; each step's output triggers the next.
# State files in evaluation_outputs/paper_a/ enable --resume on partial fails.

# Removed set -u (was causing exit on unset PYTHONPATH at startup).
# We DON'T set -e either — we want partial progress through stages.
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="D:/Users/Chengwei/anaconda3/envs/mesa-demo/python.exe"
LOGDIR="$ROOT/evaluation_outputs/paper_a"
mkdir -p "$LOGDIR"

NIGHT_LOG="$LOGDIR/overnight_run.log"
SUMMARY_MD="$LOGDIR/overnight_summary.md"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "===== Overnight Paper A pipeline =====" | tee -a "$NIGHT_LOG"
echo "Started:   $(ts)" | tee -a "$NIGHT_LOG"
echo "ROOT:      $ROOT" | tee -a "$NIGHT_LOG"
echo "" | tee -a "$NIGHT_LOG"

# Step 1: Synthetic recovery — W4 hard checkpoint, n_seeds=20
echo "----- [1/5] synthetic_recovery (W4 gate, n_seeds=20) -----" | tee -a "$NIGHT_LOG"
echo "$(ts) START" | tee -a "$NIGHT_LOG"
"$PYTHON" experiments/paper_a/synthetic_recovery.py --n_seeds 20 --device cpu \
  >> "$NIGHT_LOG" 2>&1
SYN_RC=$?
echo "$(ts) END synthetic_recovery rc=$SYN_RC" | tee -a "$NIGHT_LOG"
echo "" | tee -a "$NIGHT_LOG"

# Step 2: GNN ablation — W8 GO/NO-GO, n_seeds=5 (smaller because real London data is heavier)
echo "----- [2/5] gnn_ablation (W8 gate, n_seeds=5) -----" | tee -a "$NIGHT_LOG"
echo "$(ts) START" | tee -a "$NIGHT_LOG"
"$PYTHON" experiments/paper_a/gnn_ablation.py --n_seeds 5 --device cpu \
  >> "$NIGHT_LOG" 2>&1
GNN_RC=$?
echo "$(ts) END gnn_ablation rc=$GNN_RC" | tee -a "$NIGHT_LOG"
echo "" | tee -a "$NIGHT_LOG"

# Step 3: Choice-set ablation
echo "----- [3/5] choice_set_ablation -----" | tee -a "$NIGHT_LOG"
echo "$(ts) START" | tee -a "$NIGHT_LOG"
"$PYTHON" experiments/paper_a/choice_set_ablation.py --n_seeds 3 --device cpu \
  >> "$NIGHT_LOG" 2>&1
CSA_RC=$?
echo "$(ts) END choice_set_ablation rc=$CSA_RC" | tee -a "$NIGHT_LOG"
echo "" | tee -a "$NIGHT_LOG"

# Step 4: Temporal holdout
echo "----- [4/5] temporal_holdout -----" | tee -a "$NIGHT_LOG"
echo "$(ts) START" | tee -a "$NIGHT_LOG"
"$PYTHON" experiments/paper_a/temporal_holdout.py --n_seeds 3 --device cpu \
  >> "$NIGHT_LOG" 2>&1
TH_RC=$?
echo "$(ts) END temporal_holdout rc=$TH_RC" | tee -a "$NIGHT_LOG"
echo "" | tee -a "$NIGHT_LOG"

# Step 5: Validation orchestrator
echo "----- [5/5] run_all_validation -----" | tee -a "$NIGHT_LOG"
echo "$(ts) START" | tee -a "$NIGHT_LOG"
"$PYTHON" validation/paper_a/run_all_validation.py \
  >> "$NIGHT_LOG" 2>&1
VAL_RC=$?
echo "$(ts) END run_all_validation rc=$VAL_RC" | tee -a "$NIGHT_LOG"
echo "" | tee -a "$NIGHT_LOG"

# Final: build summary markdown
echo "----- writing overnight_summary.md -----" | tee -a "$NIGHT_LOG"
"$PYTHON" data/scripts/write_overnight_summary.py \
  --log "$NIGHT_LOG" \
  --out "$SUMMARY_MD" \
  --syn_rc $SYN_RC --gnn_rc $GNN_RC --csa_rc $CSA_RC --th_rc $TH_RC --val_rc $VAL_RC \
  >> "$NIGHT_LOG" 2>&1
SUM_RC=$?
echo "$(ts) END summary rc=$SUM_RC" | tee -a "$NIGHT_LOG"

echo "" | tee -a "$NIGHT_LOG"
echo "===== Overnight pipeline complete =====" | tee -a "$NIGHT_LOG"
echo "Finished: $(ts)" | tee -a "$NIGHT_LOG"
echo "synthetic_recovery rc=$SYN_RC; gnn_ablation rc=$GNN_RC; choice_set rc=$CSA_RC; temporal rc=$TH_RC; validation rc=$VAL_RC; summary rc=$SUM_RC" | tee -a "$NIGHT_LOG"
echo "Summary: $SUMMARY_MD" | tee -a "$NIGHT_LOG"

# Post-overnight sanity check — added 2026-05-05 because the previous
# run had rc=0 on stages 2-5 but all four were silent fails. This script
# checks each stage CSV/JSON has real content (not placeholder, not empty).
# Sets OVERNIGHT_HEALTHY=1 if all checks pass, 0 otherwise. Does NOT exit
# the shell; lets the user (or CI) decide based on the printed verdict.
echo "" | tee -a "$NIGHT_LOG"
echo "----- Sanity check: are stages 2-5 really populated? -----" | tee -a "$NIGHT_LOG"
"$PYTHON" evaluation_outputs/paper_a/check_overnight_outputs.py \
  >> "$NIGHT_LOG" 2>&1
CHECK_RC=$?
if [ $CHECK_RC -eq 0 ]; then
  echo "$(ts) Sanity check PASSED — all stages produced real output" | tee -a "$NIGHT_LOG"
else
  echo "$(ts) Sanity check FAILED — see $NIGHT_LOG (one or more stages silent-failed)" | tee -a "$NIGHT_LOG"
fi
