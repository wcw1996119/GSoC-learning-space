#!/usr/bin/env bash
# Smoke test wrapper. Sets PYTHONPATH explicitly because background bash
# does not propagate inline env reliably on this system.
set -e
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
exec "D:/Users/Chengwei/anaconda3/envs/mesa-demo/python.exe" experiments/paper_a/synthetic_recovery.py --n_seeds 1 --device cpu "$@"
