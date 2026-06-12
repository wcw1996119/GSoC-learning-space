#!/bin/bash
# 全套可识别性 profile (本地 CPU, mesa-demo env). 结果追加到 _profile_results.txt
cd "$(dirname "$0")/.."
OUT=methodology/_profile_results.txt
echo "=== profile run $(date) ===" > $OUT
run() {
  echo ""; echo ">>> $* <<<"
  PYTHONIOENCODING=utf-8 conda run -n mesa-demo python experiments/profile_param_reopt.py "$@" 2>&1 \
    | grep -v -i "warning" | grep -E "param=|span|^ +[-0-9]"
}
{
  run --param gamma_M   --tier 0
  run --param gamma_M   --tier 1
  run --param gamma_M   --tier 2
  run --param T_max     --tier 0
  run --param T_max     --tier 1
  run --param T_max     --tier 2
  run --param self_loop
  run --param alpha_W   --tier 1
  run --param nu_D      --tier 1
} | tee -a $OUT
echo "ALL_PROFILES_DONE" | tee -a $OUT
