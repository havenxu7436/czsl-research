#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs "$CPF_WORK"
exec 9>logs/queue.lock
flock -n 9 || { printf '%s\n' 'Queue already running'; exit 1; }
if [[ "$#" -eq 0 ]]; then set -- 0 1 2; fi
if [[ "$#" -ne 3 ]]; then printf '%s\n' 'Usage: bash scripts/run_all.sh GPU0 GPU1 GPU2'; exit 2; fi
gpus=("$1" "$2" "$3")
seeds=(3351 3352 3353)
pids=()
for i in 0 1 2; do
  gpu="${gpus[$i]}"; seed="${seeds[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$CPF_PY" run.py run-seed --seed "$seed" >>"logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"logs/seed${seed}.pid"
  printf 'START seed=%s physical_gpu=%s pid=%s\n' "$seed" "$gpu" "$!"
done
result=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else result=1; fi
done
"$CPF_PY" run.py report
if [[ "$result" -eq 0 ]]; then
  "$CPF_PY" scripts/check_complete.py
  printf '%s\n' 'QUEUE_COMPLETE'
else
  printf '%s\n' 'QUEUE_FAILED: inspect logs; rerun the same command to resume.'
fi
exit "$result"
