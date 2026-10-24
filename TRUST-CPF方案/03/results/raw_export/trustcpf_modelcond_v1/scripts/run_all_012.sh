#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs "$CPF_WORK"
exec 9>logs/queue.lock
flock -n 9 || { printf '%s\n' 'Queue already running'; exit 1; }
pids=()
for spec in '0 3351' '1 3352' '2 3353'; do
  read -r gpu seed <<< "$spec"
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
  printf '%s\n' 'QUEUE_FAILED: inspect seed logs; rerun run_all.sh to resume.'
fi
exit "$result"
