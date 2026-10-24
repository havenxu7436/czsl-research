#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs "$CPF_WORK"
exec 9>logs/queue.lock
flock -n 9 || { printf '%s\n' 'Queue already running'; exit 1; }
pids=()
for spec in '1 3351' '2 3352' '3 3353'; do
  read -r gpu seed <<< "$spec"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_seed.sh "$seed" >>"logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"logs/seed${seed}.pid"
  printf 'START seed=%s physical_gpu=%s pid=%s\n' "$seed" "$gpu" "$!"
done
status=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else status=1; fi
done
"$CPF_PY" run.py report --work "$CPF_WORK"
if [[ "$status" -eq 0 ]]; then
  "$CPF_PY" scripts/check_complete.py
  printf '%s\n' 'QUEUE_COMPLETE'
else
  printf '%s\n' 'QUEUE_FAILED: inspect seed logs, then rerun this queue to resume.'
fi
exit "$status"
