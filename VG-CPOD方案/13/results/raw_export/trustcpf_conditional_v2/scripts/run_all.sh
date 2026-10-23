#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs "$CPF_WORK"
# Hold a supervisor lock so repeated launches cannot write to the same log files.
exec 9>logs/queue.lock
flock -n 9 || { printf '%s\n' 'A queue is already running.'; exit 1; }
pids=()
for spec in '1 3351' '2 3352' '3 3353'; do
  read -r gpu seed <<< "$spec"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_seed.sh "$seed" >>"logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"logs/seed${seed}.pid"
  printf 'START seed=%s gpu=%s pid=%s\n' "$seed" "$gpu" "$!"
done
status=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else status=1; fi
done
"$CPF_PY" run.py report --work "$CPF_WORK"
if [[ "$status" -eq 0 ]]; then
  printf '%s\n' 'QUEUE_COMPLETE: all three seed queues finished.'
else
  printf '%s\n' 'QUEUE_FAILED: see seed logs; restart the same command to resume.'
fi
exit "$status"
