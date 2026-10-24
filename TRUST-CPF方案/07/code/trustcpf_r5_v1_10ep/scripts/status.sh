#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
"$CPF_PY" run.py status
for seed in 3351 3352 3353; do
  printf '\nseed %s\n' "$seed"
  if [[ -f "logs/seed${seed}.log" ]]; then tail -n 4 "logs/seed${seed}.log"; fi
done
