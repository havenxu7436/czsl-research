#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
source scripts/env.sh
for MODEL in V2 V5; do
  for RANK in 16 64; do
    DEST="$VG_WORK/sensitivity/rank$RANK"
    EXTRA=()
    [[ ! -f "$DEST/ut-zap50k/$MODEL/seed3351/checkpoints/last.pt" ]] || EXTRA=(--resume)
    "$CPF_PY" -u run.py train --cache "$VG_CACHE" --out "$DEST" \
      --variant "$MODEL" --seed 3351 --rank "$RANK" "${EXTRA[@]}"
  done
done
for TEMP in 0.05 0.2; do
  DEST="$VG_WORK/sensitivity/tau_pi_$TEMP"
  EXTRA=()
  [[ ! -f "$DEST/ut-zap50k/V5/seed3351/checkpoints/last.pt" ]] || EXTRA=(--resume)
  "$CPF_PY" -u run.py train --cache "$VG_CACHE" --out "$DEST" \
    --variant V5 --seed 3351 --tau-pi "$TEMP" "${EXTRA[@]}"
done
