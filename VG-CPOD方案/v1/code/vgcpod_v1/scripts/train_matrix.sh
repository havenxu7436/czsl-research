#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
source scripts/env.sh
# pilot: six groups, seed3351; full: nine groups, seed3351/3352/3353.
MODE="${1:-pilot}"
if [[ "$MODE" == pilot ]]; then
  SEEDS=(3351)
  MODELS=(V0 V1 V1L V2 V4 V5)
elif [[ "$MODE" == full ]]; then
  SEEDS=(3351 3352 3353)
  MODELS=(V0 V1 V1L V2 V3 V4 V5 V6 V7)
else
  echo "Usage: bash scripts/train_matrix.sh [pilot|full]" >&2
  exit 2
fi
mkdir -p "$VG_WORK/logs"
for SEED in "${SEEDS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    RUN="$VG_OUT/ut-zap50k/$MODEL/seed$SEED"
    EXTRA=()
    if [[ -f "$RUN/checkpoints/last.pt" ]]; then
      EXTRA=(--resume)
    elif [[ -f "$RUN/protocol.json" ]]; then
      EXTRA=(--resume)
    fi
    "$CPF_PY" -u run.py train --cache "$VG_CACHE" --out "$VG_OUT" \
      --variant "$MODEL" --seed "$SEED" --device cuda "${EXTRA[@]}" \
      2>&1 | tee -a "$VG_WORK/logs/${MODEL}_seed${SEED}.log"
  done
done
"$CPF_PY" run.py summarize --out "$VG_OUT"
"$CPF_PY" run.py compare --cache "$VG_CACHE" --out "$VG_OUT" --seeds "${SEEDS[@]}"
