#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VG_GPU="$1"
shift
source scripts/env.sh

for SEED in 3351 3352 3353; do
    for MODEL in "$@"; do
        RUN="$VG_OUT/ut-zap50k/$MODEL/seed$SEED"
        EXTRA=()
        if [[ -f "$RUN/protocol.json" ]]; then
            EXTRA=(--resume)
        fi

        "$CPF_PY" -u run.py train \
            --cache "$VG_CACHE" \
            --out "$VG_OUT" \
            --variant "$MODEL" \
            --seed "$SEED" \
            --device cuda \
            "${EXTRA[@]}"
    done
done
