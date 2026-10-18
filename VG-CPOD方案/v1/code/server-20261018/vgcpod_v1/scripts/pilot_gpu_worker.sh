#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VG_GPU="$1"
shift
source scripts/env.sh

for MODEL in "$@"; do
    EXTRA=()
    if [[ -f "$VG_OUT/ut-zap50k/$MODEL/seed3351/protocol.json" ]]; then
        EXTRA=(--resume)
    fi

    "$CPF_PY" -u run.py train \
        --cache "$VG_CACHE" --out "$VG_OUT" \
        --variant "$MODEL" --seed 3351 \
        --device cuda "${EXTRA[@]}"
done
