#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VG_GPU="$1"
MODEL="$2"
TYPE="$3"
source scripts/env.sh

case "$TYPE" in
    rank)
        VALUES=(16 64)
        OPTION="--rank"
        PREFIX="rank"
        ;;
    tau)
        VALUES=(0.05 0.2)
        OPTION="--tau-pi"
        PREFIX="tau_pi_"
        ;;
    *)
        echo "Unknown sensitivity type: $TYPE" >&2
        exit 2
        ;;
esac

for VALUE in "${VALUES[@]}"; do
    DEST="$VG_WORK/sensitivity/${PREFIX}${VALUE}"
    EXTRA=()
    if [[ -f "$DEST/ut-zap50k/$MODEL/seed3351/protocol.json" ]]; then
        EXTRA=(--resume)
    fi

    "$CPF_PY" -u run.py train \
        --cache "$VG_CACHE" \
        --out "$DEST" \
        --variant "$MODEL" \
        --seed 3351 \
        --device cuda \
        "$OPTION" "$VALUE" \
        "${EXTRA[@]}"
done
