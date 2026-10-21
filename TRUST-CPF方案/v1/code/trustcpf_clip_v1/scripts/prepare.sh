#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
"$CPF_PY" run.py preflight --cache "$CACHE" --data-root "$DATA_ROOT" --checkpoint "$CLIP_CKPT"
"$CPF_PY" run.py features --cache "$CACHE" --data-root "$DATA_ROOT" --checkpoint "$CLIP_CKPT" --batch-size 16 --workers 4
"$CPF_PY" run.py graph --cache "$CACHE" --checkpoint "$CLIP_CKPT"
