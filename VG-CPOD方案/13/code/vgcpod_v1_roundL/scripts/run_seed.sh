#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
SEED="${1:?Pass 3351, 3352 or 3353}"
"$CPF_PY" run.py train --cache "$CPF_CACHE" --work "$CPF_WORK" --seed "$SEED" --variant all
