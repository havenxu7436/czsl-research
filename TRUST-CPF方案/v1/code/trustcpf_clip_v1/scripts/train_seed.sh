#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
SEED="${1:-3351}"
"$CPF_PY" run.py pipeline --cache "$CACHE" --work "$WORK_ROOT/seed$SEED" --seed "$SEED"
"$CPF_PY" run.py report --cache "$CACHE" --work "$WORK_ROOT/seed$SEED"
