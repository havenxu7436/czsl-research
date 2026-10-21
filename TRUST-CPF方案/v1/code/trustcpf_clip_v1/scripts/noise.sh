#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
SEED="${1:-3351}"
for VARIANT in dense drop mean robust; do
  "$CPF_PY" run.py noise --cache "$CACHE" --work "$WORK_ROOT/seed$SEED" --seed "$SEED" --variant "$VARIANT"
  "$CPF_PY" run.py noise --cache "$CACHE" --work "$WORK_ROOT/seed$SEED" --seed "$SEED" --variant "$VARIANT" --deletion
done
