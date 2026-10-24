#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
"$CPF_PY" run.py run-seed --cache "$CPF_CACHE" --work "$CPF_WORK" --seed "${1:?seed required}"
