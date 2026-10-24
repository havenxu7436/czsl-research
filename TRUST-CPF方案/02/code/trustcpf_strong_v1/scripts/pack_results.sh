#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
"$CPF_PY" run.py report --work "$CPF_WORK"
"$CPF_PY" scripts/check_complete.py
"$CPF_PY" scripts/pack_results.py
