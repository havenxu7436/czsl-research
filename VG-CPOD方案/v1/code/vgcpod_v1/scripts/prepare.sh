#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
source scripts/env.sh
mkdir -p "$VG_WORK/logs"
"$CPF_PY" run.py doctor --data-root "$VG_DATA" --checkpoint "$VG_CLIP"
"$CPF_PY" -m unittest discover -s tests -v
"$CPF_PY" -u run.py prepare --data-root "$VG_DATA" --checkpoint "$VG_CLIP" \
  --cache "$VG_CACHE" --batch-size 16 --workers 4 --device cuda \
  2>&1 | tee "$VG_WORK/logs/prepare.log"
"$CPF_PY" run.py smoke --cache "$VG_CACHE" --out "$VG_WORK/preflight" --device cuda
"$CPF_PY" run.py baselines --cache "$VG_CACHE" --out "$VG_OUT" --device cuda
