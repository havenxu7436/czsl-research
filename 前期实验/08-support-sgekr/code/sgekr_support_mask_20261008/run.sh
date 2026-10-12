#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=/data/xhw/miniconda3/envs/cpf/bin/python
WORK=/var/tmp/czsl_xhw_name_ekr_20261007
OUT=${SGEKR_OUT:-/var/tmp/czsl_xhw_sgekr_20261008}
PHASE=${1:-all}
if [[ "$OUT" != /var/tmp/czsl_xhw_sgekr_* ]]; then
  echo 'USE A SEPARATE /var/tmp/czsl_xhw_sgekr_ OUTPUT VERSION' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$PY" "$BUNDLE/verify_release.py"
mkdir -p -- "${OUT}.launch_logs"
LOG="${OUT}.launch_logs/${PHASE}_$(date +%Y%m%d_%H%M%S)_$$.log"
run_phase() {
  "$PY" -u "$BUNDLE/pilot.py" "$1" --work "$WORK" --out "$OUT"
}
if [[ "$PHASE" == all ]]; then
  { run_phase prepare; run_phase train; run_phase report; } 2>&1 | tee "$LOG"
elif [[ "$PHASE" == prepare || "$PHASE" == train || "$PHASE" == resume || "$PHASE" == report || "$PHASE" == pack ]]; then
  run_phase "$PHASE" 2>&1 | tee "$LOG"
else
  echo 'UNKNOWN PHASE; USE all, prepare, train, resume, report, or pack' >&2
  exit 2
fi
