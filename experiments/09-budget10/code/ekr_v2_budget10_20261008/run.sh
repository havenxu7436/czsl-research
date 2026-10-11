#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=/data/xhw/miniconda3/envs/cpf/bin/python
WORK=/var/tmp/czsl_xhw_name_ekr_20261007
OUT=${V2_OUT:-/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008}
PHASE=${1:-all}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$PY" "$BUNDLE/verify_release.py"
mkdir -p -- "${OUT}.launch_logs"
exec 9>"${OUT}.launch_logs/controller.lock"
flock -n 9 || { echo 'ANOTHER BUDGET10 CONTROLLER IS RUNNING' >&2; exit 2; }
LOG="${OUT}.launch_logs/${PHASE}_$(date +%Y%m%d_%H%M%S)_$$.log"
run_phase() {
  "$PY" -u "$BUNDLE/pilot.py" "$1" --work "$WORK" --out "$OUT"
}
if [[ "$PHASE" == all ]]; then
  {
    if [[ ! -e "$OUT" ]]; then run_phase prepare; fi
    run_phase import
    run_phase resume
    run_phase report
  } 2>&1 | tee "$LOG"
elif [[ "$PHASE" == prepare || "$PHASE" == import || "$PHASE" == resume || "$PHASE" == report || "$PHASE" == pack ]]; then
  run_phase "$PHASE" 2>&1 | tee "$LOG"
else
  echo 'USE all, prepare, import, resume, report, or pack' >&2
  exit 2
fi
