#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CPF_PY=${CPF_PY:-/data/xhw/miniconda3/envs/cpf/bin/python}
DIAG_OLD_WORK=${DIAG_OLD_WORK:-/var/tmp/czsl_xhw_name_ekr_20261007}
DIAG_OUT=${DIAG_OUT:-/var/tmp/czsl_xhw_name_ekr_diag_20261007_v1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
if [[ -e "$DIAG_OUT" || -e "${DIAG_OUT}.launch.log" ]]; then
  printf 'OUTPUT EXISTS; PRESERVE IT AND USE A NEW DIAG_OUT VERSION: %s\n' "$DIAG_OUT" >&2
  exit 2
fi
"$CPF_PY" "$BUNDLE/verify_release.py"
cd -- "$DIAG_OLD_WORK/src"
"$CPF_PY" -u "$BUNDLE/diagnose.py" run --work "$DIAG_OLD_WORK" --out "$DIAG_OUT" \
  2>&1 | tee "${DIAG_OUT}.launch.log"
