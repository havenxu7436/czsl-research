#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
BUNDLE_DIR=/tmp/ekr_train_support_probe_20261008
PROBE_OUT=${PROBE_OUT:-/var/tmp/czsl_xhw_ekr_train_support_20261008}
"$CPF_PY" "$BUNDLE_DIR/tests.py"
"$CPF_PY" "$BUNDLE_DIR/probe.py" --out "$PROBE_OUT"
