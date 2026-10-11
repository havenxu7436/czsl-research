#!/usr/bin/env bash
# Absolute environment Python works even if the shell still displays (base).
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CPF_PY=${CPF_PY:-/data/xhw/miniconda3/envs/cpf/bin/python}
export NAME_PREVIOUS=${NAME_PREVIOUS:-/var/tmp/czsl_xhw_ut_v2_20261006}
export NAME_WORK=${NAME_WORK:-/var/tmp/czsl_xhw_name_ekr_20261007}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
