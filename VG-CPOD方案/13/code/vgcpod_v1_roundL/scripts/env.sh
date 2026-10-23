#!/usr/bin/env bash
CPF_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$CPF_PROJECT/server_env.sh" ]]; then
  source "$CPF_PROJECT/server_env.sh"
fi
export CPF_PY="${CPF_PY:-python}"
export CPF_WORK="${CPF_WORK:-$CPF_PROJECT/work}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
