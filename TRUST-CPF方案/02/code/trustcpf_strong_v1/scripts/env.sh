#!/usr/bin/env bash
set -euo pipefail
CPF_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CPF_PROJECT/server_env.sh"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
