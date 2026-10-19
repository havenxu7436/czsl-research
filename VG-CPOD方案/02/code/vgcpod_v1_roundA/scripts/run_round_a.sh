#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
PACKAGE_ROOT="$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
VG_PYTHON="${VG_PYTHON:-python}"
VG_CACHE="${VG_CACHE:-/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14}"
VG_A_OUT="${VG_A_OUT:-/data/xhw/vgcpod_v1_runs/roundA_20261019}"
exec "$VG_PYTHON" -u round_a.py run --cache "$VG_CACHE" --out "$VG_A_OUT" "$@"
