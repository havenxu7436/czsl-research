#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
VG_PYTHON="${VG_PYTHON:-python}"
VG_CACHE="${VG_CACHE:-/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14}"
VG_C_OUT="${VG_C_OUT:-/data/xhw/vgcpod_v1_runs/roundC_20261020}"
VG_A_OUT="${VG_A_OUT:-/data/xhw/vgcpod_v1_runs/roundA_20261019}"
VG_B_OUT="${VG_B_OUT:-/data/xhw/vgcpod_v1_runs/roundB_20261019}"
VG_B_ROOT="${VG_B_ROOT:-/data/xhw/vgcpod_project/vgcpod_v1_roundB}"
exec "$VG_PYTHON" -u round_c.py run --cache "$VG_CACHE" --out "$VG_C_OUT" \
  --a-out "$VG_A_OUT" --b-out "$VG_B_OUT" --b-root "$VG_B_ROOT" "$@"
