#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
VG_PYTHON="${VG_PYTHON:-python}"
VG_CACHE="${VG_CACHE:-/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14}"
VG_D_OUT="${VG_D_OUT:-/data/xhw/vgcpod_v1_runs/roundD_20261020}"
VG_C_OUT="${VG_C_OUT:-/data/xhw/vgcpod_v1_runs/roundC_20261020}"
VG_C_ROOT="${VG_C_ROOT:-/data/xhw/vgcpod_project/vgcpod_v1_roundC}"
VG_A_OUT="${VG_A_OUT:-/data/xhw/vgcpod_v1_runs/roundA_20261019}"
VG_B_OUT="${VG_B_OUT:-/data/xhw/vgcpod_v1_runs/roundB_20261019}"
exec "$VG_PYTHON" -u round_d.py run --cache "$VG_CACHE" --out "$VG_D_OUT" \
  --c-out "$VG_C_OUT" --c-root "$VG_C_ROOT" --a-out "$VG_A_OUT" --b-out "$VG_B_OUT" "$@"
