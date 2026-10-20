#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
PACKAGE_ROOT="$PWD"
SESSION="${VG_C_SESSION:-vgcpod_C}"
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_CACHE="${VG_CACHE:-/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14}"
VG_C_OUT="${VG_C_OUT:-/data/xhw/vgcpod_v1_runs/roundC_20261020}"
VG_A_OUT="${VG_A_OUT:-/data/xhw/vgcpod_v1_runs/roundA_20261019}"
VG_B_OUT="${VG_B_OUT:-/data/xhw/vgcpod_v1_runs/roundB_20261019}"
VG_B_ROOT="${VG_B_ROOT:-/data/xhw/vgcpod_project/vgcpod_v1_roundB}"
mkdir -p "$VG_C_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Session exists. Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PACKAGE_ROOT/scripts/tmux_round_c_entry.sh" \
  "$VG_PYTHON" "$VG_CACHE" "$VG_C_OUT" "$VG_A_OUT" "$VG_B_OUT" "$VG_B_ROOT" "$@"
tmux new-session -d -s "$SESSION" -c "$PACKAGE_ROOT" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
