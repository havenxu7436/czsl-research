#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
PACKAGE_ROOT="$PWD"
SESSION="${VG_B_SESSION:-vgcpod_B}"
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_CACHE="${VG_CACHE:-/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14}"
VG_B_OUT="${VG_B_OUT:-/data/xhw/vgcpod_v1_runs/roundB_20261019}"
VG_A_OUT="${VG_A_OUT:-/data/xhw/vgcpod_v1_runs/roundA_20261019}"
mkdir -p "$VG_B_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Session exists. Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PACKAGE_ROOT/scripts/tmux_round_b_entry.sh" "$VG_PYTHON" "$VG_CACHE" "$VG_B_OUT" "$VG_A_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PACKAGE_ROOT" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
