#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_J_OUT="$3"
  shift 3
  mkdir -p "$VG_J_OUT"
  printf 'RUNNING\n' > "$VG_J_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_j.py run --out "$VG_J_OUT" "$@" 2>&1 | tee -a "$VG_J_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_J_OUT/launcher.exitcode"
  printf '\nROUND_J_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_J_OUT="${VG_J_OUT:-/data/xhw/vgcpod_v1_runs/roundJ_20261021}"
SESSION="${VG_J_SESSION:-vgcpod_J}"
mkdir -p "$VG_J_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_j_tmux.sh" --inside "$VG_PYTHON" "$VG_J_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
