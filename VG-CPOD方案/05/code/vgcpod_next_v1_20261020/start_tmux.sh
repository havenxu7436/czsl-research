#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_NEXT_OUT="$3"
  shift 3
  mkdir -p "$VG_NEXT_OUT"
  printf 'RUNNING\n' > "$VG_NEXT_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u next_round.py --out "$VG_NEXT_OUT" "$@" 2>&1 | tee -a "$VG_NEXT_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_NEXT_OUT/launcher.exitcode"
  printf '\nNEXT_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_NEXT_OUT="${VG_NEXT_OUT:-/data/xhw/vgcpod_v1_runs/next_20261020}"
SESSION="${VG_NEXT_SESSION:-vgcpod_next}"
mkdir -p "$VG_NEXT_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/start_tmux.sh" --inside "$VG_PYTHON" "$VG_NEXT_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
