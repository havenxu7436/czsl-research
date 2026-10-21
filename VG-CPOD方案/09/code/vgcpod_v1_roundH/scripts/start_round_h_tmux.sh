#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_H_OUT="$3"
  shift 3
  mkdir -p "$VG_H_OUT"
  printf 'RUNNING\n' > "$VG_H_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_h.py run --out "$VG_H_OUT" "$@" 2>&1 | tee -a "$VG_H_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_H_OUT/launcher.exitcode"
  printf '\nROUND_H_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_H_OUT="${VG_H_OUT:-/data/xhw/vgcpod_v1_runs/roundH_20261020}"
SESSION="${VG_H_SESSION:-vgcpod_H}"
mkdir -p "$VG_H_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_h_tmux.sh" --inside "$VG_PYTHON" "$VG_H_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
