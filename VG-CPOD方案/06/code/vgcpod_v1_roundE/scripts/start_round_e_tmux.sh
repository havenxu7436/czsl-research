#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_E_OUT="$3"
  shift 3
  mkdir -p "$VG_E_OUT"
  printf 'RUNNING\n' > "$VG_E_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_e.py run --out "$VG_E_OUT" "$@" 2>&1 | tee -a "$VG_E_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_E_OUT/launcher.exitcode"
  printf '\nROUND_E_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_E_OUT="${VG_E_OUT:-/data/xhw/vgcpod_v1_runs/roundE_20261020}"
SESSION="${VG_E_SESSION:-vgcpod_E}"
mkdir -p "$VG_E_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_e_tmux.sh" --inside "$VG_PYTHON" "$VG_E_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
