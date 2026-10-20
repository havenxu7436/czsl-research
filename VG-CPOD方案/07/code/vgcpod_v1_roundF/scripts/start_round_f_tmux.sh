#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_F_OUT="$3"
  shift 3
  mkdir -p "$VG_F_OUT"
  printf 'RUNNING\n' > "$VG_F_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_f.py run --out "$VG_F_OUT" "$@" 2>&1 | tee -a "$VG_F_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_F_OUT/launcher.exitcode"
  printf '\nROUND_F_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_F_OUT="${VG_F_OUT:-/data/xhw/vgcpod_v1_runs/roundF_20261020}"
SESSION="${VG_F_SESSION:-vgcpod_F}"
mkdir -p "$VG_F_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_f_tmux.sh" --inside "$VG_PYTHON" "$VG_F_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
