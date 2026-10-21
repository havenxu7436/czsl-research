#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_I_OUT="$3"
  shift 3
  mkdir -p "$VG_I_OUT"
  printf 'RUNNING\n' > "$VG_I_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_i.py run --out "$VG_I_OUT" "$@" 2>&1 | tee -a "$VG_I_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_I_OUT/launcher.exitcode"
  printf '\nROUND_I_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_I_OUT="${VG_I_OUT:-/data/xhw/vgcpod_v1_runs/roundI_20261021}"
SESSION="${VG_I_SESSION:-vgcpod_I}"
mkdir -p "$VG_I_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_i_tmux.sh" --inside "$VG_PYTHON" "$VG_I_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
