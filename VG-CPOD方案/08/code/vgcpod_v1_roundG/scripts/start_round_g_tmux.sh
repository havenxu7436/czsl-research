#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_G_OUT="$3"
  shift 3
  mkdir -p "$VG_G_OUT"
  printf 'RUNNING\n' > "$VG_G_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_g.py run --out "$VG_G_OUT" "$@" 2>&1 | tee -a "$VG_G_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_G_OUT/launcher.exitcode"
  printf '\nROUND_G_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_G_OUT="${VG_G_OUT:-/data/xhw/vgcpod_v1_runs/roundG_20261020}"
SESSION="${VG_G_SESSION:-vgcpod_G}"
mkdir -p "$VG_G_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_g_tmux.sh" --inside "$VG_PYTHON" "$VG_G_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
