#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${1:-}" = "--inside" ]; then
  VG_PYTHON="$2"
  VG_K_OUT="$3"
  shift 3
  mkdir -p "$VG_K_OUT"
  printf 'RUNNING\n' > "$VG_K_OUT/launcher.exitcode"
  set +e
  "$VG_PYTHON" -u round_k.py run --out "$VG_K_OUT" "$@" 2>&1 | tee -a "$VG_K_OUT/launcher.log"
  RESULT=${PIPESTATUS[0]}
  printf '%s\n' "$RESULT" > "$VG_K_OUT/launcher.exitcode"
  printf '\nROUND_K_EXIT_CODE=%s\n' "$RESULT"
  exec bash -i
fi
VG_PYTHON="$(command -v "${VG_PYTHON:-python}")"
VG_K_OUT="${VG_K_OUT:-/data/xhw/vgcpod_v1_runs/roundK_20261021}"
SESSION="${VG_K_SESSION:-vgcpod_K}"
mkdir -p "$VG_K_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Attach with: tmux attach -t %q\n' "$SESSION"
  exit 0
fi
printf -v LAUNCH '%q ' bash "$PWD/scripts/start_round_k_tmux.sh" --inside "$VG_PYTHON" "$VG_K_OUT" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$LAUNCH"
printf 'Started. Attach with: tmux attach -t %q\n' "$SESSION"
