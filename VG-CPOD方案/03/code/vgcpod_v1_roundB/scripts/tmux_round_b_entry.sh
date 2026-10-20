#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
export VG_PYTHON="$1" VG_CACHE="$2" VG_B_OUT="$3" VG_A_OUT="$4"
shift 4
mkdir -p "$VG_B_OUT"
printf 'RUNNING\n' > "$VG_B_OUT/launcher.exitcode"
bash scripts/run_round_b.sh "$@" 2>&1 | tee -a "$VG_B_OUT/launcher.log"
RESULT=${PIPESTATUS[0]}
printf '%s\n' "$RESULT" > "$VG_B_OUT/launcher.exitcode"
printf '\nROUND_B_EXIT_CODE=%s\nResults: %s\n' "$RESULT" "$VG_B_OUT"
exec bash -i
