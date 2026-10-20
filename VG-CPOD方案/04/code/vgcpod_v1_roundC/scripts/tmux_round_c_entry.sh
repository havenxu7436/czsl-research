#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
export VG_PYTHON="$1" VG_CACHE="$2" VG_C_OUT="$3" VG_A_OUT="$4" VG_B_OUT="$5" VG_B_ROOT="$6"
shift 6
mkdir -p "$VG_C_OUT"
printf 'RUNNING\n' > "$VG_C_OUT/launcher.exitcode"
bash scripts/run_round_c.sh "$@" 2>&1 | tee -a "$VG_C_OUT/launcher.log"
RESULT=${PIPESTATUS[0]}
printf '%s\n' "$RESULT" > "$VG_C_OUT/launcher.exitcode"
printf '\nROUND_C_EXIT_CODE=%s\nResults: %s\n' "$RESULT" "$VG_C_OUT"
exec bash -i
