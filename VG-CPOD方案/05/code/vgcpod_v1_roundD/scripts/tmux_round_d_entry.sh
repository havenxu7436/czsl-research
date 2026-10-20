#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VG_PYTHON="$1" VG_CACHE="$2" VG_D_OUT="$3" VG_C_OUT="$4" VG_C_ROOT="$5" VG_A_OUT="$6" VG_B_OUT="$7"
shift 7
mkdir -p "$VG_D_OUT"
printf 'RUNNING\n' > "$VG_D_OUT/launcher.exitcode"
bash scripts/run_round_d.sh "$@" 2>&1 | tee -a "$VG_D_OUT/launcher.log"
RESULT=${PIPESTATUS[0]}
printf '%s\n' "$RESULT" > "$VG_D_OUT/launcher.exitcode"
printf '\nROUND_D_EXIT_CODE=%s\nResults: %s\n' "$RESULT" "$VG_D_OUT"
exec bash -i
