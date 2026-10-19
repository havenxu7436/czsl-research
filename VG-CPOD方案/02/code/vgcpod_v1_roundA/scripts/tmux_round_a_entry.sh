#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."
PACKAGE_ROOT="$PWD"
export VG_PYTHON="$1" VG_CACHE="$2" VG_A_OUT="$3"
shift 3
mkdir -p "$VG_A_OUT"
printf 'RUNNING\n' > "$VG_A_OUT/launcher.exitcode"
bash scripts/run_round_a.sh "$@" 2>&1 | tee -a "$VG_A_OUT/launcher.log"
RESULT=${PIPESTATUS[0]}
printf '%s\n' "$RESULT" > "$VG_A_OUT/launcher.exitcode"
printf '\nROUND_A_EXIT_CODE=%s\nResults: %s\n' "$RESULT" "$VG_A_OUT"
exec bash -i
