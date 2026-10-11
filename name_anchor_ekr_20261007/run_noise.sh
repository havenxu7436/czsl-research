#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$CPF_PY" "$BUNDLE/verify_release.py"
cd -- "$NAME_WORK/src"
mkdir -p -- "$NAME_WORK/launch_logs"
LOG="$NAME_WORK/launch_logs/noise_$(date +%Y%m%d_%H%M%S).log"
"$CPF_PY" -u -m transition.name_plan noise --work "$NAME_WORK" 2>&1 | tee "$LOG"
