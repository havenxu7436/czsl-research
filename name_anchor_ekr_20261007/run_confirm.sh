#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$CPF_PY" "$BUNDLE/verify_release.py"
cd -- "$NAME_WORK/src"
mkdir -p -- "$NAME_WORK/launch_logs"
LOG="$NAME_WORK/launch_logs/confirm_$(date +%Y%m%d_%H%M%S).log"
"$CPF_PY" -u -m transition.name_plan run --work "$NAME_WORK" --seeds 3351 2026 \
  --variants N_name N_clean N_shuffle 2>&1 | tee "$LOG"
"$CPF_PY" -u -m transition.name_plan summary --work "$NAME_WORK" --stage confirm
