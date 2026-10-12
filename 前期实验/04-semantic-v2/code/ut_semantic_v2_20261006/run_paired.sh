#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CPF_PY=${CPF_PY:-/data/xhw/miniconda3/envs/cpf/bin/python}
UT_V2_ROOT=${UT_V2_ROOT:-/data/xhw/czsl_repro}
UT_V2_WORK=${UT_V2_WORK:-/var/tmp/czsl_xhw_ut_v2_20261006}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$CPF_PY" "$BUNDLE/verify_release.py"
cd -- "$UT_V2_WORK/src"
"$CPF_PY" -c 'import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; print("OUTPUT FREE GiB",round(free/2**30,2)); assert free>=20*2**30,"Need at least 20 GiB free on the output filesystem"' "$UT_V2_WORK"
mkdir -p -- "$UT_V2_WORK/launch_logs"
LOG="$UT_V2_WORK/launch_logs/paired_$(date +%Y%m%d_%H%M%S).log"
(
  "$CPF_PY" -u -m transition.v2_checks gpu-smoke --root "$UT_V2_ROOT" --work "$UT_V2_WORK"
  mapfile -t CONFIGS < <("$CPF_PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("\n".join(p.values()))' "$UT_V2_WORK/experiments/semantic_v2/config_index_semantic_e10.json")
  [[ ${#CONFIGS[@]} -eq 12 ]]
  "$CPF_PY" -u -m transition.cli run-plan --configs "${CONFIGS[@]}"
) 2>&1 | tee "$LOG"
bash "$BUNDLE/summarize.sh"
