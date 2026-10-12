#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CPF_PY=${CPF_PY:-/data/xhw/miniconda3/envs/cpf/bin/python}
UT_V2_ROOT=${UT_V2_ROOT:-/data/xhw/czsl_repro}
UT_V2_WORK=${UT_V2_WORK:-/var/tmp/czsl_xhw_ut_v2_20261006}
"$CPF_PY" "$BUNDLE/verify_release.py"
"$CPF_PY" -c 'import sys,torch,torchvision; from torch.nn.modules.pooling import MaxPool2d; assert sys.version_info[:2]==(3,8); assert torch.__version__.startswith("1.13.1"); assert torchvision.__version__.startswith("0.14.1"); print("CPU ENVIRONMENT PASSED",sys.executable)'
mkdir -p -- "$UT_V2_WORK"
if [[ ! -f "$UT_V2_WORK/src/transition_install.json" ]]; then
  "$CPF_PY" "$BUNDLE/install.py" --source "$UT_V2_ROOT/src/KCC0601_v1" --destination "$UT_V2_WORK/src"
fi
cd -- "$UT_V2_WORK/src"
"$CPF_PY" -u -m transition.v2_prepare --root "$UT_V2_ROOT" --work "$UT_V2_WORK"
printf 'PREPARE COMPLETE. Run: bash %s/run_paired.sh\n' "$BUNDLE"
