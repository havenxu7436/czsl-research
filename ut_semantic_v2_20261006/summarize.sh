#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CPF_PY=${CPF_PY:-/data/xhw/miniconda3/envs/cpf/bin/python}
UT_V2_ROOT=${UT_V2_ROOT:-/data/xhw/czsl_repro}
UT_V2_WORK=${UT_V2_WORK:-/var/tmp/czsl_xhw_ut_v2_20261006}
"$CPF_PY" "$BUNDLE/verify_release.py"
cd -- "$UT_V2_WORK/src"
EXP="$UT_V2_WORK/experiments/semantic_v2"
REPORT="$EXP/reports/paired_v2"
GATE="$EXP/protocol/semantic_gate_v2.json"
"$CPF_PY" -u -m transition.v2_checks post-audit --root "$UT_V2_ROOT" --work "$UT_V2_WORK" --out "$EXP/protocol/integrity_v2.json"
"$CPF_PY" -u -m transition.cli summarize --exp "$EXP" --out "$REPORT"
if [[ ! -f "$GATE" ]]; then
  "$CPF_PY" -u -m transition.cli gate --report "$REPORT" --out "$GATE" --datasets ut-zappos --seeds 0 3351 2026
else
  "$CPF_PY" -c 'import json,sys; g=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); assert g["report_fingerprint"]==r["fingerprint"]; print(json.dumps(g,indent=2))' "$GATE" "$REPORT/collection.json"
fi
"$CPF_PY" -c 'import json,sys; g=json.load(open(sys.argv[1])); print("V2 RESEARCH GATE PROCEED",g["datasets"]["ut-zappos"]["proceed"])' "$GATE"
"$CPF_PY" - "$UT_V2_WORK" <<'PY'
import sys, zipfile
from pathlib import Path
work = Path(sys.argv[1])
out = work / 'ut_v2_results_review.zip'
paths = [work/'prepared.json',work/'gpu_smoke.json']
for folder in ('experiments/semantic_v2/reports','experiments/semantic_v2/protocol',
               'src/v2_assets/knowledge','cache/ut-zap50k','bank/ut-zap50k'):
    paths += [p for p in (work/folder).rglob('*') if p.is_file() and p.suffix != '.pt']
for run in (work/'experiments/semantic_v2/runs').iterdir():
    paths += [run/name for name in ('effective_config.yml','train_complete.json','status.json',
              'eval_val_cw/metrics.json','eval_val_ow/metrics.json','meta/code_manifest.json',
              'meta/environment.json')]
with zipfile.ZipFile(out.with_suffix('.building.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(set(paths)):
        z.write(path,path.relative_to(work).as_posix())
out.with_suffix('.building.zip').replace(out)
print('REVIEW RESULT ZIP',out)
print('KEEP SERVER CHECKPOINTS FOR POSSIBLE EKR; THIS SMALL ZIP DOES NOT INCLUDE WEIGHTS')
PY
