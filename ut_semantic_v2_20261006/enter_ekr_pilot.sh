#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CPF_PY=${CPF_PY:-/data/xhw/miniconda3/envs/cpf/bin/python}
UT_V2_ROOT=${UT_V2_ROOT:-/data/xhw/czsl_repro}
UT_V2_WORK=${UT_V2_WORK:-/var/tmp/czsl_xhw_ut_v2_20261006}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$CPF_PY" "$BUNDLE/verify_release.py"
cd -- "$UT_V2_WORK/src"
SEMANTIC_EXP="$UT_V2_WORK/experiments/semantic_v2"
GATE="$SEMANTIC_EXP/protocol/semantic_gate_v2.json"
"$CPF_PY" -u -m transition.v2_checks post-audit --root "$UT_V2_ROOT" --work "$UT_V2_WORK" --out "$SEMANTIC_EXP/protocol/integrity_v2.json"
"$CPF_PY" - "$GATE" "$SEMANTIC_EXP/reports/paired_v2/collection.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2]))
assert g['report_fingerprint']==r['fingerprint'], 'Gate belongs to another report'
assert g['datasets']['ut-zappos']['proceed'], 'STOP: v2 semantic research gate did not pass'
PY
EXP="$UT_V2_WORK/experiments/ekr_v2_pilot"
"$CPF_PY" -u -m transition.cli make-configs --root "$UT_V2_ROOT" --exp "$EXP" \
  --legacy-exp "$UT_V2_ROOT/experiments/kcc0601_glove_mvp_v1" \
  --semantic-exp "$SEMANTIC_EXP" --cache-root "$UT_V2_WORK/cache" --bank-root "$UT_V2_WORK/bank" \
  --stage branch --datasets ut-zappos --seeds 0 --epochs 3 --variants R0 R3 R5 R7 --regime frozen --gate "$GATE"
SMOKE="$UT_V2_WORK/smoke/ekr_v2_seed0.json"
if [[ ! -f "$SMOKE" ]]; then
  "$CPF_PY" -u -m transition.cli smoke --cfg "$EXP/configs/branch_ut-zappos_R5_s0_e3.yml" --out "$SMOKE"
else
  "$CPF_PY" - "$SMOKE" "$UT_V2_WORK/bank/ut-zap50k/manifest.json" "$SEMANTIC_EXP/runs/semantic_ut-zappos_C_text_s0_e10/checkpoints/best_auc.pth" <<'PY'
import json,sys,hashlib
s=json.load(open(sys.argv[1]))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert s['passed'] and s['checkpoint_sha256']==sha(sys.argv[3])
assert s['bank_manifest_sha256']==sha(sys.argv[2])
PY
fi
mapfile -t CONFIGS < <("$CPF_PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("\n".join(p.values()))' "$EXP/config_index_branch_e3.json")
[[ ${#CONFIGS[@]} -eq 4 ]]
LOG="$UT_V2_WORK/launch_logs/ekr_pilot_$(date +%Y%m%d_%H%M%S).log"
"$CPF_PY" -u -m transition.cli run-plan --configs "${CONFIGS[@]}" 2>&1 | tee "$LOG"
"$CPF_PY" -u -m transition.cli summarize --exp "$EXP" --out "$EXP/reports/seed0_pilot"
printf 'EKR SEED0 PILOT COMPLETE. Inspect %s/reports/seed0_pilot/paired_deltas.csv before adding seeds.\n' "$EXP"
