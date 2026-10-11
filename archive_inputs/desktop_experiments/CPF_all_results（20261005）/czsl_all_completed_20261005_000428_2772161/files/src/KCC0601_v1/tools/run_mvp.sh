#!/usr/bin/env bash
set -euo pipefail
cfg=${1:?usage: run_mvp.sh CONFIG PHYSICAL_GPU}
gpu=${2:?usage: run_mvp.sh CONFIG PHYSICAL_GPU}
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1

info_text=$(python - "$cfg" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
assert c['TRAIN']['kcc_test_at_end'] is False
assert c['MODEL']['kcc_feature'] is False and c['MODEL']['kcc_logic'] is False
assert c['MODEL']['kcc_score_mode'] == 'legacy'
print(c['TRAIN']['kcc_run_dir'])
print(c['TRAIN']['checkpoint_dir']+'/'+c['config_name']+'_'+c['TRAIN']['train_name']+'/best_auc.pth')
print(c['DATASET']['root_dir'])
print(c['MODEL']['kcc_cache'])
PY
)
mapfile -t info <<< "$info_text"
[ "${#info[@]}" -eq 4 ] || { echo 'Invalid config'; exit 2; }
run=${info[0]}; ck=${info[1]}; data=${info[2]}; cache=${info[3]}
[ ! -e "$run" ] || { echo "Run exists, refusing overwrite: $run"; exit 2; }
test -s models/kcc_cpf.py
test -s "$cache/manifest.json"
python tools/build_knowledge.py --output-dir "$cache" --verify-only
mkdir -p "$run/meta"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "failed exit_code=%s\n" "$rc" > "$run/status.txt"; fi' EXIT
printf 'running\n' > "$run/status.txt"
cp "$cfg" "$run/config_requested.yml"
cp "$cache/cache_manifest.json" "$run/meta/cache_manifest.json"
python -m pip freeze > "$run/meta/pip_freeze.txt"
nvidia-smi > "$run/meta/nvidia-smi.txt"
python - "$run" "$data" <<'PY'
import hashlib, sys, tarfile
from pathlib import Path
run, data = map(Path, sys.argv[1:])
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
files=[]
for folder in ('models','config','utils','tools','tests','clip_modules'):
    if Path(folder).is_dir():
        files.extend(p for p in Path(folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py','.yml','.yaml','.sh'))
files.extend(Path(x) for x in ('train.py','test.py','dataset.py','evaluator_ge.py'))
files=sorted(set(files))
(run/'meta/source.sha256').write_text(''.join(digest(p)+'  '+str(p)+'\n' for p in files))
with tarfile.open(run/'meta/source_snapshot.tar.gz','w:gz') as archive:
    for p in files: archive.add(p,arcname=str(p))
datafiles=[data/'compositional-split-natural'/(s+'_pairs.txt') for s in ('train','val','test')]
datafiles.append(data/'metadata_compositional-split-natural.t7')
(run/'meta/dataset_hashes.txt').write_text(''.join(digest(p)+'  '+str(p)+'\n' for p in datafiles))
PY
printf 'CUDA_VISIBLE_DEVICES=%q python -u train.py --cfg %q\n' "$gpu" "$cfg" > "$run/command_train.txt"
date -Is > "$run/start_time.txt"
python -u train.py --cfg "$cfg" 2>&1 | tee "$run/train.log"
test -s "$ck"
sha256sum "$ck" > "$run/meta/checkpoint.sha256"
for setting in cw ow; do
    ow=False; [ "$setting" = ow ] && ow=True
    python tools/eval_mvp.py --cfg "$cfg" --load "$ck" --phase val --setting "$setting" \
        --out "$run/eval_val_$setting" EVAL.open_world "$ow" \
        2>&1 | tee "$run/eval_val_${setting}.log"
done
date -Is > "$run/end_time.txt"
printf 'train_and_val_complete\n' > "$run/status.txt"
echo "MVP TRAIN + VAL COMPLETE: $run"
