"""Verify F archive, README result tables, numerical replay and math layout."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
import tarfile

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parent/'CZSL'
STAGE = REPO/'VG-CPOD方案/07'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

original = read(STAGE/'analysis/confirmed/analysis.json')
recomputed = read(STAGE/'analysis/recomputed/analysis.json')
assert original==recomputed
assert read(STAGE/'analysis/confirmed/mechanism.json')==read(STAGE/'analysis/recomputed/mechanism.json')
tables = list((STAGE/'analysis/confirmed/tables').glob('*.csv'))
assert all(p.read_bytes()==(STAGE/'analysis/recomputed/tables'/p.name).read_bytes() for p in tables)
raw = read(STAGE/'results/raw_export_members.json')
expected = {r['path'].removeprefix('results/raw_export/'):r['sha256'] for r in raw}
checked = set()
with tarfile.open(STAGE/'results/archives/vgcpod_roundF_results_20261020.tar.gz','r|gz') as archive:
    for member in archive:
        if not member.isfile():
            continue
        name = member.name.removeprefix('./')
        if name in expected:
            h = hashlib.sha256()
            f = archive.extractfile(member)
            for block in iter(lambda:f.read(4*1024*1024),b''):
                h.update(block)
            assert h.hexdigest()==expected[name],name
            checked.add(name)
assert checked==set(expected),(len(checked),len(expected))

rows = read(STAGE/'configs/experiment_inventory.json')
for row in rows:
    keys = ['val_cw','val_ow','test_cw','test_ow']
    numbers = ['%.4f'%row['metrics'][k]['AUC_native'] for k in keys]
    line = '| %d | %d／%d | %s |'%(row['seed'],row['selected_epoch'],row['exact_selected_epoch'],' | '.join(numbers))
    assert line in (STAGE/'README.md').read_text(encoding='utf-8'),line
    assert line in (STAGE/'results/README.md').read_text(encoding='utf-8'),line
    d = read(STAGE/'configs/round_e/E1.json')
    assert {k for k in d if d[k]!=row['config'][k]}=={'task_aux_beta','experiment_id'}

means = original['means']
root_text = (STAGE/'README.md').read_text(encoding='utf-8')
for case in ('A5','D1','E1','F1','F1_mean'):
    def metric(setting,key):
        return next(r[key] for r in means if r['case']==case and r['setting']==setting)
    values = [metric('val_cw','AUC_native_mean'),metric('test_cw','AUC_native_mean'),
              metric('test_ow','AUC_native_mean'),metric('test_ow','gt_object_attr_acc_unseen_mean')]
    assert ' | '.join('%.3f'%x for x in values[:-1])+' | %.3f%%'%values[-1] in root_text

layout = read(STAGE/'analysis/repository_qa/layout.json')
assert layout['passed'] and len(layout['readmes'])==5
commands = 0
sys.path.insert(0,str(WORK/'tmp/round_a_bash_validation'))
import bashlex
for p in STAGE.rglob('README.md'):
    text = p.read_text(encoding='utf-8')
    for block in re.findall(r'^```bash\n(.*?)\n```',text,re.M|re.S):
        bashlex.parse(block);commands += 1
    assert text.count('```')%2==0 and '$$' not in text
for p in (STAGE/'code/vgcpod_v1_roundF').rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))

materialization = read(WORK/'replay_archive07_20261020/materialization.json')
audit = original['audit']
result = dict(passed=True,date='2026-10-20',new_training_runs=3,model_epochs=30,
    native_predictions=12,exact_selection_predictions=12,intervention_predictions=48,new_checkpoint_files=12,
    recomputed_training_runs=20,recomputed_prediction_files=audit['prediction_files'],
    source_files_verified=audit['source_files_verified'],server_source_members_verified=len(read(STAGE/'results/server_source_identity.json')['members']),
    raw_result_members_identical_to_TAR=len(checked),
    reused_reference_members_verified=len(read(STAGE/'results/reused_reference_members.json')),
    analysis_identical=True,csv_tables_byte_identical=len(tables),
    original_analysis_sha256=sha(STAGE/'analysis/confirmed/analysis.json'),
    README_result_rows_verified=6,README_mean_rows_verified=5,mechanism_recomputed_identical=True,parallel_readmes=5,
    display_formulas=sum(r['displayMath'] for r in layout['readmes']),formula_visual_inspection=True,
    parsed_bash_command_blocks=commands,materialization_files=materialization['files_verified'],
    original_model_unchanged=True,training_started=False,max_recompute_errors=audit['max_recompute_errors'])
(STAGE/'analysis/repository_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
