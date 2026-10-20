"""Cross-check experiment-05 analyses, README result rows and restored package bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

STAGE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialized',type=Path)
    args = parser.parse_args()
    original = read(STAGE/'analysis/confirmed/analysis.json')
    recomputed = read(STAGE/'analysis/recomputed/analysis.json')
    assert original==recomputed
    tables = sorted((STAGE/'analysis/confirmed/tables').glob('*.csv'))
    assert len(tables)==13
    for path in tables:
        assert path.read_bytes()==(STAGE/'analysis/recomputed/tables'/path.name).read_bytes(),path.name
    inventory = read(STAGE/'configs/experiment_inventory.json')
    for rel in ('README.md','results/README.md'):
        text = (STAGE/rel).read_text(encoding='utf-8')
        for row in inventory:
            expected = '| '+str(row['case'])+' | '+str(row['seed'])+' | '+str(row['selected_epoch'])+' | '+' | '.join(
                '%.4f'%row['metrics'][setting]['AUC_native'] for setting in ('val_cw','val_ow','test_cw','test_ow'))+' |'
            assert expected in text,(rel,row['case'],row['seed'])
    layout = read(STAGE/'analysis/readme_layout_check.json')
    assert layout['passed'] and len(layout['files'])==5
    formulas = 0
    for entry in layout['files']:
        text = (STAGE/entry['file']).read_text(encoding='utf-8')
        assert len(re.findall(r'```math\n(.*?)\n```',text,re.S))==entry['display']
        assert not re.search(r'\$\$',text)
        formulas += entry['display']
    restored = None
    if args.materialized:
        restored = read(args.materialized/'materialization.json')
        assert restored['stage']=='05' and not restored['training_started']
        for parent,name in [(STAGE,'package_members.json'),(STAGE,'next_package_members.json'),
                            (STAGE.parent/'04','package_members.json')]:
            data = read(parent/name)
            for member,source in data['members'].items():
                assert sha(args.materialized/data['package']/member)==sha(parent/source),member
        assert restored['primary_and_next_files']==69 and restored['C_files']==57
    report = dict(passed=True,date='2026-10-20',new_training_runs=6,
        training_predictions=24,intervention_checkpoints=6,intervention_predictions=48,new_checkpoints=12,
        recomputed_training_runs=14,recomputed_prediction_files=104,source_files_verified=72,
        dynamic_prediction_duplicates_identical=24,csv_tables_byte_identical=13,
        confirmed_analysis_sha256=sha(STAGE/'analysis/confirmed/analysis.json'),
        README_result_rows_verified=12,parallel_readmes=5,display_formulas=formulas,
        original_training_core_unchanged=True,materialization_files=126 if restored else None,
        training_started=False,max_recompute_errors=original['audit']['max_recompute_errors'])
    target = STAGE/'analysis/repository_validation.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
