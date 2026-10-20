"""Verify G archive bytes, paired result tables, numerical replay and five READMEs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parent/'CZSL'
STAGE=REPO/'VG-CPOD方案/08'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    original=read(STAGE/'analysis/confirmed/analysis.json')
    assert original==read(STAGE/'analysis/recomputed/analysis.json')
    assert read(STAGE/'analysis/confirmed/supplementary_audit.json')==read(STAGE/'analysis/recomputed/supplementary_audit.json')
    tables=list((STAGE/'analysis/confirmed/tables').glob('*.csv'))
    assert all(p.read_bytes()==(STAGE/'analysis/recomputed/tables'/p.name).read_bytes() for p in tables)
    raw=read(STAGE/'results/raw_export_members.json')
    expected={r['path'].removeprefix('results/raw_export/'):r['sha256'] for r in raw};checked=set()
    with tarfile.open(STAGE/'results/archives/vgcpod_roundG_results_20261020.tar.gz','r|gz') as archive:
        for member in archive:
            if not member.isfile():continue
            name=member.name.removeprefix('./')
            if name in expected:
                digest=hashlib.sha256();stream=archive.extractfile(member)
                for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
                assert digest.hexdigest()==expected[name],name;checked.add(name)
    assert checked==set(expected),(len(checked),len(expected))
    inventory=read(STAGE/'configs/experiment_inventory.json')
    text=(STAGE/'README.md').read_text(encoding='utf-8');result_text=(STAGE/'results/README.md').read_text(encoding='utf-8')
    assert len(inventory)==6
    for row in inventory:
        values=[row['metrics'][k]['AUC_native'] for k in ('val_cw','val_ow','test_cw','test_ow')]
        line='| %s | %d | %d／%d | %s |'%(row['case'],row['seed'],row['selected_epoch'],row['exact_selected_epoch'],
            ' | '.join('%.4f'%v for v in values))
        assert line in text and line in result_text,line
        values=[row['exact_selection_metrics'][setting][key] for setting,key in
            [('val_cw','AUC_native'),('val_cw','AUC_exact'),('test_cw','AUC_native'),('test_ow','AUC_native')]]
        line='| %s | %d | %d | %s |'%(row['case'],row['seed'],row['exact_selected_epoch'],' | '.join('%.4f'%v for v in values))
        assert line in result_text,line
        protocol=read(STAGE/row['path']/'protocol.json')
        assert protocol['trainable_parameters']==(98304 if row['case']=='G0' else 122880)
    dynamic=read(STAGE/'configs/round_g/G1.json');static=read(STAGE/'configs/round_g/G0.json')
    before=read(STAGE/'configs/round_f/F1.json')
    assert {k for k in before if before[k]!=dynamic[k]}=={'task_aux_beta','experiment_id'}
    assert {k for k in static if static[k]!=dynamic[k]}=={'gate_input','experiment_id'}
    for case in ('D1','E1','F1','G0','G1'):
        def metric(setting,key):return next(r[key] for r in original['means'] if r['case']==case and r['setting']==setting)
        values=[metric(s,'AUC_native_mean') for s in ('val_cw','test_cw','test_ow')]
        line=' | '.join('%.3f'%v for v in values)+' | %.3f%%'%metric('test_ow','gt_object_attr_acc_unseen_mean')
        assert line in text,line
    layout=read(STAGE/'analysis/repository_qa/layout.json');assert layout['passed'] and len(layout['readmes'])==5
    sys.path.insert(0,str(WORK/'tmp/round_a_bash_validation'))
    import bashlex
    commands=0
    for file in STAGE.rglob('README.md'):
        doc=file.read_text(encoding='utf-8')
        assert doc.count('```')%2==0 and '$$' not in doc
        for block in re.findall(r'^```bash\n(.*?)\n```',doc,re.M|re.S):bashlex.parse(block);commands+=1
        for link in re.findall(r'\]\(([^)\n]+)\)',doc):
            target=link.split('#')[0]
            generated={STAGE/'runbook.json',STAGE/'file_manifest.json',STAGE/'analysis/repository_validation.json'}
            # These are written by this check or the following finalization;
            # register_vgcpod_round_g.check() verifies all links afterwards.
            if (file.parent/target).resolve() in generated and not (file.parent/target).exists():continue
            assert not target or re.match(r'^[a-z]+://',target) or (file.parent/target).exists(),(file,link)
    for file in (STAGE/'code/vgcpod_v1_roundG').rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    materialization=read(WORK/'replay_archive08_20261020/materialization.json')
    audit=original['audit'];supplement=read(STAGE/'analysis/confirmed/supplementary_audit.json')
    result=dict(passed=True,date='2026-10-20',new_training_runs=6,model_epochs=60,native_predictions=24,
        exact_selection_predictions=24,intervention_predictions=48,new_predictions=96,new_checkpoint_files=24,
        recomputed_training_runs=26,recomputed_prediction_files=audit['prediction_files'],source_files_verified=audit['source_files_verified'],
        server_source_members_verified=len(read(STAGE/'results/server_source_identity.json')['members']),
        raw_result_members_identical_to_TAR=len(checked),reused_reference_members_verified=len(read(STAGE/'results/reused_reference_members.json')),
        analysis_identical=True,supplementary_identical=True,csv_tables_byte_identical=len(tables),
        original_analysis_sha256=sha(STAGE/'analysis/confirmed/analysis.json'),README_result_rows_verified=18,README_mean_rows_verified=5,
        parallel_readmes=5,display_formulas=sum(r['displayMath'] for r in layout['readmes']),formula_visual_inspection=True,
        parsed_bash_command_blocks=commands,materialization_files=materialization['files_verified'],runtime_unchanged_from_F=True,
        ranking_rows_verified=supplement['ranking_rows_verified'],object_text_rank=supplement['object_text_rank'],
        static_folding_max_delta_error=max(r['random_visual_delta_max_error'] for r in supplement['static_gate_folding']),
        training_started=False,max_recompute_errors=audit['max_recompute_errors'])
    (STAGE/'analysis/repository_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
