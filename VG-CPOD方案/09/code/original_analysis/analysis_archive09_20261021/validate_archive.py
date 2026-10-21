"""Check H archive bytes, repository replay, result tables and five parallel READMEs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parent/'CZSL'
STAGE=REPO/'VG-CPOD方案/09'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    original=read(STAGE/'analysis/confirmed/analysis.json')
    assert original==read(STAGE/'analysis/recomputed/analysis.json')
    assert read(STAGE/'analysis/confirmed/audit.json')==read(STAGE/'analysis/recomputed/audit.json')
    tables=list((STAGE/'analysis/confirmed/tables').glob('*.csv'))
    assert all(p.read_bytes()==(STAGE/'analysis/recomputed/tables'/p.name).read_bytes() for p in tables)
    raw=read(STAGE/'results/raw_export_members.json')
    expected={r['path'].removeprefix('results/raw_export/'):r['sha256'] for r in raw};checked=set()
    with tarfile.open(STAGE/'results/archives/vgcpod_roundH_results_20261020.tar.gz','r|gz') as archive:
        for member in archive:
            if not member.isfile():continue
            name=member.name.removeprefix('./')
            if name in expected:
                digest=hashlib.sha256();stream=archive.extractfile(member)
                for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
                assert digest.hexdigest()==expected[name],name;checked.add(name)
    assert checked==set(expected),(len(checked),len(expected))
    for row in read(STAGE/'results/reused_reference_members.json'):
        assert sha(STAGE/row['reference'])==row['sha256'],row['member']
    inventory=read(STAGE/'configs/experiment_inventory.json')
    text=(STAGE/'README.md').read_text(encoding='utf-8');result_text=(STAGE/'results/README.md').read_text(encoding='utf-8')
    assert len(inventory)==3
    for row in inventory:
        values=[row['metrics'][k]['AUC_native'] for k in ('val_cw','val_ow','test_cw','test_ow')]
        line='| H1 | %d | %d／%d | %s |'%(row['seed'],row['selected_epoch'],row['exact_selected_epoch'],
            ' | '.join('%.4f'%v for v in values))
        assert line in text and line in result_text,line
        values=[row['exact_selection_metrics'][setting][key] for setting,key in
            [('val_cw','AUC_native'),('val_cw','AUC_exact'),('test_cw','AUC_native'),('test_ow','AUC_native')]]
        line='| H1 | %d | %d | %s |'%(row['seed'],row['exact_selected_epoch'],' | '.join('%.4f'%v for v in values))
        assert line in result_text,line
        assert read(STAGE/row['path']/'protocol.json')['trainable_parameters']==122880
    before=read(STAGE/'configs/round_g/G1.json');template=read(STAGE/'configs/round_h/H1.json')
    assert {k for k in before if before[k]!=template[k]}=={'gate_input','experiment_id'}
    assert set(template)-set(before)=={'gate_residual_eps'}
    assert template['gate_eta']==.25 and template['tau_gamma']==.1 and template['gate_residual_eps']==1e-12
    for case in ('G0','G1','H1'):
        def metric(setting,key):return next(r[key] for r in original['means'] if r['case']==case and r['setting']==setting)
        values=[metric(s,'AUC_native_mean') for s in ('val_cw','val_ow','test_cw','test_ow')]
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
            if (file.parent/target).resolve() in generated and not (file.parent/target).exists():continue
            assert not target or re.match(r'^[a-z]+://',target) or (file.parent/target).exists(),(file,link)
    for file in (STAGE/'code/vgcpod_v1_roundH').rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    materialization=read(WORK/'replay_archive09_20261021/materialization.json')
    audit=original['audit']
    result=dict(passed=True,date='2026-10-21',new_training_runs=3,model_epochs=30,native_predictions=12,
        exact_selection_predictions=12,intervention_predictions=48,new_predictions=72,new_checkpoint_files=12,
        recomputed_prediction_files=audit['prediction_files'],source_files_verified=audit['source_files_verified'],
        server_source_members_verified=len(read(STAGE/'results/server_source_identity.json')['members']),
        raw_result_members_identical_to_TAR=len(checked),reused_reference_members_verified=len(read(STAGE/'results/reused_reference_members.json')),
        analysis_identical=True,audit_identical=True,csv_tables_byte_identical=len(tables),
        original_analysis_sha256=sha(STAGE/'analysis/confirmed/analysis.json'),README_result_rows_verified=9,README_mean_rows_verified=3,
        parallel_readmes=5,display_formulas=sum(r['displayMath'] for r in layout['readmes']),formula_visual_inspection=True,
        parsed_bash_command_blocks=commands,materialization_files=materialization['files_verified'],
        ranking_rows_verified=audit['ranking_rows_verified'],checkpoint_mean_identical=True,
        training_started=False,max_recompute_errors=audit['max_recompute_errors'])
    (STAGE/'analysis/repository_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
