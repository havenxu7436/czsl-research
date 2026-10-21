"""Archive experiment 12: grouped dynamic-readout learning rates and update scales."""
import argparse
import ast
import difflib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from register_vgcpod_round_f import read, write, sha, copy, package_map

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'VG-CPOD方案/12'
TITLE='动态读出学习率分组与更新尺度对照'
PACKAGE='vgcpod_v1_roundK'
ZIP='vgcpod_v1_roundK_20261021.zip'
TAR='vgcpod_roundK_results_20261021.tar.gz'
SEEDS=(3351,3352,3353)
CASES=('K1','K2')


def import_data(source):
    source=Path(source).resolve()
    if STAGE.exists():raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True);mapping={}
    with zipfile.ZipFile(source/ZIP) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('/'):continue
            path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name and path.parts[0]==PACKAGE
            member=PurePosixPath(*path.parts[1:]).as_posix()
            if member.startswith('configs/'):target=member
            elif member.startswith(('assets/','references/')):target='configs/'+member
            elif member.startswith('validation/') or member=='release_manifest.json':target='results/package_validation/K/'+member
            elif Path(member).suffix in ('.md','.html'):target='results/package_support/K/'+member+'.txt'
            else:target='code/'+PACKAGE+'/'+member
            file=STAGE/target;file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(archive.read(name));mapping[member]=target
    write(STAGE/'package_members.json',dict(package=PACKAGE,members=mapping))
    for name in (ZIP,TAR):
        file=STAGE/'results/archives'/name;copy(source/name,file)
        file.with_name(name+'.sha256').write_text(sha(file)+'  '+name+'\n',encoding='ascii')
    export=source/'server_results_roundK_20261021';raw=[]
    for file in sorted((export/'roundK_20261021').rglob('*')):
        if not file.is_file() or file.name.endswith('.lock') or '__pycache__' in file.parts:continue
        target=STAGE/'results/raw_export'/file.relative_to(export);copy(file,target)
        raw.append(dict(path=target.relative_to(STAGE).as_posix(),sha256=sha(target),bytes=target.stat().st_size))
    write(STAGE/'results/raw_export_members.json',raw)
    members=[]
    for member,target in mapping.items():
        server=export/PACKAGE/member
        if server.is_file():
            assert server.read_bytes()==(STAGE/target).read_bytes(),member
            members.append(dict(package=PACKAGE,member=member,stage='12',sha256=sha(server)))
    before=STAGE.parent/'11';oldmap=package_map(before);changed=[];unchanged=[];diff=[]
    for member,target in sorted(mapping.items()):
        previous=(before/oldmap[member]).read_bytes() if member in oldmap else None
        current=(STAGE/target).read_bytes()
        if member.startswith('vgcpod/') or member in ('run.py','mean_gate_eval.py'):
            (unchanged if current==previous else changed).append(member)
        if member.endswith(('.py','.sh')) or member.startswith('configs/round_k/') or member=='references/pre_k_complete.json':
            old=previous.decode('utf-8') if previous is not None else ''
            diff.extend(difflib.unified_diff(old.splitlines(True),current.decode('utf-8').splitlines(True),
                fromfile='a/'+member if previous is not None else '/dev/null',tofile='b/'+member))
    assert set(changed)=={'vgcpod/engine.py'},changed
    (STAGE/'code/source_changes.patch').write_text(''.join(diff),encoding='utf-8')
    write(STAGE/'results/server_source_identity.json',dict(source_bytes_preserved=True,previous_stage='11',
        changed_runtime_files=changed,unchanged_runtime_files=unchanged,members=members))
    refs=['../11/analysis/confirmed/analysis.json','../11/analysis/confirmed/audit.json',
        '../11/results/archives/vgcpod_roundJ_results_20261021.tar.gz',
        '../11/results/raw_export/roundJ_20261021/feature_statistics/train_object_query_mean.npz']
    write(STAGE/'results/reused_reference_members.json',[dict(reference=p,sha256=sha(STAGE/p)) for p in refs])
    analysis=source/'analysis_roundK_results_20261021'
    for file in analysis.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts and 'qa' not in file.relative_to(analysis).parts and file.suffix in ('.json','.csv','.png','.svg','.log','.txt'):
            copy(file,STAGE/'analysis/confirmed'/file.relative_to(analysis))
    for folder in ('analysis_roundK_20261021','analysis_roundK_results_20261021'):
        for file in (source/folder).glob('*'):
            if file.suffix in ('.py','.cjs','.md'):copy(file,STAGE/'code/original_analysis'/folder/file.name)
    for file in source.glob('VG-CPOD_v1_K组*.md'):
        copy(file,STAGE/'results/package_support/original_reports'/(file.name+'.txt'))
    raw=STAGE/'results/raw_export/roundK_20261021';inventory=[]
    for case in CASES:
        copy(raw/('configs/'+case+'.json'),STAGE/('configs/runtime/'+case+'.json'))
        for seed in SEEDS:
            run=raw/case/'ut-zap50k/V1'/('seed%d'%seed)
            protocol,saved,exact=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'exact_selection/metrics.json')
            inventory.append(dict(case=case,seed=seed,completed_epochs=10,selected_epoch=saved['selected_epoch'],
                exact_selected_epoch=exact['selected_epoch'],path=run.relative_to(STAGE).as_posix(),
                config=protocol['config'],metrics=saved['metrics'],exact_selection_metrics=exact['metrics']))
    write(STAGE/'configs/experiment_inventory.json',inventory)
    write(STAGE/'configs/intervention_inventory.json',[
        dict(case=case,seed=seed,selection=selection,path='results/raw_export/roundK_20261021/'+folder+'/'+case+'/seed'+str(seed),
            conditions=['dynamic','train_mean'],optimizer_steps=0)
        for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')] for case in CASES for seed in SEEDS])
    write(STAGE/'external_inputs.json',dict(framework='frozen OpenAI CLIP ViT-L/14',
        cache='/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        feature_manifest_sha256='bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        J_references='../11/results/raw_export/roundJ_20261021',
        historical_references=['02','04','05','06','07','08','09','10','11'],
        cache_metadata='../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14',
        training_statistics='results/raw_export/roundK_20261021/feature_statistics/train_object_query_mean.npz',
        training_inputs='Reuse server CLIP feature arrays and the completed J-group TRAIN query mean.'))
    (STAGE/'.gitignore').write_text('analysis/recomputed/\n',encoding='utf-8')
    create_recompute(source)


def create_recompute(source):
    original=source/'analysis_roundK_results_20261021'
    common="""STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
ARCHIVE=STAGE/'results/raw_export'
RUNS=ARCHIVE/'roundK_20261021'
SOURCE=STAGE/'code/vgcpod_v1_roundK'
OLD=STAGE.parent/'11/results/raw_export'
CACHE=STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'
sys.path.insert(0,str(SOURCE))

def source_file(member):
    mapping=json.loads((STAGE/'package_members.json').read_text(encoding='utf-8'))['members']
    return STAGE/mapping[member]

"""
    text=(original/'analyze_k.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT=Path('),text.index('from ranking_diagnostics')
    text=text[:start]+common+text[end:]
    replacements={
        "ROOT/'analysis_20261018/analyze_results.py'":"STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "ROOT/'analysis_roundJ_results_20261021/analysis.json'":"STAGE.parent/'11/analysis/confirmed/analysis.json'",
        "ROOT/'vgcpod_roundJ_results_20261021.tar.gz'":"STAGE.parent/'11/results/archives/vgcpod_roundJ_results_20261021.tar.gz'",
        "ROOT/'vgcpod_roundK_results_20261021.tar.gz'":"STAGE/'results/archives/vgcpod_roundK_results_20261021.tar.gz'",
        "SOURCE/'release_manifest.json'":"source_file('release_manifest.json')",
        "SOURCE/name":"source_file(name)",
        "SOURCE/'mean_gate_eval.py'":"source_file('mean_gate_eval.py')"}
    for old,new in replacements.items():
        assert old in text,old;text=text.replace(old,new)
    text=text.replace('np.trapezoid(',"getattr(np,'trapezoid',np.trapz)(")
    (STAGE/'analysis/recompute_metrics.py').write_text('# Repository paths adapted; numerical analysis preserved.\n'+text,encoding='utf-8')
    text=(original/'supplement.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT=Path('),text.index('from readout_diagnostics')
    text=text[:start]+"""STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundK'))
"""+text[end:]
    text=text.replace('from analyze_k import csv_write,aggregate','from recompute_metrics import csv_write,aggregate')
    start,end=text.index('    stage=case[0]'),text.index('def load')
    text=text[:start]+"""    stages={'D1':('05','D','20261020'),'G0':('08','G','20261020'),
            'I2':('10','I','20261021'),'J1':('11','J','20261021'),
            'K1':('12','K','20261021'),'K2':('12','K','20261021')}
    number,letter,date=stages[case]
    return STAGE.parent/number/'results/raw_export'/('round%s_%s'%(letter,date))/case/'ut-zap50k/V1'/('seed%d'%seed)


"""+text[end:]
    text=text.replace("ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json'",
        "STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json'")
    (STAGE/'analysis/recompute_supplement.py').write_text(text,encoding='utf-8')
    text=(original/'make_figures.py').read_text(encoding='utf-8').replace('OUT=Path(__file__).resolve().parent',
        "OUT=Path(__file__).resolve().parent/'recomputed'")
    (STAGE/'analysis/recompute_figures.py').write_text(text,encoding='utf-8')
    entry='''"""Recompute archived K1/K2 predictions, updates, error decomposition and figures."""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    sys.path.insert(0,str(folder))
    (folder/'recomputed').mkdir(exist_ok=True)
    for name in ('recompute_metrics.py','recompute_supplement.py','recompute_figures.py'):
        runpy.run_path(str(folder/name),run_name='__main__')
'''
    (STAGE/'analysis/recompute.py').write_text(entry,encoding='utf-8')


def finalize():
    write(STAGE/'inputs.json',dict(line='VG-CPOD方案',stage='12',version='v1',title=TITLE,knowledge=[],
        completed_runs=6,model_epochs=60,training_evaluations=24,secondary_selection_evaluations=24,
        intervention_checkpoint_selections=12,intervention_conditions=24,intervention_evaluations=96,
        predictions=144,checkpoint_files=24,completed_cases=list(CASES),completed_seeds=list(SEEDS),
        run_matrix=[dict(case=c,seed=s) for c in CASES for s in SEEDS],package_member_map='package_members.json',
        experiment_inventory='configs/experiment_inventory.json',
        code_files=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        configs=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'configs').rglob('*.json'))],
        documents=[p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='12',primary_package=PACKAGE,
        commands=[dict(source='README.md',language=language,text=text) for language,text in
            re.findall(r'```(bash|powershell)\n(.*?)```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)]))
    files=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in files if p.name!='file_manifest.json'])
    stages=[s for s in read(ROOT/'provenance/stages.json') if s.get('id')!='31']
    stages.append(dict(id='31',line='VG-CPOD方案',number='12',version='v1',title='12 · '+TITLE,
        path='VG-CPOD方案/12',primary_package=PACKAGE,status='K1/K2各三seed、动态读出独立学习率、双准则及十二项均值评估归档',
        completed_runs=6,model_epochs=60,evaluations=144,predictions=144,checkpoints=24))
    write(ROOT/'provenance/stages.json',stages)
    write(ROOT/'provenance/updates/20261021_vgcpod_12.json',dict(date='2026-10-21',line='VG-CPOD方案',number='12',
        version='v1',title=TITLE,imported_from='K release ZIP and downloaded K server export',
        new_runs=[dict(case=c,seed=s) for c in CASES for s in SEEDS],reference_stages=['02','04','05','06','07','08','09','10','11'],
        formula_revision='Forward and loss preserved from J1; AdamW groups: alpha_d=m_d*alpha_CP, m_d=.3/.1; diagnostics reconstruct actual adaptive update',
        trainable_parameters=147456,added_parameters=0,
        planned_only=['Trainable CLIP local discriminant evidence; R0/R1/R2 formal comparison; no training code implemented']))
    owned={p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/12/file_manifest.json','README.md','VG-CPOD方案/README.md','tools/reproduce.py',
        'tools/register_vgcpod_round_k.py','provenance/stages.json','provenance/updates/20261021_vgcpod_12.json'])
    current={r['path']:r for r in read(ROOT/'provenance/current_manifest.json')}
    for relative in sorted(owned):
        file=ROOT/relative;current[relative]=dict(path=relative,sha256=sha(file),bytes=file.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[p] for p in sorted(current)])


def check():
    files=read(STAGE/'file_manifest.json')
    for row in files:assert sha(STAGE/row['path'])==row['sha256'],row['path']
    mapping=package_map(STAGE);release=read(STAGE/mapping['release_manifest.json'])['files']
    assert set(mapping)==set(release)|{'release_manifest.json'}
    with zipfile.ZipFile(STAGE/'results/archives'/ZIP) as archive:
        for name,target in mapping.items():assert archive.read(PACKAGE+'/'+name)==(STAGE/target).read_bytes(),name
    for name,fingerprint in release.items():assert sha(STAGE/mapping[name])==fingerprint,name
    for name in (ZIP,TAR):
        file=STAGE/'results/archives'/name;assert sha(file)==file.with_name(name+'.sha256').read_text().split()[0]
    raw=STAGE/'results/raw_export/roundK_20261021'
    assert len(list(raw.rglob('predictions/*.npz')))==144 and len(list(raw.rglob('checkpoints/*.pt')))==24
    assert (raw/'launcher.exitcode').read_text().strip()=='0'
    assert sha(raw/'feature_statistics/train_object_query_mean.npz')==sha(STAGE.parent/'11/results/raw_export/roundJ_20261021/feature_statistics/train_object_query_mean.npz')
    for case in CASES:
        for seed in SEEDS:
            run=raw/case/'ut-zap50k/V1'/('seed%d'%seed)
            p,m,h,e=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json'),read(run/'exact_selection/metrics.json')
            assert len(h)==10 and h[-1]['step']==3600 and p['trainable_parameters']==147456
            assert m['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_native'])+1
            assert e['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_exact'])+1
            assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')==m['checkpoint_sha256']
            assert sha(run/'checkpoints/best_exact.pt')==e['checkpoint_sha256']
            for name,fingerprint in p['source_sha256'].items():assert sha(STAGE/mapping[name])==fingerprint,(case,seed,name)
    docs=list(STAGE.rglob('README.md'));assert len(docs)==5
    for file in docs:
        text=file.read_text(encoding='utf-8');assert text.count('```')%2==0 and '$$' not in text
        for link in re.findall(r'\]\(([^)\n]+)\)',text):
            link=link.split('#')[0];assert not link or re.match(r'^[a-z]+://',link) or (file.parent/link).exists(),(file,link)
    for file in list((STAGE/'code').rglob('*.py'))+list((STAGE/'analysis').glob('*.py')):
        ast.parse(file.read_text(encoding='utf-8-sig'))
    return dict(passed=True,files_verified=len(files),trained_runs=6,training_predictions=24,
        secondary_selection_predictions=24,intervention_predictions=96,predictions=144,checkpoint_files=24,
        readmes=5,display_formulas=sum(p.read_text(encoding='utf-8').count('```math\n') for p in docs),training_started=False)


def materialize(destination):
    destination=Path(destination).expanduser()
    if not destination.is_absolute():raise ValueError('Use an absolute destination')
    destination=destination.resolve()
    if destination.exists() or destination==ROOT or ROOT in destination.parents:raise FileExistsError(destination)
    mapping=package_map(STAGE)
    for member,relative in mapping.items():
        target=destination/PACKAGE/member;copy(STAGE/relative,target)
        if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
    for member,fingerprint in read(destination/PACKAGE/'release_manifest.json')['files'].items():
        assert sha(destination/PACKAGE/member)==fingerprint,member
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='12',package=PACKAGE,files_verified=len(mapping),training_started=False))
    return dict(line='VG-CPOD方案',stage='12',package=str(destination/PACKAGE),files_verified=len(mapping),training_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--finalize',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--materialize',type=Path);args=parser.parse_args()
    if args.import_data and args.source:import_data(args.source);result=dict(imported=True,stage='12')
    elif args.finalize:finalize();result=check()
    elif args.check:result=check()
    elif args.materialize:result=materialize(args.materialize)
    else:parser.error('Choose an action')
    print(json.dumps(result,ensure_ascii=False))
