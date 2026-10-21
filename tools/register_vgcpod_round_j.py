"""Archive experiment 11: independent static and dynamic CP readouts."""
import argparse
import ast
import difflib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from register_vgcpod_round_f import read, write, sha, copy, package_map

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'VG-CPOD方案/11'
TITLE='静态与动态CP独立读出'
PACKAGE='vgcpod_v1_roundJ'
ZIP='vgcpod_v1_roundJ_20261021.zip'
TAR='vgcpod_roundJ_results_20261021.tar.gz'
SEEDS=(3351,3352,3353)


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
            elif member.startswith('validation/') or member=='release_manifest.json':target='results/package_validation/J/'+member
            elif Path(member).suffix in ('.md','.html'):target='results/package_support/J/'+member+'.txt'
            else:target='code/'+PACKAGE+'/'+member
            file=STAGE/target;file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(archive.read(name));mapping[member]=target
    write(STAGE/'package_members.json',dict(package=PACKAGE,members=mapping))
    for name in (ZIP,TAR):
        file=STAGE/'results/archives'/name;copy(source/name,file)
        file.with_name(name+'.sha256').write_text(sha(file)+'  '+name+'\n',encoding='ascii')
    export=source/'server_results_roundJ_20261021';raw=[]
    for file in sorted((export/'roundJ_20261021').rglob('*')):
        if not file.is_file() or file.name.endswith('.lock') or '__pycache__' in file.parts:continue
        target=STAGE/'results/raw_export'/file.relative_to(export);copy(file,target)
        raw.append(dict(path=target.relative_to(STAGE).as_posix(),sha256=sha(target),bytes=target.stat().st_size))
    write(STAGE/'results/raw_export_members.json',raw)
    members=[]
    for member,target in mapping.items():
        server=export/PACKAGE/member
        if server.is_file():
            assert server.read_bytes()==(STAGE/target).read_bytes(),member
            members.append(dict(package=PACKAGE,member=member,stage='11',sha256=sha(server)))
    before=STAGE.parent/'10';oldmap=package_map(before);changed=[];unchanged=[];diff=[]
    for member,target in sorted(mapping.items()):
        previous=(before/oldmap[member]).read_bytes() if member in oldmap else None
        current=(STAGE/target).read_bytes()
        if member.startswith('vgcpod/') or member in ('run.py','mean_gate_eval.py'):
            (unchanged if current==previous else changed).append(member)
        if member.endswith(('.py','.sh')) or member.startswith('configs/round_j/') or member=='references/pre_j_complete.json':
            old=previous.decode('utf-8') if previous is not None else ''
            diff.extend(difflib.unified_diff(old.splitlines(True),current.decode('utf-8').splitlines(True),
                fromfile='a/'+member if previous is not None else '/dev/null',tofile='b/'+member))
    assert set(changed)=={'vgcpod/model.py','vgcpod/engine.py','mean_gate_eval.py'}
    (STAGE/'code/source_changes.patch').write_text(''.join(diff),encoding='utf-8')
    write(STAGE/'results/server_source_identity.json',dict(source_bytes_preserved=True,previous_stage='10',
        changed_runtime_files=changed,unchanged_runtime_files=unchanged,members=members))
    references=[]
    for relative in ('../10/analysis/confirmed/analysis.json','../10/analysis/confirmed/audit.json',
        '../10/results/archives/vgcpod_roundI_results_20261021.tar.gz',
        '../10/results/raw_export/roundI_20261021/feature_statistics/train_object_query_mean.npz'):
        references.append(dict(reference=relative,sha256=sha(STAGE/relative)))
    write(STAGE/'results/reused_reference_members.json',references)
    analysis=source/'analysis_roundJ_results_20261021'
    for file in analysis.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts and file.suffix in ('.json','.csv','.png','.log','.txt'):
            copy(file,STAGE/'analysis/confirmed'/file.relative_to(analysis))
    for folder in ('analysis_roundJ_20261021','analysis_roundJ_results_20261021'):
        for file in (source/folder).glob('*'):
            if file.suffix in ('.py','.cjs'):copy(file,STAGE/'code/original_analysis'/folder/file.name)
    for file in source.glob('VG-CPOD_v1_J组*.md'):
        copy(file,STAGE/'results/package_support/original_reports'/(file.name+'.txt'))
    raw=STAGE/'results/raw_export/roundJ_20261021'
    copy(raw/'configs/J1.json',STAGE/'configs/runtime/J1.json')
    inventory=[]
    for seed in SEEDS:
        run=raw/'J1/ut-zap50k/V1'/('seed%d'%seed)
        protocol,saved,exact=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'exact_selection/metrics.json')
        inventory.append(dict(case='J1',seed=seed,completed_epochs=10,selected_epoch=saved['selected_epoch'],
            exact_selected_epoch=exact['selected_epoch'],path=run.relative_to(STAGE).as_posix(),
            config=protocol['config'],metrics=saved['metrics'],exact_selection_metrics=exact['metrics']))
    write(STAGE/'configs/experiment_inventory.json',inventory)
    write(STAGE/'configs/intervention_inventory.json',[
        dict(case='J1',seed=seed,selection=selection,path='results/raw_export/roundJ_20261021/'+folder+'/J1/seed'+str(seed),
            conditions=['dynamic','train_mean'],optimizer_steps=0)
        for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')] for seed in SEEDS])
    write(STAGE/'external_inputs.json',dict(framework='frozen OpenAI CLIP ViT-L/14',
        cache='/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        feature_manifest_sha256='bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        I_references='../10/results/raw_export/roundI_20261021',
        historical_references=['02','04','05','06','07','08','09','10'],
        cache_metadata='../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14',
        training_statistics='results/raw_export/roundJ_20261021/feature_statistics/train_object_query_mean.npz',
        training_inputs='Reuse server CLIP feature arrays and the completed I-group TRAIN query mean.'))
    (STAGE/'.gitignore').write_text('analysis/recomputed/\n',encoding='utf-8')
    create_recompute(source)


def create_recompute(source):
    original=source/'analysis_roundJ_results_20261021'
    common="""STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
ARCHIVE=STAGE/'results/raw_export'
RUNS=ARCHIVE/'roundJ_20261021'
SOURCE=STAGE/'code/vgcpod_v1_roundJ'
OLD=STAGE.parent/'10/results/raw_export'
CACHE=STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'
sys.path.insert(0,str(SOURCE))

def source_file(member):
    mapping=json.loads((STAGE/'package_members.json').read_text(encoding='utf-8'))['members']
    return STAGE/mapping[member]

"""
    text=(original/'analyze_j.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT=Path('),text.index('from ranking_diagnostics')
    text=text[:start]+common+text[end:]
    replacements={
        "ROOT/'analysis_20261018/analyze_results.py'":"STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "ROOT/'analysis_roundI_results_20261021/analysis.json'":"STAGE.parent/'10/analysis/confirmed/analysis.json'",
        "ROOT/'vgcpod_roundI_results_20261021.tar.gz'":"STAGE.parent/'10/results/archives/vgcpod_roundI_results_20261021.tar.gz'",
        "ROOT/'vgcpod_roundJ_results_20261021.tar.gz'":"STAGE/'results/archives/vgcpod_roundJ_results_20261021.tar.gz'",
        "SOURCE/'release_manifest.json'":"source_file('release_manifest.json')",
        "SOURCE/name":"source_file(name)",
        "SOURCE/'mean_gate_eval.py'":"source_file('mean_gate_eval.py')"}
    for old,new in replacements.items():
        assert old in text,old;text=text.replace(old,new)
    text=text.replace('np.trapezoid(',"getattr(np,'trapezoid',np.trapz)(")
    (STAGE/'analysis/recompute_metrics.py').write_text('# Repository paths adapted; numerical analysis preserved.\n'+text,encoding='utf-8')
    text=(original/'supplement.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT=Path('),text.index('from vgcpod.common')
    text=text[:start]+"""STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
NEW=STAGE/'results/raw_export/roundJ_20261021'
OLD=STAGE.parent/'10/results/raw_export/roundI_20261021'
sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundJ'))
"""+text[end:]
    text=text.replace("ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json'",
        "STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json'")
    (STAGE/'analysis/recompute_supplement.py').write_text(text,encoding='utf-8')
    text=(original/'make_figures.py').read_text(encoding='utf-8').replace('OUT=Path(__file__).resolve().parent',
        "OUT=Path(__file__).resolve().parent/'recomputed'")
    (STAGE/'analysis/recompute_figures.py').write_text(text,encoding='utf-8')
    entry='''"""Recompute archived J1 metrics, optimizer measurements and figures without training."""
from pathlib import Path
import runpy

if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    (folder/'recomputed').mkdir(exist_ok=True)
    for name in ('recompute_metrics.py','recompute_supplement.py','recompute_figures.py'):
        runpy.run_path(str(folder/name),run_name='__main__')
'''
    (STAGE/'analysis/recompute.py').write_text(entry,encoding='utf-8')


def finalize():
    write(STAGE/'inputs.json',dict(line='VG-CPOD方案',stage='11',version='v1',title=TITLE,knowledge=[],
        completed_runs=3,model_epochs=30,training_evaluations=12,secondary_selection_evaluations=12,
        intervention_checkpoint_selections=6,intervention_conditions=12,intervention_evaluations=48,
        predictions=72,checkpoint_files=12,completed_cases=['J1'],completed_seeds=list(SEEDS),
        run_matrix=[dict(case='J1',seed=s) for s in SEEDS],package_member_map='package_members.json',
        experiment_inventory='configs/experiment_inventory.json',
        code_files=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        configs=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'configs').rglob('*.json'))],
        documents=[p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='11',primary_package=PACKAGE,
        commands=[dict(source='README.md',language=language,text=text) for language,text in
            re.findall(r'```(bash|powershell)\n(.*?)```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)]))
    files=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in files if p.name!='file_manifest.json'])
    stages=[s for s in read(ROOT/'provenance/stages.json') if s.get('id')!='30']
    stages.append(dict(id='30',line='VG-CPOD方案',number='11',version='v1',title='11 · '+TITLE,
        path='VG-CPOD方案/11',primary_package=PACKAGE,status='J1三seed、静态与动态独立读出、双准则及六项均值评估归档',
        completed_runs=3,model_epochs=30,evaluations=72,predictions=72,checkpoints=12))
    write(ROOT/'provenance/stages.json',stages)
    write(ROOT/'provenance/updates/20261021_vgcpod_11.json',dict(date='2026-10-21',line='VG-CPOD方案',number='11',
        version='v1',title=TITLE,imported_from='J release ZIP and downloaded J server export',
        new_runs=[dict(case='J1',seed=s) for s in SEEDS],reference_stages=['02','04','05','06','07','08','09','10'],
        formula_revision='Separate static Wc and dynamic Wd readouts, shared Au/Av; I2 residual gates with tau=.56 and eta=.25',
        trainable_parameters=147456,added_parameters=24576,
        planned_only=['K1: Wd learning-rate multiplier .3','K2: Wd learning-rate multiplier .1']))
    owned={p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/11/file_manifest.json','README.md','VG-CPOD方案/README.md','tools/reproduce.py',
        'tools/register_vgcpod_round_j.py','provenance/stages.json','provenance/updates/20261021_vgcpod_11.json'])
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
    raw=STAGE/'results/raw_export/roundJ_20261021'
    assert len(list(raw.rglob('predictions/*.npz')))==72 and len(list(raw.rglob('checkpoints/*.pt')))==12
    assert (raw/'launcher.exitcode').read_text().strip()=='0'
    stats=raw/'feature_statistics/train_object_query_mean.npz'
    assert sha(stats)==sha(STAGE.parent/'10/results/raw_export/roundI_20261021/feature_statistics/train_object_query_mean.npz')
    for seed in SEEDS:
        run=raw/'J1/ut-zap50k/V1'/('seed%d'%seed)
        p,m,h,e=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json'),read(run/'exact_selection/metrics.json')
        assert len(h)==10 and h[-1]['step']==3600 and p['trainable_parameters']==147456
        assert m['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_native'])+1
        assert e['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_exact'])+1
        assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')==m['checkpoint_sha256']
        assert sha(run/'checkpoints/best_exact.pt')==e['checkpoint_sha256']
        for name,fingerprint in p['source_sha256'].items():assert sha(STAGE/mapping[name])==fingerprint,(seed,name)
    docs=list(STAGE.rglob('README.md'));assert len(docs)==5
    for file in docs:
        text=file.read_text(encoding='utf-8');assert text.count('```')%2==0 and '$$' not in text
        for link in re.findall(r'\]\(([^)\n]+)\)',text):
            link=link.split('#')[0];assert not link or re.match(r'^[a-z]+://',link) or (file.parent/link).exists(),(file,link)
    for file in list((STAGE/'code').rglob('*.py'))+list((STAGE/'analysis').glob('*.py')):
        ast.parse(file.read_text(encoding='utf-8-sig'))
    return dict(passed=True,files_verified=len(files),trained_runs=3,training_predictions=12,
        secondary_selection_predictions=12,intervention_predictions=48,predictions=72,checkpoint_files=12,
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
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='11',package=PACKAGE,files_verified=len(mapping),training_started=False))
    return dict(line='VG-CPOD方案',stage='11',package=str(destination/PACKAGE),files_verified=len(mapping),training_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--finalize',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--materialize',type=Path);args=parser.parse_args()
    if args.import_data and args.source:import_data(args.source);result=dict(imported=True,stage='11')
    elif args.finalize:finalize();result=check()
    elif args.check:result=check()
    elif args.materialize:result=materialize(args.materialize)
    else:parser.error('Choose an action')
    print(json.dumps(result,ensure_ascii=False))
