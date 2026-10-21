"""Archive experiment 10: residual magnitude preservation and gate temperature comparison."""
import argparse
import ast
import difflib
import json
from pathlib import Path,PurePosixPath
import re
import zipfile

from register_vgcpod_round_f import read,write,sha,copy,package_map

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'VG-CPOD方案/10'
TITLE='残差幅值保留与门控温度对照'
PACKAGE='vgcpod_v1_roundI'
ZIP='vgcpod_v1_roundI_20261021.zip'
TAR='vgcpod_roundI_results_20261021.tar.gz'
SEEDS=(3351,3352,3353)
SOURCE_STAGES={'vgcpod_v1_roundC':'04','vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06',
               'vgcpod_v1_roundF':'07','vgcpod_v1_roundG':'08','vgcpod_v1_roundH':'09','vgcpod_v1_roundI':'10'}


def import_data(source):
    source=Path(source).resolve()
    if (STAGE/'README.md').exists():raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True,exist_ok=True);mapping={}
    with zipfile.ZipFile(source/ZIP) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('/'):continue
            path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name
            assert path.parts[0]==PACKAGE
            member=PurePosixPath(*path.parts[1:]).as_posix()
            if member.startswith('configs/'):target=member
            elif member.startswith(('assets/','references/')):target='configs/'+member
            elif member.startswith('validation/') or member=='release_manifest.json':target='results/package_validation/I/'+member
            elif Path(member).suffix in ('.md','.html'):target='results/package_support/I/'+member+'.txt'
            else:target='code/'+PACKAGE+'/'+member
            file=STAGE/target;file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(archive.read(name));mapping[member]=target
    write(STAGE/'package_members.json',dict(package=PACKAGE,members=mapping))
    for name in (ZIP,TAR):
        file=STAGE/'results/archives'/name;copy(source/name,file)
        file.with_name(name+'.sha256').write_text(sha(file)+'  '+name+'\n',encoding='ascii')
    export=source/'server_results_roundI_20261021';raw=[]
    for file in sorted((export/'roundI_20261021').rglob('*')):
        if not file.is_file() or file.name.endswith('.lock') or '__pycache__' in file.parts:continue
        target=STAGE/'results/raw_export'/file.relative_to(export);copy(file,target)
        raw.append(dict(path=target.relative_to(STAGE).as_posix(),sha256=sha(target),bytes=target.stat().st_size))
    write(STAGE/'results/raw_export_members.json',raw)
    sources,reused=[],[]
    for package,number in SOURCE_STAGES.items():
        stage=STAGE.parent/number
        for member,target in package_map(stage).items():
            server=export/package/member
            if server.is_file():
                assert server.read_bytes()==(stage/target).read_bytes(),(package,member)
                sources.append(dict(package=package,member=member,stage=number,sha256=sha(server)))
    for group in ('roundC_20261020','roundD_20261020','roundE_20261020','roundF_20261020','roundG_20261020','roundH_20261020'):
        for file in sorted((export/group).rglob('*')):
            if not file.is_file() or file.name.endswith('.lock') or '__pycache__' in file.parts:continue
            rel=file.relative_to(export);fingerprint=sha(file)
            candidates=[STAGE.parent/n/'results/raw_export'/rel for n in ('04','05','06','07','08','09')]
            match=next((p for p in candidates if p.is_file() and sha(p)==fingerprint),None)
            if match:
                target='../'+match.relative_to(STAGE.parent).as_posix()
            else:
                destination=STAGE/'results/reference_exports'/rel;copy(file,destination)
                target=destination.relative_to(STAGE).as_posix()
            reused.append(dict(member=rel.as_posix(),reference=target,sha256=fingerprint))
    before=STAGE.parent/'09';before_map=package_map(before);unchanged=[];changed=[]
    for member,target in mapping.items():
        if member.startswith('vgcpod/') or member in ('run.py','mean_gate_eval.py'):
            if member in before_map and (STAGE/target).read_bytes()==(before/before_map[member]).read_bytes():unchanged.append(member)
            else:changed.append(member)
    assert set(changed)=={'vgcpod/gate_features.py','vgcpod/engine.py','vgcpod/model.py'}
    write(STAGE/'results/server_source_identity.json',dict(source_bytes_preserved=True,
        previous_stage='09',changed_runtime_files=changed,unchanged_runtime_files=unchanged,members=sources))
    write(STAGE/'results/reused_reference_members.json',reused)
    diff=[]
    for member,target in sorted(mapping.items()):
        if not (member.endswith(('.py','.sh')) or member.startswith('configs/round_i/') or member=='references/pre_i_complete.json'):continue
        old=(before/before_map[member]).read_text(encoding='utf-8') if member in before_map else ''
        new=(STAGE/target).read_text(encoding='utf-8')
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='a/'+member if member in before_map else '/dev/null',tofile='b/'+member))
    (STAGE/'code/source_changes.patch').write_text(''.join(diff),encoding='utf-8')
    analysis=source/'analysis_roundI_results_20261021'
    for file in analysis.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts and file.suffix in ('.json','.csv','.png','.log','.txt'):
            copy(file,STAGE/'analysis/confirmed'/file.relative_to(analysis))
    for folder in ('analysis_roundI_20261021','analysis_roundI_results_20261021'):
        for file in (source/folder).glob('*'):
            if file.suffix in ('.py','.cjs'):copy(file,STAGE/'code/original_analysis'/folder/file.name)
    for file in source.glob('VG-CPOD_v1_I组*.md'):
        copy(file,STAGE/'results/package_support/original_reports'/(file.name+'.txt'))
    inventory=[]
    for case in ('I1','I2'):
        copy(STAGE/('results/raw_export/roundI_20261021/configs/'+case+'.json'),STAGE/('configs/runtime/'+case+'.json'))
        for seed in SEEDS:
            run=STAGE/'results/raw_export/roundI_20261021'/case/'ut-zap50k/V1'/('seed%d'%seed)
            protocol,saved,exact=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'exact_selection/metrics.json')
            inventory.append(dict(case=case,seed=seed,completed_epochs=10,selected_epoch=saved['selected_epoch'],
                exact_selected_epoch=exact['selected_epoch'],path=run.relative_to(STAGE).as_posix(),
                config=protocol['config'],metrics=saved['metrics'],exact_selection_metrics=exact['metrics']))
    write(STAGE/'configs/experiment_inventory.json',inventory)
    write(STAGE/'configs/intervention_inventory.json',[
        dict(case=case,seed=s,selection=selection,path='results/raw_export/roundI_20261021/'+folder+'/'+case+'/seed'+str(s),
             conditions=['dynamic','train_mean'],optimizer_steps=0)
        for case in ('I1','I2') for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')] for s in SEEDS])
    write(STAGE/'external_inputs.json',dict(framework='frozen OpenAI CLIP ViT-L/14',
        cache='/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        feature_manifest_sha256='bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        A_references='../02/results/raw_export/roundA_20261019',C_first_seed='../04/results/raw_export/roundC_20261020',
        C_additional_and_D='../05/results/raw_export',E_references='../06/results/raw_export/roundE_20261020',
        F_references='../07/results/raw_export/roundF_20261020',G_references='../08/results/raw_export/roundG_20261020',H_references='../09/results/raw_export/roundH_20261020',
        cache_metadata='../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14',
        training_statistics='results/raw_export/roundI_20261021/feature_statistics/train_object_query_mean.npz',
        training_inputs='Reuse server feature arrays; shared TRAIN mean is computed once before the three seed jobs.'))
    (STAGE/'.gitignore').write_text('analysis/recomputed/\n',encoding='utf-8')
    create_recompute(source)


def create_recompute(source):
    common="""STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
ARCHIVE=STAGE/'results/raw_export'
I=ARCHIVE/'roundI_20261021'
H=STAGE.parent/'09/results/raw_export/roundH_20261020'
CACHE=STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'
sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundI'))


def source_file(package,member):
    number={'vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06','vgcpod_v1_roundF':'07',
            'vgcpod_v1_roundG':'08','vgcpod_v1_roundH':'09','vgcpod_v1_roundI':'10'}[package]
    folder=STAGE.parent/number
    mapping=json.loads((folder/'package_members.json').read_text(encoding='utf-8'))['members']
    return folder/mapping[member]


def result_base(stage):
    number={'D':'05','E':'06','F':'07','G':'08','H':'09','I':'10'}[stage]
    suffix='_20261021' if stage=='I' else '_20261020'
    return STAGE.parent/number/'results/raw_export'/('round'+stage+suffix)

"""
    original=source/'analysis_roundI_results_20261021'
    text=(original/'analyze_i.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT=Path('),text.index('from ranking_diagnostics')
    text=text[:start]+common+text[end:]
    replacements={
        "ROOT/'analysis_20261018/analyze_results.py'":"STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "ROOT/'vgcpod_roundI_results_20261021.tar.gz'":"STAGE/'results/archives/vgcpod_roundI_results_20261021.tar.gz'",
        "read(source/'release_manifest.json')":"read(source_file(source.name,'release_manifest.json'))",
        "sha(source/name)":"sha(source_file(source.name,name))",
        "base=ARCHIVE/('round'+stage+('_20261021' if stage=='I' else '_20261020'))":"base=result_base(stage)",
        "base=ARCHIVE/('round'+case[0]+('_20261021' if case[0]=='I' else '_20261020'))":"base=result_base(case[0])",
        "sha(source/'mean_gate_eval.py')":"sha(source_file(source.name,'mean_gate_eval.py'))"}
    for old,new in replacements.items():
        assert old in text,old
        text=text.replace(old,new)
    text=text.replace('np.trapezoid(',"getattr(np,'trapezoid',np.trapz)(")
    (STAGE/'analysis/recompute_metrics.py').write_text('# Numerical calculations unchanged; repository paths adapted.\n'+text,encoding='utf-8')
    text=(original/'make_figures.py').read_text(encoding='utf-8').replace('OUT=Path(__file__).resolve().parent',"OUT=Path(__file__).resolve().parent/'recomputed'")
    (STAGE/'analysis/recompute_figures.py').write_text(text,encoding='utf-8')
    entry='"""Recompute archived I results without training."""\nfrom pathlib import Path\nimport runpy\n\nif __name__=="__main__":\n    output=Path(__file__).resolve().parent/"recomputed"\n    output.mkdir(exist_ok=True)\n    for name in ("recompute_metrics.py","recompute_figures.py"):\n        runpy.run_path(str(Path(__file__).resolve().parent/name),run_name="__main__")\n'
    (STAGE/'analysis/recompute.py').write_text(entry,encoding='utf-8')


def finalize():
    write(STAGE/'inputs.json',dict(line='VG-CPOD方案',stage='10',version='v1',title=TITLE,knowledge=[],
        completed_runs=6,model_epochs=60,training_evaluations=24,secondary_selection_evaluations=24,
        intervention_checkpoint_selections=12,intervention_conditions=24,intervention_evaluations=96,
        predictions=144,checkpoint_files=24,completed_cases=['I1','I2'],completed_seeds=list(SEEDS),
        run_matrix=[dict(case=case,seed=seed) for case in ('I1','I2') for seed in SEEDS],package_member_map='package_members.json',
        experiment_inventory='configs/experiment_inventory.json',
        code_files=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        configs=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'configs').rglob('*.json'))],
        documents=[p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='10',primary_package=PACKAGE,
        commands=[dict(source='README.md',language=language,text=text) for language,text in
            re.findall(r'```(bash|powershell)\n(.*?)```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)]))
    files=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in files if p.name!='file_manifest.json'])
    stages=[s for s in read(ROOT/'provenance/stages.json') if s.get('id')!='29']
    stages.append(dict(id='29',line='VG-CPOD方案',number='10',version='v1',title='10 · '+TITLE,
        path='VG-CPOD方案/10',primary_package=PACKAGE,status='I1/I2各三seed、残差幅值与门控温度对照、双准则及十二项均值评估归档',
        completed_runs=6,model_epochs=60,evaluations=144,predictions=144,checkpoints=24))
    write(ROOT/'provenance/stages.json',stages)
    write(ROOT/'provenance/updates/20261021_vgcpod_10.json',dict(date='2026-10-21',line='VG-CPOD方案',number='10',
        version='v1',title=TITLE,imported_from='I release ZIP and downloaded I server export',
        new_runs=[dict(case=case,seed=seed) for case in ('I1','I2') for seed in SEEDS],reference_stages=['02','04','05','06','07','08','09'],
        formula_revision='I1 keeps TRAIN residual magnitude with tau=.1; I2 normalizes residual with tau=.56; centered eta=.25 gate and global CP readout',
        gate_mean_shape=[12,768],gate_mean_trainable=False,planned_only=['J1: separate static/dynamic CP readouts with shared attribute/object factors']))
    owned={p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/10/file_manifest.json','README.md','VG-CPOD方案/README.md','tools/reproduce.py',
        'tools/register_vgcpod_round_i.py','provenance/stages.json','provenance/updates/20261021_vgcpod_10.json'])
    current={r['path']:r for r in read(ROOT/'provenance/current_manifest.json')}
    for relative in sorted(owned):
        file=ROOT/relative;current[relative]=dict(path=relative,sha256=sha(file),bytes=file.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[p] for p in sorted(current)])


def check():
    files=read(STAGE/'file_manifest.json')
    for row in files:assert sha(STAGE/row['path'])==row['sha256'],row['path']
    mapping=package_map(STAGE);release=read(STAGE/mapping['release_manifest.json'])['files_sha256']
    assert set(mapping)==set(release)|{'release_manifest.json'}
    with zipfile.ZipFile(STAGE/'results/archives'/ZIP) as archive:
        for name,target in mapping.items():assert archive.read(PACKAGE+'/'+name)==(STAGE/target).read_bytes(),name
    for name,fingerprint in release.items():assert sha(STAGE/mapping[name])==fingerprint,name
    for name in (ZIP,TAR):
        file=STAGE/'results/archives'/name;assert sha(file)==file.with_name(name+'.sha256').read_text().split()[0]
    raw=STAGE/'results/raw_export/roundI_20261021'
    assert len(list(raw.rglob('predictions/*.npz')))==144
    assert len(list(raw.rglob('checkpoints/*.pt')))==24
    assert (raw/'launcher.exitcode').read_text().strip()=='0'
    statistics=raw/'feature_statistics/train_object_query_mean.npz'
    for case in ('I1','I2'):
        assert read(raw/('configs/'+case+'.json'))['gate_statistics_sha256']==sha(statistics)
        for seed in SEEDS:
            run=raw/case/'ut-zap50k/V1'/('seed%d'%seed)
            protocol,saved,history,exact=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json'),read(run/'exact_selection/metrics.json')
            assert len(history)==10 and history[-1]['step']==3600 and protocol['trainable_parameters']==122880
            assert saved['selected_epoch']==max(range(10),key=lambda i:history[i]['val_cw']['AUC_native'])+1
            assert exact['selected_epoch']==max(range(10),key=lambda i:history[i]['val_cw']['AUC_exact'])+1
            assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')==saved['checkpoint_sha256']
            assert sha(run/'checkpoints/best_exact.pt')==exact['checkpoint_sha256']
            assert protocol['config']['gate_statistics_sha256']==sha(statistics)
            for name,fingerprint in protocol['source_sha256'].items():assert sha(STAGE/mapping[name])==fingerprint,(case,seed,name)
    docs=list(STAGE.rglob('README.md'));assert len(docs)==5
    formulas=0
    for file in docs:
        text=file.read_text(encoding='utf-8');assert text.count('```')%2==0 and '$$' not in text
        formulas+=text.count('```math\n')
        for link in re.findall(r'\]\(([^)\n]+)\)',text):
            link=link.split('#')[0];assert not link or re.match(r'^[a-z]+://',link) or (file.parent/link).exists(),(file,link)
    for file in list((STAGE/'code').rglob('*.py'))+list((STAGE/'analysis').glob('*.py')):ast.parse(file.read_text(encoding='utf-8-sig'))
    return dict(passed=True,files_verified=len(files),trained_runs=6,training_predictions=24,
        secondary_selection_predictions=24,intervention_predictions=96,predictions=144,checkpoint_files=24,
        readmes=5,display_formulas=formulas,training_started=False)


def materialize(destination):
    destination=Path(destination).expanduser()
    if not destination.is_absolute():raise ValueError('Use an absolute destination')
    destination=destination.resolve()
    if destination.exists() or destination==ROOT or ROOT in destination.parents:raise FileExistsError(destination)
    mapping=package_map(STAGE)
    for member,relative in mapping.items():
        target=destination/PACKAGE/member;copy(STAGE/relative,target)
        if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
    for member,fingerprint in read(destination/PACKAGE/'release_manifest.json')['files_sha256'].items():
        assert sha(destination/PACKAGE/member)==fingerprint,member
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='10',package=PACKAGE,files_verified=len(mapping),training_started=False))
    return dict(line='VG-CPOD方案',stage='10',package=str(destination/PACKAGE),files_verified=len(mapping),training_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--finalize',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--materialize',type=Path);args=parser.parse_args()
    if args.import_data and args.source:import_data(args.source);result=dict(imported=True,stage='10')
    elif args.finalize:finalize();result=check()
    elif args.check:result=check()
    elif args.materialize:result=materialize(args.materialize)
    else:parser.error('Choose an action')
    print(json.dumps(result,ensure_ascii=False))
