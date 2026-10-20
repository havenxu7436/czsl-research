"""Archive experiment 08: intermediate support weight and matched gate training."""
import argparse
import ast
import difflib
import json
from pathlib import Path,PurePosixPath
import re
import zipfile

from register_vgcpod_round_f import read,write,sha,copy,package_map

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'VG-CPOD方案/08'
TITLE='支持集权重折中与门控训练对照'
PACKAGE='vgcpod_v1_roundG'
ZIP='vgcpod_v1_roundG_20261020.zip'
TAR='vgcpod_roundG_results_20261020.tar.gz'
SEEDS=(3351,3352,3353)
CASES=('G0','G1')
SOURCE_STAGES={'vgcpod_v1_roundC':'04','vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06',
               'vgcpod_v1_roundF':'07','vgcpod_v1_roundG':'08'}


def import_data(source):
    source=Path(source).resolve()
    if STAGE.exists():raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True);mapping={}
    with zipfile.ZipFile(source/ZIP) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('/'):continue
            path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name
            assert path.parts[0]==PACKAGE
            member=PurePosixPath(*path.parts[1:]).as_posix()
            if member.startswith('configs/'):target=member
            elif member.startswith(('assets/','references/')):target='configs/'+member
            elif member.startswith('validation/') or member=='release_manifest.json':target='results/package_validation/G/'+member
            elif Path(member).suffix in ('.md','.html'):target='results/package_support/G/'+member+'.txt'
            else:target='code/'+PACKAGE+'/'+member
            file=STAGE/target;file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(archive.read(name));mapping[member]=target
    write(STAGE/'package_members.json',dict(package=PACKAGE,members=mapping))
    for name in (ZIP,TAR):
        file=STAGE/'results/archives'/name;copy(source/name,file)
        file.with_name(name+'.sha256').write_text(sha(file)+'  '+name+'\n',encoding='ascii')
    export=source/'server_results_roundG_20261020';raw=[]
    for file in sorted((export/'roundG_20261020').rglob('*')):
        if not file.is_file() or file.name.endswith('.lock') or '__pycache__' in file.parts:continue
        target=STAGE/'results/raw_export'/file.relative_to(export);copy(file,target)
        raw.append(dict(path=target.relative_to(STAGE).as_posix(),sha256=sha(target),bytes=target.stat().st_size))
    write(STAGE/'results/raw_export_members.json',raw)
    sources,reused=[],[]
    for package,number in SOURCE_STAGES.items():
        stage=STAGE.parent/number;members=package_map(stage)
        for member,target in members.items():
            server=export/package/member
            if server.is_file():
                assert server.read_bytes()==(stage/target).read_bytes(),(package,member)
                sources.append(dict(package=package,member=member,stage=number,sha256=sha(server)))
    for group in ('roundC_20261020','roundD_20261020','roundE_20261020','roundF_20261020'):
        for file in sorted((export/group).rglob('*')):
            if not file.is_file() or file.name.endswith('.lock') or '__pycache__' in file.parts:continue
            rel=file.relative_to(export);fingerprint=sha(file)
            candidates=[STAGE.parent/n/'results/raw_export'/rel for n in ('04','05','06','07')]
            match=next((p for p in candidates if p.is_file() and sha(p)==fingerprint),None)
            if match:
                target='../'+match.relative_to(STAGE.parent).as_posix()
            else:
                destination=STAGE/'results/reference_exports'/rel;copy(file,destination)
                target=destination.relative_to(STAGE).as_posix()
            reused.append(dict(member=rel.as_posix(),reference=target,sha256=fingerprint))
    before=STAGE.parent/'07';before_map=package_map(before)
    unchanged=[]
    for member,target in mapping.items():
        if member.startswith('vgcpod/') or member in ('run.py','mean_gate_eval.py'):
            assert (STAGE/target).read_bytes()==(before/before_map[member]).read_bytes(),member
            unchanged.append(member)
    write(STAGE/'results/server_source_identity.json',dict(source_bytes_preserved=True,
        runtime_identical_to_F=True,runtime_files=unchanged,members=sources))
    write(STAGE/'results/reused_reference_members.json',reused)
    diff=[]
    for member,target in sorted(mapping.items()):
        if not (member.endswith(('.py','.sh')) or member.startswith('configs/round_g/') or member=='references/pre_g_complete.json'):continue
        old=(before/before_map[member]).read_text(encoding='utf-8') if member in before_map else ''
        new=(STAGE/target).read_text(encoding='utf-8')
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='a/'+member if member in before_map else '/dev/null',tofile='b/'+member))
    (STAGE/'code/source_changes.patch').write_text(''.join(diff),encoding='utf-8')
    analysis=source/'analysis_roundG_results_20261020'
    for file in analysis.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts and file.suffix in ('.json','.csv','.png','.log','.txt'):
            copy(file,STAGE/'analysis/confirmed'/file.relative_to(analysis))
    for folder in ('analysis_roundG_20261020','analysis_roundG_results_20261020'):
        for file in (source/folder).glob('*'):
            if file.suffix in ('.py','.cjs'):copy(file,STAGE/'code/original_analysis'/folder/file.name)
    for file in source.glob('VG-CPOD_v1_G组*.md'):
        copy(file,STAGE/'results/package_support/original_reports'/(file.name+'.txt'))
    inventory=[]
    for case in CASES:
        for seed in SEEDS:
            run=STAGE/'results/raw_export/roundG_20261020'/case/'ut-zap50k/V1'/('seed%d'%seed)
            protocol,saved,exact=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'exact_selection/metrics.json')
            inventory.append(dict(case=case,seed=seed,completed_epochs=10,selected_epoch=saved['selected_epoch'],
                exact_selected_epoch=exact['selected_epoch'],path=run.relative_to(STAGE).as_posix(),
                config=protocol['config'],metrics=saved['metrics'],exact_selection_metrics=exact['metrics']))
    write(STAGE/'configs/experiment_inventory.json',inventory)
    write(STAGE/'configs/intervention_inventory.json',[
        dict(case='G1',seed=s,selection=selection,path='results/raw_export/roundG_20261020/'+folder+'/G1/seed'+str(s),
             conditions=['dynamic','train_mean'],optimizer_steps=0)
        for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')] for s in SEEDS])
    write(STAGE/'external_inputs.json',dict(framework='frozen OpenAI CLIP ViT-L/14',
        cache='/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        feature_manifest_sha256='bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        A_references='../02/results/raw_export/roundA_20261019',C_first_seed='../04/results/raw_export/roundC_20261020',
        C_additional_and_D='../05/results/raw_export',E_references='../06/results/raw_export/roundE_20261020',
        F_references='../07/results/raw_export/roundF_20261020',
        cache_metadata='../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14',
        training_inputs='Reuse server feature arrays; repository stores cache metadata and text features.'))
    (STAGE/'.gitignore').write_text('analysis/recomputed/\n',encoding='utf-8')
    create_recompute(source)


def create_recompute(source):
    common="""STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent/'recomputed'
ARCHIVE = STAGE/'results/raw_export'
A_OUT = STAGE.parent/'02/results/raw_export/roundA_20261019'
C_OUT = STAGE.parent/'05/results/raw_export/roundC_20261020'
D_OUT = STAGE.parent/'05/results/raw_export/roundD_20261020'
E_OUT = STAGE.parent/'06/results/raw_export/roundE_20261020'
F_OUT = STAGE.parent/'07/results/raw_export/roundF_20261020'
G_OUT = ARCHIVE/'roundG_20261020'
CACHE = STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'


def source_file(package,member):
    number = {'vgcpod_v1_roundC':'04','vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06','vgcpod_v1_roundF':'07','vgcpod_v1_roundG':'08'}[package]
    stage = STAGE.parent/number
    mapping = json.loads((stage/'package_members.json').read_text(encoding='utf-8'))['members']
    return stage/mapping[member]


def reference_base(case,seed):
    if case.startswith('A'):return A_OUT
    if case.startswith('C'):
        return STAGE.parent/('04' if seed==3351 else '05')/'results/raw_export/roundC_20261020'
    return {'D':D_OUT,'E':E_OUT,'F':F_OUT,'G':G_OUT}[case[0]]

"""
    original=source/'analysis_roundG_results_20261020'
    text=(original/'analyze_g.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT = Path('),text.index('SETTINGS =')
    text=text[:start]+common+text[end:]
    start,end=text.index('    base = A_OUT'),text.index('    return next(',text.index('def folder'))
    text=text[:start]+"    base = reference_base(case,seed)\n"+text[end:]
    replacements={"ROOT/'analysis_20261018/analyze_results.py'":"STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "ARCHIVE/'vgcpod_v1_roundD/vgcpod/metrics.py'":"source_file('vgcpod_v1_roundD','vgcpod/metrics.py')",
        "ROOT/'vgcpod_roundG_results_20261020.tar.gz'":"STAGE/'results/archives/vgcpod_roundG_results_20261020.tar.gz'",
        "read(source/'configs'/sub/(case+'.json'))":"read(source_file(source.name,'configs/'+sub+'/'+case+'.json'))",
        "helper.sha(source/name)==fingerprint==helper.sha(ROOT/source.name/name)":"helper.sha(source_file(source.name,name))==fingerprint",
        "helper.sha(ARCHIVE/source/'mean_gate_eval.py')":"helper.sha(source_file(source,'mean_gate_eval.py'))"}
    for old,new in replacements.items():assert old in text,old;text=text.replace(old,new)
    (STAGE/'analysis/recompute_metrics.py').write_text('# Numerical calculations unchanged; repository paths adapted.\n'+text,encoding='utf-8')
    text=(original/'supplementary_analysis.py').read_text(encoding='utf-8')
    start,end=text.index('ROOT=Path('),text.index('from ranking_diagnostics')
    text=text[:start]+common+"sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundG'))\n"+text[end:]
    text=text.replace("ROOT/'analysis_20261018/analyze_results.py'","STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'")
    start,end=text.index("        if case.startswith('A')"),text.index("        if condition=='train_mean'")
    text=text[:start]+"        base=reference_base(case,seed)\n"+text[end:]
    text=text.replace("read(ARCHIVE/'vgcpod_v1_roundG/release_manifest.json')","read(source_file('vgcpod_v1_roundG','release_manifest.json'))")
    text=text.replace("(ARCHIVE/'vgcpod_v1_roundG'/name).read_bytes()","source_file('vgcpod_v1_roundG',name).read_bytes()")
    (STAGE/'analysis/recompute_supplementary.py').write_text('# Numerical calculations unchanged; repository paths adapted.\n'+text,encoding='utf-8')
    text=(original/'make_figures.py').read_text(encoding='utf-8')
    text=text.replace('OUT=Path(__file__).resolve().parent',"OUT=Path(__file__).resolve().parent/'recomputed'")
    (STAGE/'analysis/recompute_figures.py').write_text(text,encoding='utf-8')
    (STAGE/'analysis/recompute.py').write_text('''"""Recompute the G archive metrics, static-gate equivalence and figures."""
from pathlib import Path
import runpy
import shutil

if __name__=='__main__':
    for name in ('recompute_metrics.py','recompute_supplementary.py','recompute_figures.py'):
        runpy.run_path(str(Path(__file__).resolve().parent/name),run_name='__main__')
    stage=Path(__file__).resolve().parents[1]
    for name in ('ranking_diagnostics.csv','ranking_diagnostics_mean_sd.csv',
                 'ranking_diagnostic_deltas.csv','ranking_diagnostic_paired_mean_sd.csv'):
        shutil.copy2(stage/'results/raw_export/roundG_20261020'/name,
                     stage/'analysis/recomputed/tables'/name)
''',encoding='utf-8')


def finalize():
    write(STAGE/'inputs.json',dict(line='VG-CPOD方案',stage='08',version='v1',title=TITLE,knowledge=[],
        completed_runs=6,model_epochs=60,training_evaluations=24,secondary_selection_evaluations=24,
        intervention_checkpoint_selections=6,intervention_conditions=12,intervention_evaluations=48,
        predictions=96,checkpoint_files=24,completed_cases=list(CASES),completed_seeds=list(SEEDS),
        run_matrix=[dict(case=case,seed=seed) for case in CASES for seed in SEEDS],
        package_member_map='package_members.json',experiment_inventory='configs/experiment_inventory.json',
        code_files=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        configs=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'configs').rglob('*.json'))],
        documents=[p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='08',primary_package=PACKAGE,
        commands=[dict(source='README.md',language=language,text=text) for language,text in
            re.findall(r'```(bash|powershell)\n(.*?)```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)]))
    files=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in files if p.name!='file_manifest.json'])
    stages=[s for s in read(ROOT/'provenance/stages.json') if s.get('id')!='27']
    stages.append(dict(id='27',line='VG-CPOD方案',number='08',version='v1',title='08 · '+TITLE,
        path='VG-CPOD方案/08',primary_package=PACKAGE,status='G0/G1六次训练、beta中间值与门控模块对照、双准则及六项均值评估归档',
        completed_runs=6,model_epochs=60,evaluations=96,predictions=96,checkpoints=24))
    write(ROOT/'provenance/stages.json',stages)
    write(ROOT/'provenance/updates/20261020_vgcpod_08.json',dict(date='2026-10-20',line='VG-CPOD方案',number='08',
        version='v1',title=TITLE,imported_from='G release ZIP and downloaded G server export',
        new_runs=[dict(case=case,seed=seed) for case in CASES for seed in SEEDS],reference_stages=['02','04','05','06','07'],
        formula_revision='beta=.075 gives pair=3/2 and support=9/86; G0 gamma=1, G1 mode-centered eta=.25; separate native/exact selection',
        runtime_unchanged_from_F=True,planned_only=['H1: TRAIN-mean-centered local visual residual gate; three seeds']))
    owned={p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/08/file_manifest.json','README.md','VG-CPOD方案/README.md','tools/reproduce.py',
        'tools/register_vgcpod_round_g.py','provenance/stages.json','provenance/updates/20261020_vgcpod_08.json'])
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
    raw=STAGE/'results/raw_export/roundG_20261020'
    assert len(list(raw.rglob('predictions/*.npz')))==96
    assert len(list(raw.rglob('checkpoints/*.pt')))==24
    assert (raw/'launcher.exitcode').read_text().strip()=='0'
    for case in CASES:
        for seed in SEEDS:
            run=raw/case/'ut-zap50k/V1'/('seed%d'%seed)
            protocol,saved,history,exact=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json'),read(run/'exact_selection/metrics.json')
            assert len(history)==10 and history[-1]['step']==3600
            assert saved['selected_epoch']==max(range(10),key=lambda i:history[i]['val_cw']['AUC_native'])+1
            assert exact['selected_epoch']==max(range(10),key=lambda i:history[i]['val_cw']['AUC_exact'])+1
            assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')==saved['checkpoint_sha256']
            assert sha(run/'checkpoints/best_exact.pt')==exact['checkpoint_sha256']
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
        secondary_selection_predictions=24,intervention_predictions=48,predictions=96,checkpoint_files=24,
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
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='08',package=PACKAGE,files_verified=len(mapping),training_started=False))
    return dict(line='VG-CPOD方案',stage='08',package=str(destination/PACKAGE),files_verified=len(mapping),training_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--finalize',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--materialize',type=Path);args=parser.parse_args()
    if args.import_data and args.source:import_data(args.source);result=dict(imported=True,stage='08')
    elif args.finalize:finalize();result=check()
    elif args.check:result=check()
    elif args.materialize:result=materialize(args.materialize)
    else:parser.error('Choose an action')
    print(json.dumps(result,ensure_ascii=False))
