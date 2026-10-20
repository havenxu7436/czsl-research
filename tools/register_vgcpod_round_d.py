"""Archive experiment 05: centered CP gain coupling and TRAIN-mean gate evaluation."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'VG-CPOD方案/05'
C_STAGE = ROOT/'VG-CPOD方案/04'
TITLE = '中心化 CP 幅度联动与均值门控评估'
PACKAGES = {'vgcpod_v1_roundD':'vgcpod_v1_roundD_20261020.zip',
            'vgcpod_next_v1_20261020':'vgcpod_next_v1_20261020.zip'}
ARCHIVES = list(PACKAGES.values())+['vgcpod_roundD_results_20261020.tar.gz','vgcpod_next_results_20261020.tar.gz']
RUNS = [('C1',3352),('C2',3352),('C2',3353),('D1',3351),('D1',3352),('D1',3353)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def member_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name:
        raise ValueError('Unsafe member: '+name)
    return p


def import_data(source):
    source = Path(source).resolve()
    if STAGE.exists():
        raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    maps = {}
    for package,archive_name in PACKAGES.items():
        mapping = {}
        kind = 'D' if package=='vgcpod_v1_roundD' else 'next'
        with zipfile.ZipFile(source/archive_name) as z:
            for item in sorted(z.namelist()):
                if item.endswith('/'):
                    continue
                parts = member_path(item).parts
                assert parts[0]==package
                name = PurePosixPath(*parts[1:]).as_posix()
                if name.startswith('configs/'):
                    target = name
                elif name.startswith(('assets/','references/')):
                    target = 'configs/'+name
                elif name.startswith('validation/') or name.startswith('validation') or name=='release_manifest.json':
                    target = 'results/package_validation/'+kind+'/'+name
                elif Path(name).suffix in ('.md','.html'):
                    target = 'results/package_support/'+kind+'/'+name+'.txt'
                else:
                    target = 'code/'+package+'/'+name
                path = STAGE/target;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(z.read(item));mapping[name] = target
        maps[package] = mapping
        write(STAGE/('package_members.json' if kind=='D' else 'next_package_members.json'),
              dict(package=package,members=mapping))
    for name in ARCHIVES:
        target = STAGE/'results/archives'/name
        copy(source/name,target)
        target.with_name(name+'.sha256').write_text(sha(target)+'  '+name+'\n',encoding='ascii')
    c_map = read(C_STAGE/'package_members.json')['members']
    raw_members,source_checks,reused = [],[],[]
    with tarfile.open(source/'vgcpod_next_results_20261020.tar.gz','r:gz') as archive:
        for item in archive:
            if not item.isfile():
                continue
            parts = member_path(item.name).parts
            name = PurePosixPath(*parts[1:]).as_posix()
            if parts[0] in maps or parts[0]=='vgcpod_v1_roundC':
                stage,mapping = (C_STAGE,c_map) if parts[0]=='vgcpod_v1_roundC' else (STAGE,maps[parts[0]])
                if name in mapping:
                    assert archive.extractfile(item).read()==(stage/mapping[name]).read_bytes(),item.name
                    source_checks.append(dict(package=parts[0],member=name,sha256=sha(stage/mapping[name])))
                continue
            if parts[0] not in ('roundC_20261020','roundD_20261020','next_20261020') or parts[-1].endswith('.lock'):
                continue
            if parts[0]=='roundC_20261020':
                if 'seed3351' in parts and parts[1] in ('C1','C2'):
                    old = C_STAGE/'results/raw_export'/PurePosixPath(*parts)
                    payload = archive.extractfile(item).read()
                    assert payload==old.read_bytes(),item.name
                    reused.append(dict(member=item.name,reference='../04/'+old.relative_to(C_STAGE).as_posix(),sha256=sha(old)))
                    continue
                if parts[1:2]==('logs',) and parts[-1] not in ('C1_seed3352.log','C2_seed3352.log','C2_seed3353.log'):
                    continue
            payload = archive.extractfile(item).read()
            target = STAGE/'results/raw_export'/PurePosixPath(*parts)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
            raw_members.append(dict(path=target.relative_to(STAGE).as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
    write(STAGE/'results/raw_export_members.json',raw_members)
    write(STAGE/'results/reused_stage04_members.json',reused)
    write(STAGE/'results/server_source_identity.json',dict(source_bytes_preserved=True,
        D_training_core_identical_to_C=True,members=source_checks))
    for name in ('run.py',)+tuple('vgcpod/'+p.name for p in (source/'vgcpod_v1_roundD/vgcpod').glob('*.py')):
        assert (STAGE/maps['vgcpod_v1_roundD'][name]).read_bytes()==(C_STAGE/c_map[name]).read_bytes(),name
    for src,sub in [('analysis_roundD_results_20261020','pilot'),('analysis_next_results_20261020','confirmed')]:
        for path in (source/src).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.json','.csv','.png','.log','.txt'):
                copy(path,STAGE/'analysis'/sub/path.relative_to(source/src))
    for src in ('analysis_roundD_results_20261020','analysis_next_results_20261020','analysis_roundD_20261020','analysis_next_20261020'):
        for path in (source/src).glob('*'):
            if path.is_file() and path.suffix in ('.py','.cjs'):
                copy(path,STAGE/'code/original_analysis'/src/path.name)
    for pattern in ('VG-CPOD_v1_D*.md','VG-CPOD_v1_下一轮三卡*.md'):
        for path in source.glob(pattern):
            copy(path,STAGE/'results/package_support/original_reports'/(path.name+'.txt'))
    diff = []
    for name,target in sorted(maps['vgcpod_v1_roundD'].items()):
        if not (name.endswith(('.py','.sh')) or name=='configs/round_d/D1.json'):
            continue
        before = (C_STAGE/c_map[name]).read_text(encoding='utf-8') if name in c_map else ''
        after = (STAGE/target).read_text(encoding='utf-8')
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile='a/'+name if name in c_map else '/dev/null',tofile='b/'+name))
    for name,target in maps['vgcpod_next_v1_20261020'].items():
        if name.endswith(('.py','.sh')):
            diff.extend(difflib.unified_diff([], (STAGE/target).read_text(encoding='utf-8').splitlines(True),
                fromfile='/dev/null',tofile='b/next/'+name))
    (STAGE/'code/source_changes.patch').write_text(''.join(diff),encoding='utf-8')
    inventory = []
    for case,seed in RUNS:
        group = 'roundD_20261020' if case=='D1' else 'roundC_20261020'
        run = STAGE/'results/raw_export'/group/case/'ut-zap50k/V1'/('seed%d'%seed)
        p,m = read(run/'protocol.json'),read(run/'metrics.json')
        inventory.append(dict(case=case,seed=seed,completed_epochs=10,selected_epoch=m['selected_epoch'],
            path=run.relative_to(STAGE).as_posix(),runtime_source_stage='05' if case=='D1' else '04',
            config=p['config'],metrics=m['metrics']))
    write(STAGE/'configs/experiment_inventory.json',inventory)
    write(STAGE/'configs/intervention_inventory.json',[
        dict(case=c,seed=s,path='results/raw_export/roundD_20261020/mean_gate/'+c+'/seed'+str(s),
             conditions=['dynamic','train_mean'],optimizer_steps=0)
        for c in ('C2','D1') for s in (3351,3352,3353)])
    write(STAGE/'external_inputs.json',dict(framework='frozen CLIP ViT-L/14',
        cache='/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        feature_manifest_sha256='bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        A_references='../02/results/raw_export/roundA_20261019',
        C_seed3351_references='../04/results/raw_export/roundC_20261020',
        C_runtime_source='../04/package_members.json',
        cache_metadata='../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'))
    (STAGE/'.gitignore').write_text('analysis/recomputed/\n',encoding='utf-8')
    create_recompute(source)


def create_recompute(source):
    text = (source/'analysis_next_results_20261020/analyze_next.py').read_text(encoding='utf-8')
    start,end = text.index('ROOT = Path('),text.index('SETTINGS =')
    text = text[:start]+'''STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent/'recomputed'
ARCHIVE = STAGE/'results/raw_export'
A_OUT = STAGE.parent/'02/results/raw_export/roundA_20261019'
C_OUT = ARCHIVE/'roundC_20261020'
D_OUT = ARCHIVE/'roundD_20261020'
C_STAGE = STAGE.parent/'04'
D_MAP = json.loads((STAGE/'package_members.json').read_text(encoding='utf-8'))['members']
C_MAP = json.loads((C_STAGE/'package_members.json').read_text(encoding='utf-8'))['members']
CACHE = STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'
'''+text[end:]
    replacements = {
        "base = A_OUT if case.startswith('A') else C_OUT if case.startswith('C') else D_OUT":
            "base = A_OUT if case.startswith('A') else (C_STAGE/'results/raw_export/roundC_20261020' if case.startswith('C') and seed==3351 else C_OUT) if case.startswith('C') else D_OUT",
        "ROOT/'analysis_20261018/analyze_results.py'":"STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "ARCHIVE/'vgcpod_v1_roundD/vgcpod/metrics.py'":"STAGE/D_MAP['vgcpod/metrics.py']",
        "ROOT/'vgcpod_next_results_20261020.tar.gz'":"STAGE/'results/archives/vgcpod_next_results_20261020.tar.gz'",
        "source = ARCHIVE/('vgcpod_v1_roundC' if case.startswith('C') else 'vgcpod_v1_roundD')":
            "source = C_STAGE if case.startswith('C') else STAGE\n            source_map = C_MAP if case.startswith('C') else D_MAP",
        "assert helper.sha(source/name)==fingerprint==helper.sha(ROOT/source.name/name)":
            "assert helper.sha(source/source_map[name])==fingerprint",
        "ARCHIVE/'vgcpod_v1_roundD/mean_gate_eval.py'":"STAGE/D_MAP['mean_gate_eval.py']"}
    for before,after in replacements.items():
        assert before in text,before;text = text.replace(before,after)
    (STAGE/'analysis/recompute.py').write_text('# Numerical calculations unchanged; repository paths adapted.\n'+text,encoding='utf-8')
    bias = (source/'analysis_next_results_20261020/analyze_bias.py').read_text(encoding='utf-8')
    bias = bias.replace("ROOT = Path(__file__).resolve().parents[1]", "STAGE = Path(__file__).resolve().parents[1]\nROOT = STAGE.parent.parent")
    bias = bias.replace("OUT = Path(__file__).resolve().parent", "OUT = Path(__file__).resolve().parent/'recomputed'")
    bias = bias.replace("NEW = ROOT/'server_results_next_20261020'", "NEW = STAGE/'results/raw_export'")
    bias = bias.replace("ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json'", "STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json'")
    bias = bias.replace("ROOT/'server_results_roundA_confirm_20261019/roundA_20261019/A5/ut-zap50k/V1'", "STAGE.parent/'02/results/raw_export/roundA_20261019/A5/ut-zap50k/V1'")
    bias = bias.replace("metrics = read(run/'metrics.json')['metrics']", "if case=='C2' and seed==3351:\n                run = STAGE.parent/'04/results/raw_export/roundC_20261020/C2/ut-zap50k/V1/seed3351'\n            metrics = read(run/'metrics.json')['metrics']")
    (STAGE/'analysis/recompute_bias.py').write_text(bias,encoding='utf-8')


def finalize():
    write(STAGE/'inputs.json',dict(line='VG-CPOD方案',stage='05',version='v1',title=TITLE,knowledge=[],
        completed_runs=6,new_D1_runs=3,companion_C_runs=3,model_epochs=60,
        training_evaluations=24,intervention_checkpoints=6,intervention_conditions=12,
        intervention_evaluations=48,predictions=72,checkpoints=12,
        completed_cases=['C1','C2','D1'],completed_seeds=[3351,3352,3353],
        run_matrix=[dict(case=c,seed=s) for c,s in RUNS],package_member_map='package_members.json',
        next_package_member_map='next_package_members.json',experiment_inventory='configs/experiment_inventory.json',
        code_files=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        configs=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'configs').rglob('*.json'))],
        documents=[p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='05',primary_package='vgcpod_v1_roundD',
        companion_source_stage='04',next_package='vgcpod_next_v1_20261020',
        commands=[dict(source='README.md',language=lang,text=text) for lang,text in
                  re.findall(r'```(bash|powershell)\n(.*?)```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)]))
    files = sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in files if p.name!='file_manifest.json'])
    stages = [r for r in read(ROOT/'provenance/stages.json') if r.get('id')!='24']
    stages.append(dict(id='24',line='VG-CPOD方案',number='05',version='v1',title='05 · '+TITLE,
        path='VG-CPOD方案/05',primary_package='vgcpod_v1_roundD',
        status='6次新增训练、6个checkpoint均值替换、D1三seed及两seed四格交互归档',
        completed_runs=6,model_epochs=60,evaluations=72,predictions=72,checkpoints=12))
    write(ROOT/'provenance/stages.json',stages)
    update = dict(date='2026-10-20',line='VG-CPOD方案',number='05',version='v1',title=TITLE,
        imported_from='VG-CPOD workspace D and completion-round server exports',
        new_runs=[dict(case=c,seed=s) for c,s in RUNS],reused_stage='04',
        core_source='D vgcpod/*.py and run.py byte-identical to C',
        planned_only=['E1 rho=0.15','best_native.pt and best_exact.pt dual checkpoint saving'])
    write(ROOT/'provenance/updates/20261020_vgcpod_05.json',update)
    owned = {p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/05/file_manifest.json','README.md','VG-CPOD方案/README.md','tools/reproduce.py',
        'tools/register_vgcpod_round_d.py','provenance/stages.json','provenance/updates/20261020_vgcpod_05.json'])
    current = {r['path']:r for r in read(ROOT/'provenance/current_manifest.json')}
    for rel in sorted(owned):
        p = ROOT/rel;current[rel] = dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[p] for p in sorted(current)])


def check():
    entries = read(STAGE/'file_manifest.json')
    for row in entries:
        assert sha(STAGE/row['path'])==row['sha256'],row['path']
    for package,archive in PACKAGES.items():
        mapping = read(STAGE/('package_members.json' if package=='vgcpod_v1_roundD' else 'next_package_members.json'))['members']
        release = read(STAGE/mapping['release_manifest.json'])['files_sha256']
        assert set(mapping)==set(release)|{'release_manifest.json'}
        with zipfile.ZipFile(STAGE/'results/archives'/archive) as z:
            for name,target in mapping.items():
                assert z.read(package+'/'+name)==(STAGE/target).read_bytes(),name
        for name,digest in release.items():
            assert sha(STAGE/mapping[name])==digest,name
    for name in ARCHIVES:
        p = STAGE/'results/archives'/name
        assert sha(p)==p.with_name(name+'.sha256').read_text().split()[0]
    for row in read(STAGE/'results/raw_export_members.json'):
        assert sha(STAGE/row['path'])==row['sha256'],row['path']
    runs = list((STAGE/'results/raw_export').glob('*/?*/ut-zap50k/V*/seed*/metrics.json'))
    assert len(runs)==6
    assert len(list((STAGE/'results/raw_export').rglob('predictions/*.npz')))==72
    assert len(list((STAGE/'results/raw_export').rglob('checkpoints/*.pt')))==12
    assert (STAGE/'results/raw_export/next_20261020/launcher.exitcode').read_text().strip()=='0'
    dmap,cmap = read(STAGE/'package_members.json')['members'],read(C_STAGE/'package_members.json')['members']
    for path in runs:
        p,m,h = read(path.parent/'protocol.json'),read(path),read(path.parent/'history.json')
        case = p['config']['experiment_id'];source,mapping = (STAGE,dmap) if case=='D1' else (C_STAGE,cmap)
        assert len(h)==10 and h[-1]['step']==3600
        assert m['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_native'])+1
        assert sha(path.parent/'checkpoints/best.pt')==m['checkpoint_sha256']
        for name,digest in p['source_sha256'].items():
            assert sha(source/mapping[name])==digest,(case,name)
    docs = list(STAGE.rglob('README.md'));assert len(docs)==5
    for path in docs:
        text = path.read_text(encoding='utf-8');assert text.count('```')%2==0
        for link in re.findall(r'\]\(([^)\n]+)\)',text):
            link = link.split('#')[0]
            assert not link or re.match(r'^[a-z]+://',link) or (path.parent/link).exists(),(path,link)
    for path in list((STAGE/'code').rglob('*.py'))+list((STAGE/'analysis').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    return dict(passed=True,files_verified=len(entries),trained_runs=6,training_predictions=24,
        intervention_predictions=48,predictions=72,checkpoints=12,readmes=5)


def materialize(destination):
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Use an absolute destination')
    destination = destination.resolve()
    if destination.exists() or destination==ROOT or ROOT in destination.parents:
        raise FileExistsError(destination)
    total = 0
    for name in ('package_members.json','next_package_members.json'):
        data = read(STAGE/name);package,mapping = data['package'],data['members']
        for member,rel in mapping.items():
            target = destination/package/member;copy(STAGE/rel,target)
            if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
        for member,digest in read(destination/package/'release_manifest.json')['files_sha256'].items():
            assert sha(destination/package/member)==digest
        total += len(mapping)
    # Restore only the required C package; its B3 job is disabled in D queues.
    c = read(C_STAGE/'package_members.json')
    for member,rel in c['members'].items():
        target = destination/c['package']/member;copy(C_STAGE/rel,target)
        if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='05',
        packages=['vgcpod_v1_roundD','vgcpod_next_v1_20261020','vgcpod_v1_roundC'],
        primary_and_next_files=total,C_files=len(c['members']),training_started=False))
    return dict(line='VG-CPOD方案',stage='05',package=str(destination/'vgcpod_v1_roundD'),
        next_package=str(destination/'vgcpod_next_v1_20261020'),C_source=str(destination/'vgcpod_v1_roundC'),
        files_verified=total+len(c['members']),training_started=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--finalize',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--materialize',type=Path);args = parser.parse_args()
    if args.import_data and args.source:
        import_data(args.source);result = dict(imported=True,stage='05')
    elif args.finalize:
        finalize();result = check()
    elif args.check:
        result = check()
    elif args.materialize:
        result = materialize(args.materialize)
    else:
        parser.error('Choose an action')
    print(json.dumps(result,ensure_ascii=False))
