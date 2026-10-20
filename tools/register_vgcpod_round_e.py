"""Archive experiment 06: centered CP strength and dual checkpoint selection."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'VG-CPOD方案/06'
TITLE = '中心化 CP 强度扩展与双准则选权重'
PACKAGE = 'vgcpod_v1_roundE'
ZIP = 'vgcpod_v1_roundE_20261020.zip'
TAR = 'vgcpod_roundE_results_20261020.tar.gz'
SEEDS = (3351,3352,3353)


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


def package_map(stage):
    return read(stage/'package_members.json')['members']


def import_data(source):
    source = Path(source).resolve()
    if STAGE.exists():
        raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    mapping = {}
    with zipfile.ZipFile(source/ZIP) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('/'):
                continue
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name
            assert path.parts[0]==PACKAGE
            member = PurePosixPath(*path.parts[1:]).as_posix()
            if member.startswith('configs/'):
                target = member
            elif member.startswith(('assets/','references/')):
                target = 'configs/'+member
            elif member.startswith('validation/') or member=='release_manifest.json':
                target = 'results/package_validation/E/'+member
            elif Path(member).suffix in ('.md','.html'):
                target = 'results/package_support/E/'+member+'.txt'
            else:
                target = 'code/'+PACKAGE+'/'+member
            p = STAGE/target;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(archive.read(name));mapping[member] = target
    write(STAGE/'package_members.json',dict(package=PACKAGE,members=mapping))
    for name in (ZIP,TAR):
        p = STAGE/'results/archives'/name;copy(source/name,p)
        p.with_name(name+'.sha256').write_text(sha(p)+'  '+name+'\n',encoding='ascii')
    export = source/'server_results_roundE_20261020'
    raw = []
    for p in sorted((export/'roundE_20261020').rglob('*')):
        if not p.is_file() or p.name.endswith('.lock') or '__pycache__' in p.parts:
            continue
        target = STAGE/'results/raw_export'/p.relative_to(export)
        copy(p,target)
        raw.append(dict(path=target.relative_to(STAGE).as_posix(),sha256=sha(target),bytes=target.stat().st_size))
    write(STAGE/'results/raw_export_members.json',raw)
    sources,reused = [],[]
    for package,number in [('vgcpod_v1_roundC','04'),('vgcpod_v1_roundD','05'),(PACKAGE,'06')]:
        stage = STAGE.parent/number;members = package_map(stage)
        for member,target in members.items():
            server = export/package/member
            if server.is_file():
                assert server.read_bytes()==(stage/target).read_bytes(),(package,member)
                sources.append(dict(package=package,member=member,stage=number,sha256=sha(server)))
    for group in ('roundC_20261020','roundD_20261020'):
        for p in sorted((export/group).rglob('*')):
            if not p.is_file() or p.name.endswith('.lock') or '__pycache__' in p.parts:
                continue
            rel = p.relative_to(export)
            candidates = [STAGE.parent/n/'results/raw_export'/rel for n in ('04','05')]
            matches = [q for q in candidates if q.is_file() and sha(q)==sha(p)]
            if matches:
                reused.append(dict(member=rel.as_posix(),reference='../'+matches[0].relative_to(STAGE.parent).as_posix(),sha256=sha(p)))
            else:
                # Later aggregate files can differ; retain their exact downloaded bytes.
                target = STAGE/'results/reference_exports'/rel;copy(p,target)
                reused.append(dict(member=rel.as_posix(),reference=target.relative_to(STAGE).as_posix(),sha256=sha(p)))
    write(STAGE/'results/server_source_identity.json',dict(source_bytes_preserved=True,
        model_py_identical_to_D=True,members=sources))
    write(STAGE/'results/reused_reference_members.json',reused)
    d = STAGE.parent/'05';dmap = package_map(d)
    assert (STAGE/mapping['vgcpod/model.py']).read_bytes()==(d/dmap['vgcpod/model.py']).read_bytes()
    diff = []
    for member,target in sorted(mapping.items()):
        if not (member.endswith(('.py','.sh')) or member=='configs/round_e/E1.json' or member=='references/pre_e_complete.json'):
            continue
        before = (d/dmap[member]).read_text(encoding='utf-8') if member in dmap else ''
        after = (STAGE/target).read_text(encoding='utf-8')
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile='a/'+member if member in dmap else '/dev/null',tofile='b/'+member))
    (STAGE/'code/source_changes.patch').write_text(''.join(diff),encoding='utf-8')
    analysis = source/'analysis_roundE_results_20261020'
    for p in analysis.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.json','.csv','.png','.log','.txt'):
            copy(p,STAGE/'analysis/confirmed'/p.relative_to(analysis))
    for folder in ('analysis_roundE_20261020','analysis_roundE_results_20261020'):
        for p in (source/folder).glob('*'):
            if p.suffix in ('.py','.cjs'):
                copy(p,STAGE/'code/original_analysis'/folder/p.name)
    for p in source.glob('VG-CPOD_v1_E组*.md'):
        copy(p,STAGE/'results/package_support/original_reports'/(p.name+'.txt'))
    inventory = []
    for seed in SEEDS:
        run = STAGE/'results/raw_export/roundE_20261020/E1/ut-zap50k/V1'/('seed%d'%seed)
        p,m,ex = read(run/'protocol.json'),read(run/'metrics.json'),read(run/'exact_selection/metrics.json')
        inventory.append(dict(case='E1',seed=seed,completed_epochs=10,selected_epoch=m['selected_epoch'],
            exact_selected_epoch=ex['selected_epoch'],path=run.relative_to(STAGE).as_posix(),
            config=p['config'],metrics=m['metrics'],exact_selection_metrics=ex['metrics']))
    write(STAGE/'configs/experiment_inventory.json',inventory)
    write(STAGE/'configs/intervention_inventory.json',[
        dict(case='E1',seed=s,selection=selection,
             path='results/raw_export/roundE_20261020/'+folder+'/E1/seed'+str(s),
             conditions=['dynamic','train_mean'],optimizer_steps=0)
        for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')] for s in SEEDS])
    write(STAGE/'external_inputs.json',dict(framework='frozen OpenAI CLIP ViT-L/14',
        cache='/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        feature_manifest_sha256='bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        A_references='../02/results/raw_export/roundA_20261019',
        C_first_seed='../04/results/raw_export/roundC_20261020',
        C_additional_and_D='../05/results/raw_export',
        cache_metadata='../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14',
        training_inputs='Reuse server feature arrays; repository contains cache metadata and text features.'))
    (STAGE/'.gitignore').write_text('analysis/recomputed/\n',encoding='utf-8')
    create_recompute(source)


def create_recompute(source):
    text = (source/'analysis_roundE_results_20261020/analyze_e.py').read_text(encoding='utf-8')
    start,end = text.index('ROOT = Path('),text.index('SETTINGS =')
    text = text[:start]+'''STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent/'recomputed'
ARCHIVE = STAGE/'results/raw_export'
A_OUT = STAGE.parent/'02/results/raw_export/roundA_20261019'
C_OUT = STAGE.parent/'05/results/raw_export/roundC_20261020'
D_OUT = STAGE.parent/'05/results/raw_export/roundD_20261020'
E_OUT = ARCHIVE/'roundE_20261020'
CACHE = STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'


def source_file(package,member):
    number = {'vgcpod_v1_roundC':'04','vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06'}[package]
    stage = STAGE.parent/number
    mapping = json.loads((stage/'package_members.json').read_text(encoding='utf-8'))['members']
    return stage/mapping[member]

'''+text[end:]
    replacements = {
        "return next((base/case).glob('ut-zap50k/V*/seed%d'%seed))":
            "if case.startswith('C') and seed==3351:\n        base = STAGE.parent/'04/results/raw_export/roundC_20261020'\n    return next((base/case).glob('ut-zap50k/V*/seed%d'%seed))",
        "ROOT/'analysis_20261018/analyze_results.py'":"STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "ARCHIVE/'vgcpod_v1_roundD/vgcpod/metrics.py'":"source_file('vgcpod_v1_roundD','vgcpod/metrics.py')",
        "ROOT/'vgcpod_roundE_results_20261020.tar.gz'":"STAGE/'results/archives/vgcpod_roundE_results_20261020.tar.gz'",
        "read(source/'configs'/sub/(case+'.json'))":"read(source_file(source.name,'configs/'+sub+'/'+case+'.json'))",
        "helper.sha(source/name)==fingerprint==helper.sha(ROOT/source.name/name)":"helper.sha(source_file(source.name,name))==fingerprint",
        "helper.sha(ARCHIVE/source/'mean_gate_eval.py')":"helper.sha(source_file(source,'mean_gate_eval.py'))"}
    for before,after in replacements.items():
        assert before in text,before;text = text.replace(before,after)
    (STAGE/'analysis/recompute.py').write_text('# Numerical calculations unchanged; repository paths adapted.\n'+text,encoding='utf-8')


def finalize():
    write(STAGE/'inputs.json',dict(line='VG-CPOD方案',stage='06',version='v1',title=TITLE,knowledge=[],
        completed_runs=3,model_epochs=30,training_evaluations=12,secondary_selection_evaluations=12,
        intervention_checkpoint_selections=6,intervention_conditions=12,intervention_evaluations=48,
        predictions=72,checkpoint_files=12,completed_cases=['E1'],completed_seeds=list(SEEDS),
        run_matrix=[dict(case='E1',seed=s) for s in SEEDS],package_member_map='package_members.json',
        experiment_inventory='configs/experiment_inventory.json',
        code_files=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        configs=[p.relative_to(STAGE).as_posix() for p in sorted((STAGE/'configs').rglob('*.json'))],
        documents=[p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='06',primary_package=PACKAGE,
        commands=[dict(source='README.md',language=lang,text=text) for lang,text in
            re.findall(r'```(bash|powershell)\n(.*?)```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)]))
    files = sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in files if p.name!='file_manifest.json'])
    stages = [s for s in read(ROOT/'provenance/stages.json') if s.get('id')!='25']
    stages.append(dict(id='25',line='VG-CPOD方案',number='06',version='v1',title='06 · '+TITLE,
        path='VG-CPOD方案/06',primary_package=PACKAGE,status='E1三seed、双准则选权重及六项均值评估归档',
        completed_runs=3,model_epochs=30,evaluations=72,predictions=72,checkpoints=12))
    write(ROOT/'provenance/stages.json',stages)
    write(ROOT/'provenance/updates/20261020_vgcpod_06.json',dict(date='2026-10-20',line='VG-CPOD方案',
        number='06',version='v1',title=TITLE,imported_from='E release ZIP and downloaded server export',
        new_runs=[dict(case='E1',seed=s) for s in SEEDS],reference_stages=['02','04','05'],
        formula_revision='rho 0.125 -> 0.15; independent first-max native/exact checkpoint rules',
        model_py_unchanged=True,planned_only=['F1 rho=0.15, beta=0.05, three seeds']))
    owned = {p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/06/file_manifest.json','README.md','VG-CPOD方案/README.md','tools/reproduce.py',
        'tools/register_vgcpod_round_e.py','provenance/stages.json','provenance/updates/20261020_vgcpod_06.json'])
    current = {r['path']:r for r in read(ROOT/'provenance/current_manifest.json')}
    for rel in sorted(owned):
        p = ROOT/rel;current[rel] = dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[p] for p in sorted(current)])


def check():
    entries = read(STAGE/'file_manifest.json')
    for row in entries:
        assert sha(STAGE/row['path'])==row['sha256'],row['path']
    mapping = package_map(STAGE)
    release = read(STAGE/mapping['release_manifest.json'])['files_sha256']
    assert set(mapping)==set(release)|{'release_manifest.json'}
    with zipfile.ZipFile(STAGE/'results/archives'/ZIP) as z:
        for name,target in mapping.items():
            assert z.read(PACKAGE+'/'+name)==(STAGE/target).read_bytes(),name
    for name,digest in release.items():
        assert sha(STAGE/mapping[name])==digest,name
    for name in (ZIP,TAR):
        p = STAGE/'results/archives'/name
        assert sha(p)==p.with_name(name+'.sha256').read_text().split()[0]
    raw = STAGE/'results/raw_export/roundE_20261020'
    assert len(list(raw.rglob('predictions/*.npz')))==72
    assert len(list(raw.rglob('checkpoints/*.pt')))==12
    assert (raw/'launcher.exitcode').read_text().strip()=='0'
    for seed in SEEDS:
        run = raw/'E1/ut-zap50k/V1'/('seed%d'%seed)
        p,m,h,ex = read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json'),read(run/'exact_selection/metrics.json')
        assert len(h)==10 and h[-1]['step']==3600
        assert m['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_native'])+1
        assert ex['selected_epoch']==max(range(10),key=lambda i:h[i]['val_cw']['AUC_exact'])+1
        assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')==m['checkpoint_sha256']
        assert sha(run/'checkpoints/best_exact.pt')==ex['checkpoint_sha256']
        for name,digest in p['source_sha256'].items():
            assert sha(STAGE/mapping[name])==digest,(seed,name)
    docs = list(STAGE.rglob('README.md'));assert len(docs)==5
    formulas = 0
    for p in docs:
        text = p.read_text(encoding='utf-8');assert text.count('```')%2==0
        assert '$$' not in text
        formulas += text.count('```math\n')
        for link in re.findall(r'\]\(([^)\n]+)\)',text):
            link = link.split('#')[0]
            assert not link or re.match(r'^[a-z]+://',link) or (p.parent/link).exists(),(p,link)
    for p in list((STAGE/'code').rglob('*.py'))+list((STAGE/'analysis').glob('*.py')):
        ast.parse(p.read_text(encoding='utf-8-sig'))
    return dict(passed=True,files_verified=len(entries),trained_runs=3,training_predictions=12,
        secondary_selection_predictions=12,intervention_predictions=48,predictions=72,
        checkpoint_files=12,readmes=5,display_formulas=formulas,training_started=False)


def materialize(destination):
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Use an absolute destination')
    destination = destination.resolve()
    if destination.exists() or destination==ROOT or ROOT in destination.parents:
        raise FileExistsError(destination)
    mapping = package_map(STAGE)
    for member,rel in mapping.items():
        target = destination/PACKAGE/member;copy(STAGE/rel,target)
        if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
    for member,digest in read(destination/PACKAGE/'release_manifest.json')['files_sha256'].items():
        assert sha(destination/PACKAGE/member)==digest,member
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='06',package=PACKAGE,
        files_verified=len(mapping),training_started=False))
    return dict(line='VG-CPOD方案',stage='06',package=str(destination/PACKAGE),
        files_verified=len(mapping),training_started=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--finalize',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--materialize',type=Path);args = parser.parse_args()
    if args.import_data and args.source:
        import_data(args.source);result = dict(imported=True,stage='06')
    elif args.finalize:
        finalize();result = check()
    elif args.check:
        result = check()
    elif args.materialize:
        result = materialize(args.materialize)
    else:
        parser.error('Choose an action')
    print(json.dumps(result,ensure_ascii=False))
