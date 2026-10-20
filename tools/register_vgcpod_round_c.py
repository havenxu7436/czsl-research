"""Archive experiment 04 and restore the exact C-group training package."""
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
STAGE = ROOT / 'VG-CPOD方案/04'
PREVIOUS = ROOT / 'VG-CPOD方案/03'
PACKAGE = 'vgcpod_v1_roundC'
TITLE = '静态幅度对照与模式中心化门控'
ZIP = 'vgcpod_v1_roundC_20261020.zip'
TAR = 'vgcpod_roundC_results_20261020.tar.gz'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def relative_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unexpected archive member: ' + name)
    return path


def import_data(source):
    source = Path(source).resolve()
    if STAGE.exists():
        raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    mapping = {}
    with zipfile.ZipFile(source / ZIP) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith('/'):
                continue
            parts = relative_member(member).parts
            assert parts[0] == PACKAGE, member
            name = PurePosixPath(*parts[1:]).as_posix()
            if name.startswith('configs/'):
                target = name
            elif name.startswith(('assets/', 'references/')):
                target = 'configs/' + name
            elif name.startswith('validation/') or name == 'release_manifest.json':
                target = 'results/package_validation/' + name
            elif Path(name).suffix in ('.md', '.html'):
                target = 'results/package_support/' + name + '.txt'
            else:
                target = 'code/' + PACKAGE + '/' + name
            destination = STAGE / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            mapping[name] = target
    write(STAGE / 'package_members.json', {'package': PACKAGE, 'members': mapping})
    for name in (ZIP, ZIP + '.sha256', TAR):
        copy(source / name, STAGE / 'results/archives' / name)
    (STAGE / 'results/archives' / (TAR + '.sha256')).write_text(sha(source / TAR) + '  ' + TAR + '\n', encoding='ascii')
    previous_map = read(PREVIOUS / 'package_members.json')['members']
    raw_manifest, matched_c, matched_b = [], [], []
    with tarfile.open(source / TAR, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = relative_member(member.name).parts
            in_c = parts[0] == 'roundC_20261020' and parts[-1] != '.round_c.lock'
            in_b = parts[0] == 'roundB_20261019' and (
                (parts[1:2] == ('B3',) and 'seed3352' in parts) or parts[1:] == ('logs', 'B3_seed3352.log'))
            if in_c or in_b:
                payload = archive.extractfile(member).read()
                target = STAGE / 'results/raw_export' / PurePosixPath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                raw_manifest.append({'path': target.relative_to(STAGE).as_posix(),
                                     'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
            elif parts[0] in (PACKAGE, 'vgcpod_v1_roundB'):
                name = PurePosixPath(*parts[1:]).as_posix()
                reference, member_map = (STAGE, mapping) if parts[0] == PACKAGE else (PREVIOUS, previous_map)
                if name in member_map:
                    assert archive.extractfile(member).read() == (reference / member_map[name]).read_bytes(), member.name
                    (matched_c if parts[0] == PACKAGE else matched_b).append(name)
    write(STAGE / 'results/raw_export_members.json', raw_manifest)
    write(STAGE / 'results/server_source_identity.json', {'source_bytes_preserved': True,
        'C_package': PACKAGE, 'C_members_matched_to_zip': sorted(matched_c),
        'B3_package_reference': '../03/package_members.json', 'B_members_matched_to_03': sorted(matched_b)})
    for folder in ('analysis_roundC_results_20261020',):
        for path in sorted((source / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.csv', '.json', '.png', '.txt'):
                copy(path, STAGE / 'analysis/pilot' / path.relative_to(source / folder))
    for folder in ('analysis_roundC_results_20261020', 'analysis_roundC_20261020'):
        for path in sorted((source / folder).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.cjs'):
                copy(path, STAGE / 'code/original_analysis' / folder / path.name)
    for path in sorted(source.glob('VG-CPOD_v1_C组*.md')):
        copy(path, STAGE / 'results/package_support/original_reports' / (path.name + '.txt'))
    patches = []
    for name, target in sorted(mapping.items()):
        if not (name.endswith(('.py', '.sh')) or name.startswith('configs/round_c/')):
            continue
        previous = (PREVIOUS / previous_map[name]).read_text(encoding='utf-8') if name in previous_map else ''
        current = (STAGE / target).read_text(encoding='utf-8')
        patches.extend(difflib.unified_diff(previous.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile='a/' + name if name in previous_map else '/dev/null', tofile='b/' + name))
    (STAGE / 'code/source_changes.patch').write_text(''.join(patches), encoding='utf-8')
    create_recompute(source)
    inventory = []
    for path in sorted((STAGE / 'results/raw_export').glob('*/?*/ut-zap50k/V*/seed*/metrics.json')):
        saved, protocol = read(path), read(path.parent / 'protocol.json')
        inventory.append({'case': protocol['config']['experiment_id'], 'seed': protocol['seed'],
            'selected_epoch': saved['selected_epoch'], 'completed_epochs': 10,
            'path': path.parent.relative_to(STAGE).as_posix(), 'config': protocol['config'],
            'runtime_source_stage': '03' if protocol['config']['experiment_id'] == 'B3' else '04',
            'metrics': saved['metrics']})
    assert len(inventory) == 3
    write(STAGE / 'configs/experiment_inventory.json', inventory)
    write(STAGE / 'external_inputs.json', {'framework': 'frozen CLIP ViT-L/14',
        'cache': '/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        'feature_manifest_sha256': 'bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        'feature_arrays': 'Reuse the existing frozen CLIP server cache; raw images and CLIP weights rebuild it.',
        'A_references': '../02/results/raw_export/roundA_20261019',
        'B3_seed3351_reference': '../03/results/raw_export/roundB_20261019/B3',
        'B3_runtime_source': '../03/package_members.json',
        'cache_metadata': '../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'})
    (STAGE / '.gitignore').write_text('analysis/recomputed/\n', encoding='utf-8')


def create_recompute(source):
    original = (source / 'analysis_roundC_results_20261020/analyze_c.py').read_text(encoding='utf-8')
    start, end = original.index('ROOT = Path('), original.index('ITEMS =')
    original = original[:start] + """STAGE = Path(__file__).resolve().parent.parent
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent / 'recomputed'
NEW = STAGE / 'results/raw_export'
BR = STAGE.parent / '03/results/raw_export/roundB_20261019'
AR = STAGE.parent / '02/results/raw_export/roundA_20261019'
OLD = STAGE.parent / 'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs'
C_MAP = json.loads((STAGE / 'package_members.json').read_text(encoding='utf-8'))['members']
B_STAGE = STAGE.parent / '03'
B_MAP = json.loads((B_STAGE / 'package_members.json').read_text(encoding='utf-8'))['members']
""" + original[end:]
    replacements = {
        "base=AR if case.startswith('A') else NEW/('roundB_20261019' if case=='B3' else 'roundC_20261020')":
            "base=AR if case.startswith('A') else (BR if case=='B3' and seed==3351 else NEW/('roundB_20261019' if case=='B3' else 'roundC_20261020'))",
        "(OUT/'tables').mkdir(parents=True,exist_ok=True)": "(OUT/'tables').mkdir(parents=True,exist_ok=True)",
        "ROOT/'analysis_20261018/analyze_results.py'": "STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'",
        "NEW/'vgcpod_v1_roundC/vgcpod/metrics.py'": "STAGE/C_MAP['vgcpod/metrics.py']",
        "ROOT/'vgcpod_roundC_results_20261020.tar.gz'": "STAGE/'results/archives/vgcpod_roundC_results_20261020.tar.gz'",
        "ROOT/'server_results_roundB_20261019/roundB_20261019/B3/ut-zap50k/V1/seed3351/metrics.json'": "BR/'B3/ut-zap50k/V1/seed3351/metrics.json'",
        "source=NEW/('vgcpod_v1_roundB' if case=='B3' else 'vgcpod_v1_roundC')\n            local=ROOT/source.name":
            "source=B_STAGE if case=='B3' else STAGE\n            member_map=B_MAP if case=='B3' else C_MAP",
        "assert cfg==read(local/('configs/round_b' if case=='B3' else 'configs/round_c')/(case+'.json'))":
            "assert cfg==read(source/('configs/round_b' if case=='B3' else 'configs/round_c')/(case+'.json'))",
        "assert sha(source/name)==fingerprint==sha(local/name)": "assert sha(source/member_map[name])==fingerprint",
    }
    for before, after in replacements.items():
        assert before in original, before
        original = original.replace(before, after)
    (STAGE / 'analysis/recompute.py').write_text(
        '# Repository paths adapted; archived numerical calculations are unchanged.\n' + original, encoding='utf-8')


def finalize():
    write(STAGE / 'inputs.json', {'line': 'VG-CPOD方案', 'stage': '04', 'version': 'v1', 'title': TITLE,
        'knowledge': [], 'completed_runs': 3, 'new_C_runs': 2, 'companion_B3_runs': 1,
        'C_launcher_training_runs': 2, 'C_launcher_reused_runs': 1,
        'model_epochs': 30, 'evaluations': 12, 'predictions': 12, 'checkpoints': 6,
        'completed_cases': ['C1', 'C2', 'B3'], 'completed_seeds': [3351, 3352],
        'run_matrix': [{'case':'C1','seed':3351}, {'case':'C2','seed':3351}, {'case':'B3','seed':3352}],
        'package_member_map': 'package_members.json', 'experiment_inventory': 'configs/experiment_inventory.json',
        'code_files': [p.relative_to(STAGE).as_posix() for p in sorted((STAGE / 'code').rglob('*')) if p.is_file()],
        'configs': [p.relative_to(STAGE).as_posix() for p in sorted((STAGE / 'configs').rglob('*.json'))],
        'documents': [p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]})
    write(STAGE / 'runbook.json', {'line':'VG-CPOD方案', 'stage':'04', 'primary_package':PACKAGE,
        'companion_source_stage':'03', 'commands':[{'source':'README.md','language':lang,'text':text}
        for lang,text in re.findall(r'```(bash|powershell)\n(.*?)```', (STAGE / 'README.md').read_text(encoding='utf-8'), re.S)]})
    files = sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE / 'file_manifest.json', [{'path':p.relative_to(STAGE).as_posix(), 'sha256':sha(p), 'bytes':p.stat().st_size}
          for p in files if p.name != 'file_manifest.json'])
    stages = [s for s in read(ROOT / 'provenance/stages.json') if s.get('id') != '23']
    stages.append({'id':'23','line':'VG-CPOD方案','number':'04','version':'v1','title':'04 · '+TITLE,
        'path':'VG-CPOD方案/04','primary_package':PACKAGE,
        'status':'C1/C2及B3补seed共3次训练结果归档；12份评价与预测、6份权重及数据分析',
        'completed_runs':3,'new_C_runs':2,'companion_B3_runs':1,'model_epochs':30,'evaluations':12,
        'predictions':12,'checkpoints':6,'run_matrix':[{'case':'C1','seed':3351},{'case':'C2','seed':3351},{'case':'B3','seed':3352}]})
    write(ROOT / 'provenance/stages.json', stages)
    owned = {p.relative_to(ROOT).as_posix() for p in files}
    owned.update(['VG-CPOD方案/04/file_manifest.json','README.md','VG-CPOD方案/README.md',
                  'tools/reproduce.py','tools/register_vgcpod_round_c.py','provenance/stages.json'])
    current = {r['path']:r for r in read(ROOT / 'provenance/current_manifest.json')}
    for rel in sorted(owned):
        path = ROOT / rel
        current[rel] = {'path':rel,'sha256':sha(path),'bytes':path.stat().st_size}
    write(ROOT / 'provenance/current_manifest.json', [current[p] for p in sorted(current)])


def check():
    entries = read(STAGE / 'file_manifest.json')
    for row in entries:
        assert sha(STAGE / row['path']) == row['sha256'], row['path']
    mapping = read(STAGE / 'package_members.json')['members']
    release = read(STAGE / mapping['release_manifest.json'])['files_sha256']
    assert set(mapping) == set(release) | {'release_manifest.json'}
    with zipfile.ZipFile(STAGE / 'results/archives' / ZIP) as archive:
        assert {n for n in archive.namelist() if not n.endswith('/')} == {PACKAGE+'/'+n for n in mapping}
        for name,target in mapping.items():
            assert archive.read(PACKAGE+'/'+name) == (STAGE / target).read_bytes(), name
    for name,digest in release.items():
        assert sha(STAGE / mapping[name]) == digest, name
    for name in (ZIP, TAR):
        assert sha(STAGE / 'results/archives' / name) == (STAGE / 'results/archives' / (name+'.sha256')).read_text().split()[0]
    for row in read(STAGE / 'results/raw_export_members.json'):
        assert sha(STAGE / row['path']) == row['sha256'], row['path']
    runs = list((STAGE / 'results/raw_export').glob('*/?*/ut-zap50k/V*/seed*/metrics.json'))
    assert len(runs) == 3
    assert len(list((STAGE / 'results/raw_export').rglob('predictions/*.npz'))) == 12
    assert len(list((STAGE / 'results/raw_export').rglob('checkpoints/*.pt'))) == 6
    assert (STAGE / 'results/raw_export/roundC_20261020/launcher.exitcode').read_text().strip() == '0'
    bmap = read(PREVIOUS / 'package_members.json')['members']
    for path in runs:
        metrics, protocol, history = read(path), read(path.parent/'protocol.json'), read(path.parent/'history.json')
        assert len(history) == 10 and history[-1]['step'] == 3600
        assert metrics['selected_epoch'] == max(range(10), key=lambda i:history[i]['val_cw']['AUC_native']) + 1
        assert sha(path.parent/'checkpoints/best.pt') == metrics['checkpoint_sha256']
        case = protocol['config']['experiment_id']
        source, members = (PREVIOUS,bmap) if case == 'B3' else (STAGE,mapping)
        assert protocol['config'] == read(source / ('configs/round_b' if case=='B3' else 'configs/round_c') / (case+'.json'))
        for name,digest in protocol['source_sha256'].items():
            assert sha(source/members[name]) == digest, (case,name)
    broken = []
    docs = list(STAGE.rglob('README.md'))
    assert len(docs) == 5
    for path in docs:
        text = path.read_text(encoding='utf-8')
        assert text.count('```') % 2 == 0, path
        assert '\\operatorname' not in text, path
        for tex in re.findall(r'```math\n(.*?)\n```', text, re.S):
            assert not any(line.endswith(chr(92)) for line in tex.splitlines()), path
        for link in re.findall(r'\]\(([^)\n]+)\)', text):
            link = link.split('#')[0]
            if link and not re.match(r'^[a-z]+://',link) and not (path.parent/link).exists():
                broken.append((str(path.relative_to(STAGE)),link))
    assert not broken, broken
    for path in list((STAGE / 'code').rglob('*.py')) + [STAGE / 'analysis/recompute.py']:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    return {'passed':True,'files_verified':len(entries),'package_members':len(mapping),
        'trained_runs':3,'new_C_runs':2,'companion_B3_runs':1,'evaluations':12,'predictions':12,'checkpoints':6,'readmes':5}


def materialize(destination):
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Use an absolute destination')
    destination = destination.resolve()
    if destination.exists() or destination == ROOT or ROOT in destination.parents:
        raise FileExistsError(destination)
    mapping = read(STAGE / 'package_members.json')['members']
    for name,source in sorted(mapping.items()):
        target = destination / PACKAGE / name
        copy(STAGE/source, target)
        if target.suffix == '.sh':
            target.chmod(target.stat().st_mode | 0o111)
    assert all(sha(destination/PACKAGE/name)==digest for name,digest in read(destination/PACKAGE/'release_manifest.json')['files_sha256'].items())
    # The original B source is required by round_c.py's B3 companion job.
    from register_vgcpod_round_b import materialize as materialize_b
    companion = materialize_b(destination / 'companion_03')
    write(destination/'materialization.json', {'line':'VG-CPOD方案','stage':'04','package':PACKAGE,
        'files':len(mapping),'companion_B_source':companion['package'],'training_started':False})
    return {'line':'VG-CPOD方案','stage':'04','package':str(destination/PACKAGE),
        'files_verified':len(mapping),'companion_B_source':companion['package'],'training_started':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--import-data', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--materialize', type=Path)
    args = parser.parse_args()
    if args.import_data and args.source:
        import_data(args.source)
        result = {'imported':True,'stage':'04','next':'Write five parallel READMEs, then --finalize'}
    elif args.finalize:
        finalize()
        result = check()
    elif args.check:
        result = check()
    elif args.materialize:
        result = materialize(args.materialize)
    else:
        parser.error('Choose --source PATH --import-data, --check, --finalize, or --materialize PATH')
    print(json.dumps(result, ensure_ascii=False))
