"""Lossless, hash-checked import of the 2026-10-15 v2 seed3351 export.

No training, pickle loading, metric recomputation or network access. Original
members are either preserved byte-for-byte or linked to identical canonical
assets. Run --check after import to verify the complete member maps offline.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).absolute().parents[1]
STAGE = '0601方案/v2'
BATCH = 'seed3351-10epoch-20261015'
RESULT = STAGE + '/results/' + BATCH
FIX = 'revised_0601_20261014_fix'
WORK = 'czsl_0601_revision_all_20261014_fix'
EXPORT_SHA = 'f18d322db5e15255de4d8678f526f4353e66f403b3020992e36c9af53e6325bb'
RELEASE_SHA = '21c1dcffb9275cb5b0838d9ca72739e6a6f255c89a684282885dda8acc77b4ad'
RECEIPT = 'provenance/updates/20261015_v2_seed3351.json'
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|https?://[^\s/@:]+:[^\s/@]+@)')
TEXT = {'.json', '.txt', '.log', '.md', '.yml', '.yaml', '.py', '.sh'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def write(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or ':' in name or '\x00' in name or name != path.as_posix()):
        raise ValueError('Unsafe archive path: ' + repr(name))
    return path


def load_zip(path, expected):
    if file_sha(path) != expected:
        raise ValueError('ZIP SHA256 mismatch: ' + path.name)
    sidecar = path.with_suffix(path.suffix + '.sha256')
    if sidecar.read_text(encoding='utf-8-sig').split()[0] != expected:
        raise ValueError('SHA256 sidecar mismatch: ' + path.name)
    with zipfile.ZipFile(path) as handle:
        infos = handle.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        payloads = {}
        for item in infos:
            safe_name(item.filename.rstrip('/'))
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Archive symlink is not an evidence file')
            if item.is_dir():
                continue
            name = item.filename
            suffix = PurePosixPath(name).suffix.lower()
            if suffix in {'.pth', '.pem', '.key', '.jpg', '.jpeg', '.png'} or PurePosixPath(name).name in {'.env', '.git-credentials'}:
                raise ValueError('Out-of-scope weight/image/credential: ' + name)
            data = handle.read(item)  # Also verifies the member CRC.
            if suffix in TEXT and SECRET.search(data):
                raise ValueError('Possible credential in ' + name)
            payloads[name] = data
    return payloads


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(export, release):
    e = load_zip(export, EXPORT_SHA)
    original = load_zip(release, RELEASE_SHA)
    prefix = 'revised_0601_20261013/'
    require(all(n.startswith(prefix) for n in original), 'Unexpected release root')
    r = {n[len(prefix):]: data for n, data in original.items()}
    manifest = json.loads(e['export_manifest.json'])['files_sha256']
    require(set(e) == set(manifest) | {'export_manifest.json'}, 'Export member set differs from manifest')
    require(all(sha(e[n]) == digest for n, digest in manifest.items()), 'Export member SHA256 mismatch')
    released = json.loads(r['release_manifest.json'])
    require(set(r) == set(released) | {'release_manifest.json'}, 'Release member set differs from manifest')
    require(all(sha(r[n]) == digest for n, digest in released.items()), 'Release member SHA256 mismatch')
    source = [n for n in e if n.startswith('source/')]
    require(all(n in r and e[n] == r[n] for n in source), 'Executed source differs from fixed release')
    protocol = json.loads(e['protocol.json'])
    summary = json.loads(e['summary/summary.json'])
    require(protocol['epochs'] == 10 and protocol['seeds'] == [3351, 3352, 3353], 'Wrong protocol budget or seeds')
    require(len(protocol['jobs']) == 96, 'Expected 96 planned jobs, not 96 completed jobs')
    require(summary['protocol_sha256'] == sha(e['protocol.json']), 'Summary protocol identity mismatch')
    jobs = [j for j in protocol['jobs'] if j['seed'] == 3351]
    require(len(jobs) == 32 and len({j['variant'] for j in jobs}) == 32, 'Incomplete seed3351 matrix')
    names = {j['variant'] for j in jobs}
    require({n.split('/')[1] for n in e if n.startswith('runs/')} == names, 'Unexpected/missing run folders')
    states = {s['variant']: s for s in summary['states']}
    rows = {(v['variant'], v['phase'], v['setting']): v for v in summary['rows']}
    require(len(rows) == len(summary['rows']) == 128, 'Expected 128 distinct evaluations')
    require(not summary['all_planned_complete'], 'Do not claim all seeds complete')
    require(all(s['state'] == 'not_started' for s in summary['states'] if s['seed'] != 3351), 'Export contains another seed result')
    frozen_checked = 0
    external = []
    for path, digest in protocol['frozen_inputs'].items():
        marker = '/'+WORK+'/'
        if marker in path:
            name = path.split(marker, 1)[1]
            require(name in e and sha(e[name]) == digest, 'Frozen exported input mismatch: ' + name)
            frozen_checked += 1
        else:
            external.append({'path': path, 'sha256': digest})
    for job in protocol['jobs']:
        cfg = 'configs/' + job['variant'] + '.yml'
        require(cfg in e and sha(e[cfg]) == job['cfg_sha256'], 'Frozen job config differs: ' + cfg)
    job_checks = []
    for job in jobs:
        name = job['variant']
        base = 'runs/' + name + '/'
        state = json.loads(e[base + 'status.json'])
        complete = json.loads(e[base + 'train_complete.json'])
        checkpoint = json.loads(e[base + 'meta/checkpoint.json'])
        smoke = json.loads(e[base + 'smoke.json'])
        require(state['state'] == states[name]['state'] == 'complete', 'Run not complete: ' + name)
        require(states[name]['epochs'] == 10 and complete['epochs'] == list(range(1, 11)), 'Incomplete epochs: ' + name)
        require(complete['best_sha256'] == checkpoint['sha256'] == states[name]['checkpoint_sha256'], 'Checkpoint identity differs: ' + name)
        require(e[base + 'config_requested.yml'] == e['configs/' + name + '.yml'], 'Requested config differs: ' + name)
        require(e[base + 'meta/protocol.json'] == e['protocol.json'], 'Per-run protocol differs: ' + name)
        require(smoke['passed'] and smoke['training_acceptance']['passed'], 'CUDA smoke/optimizer-resume acceptance failed: ' + name)
        epochs = [int(v) for v in re.findall(rb'FULL_EPOCH_FINISHED epoch=(\d+)', e[base+'train.log'])]
        require(epochs == list(range(1, 11)), 'Training log lacks ten completed epochs: ' + name)
        best_epochs = set()
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                folder = base + 'eval_' + phase + '_' + setting + '/'
                metric = json.loads(e[folder + 'metrics.json'])
                row = rows[(name, phase, setting)]
                require(metric['checkpoint_sha256'] == row['checkpoint_sha256'] == checkpoint['sha256'], 'Evaluation checkpoint differs: ' + folder)
                require(metric['seed'] == 3351 and metric['phase'] == phase and metric['setting'] == setting, 'Evaluation scope mismatch: ' + folder)
                require(metric['best_epoch'] == row['best_epoch'] and 1 <= row['best_epoch'] <= 10, 'Best epoch mismatch: ' + folder)
                require(sha(e[folder+'config_eval_effective.yml']) == metric['config_sha256'], 'Evaluation config hash mismatch: ' + folder)
                require(metric['scores_saved'] and metric['score_dtype'] == 'float32', 'Missing raw float32 scores: ' + folder)
                require(len(e[folder+'scores.dat']) == metric['score_shape'][0] * metric['score_shape'][1] * 4, 'Score matrix byte-size mismatch: ' + folder)
                require(abs(metric['metrics_raw']['AUC'] * 100 - row['AUC_pct']) < 1e-8, 'Summary AUC differs from raw metric: ' + folder)
                best_epochs.add(row['best_epoch'])
        require(len(best_epochs) == 1, 'Four evaluations used different epochs: ' + name)
        job_checks.append({'variant': name, 'epochs': epochs, 'best_epoch': best_epochs.pop(),
                           'checkpoint_sha256': checkpoint['sha256'], 'cuda_smoke_passed': True,
                           'optimizer_resume_acceptance_passed': True, 'evaluations': 4})
    result = {'passed': True, 'date': '2026-10-15', 'source_zip_sha256': EXPORT_SHA,
              'fixed_release_zip_sha256': RELEASE_SHA, 'export_members': len(e),
              'export_manifest_hashes_verified': len(manifest), 'release_manifest_hashes_verified': len(released),
              'executed_source_files_identical_to_fixed_release': len(source),
              'frozen_exported_inputs_verified': frozen_checked, 'external_frozen_inputs_not_in_export': external,
              'completed_seeds': [3351], 'completed_runs': 32, 'evaluations': 128,
              'planned_runs': 96, 'all_planned_complete': False, 'scope_scan_passed': True,
              'server_training_started': False, 'raw_bytes_modified': False,
              'metric_recomputation_performed': False, 'checkpoint_binaries_included': False,
              'source_manifest_fields_null_in_metrics': all(json.loads(e[n])['source_manifest_sha256'] is None for n in e if n.endswith('/metrics.json')),
              'identity_note': 'Source identity is independently checked against frozen_inputs and the fixed release; null metric fields are preserved, not fabricated.',
              'jobs': job_checks}
    return e, r, result


def import_data(export, release):
    e, r, report = audit(export, release)
    require(not (ROOT / RESULT / 'export_member_map.json').exists(), 'Already imported; use --check')
    registry = read('knowledge/registry.json')
    layout = read('provenance/layout.json')
    inputs = read(STAGE + '/inputs.json')
    by_hash = {}
    for asset in registry['assets']:
        by_hash.setdefault(asset['sha256'], asset)
    categories = {'knowledge': set(), 'configs': set(), 'code_files': set(), 'data_files': set()}
    created = set()
    records = []

    def store(name, data, target, runtime, archive_name, category):
        safe_name(name)
        safe_name(target)
        digest = sha(data)
        if category == 'knowledge':
            asset = by_hash.get(digest)
            if asset is None:
                asset = {'path': target, 'sha256': digest, 'bytes': len(data), 'dataset': 'ut-zap50k', 'aliases': [], 'stages': []}
                registry['assets'].append(asset)
                by_hash[digest] = asset
            target = asset['path']
            alias = archive_name + '::' + name
            if alias not in asset['aliases']:
                asset['aliases'].append(alias)
            if '19' not in asset['stages']:
                asset['stages'].append('19')
        path = ROOT / target
        if path.exists():
            require(path.is_file() and file_sha(path) == digest, 'Preserve existing different content: ' + target)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            created.add(target)
        categories[category].add(target)
        records.append({'original_path': archive_name+'::'+name, 'runtime_path': runtime,
                        'current_path': target, 'sha256': digest, 'bytes': len(data),
                        'stage': '19', 'kind': 'v2_fixed_release_member' if archive_name == release.name else 'v2_seed3351_export_member'})
        return target

    release_map = {}
    for name, data in sorted(r.items()):
        if name.startswith('assets/logic_bank/'):
            target, category = 'knowledge/ut-zap50k/logic/0601-v2-20261014-fix/'+name.split('logic_bank/', 1)[1], 'knowledge'
        elif name in ('assets/knowledge_revision.json', 'assets/additional_descriptions.json'):
            target, category = 'knowledge/ut-zap50k/definitions/0601-v2-20261014-fix/'+PurePosixPath(name).name, 'knowledge'
        elif PurePosixPath(name).suffix in ('.yml', '.yaml') or name == 'configs/variant_options.json':
            target, category = STAGE+'/configs/'+FIX+'/'+name, 'configs'
        elif PurePosixPath(name).suffix in ('.py', '.sh'):
            target, category = STAGE+'/code/'+FIX+'/'+name, 'code_files'
        elif PurePosixPath(name).suffix == '.md':
            target, category = STAGE+'/results/package_support/'+FIX+'/'+name+'.txt', 'data_files'
        else:
            target, category = STAGE+'/results/'+FIX+'/'+name, 'data_files'
        release_map[name] = store('revised_0601_20261013/'+name, data, target, FIX+'/'+name, release.name, category)
    mapping, source_links = {}, {}
    for name, data in sorted(e.items()):
        suffix = PurePosixPath(name).suffix
        if name.startswith('cache_') or name.startswith('logic_bank/'):
            kind = 'logic' if name.startswith('logic_bank/') else 'text-cache'
            target, category = 'knowledge/ut-zap50k/'+kind+'/0601-v2-20261014-fix/'+name, 'knowledge'
        elif name.startswith('source/'):
            # All 50 source members were independently compared with the fixed release.
            target = release_map[name]
            category = 'configs' if suffix in ('.yml', '.yaml') else ('code_files' if suffix == '.py' else 'data_files')
            source_links[name] = target
        elif suffix in ('.yml', '.yaml'):
            target, category = STAGE+'/configs/'+BATCH+'/'+name, 'configs'
        elif suffix == '.md':
            target, category = RESULT+'/package_support/'+name+'.txt', 'data_files'
        else:
            target, category = RESULT+'/raw_export/'+name, 'data_files'
        mapping[name] = store(name, data, target, WORK+'/'+name, export.name, category)
    layout['files'].extend(records)
    write('provenance/layout.json', layout)
    write('knowledge/registry.json', registry)
    write(RESULT+'/export_member_map.json', {'source_zip_sha256': EXPORT_SHA, 'members': mapping})
    write(RESULT+'/fixed_release_member_map.json', {'source_zip_sha256': RELEASE_SHA, 'original_package': 'revised_0601_20261013', 'runtime_package': FIX, 'members': release_map})
    write(RESULT+'/source_links.json', source_links)
    write(RESULT+'/audit.json', report)
    write(RESULT+'/upload_scope.json', {'original_zip_not_duplicated': True, 'raw_export_members_preserved_or_canonical_linked': len(mapping),
          'binary_scores_and_knowledge_use_git_lfs': True, 'model_weights_dataset_images_credentials_included': False,
          'completed_seeds': [3351], 'other_seed_configs_are_plans_only': True,
          'summary_is_historical_export_snapshot_not_live_server_status': True,
          'posthoc_analysis_included': False, 'trained_source_release': FIX,
          'initial_20261013_release_preserved': True, 'all_member_bytes_unchanged': True})
    (ROOT/RESULT/'original_zip.sha256').write_bytes(export.with_suffix('.zip.sha256').read_bytes())
    deployment = STAGE+'/results/deployment/'+release.name
    (ROOT/deployment).write_bytes(release.read_bytes())
    (ROOT/(deployment+'.sha256')).write_bytes(release.with_suffix('.zip.sha256').read_bytes())
    categories['data_files'].update((deployment, deployment+'.sha256'))
    for category, paths in categories.items():
        inputs[category] = sorted(set(inputs[category]) | paths)
        if category == 'configs':
            inputs[category] = [name for name in inputs[category]
                               if not re.search(r'_s\d+\.yml$', name) or name.endswith('_s3351.yml')]
    inputs.update(status='seed3351_complete', completed_runs=32, evaluations=128,
                  completed_seeds=[3351], active_package=FIX,
                  result_batches=[{'path': RESULT, 'seed': 3351, 'variants': 32, 'epochs': 10, 'evaluations': 128,
                                   'source_zip_sha256': EXPORT_SHA, 'fixed_release_zip_sha256': RELEASE_SHA}])
    write(STAGE+'/inputs.json', inputs)
    write(RECEIPT, {'date': '2026-10-15', 'archive_stage': '19', 'stage': 'v2',
          'source_zip_sha256': EXPORT_SHA, 'fixed_release_zip_sha256': RELEASE_SHA,
          'export_members': len(mapping), 'release_members_including_manifest': len(release_map),
          'completed_runs': 32, 'evaluations': 128, 'completed_seeds': [3351],
          'planned_runs': 96, 'all_planned_complete': False, 'member_bytes_modified': False,
          'server_training_started': False, 'checkpoints_included': False,
          'scope_scan_passed': True, 'metric_recomputation_performed': False,
          'baseline_repository_issue': 'Remote commit 2959576 changed 0601方案/README.md without refreshing current_manifest; current documentation hashes are refreshed in this import.',
          'new_canonical_payload_files': sorted(created), 'new_layout_records': len(records),
          'audit': RESULT+'/audit.json', 'mapping': RESULT+'/export_member_map.json'})
    print(json.dumps({k: report[k] for k in ('passed', 'export_members', 'completed_runs', 'evaluations', 'frozen_exported_inputs_verified')}, ensure_ascii=False))


def check():
    mapped = read(RESULT+'/export_member_map.json')['members']
    release_map = read(RESULT+'/fixed_release_member_map.json')['members']
    manifest = read(mapped['export_manifest.json'])['files_sha256']
    release_manifest = read(release_map['release_manifest.json'])
    require(set(mapped) == set(manifest) | {'export_manifest.json'}, 'Export mapping is incomplete')
    require(set(release_map) == set(release_manifest) | {'release_manifest.json'}, 'Fixed release mapping is incomplete')
    checked = 0
    for mapping, hashes in ((mapped, manifest), (release_map, release_manifest)):
        for name, digest in hashes.items():
            require(file_sha(ROOT/mapping[name]) == digest, 'Mapped bytes changed: '+name)
            checked += 1
    mapped_layout = {r['original_path'].split('::', 1)[1]: r for r in read('provenance/layout.json')['files']
                     if r['kind'] == 'v2_seed3351_export_member'}
    require(set(mapped_layout) == set(mapped), 'Export layout records are incomplete')
    for name, target in mapped.items():
        record = mapped_layout[name]
        require(target == record['current_path'] and file_sha(ROOT/target) == record['sha256'], 'Layout identity differs: '+name)
    receipt = read(RECEIPT)
    require(receipt['source_zip_sha256'] == EXPORT_SHA, 'Source ZIP identity changed')
    print(json.dumps({'passed': True, 'export_members_mapped': len(mapped), 'fixed_release_members_mapped': len(release_map),
                      'manifest_hashes_verified': checked, 'training_started': False}))


def restore_check():
    """Exercise the repository's restoration path rules for this import only."""
    from reproduce import safe_child
    selected = [r for r in read('provenance/layout.json')['files']
                if r['kind'] in ('v2_seed3351_export_member', 'v2_fixed_release_member')]
    destination = Path(tempfile.mkdtemp(prefix='czsl-v2-seed3351-restore-'))
    for record in selected:
        target = safe_child(destination, record['runtime_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), 'New runtime path collision: '+record['runtime_path'])
        shutil.copy2(ROOT/record['current_path'], target)
        require(file_sha(target) == record['sha256'], 'Restored member hash differs')
    write(RESULT+'/restoration_check.json', {'passed': True, 'date': '2026-10-15',
          'new_member_copies_verified': len(selected), 'export_members': 1766, 'fixed_release_members': 126,
          'restoration_uses_repository_safe_child': True, 'runtime_path_collisions': 0,
          'historical_stages_not_recopied_by_this_check': True, 'training_started': False})
    print(json.dumps({'passed': True, 'files_restored_and_sha256_verified': len(selected),
                      'temporary_check_directory': str(destination), 'training_started': False}))


def finalize():
    """Refresh only import-owned/documented paths, retaining unrelated hashes."""
    inputs = read(STAGE+'/inputs.json')
    extra = [p.relative_to(ROOT).as_posix() for p in (ROOT/RESULT).rglob('*') if p.is_file()]
    inputs['data_files'] = sorted(set(inputs['data_files']) | set(extra))
    write(STAGE+'/inputs.json', inputs)
    stages = read('provenance/stages.json')
    for stage in stages:
        if stage['id'] == '19':
            stage.update(status='seed3351：32组×10epoch完成，128份评价已归档',
                         active_package=FIX, completed_seeds=[3351], completed_runs=32, evaluations=128,
                         knowledge_assets=len(inputs['knowledge']), configs=len(inputs['configs']), data_files=len(inputs['data_files']))
    write('provenance/stages.json', stages)
    current = {r['path']: r for r in read('provenance/current_manifest.json')}
    selected = {r['current_path'] for r in read('provenance/layout.json')['files'] if r['stage'] == '19' and r['current_path']}
    selected.update(extra)
    selected.update(inputs['data_files'])
    selected.update(('README.md', '0601方案/README.md', STAGE+'/README.md', STAGE+'/results/README.md',
                     STAGE+'/inputs.json', STAGE+'/runbook.json', 'knowledge/README.md', 'knowledge/registry.json',
                     'provenance/layout.json', 'provenance/stages.json', RECEIPT, 'tools/import_v2_seed3351.py'))
    for name in sorted(selected):
        path = ROOT/name
        current[name] = {'path': name, 'sha256': file_sha(path), 'bytes': path.stat().st_size}
    write('provenance/current_manifest.json', [current[n] for n in sorted(current)])
    check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--import-data', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--restore-check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    elif args.finalize:
        finalize()
    elif args.restore_check:
        restore_check()
    elif args.export and args.release:
        if args.import_data:
            import_data(args.export, args.release)
        else:
            _, _, result = audit(args.export, args.release)
            print(json.dumps({k: v for k, v in result.items() if k not in ('jobs', 'external_frozen_inputs_not_in_export')}, ensure_ascii=False))
    else:
        parser.error('Choose --check/--finalize or provide --export and --release (audit only unless --import-data)')
