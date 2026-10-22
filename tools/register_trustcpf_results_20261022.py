"""Register and verify the complete 2026-10-22 TRUST-CPF server result batch."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from register_trustcpf_v1 import ROOT, STAGE, sha, read, write, safe_name

BATCH = 'server-results-20261022'
VERSION = ROOT / STAGE
RESULTS = VERSION / 'results' / BATCH
ANALYSIS = VERSION / 'analysis' / BATCH
MAP = RESULTS / 'server_member_map.json'
RECEIPT = ROOT / 'provenance/updates/20261022_trustcpf_results.json'
ARCHIVE = 'trustcpf_clip_v1_results_20261022.tar.gz'


def copy_identical(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = sha(source)
    if target.exists():
        if sha(target) != fingerprint:
            raise ValueError('Conflicting archived bytes: ' + str(target))
    else:
        shutil.copy2(source, target)
        if sha(target) != fingerprint:
            raise ValueError('Copy differs: ' + str(target))
    return fingerprint


def import_source(source):
    source = Path(source).resolve()
    export = source / 'server_results_20261022/trustcpf_clip_v1'
    release = read(VERSION / 'results/release_member_map.json')['members']
    files = sorted(p for p in export.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    if len(files) != 434:
        raise ValueError('Expected 434 original server members, found ' + str(len(files)))
    entries = []
    for p in files:
        member = p.relative_to(export).as_posix()
        safe_name(member)
        if member in release:
            target = ROOT / release[member]
            if sha(p) != sha(target):
                raise ValueError('Frozen release differs from server: ' + member)
            kind = 'reused_frozen_release_member'
        else:
            target = RESULTS / 'raw_export/trustcpf_clip_v1' / member
            copy_identical(p, target)
            kind = 'server_training_result'
        entries.append(dict(runtime_path='trustcpf_clip_v1/' + member,
                            current_path=target.relative_to(ROOT).as_posix(),
                            sha256=sha(target), bytes=target.stat().st_size, kind=kind))
    archive = RESULTS / 'archives' / ARCHIVE
    fingerprint = copy_identical(source / ARCHIVE, archive)
    archive.with_name(ARCHIVE + '.sha256').write_text(fingerprint + '  ' + ARCHIVE + '\n', encoding='ascii')
    write(MAP, dict(date='2026-10-22', source_archive=archive.relative_to(ROOT).as_posix(),
                    source_archive_sha256=fingerprint, source_files=len(entries),
                    source_regular_files=424, source_hardlinks=10, source_bytes=sum(e['bytes'] for e in entries),
                    members=entries))
    for p in sorted((source / 'analysis_TRUST_CPF_v1_20261022').glob('*')):
        if p.is_file():
            copy_identical(p, ANALYSIS / 'confirmed' / p.name)
    for p in sorted((source / 'analysis_TRUST_CPF_v1_20261022/publication').rglob('*')):
        if p.is_file():
            copy_identical(p, ANALYSIS / 'confirmed' / p.relative_to(source / 'analysis_TRUST_CPF_v1_20261022/publication'))
    copy_identical(source / 'output/server_archive_utf8_repair_20261022.json', RESULTS / 'validation/source_archive_integrity.json')
    report = source / 'output/TRUST-CPF_v1_完整结果分析与下一轮实验方向_20261022.md'
    copy_identical(report, RESULTS / 'package_support' / (report.name + '.txt'))
    for name in ['analyze_trustcpf_complete.py', 'write_trustcpf_analysis_report.py', 'make_trustcpf_result_figures.py']:
        copy_identical(source / 'tmp' / name, VERSION / 'code/original_analysis/server-results-20261022' / name)
    copy_identical(source / 'tmp/trustcpf_complete_analysis.log', ANALYSIS / 'confirmed/original_analysis_output.log')
    write(RECEIPT, dict(date='2026-10-22', line='TRUST-CPF方案', stage='v1', archive_stage='32',
                        batch=BATCH, original_archive_sha256=fingerprint, original_members=434,
                        byte_identity_preserved=True, completed_main_runs=21, completed_shuffled_runs=1,
                        utility_ablation_audits=2, noise_evaluations=600, noise_json_evaluations=529,
                        recovered_ow_evaluations=600, complete_robust_mismatch_cw_available=False,
                        next_experiment_planned_only=True, next_gpu_ids=[1, 2, 3], training_started_by_import=False))
    print(json.dumps({'imported_server_members': len(entries), 'reused_release_members': sum(e['kind'].startswith('reused') for e in entries),
                      'source_bytes': sum(e['bytes'] for e in entries)}, ensure_ascii=False))


def finalize():
    from register_trustcpf_v1 import finalize as finalize_release
    inputs = read(VERSION / 'inputs.json')
    inputs.update(status='server_training_and_complete_analysis_archived',
                  formal_training_results_included=True, completed_runs=21,
                  completed_shuffled_runs=1, completed_utility_ablation_audits=2,
                  noise_evaluations=600, noise_json_evaluations=529,
                  recovered_ow_evaluations=600, latest_results_batch=BATCH,
                  server_member_map=MAP.relative_to(ROOT).as_posix())
    write(VERSION / 'inputs.json', inputs)
    stages = read(ROOT / 'provenance/stages.json')
    for stage in stages:
        if stage['id'] == '32':
            stage.update(status='21组主实验、知识打乱、两项效用消融、600次噪声评估及完整分析已归档',
                         completed_runs=21, completed_shuffled_runs=1,
                         utility_ablation_audits=2, noise_evaluations=600,
                         latest_results_batch=BATCH)
    write(ROOT / 'provenance/stages.json', stages)
    layout = read(ROOT / 'provenance/layout.json')
    layout['files'] = [r for r in layout['files'] if r.get('kind') != 'trustcpf_v1_server_result_20261022']
    for row in read(MAP)['members']:
        layout['files'].append(dict(original_path=ARCHIVE + '::' + row['runtime_path'],
                                    runtime_path=row['runtime_path'], current_path=row['current_path'],
                                    sha256=row['sha256'], bytes=row['bytes'], stage='32',
                                    kind='trustcpf_v1_server_result_20261022'))
    write(ROOT / 'provenance/layout.json', layout)
    index = ROOT / 'provenance/metrics_index.csv'
    with index.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, [r for r in reader if r.get('route') != 'TRUST-CPF-server-20261022']
    raw = RESULTS / 'raw_export/trustcpf_clip_v1/work'
    for p in sorted(raw.glob('*/runs/*/metrics.json')):
        value = read(p)
        for phase in ['val', 'test']:
            for world in ['cw', 'ow']:
                m = value[phase][world]
                row = {k: '' for k in fields}
                row.update(stage='32', route='TRUST-CPF-server-20261022', dataset='ut-zap50k',
                           seed=value['seed'], variant=value['variant'], phase=phase, setting=world,
                           selected_epoch=value['epoch'], auc_pct=m['AUC_native'], hm_pct=m['at_val_bias']['HM'],
                           source=p.relative_to(ROOT).as_posix(), note='Native AUC; HM at fixed VAL bias; independent run')
                rows.append(row)
    with index.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finalize_release()
    owned = sorted(p for p in VERSION.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    manifest_path = RESULTS / 'file_manifest.json'
    write(manifest_path, [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
                          for p in owned if p != manifest_path])
    current = {r['path']: r for r in read(ROOT / 'provenance/current_manifest.json')}
    extra = [ROOT / 'README.md', ROOT / 'VG-CPOD方案/README.md', ROOT / 'TRUST-CPF方案/README.md',
             ROOT / 'tools/register_trustcpf_results_20261022.py', ROOT / 'tools/register_trustcpf_v1.py',
             ROOT / 'tools/reproduce.py', ROOT / 'provenance/layout.json', ROOT / 'provenance/stages.json',
             ROOT / 'provenance/metrics_index.csv', ROOT / 'provenance/documents/trustcpf_results_20261022.json', RECEIPT]
    for p in owned + extra:
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            current[rel] = dict(path=rel, sha256=sha(p), bytes=p.stat().st_size)
    write(ROOT / 'provenance/current_manifest.json', [current[p] for p in sorted(current)])
    print(json.dumps({'indexed_stage_files': len(owned), 'main_results': 21, 'next_gpu_ids': [1, 2, 3]}))


def check():
    record = read(MAP)
    failures = []
    for row in record['members']:
        p = ROOT / row['current_path']
        if not p.is_file() or sha(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            failures.append(row['runtime_path'])
    for row in read(RESULTS / 'file_manifest.json'):
        p = ROOT / row['path']
        if not p.is_file() or sha(p) != row['sha256']:
            failures.append(row['path'])
    archive = ROOT / record['source_archive']
    if sha(archive) != record['source_archive_sha256']:
        failures.append(record['source_archive'])
    raw = RESULTS / 'raw_export/trustcpf_clip_v1'
    main_metrics = list((raw / 'work').glob('seed*/runs/*/metrics.json'))
    shuffled = list((raw / 'work').glob('shuffled*/runs/*/metrics.json'))
    audit = [raw / ('work/' + name + '3351/utility/audit.json') for name in ['semantic', 'unmeasured']]
    analysis = read(ANALYSIS / 'confirmed/analysis.json')
    assert len(main_metrics) == 21 and len(shuffled) == 1 and all(p.is_file() for p in audit)
    assert analysis['noise_log_recovery']['recovered_ow_records'] == 600
    assert sum(analysis['noise_coverage'].values()) == 529
    assert all(v == 0 for v in analysis['noise_log_recovery']['log_vs_json_max_errors'].values())
    assert len(record['members']) == 434
    assert not failures, failures
    return dict(passed=True, source_members=434, main_runs=21, shuffled_runs=1,
                utility_ablation_audits=2, recovered_ow_noise_evaluations=600,
                noise_json_evaluations=529, original_bytes_preserved=True,
                complete_robust_mismatch_cw_available=False, training_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.source:
        import_source(args.source)
    elif args.finalize:
        finalize()
    elif args.check:
        print(json.dumps(check(), ensure_ascii=False))
    else:
        parser.error('Choose --source, --finalize or --check')
