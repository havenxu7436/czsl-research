#!/usr/bin/env python3
"""Read existing B/F records and C_text inputs; export evidence, never train.

Python 3.8+, standard library only. No imports from experiment source code,
no torch.load, optimizer, CUDA, evaluation, image reads, or source edits.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


BF_VARIANTS = ('A_ref', 'B', 'E_ref', 'F')
C_RUN = 'semantic_ut-zappos_C_text_s0_e10'
C_REFERENCE = {
    'checkpoint_sha256': '45de209c4c38af6d7cfa62f6ff51f62ec8f095ee18edc22d893f43184f3199a6',
    'bank_manifest_sha256': 'ed17757749ffe48fb0d3e2303d7032ea6fce12a27eabbdbb6cdfc87fc4a8a71e',
    'bank_sha256': '94259cca60bdc132d26a6c26a799d823826152d4a9dcf60fe8bcc93e0cca0525',
    'config_sha256': '4bd532e1faedd7965bd25ec87c53c358fe2758b5816a92e4ac858e6c82032f44',
    'cw_auc': 0.45440386970321,
    'ow_auc': 0.34937344606543497,
    'best_epoch': 6,
}
C_SOURCE_REFERENCE = {
    'dataset.py': '430ac2b7f388ce28dd84e1218a7e3180b87d4620a86f9ab339240f204948086f',
    'evaluator_ge.py': '24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978',
    'models/semantic_control.py': '0206bb561f77d7fa104460f5c789d9a2432fa1b7057a09dd81fda3ceb6ae184c',
    'transition/runtime.py': '2ef3875c4a7aa62d1879b72f07724ec063dd4d6ac2d3c785b2ce6c0b39106afe',
}
LOGIC_PATTERN = re.compile(
    r'BF_LOGIC epoch=(\d+) iter=(\d+) cpf=([^ ]+) oi=([^ ]+) '
    r'oe=([^ ]+) ae=([^ ]+) weighted=([^\s]+)')
MAX_FILE_BYTES = 64 * 1024 ** 2
MAX_EXPORT_BYTES = 256 * 1024 ** 2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def json_bytes(value):
    def serializable(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {'nonfinite_numeric': repr(item)}
        if isinstance(item, dict):
            return {k: serializable(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [serializable(v) for v in item]
        return item
    return (json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


class Evidence:
    def __init__(self):
        self.files = {}
        self.checks = []
        self.missing = []
        self.notes = []
        self.total_bytes = 0

    def check(self, section, name, passed, detail=None):
        self.checks.append(dict(section=section, name=name, passed=bool(passed), detail=detail))

    def collect(self, path, archive_name, root, required=True):
        path = Path(path)
        name = PurePosixPath(archive_name)
        if name.is_absolute() or '..' in name.parts or '\\' in archive_name:
            raise ValueError('Unsafe archive name: ' + archive_name)
        if not inside(path, root):
            self.check('safety', 'file_stays_inside_named_root', False, str(path))
            return False
        if not path.is_file():
            if required:
                self.missing.append(dict(path=str(path), archive_name=archive_name))
            return False
        if archive_name in self.files:
            if Path(self.files[archive_name]['path']).resolve() != path.resolve():
                raise ValueError('Archive name collision: ' + archive_name)
            return True
        size = path.stat().st_size
        if size > MAX_FILE_BYTES or self.total_bytes + size > MAX_EXPORT_BYTES:
            raise ValueError('Evidence size budget exceeded; preserve originals: ' + str(path))
        self.files[archive_name] = dict(path=str(path.resolve()), size=size, sha256=sha(path))
        self.total_bytes += size
        return True

    def read_json(self, path, name, root, required=True):
        if not self.collect(path, name, root, required):
            return None
        try:
            return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except (ValueError, UnicodeError) as error:
            self.check('parse', 'json_parse', False, dict(path=str(path), error=str(error)))
            return None

    def tree(self, folder, prefix, root, extensions=None):
        folder = Path(folder)
        if not inside(folder, root) or not folder.is_dir():
            self.missing.append(dict(path=str(folder), archive_name=prefix))
            return
        for path in sorted(folder.rglob('*')):
            if path.is_file() and (extensions is None or path.suffix in extensions):
                if '__pycache__' not in path.parts and '.git' not in path.parts:
                    self.collect(path, prefix + '/' + path.relative_to(folder).as_posix(), root)

    def fingerprint(self, section, path, expected, root):
        ok = (bool(expected) and inside(path, root) and Path(path).is_file()
              and sha(path) == expected)
        self.check(section, 'sha256', ok, dict(path=str(path), expected=expected))
        return ok

    def passed(self, section):
        checks = [item for item in self.checks if item['section'] == section]
        missing = [item for item in self.missing if item['archive_name'].startswith(section + '/')]
        unsafe = any(not item['passed'] for item in self.checks if item['section'] in ('safety', 'parse'))
        return bool(checks) and not missing and not unsafe and all(item['passed'] for item in checks)


def inspect_bf(evidence, work, seed):
    protocol = evidence.read_json(work / 'experiments/protocol' / ('seed_%d.json' % seed),
                                 'bf/protocol.json', work)
    if not isinstance(protocol, dict):
        evidence.check('bf', 'protocol_available', False)
        return {}
    evidence.check('bf', 'frozen_protocol_identity',
                   protocol.get('seed') == seed and protocol.get('epochs') == 10
                   and protocol.get('q') == 3 and protocol.get('beta') == 0.01
                   and set(protocol.get('configs', {})) == set(BF_VARIANTS))
    source = Path(protocol.get('source', str(work / 'src')))
    install = evidence.read_json(source / 'bf_install.json', 'bf/source/bf_install.json', work)
    evidence.fingerprint('bf', source / 'bf_install.json', protocol.get('install_sha256'), work)
    if isinstance(install, dict):
        for relative, expected in install.get('output_hashes', {}).items():
            evidence.fingerprint('bf', source / relative, expected, work)
    evidence.tree(source, 'bf/source', work, {'.py', '.yml', '.yaml'})
    for label, folder in (('logic_bank', work / 'logic_bank/ut-zap50k'),):
        for name in ('manifest.json', 'coverage.json', 'bank.pt'):
            evidence.collect(folder / name, 'bf/' + label + '/' + name, work)
    evidence.fingerprint('bf', work / 'logic_bank/ut-zap50k/manifest.json',
                         protocol.get('logic_bank_manifest_sha256'), work)
    bank_manifest = evidence.read_json(work / 'logic_bank/ut-zap50k/manifest.json',
                                       'bf/logic_bank/manifest.json', work)
    if isinstance(bank_manifest, dict):
        evidence.fingerprint('bf', work / 'logic_bank/ut-zap50k/bank.pt',
                             bank_manifest.get('bank_sha256'), work)
    report = {}
    for variant in BF_VARIANTS:
        spec = protocol.get('configs', {}).get(variant)
        if not isinstance(spec, dict) or not spec.get('path') or not spec.get('run'):
            evidence.check('bf', 'run_spec_' + variant, False)
            continue
        run, cfg = Path(spec['run']), Path(spec['path'])
        prefix = 'bf/runs/' + variant
        evidence.collect(cfg, prefix + '/config_protocol.yml', work)
        evidence.fingerprint('bf', cfg, spec.get('sha256'), work)
        for name in ('config_requested.yml', 'train.log', 'status.txt', 'start_time.txt', 'end_time.txt',
                     'eval_val_cw.log', 'eval_val_ow.log'):
            evidence.collect(run / name, prefix + '/' + name, work)
        for name in ('config_effective.yml', 'config_effective.yaml', 'training_metrics.json'):
            evidence.collect(run / name, prefix + '/' + name, work, required=False)
        evidence.tree(run / 'meta', prefix + '/meta', work, {'.json', '.txt', '.sha256', '.gz'})
        meta = evidence.read_json(run / 'meta/checkpoint.json', prefix + '/meta/checkpoint.json', work)
        complete = ((run / 'status.txt').read_text().strip() == 'train_and_val_complete'
                    if inside(run, work) and (run / 'status.txt').is_file() else False)
        evidence.check('bf', variant + '_reported_complete', complete)
        evaluations = {}
        for setting in ('cw', 'ow'):
            folder = run / ('eval_val_' + setting)
            for name in ('config_eval_effective.yml', 'candidates.json', 'image_ids.json', 'labels.npz'):
                evidence.collect(folder / name, prefix + '/eval_val_' + setting + '/' + name, work)
            evidence.collect(folder / 'scores.dat', prefix + '/eval_val_' + setting + '/scores.dat',
                             work, required=False)
            result = evidence.read_json(folder / 'metrics.json',
                                        prefix + '/eval_val_' + setting + '/metrics.json', work)
            if isinstance(result, dict):
                raw = result.get('metrics_raw', {})
                good_metrics = all(isinstance(raw.get(k), (int, float)) and math.isfinite(raw[k])
                                   and 0 <= raw[k] <= 1
                                   for k in ('AUC', 'best_hm', 'best_seen', 'best_unseen'))
                evidence.check('bf', variant + '_' + setting + '_result_identity',
                               result.get('phase') == 'val' and result.get('seed') == seed
                               and result.get('variant') == variant and result.get('setting') == setting
                               and result.get('dataset') == 'ut-zappos' and good_metrics)
                evaluations[setting] = result
        if isinstance(meta, dict) and meta.get('path'):
            checkpoint = Path(meta['path'])
            print('HASH_CHECKPOINT', variant, str(checkpoint), flush=True)
            evidence.fingerprint('bf', checkpoint, meta.get('sha256'), run)
            evidence.check('bf', variant + '_same_checkpoint_cw_ow',
                           len(evaluations) == 2 and all(
                               value.get('checkpoint_sha256') == meta.get('sha256')
                               for value in evaluations.values()))
        else:
            evidence.check('bf', variant + '_checkpoint_record', False)
        log_path = run / 'train.log'
        observations = []
        if inside(log_path, work) and log_path.is_file():
            text = log_path.read_text(encoding='utf-8', errors='replace')
            for match in LOGIC_PATTERN.findall(text):
                values = {}
                for key, text_value in zip(('cpf', 'oi', 'oe', 'ae', 'weighted'), match[2:]):
                    try:
                        values[key] = float(text_value)
                    except ValueError:
                        values[key] = None
                observations.append(dict(epoch=int(match[0]), iteration=int(match[1]), **values))
        if variant in ('B', 'F'):
            evidence.check('bf', variant + '_finite_active_logic_through_epoch10',
                           bool(observations) and max(row['epoch'] for row in observations) == 10
                           and all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) for row in observations
                                   for k in ('cpf', 'oi', 'oe', 'ae', 'weighted'))
                           and any(isinstance(row['weighted'], (int, float)) and row['weighted'] > 0
                                   for row in observations))
            smoke = evidence.read_json(work / 'experiments/smoke' / ('%s_s%d.json' % (variant, seed)),
                                       'bf/smoke/' + variant + '.json', work)
            good_smoke = (isinstance(smoke, dict) and smoke.get('passed') is True
                          and smoke.get('config_sha256') == spec.get('sha256')
                          and smoke.get('install_sha256') == protocol.get('install_sha256')
                          and smoke.get('logic_manifest_sha256') == protocol.get('logic_bank_manifest_sha256'))
            gradients = smoke.get('individual_visual_gradient_norm', {}) if isinstance(smoke, dict) else {}
            good_smoke = good_smoke and all(
                isinstance(gradients.get(k), (int, float)) and math.isfinite(gradients[k])
                and gradients[k] > 0 for k in ('loss_oi', 'loss_oe', 'loss_ae'))
            evidence.check('bf', variant + '_saved_cuda_smoke_identity_and_gradients', good_smoke)
        report[variant] = dict(evaluations=evaluations, logic_observations=observations,
                               reported_complete=complete, checkpoint_record=meta)
    return report


def inspect_conditional_inputs(evidence, work):
    source = work / 'src'
    exp = work / 'experiments/semantic_v2'
    run = exp / 'runs' / C_RUN
    prepared = evidence.read_json(work / 'prepared.json', 'conditional/prepared.json', work)
    index = evidence.read_json(exp / 'config_index_semantic_e10.json', 'conditional/config_index.json', work)
    evidence.check('conditional', 'C_text_config_index_identity',
                   isinstance(index, dict) and C_RUN in index)
    cfg_path = Path(index[C_RUN]) if isinstance(index, dict) and C_RUN in index else exp / 'configs' / (C_RUN + '.yml')
    evidence.collect(cfg_path, 'conditional/C_text_s0/config_protocol.yml', work)
    evidence.fingerprint('conditional', cfg_path, C_REFERENCE['config_sha256'], work)
    evidence.tree(source, 'conditional/source', work, {'.py', '.yml', '.yaml'})
    for name in ('transition_install.json', 'v2_assets/assets_sha256.json'):
        evidence.collect(source / name, 'conditional/source/' + name, work)
    if isinstance(prepared, dict):
        for relative, expected in prepared.get('code_manifest', {}).items():
            evidence.fingerprint('conditional', source / relative, expected, work)
    for relative, expected in C_SOURCE_REFERENCE.items():
        evidence.fingerprint('conditional', source / relative, expected, work)
    for label, folder in (('bank', work / 'bank/ut-zap50k'), ('cache', work / 'cache/ut-zap50k')):
        for name in ('manifest.json', 'labels.json', 'descriptions.json', 'report.json', 'cache_manifest.json'):
            evidence.collect(folder / name, 'conditional/' + label + '/' + name, work,
                             required=(name == 'manifest.json' or (label == 'bank' and name in ('labels.json', 'descriptions.json'))))
    evidence.collect(work / 'bank/ut-zap50k/bank.pt', 'conditional/bank/bank.pt', work)
    evidence.fingerprint('conditional', work / 'bank/ut-zap50k/manifest.json',
                         C_REFERENCE['bank_manifest_sha256'], work)
    evidence.fingerprint('conditional', work / 'bank/ut-zap50k/bank.pt',
                         C_REFERENCE['bank_sha256'], work)
    for name in ('train.log', 'status.json'):
        evidence.collect(run / name, 'conditional/C_text_s0/' + name, work)
    evidence.tree(run / 'meta', 'conditional/C_text_s0/meta', work, {'.json', '.txt', '.sha256'})
    evaluations = {}
    for setting in ('cw', 'ow'):
        folder = run / ('eval_val_' + setting)
        prefix = 'conditional/C_text_s0/eval_val_' + setting
        for name in ('config_eval_effective.yml', 'candidates.json', 'image_ids.json', 'labels.npz'):
            evidence.collect(folder / name, prefix + '/' + name, work)
        evidence.collect(folder / 'scores.dat', prefix + '/scores.dat', work, required=False)
        result = evidence.read_json(folder / 'metrics.json', prefix + '/metrics.json', work)
        if isinstance(result, dict):
            auc = result.get('metrics_raw', {}).get('AUC')
            evidence.check('conditional', 'C_text_' + setting + '_known_anchor',
                           result.get('run_id') == C_RUN and result.get('variant') == 'C_text'
                           and result.get('seed') == 0 and result.get('phase') == 'val'
                           and result.get('setting') == setting
                           and result.get('best_epoch') == C_REFERENCE['best_epoch']
                           and result.get('checkpoint_sha256') == C_REFERENCE['checkpoint_sha256']
                           and result.get('bank_manifest_sha256') == C_REFERENCE['bank_manifest_sha256']
                           and isinstance(auc, (int, float)) and math.isfinite(auc)
                           and abs(auc - C_REFERENCE[setting + '_auc']) < 1e-12)
            evaluations[setting] = result
    checkpoint = run / 'checkpoints/best_auc.pth'
    print('HASH_CHECKPOINT C_text_s0', str(checkpoint), flush=True)
    evidence.fingerprint('conditional', checkpoint, C_REFERENCE['checkpoint_sha256'], run)
    return dict(run_id=C_RUN, config_path=str(cfg_path), checkpoint_path=str(checkpoint),
                evaluations=evaluations, reference=C_REFERENCE,
                cache_extraction_performed=False, new_model_implemented=False)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError('ZIP duplicate member or CRC failure')
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or '..' in parsed.parts or '\\' in name:
                raise ValueError('Unsafe ZIP member')
        manifest = json.loads(archive.read('manifest.json'))
        expected = set(manifest['files']) | {'manifest.json'}
        if set(names) != expected:
            raise ValueError('ZIP inventory differs from manifest')
        for name, record in manifest['files'].items():
            data = archive.read(name)
            if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('ZIP member hash mismatch: ' + name)
    return dict(files=len(manifest['files']), sha256=sha(path))


def export(bf_work, ut_work, out, seed=3351):
    bf_work, ut_work, out = Path(bf_work).resolve(), Path(ut_work).resolve(), Path(out).absolute()
    sidecar = Path(str(out) + '.sha256')
    if out.exists() or sidecar.exists():
        raise FileExistsError('Preserve existing export; use a new --out filename: ' + str(out))
    if inside(out, bf_work) or inside(out, ut_work):
        raise ValueError('Output must be outside both original experiment work directories')
    if not bf_work.is_dir():
        raise FileNotFoundError('B/F work directory not found; specify its actual --bf-work: ' + str(bf_work))
    if not out.parent.is_dir():
        raise FileNotFoundError('Output parent must exist: ' + str(out.parent))
    if shutil.disk_usage(out.parent).free < 512 * 1024 ** 2:
        raise ValueError('Need 512 MiB free for a bounded evidence export; do not delete experiment weights')
    evidence = Evidence()
    start = time.monotonic()
    bf_report = inspect_bf(evidence, bf_work, seed)
    conditional_report = inspect_conditional_inputs(evidence, ut_work)
    report = dict(
        schema='bf_conditional_preflight_v1', date_utc=datetime.now(timezone.utc).isoformat(),
        bf_work=str(bf_work), ut_work=str(ut_work), bf_seed=seed,
        bf_record_checks_passed=evidence.passed('bf'),
        conditional_input_checks_passed=evidence.passed('conditional'),
        checks=evidence.checks, missing=evidence.missing, notes=evidence.notes,
        bf=bf_report, conditional=conditional_report,
        new_training=False, new_optimizer_steps=0, new_evaluation=False,
        test_images_read=0, model_weights_in_export=False,
        interpretation_limits=[
            'Saved smoke gradients are pre-training connection checks, not gradients at the selected checkpoint.',
            'This script does not independently rerun the evaluator or verify image/data tensor hashes.',
            'A consistent export does not prove efficacy, convergence, or gradient alignment.',
            'C_text input location and hashes are not real-data replay of a new conditional model.',
            'Missing files and failed checks remain explicit; do not train past a known provenance mismatch.',
        ])
    # No source is executed. All archived bytes must retain their pre-export hash.
    files_manifest = {}
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, record in sorted(evidence.files.items()):
            data = Path(record['path']).read_bytes()
            if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('Source changed during export; preserve the partial ZIP: ' + record['path'])
            archive.writestr(name, data)
            files_manifest[name] = record
        for name, data in (('audit.json', json_bytes(report)), ('export_script.py', Path(__file__).read_bytes())):
            archive.writestr(name, data)
            files_manifest[name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
        for name, record in evidence.files.items():
            if sha(record['path']) != record['sha256']:
                raise ValueError('Source changed before export commit: ' + record['path'])
        report['copied_sources_unchanged'] = True
        # Explicit final read-only audit is separate from the earlier audit.json.
        final_data = json_bytes(dict(copied_sources_unchanged=True,
                                     copied_files=len(evidence.files),
                                     export_seconds=time.monotonic() - start))
        archive.writestr('export_integrity.json', final_data)
        files_manifest['export_integrity.json'] = dict(size=len(final_data), sha256=hashlib.sha256(final_data).hexdigest())
        archive.writestr('manifest.json', json_bytes(dict(schema='sha256_inventory_v1', files=files_manifest)))
    verified = verify_archive(out)
    with sidecar.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(verified['sha256'] + '  ' + out.name + '\n')
    print('EXPORT_COMPLETE', out, flush=True)
    print('BF_RECORD_CHECKS_PASSED', report['bf_record_checks_passed'], flush=True)
    print('C_TEXT_INPUT_CHECKS_PASSED', report['conditional_input_checks_passed'], flush=True)
    print('MISSING_FILES', len(report['missing']), flush=True)
    print('NEW_TRAINING=0 NEW_EVALUATION=0 TEST_IMAGES_READ=0', flush=True)
    print('ZIP_SHA256', verified['sha256'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bf-work', type=Path, default=Path('/var/tmp/czsl_xhw_bf_20261008'))
    parser.add_argument('--ut-work', type=Path, default=Path('/var/tmp/czsl_xhw_ut_v2_20261006'))
    parser.add_argument('--out', type=Path, default=Path('/var/tmp/czsl_bf_conditional_preflight_20261009.zip'))
    parser.add_argument('--seed', type=int, choices=[3351], default=3351)
    parser.add_argument('--verify-zip', type=Path, help='Validate a downloaded evidence ZIP; no experiment reads')
    args = parser.parse_args()
    if args.verify_zip:
        print('ZIP_VERIFIED', json.dumps(verify_archive(args.verify_zip)), flush=True)
    else:
        export(args.bf_work, args.ut_work, args.out, args.seed)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print('STOP:', str(error), file=sys.stderr, flush=True)
        sys.exit(1)
