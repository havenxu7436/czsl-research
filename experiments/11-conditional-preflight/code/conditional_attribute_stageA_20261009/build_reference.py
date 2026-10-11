"""Build fixed server fingerprints from the locally verified evidence ZIP."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    if (root / 'reference.json').exists():
        raise FileExistsError('Preserve existing reference.json')
    with zipfile.ZipFile(a.zip) as z:
        if z.testzip() or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Invalid ZIP')
        for name in z.namelist():
            q = PurePosixPath(name)
            if q.is_absolute() or '..' in q.parts or '\\' in name:
                raise ValueError('Unsafe ZIP member')
        manifest = json.loads(z.read('manifest.json'))['files']
        if set(manifest) != set(z.namelist()) - {'manifest.json'}:
            raise ValueError('Manifest inventory mismatch')
        for name, record in manifest.items():
            value = z.read(name)
            if len(value) != record['size'] or hashlib.sha256(value).hexdigest() != record['sha256']:
                raise ValueError('ZIP fingerprint mismatch: ' + name)
        audit = json.loads(z.read('audit.json'))
        if not (audit['bf_record_checks_passed'] and audit['conditional_input_checks_passed']):
            raise ValueError('Preflight evidence failed')
        if audit['missing'] or any(not x['passed'] for x in audit['checks']):
            raise ValueError('Missing/failed evidence')
        protocol = json.loads(z.read('bf/protocol.json'))
        cfg = audit['conditional']['config_path']
        checkpoint = audit['conditional']['checkpoint_path']
        def server_files(prefix):
            return {v['path']: v['sha256'] for k, v in manifest.items()
                    if k.startswith(prefix) and 'path' in v}
        cfiles = server_files('conditional/source/')
        cfiles.update(server_files('conditional/bank/'))
        cfiles.update(server_files('conditional/cache/'))
        cfiles[cfg] = manifest['conditional/C_text_s0/config_protocol.yml']['sha256']
        cfiles[checkpoint] = audit['conditional']['reference']['checkpoint_sha256']
        cfiles.update(protocol['split_sha256'])
        cfiles.update(protocol['metadata_sha256'])
        bank_manifest = json.loads(z.read('conditional/bank/manifest.json'))
        cfiles['/data/xhw/czsl_repro/data/glove/glove.6B.300d.txt'] = bank_manifest['glove_sha256']
        cache_manifest = json.loads(z.read('conditional/cache/manifest.json'))
        for name in ('W_a_name_raw.pt', 'W_o_name_raw.pt', 'W_a_K.pt', 'W_o_K.pt'):
            cfiles['/var/tmp/czsl_xhw_ut_v2_20261006/cache/ut-zap50k/' + name] = cache_manifest['files_sha256'][name]
        bf_files = server_files('bf/source/')
        bf_files.update(server_files('bf/logic_bank/'))
        bf_files.update(protocol['split_sha256'])
        bf_files.update(protocol['metadata_sha256'])
        bf_files['/data/xhw/czsl_repro/data/glove/glove.6B.300d.txt'] = bank_manifest['glove_sha256']
        bf = {}
        for variant in ('B', 'F'):
            spec = protocol['configs'][variant]
            bf_files[spec['path']] = spec['sha256']
            record = audit['bf'][variant]['checkpoint_record']
            bf_files[record['path']] = record['sha256']
            bf[variant] = dict(config=spec['path'], checkpoint=record['path'],
                               checkpoint_sha256=record['sha256'],
                               best_epoch=audit['bf'][variant]['evaluations']['cw']['best_epoch'])
        labels = json.loads(z.read('conditional/bank/labels.json'))
        reference = dict(schema='conditional_attribute_stageA_v1',
                         preflight_zip_sha256=hashlib.sha256(a.zip.read_bytes()).hexdigest(),
                         checks=128, c_source='/var/tmp/czsl_xhw_ut_v2_20261006/src',
                         bf_source=protocol['source'], config=cfg, checkpoint=checkpoint,
                         bank='/var/tmp/czsl_xhw_ut_v2_20261006/bank/ut-zap50k',
                         c_files=cfiles, bf_files=bf_files, bf=bf, labels=labels,
                         c_reference=audit['conditional']['reference'],
                         metrics={s: audit['conditional']['evaluations'][s]['metrics_raw'] for s in ('cw', 'ow')},
                         val_evidence={})
        for setting in ('cw', 'ow'):
            prefix = 'conditional/C_text_s0/eval_val_' + setting + '/'
            reference['val_evidence'][setting] = {
                'candidates': json.loads(z.read(prefix + 'candidates.json')),
                'image_ids': json.loads(z.read(prefix + 'image_ids.json')),
                'labels_file': manifest[prefix + 'labels.npz']['path'],
                'labels_sha256': manifest[prefix + 'labels.npz']['sha256']}
            cfiles[reference['val_evidence'][setting]['labels_file']] = reference['val_evidence'][setting]['labels_sha256']
        (root / 'reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Source/control differences are compared excluding run/log/output/seed identity.
        import yaml
        configs = {v: yaml.safe_load(z.read('bf/runs/' + v + '/config_protocol.yml'))
                   for v in ('A_ref', 'B', 'E_ref', 'F')}
        ignored = {('TRAIN', k) for k in ('checkpoint_dir', 'log_dir', 'train_name', 'kcc_variant', 'kcc_run_dir')}
        diffs = {}
        for left, right in (('A_ref', 'B'), ('E_ref', 'F')):
            changes = []
            for section in sorted(set(configs[left]) | set(configs[right])):
                x, y = configs[left].get(section), configs[right].get(section)
                if isinstance(x, dict) and isinstance(y, dict):
                    for key in sorted(set(x) | set(y)):
                        if x.get(key) != y.get(key) and (section, key) not in ignored:
                            changes.append(dict(field=section + '.' + key, left=x.get(key), right=y.get(key)))
                elif x != y:
                    changes.append(dict(field=section, left=x, right=y))
            diffs[left + '_vs_' + right] = changes
        curves = {}
        for v in configs:
            log = z.read('bf/runs/' + v + '/train.log').decode('utf-8', errors='replace')
            curves[v] = [{'epoch': int(e), 'cw_auc_pct_rounded': 100 * float(auc)}
                         for e, auc in re.findall(r'Val/Test Epoch (\d+): AUC=([\d.]+)', log)]
        report = dict(preflight_passed=True, independent_scores_recomputed=False,
                      paired_config_differences=diffs, curves_from_rounded_logs=curves,
                      bf={v: audit['bf'][v]['evaluations'] for v in configs},
                      saved_smoke_is_pretraining=True, trained_checkpoint_gradient_probe_pending=True,
                      real_conditional_replay_pending=True)
        (root / 'audit_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('REFERENCE_BUILT', len(cfiles), len(bf_files))


if __name__ == '__main__':
    main()
