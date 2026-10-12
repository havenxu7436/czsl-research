"""Read-only verification of curated seed3351 evidence; optional native CPU replay."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parents[4]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=ROOT / 'experiments/17-formula-aligned-0601/results/seed3351-30epoch-20261012')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    result = args.result.absolute()
    raw = result / 'raw_export'
    mapping = read(result / 'export_member_map.json')
    expected = read(raw / 'export_manifest.json')['files_sha256']
    if set(mapping['members']) != set(expected) | {'export_manifest.json'}:
        raise ValueError('Mapped original member set differs')
    for member, digest in expected.items():
        relative = Path(mapping['members'][member])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise ValueError('Unsafe mapped path')
        if sha(ROOT / relative) != digest:
            raise ValueError('Evidence/source checksum differs: ' + member)
    audit = read(result / 'audit.json')
    if not audit['passed'] or mapping['source_zip_sha256'] != audit['source_zip_sha256']:
        raise ValueError('Audit identity differs')
    report = {'passed': True, 'original_export_files_verified': len(expected),
              'completed_runs': audit['completed_seed3351_jobs'], 'epochs_actually_executed_per_run': 30,
              'recommended_next_epochs': 10, 'training_started': False,
              'cpu_replay': 'not requested'}
    if args.replay:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        helper = load(ROOT / 'experiments/16-residual-0601/code/review_0601_20261011/audit.py', 'existing_metric_helpers')
        ev = load(ROOT / mapping['members']['source/evaluator_ge.py'], 'verified_seed_evaluator')
        if ev.device != 'cpu':
            raise ValueError('This verifier is CPU-only')
        split_dir = ROOT / 'experiments/01-baseline/results/data/ut-zap50k/compositional-split-natural'
        protocol = read(raw / 'protocol.json')
        splits = {}
        for phase in ('train', 'val', 'test'):
            path = split_dir / (phase + '_pairs.txt')
            server_path = '/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural/' + path.name
            if sha(path) != protocol['frozen_inputs'][server_path]:
                raise ValueError('Split differs: ' + phase)
            splits[phase] = [tuple(line.split()) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        labels = read(raw / 'runs/A_s3351/meta/labels.json')
        differences = []
        for job in protocol['jobs']:
            if job['seed'] != 3351:
                continue
            for phase in ('val', 'test'):
                for setting in ('cw', 'ow'):
                    folder = raw / 'runs' / job['variant'] / ('eval_' + phase + '_' + setting)
                    meta = read(folder / 'metrics.json')
                    ds = helper.dataset(labels, splits, phase, setting)
                    with np.load(folder / 'labels.npz', allow_pickle=False) as truth:
                        matrix = np.memmap(folder / 'scores.dat', mode='r', dtype=np.float32, shape=tuple(meta['score_shape']))
                        scores = helper.evaluate(matrix, truth, ds, ev.Evaluator)
                    differences.append(max(abs(scores[key] - meta['metrics_raw'][key]) for key in scores))
        if len(differences) != 56 or max(differences) >= 2e-6:
            raise ValueError('Native metrics differ')
        report['cpu_replay'] = {'evaluations': len(differences), 'max_abs_difference': max(differences)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
