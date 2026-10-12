"""Exercise actual TRAIN loader against archived bytes; no encoder or VAL read."""
import json
import zipfile
from pathlib import Path
import numpy as np
from common import ROOT, Inputs, write_json, require
from analysis import load_train, make_prompts, VARIANTS


def main():
    workspace = ROOT.parent
    with zipfile.ZipFile(workspace / 'output/train_knowledge_audit_release_20261010.zip') as release:
        prefix = ROOT.name + '/'
        protocol = json.loads(release.read(prefix + 'protocol.json'))
        expected = json.loads(release.read(prefix + 'train_input_manifest.json'))['files_sha256']
        ae, oe = (json.loads(release.read(prefix + 'assets/' + n)) for n in ('ae_dict.json', 'oe_dict.json'))
    with zipfile.ZipFile(workspace / 'output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip') as old:
        inputs = Inputs(expected, archive=old)
        inputs.verify_all()
        require(inputs.js('prompts.json') == make_prompts(protocol, ae, oe), 'Actual prompt identity')
        records, features, a, o, tables, scores = load_train(inputs, protocol)
        replay = {}
        for v in VARIANTS:
            x = np.empty_like(scores[v])
            for oi in range(12):
                ix = np.flatnonzero(o == oi)
                x[ix] = features[ix] @ tables[v][oi].T
            difference = float(np.abs(x - scores[v]).max())
            mismatches = int((x.argmax(1) != scores[v].argmax(1)).sum())
            require(difference < 2e-5 and mismatches == 0, 'Actual pooled TRAIN replay')
            replay[v] = {'max_abs_difference': difference, 'top1_mismatches': mismatches}
        report = {'passed': True, 'allowed_files_verified': len(inputs.reads), 'train_images': len(records),
                  'features_shape': list(features.shape), 'replay': replay,
                  'val_files_read': [], 'test_files_read': [], 'encoder_calls': 0,
                  'limit': 'Local ZIP backend. Remote paths/deps/checkpoint and new text encoder not exercised.'}
        write_json(ROOT / 'runtime_input_validation.json', report)
        print(json.dumps(report))


if __name__ == '__main__':
    main()
