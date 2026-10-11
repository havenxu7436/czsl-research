"""Build only from local authored source + exactly verified archived r2 assets."""
import hashlib
import json
import zipfile
from pathlib import Path
from common import ROOT, sha, require, read_json, write_json
from analysis import VARIANTS

WORKSPACE = ROOT.parent
ARCHIVE = WORKSPACE / 'output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip'
EXPECTED = '7fb5d14a81e83b8b0c62fba93ec441bc3f056e0c04cf2749246d31acc6a4bb9f'


def build():
    require(sha(ARCHIVE) == EXPECTED, 'Archived r2 SHA changed')
    with zipfile.ZipFile(ARCHIVE) as z:
        files_sha = json.loads(z.read('export_manifest.json'))['files_sha256']
        def verified(name):
            content = z.read(name)
            require(hashlib.sha256(content).hexdigest() == files_sha[name], 'Archived member differs: ' + name)
            return content
        old = json.loads(verified('source/protocol.json'))
        protocol = {k: old[k] for k in ('labels', 'names', 'permutations', 'data', 'clip_source_commit', 'clip_checkpoint_sha256')}
        protocol.update(schema='train_knowledge_sentence_audit_v1',
                        probe_work='/var/tmp/czsl_clip_knowledge_probe_20261010_r2',
                        checkpoint='/var/tmp/czsl_clip_knowledge_probe_20261010_r2/model/ViT-L-14.pt',
                        work_basename='czsl_train_knowledge_audit_20261010',
                        max_train_image_decodes=96, optimizer_steps=0, image_encoder_calls=0,
                        no_val_or_test_access=True, no_weights_or_prompts_fitted=True,
                        audit_only_not_a_teacher_proceed_gate=True)
        names = ['train_records.json', 'teacher/train_truth.npz', 'text_features.npz', 'prompts.json',
                 'environment.json', 'dependencies.json']
        names += ['teacher/train_' + v + '.npz' for v in VARIANTS]
        for start in range(0, 22998, 512):
            names += ['features/train/%06d%s' % (start, suffix) for suffix in ('.npz', '.json')]
        expected_inputs = {'source_archive_sha256': EXPECTED,
                           'files_sha256': {name: files_sha['result/' + name] for name in sorted(names)}}
        # No VAL metrics, images, masks or predictions are read to design this audit.
        content = {p.name: p.read_bytes() for p in ROOT.glob('*.py')}
        content['protocol.json'] = (json.dumps(protocol, ensure_ascii=False, indent=2) + '\n').encode()
        content['train_input_manifest.json'] = (json.dumps(expected_inputs, indent=2) + '\n').encode()
        for name in z.namelist():
            if name.startswith('source/vendor/clip/') or name in ('source/assets/ae_dict.json', 'source/assets/oe_dict.json'):
                content[name[len('source/'):]] = verified(name)
        local_tests = read_json(ROOT / 'local_test_report.json')
        require(local_tests['passed'], 'Local tests have not passed')
        content['local_test_report.json'] = (json.dumps(local_tests, indent=2) + '\n').encode()
    manifest = {'files_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in sorted(content.items())},
                'scientific_scope': 'TRAIN-only descriptive, <=96 TRAIN image decodes; frozen text re-encoding; no optimizer',
                'gpu_end_to_end_run_locally': False}
    content['release_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    target = WORKSPACE / 'output/train_knowledge_audit_release_20261010.zip'
    require(not target.exists() and not Path(str(target) + '.sha256').exists(), 'Release exists; preserve it')
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, blob in sorted(content.items()):
            archive.writestr(ROOT.name + '/' + name, blob)
    with zipfile.ZipFile(target) as archive:
        require(archive.testzip() is None, 'Release CRC failure')
        require(len(archive.namelist()) == len(content), 'Release member count')
        for name, digest in manifest['files_sha256'].items():
            require(hashlib.sha256(archive.read(ROOT.name + '/' + name)).hexdigest() == digest, 'Release member SHA')
    Path(str(target) + '.sha256').write_text(sha(target) + '  ' + target.name + '\n', encoding='ascii')
    write_json(ROOT / 'build_receipt.json', {'release': str(target), 'bytes': target.stat().st_size,
                                            'sha256': sha(target), 'members': len(content),
                                            'local_tests': local_tests, 'gpu_end_to_end_run': False})
    print(json.dumps(read_json(ROOT / 'build_receipt.json'), ensure_ascii=False))


if __name__ == '__main__':
    build()
