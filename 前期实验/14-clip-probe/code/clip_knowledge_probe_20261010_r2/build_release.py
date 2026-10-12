"""Build the frozen release from already-reviewed local artifacts."""
import ast
import json
import shutil
import sys
import zipfile
from pathlib import Path
from common import ROOT, sha, read_json, write_json, require


WORKSPACE = ROOT.parent
SHA_MODEL = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
COMMIT = 'a1d071733d7111c9c014f024669f959182114e33'


def build(prepare_only=False):
    reference = read_json(WORKSPACE / 'generic_mixture_controls_20261009/legacy/reference.json')
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    for name in ('ae_dict.json', 'oe_dict.json'):
        shutil.copyfile(WORKSPACE / 'ut_semantic_v2_20261006/knowledge/ut-zap50k' / name, assets / name)
    shutil.copyfile(WORKSPACE / 'ut_train_path_aliases_20261006.json', assets / 'path_aliases.json')
    wheels = ROOT / 'vendor/wheels'
    wheels.mkdir(exist_ok=True)
    wheel_sources = list((WORKSPACE / 'tmp/clip_probe_linux38_wheels').glob('*.whl'))
    require(len(wheel_sources) == 5, 'Five pinned Linux Python3.8 dependency wheels required')
    for p in wheel_sources:
        shutil.copyfile(p, wheels / p.name)
    bank_manifest = read_json(WORKSPACE / 'output/BF_conditional_preflight_20261009/extracted/conditional/bank/manifest.json')
    for name in ('ae_dict.json', 'oe_dict.json'):
        require(sha(assets / name) == bank_manifest['knowledge_inputs'][name], 'Reviewed description mismatch')
    data = '/data/xhw/czsl_repro/data/ut-zap50k'
    server_files = {p: h for p, h in reference['c_files'].items() if
                    p.endswith('metadata_compositional-split-natural.t7') or
                    p.endswith('/train_pairs.txt') or p.endswith('/val_pairs.txt') or
                    p == reference['bank'] + '/manifest.json'}
    for name in ('train.pt', 'val.pt', 'train_image_ids.json', 'val_image_ids.json', 'labels.json', 'labels_ow.npz'):
        p = reference['stageA_base'] + '/' + name
        server_files[p] = reference['stageA_files'][p]
    server_files[reference['bank'] + '/manifest.json'] = sha(
        WORKSPACE / 'output/BF_conditional_preflight_20261009/extracted/conditional/bank/manifest.json')
    labels = reference['labels']
    require(labels['attrs'] == sorted(labels['attrs']) and labels['objs'] == sorted(labels['objs']), 'Unexpected order')
    protocol = {'schema': 'frozen_clip_knowledge_probe_v2', 'data': data, 'stageA_base': reference['stageA_base'],
                'work_basename': 'czsl_clip_knowledge_probe_20261010_r2',
                'split_expectations': {'train_pairs': 83, 'val_pairs': 30, 'val_seen_pairs': 15,
                                       'val_unseen_pairs': 15, 'train_images': 22998, 'val_images': 3214,
                                       'val_seen_images': 877, 'val_unseen_images': 2337},
                'bank': reference['bank'], 'labels': labels, 'temperature': 0.05,
                'names': {'a': ['canvas', 'cotton', 'faux fur', 'faux leather', 'full grain leather', 'calf hair',
                               'leather', 'nubuck', 'nylon', 'patent leather', 'rubber', 'satin', 'sheepskin',
                               'suede', 'synthetic material', 'wool'],
                          'o': ['ankle boots', 'knee high boots', 'mid calf boots', 'sandals', 'boat shoes',
                                'clogs and mules', 'flats', 'heels', 'loafers', 'oxfords', 'sneakers', 'slippers']},
                'permutations': {k: reference['control_protocol']['permutations'][k]
                                 for k in ('P_K', 'P_perm1', 'P_perm2')},
                'asset_sha256': {name: sha(assets / name) for name in ('ae_dict.json', 'oe_dict.json', 'path_aliases.json')},
                'server_files_sha256': server_files,
                'clip_source_commit': COMMIT, 'clip_model': 'ViT-L/14', 'image_resolution': 224,
                'clip_checkpoint_sha256': SHA_MODEL,
                'clip_checkpoint_url': 'https://openaipublic.azureedge.net/clip/models/' + SHA_MODEL + '/ViT-L-14.pt',
                'aggregation': 'mean of nine unit-normalized text cosine scores; no pooled renormalization',
                'primary_family': 'T_K OE+AE', 'secondary_family': 'T_AE attribute-only, predeclared',
                'temperature_fit': False, 'prompt_search': False, 'optimizer_steps': 0,
                'test_images_or_scores_allowed': False, 'val_role': 'development screen, not final test',
                'generic_reference_preserved': {'architecture_parameters': 60864, 'readout_alpha': 0.25,
                                               'backbone_update_this_probe': False,
                                               'formal_teacher_training_in_this_release': False}}
    write_json(ROOT / 'protocol.json', protocol)
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 8))
    if prepare_only:
        print('ASSETS_AND_PROTOCOL_PREPARED')
        return
    tests = read_json(ROOT / 'local_test_report.json')
    require(tests['passed'] and tests['tests_run'] >= 20 and tests['real_split_preflight_passed'],
            'R2 real split regression suite must pass before packaging')
    files = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
             for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
             p.suffix != '.pyc' and p.name != 'release_manifest.json'}
    write_json(ROOT / 'release_manifest.json', {'schema': 'clip_probe_release_v2', 'files_sha256': files,
                                             'gpu_execution_tested_locally': False,
                                             'official_clip_source': 'https://github.com/openai/CLIP/tree/' + COMMIT})
    archive = WORKSPACE / 'output/clip_knowledge_probe_release_20261010_r2.zip'
    require(not archive.exists(), 'Release exists; do not silently replace a delivered package')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(list(files) + ['release_manifest.json']):
            z.write(ROOT / name, ROOT.name + '/' + name)
    Path(str(archive) + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii')
    print(json.dumps({'release': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'python38_syntax_verified': True}))


if __name__ == '__main__':
    build('--prepare-only' in sys.argv)
