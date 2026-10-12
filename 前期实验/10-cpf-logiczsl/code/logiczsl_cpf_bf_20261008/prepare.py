#!/usr/bin/env python3
"""Prepare UT-Zappos A_ref/B/E_ref/F without modifying legacy runs."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from common import sha, write_new, write_json
from build_bank import build
from install import install

BUNDLE = Path(__file__).resolve().parent


def verify_install(source):
    record = json.loads((source / 'bf_install.json').read_text(encoding='utf-8'))
    for name, digest in record['output_hashes'].items():
        if sha(source / name) != digest:
            raise ValueError('Installed source modified: ' + name)


def prepare(root, work, seed, beta=0.01, allow_reconstructed=False):
    if not allow_reconstructed:
        raise ValueError('Read/review assets first, then explicitly use --allow-reconstructed')
    if beta != 0.01:
        raise ValueError('This frozen first-round protocol uses beta=0.01; new beta needs a separately versioned experiment')
    root, work = Path(root).resolve(), Path(work).resolve()
    if work == root or root in work.parents or work in root.parents:
        raise ValueError('Use a separate work directory outside the existing project')
    source = work / 'src'
    if source.exists():
        verify_install(source)
    else:
        install(root / 'src/KCC0601_v1', source)
    legacy = root / 'experiments/kcc0601_glove_mvp_v1'
    refs = {
        'A': legacy / 'configs/mvp_ut-zappos_A_legacy_s3351_l0.yml',
        'E': legacy / 'configs/mvp_ut-zappos_E_legacy_s3351_l0.1.yml'
    }
    reference = {name: yaml.safe_load(path.read_text(encoding='utf-8')) for name, path in refs.items()}
    cache = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    labels_path = cache / 'labels.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    if len(labels['objs']) != 12 or len(labels['attrs']) != 16:
        raise ValueError('This release is UT-Zappos only')
    knowledge = BUNDLE / 'assets/ut-zap50k/knowledge.json'
    bank_dir = work / 'logic_bank/ut-zap50k'
    glove = root / 'data/glove/glove.6B.300d.txt'
    if bank_dir.exists():
        manifest = json.loads((bank_dir / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['knowledge_sha256'] != sha(knowledge) or manifest['labels_sha256'] != sha(labels_path):
            raise ValueError('Existing bank has different knowledge/labels; use a new work directory')
        if manifest['bank_sha256'] != sha(bank_dir / 'bank.pt'):
            raise ValueError('Existing logic bank changed')
    else:
        build(knowledge, labels_path, glove, bank_dir)
    configs = {}
    for variant, anchor in (('A_ref', 'A'), ('B', 'A'), ('E_ref', 'E'), ('F', 'E')):
        c = copy.deepcopy(reference[anchor])
        if c['MODEL']['wordembs'] != 'glove' or c['MODEL']['cosine_cls_temp'] != 0.05:
            raise ValueError('Requires audited native GloVe / temperature 0.05')
        if c['TRAIN']['max_epoch'] != 10 or c['TRAIN']['aug_epoch'] <= 10:
            raise ValueError('Ten native epochs, without augmentation, required')
        if c['MODEL']['kcc_text'] != (anchor == 'E') or c['MODEL']['kcc_calib'] != (anchor == 'E'):
            raise ValueError('Incorrect legacy reference flags')
        expected_lambda = 0.1 if anchor == 'E' else 0.0
        if c['MODEL']['kcc_lambda'] != expected_lambda:
            raise ValueError('Do not silently redefine F with lambda=0')
        enabled = variant in ('B', 'F')
        rid = f'bf_ut-zappos_{variant}_s{seed}_b{beta if enabled else 0:g}_q3_l{expected_lambda:g}'
        run = work / 'experiments/runs' / rid
        c['DATASET']['root_dir'] = str(root / 'data/ut-zap50k')
        c['EVAL'].update(open_world=False, topk=1)
        c['MODEL'].update(kcc_logic=enabled, kcc_beta=beta if enabled else 0.0,
                          kcc_feature=False, kcc_control='none', kcc_cache=str(cache),
                          bf_logic_bank=str(bank_dir), bf_logic_q=3.0, bf_allow_reconstructed=True)
        c['TRAIN'].update(seed=seed, start_epoch=1, max_epoch=10, final_max_epoch=10,
                          train_name=rid, kcc_run_dir=str(run), kcc_variant=variant,
                          kcc_stage='paper_method_reconstructed_BF', kcc_test_at_end=False,
                          log_dir=str(run / 'logs'), checkpoint_dir=str(run / 'checkpoints'))
        path = work / 'experiments/configs' / f'{variant}_s{seed}.yml'
        content = yaml.safe_dump(c, allow_unicode=True, sort_keys=True)
        if path.exists():
            if path.read_text(encoding='utf-8') != content:
                raise FileExistsError('Config already exists with different values: ' + str(path))
        else:
            write_new(path, content)
        configs[variant] = {'path': str(path), 'sha256': sha(path), 'run': str(run)}
    split_dir = root / 'data/ut-zap50k/compositional-split-natural'
    protocol = {
        'release': BUNDLE.name, 'dataset': 'ut-zappos', 'seed': seed, 'configs': configs,
        'source': str(source), 'install_sha256': sha(source / 'bf_install.json'),
        'logic_bank_manifest_sha256': sha(bank_dir / 'manifest.json'),
        'legacy_cache_manifest_sha256': sha(cache / 'cache_manifest.json'),
        'legacy_tensor_manifest_sha256': sha(cache / 'manifest.json'),
        'reference_configs_sha256': {k: sha(v) for k, v in refs.items()},
        'split_sha256': {str(split_dir / (s + '_pairs.txt')): sha(split_dir / (s + '_pairs.txt'))
                         for s in ('train', 'val', 'test')},
        'metadata_sha256': {str(root / 'data/ut-zap50k/metadata_compositional-split-natural.t7'):
                            sha(root / 'data/ut-zap50k/metadata_compositional-split-natural.t7')},
        'epochs': 10, 'q': 3.0, 'beta': beta, 'selection': 'first strict maximum VAL CW AUC',
        'formal_paper_result_reproduction': False, 'test_evaluation': 'not provided in this development release',
        'knowledge_policy': 'same reconstructed logic bank for B/F; old E text/calibration cache unchanged',
        'rng_policy': 'Python/NumPy/Torch seed; cudnn benchmark=False; strict bitwise GPU determinism not guaranteed'
    }
    target = work / 'experiments/protocol' / f'seed_{seed}.json'
    content = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
    if target.exists():
        if target.read_text(encoding='utf-8') != content:
            raise ValueError('Protocol changed; use a new version')
    else:
        write_new(target, content)
    print('PREPARED', target)
    for variant, spec in configs.items():
        print(variant, spec['path'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/xhw/czsl_repro'))
    p.add_argument('--work', type=Path, default=Path('/var/tmp/czsl_xhw_bf_20261008'))
    p.add_argument('--seed', type=int, default=3351)
    p.add_argument('--allow-reconstructed', action='store_true')
    a = p.parse_args()
    prepare(a.root, a.work, a.seed, allow_reconstructed=a.allow_reconstructed)
