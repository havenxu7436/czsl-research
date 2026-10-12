"""Freeze UT A-H and prioritized checks, in an isolated new directory."""
import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from common import read, sha, write, verify_release, verify_files

HERE = Path(__file__).resolve().parent
CORE = ['G', 'H', 'C', 'D', 'A', 'B', 'E', 'F']
SANITY = ['H_no_pair', 'H_shuffle', 'H_random_prior', 'H_uniform_prior']
EXTRA = ['H_noise10', 'H_noise30', 'H_noise50']


def verify_cache(cache):
    manifest = read(cache / 'manifest.json')
    if manifest['pair_axis_order'] != ['object', 'attribute'] or manifest['dimension'] != 300:
        raise ValueError('Cache axes/dimension mismatch')
    for name, fingerprint in manifest['files_sha256'].items():
        if sha(cache / name) != fingerprint:
            raise ValueError('Cache tensor changed: ' + name)
    if len(manifest['objs']) != 12 or len(manifest['attrs']) != 16:
        raise ValueError('First release is UT only')
    return manifest


def derangement(n, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(1000):
        p = torch.randperm(n, generator=g)
        if bool((p != torch.arange(n)).all()):
            return p
    raise ValueError('Unable to make a derangement')


def altered_cache(original, dest, kind, builder, glove):
    manifest = verify_cache(original)
    dictionaries = {}
    inputs = {}
    for which in ('oe', 'ae'):
        matches = [(Path(p), h) for p, h in manifest['input_sha256'].items() if p.endswith('/' + which + '_dict.json')]
        if len(matches) != 1 or sha(matches[0][0]) != matches[0][1]:
            raise ValueError('Missing/changed frozen description input: ' + which)
        inputs[str(matches[0][0])] = matches[0][1]
        dictionaries[which] = read(matches[0][0])
    needed = {t for d in dictionaries.values() for texts in d.values() for s in texts for t in builder.description_tokens(s)}
    vectors, fingerprint = builder.load_needed_glove(glove, needed)
    if fingerprint != manifest['input_sha256'].get(str(glove)):
        raise ValueError('GloVe bytes changed')
    shutil.copytree(original, dest)
    permutations = {'oe': derangement(12, 603352), 'ae': derangement(16, 603353)}
    masks = {}
    for which, labels, filename in (('oe', manifest['objs'], 'W_o_K.pt'), ('ae', manifest['attrs'], 'W_a_K.pt')):
        original_texts = [dictionaries[which][label] for label in labels]
        if kind == 'shuffle':
            texts = [original_texts[i] for i in permutations[which].tolist()]
        else:
            fraction = int(kind.replace('noise', '')) / 100
            slots = sum(len(t) for t in original_texts)
            g = torch.Generator().manual_seed(603360 + len(labels))
            removed = set(torch.randperm(slots, generator=g)[:round(slots * fraction)].tolist())
            texts, offset = [], 0
            for group in original_texts:
                texts.append([s for j, s in enumerate(group) if offset + j not in removed])
                offset += len(group)
            masks[which] = sorted(removed)
        name_file = 'W_o_name_raw.pt' if which == 'oe' else 'W_a_name_raw.pt'
        names = torch.load(original / name_file, weights_only=True)
        table = []
        for name, group in zip(names, texts):
            values = [F.normalize(name, dim=0)]
            for sentence in group:
                kept = [vectors[t] for t in builder.description_tokens(sentence) if t in vectors]
                value = torch.stack(kept).mean(0) if kept else name
                values.append(F.normalize(value, dim=0))
            table.append(F.normalize(torch.stack(values).mean(0), dim=0))
        torch.save(torch.stack(table), dest / filename)
        manifest['files_sha256'][filename] = sha(dest / filename)
    manifest.update(knowledge_status='0601_control_' + kind,
                    control={'kind': kind, 'parent_manifest_sha256': sha(original / 'manifest.json'),
                             'names_preserved': True, 'pair_templates_preserved': True,
                             'prior_preserved': True,
                             'permutations': {k: p.tolist() for k, p in permutations.items()} if kind == 'shuffle' else {},
                             'deleted_description_slots': masks,
                             'scope': 'text inputs and feature primitive K; logic descriptions also shuffled for shuffle; noise leaves logic unchanged'})
    write(dest / 'manifest.json', manifest)
    write(dest / 'cache_manifest.json', manifest)
    return inputs


def prepare(root, work, tier):
    verify_release(HERE)
    root, work = root.resolve(), work.resolve()
    if root == work or root in work.parents or work in root.parents:
        raise ValueError('New work directory must be separate from the old project')
    if work.exists():
        raise FileExistsError('Work exists: inspect/status or use runner --resume; never overwrite ' + str(work))
    cache = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    expected_cache = HERE / 'assets/legacy_cache_manifest.json'
    if sha(cache / 'manifest.json') != sha(expected_cache):
        raise ValueError('UT cache version differs from audited BF inputs')
    manifest = verify_cache(cache)
    original_hashes = read(HERE / 'assets/source_origin.json')
    for name in ('models/ddm.py', 'models/word_embedding_utils.py', 'models/ddm_utils.py', 'dataset.py', 'evaluator_ge.py', 'config/defaults.py'):
        if sha(root / 'src/KCC0601_v1' / name) != original_hashes[name]:
            raise ValueError('CPF baseline version changed: ' + name)
    data = root / 'data/ut-zap50k'
    frozen = {str(cache / 'manifest.json'): sha(cache / 'manifest.json')}
    frozen.update({str(cache / name): h for name, h in manifest['files_sha256'].items()})
    for part in ('train', 'val', 'test'):
        path = data / 'compositional-split-natural' / (part + '_pairs.txt')
        expected = manifest['input_sha256'].get(str(path))
        if expected is None or sha(path) != expected:
            raise ValueError('Split changed: ' + str(path))
        frozen[str(path)] = expected
    metadata = data / 'metadata_compositional-split-natural.t7'
    if sha(metadata) != '7fbe052a107170ad28f5b85d2237e584c7ea306ca791844e691606e89635aa5b':
        raise ValueError('UT metadata differs from audited BF dataset')
    frozen[str(metadata)] = sha(metadata)
    work.mkdir(parents=True)
    shutil.copytree(HERE / 'source', work / 'source')
    shutil.copytree(HERE / 'assets/logic_bank', work / 'logic_bank')
    if sha(work / 'logic_bank/bank.pt') != read(work / 'logic_bank/manifest.json')['bank_sha256']:
        raise ValueError('Logic bank differs from audited bank')
    variants = CORE + (SANITY if tier >= 1 else []) + (EXTRA if tier >= 2 else [])
    cache_by_variant, bank_by_variant = {}, {}
    if tier >= 1:
        p = work / 'source/tools/build_knowledge.py'
        spec = importlib.util.spec_from_file_location('cache_builder', p)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        frozen.update(altered_cache(cache, work / 'cache_shuffle', 'shuffle', builder, root / 'data/glove/glove.6B.300d.txt'))
        cache_by_variant['H_shuffle'] = work / 'cache_shuffle'
        shutil.copytree(work / 'logic_bank', work / 'logic_bank_shuffle')
        bank = torch.load(work / 'logic_bank/bank.pt', weights_only=True)
        bank['oe'] = bank['oe'][derangement(12, 603352)]
        bank['ae'] = bank['ae'][derangement(16, 603353)]
        torch.save(bank, work / 'logic_bank_shuffle/bank.pt')
        bm = read(work / 'logic_bank/manifest.json')
        bm['bank_sha256'] = sha(work / 'logic_bank_shuffle/bank.pt')
        bm['control'] = 'Same primitive derangements as text inputs; OI/labels/names unchanged'
        write(work / 'logic_bank_shuffle/manifest.json', bm)
        bank_by_variant['H_shuffle'] = work / 'logic_bank_shuffle'
        for kind in ('random', 'uniform'):
            dest = work / ('cache_' + kind)
            shutil.copytree(cache, dest)
            k = torch.rand(12, 16, generator=torch.Generator().manual_seed(603354)) if kind == 'random' else torch.ones(12, 16)
            prior = builder.centered_prior(k)
            torch.save(k, dest / 'K_oa_text.pt')
            torch.save(prior, dest / 'K_oa_prior_centered.pt')
            cm = copy.deepcopy(manifest)
            for name in ('K_oa_text.pt', 'K_oa_prior_centered.pt'):
                cm['files_sha256'][name] = sha(dest / name)
            cm['knowledge_status'] = '0601_' + kind + '_prior'
            cm['control'] = {'kind': kind, 'seed': 603354, 'other_embeddings_unchanged': True}
            write(dest / 'manifest.json', cm)
            write(dest / 'cache_manifest.json', cm)
            cache_by_variant['H_' + kind + '_prior'] = dest
    if tier >= 2:
        for level in (10, 30, 50):
            dest = work / ('cache_noise' + str(level))
            frozen.update(altered_cache(cache, dest, 'noise' + str(level), builder, root / 'data/glove/glove.6B.300d.txt'))
            cache_by_variant['H_noise' + str(level)] = dest
    base = yaml.safe_load((HERE / 'assets/A_ref.yml').read_text(encoding='utf-8'))
    jobs = []
    for variant in variants:
        identity = variant if len(variant) == 1 else 'H'
        text = identity in 'CEFGH'
        calib = identity in 'DEFGH'
        logic = identity in 'BFH'
        feature = identity in 'GH'
        config = copy.deepcopy(base)
        run = work / 'runs' / variant
        config['DATASET']['root_dir'] = str(data)
        config['MODEL'].update(kcc_text=text, kcc_calib=calib, kcc_feature=feature,
                               kcc_logic=logic, kcc_lambda=0.1 if calib else 0.0,
                               kcc_beta=0.01 if logic else 0.0, kcc_control='none',
                               kcc_cache=str(cache_by_variant.get(variant, cache)),
                               bf_logic_bank=str(bank_by_variant.get(variant, work / 'logic_bank')),
                               bf_allow_reconstructed=True, bf_logic_q=3.0,
                               full_use_pair=variant != 'H_no_pair', full_gain=1.0)
        config['TRAIN'].update(seed=3351, max_epoch=10, final_max_epoch=10, start_epoch=1,
                               kcc_stage='0601_full_reproduction_20261010', kcc_variant=variant,
                               kcc_test_at_end=False, train_name='0601_' + variant,
                               kcc_run_dir=str(run), log_dir=str(run / 'logs'), checkpoint_dir=str(run / 'checkpoints'))
        path = work / 'configs' / (variant + '.yml')
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding='utf-8')
        jobs.append({'variant': variant, 'tier': 0 if variant in CORE else 1 if variant in SANITY else 2,
                     'cfg': str(path), 'run': str(run), 'cfg_sha256': sha(path)})
    for directory in [work / 'source', work / 'logic_bank'] + list(set(cache_by_variant.values())) + list(set(bank_by_variant.values())):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                frozen[str(path)] = sha(path)
    protocol = {'release': HERE.name, 'dataset': 'ut-zappos', 'seed': 3351, 'epochs': 10,
                'jobs': jobs, 'frozen_inputs': frozen, 'source': str(work / 'source'), 'root': str(root),
                'tier': tier, 'selection': 'first strict maximum VAL CW AUC, same checkpoint for CW/OW and TEST',
                'test_policy': 'config fixed before run; TEST never selects checkpoints or hyperparameters',
                'feature': 'section 6.1 hidden512, Bx300 shared visual branch, add factorized K score to native composition score',
                'loss': 'native composition CE + 0.4 object CE + 0.6 GT-object conditional attribute CE; beta0.01 logic for H',
                'logic_adapter': 'B/F native primitive attr; H actual train pair-score logsumexp over objects',
                'knowledge': 'audited legacy text/prior + reconstructed BF logic; template-only pair knowledge',
                'backbone': 'native ViT B16; finetune_backbone=True as original BF configs',
                'sanity': 'H template-only is the main run; H shuffle/no-pair/random/uniform are fresh 10-epoch runs',
                'extra_limit': 'noise affects text/feature primitive descriptions only; prior, templates, and logic kept fixed',
                'not_covered': ['MIT/C-GQA full A-H', 'multi-seed', 'description counts 1/3/5', 'OI depths 1/2/3',
                                'LLM-only/mixed feasibility (no valid LLM scores)', 'optional cross-benchmark transfer'],
                'historical_results': 'Not copied into this paired main table. All eight groups retrained.'}
    write(work / 'protocol.json', protocol)
    print('PREPARED_0601', len(jobs), 'jobs', work, flush=True)
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/xhw/czsl_repro'))
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--tier', type=int, choices=[0, 1, 2], default=1)
    a = p.parse_args()
    prepare(a.root, a.work, a.tier)
