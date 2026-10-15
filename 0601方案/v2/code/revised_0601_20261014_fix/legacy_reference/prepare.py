"""Freeze a fresh formula protocol. Never edit or resume old residual runs."""
import argparse
import copy
import shutil
from pathlib import Path
import torch
import yaml
from common import read, sha, write, verify_release
import legacy_prepare

HERE = Path(__file__).resolve().parent
CORE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
CONTROLS = ['H_no_pair', 'H_shuffle', 'H_random_prior', 'H_uniform_prior']
INTERFACE = ['J0', 'Jtext']


def prior(k):
    return (k - k.mean()) / (k.std() + 1e-8)


def update_tensor(cache, name, value):
    torch.save(value, cache / name)
    m = read(cache / 'manifest.json')
    m['files_sha256'][name] = sha(cache / name)
    m['knowledge_status'] = '0601_formula_reconstruction_20261011'
    m['formula_revision'] = 'minmax+1e-8; default sample std+1e-8 centered; raw files retained'
    write(cache / 'manifest.json', m)
    write(cache / 'cache_manifest.json', m)


def prepare(root, work, epochs, seeds):
    if epochs < 1 or epochs >= 37:
        raise ValueError('Choose 1..36 epochs. This protocol does not cover augmentation >=37')
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Distinct nonnegative seeds required')
    verify_release(HERE)
    root, work = root.resolve(), work.resolve()
    old = legacy_prepare.prepare(root, work, tier=1)
    original = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    main = work / 'cache_formula'
    shutil.copytree(original, main)
    eo = torch.load(main / 'W_o_K.pt', weights_only=True)
    ea = torch.load(main / 'W_a_K.pt', weights_only=True)
    ep = torch.stack([torch.load(main / ('W_ao_K/obj_{:05d}.pt'.format(i)), weights_only=True)
                      for i in range(len(eo))])
    proto = torch.nn.functional.normalize((eo[:, None] + ea[None, :]) / 2, dim=-1)
    raw = torch.nn.functional.cosine_similarity(ep, proto, dim=-1)
    k = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    update_tensor(main, 'K_oa_text.pt', k)
    update_tensor(main, 'K_oa_prior_centered.pt', prior(k))
    # Shuffled-description control holds prior and pair-template bytes fixed.
    update_tensor(work / 'cache_shuffle', 'K_oa_text.pt', k)
    update_tensor(work / 'cache_shuffle', 'K_oa_prior_centered.pt', prior(k))
    for kind in ('random', 'uniform'):
        cache = work / ('cache_' + kind)
        control_k = torch.load(cache / 'K_oa_text.pt', weights_only=True)
        update_tensor(cache, 'K_oa_prior_centered.pt', prior(control_k))
    jobs = []
    by_name = {j['variant']: j for j in old['jobs']}
    # First get one complete core/interface matrix, then replicate it. Controls
    # wait for ALL core jobs to complete; no dependence on positive performance.
    for seed in seeds:
        for base_variant in ['A', 'G', 'H', 'E', 'B', 'F', 'C', 'D'] + INTERFACE + CONTROLS:
            source_job = by_name.get(base_variant, by_name['E' if base_variant == 'Jtext' else 'D'])
            c = yaml.safe_load(Path(source_job['cfg']).read_text(encoding='utf-8'))
            run_id = base_variant + '_s' + str(seed)
            run = work / 'runs' / run_id
            cache = Path(c['MODEL']['kcc_cache'])
            if cache == original:
                cache = main
            c['MODEL']['kcc_cache'] = str(cache)
            c['MODEL']['formula_joint'] = base_variant not in ('A', 'B', 'C')
            c['MODEL']['kcc_score_mode'] = '0601_formula' if c['MODEL']['formula_joint'] else 'legacy'
            if base_variant in INTERFACE:
                c['MODEL'].update(kcc_text=base_variant == 'Jtext', kcc_calib=False,
                                  kcc_lambda=0.0, kcc_feature=False, kcc_logic=False, kcc_beta=0.0)
            c['TRAIN'].update(seed=seed, max_epoch=epochs, final_max_epoch=epochs,
                              start_epoch=1, kcc_stage='0601_formula_reconstruction_20261011',
                              kcc_variant=run_id, kcc_test_at_end=False,
                              train_name='formula_' + run_id, kcc_run_dir=str(run),
                              log_dir=str(run / 'logs'), checkpoint_dir=str(run / 'checkpoints'))
            path = work / 'configs' / (run_id + '.yml')
            path.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=True), encoding='utf-8')
            jobs.append({'variant': run_id, 'base_variant': base_variant, 'seed': seed,
                         'tier': 1 if base_variant in CONTROLS else 0,
                         'cfg': str(path), 'cfg_sha256': sha(path), 'run': str(run)})
    frozen = old['frozen_inputs']
    # Control manifests have been deliberately regenerated before freezing.
    for directory in (work / 'source', main, work / 'cache_shuffle', work / 'cache_random', work / 'cache_uniform'):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                frozen[str(path)] = sha(path)
    protocol = {
        'release': HERE.name, 'dataset': 'ut-zappos', 'epochs': epochs, 'seeds': seeds,
        'jobs': jobs, 'frozen_inputs': frozen, 'source': str(work / 'source'), 'root': str(root),
        'selection': 'first strict maximum VAL CW AUC; one checkpoint for all VAL/TEST CW/OW evaluations',
        'test_policy': 'No TEST tuning. Prior TEST results already observed: this is a declared follow-up, not an unseen holdout claim.',
        'formula': 'G/H: score=z_obj+head_raw+lambda*prior; attribute CE=head_raw[GT_object]; NO native composition/attribute residual',
        'temperature': 'z_obj=native cosine/T, T=0.05 BOTH train and infer; section6 head raw, T_head=1; MVP native comp/attr cosine/T BOTH train and infer',
        'mvp_interface': 'D/E/F/J0/Jtext conditional=full-cartesian native CPF composition logit+broadcast attribute logit. Original section5 does not define this function; interface choice, not a uniquely prescribed formula.',
        'native_baseline': 'A/C keep original CPF task losses and original CPF inference; B keeps native CPF task CE and inference, replaces only logic grounding',
        'logic_adapter': 'object softmax; attribute softmax(logsumexp masked joint over objects); pair sigmoid; descriptions sigmoid(projected_global_CLS @ frozen_GloVe_desc.T); parent similarity softmax',
        'description_interface': 'native objc_embedder projects raw ViT global CLS768 to GloVe300; one shared global vector for OE and AE, no added parameter, no cosine/temperature/class-slot softmax',
        'logic_operators': 'Existing reconstructed OI/OE/AE p-mean violation operators q=3; probabilities consumed directly. Pair node exposed, no additional unspecified fourth loss.',
        'head': 'section6.1 hidden512; native shared Bx300 visual vectors broadcast to primitives; head init seed=600000+run_seed within fork_rng',
        'loss': 'composition CE+0.6 GT-conditioned attribute CE+0.4 object CE+0.01 logic (once); B native attribute CE',
        'knowledge': 'mean aggregation, reconstructed GloVe300 descriptions/hierarchy; all-cartesian template-only pairs; text-only prior; no recovered author assets or CLIP-description claim',
        'controls': 'four fresh H controls for EVERY seed; H itself covers template-only; J0/Jtext isolate MVP scoring-interface change from calibration/text comparisons',
        'training_budget': 'epochs chosen explicitly, NOT specified in 0601; 30 recommended as a larger fixed budget than the previous10; original milestone23/30 schedule retained (decay runs after each epoch); all groups freshly trained under same budget; not a convergence guarantee',
        'backbone': 'original local CPF ddm.py ViT B16, fully fine-tuned; do not replace simplified section2 recap with a new backbone',
        'augmentation': 'epochs<37; minority augmentation not covered, never silently allowed to bypass formula',
        'not_covered': ['MIT-States/C-GQA matrix', 'author original lost knowledge assets', 'LLM-only/mixed feasibility',
                        'CLIP description encoder', 'mean vs attention', 'centered vs clamped-log training comparison',
                        'learnable lambda', 'delayed beta schedule', 'marginalized attribute CE comparison',
                        'description count1/3/5 and OI depth1/2/3', 'noise10/30/50 under new formula',
                        'qualitative failure taxonomy and optional cross-benchmark transfer'],
        'historical_results': 'Never imported as main-table results; old10epoch residual runs remain untouched.',
    }
    write(work / 'protocol.json', protocol)
    write(work / 'checks/prior_formula.json', {
        'centered_default_sample_std': True, 'epsilon': 1e-8,
        'constant_centered_max_abs': float(prior(torch.ones_like(k)).abs().max()),
        'sample_std_of_centered': float(prior(k).std()),
        'main_cache': str(main), 'parent_manifest_sha256': sha(original / 'manifest.json')})
    print('FORMULA_PREPARED jobs={} epochs={} seeds={} work={}'.format(len(jobs), epochs, seeds, work), flush=True)
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/xhw/czsl_repro'))
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seeds', default='3351,3352,3353')
    a = p.parse_args()
    prepare(a.root, a.work, a.epochs, [int(s) for s in a.seeds.split(',')])
