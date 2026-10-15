"""Freeze all formula corrections together and explicit one-factor ablations."""
import argparse
import copy
from pathlib import Path
import torch
import yaml
import legacy_prepare
from common import read, write, sha, verify_release
from build_revision_assets import encode_assets, materialize, external_prior

HERE = Path(__file__).absolute().parent
CORE = ['A_native','R0','Rtext','Rprior','Rmvp','Rexternal','Rmvp_logic','Rfeature','Rfull',
        'Rfull_nopair_head','Rfull_shuffle','Rfull_random_prior','Rfull_uniform_prior']
SMALL = ['Rfull_noise10','Rfull_noise30','Rfull_noise50','Rfull_count1','Rfull_count5',
         'Rfull_depth1','Rfull_depth2','Rfull_attention_K','Rfull_clamped_log','Rfull_learn_lambda',
         'Rfull_delayed_beta','Rfull_marginal_attr','Rfull_static_head','Rfull_native_attention',
         'Rfull_prior_off','Rfull_nopair_all','Rfull_descT005','Rfull_descT020','R0_native_attention']
EXTERNAL = ['Rfull_llm_prior','Rfull_mixed_prior']


def variant_options(name):
    opt = dict(text=True, prior=True, feature=True, logic=True, use_pair=True,
               count=3, noise=0, shuffle=False, aggregation='mean', kind='text',
               rev_enabled=True, rev_prior_mode='centered', rev_prior_floor=1e-3,
               rev_gain=0.1, rev_rank=128, rev_object_chunk=32, rev_static_head=False,
               rev_desc_temperature=0.1, rev_anchor=0.1, rev_attr_mode='gt',
               rev_learn_lambda=False, rev_beta_delay=0, rev_beta_ramp=0, rev_oi_depth=3,
               rev_visual_attention='cosine', rev_attention_temperature=0.07)
    matrix = {'A_native': (False,False,False,False), 'R0': (False,False,False,False),
              'Rtext': (True,False,False,False), 'Rprior': (False,True,False,False),
              'Rmvp': (True,True,False,False), 'Rexternal': (False,False,False,True),
              'Rmvp_logic': (True,True,False,True), 'Rfeature': (True,True,True,False)}
    identity = 'R0' if name == 'R0_native_attention' else name
    if identity in matrix:
        opt.update(zip(('text','prior','feature','logic'),matrix[identity]))
    if name == 'A_native': opt['rev_enabled'] = False
    if name in ('Rfull_nopair_head','Rfull_nopair_all'): opt['use_pair'] = False
    if name in ('Rfull_prior_off','Rfull_nopair_all'): opt['prior'] = False
    if '_noise' in name: opt['noise'] = int(name.split('_noise')[1])
    if '_count' in name: opt['count'] = int(name.split('_count')[1])
    if '_depth' in name: opt['rev_oi_depth'] = int(name.split('_depth')[1])
    if name == 'Rfull_shuffle': opt['shuffle'] = True
    if name == 'Rfull_attention_K': opt['aggregation'] = 'attention'
    if name == 'Rfull_clamped_log': opt['rev_prior_mode'] = 'clamped_log'
    if name == 'Rfull_learn_lambda': opt['rev_learn_lambda'] = True
    if name == 'Rfull_delayed_beta': opt.update(rev_beta_delay=2, rev_beta_ramp=3)
    if name == 'Rfull_marginal_attr': opt['rev_attr_mode'] = 'marginal'
    if name == 'Rfull_static_head': opt['rev_static_head'] = True
    if name in ('Rfull_native_attention','R0_native_attention'): opt['rev_visual_attention'] = 'native'
    if name == 'Rfull_descT005': opt['rev_desc_temperature'] = 0.05
    if name == 'Rfull_descT020': opt['rev_desc_temperature'] = 0.20
    if name == 'Rfull_random_prior': opt['kind'] = 'random'
    if name == 'Rfull_uniform_prior': opt['kind'] = 'uniform'
    if name in EXTERNAL: opt['kind'] = 'llm' if name == EXTERNAL[0] else 'mixed'
    if name not in CORE + SMALL + EXTERNAL: raise ValueError('Unknown variant ' + name)
    return opt


def prepare(root, work, epochs, seeds, suite='core', llm=None):
    if not 1 <= epochs < 37 or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('Require distinct nonnegative seeds and epochs 1..36')
    verify_release(HERE)
    root, work = root.resolve(), work.resolve()
    if suite == 'external' and llm is None:
        raise ValueError('External suite requires --llm-prior with actual independently generated scores')
    old = legacy_prepare.prepare(root, work, tier=0)
    write(work/'checks/split_pairs.json', {part:[line.split() for line in
          (root/'data/ut-zap50k/compositional-split-natural'/(part+'_pairs.txt')).read_text().splitlines() if line.strip()]
          for part in ('train','val','test')})
    original = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    encoded = encode_assets(original, root / 'data/glove/glove.6B.300d.txt')
    base_cache = work / 'cache_revision_base'
    base_k = materialize(original, base_cache, encoded)
    labels = read(original / 'labels.json')
    names = CORE if suite == 'core' else ['Rfull'] + SMALL if suite == 'small' else ['Rfull'] + EXTERNAL if suite == 'external' else CORE + SMALL
    cache_paths = {}
    for name in names:
        opt = variant_options(name)
        signature = (opt['count'],opt['noise'],opt['shuffle'],opt['aggregation'],opt['kind'])
        if signature == (3,0,False,'mean','text'):
            cache_paths[name] = base_cache
            continue
        dest = work / ('cache_' + name)
        k = base_k
        if opt['kind'] == 'random': k = torch.rand(base_k.shape, generator=torch.Generator().manual_seed(603354))
        if opt['kind'] == 'uniform': k = torch.ones_like(base_k)
        if opt['kind'] in ('llm','mixed'): k = external_prior(llm, labels, opt['kind'], base_k)
        materialize(original, dest, encoded, opt['count'],opt['noise'],opt['shuffle'],opt['aggregation'],fixed_k=k)
        cache_paths[name] = dest
    base = yaml.safe_load((HERE / 'assets/A_ref.yml').read_text(encoding='utf-8'))
    jobs = []
    for seed in seeds:
        for name in names:
            opt, cfg = variant_options(name), copy.deepcopy(base)
            run_id = name + '_s' + str(seed)
            run = work / 'runs' / run_id
            cfg['DATASET']['root_dir'] = str(root / 'data/ut-zap50k')
            cfg['MODEL'].update(kcc_text=opt['text'],kcc_calib=opt['prior'],kcc_feature=opt['feature'],
                kcc_logic=opt['logic'],kcc_lambda=0.1 if opt['prior'] else 0.0,kcc_beta=0.01 if opt['logic'] else 0.0,
                kcc_cache=str(cache_paths[name]),kcc_control='none',
                bf_logic_bank=str(work/'logic_bank'),bf_allow_reconstructed=True,bf_logic_q=3.0,
                formula_joint=opt['rev_enabled'], kcc_score_mode='0601_formula' if opt['rev_enabled'] else 'legacy',
                full_use_pair=opt['use_pair'], full_gain=1.0)
            cfg['MODEL'].update({key:value for key,value in opt.items() if key.startswith('rev_')})
            cfg['TRAIN'].update(seed=seed,max_epoch=epochs,final_max_epoch=epochs,start_epoch=1,
                kcc_stage='unified_0601_revision_20261013',kcc_variant=run_id,kcc_test_at_end=False,
                train_name='rev_' + run_id,kcc_run_dir=str(run),log_dir=str(run/'logs'),checkpoint_dir=str(run/'checkpoints'))
            path = work / 'configs' / (run_id + '.yml')
            path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=True),encoding='utf-8')
            jobs.append(dict(variant=run_id,base_variant=name,seed=seed,cfg=str(path),cfg_sha256=sha(path),run=str(run),
                             tier=0 if name in CORE[:9] or name == 'Rfull' else 1))
    frozen = old['frozen_inputs']
    frozen[str(work/'checks/split_pairs.json')] = sha(work/'checks/split_pairs.json')
    frozen[str(root / 'data/glove/glove.6B.300d.txt')] = encoded[-1]['glove_sha256']
    for directory in (work/'source',work/'logic_bank') + tuple(set(cache_paths.values())):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts: frozen[str(path)] = sha(path)
    if llm: frozen[str(llm.resolve())] = sha(llm)
    p = dict(release=HERE.name,revision='unified_20261013',root=str(root),source=str(work/'source'),
             dataset='ut-zappos',epochs=epochs,seeds=seeds,suite=suite,jobs=jobs,frozen_inputs=frozen,
             formula='logP=log_softmax(z_object)+log_softmax(U_native+0.1*coupled_delta+lambda*prior,attributes)',
             loss='candidate-restricted pair CE +0.4 object NLL +0.6 conditional/marginal attribute NLL + beta*semantic constraints',
             logic='JS categorical consistency; disjoint child probabilities summed; TRAIN-label descriptor CE anchors0.1; no pair sigmoid',
             knowledge='unified assistant-reconstructed GloVe300 corpus; pair templates retained; not original author knowledge',
             selection='first strict maximum VAL CW AUC; one checkpoint for all four CW/OW VAL/TEST evaluations',
             test_policy='No TEST model/budget/bias selection. Prior TEST results already observed; follow-up is not a fresh unseen holdout claim.',
             budget='Fresh same 10epoch budget across groups, explicit choice not a convergence guarantee; native milestones23/30 inactive at10',
             corrected_together=True, augmentation='excluded; epochs<37',
             not_covered=['Original-author lost corpus / original Troika LogiCZSL score replication',
                          'MIT-States/C-GQA end-to-end cache, CUDA memory and full dataset matrix',
                          'CLIP image+text dual-encoder semantic teacher integration',
                          'Cross-benchmark transfer and human-reviewed qualitative visual failure taxonomy'] +
                         ([] if suite=='external' else ['LLM/mixed feasibility requires actual --llm-prior asset; code provided, not run']))
    write(work/'protocol.json',p)
    print('REVISION_PREPARED unified=True suite={} jobs={} epochs={} seeds={} work={}'.format(suite,len(jobs),epochs,seeds,work),flush=True)
    return p


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--root',type=Path,default=Path('/data/xhw/czsl_repro'))
    a.add_argument('--work',type=Path,required=True)
    a.add_argument('--epochs',type=int,default=10)
    a.add_argument('--seeds',default='3351,3352,3353')
    a.add_argument('--suite',choices=['core','small','all','external'],default='core')
    a.add_argument('--llm-prior',type=Path)
    args = a.parse_args()
    prepare(args.root,args.work,args.epochs,[int(s) for s in args.seeds.split(',')],args.suite,args.llm_prior)
