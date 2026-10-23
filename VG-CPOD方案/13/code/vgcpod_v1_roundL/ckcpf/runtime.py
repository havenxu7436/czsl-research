import contextlib
import csv
import math
import os
import platform
import shlex
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json, sha, digest, load_torch, save_torch, seed_all
from .data import validate_cache, loader, train_mask
from .model import ConditionalCPF
from .evaluation import predict, evaluate, selection_auc

VARIANTS = ('base', 'knowledge', 'shuffle')
SEEDS = (3351, 3352, 3353)


@contextlib.contextmanager
def lock_run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    f = (folder / '.lock').open('a+b')
    try:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    finally:
        f.close()


def source_identity():
    files = sorted((ROOT / 'ckcpf').glob('*.py')) + [ROOT / 'run.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def schedule(opt, epoch, epochs):
    factor = .01 + .99 * (1 + math.cos(math.pi * epoch / max(1, epochs-1))) / 2
    for group in opt.param_groups:
        group['lr'] = group['initial_lr'] * factor


def make_model(seed, variant, cfg, device):
    initial = load_torch(ROOT / f'initial/g0_seed{seed}.pt')
    bank = load_torch(ROOT / ('assets/shuffle.pt' if variant == 'shuffle' else 'assets/knowledge.pt'))
    return ConditionalCPF(initial, bank, variant, cfg, seed).to(device)


def train_one(args, manifest, cfg, variant):
    run = Path(args.work) / f'seed{args.seed}' / variant
    identity = dict(seed=args.seed, variant=variant, config=cfg, source=source_identity(),
                    initial_sha256=sha(ROOT / f'initial/g0_seed{args.seed}.pt'),
                    bank_sha256=sha(ROOT / ('assets/shuffle.pt' if variant == 'shuffle' else 'assets/knowledge.pt')),
                    input_sha256={k: manifest['files_sha256'][k] for k in ['text_features.npz'] +
                        [f'{p}/{n}.npy' for p in ('train','val','test') for n in ('g',)] +
                        [f'{p}_records.json' for p in ('train','val','test')]})
    with lock_run(run):
        if (run / 'identity.json').exists() and read_json(run / 'identity.json') != identity:
            raise ValueError('Existing run uses a different recipe; choose a new --work: ' + str(run))
        write_json(run / 'identity.json', identity)
        if (run / 'DONE.json').exists():
            print('ALREADY_COMPLETE', args.seed, variant, flush=True)
            return
        seed_all(args.seed)
        model = make_model(args.seed, variant, cfg, args.device)
        opt = model.optimizer()
        mask = train_mask(manifest).to(args.device)
        history, start, best = [], 0, -float('inf')
        if (run / 'last.pt').exists():
            state = load_torch(run / 'last.pt')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            history, start, best = state['history'], state['epoch'], state['best_auc']
            print('RESUME', args.seed, variant, 'after_epoch', start, flush=True)
        else:
            p0 = predict(model, args.cache, 'val', cfg, args.seed, args.device)
            write_json(run / 'initial.json', dict(val_cw_auc=selection_auc(p0, manifest),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, cfg['epochs']):
            tic = time.perf_counter()
            schedule(opt, epoch, cfg['epochs'])
            model.train(); total = 0.; count = 0
            for g, a, o in loader(args.cache, 'train', cfg, args.seed, epoch):
                g, a, o = g.to(args.device), a.to(args.device), o.to(args.device)
                loss = model.loss(model(g), a, o, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                opt.step(); total += float(loss.detach())*len(a); count += len(a)
            train_seconds = time.perf_counter()-tic
            val = predict(model, args.cache, 'val', cfg, args.seed, args.device)
            auc = selection_auc(val, manifest)
            record = dict(epoch=epoch+1, loss=total/count, val_cw_auc=auc,
                train_seconds=train_seconds, epoch_seconds=time.perf_counter()-tic,
                alpha=float(model.alpha.detach()) if variant != 'base' else 0.,
                peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if str(args.device).startswith('cuda') else 0.)
            history.append(record)
            if auc > best:
                best = auc
                save_torch(run / 'best.pt', dict(model=model.state_dict(), epoch=epoch+1, val_cw_auc=auc, identity=identity))
            save_torch(run / 'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch+1,
                best_auc=best, history=history, identity=identity))
            write_json(run / 'history.json', history)
            print('EPOCH', args.seed, variant, epoch+1, 'loss', record['loss'], 'VAL_CW_AUC', auc,
                  'alpha', record['alpha'], 'seconds', record['epoch_seconds'], flush=True)
        state = load_torch(run / 'best.pt'); model.load_state_dict(state['model'])
        val_pred = predict(model, args.cache, 'val', cfg, args.seed, args.device)
        val, val_curves = evaluate(val_pred, manifest, 'val')
        test_pred = predict(model, args.cache, 'test', cfg, args.seed, args.device)
        test, test_curves = evaluate(test_pred, manifest, 'test', val)
        for phase, pred in [('val', val_pred), ('test', test_pred)]:
            tmp = run / (phase + '_predictions.writing.npz')
            np.savez_compressed(tmp, **pred)
            os.replace(tmp, run / (phase + '_predictions.npz'))
            # Ordered image IDs and the full [object, attribute] axes reconstruct every score.
            write_json(run / (phase + '_records.json'), read_json(Path(args.cache) / (phase + '_records.json')))
        write_json(run / 'prediction_axes.json', manifest['labels'])
        result = dict(seed=args.seed, variant=variant, selected_epoch=state['epoch'], val=val, test=test,
            initial_val_cw_auc=read_json(run/'initial.json')['val_cw_auc'],
            alpha=float(model.alpha.detach()) if variant != 'base' else 0.,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            recorded_training_seconds=sum(x['epoch_seconds'] for x in history),
            peak_gpu_mb=max(x['peak_gpu_mb'] for x in history))
        write_json(run / 'metrics.json', result)
        write_json(run / 'curves.json', dict(val=val_curves, test=test_curves))
        write_json(run / 'DONE.json', dict(seed=args.seed, variant=variant, selected_epoch=state['epoch'], metrics_sha256=sha(run/'metrics.json')))
        print('COMPLETE', args.seed, variant, 'CW', test['cw']['AUC_native'], 'OW', test['ow']['AUC_native'], flush=True)


def train(args):
    cfg = read_json(args.config)
    if args.epochs is not None: cfg['epochs'] = args.epochs
    if cfg['epochs'] <= 0: raise ValueError('epochs must be positive')
    manifest = validate_cache(args.cache)
    print('DEVICE', args.seed, os.getenv('CUDA_VISIBLE_DEVICES'),
          torch.cuda.get_device_name(0) if str(args.device).startswith('cuda') else args.device, flush=True)
    for variant in VARIANTS if args.variant == 'all' else [args.variant]:
        train_one(args, manifest, cfg, variant)


def report(args):
    root = Path(args.work); root.mkdir(parents=True, exist_ok=True)
    rows, found = [], {}
    for seed in SEEDS:
        for variant in VARIANTS:
            folder = root / f'seed{seed}' / variant
            if not (folder / 'DONE.json').exists(): continue
            result = read_json(folder / 'metrics.json'); found[(seed, variant)] = result
            for phase in ('val', 'test'):
                for world in ('cw', 'ow'):
                    m = result[phase][world]
                    rows.append(dict(seed=seed, variant=variant, phase=phase, world=world,
                        initial_val_cw_auc=result['initial_val_cw_auc'],
                        selected_epoch=result['selected_epoch'], AUC_native=m['AUC_native'], AUC_exact=m['AUC_exact'],
                        HM_val_bias=m['at_val_bias']['HM'], seen_val_bias=m['at_val_bias']['seen'], unseen_val_bias=m['at_val_bias']['unseen'],
                        NLL=m['calibrated']['NLL'], ECE=m['calibrated']['ECE'], gt_attr_unseen=m['gt_object_attr_acc_unseen'],
                        margin_unseen=m['gt_object_margin_unseen'], object_marginal_acc=m['object_marginal_accuracy'],
                        joint_object_acc=m['joint_object_accuracy'], delta_mean_abs=m['delta_mean_abs'], alpha=result['alpha'],
                        trainable_parameters=result['trainable_parameters'], train_seconds=result['recorded_training_seconds'], peak_gpu_mb=result['peak_gpu_mb']))
    if rows:
        with (root / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    comparisons = {}
    for other in ('base', 'shuffle'):
        seeds = [s for s in SEEDS if (s,'knowledge') in found and (s,other) in found]
        comparisons['knowledge_minus_' + other] = {
            w: dict(seeds=seeds, differences=[found[s,'knowledge']['test'][w]['AUC_native']-found[s,other]['test'][w]['AUC_native'] for s in seeds])
            for w in ('cw','ow')}
    aggregates = {}
    for v in VARIANTS:
        aggregates[v] = {}
        for w in ('cw','ow'):
            values = [found[s,v]['test'][w]['AUC_native'] for s in SEEDS if (s,v) in found]
            aggregates[v][w] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None, values=values)
    write_json(root / 'paired_comparisons.json', comparisons)
    write_json(root / 'aggregates.json', aggregates)
    status = dict(completed=len(found), expected=9,
                  missing=[f'seed{s}/{v}' for s in SEEDS for v in VARIANTS if (s,v) not in found])
    write_json(root / 'status.json', status)
    print('COMPLETED', len(found), '/ 9', 'MISSING', status['missing'], flush=True)
    print('SUMMARY', root / 'summary.csv', flush=True)
    return status


def setup(args):
    candidates = [Path(args.cache)] if args.cache else [Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'),
        Path.home()/'experiments/trustcpf_clip_v1/cache/ut-zap50k', Path('/data/xhw/vgcpod_project/cache/ut-zap50k')]
    failures = []
    for cache in candidates:
        if not (cache / 'features_manifest.json').exists(): continue
        try:
            validate_cache(cache)
            break
        except (ValueError, OSError) as e:
            failures.append(str(e))
    else:
        raise FileNotFoundError('G0 cache not found. Pass --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14. ' + '; '.join(failures))
    import sys
    values = dict(CPF_PY=sys.executable, CPF_CACHE=str(cache.resolve()), CPF_WORK=str((ROOT/'work').resolve()))
    (ROOT/'server_env.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in values.items()), encoding='utf-8')
    write_json(ROOT/'setup.json', dict(paths=values, python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count()))
    print('CONFIGURED', values, flush=True)
