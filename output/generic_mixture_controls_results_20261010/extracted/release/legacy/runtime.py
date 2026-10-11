"""Cache-only head learning, native evaluation, and deterministic checkpoint state."""
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from common import ROOT, finite, read_json, seed_all, sha, state_sha, tensor_save, verify_files, write_json
from head import ConditionalHead, TEMPERATURE, VARIANTS, conditional_loss, corrected_scores, probability_checks

FIELDS = ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen')


def make_dataset(reference, setting, split):
    labels = reference['labels']
    attrs, objs = labels['attrs'], labels['objs']
    pairs = (sorted(set(split['train'] + split['val'] + split['test'])) if setting == 'cw'
             else [(a, o) for a in attrs for o in objs])
    return SimpleNamespace(attrs=attrs, objs=objs, pairs=pairs, train_pairs=split['train'],
                           val_pairs=split['val'], test_pairs=split['test'],
                           attr2idx={a: i for i, a in enumerate(attrs)}, obj2idx={o: i for i, o in enumerate(objs)},
                           pair2idx={pair: i for i, pair in enumerate(pairs)}, phase='val', open_world=setting == 'ow')


def native_metrics(grid, attrs, objects, ds, evaluator_class):
    """Full [N,O,A] distribution first, unchanged native candidate order/masks second."""
    ai = torch.tensor([ds.attr2idx[a] for a, o in ds.pairs])
    oi = torch.tensor([ds.obj2idx[o] for a, o in ds.pairs])
    matrix = grid.detach().cpu()[:, oi, ai].contiguous()
    prediction = {pair: matrix[:, i] for i, pair in enumerate(ds.pairs)}
    gt_pair = torch.tensor([ds.pair2idx[(ds.attrs[int(a)], ds.objs[int(o)])] for a, o in zip(attrs, objects)])
    evaluator = evaluator_class(ds, None)
    result = evaluator.score_model(prediction, objects, bias=1e3, topk=1)
    stats = evaluator.evaluate_predictions(result, attrs, objects, gt_pair, prediction, topk=1)
    metrics = {k: float(v) for k, v in stats.items()}
    if any(not np.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in FIELDS):
        raise ValueError('Invalid native AUC/HM/accuracy')
    return metrics


def load_inputs(reference):
    """Verify old assets, then read already-frozen tensors; NEVER decode an image."""
    import importlib.util
    verify_files(reference['c_files'])
    verify_files(reference['stageA_files'])
    base = Path(reference['stageA_base'])
    prepared = read_json(base / 'prepared.json')
    if not prepared['passed'] or prepared['formal_training_optimizer_steps'] != 0:
        raise ValueError('Unexpected Stage A state')
    labels = read_json(base / 'labels.json')
    if labels != reference['labels']:
        raise ValueError('Cached primitive order changed')
    bank = torch.load(Path(reference['bank']) / 'bank.pt', map_location='cpu', weights_only=True)
    caches = {}
    ids = {}
    for phase, n in (('train', 22998), ('val', 3214)):
        value = torch.load(base / (phase + '.pt'), map_location='cpu', weights_only=True)
        if set(value) != {'features', 'scores', 'attrs', 'objects'}:
            raise ValueError('Unexpected cache fields')
        if value['features'].shape != (n, 300) or value['scores'].shape != (n, 12, 16):
            raise ValueError('Unexpected cache dimensions')
        for k in ('features', 'scores'):
            if value[k].dtype != torch.float32:
                raise ValueError('Expected float32 frozen cache')
            finite(value[k], phase + '/' + k)
        for k, maximum in (('attrs', 16), ('objects', 12)):
            if value[k].shape != (n,) or value[k].dtype != torch.long or value[k].min() < 0 or value[k].max() >= maximum:
                raise ValueError('Invalid cached label')
        ids[phase] = read_json(base / (phase + '_image_ids.json'))
        if len(ids[phase]) != n or len(set(ids[phase])) != n:
            raise ValueError('Duplicate/missing image IDs')
        caches[phase] = value
    if set(ids['train']) & set(ids['val']):
        raise ValueError('TRAIN and VAL image identity overlap')
    split = {}
    split_root = Path('/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural')
    for phase in ('train', 'val', 'test'):
        split[phase] = [tuple(line.split()) for line in (split_root / (phase + '_pairs.txt')).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if list(map(len, (split['train'], split['val'], split['test']))) != [83, 30, 36]:
        raise ValueError('Unexpected split sizes')
    datasets = {s: make_dataset(reference, s, split) for s in ('cw', 'ow')}
    for setting, ds in datasets.items():
        if list(map(list, ds.pairs)) != read_json(base / ('candidates_' + setting + '.json')):
            raise ValueError('Candidate order differs from Stage A')
    train_pairs = {(labels['attrs'][int(a)], labels['objs'][int(o)])
                   for a, o in zip(caches['train']['attrs'], caches['train']['objects'])}
    if train_pairs != set(split['train']):
        raise ValueError('Cache TRAIN pair coverage differs')
    if any((labels['attrs'][int(a)], labels['objs'][int(o)]) not in set(split['train'] + split['val'])
           for a, o in zip(caches['val']['attrs'], caches['val']['objects'])):
        raise ValueError('Unexpected VAL pair')
    for setting, ds in datasets.items():
        with np.load(base / ('labels_' + setting + '.npz'), allow_pickle=False) as truth:
            if not np.array_equal(truth['attr'], caches['val']['attrs'].numpy()) or not np.array_equal(truth['obj'], caches['val']['objects'].numpy()):
                raise ValueError('Cache and saved evaluation labels differ')
            pair = [ds.pair2idx[(ds.attrs[int(a)], ds.objs[int(o)])] for a, o in zip(caches['val']['attrs'], caches['val']['objects'])]
            if not np.array_equal(truth['pair'], pair):
                raise ValueError('Cache pair labels differ')
        scores = np.load(base / ('val_scores_' + setting + '.npy'), allow_pickle=False)
        oi = [ds.obj2idx[o] for a, o in ds.pairs]
        ai = [ds.attr2idx[a] for a, o in ds.pairs]
        if not np.array_equal(scores, caches['val']['scores'][:, oi, ai].numpy()):
            raise ValueError('Cache scores differ from audited Stage A replay')
    source = Path(reference['c_source']) / 'evaluator_ge.py'
    spec = importlib.util.spec_from_file_location('pilot_verified_native_evaluator', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return caches, ids, bank, datasets, module.Evaluator


def fresh_head(bank, variant, controls, device):
    seed_all(0)
    head = ConditionalHead(bank, variant)
    if head.permutations != controls['permutations'][variant]:
        raise ValueError('Fixed Stage A knowledge permutation differs')
    return head.to(device)


def matched_initializations(bank, controls):
    shared, reports = None, {}
    for variant in VARIANTS:
        head = fresh_head(bank, variant, controls, 'cpu')
        state = head.state_dict()
        if shared is None:
            shared = copy.deepcopy(state)
        else:
            for k, value in shared.items():
                if not torch.equal(state[k], value):
                    raise ValueError('Common initialization differs: ' + variant + '/' + k)
        reports[variant] = dict(initial_state_sha256=state_sha(head),
                                parameter_count=sum(p.numel() for p in head.parameters()),
                                permutations=head.permutations)
    return reports


def train_fold(cache, image_ids, fraction=0.20):
    groups = {}
    for i, (a, o) in enumerate(zip(cache['attrs'].tolist(), cache['objects'].tolist())):
        groups.setdefault((a, o), []).append(i)
    fit, hold, report = [], [], []
    for pair, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError('Cannot make a per-pair TRAIN mechanics fold: ' + repr(pair))
        ordered = sorted(indices, key=lambda i: (hashlib.sha256(image_ids[i].encode()).hexdigest(), image_ids[i]))
        count = min(len(indices) - 1, max(1, int(math.floor(len(indices) * fraction))))
        hold.extend(ordered[:count])
        fit.extend(ordered[count:])
        report.append(dict(attr=pair[0], obj=pair[1], total=len(indices), fit=len(indices)-count, hold=count))
    fit, hold = sorted(fit), sorted(hold)
    if set(fit) & set(hold) or sorted(fit + hold) != list(range(len(image_ids))):
        raise ValueError('Fold overlap or coverage error')
    return torch.tensor(fit), torch.tensor(hold), report


def epoch_order(indices, epoch):
    generator = torch.Generator().manual_seed(21009000 + epoch)
    return indices[torch.randperm(len(indices), generator=generator)]


def train_epoch(head, optimizer, cache, indices, epoch, device, batch_size=256):
    head.train()
    order = epoch_order(indices, epoch)
    total, steps, active = 0.0, 0, set()
    for selected in order.split(batch_size):
        x, base = cache['features'][selected].to(device), cache['scores'][selected].to(device)
        attrs, objects = cache['attrs'][selected].to(device), cache['objects'][selected].to(device)
        optimizer.zero_grad(set_to_none=True)
        delta = head.evidence(x)
        finite(delta, 'TRAIN/evidence')
        loss = conditional_loss(base, delta, attrs, objects)
        finite(loss, 'TRAIN/nll')
        loss.backward()
        for name, p in head.named_parameters():
            if p.grad is None:
                raise ValueError('Disconnected parameter: ' + name)
            finite(p.grad, 'TRAIN/gradient/' + name)
            if float(p.grad.norm()) > 0:
                active.add(name)
        optimizer.step()
        total += float(loss.detach()) * len(selected)
        steps += 1
    for name, p in head.named_parameters():
        finite(p, 'TRAIN/parameter/' + name)
    return dict(mean_conditional_nll=total/len(order), images=len(order), steps=steps,
                epoch_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                nonzero_gradient_parameters=sorted(active))


def scores_and_diagnostics(head, cache, device, indices=None):
    if indices is None:
        indices = torch.arange(len(cache['features']))
    head.eval()
    scores, deltas = [], []
    with torch.no_grad():
        for selected in indices.split(256):
            x, base = cache['features'][selected].to(device), cache['scores'][selected].to(device)
            delta = head.evidence(x)
            value = corrected_scores(base, delta)
            finite(value, 'evaluation/score')
            finite(delta, 'evaluation/delta')
            scores.append(value.cpu())
            deltas.append(delta.cpu())
    grid, delta = torch.cat(scores), torch.cat(deltas)
    base = cache['scores'][indices]
    errors = probability_checks(base, delta)
    if max(errors.values()) > 1e-5:
        raise ValueError('Full-grid probability/marginal identity failed')
    if float(delta.abs().max()) > 0.100001:
        raise ValueError('Evidence cap violated')
    attrs, objects = cache['attrs'][indices], cache['objects'][indices]
    rows = torch.arange(len(indices))
    conditional = (base + delta)[rows, objects] / TEMPERATURE
    nll = torch.nn.functional.cross_entropy(conditional, attrs, reduction='none')
    predicted = conditional.argmax(1)
    correct = predicted.eq(attrs)
    base_correct = base[rows, objects].argmax(1).eq(attrs)
    per_attr = [float(correct[attrs == i].float().mean()) for i in range(16) if (attrs == i).any()]
    diagnostics = dict(n_images=len(indices), conditional_nll=float(nll.mean()),
                       gt_object_attr_accuracy=float(correct.float().mean()),
                       macro_attr_accuracy=float(np.mean(per_attr)), macro_attr_classes=len(per_attr),
                       corrected_from_baseline=int((correct & ~base_correct).sum()),
                       damaged_from_baseline=int((~correct & base_correct).sum()),
                       candidate_object_delta_std=float(delta.std(1, unbiased=False).mean()),
                       max_abs_delta=float(delta.abs().max()), probability_errors=errors)
    vectors = dict(indices=indices.numpy(), attrs=attrs.numpy(), objects=objects.numpy(),
                   conditional_prediction=predicted.numpy(), baseline_correct=base_correct.numpy(),
                   correct=correct.numpy(), conditional_nll=nll.numpy())
    return grid, diagnostics, vectors


def conditional_by_seen(vectors, datasets):
    ds = datasets['cw']
    seen = np.array([(ds.attrs[int(a)], ds.objs[int(o)]) in set(ds.train_pairs)
                     for a, o in zip(vectors['attrs'], vectors['objects'])])
    result = {}
    for name, mask in (('seen_pair', seen), ('unseen_pair', ~seen)):
        result[name] = dict(images=int(mask.sum()),
                            conditional_nll=float(vectors['conditional_nll'][mask].mean()) if mask.any() else None,
                            attr_accuracy=float(vectors['correct'][mask].mean()) if mask.any() else None)
    return result


def paired_predictions(left, right):
    if any(not np.array_equal(left[k], right[k]) for k in ('indices', 'attrs', 'objects', 'baseline_correct')):
        raise ValueError('Diagnostic pairing differs')
    return dict(left_correct_right_wrong=int((left['correct'] & ~right['correct']).sum()),
                left_wrong_right_correct=int((~left['correct'] & right['correct']).sum()),
                both_correct=int((left['correct'] & right['correct']).sum()),
                both_wrong=int((~left['correct'] & ~right['correct']).sum()))


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                torch=torch.get_rng_state(),
                cuda={str(torch.cuda.current_device()): torch.cuda.get_rng_state()} if torch.cuda.is_available() else {})


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    for device, value in state['cuda'].items():
        torch.cuda.set_rng_state(value, int(device))


def cpu_state(head):
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


def optimizer(head, settings):
    return torch.optim.AdamW(head.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])


def environment():
    import scipy
    return dict(python=__import__('sys').version, torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
