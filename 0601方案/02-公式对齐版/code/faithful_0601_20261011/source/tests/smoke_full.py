"""Real TRAIN-only CUDA wiring check, no optimizer step."""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg as defaults
from dataset import CompositionDataset
from models.kcc_cpf import DDM


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False


def norm(grads):
    return sum(float(g.detach().float().square().sum()) for g in grads if g is not None) ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    target = Path(a.out)
    if target.exists():
        raise FileExistsError('Preserve existing smoke report ' + str(target))
    c = defaults.clone()
    c.merge_from_file(a.cfg)
    if not c.MODEL.kcc_feature:
        raise ValueError('Smoke needs a full feature configuration')
    ds = CompositionDataset(phase='train', split=c.DATASET.splitname, open_world=False, cfg=c)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    off = c.clone()
    off.MODEL.kcc_feature = False
    off.MODEL.kcc_logic = False
    off.MODEL.kcc_beta = 0.0
    seed(c.TRAIN.seed)
    baseline = DDM(ds, off).cuda().eval()
    native_cpu_rng = torch.get_rng_state()
    native_cuda_rng = torch.cuda.get_rng_state_all()
    seed(c.TRAIN.seed)
    enabled = DDM(ds, c).cuda().eval()
    if not torch.equal(native_cpu_rng, torch.get_rng_state()) or not all(
            torch.equal(a, b) for a, b in zip(native_cuda_rng, torch.cuda.get_rng_state_all())):
        raise ValueError('Feature initialization changed CPU/CUDA RNG stream')
    native_state = baseline.state_dict()
    for name, value in native_state.items():
        if not torch.equal(value, enabled.state_dict()[name]):
            raise ValueError('CPF initialization differs: ' + name)
    for model in (baseline, enabled):
        model.val_attrs, model.val_objs, model.val_pairs = model.train_attrs, model.train_objs, ds.train_pairs
    with torch.no_grad():
        enabled.full_gain = 0.0
        bp = baseline.val_forward(batch)['pred']
        ep = enabled.val_forward(batch)['pred']
        if not torch.equal(bp, ep):
            raise ValueError('gain0 does not reproduce CPF inference')
        seed(710)
        b = baseline.train_forward(batch)['loss_cpf']
        seed(710)
        e = enabled.train_forward(batch)['loss_cpf']
        if not torch.equal(b, e):
            raise ValueError('gain0 does not reproduce CPF task loss')
    del baseline
    torch.cuda.empty_cache()
    enabled.full_gain = 1.0
    seed(710)
    out = enabled.train_forward(batch)
    head = [v for v in enabled.full_head.parameters() if v.requires_grad]
    visual = [v for name, v in enabled.named_parameters() if v.requires_grad and name.startswith(('objc_embedder.', 'attrc_embedder.'))]
    backbone = [v for name, v in enabled.named_parameters() if v.requires_grad and name.startswith('feat_extractor.')]
    gradients = {'feature': norm(torch.autograd.grad(out['loss_cpf'], head, retain_graph=True, allow_unused=True)),
                 'visual': norm(torch.autograd.grad(out['loss_cpf'], visual, retain_graph=True, allow_unused=True))}
    if c.TRAIN.finetune_backbone:
        gradients['backbone'] = norm(torch.autograd.grad(out['loss_cpf'], backbone, retain_graph=True, allow_unused=True))
    if c.MODEL.kcc_logic:
        gradients['AE_to_feature'] = norm(torch.autograd.grad(out['loss_ae'], head, retain_graph=True, allow_unused=True))
        for term in ('loss_oi', 'loss_oe', 'loss_ae'):
            gradients[term + '_visual'] = norm(torch.autograd.grad(out[term], visual, retain_graph=True, allow_unused=True))
    if not all(np.isfinite(value) and value > 0 for value in gradients.values()):
        raise ValueError('Missing/nonfinite gradient ' + str(gradients))
    if not torch.allclose(out['loss_total'], out['loss_cpf'] + out['loss_logic_weighted']):
        raise ValueError('Logic applied incorrectly')
    report = {'passed': True, 'variant': c.TRAIN.kcc_variant, 'n_train_images': 4,
              'same_native_initialization': True, 'cpu_cuda_rng_preserved': True, 'gain0_inference_identity': True,
              'gain0_task_loss_identity': True, 'gradients': gradients,
              'losses': {k: float(v.detach()) for k, v in out.items() if k.startswith('loss')},
              'optimizer_steps': 0, 'test_images_read': 0, 'warning': 'Wiring check, not evidence of improved accuracy'}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('FULL_REAL_CUDA_SMOKE_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
