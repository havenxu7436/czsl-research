#!/usr/bin/env python3
"""Real-data CUDA identity and individual gradient checks, without training."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg as default_cfg
from dataset import CompositionDataset
from models.kcc_cpf import DDM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    output = Path(a.out)
    if output.exists():
        raise FileExistsError('Smoke report exists; preserve it and use a new name')
    if not torch.cuda.is_available():
        raise RuntimeError('Real-data smoke requires the CPF CUDA environment')
    c = default_cfg.clone()
    c.merge_from_file(a.cfg)
    if not c.MODEL.kcc_logic or c.MODEL.kcc_beta != 0.01:
        raise ValueError('Run smoke with the enabled B or F config')
    seed_all(c.TRAIN.seed)
    ds = CompositionDataset(phase='train', split=c.DATASET.splitname, open_world=False, cfg=c)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    base_config = c.clone()
    base_config.MODEL.kcc_logic = False
    base_config.MODEL.kcc_beta = 0.0
    seed_all(c.TRAIN.seed)
    baseline = DDM(ds, base_config).cuda().eval()
    seed_all(c.TRAIN.seed)
    enabled = DDM(ds, c).cuda().eval()
    if list(baseline.state_dict()) != list(enabled.state_dict()):
        raise ValueError('Unexpected extra persistent parameters/buffers')
    if not all(torch.equal(v, enabled.state_dict()[k]) for k, v in baseline.state_dict().items()):
        raise ValueError('Initializations not matched')
    if sum(p.numel() for p in baseline.parameters()) != sum(p.numel() for p in enabled.parameters()):
        raise ValueError('Extra parameters introduced')
    with torch.no_grad():
        seed_all(711)
        off = baseline.train_forward(batch)
        saved_beta = enabled.bf_logic.beta
        enabled.bf_logic.beta = 0.0
        seed_all(711)
        zero = enabled.train_forward(batch)
        if not torch.equal(off['loss_total'], zero['loss_total']):
            raise ValueError('beta=0 does not exactly reproduce native training loss')
        for model in (baseline, enabled):
            model.val_attrs = model.train_attrs
            model.val_objs = model.train_objs
            model.val_pairs = ds.train_pairs
        eval_off = baseline.val_forward(batch)['pred']
        enabled.bf_logic.beta = saved_beta
        eval_on = enabled.val_forward(batch)['pred']
        if not torch.equal(eval_off, eval_on):
            raise ValueError('Logic changed inference scores at fixed weights')
    del baseline
    torch.cuda.empty_cache()
    seed_all(711)
    terms = enabled.train_forward(batch)
    params = [p for n, p in enabled.named_parameters()
              if p.requires_grad and (n.startswith('objc_embedder.') or n.startswith('attrc_embedder.'))]
    gradients = {}
    for name in ('loss_oi', 'loss_oe', 'loss_ae'):
        if not torch.isfinite(terms[name]) or not terms[name].requires_grad:
            raise ValueError('Invalid or disconnected term: ' + name)
        grad = torch.autograd.grad(terms[name], params, retain_graph=True, allow_unused=True)
        norm = sum(float(g.float().square().sum()) for g in grad if g is not None) ** 0.5
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Zero/nonfinite native visual gradient: ' + name)
        gradients[name] = norm
    if not torch.allclose(terms['loss_total'], terms['loss_cpf'] + c.MODEL.kcc_beta * terms['loss_logic_raw']):
        raise ValueError('Wrong beta weighting')
    report = {'passed': True, 'variant': c.TRAIN.kcc_variant, 'config_sha256': sha(a.cfg),
              'logic_manifest_sha256': enabled.bf_logic.manifest_sha256,
              'install_sha256': sha(Path(__file__).resolve().parents[1] / 'bf_install.json'),
              'losses': {k: float(terms[k].detach()) for k in ('loss_cpf','loss_oi','loss_oe','loss_ae','loss_logic_raw','loss_logic_weighted')},
              'individual_visual_gradient_norm': gradients, 'beta0_training_identity': True,
              'fixed_weight_inference_identity': True, 'parameter_count_equal': True,
              'n_images': len(batch['img']), 'gpu': torch.cuda.get_device_name(0),
              'warning': 'No optimizer step or efficacy conclusion; train_forward called in eval mode to disable dropout for identity checks.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('REAL_CUDA_SMOKE_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
