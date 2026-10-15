"""Real TRAIN CUDA formula/init/gradient checks, no optimizer update."""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg as defaults
from dataset import CompositionDataset
from models.kcc_cpf import DDM, LegacyDDM


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
    dest = Path(a.out)
    if dest.exists():
        raise FileExistsError('Preserve smoke evidence: ' + str(dest))
    c = defaults.clone()
    c.merge_from_file(a.cfg)
    ds = CompositionDataset(phase='train', split=c.DATASET.splitname, open_world=False, cfg=c)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    off = c.clone()
    off.MODEL.kcc_feature, off.MODEL.kcc_logic, off.MODEL.kcc_beta = False, False, 0.0
    off.MODEL.formula_joint = False
    seed(c.TRAIN.seed)
    baseline = LegacyDDM(ds, off).cuda().eval()
    native_state = {k: v.detach().cpu() for k, v in baseline.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    del baseline
    torch.cuda.empty_cache()
    seed(c.TRAIN.seed)
    model = DDM(ds, c).cuda().eval()
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all())):
        raise ValueError('Model additions changed the native RNG stream')
    for name, value in native_state.items():
        if not torch.equal(value, model.state_dict()[name].detach().cpu()):
            raise ValueError('Native initialization changed: ' + name)
    del native_state
    model.val_objs, model.val_attrs, model.val_pairs = model.train_objs, model.train_attrs, ds.train_pairs
    out, logits, nodes = model.formula_train_terms(batch, return_details=True)
    expected = logits['obj'][:, :, None] + logits['conditional']
    if model.kcc_calib:
        expected = expected + model.kcc_lambda * model.K_oa_prior[None]
    if not torch.equal(logits['score'], expected):
        raise ValueError('Joint score formula mismatch')
    if c.MODEL.formula_joint:
        with torch.no_grad():
            inference = model.val_forward(batch)['pred']
        selected = logits['score'][:, model.train_objs, model.train_attrs]
        if not torch.allclose(inference, selected, atol=2e-6, rtol=1e-5):
            raise ValueError('Training/inference scoring differs in eval mode')
        conditioned = logits['conditional'][torch.arange(4, device='cuda'), batch['obj'], :]
        if not torch.equal(out['loss_conditional_attr'], F.cross_entropy(conditioned, batch['attr'])):
            raise ValueError('GT-object attribute CE mismatch')
    if not torch.equal(out['loss_total'], out['loss_cpf'] + out['loss_logic_weighted']):
        raise ValueError('Logic weight applied incorrectly')
    gradients = {}
    if c.MODEL.kcc_feature:
        head = list(model.full_head.parameters())
        gradients['head_task'] = norm(torch.autograd.grad(out['loss_cpf'], head, retain_graph=True, allow_unused=True))
        if nodes is not None:
            gradients['AE_to_head'] = norm(torch.autograd.grad(out['loss_ae'], head, retain_graph=True, allow_unused=True))
    visual = [v for n, v in model.named_parameters() if n.startswith(('objc_embedder.', 'attrc_embedder.'))]
    gradients['visual_task'] = norm(torch.autograd.grad(out['loss_cpf'], visual, retain_graph=True, allow_unused=True))
    backbone = [v for n, v in model.named_parameters() if n.startswith('feat_extractor.') and v.requires_grad]
    if c.TRAIN.finetune_backbone:
        gradients['backbone_task'] = norm(torch.autograd.grad(out['loss_cpf'], backbone, retain_graph=True, allow_unused=True))
    if nodes is not None:
        for term in ('loss_oi', 'loss_oe', 'loss_ae'):
            gradients[term + '_visual'] = norm(torch.autograd.grad(out[term], visual, retain_graph=True, allow_unused=True))
        expected_oe = torch.sigmoid(logits['image_feat'] @ model.bf_logic.oe.reshape(-1, 300).t()).reshape(4, -1, 3)
        if not torch.equal(nodes['oe'], expected_oe):
            raise ValueError('Description nodes are not literal sigmoid(dot)')
    if not all(np.isfinite(v) and v > 0 for v in gradients.values()):
        raise ValueError('Missing/nonfinite gradients ' + str(gradients))
    out['loss_total'].backward()
    if c.MODEL.kcc_feature:
        if model.projection_1.parameters().__next__().grad is not None or model.compose.parameters().__next__().grad is not None:
            raise ValueError('Native composition branch still enters G/H training')
        if not c.MODEL.full_use_pair and model.full_head.pair_bias[0].weight.grad is not None:
            raise ValueError('No-pair control consumed pair MLP')
    report = {'passed': True, 'variant': c.TRAIN.kcc_variant, 'train_images': 4,
              'same_native_initialization': True, 'native_cpu_cuda_rng_preserved': True,
              'joint_formula_exact': True, 'train_infer_formula_identity': bool(c.MODEL.formula_joint),
              'gradients': gradients, 'optimizer_steps': 0, 'test_images_read': 0,
              'losses': {k: float(v.detach()) for k, v in out.items() if k.startswith('loss')},
              'scope': 'Wiring check only, not an accuracy result or paper reproduction claim'}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('FORMULA_REAL_CUDA_SMOKE_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
