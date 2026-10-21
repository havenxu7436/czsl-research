"""Checkpoint AdamW diagnostics for static and dynamic CP readouts."""
from pathlib import Path

import torch

from vgcpod.common import load_torch, read_json
from vgcpod.engine import make_optimizer
from vgcpod.model import VGCPOD


def checkpoint_readouts(checkpoint, cfg):
    """Use optimizer group order to map saved state IDs to parameter names."""
    state = checkpoint['model']
    # Model initialization is local to this diagnostic and preserves caller RNG.
    with torch.random.fork_rng(devices=[]):
        model = VGCPOD(state['ea'], state['eo'], state['ec'], cfg, 0,
                       gate_mean=state.get('gate_fo_mean'))
        optimizer = make_optimizer(model, cfg)
    names = {id(p): name for name, p in model.named_parameters()}
    saved = checkpoint['optimizer']
    if len(optimizer.param_groups) != len(saved['param_groups']):
        raise ValueError('Checkpoint optimizer groups differ from config')
    rows = []
    for group, snapshot in zip(optimizer.param_groups, saved['param_groups']):
        if group['name'] != snapshot['name'] or len(group['params']) != len(snapshot['params']):
            raise ValueError('Checkpoint optimizer parameter order differs from config')
        for parameter, index in zip(group['params'], snapshot['params']):
            name = names[id(parameter)]
            if name not in ('Wc', 'Wd'):
                continue
            moment = saved['state'][index]
            step = int(moment['step'])
            beta1, beta2 = snapshot['betas']
            m, v = moment['exp_avg'].double(), moment['exp_avg_sq'].double()
            m_hat, v_hat = m/(1-beta1**step), v/(1-beta2**step)
            direction = m_hat/(v_hat.sqrt()+snapshot['eps'])
            norm = lambda value: float(value.norm())
            rows.append(dict(parameter=name, role='static' if name=='Wc' else 'dynamic',
                optimizer_group=snapshot['name'],optimizer_step=step,lr=snapshot['lr'],
                lr_scale=snapshot['lr_scale'],parameter_norm=norm(state[name].double()),
                first_moment_norm=norm(m),second_moment_norm=norm(v),
                corrected_first_moment_norm=norm(m_hat),corrected_second_moment_norm=norm(v_hat),
                adam_direction_norm=norm(direction),
                lr_scaled_adaptive_step_norm=snapshot['lr']*norm(direction)))
    static = next(r for r in rows if r['parameter']=='Wc')
    for row in rows:
        for key in ('parameter_norm','first_moment_norm','adam_direction_norm','lr_scaled_adaptive_step_norm'):
            row[key+'_over_static'] = row[key]/static[key] if static[key] else None
    return rows


def summarize_readout_optimizer(row):
    run = Path(row['run_dir'])
    cfg = read_json(run/'protocol.json')['config']
    result = []
    for stage, name in [('native_selected','best.pt'),('exact_selected','best_exact.pt'),('final','last.pt')]:
        checkpoint = load_torch(run/'checkpoints'/name)
        result.extend(dict(case=row['case'],seed=row['seed'],stage=stage,epoch=checkpoint['epoch'],**entry)
                      for entry in checkpoint_readouts(checkpoint,cfg))
    return result
