"""Shared optimizer and checkpoint restoration for training and acceptance."""
import random
import numpy as np
import torch


def build_optimizer(model, cfg):
    word, encoder, other, scalar = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == 'rev_lambda_raw':
            scalar.append(parameter)
        elif 'attr_embedder' in name or 'obj_embedder' in name:
            if cfg.TRAIN.lr_word_embedding > 0:
                word.append(parameter)
        elif name.startswith('feat_extractor'):
            encoder.append(parameter)
        else:
            other.append(parameter)
    groups = [{'params': encoder, 'lr': cfg.TRAIN.lr_encoder}]
    lrs = [cfg.TRAIN.lr_encoder]
    if cfg.TRAIN.lr_word_embedding > 0:
        groups.append({'params': word, 'lr': cfg.TRAIN.lr_word_embedding})
        lrs.append(cfg.TRAIN.lr_word_embedding)
    groups.append({'params': other, 'lr': cfg.TRAIN.lr})
    lrs.append(cfg.TRAIN.lr)
    optimizer = torch.optim.Adam(groups, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
    if scalar:
        # Decay on eta would otherwise pull sigmoid(eta) towards 0.5.
        optimizer.add_param_group({'params': scalar, 'lr': cfg.TRAIN.lr, 'weight_decay': 0.0})
        lrs.append(cfg.TRAIN.lr)
    return optimizer, lrs


def capture_rng_state():
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_training_state(model, optimizer, state):
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
    return int(state['epoch']) + 1
