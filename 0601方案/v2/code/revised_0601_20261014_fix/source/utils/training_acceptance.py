"""Disposable real optimizer steps and stochastic checkpoint replay."""
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from .training_state import build_optimizer, capture_rng_state, restore_training_state


def assert_state_close(actual, expected, name='state'):
    if torch.is_tensor(actual):
        left, right = actual.detach().cpu(), expected.detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError('Checkpoint tensor shape/dtype differs: ' + name)
        equal = torch.allclose(left, right, rtol=1e-5, atol=1e-6) if left.is_floating_point() else torch.equal(left, right)
        if not equal:
            raise AssertionError('Checkpoint tensor differs: ' + name)
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('Checkpoint keys differ: ' + name)
        for key in actual:
            assert_state_close(actual[key], expected[key], name + '.' + str(key))
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError('Checkpoint lengths differ: ' + name)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_state_close(left, right, name + '.' + str(index))
    elif actual != expected:
        raise AssertionError('Checkpoint value differs: ' + name)


def check_training_resume(model, batch, cfg, directory):
    optimizer, _ = build_optimizer(model, cfg)
    device = next(model.parameters()).device
    initial_mode = model.training
    first_epoch = max(1, int(getattr(cfg.MODEL, 'rev_beta_delay', 0)))

    def step(epoch):
        model.train()
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out['loss_total']
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite train-mode acceptance loss')
        loss.backward()
        watched = []
        for index, group in enumerate(optimizer.param_groups):
            active = []
            for parameter in group['params']:
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError('Nonfinite train-mode acceptance gradient')
                    if parameter.grad.abs().sum() > 0:
                        active.append(parameter)
            if active:
                watched.append((index, active[0], active[0].detach().clone()))
        if not watched:
            raise ValueError('No active optimizer parameters')
        optimizer.step()
        changed = []
        for index, parameter, before in watched:
            if torch.equal(parameter.detach(), before):
                raise ValueError('Optimizer group did not update: ' + str(index))
            changed.append(index)
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError('Nonfinite parameter after optimizer.step')
        return float(loss.detach()), changed

    def draws():
        return random.random(), float(np.random.rand()), torch.rand(4, device=device)

    try:
        with tempfile.TemporaryDirectory(prefix='training_acceptance_', dir=str(directory)) as td:
            checkpoint, expected_path = Path(td)/'resume.pth', Path(td)/'expected.pth'
            first_loss, changed = step(first_epoch)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=first_epoch)
            state.update(capture_rng_state())
            torch.save(state, checkpoint)
            del state
            expected_draws = draws()
            expected_loss, _ = step(first_epoch + 1)
            model.eval()
            with torch.no_grad():
                expected_pred = model.val_forward(batch)['pred'].detach().cpu()
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, expected_path)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            next_epoch = restore_training_state(model, optimizer, state)
            del state
            actual_draws = draws()
            if expected_draws[:2] != actual_draws[:2] or not torch.equal(expected_draws[2], actual_draws[2]):
                raise AssertionError('Python/NumPy/Torch RNG did not resume')
            actual_loss, _ = step(next_epoch)
            if not np.isclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-6):
                raise AssertionError('Resumed next-step loss differs')
            expected = torch.load(expected_path, map_location='cpu', weights_only=False)
            assert_state_close(model.state_dict(), expected['model'], 'model')
            assert_state_close(optimizer.state_dict(), expected['optimizer'], 'optimizer')
            del expected
            model.eval()
            with torch.no_grad():
                assert_state_close(model.val_forward(batch)['pred'], expected_pred, 'eval_prediction')
            return dict(passed=True, optimizer_steps=3, continuous_steps=2,
                        replayed_steps=1, changed_optimizer_groups=changed,
                        checkpoint_epoch=first_epoch, resumed_epoch=next_epoch,
                        train_mode=True, model_optimizer_rng_roundtrip=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.train(initial_mode)
