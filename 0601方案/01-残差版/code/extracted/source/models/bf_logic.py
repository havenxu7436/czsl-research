"""Training-only, parameter-free grounding into native CPF visual branches."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from .bf_logic_loss import LogicLoss


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class CPFLogicAdapter(nn.Module):
    def __init__(self, dset, cfg):
        super().__init__()
        self.enabled = bool(getattr(cfg.MODEL, 'kcc_logic', False))
        self.beta = float(getattr(cfg.MODEL, 'kcc_beta', 0.0))
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if not self.enabled and self.beta != 0:
            raise ValueError('Nonzero beta requires kcc_logic')
        self.manifest_sha256 = None
        if not self.enabled:
            return
        directory = Path(cfg.MODEL.bf_logic_bank)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['labels'] != {'attrs': list(dset.attrs), 'objs': list(dset.objs)}:
            raise ValueError('Logic bank primitive order mismatch')
        if digest(directory / 'bank.pt') != manifest['bank_sha256']:
            raise ValueError('Logic bank checksum mismatch')
        if manifest['official_author_data'] is not False:
            raise ValueError('This release must be labeled reconstructed')
        if not bool(getattr(cfg.MODEL, 'bf_allow_reconstructed', False)):
            raise ValueError('Reconstructed asset requires explicit bf_allow_reconstructed acknowledgement')
        bank = torch.load(directory / 'bank.pt', map_location='cpu', weights_only=True)
        expected = {'parent': (len(manifest['parents']), 300),
                    'oe': (len(dset.objs), 3, 300), 'ae': (len(dset.attrs), 3, 300)}
        for name, shape in expected.items():
            value = bank[name].float()
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid logic bank tensor: ' + name)
            if not torch.allclose(value.norm(dim=-1), torch.ones(shape[:-1]), atol=1e-5):
                raise ValueError('Non-unit logic bank tensor: ' + name)
            self.register_buffer(name, value, persistent=False)
        self.rules = LogicLoss(manifest['object_parent'], len(manifest['parents']),
                               float(getattr(cfg.MODEL, 'bf_logic_q', 3.0)))
        self.manifest_sha256 = digest(directory / 'manifest.json')

    def forward(self, visual_obj, visual_attr, obj_logits, attr_logits, classifier):
        zero = obj_logits.new_zeros(())
        if not self.enabled or self.beta == 0:
            return {k: zero for k in ('loss_oi', 'loss_oe', 'loss_ae', 'loss_logic_raw', 'loss_logic_weighted')}
        b = obj_logits.shape[0]
        parent_logits = classifier(visual_obj, self.parent)
        oe_logits = classifier(visual_obj, self.oe.reshape(-1, 300)).reshape(b, -1, 3)
        ae_logits = classifier(visual_attr, self.ae.reshape(-1, 300)).reshape(b, -1, 3)
        terms = self.rules(obj_logits, parent_logits, oe_logits, attr_logits, ae_logits)
        terms['loss_logic_weighted'] = self.beta * terms['loss_logic_raw']
        return terms
