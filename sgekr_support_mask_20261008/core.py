"""Frozen name anchor plus a support-masked fixed semantic increment."""
import torch
from torch import nn
from torch.nn import functional as F


class IncrementRouter(nn.Module):
    def __init__(self, bank, variant, support_mask, rank=32,
                 eta_max=0.1, temperature=0.2, object_chunk=16):
        super().__init__()
        if variant not in ('G_name', 'G_support', 'G_shuffle'):
            raise ValueError('Unknown SG-EKR variant')
        self.variant = variant
        self.eta_max, self.temperature = eta_max, temperature
        self.object_chunk = object_chunk
        a = bank['a']
        clean, clean_mask = a['units'].float().clone(), a['mask'].bool().clone()
        selected = torch.as_tensor(support_mask, dtype=torch.bool).clone()
        if clean.size(-1) != 300 or clean.size(1) != 4 or selected.shape != clean_mask.shape:
            raise ValueError('Expected a [attributes,4,300] bank and matched support mask')
        if not torch.equal(selected[:, 0], torch.ones_like(selected[:, 0])):
            raise ValueError('Every attribute must retain its own name slot')
        if (selected & ~clean_mask).any():
            raise ValueError('Support mask selected an absent clean slot')
        if variant == 'G_name':
            units = clean
            mask = torch.zeros_like(selected); mask[:, 0] = True
        elif variant == 'G_support':
            units, mask = clean, selected
        else:
            units = a['shuffle_units'].float().clone()
            mask = selected
            if (mask & ~a['shuffle_mask'].bool()).any():
                raise ValueError('Matched shuffle control lacks a selected slot')
            if not torch.allclose(units[:, 0], clean[:, 0], atol=0, rtol=0):
                raise ValueError('Shuffle control changed own attribute names')
        self.register_buffer('units', units, persistent=False)
        self.register_buffer('slot_mask', mask, persistent=False)
        self.register_buffer('object_units', bank['o']['unit'].float().clone(), persistent=False)
        self.qx = nn.Linear(300, rank, bias=False)
        self.qo = nn.Linear(300, rank, bias=False)
        self.pk = nn.Linear(300, rank, bias=False)
        self.null_key = nn.Parameter(torch.randn(rank) * 0.02)
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.raw_gain = nn.Parameter(torch.zeros(()))

    @property
    def eta(self):
        return self.eta_max * torch.tanh(self.raw_gain)

    def fixed_values(self):
        return F.normalize(self.units, dim=-1, eps=1e-6) / 2

    def forward(self, feature, attr_ids, obj_ids, force_off=False, return_diag=False):
        u = F.normalize(feature, dim=-1, eps=1e-6)
        keys = F.normalize(self.pk(self.units), dim=-1, eps=1e-6)
        values = self.fixed_values()
        evidence = torch.einsum('bd,ard->bar', u, values)
        objects = torch.unique(obj_ids, sorted=True)
        output = feature.new_zeros((feature.size(0), attr_ids.numel()))
        null_sum = feature.new_zeros(())
        count = 0
        for start in range(0, objects.numel(), self.object_chunk):
            group = objects[start:start+self.object_chunk]
            indices = torch.nonzero((obj_ids[:, None] == group[None]).any(1), as_tuple=False).flatten()
            local = (obj_ids[indices, None] == group[None]).long().argmax(1)
            context = torch.sigmoid(self.qo(self.object_units[group]))
            query = F.normalize(self.qx(u)[:, None] * context[None], dim=-1, eps=1e-6)
            logits = torch.einsum('bnk,ark->bnar', query, keys) / self.temperature
            logits = logits.masked_fill(~self.slot_mask[None, None], float('-inf'))
            null = torch.einsum('bnk,k->bn', query, F.normalize(self.null_key, dim=0, eps=1e-6))
            null = null / self.temperature + self.null_bias
            logits = torch.cat([null[:, :, None, None].expand(-1, -1, logits.size(2), 1), logits], -1)
            attention = logits.softmax(-1)
            weighted = (attention[..., 1:] * evidence[:, None]).sum(-1)
            output[:, indices] = weighted[:, local, attr_ids[indices]]
            null_sum += attention[:, local, attr_ids[indices], 0].sum()
            count += feature.size(0) * indices.numel()
        residual = output * (self.eta * 0 if force_off else self.eta)
        if not return_diag:
            return residual
        return residual, dict(eta=float(self.eta.detach()),
            null_mean=float((null_sum / max(count, 1)).detach()),
            mean_abs_residual=float(residual.detach().abs().mean()),
            max_abs_residual=float(residual.detach().abs().max()),
            value_drift_mean=0.0,
            mean_active_slots=float(self.slot_mask.float().sum(1).mean()))


class AnchorModel(nn.Module):
    def __init__(self, name_model, router):
        super().__init__()
        self.base = name_model.base
        self.name_anchor = name_model.router
        self.bank = name_model.bank
        self.router = router
        self.base.requires_grad_(False)
        self.name_anchor.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval(); self.name_anchor.eval(); self.router.train(mode)
        return self

    def components(self, batch, ai, oi):
        with torch.no_grad():
            parts = self.base.raw_forward(batch)
            feature = parts['attr_feature']
            anchor = parts['pred'] + self.name_anchor(feature, ai, oi)
        return anchor, feature

    def score(self, batch, ai, oi, force_off=False, return_diag=False):
        anchor, feature = self.components(batch, ai, oi)
        result = self.router(feature, ai, oi, force_off=force_off, return_diag=return_diag)
        if return_diag:
            residual, diag = result
            return anchor + residual, diag
        return anchor + result
