"""Frozen name anchor plus a matched raw-value semantic increment."""
import torch
from torch import nn
from torch.nn import functional as F


class IncrementRouter(nn.Module):
    def __init__(self, bank, variant, rank=32, value_rank=32,
                 eta_max=0.1, temperature=0.2, object_chunk=16):
        super().__init__()
        if variant not in ('V_name_align', 'V_clean_align', 'V_shuffle_align', 'V_clean_fixed'):
            raise ValueError('Unknown v2 variant')
        self.variant = variant
        self.eta_max, self.temperature = eta_max, temperature
        self.object_chunk = object_chunk
        a = bank['a']
        if variant == 'V_name_align':
            units = a['unit'][:, None].expand(-1, 4, -1).clone()
            mask = torch.ones(units.shape[:2], dtype=torch.bool)
        elif variant == 'V_shuffle_align':
            units, mask = a['shuffle_units'].clone(), a['shuffle_mask'].clone()
        else:
            units, mask = a['units'].clone(), a['mask'].clone()
        if units.size(-1) != 300 or units.size(1) != 4 or not mask.any(1).all():
            raise ValueError('Expected the reviewed four-slot 300D bank')
        self.register_buffer('units', units.float(), persistent=False)
        self.register_buffer('slot_mask', mask.bool(), persistent=False)
        self.register_buffer('object_units', bank['o']['unit'].float().clone(), persistent=False)
        self.qx = nn.Linear(300, rank, bias=False)
        self.qo = nn.Linear(300, rank, bias=False)
        self.pk = nn.Linear(300, rank, bias=False)
        self.null_key = nn.Parameter(torch.randn(rank) * 0.02)
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.raw_gain = nn.Parameter(torch.zeros(()))
        self.value_down = nn.Linear(300, value_rank, bias=False)
        self.value_up = nn.Linear(value_rank, 300, bias=False)
        nn.init.zeros_(self.value_up.weight)
        if variant == 'V_clean_fixed':
            self.value_down.requires_grad_(False)
            self.value_up.requires_grad_(False)

    @property
    def eta(self):
        return self.eta_max * torch.tanh(self.raw_gain)

    def projected_values(self):
        raw = self.units + self.value_up(self.value_down(self.units))
        return F.normalize(raw, dim=-1, eps=1e-6) / 2

    def forward(self, feature, attr_ids, obj_ids, force_off=False, return_diag=False):
        u = F.normalize(feature, dim=-1, eps=1e-6)
        keys = F.normalize(self.pk(self.units), dim=-1, eps=1e-6)
        values = self.projected_values()
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
        drift = (F.normalize(self.units, dim=-1, eps=1e-6)/2-values).norm(dim=-1)
        return residual, dict(eta=float(self.eta.detach()),
            null_mean=float((null_sum / max(count, 1)).detach()),
            mean_abs_residual=float(residual.detach().abs().mean()),
            max_abs_residual=float(residual.detach().abs().max()),
            value_drift_mean=float(drift.detach()[self.slot_mask].mean()))


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
        self.base.eval()
        self.name_anchor.eval()
        self.router.train(mode)
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
