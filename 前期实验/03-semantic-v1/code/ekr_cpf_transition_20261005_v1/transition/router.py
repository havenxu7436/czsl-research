import torch
from torch import nn
from torch.nn import functional as F


class Router(nn.Module):
    """Same-capacity score adapters. Values remain in raw cosine score units."""
    def __init__(self, bank, mode="ekr", rank=32, eta_max=0.1, temperature=0.2,
                 object_chunk=16):
        super().__init__()
        self.mode, self.eta_max = mode, eta_max
        self.temperature, self.object_chunk = temperature, object_chunk
        if mode not in {"ekr", "adapter", "attention", "static", "no_object",
                        "no_null", "shuffle", "name_residual", "raw_values"}:
            raise ValueError("Unknown router mode: " + mode)
        a, o = bank["a"], bank["o"]
        if mode == "adapter":
            keys = a["correct"][:, None]
            values = keys
            mask = torch.ones(keys.shape[:2], dtype=torch.bool)
        else:
            units = a["shuffle_units"] if mode == "shuffle" else a["units"]
            mask = a["shuffle_mask"] if mode == "shuffle" else a["mask"]
            mean = (units * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            residual = (units - mean[:, None]) * mask[..., None]
            keys = units if mode == "attention" else residual
            values = units if mode in {"attention", "raw_values"} else residual / 2
            if mode == "name_residual":
                keys, values, mask = keys[:, :1], values[:, :1], mask[:, :1]
        self.register_buffer("key_source", keys.float(), persistent=False)
        self.register_buffer("value_source", values.float(), persistent=False)
        self.register_buffer("slot_mask", mask.bool(), persistent=False)
        self.register_buffer("object_source", o["correct"].float(), persistent=False)
        self.qx = nn.Linear(300, rank, bias=False)
        self.qo = nn.Linear(300, rank, bias=False)
        self.pk = nn.Linear(300, rank, bias=False)
        self.null_key = nn.Parameter(torch.randn(rank) * 0.02)
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.raw_gain = nn.Parameter(torch.zeros(()))
        if mode == "static":
            self.static_query = nn.Parameter(torch.randn(rank) * 0.02)

    @property
    def eta(self):
        return self.eta_max * torch.tanh(self.raw_gain)

    def forward(self, feature, attr_ids, obj_ids, force_off=False, return_diag=False):
        u = F.normalize(feature, dim=-1, eps=1e-6)
        keys = F.normalize(self.pk(self.key_source), dim=-1, eps=1e-6)
        evidence = torch.einsum("bd,mrd->bmr", u, self.value_source)
        unique_objects = torch.unique(obj_ids, sorted=True)
        output = feature.new_zeros((feature.size(0), attr_ids.numel()))
        diag_null_sum = feature.new_zeros(())
        diag_count = 0
        qx = self.qx(u)
        for offset in range(0, unique_objects.numel(), self.object_chunk):
            objects = unique_objects[offset:offset + self.object_chunk]
            selected = (obj_ids[:, None] == objects[None, :]).any(1)
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            local_obj = (obj_ids[indices, None] == objects[None, :]).long().argmax(1)
            if self.mode == "static":
                query = self.static_query.view(1, 1, -1).expand(u.size(0), objects.numel(), -1)
            elif self.mode == "no_object":
                query = qx[:, None, :].expand(-1, objects.numel(), -1)
            else:
                context = torch.sigmoid(self.qo(self.object_source[objects]))
                query = qx[:, None, :] * context[None, :, :]
            query = F.normalize(query, dim=-1, eps=1e-6)
            logits = torch.einsum("bnk,mrk->bnmr", query, keys) / self.temperature
            logits = logits.masked_fill(~self.slot_mask[None, None], float("-inf"))
            if self.mode != "no_null":
                null = torch.einsum("bnk,k->bn", query,
                                    F.normalize(self.null_key, dim=0, eps=1e-6))
                null = null / self.temperature + self.null_bias
                logits = torch.cat([null[:, :, None, None].expand(-1, -1, logits.size(2), 1),
                                    logits], dim=-1)
                weights = logits.softmax(-1)
                delta = (weights[..., 1:] * evidence[:, None]).sum(-1)
                if return_diag:
                    diag_null_sum += weights[:, local_obj, attr_ids[indices], 0].sum()
            else:
                weights = logits.softmax(-1)
                delta = (weights * evidence[:, None]).sum(-1)
            output[:, indices] = delta[:, local_obj, attr_ids[indices]]
            diag_count += u.size(0) * indices.numel()
        residual = output * (self.eta * 0 if force_off else self.eta)
        if not return_diag:
            return residual
        return residual, {"eta": float(self.eta.detach()),
                          "null_mean": float((diag_null_sum / max(1, diag_count)).detach()),
                          "mean_abs_residual": float(residual.detach().abs().mean()),
                          "max_abs_residual": float(residual.detach().abs().max())}
