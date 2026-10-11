"""LOGICZSL Eqs. 6/9/12. beta is applied ONCE by the CPF adapter.

Probabilities follow the local author's logicloss.py: a class-axis softmax
for each of the three description slots, not a softmax across the slots.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def pmean_violation(value, q=3.0):
    # Equivalent to mean(value**q, batch)**(1/q); norm has a finite zero gradient.
    if value.shape[0] == 0 or not math.isfinite(q) or q < 1:
        raise ValueError('Nonempty batch and finite q >= 1 required')
    return torch.linalg.vector_norm(value.float(), ord=q, dim=0) / (value.shape[0] ** (1.0 / q))


class LogicLoss(nn.Module):
    def __init__(self, object_parent, n_parents, q=3.0):
        super().__init__()
        ids = torch.as_tensor(object_parent, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0 or set(ids.tolist()) != set(range(n_parents)):
            raise ValueError('Every object needs one parent and every parent needs children')
        if not math.isfinite(q) or q < 1:
            raise ValueError('Invalid aggregation q')
        self.register_buffer('object_parent', ids, persistent=False)
        self.n_parents = n_parents
        self.q = q

    def forward(self, obj_logits, parent_logits, oe_logits, attr_logits, ae_logits):
        b, o = obj_logits.shape
        a = attr_logits.shape[1]
        if parent_logits.shape != (b, self.n_parents) or oe_logits.shape != (b, o, 3):
            raise ValueError('Object/parent/OE axes mismatch')
        if attr_logits.shape[0] != b or ae_logits.shape != (b, a, 3) or o != self.object_parent.numel():
            raise ValueError('Attribute/AE axes mismatch')
        for tensor in (obj_logits, parent_logits, oe_logits, attr_logits, ae_logits):
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite logic logits')
        obj = F.softmax(obj_logits.float(), dim=1)
        parent = F.softmax(parent_logits.float(), dim=1)
        attr = F.softmax(attr_logits.float(), dim=1)
        oe = F.softmax(oe_logits.float(), dim=1)
        ae = F.softmax(ae_logits.float(), dim=1)
        oi_up = pmean_violation(obj * (1 - parent[:, self.object_parent]), self.q).mean()
        best_child = torch.stack([
            obj[:, self.object_parent == i].max(dim=1).values for i in range(self.n_parents)
        ], dim=1)
        oi_down = pmean_violation(parent * (1 - best_child), self.q).mean()
        oi = oi_up + oi_down
        oe_loss = pmean_violation(obj.unsqueeze(-1) * (1 - oe), self.q).mean()
        ae_loss = pmean_violation(attr.unsqueeze(-1) * (1 - ae), self.q).mean()
        return {'loss_oi': oi, 'loss_oe': oe_loss, 'loss_ae': ae_loss,
                'loss_logic_raw': oi + oe_loss + ae_loss}
