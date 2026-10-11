"""Candidate-object-conditioned OE/AE evidence; no pair lookup parameters."""
import math
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('P_name', 'P_K', 'P_generic', 'P_perm1', 'P_perm2')
RANK, TEMPERATURE, CAP = 32, 0.05, 0.10


def derangement(size, seed):
    # Independent RNG: input controls never advance parameter/data-order RNG.
    generator = torch.Generator().manual_seed(seed)
    for _ in range(10000):
        value = torch.randperm(size, generator=generator)
        if (value != torch.arange(size)).all():
            return value
    raise RuntimeError('Could not construct a derangement')


def inputs(bank, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown head variant')
    result, permutations = {}, {}
    for axis in ('a', 'o'):
        table = bank[axis]
        units, mask = table['units'].float(), table['mask'].float()
        if units.ndim != 3 or units.shape[1:] != (4, 300) or mask.shape != units.shape[:2]:
            raise ValueError('Expected one name slot and three description slots')
        if not torch.equal(mask, torch.ones_like(mask)):
            raise ValueError('This matched-slot pilot requires complete reviewed OE/AE')
        if not torch.isfinite(units).all():
            raise ValueError('Invalid knowledge vectors')
        name = units[:, 0].clone()
        knowledge = units[:, 1:].mean(1)
        permutation = torch.arange(len(name))
        if variant == 'P_name':
            knowledge = name.clone()
        elif variant.startswith('P_perm'):
            seed = (910011 if variant == 'P_perm1' else 910029) + (0 if axis == 'a' else 1000)
            permutation = derangement(len(name), seed)
            knowledge = knowledge[permutation]
        result[axis + '_name'] = name
        result[axis + '_knowledge'] = knowledge
        permutations[axis] = permutation.tolist()
    return result, permutations


class ConditionalHead(nn.Module):
    def __init__(self, bank, variant, rank=RANK):
        super().__init__()
        self.variant, self.rank = variant, rank
        tables, self.permutations = inputs(bank, variant)
        for name, table in tables.items():
            self.register_buffer(name, table, persistent=False)
        self.visual = nn.Linear(300, rank, bias=False)
        self.object_query = nn.Linear(300, rank, bias=False)
        self.object_name = nn.Linear(300, rank, bias=False)
        self.object_knowledge = nn.Linear(300, rank, bias=False)
        self.attribute_name = nn.Linear(300, rank, bias=False)
        self.attribute_knowledge = nn.Linear(300, rank, bias=False)
        self.query_norm = nn.LayerNorm(rank)
        self.relation_norm = nn.LayerNorm(rank)
        if variant == 'P_generic':
            # Same six input projections and an ordinary concatenation MLP.
            # Additional ACTIVE parameters are explicitly reported, not fake-matched.
            self.fusion = nn.Linear(3 * rank, rank)
        self.output = nn.Linear(rank, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def evidence(self, visual):
        q = self.query_norm(self.visual(visual)[:, None, :] + self.object_query(self.o_name)[None, :, :])
        o = self.object_name(self.o_name) + self.object_knowledge(self.o_knowledge)
        a = self.attribute_name(self.a_name) + self.attribute_knowledge(self.a_knowledge)
        if self.variant == 'P_generic':
            b, no, na = len(visual), len(o), len(a)
            combined = torch.cat((q[:, :, None, :].expand(b, no, na, -1),
                                  o[None, :, None, :].expand(b, no, na, -1),
                                  a[None, None, :, :].expand(b, no, na, -1)), dim=-1)
            hidden = self.relation_norm(F.gelu(self.fusion(combined)))
        else:
            relation = self.relation_norm(o[:, None, :] * a[None, :, :])
            hidden = q[:, :, None, :] * relation[None, :, :, :]
        raw = self.output(hidden).squeeze(-1) / math.sqrt(self.rank)
        return CAP * torch.tanh(raw)

    def forward(self, visual, base, force_off=False):
        if force_off:
            return base
        return corrected_scores(base, self.evidence(visual))


def corrected_scores(base, delta, temperature=TEMPERATURE):
    if base.shape != delta.shape or base.ndim != 3:
        raise ValueError('Full [batch, object, attribute] grid required')
    offset = temperature * (torch.logsumexp((base + delta) / temperature, dim=-1, keepdim=True)
                            - torch.logsumexp(base / temperature, dim=-1, keepdim=True))
    return base + delta - offset


def conditional_loss(base, delta, attrs, objects):
    rows = torch.arange(len(base), device=base.device)
    return F.cross_entropy((base + delta)[rows, objects] / TEMPERATURE, attrs)


def probability_checks(base, delta):
    corrected = corrected_scores(base, delta)
    pb = torch.softmax(base.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    joint = torch.softmax(corrected.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    conditional = torch.softmax((base + delta) / TEMPERATURE, dim=-1)
    expected = pb.sum(-1, keepdim=True) * conditional
    return {'joint_sum_error': float((joint.sum((1, 2)) - 1).abs().max()),
            'conditional_sum_error': float((conditional.sum(-1) - 1).abs().max()),
            'object_marginal_error': float((joint.sum(-1) - pb.sum(-1)).abs().max()),
            'joint_factorization_error': float((joint - expected).abs().max())}
