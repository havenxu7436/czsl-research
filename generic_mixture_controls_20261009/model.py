"""Same generic architecture, decoupled content controls, fixed mixture READOUT."""
import math
import bootstrap
import numpy as np
import torch
from common import finite, seed_all, state_sha
from head import ConditionalHead, TEMPERATURE, corrected_scores, inputs
import runtime as legacy_runtime

ALPHA = 0.25
MODES = {'G_K': 'P_generic', 'G_name': 'P_name', 'G_perm1': 'P_perm1', 'G_perm2': 'P_perm2'}
NEW_VARIANTS = ('G_name', 'G_perm1', 'G_perm2')


class GenericHead(ConditionalHead):
    def __init__(self, bank, variant):
        if variant not in MODES:
            raise ValueError('Unknown generic content mode')
        # Always build the ORIGINAL generic architecture, with exactly its RNG use.
        super().__init__(bank, 'P_generic')
        self.content_mode = variant
        tables, self.permutations = inputs(bank, MODES[variant])
        for name, value in tables.items():
            getattr(self, name).copy_(value)

    def forward(self, visual, base, force_off=False):
        if force_off:
            return base
        return mix_scores(base, corrected_scores(base, self.evidence(visual)))


def fresh_head(bank, variant, controls, device):
    # Legacy discarded resume selfcheck requests P_K; use generic G_K there.
    variant = 'G_K' if variant == 'P_K' else variant
    seed_all(0)
    head = GenericHead(bank, variant)
    if head.permutations != controls['permutations'][MODES[variant]]:
        raise ValueError('Original fixed content permutation changed')
    return head.to(device)


def initializations(bank, controls):
    reports, shared = {}, None
    for variant in MODES:
        head = fresh_head(bank, variant, controls, 'cpu')
        state = head.state_dict()
        if shared is None:
            shared = state
        elif any(not torch.equal(value, state[k]) for k, value in shared.items()):
            raise ValueError('Generic control initialization differs')
        count = sum(p.numel() for p in head.parameters())
        if count != 60864 or not hasattr(head, 'fusion'):
            raise ValueError('Not the original ACTIVE generic architecture')
        reports[variant] = dict(initial_state_sha256=state_sha(head), parameter_count=count,
                                permutations=head.permutations, architecture='original_P_generic')
    return reports


def mix_scores(base, trained):
    if base.shape != trained.shape or base.ndim != 3:
        raise ValueError('Full [batch,object,attribute] grid required')
    finite(base, 'mixture/base')
    finite(trained, 'mixture/trained')
    if torch.equal(base, trained):
        return base  # Exact baseline identity, not log/exp roundoff.
    log_qb = torch.log_softmax(base / TEMPERATURE, dim=-1)
    log_qt = torch.log_softmax(trained / TEMPERATURE, dim=-1)
    log_qmix = torch.logaddexp(log_qb + math.log1p(-ALPHA), log_qt + math.log(ALPHA))
    mass = TEMPERATURE * torch.logsumexp(base / TEMPERATURE, dim=-1, keepdim=True)
    result = mass + TEMPERATURE * log_qmix
    finite(result, 'mixture/result')
    return result


def from_grid(base, unmixed, attrs, objects, indices=None):
    grid = mix_scores(base, unmixed)
    pb = torch.softmax(base.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    joint = torch.softmax(grid.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    qt = torch.softmax(unmixed / TEMPERATURE, dim=-1)
    expected = (1-ALPHA)*pb + ALPHA*pb.sum(-1, keepdim=True)*qt
    errors = dict(joint_sum_error=float((joint.sum((1, 2))-1).abs().max()),
                  object_marginal_error=float((joint.sum(-1)-pb.sum(-1)).abs().max()),
                  joint_factorization_error=float((joint-expected).abs().max()))
    tv = float(.5*(joint-pb).abs().sum((1, 2)).max())
    if max(errors.values()) > 1e-5 or tv > ALPHA+1e-5:
        raise ValueError('Mixture probability identity/TV bound failed')
    rows = torch.arange(len(base))
    log_q = torch.log_softmax(grid[rows, objects] / TEMPERATURE, dim=-1)
    nll = -log_q[rows, attrs]
    correct = log_q.argmax(1).eq(attrs)
    base_correct = base[rows, objects].argmax(1).eq(attrs)
    per_attr = [float(correct[attrs == i].float().mean()) for i in range(16) if (attrs == i).any()]
    diagnostics = dict(n_images=len(base), conditional_nll=float(nll.mean()),
                       gt_object_attr_accuracy=float(correct.float().mean()),
                       macro_attr_accuracy=float(np.mean(per_attr)), macro_attr_classes=len(per_attr),
                       corrected_from_baseline=int((correct & ~base_correct).sum()),
                       damaged_from_baseline=int((~correct & base_correct).sum()),
                       probability_errors=errors, joint_total_variation_max=tv,
                       readout='fixed_arithmetic_conditional_mixture', alpha=ALPHA)
    if indices is None:
        indices = torch.arange(len(base))
    vectors = dict(indices=indices.numpy(), attrs=attrs.numpy(), objects=objects.numpy(),
                   conditional_prediction=log_q.argmax(1).numpy(), baseline_correct=base_correct.numpy(),
                   correct=correct.numpy(), conditional_nll=nll.numpy())
    return grid, diagnostics, vectors


def scored(head, cache, device, indices=None):
    if indices is None:
        indices = torch.arange(len(cache['features']))
    unmixed, raw_diag, _ = legacy_runtime.scores_and_diagnostics(head, cache, device, indices)
    grid, diag, vectors = from_grid(cache['scores'][indices], unmixed, cache['attrs'][indices],
                                   cache['objects'][indices], indices)
    diag['unmixed_head_diagnostics'] = raw_diag
    diag['candidate_object_delta_std'] = raw_diag['candidate_object_delta_std']
    diag['max_abs_delta'] = raw_diag['max_abs_delta']
    return grid, diag, vectors, unmixed


def scores_and_diagnostics(head, cache, device, indices=None):
    return scored(head, cache, device, indices)[:3]
