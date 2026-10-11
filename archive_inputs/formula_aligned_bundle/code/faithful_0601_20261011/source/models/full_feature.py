"""0601 section 6.1 factorized head, adapted to native CPF B x 300 features.

Only score grids B x N x M are created. No B x N x M x D feature expansion.
Legacy residual helpers remain for disabled-feature A/C regression only.
FormulaMixin consumes head.forward directly and does NOT use these residual helpers.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .kcc_cache import file_sha256


class FactorizedKnowledgeConditionedHead(nn.Module):
    def __init__(self, d_vis=300, d_txt=300, hidden_dim=512):
        super().__init__()
        self.attr_vis_proj = nn.Linear(d_vis, hidden_dim)
        self.attr_txt_proj = nn.Linear(d_txt, hidden_dim)
        self.obj_vis_proj = nn.Linear(d_vis, hidden_dim)
        self.obj_txt_proj = nn.Linear(d_txt, hidden_dim)
        self.pair_bias = nn.Sequential(nn.Linear(d_txt, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, v_attr, v_obj, e_attr, e_obj, e_pair=None):
        qa = F.normalize(self.attr_vis_proj(v_attr), dim=-1)
        ka = F.normalize(self.attr_txt_proj(e_attr), dim=-1)
        qo = F.normalize(self.obj_vis_proj(v_obj), dim=-1)
        ko = F.normalize(self.obj_txt_proj(e_obj), dim=-1)
        # CPF provides one shared visual vector per image/branch.
        za = qa @ ka.t()
        zo = qo @ ko.t()
        z = zo[:, :, None] + za[:, None, :]
        if e_pair is not None:
            z = z + self.pair_bias(e_pair).squeeze(-1)[None, :, :]
        return z


def attribute_marginal(candidate_scores, candidate_attrs, n_attrs):
    """Logsumexp of ACTUAL training candidate scores, grouped by attribute."""
    values = []
    for i in range(n_attrs):
        mask = candidate_attrs == i
        if not bool(mask.any()):
            raise ValueError('Training candidates do not cover attribute ' + str(i))
        values.append(torch.logsumexp(candidate_scores[:, mask], dim=1))
    return torch.stack(values, dim=1)


class FullFeatureMixin:
    def _setup_full(self, dset, cfg):
        self.full_enabled = bool(getattr(cfg.MODEL, 'kcc_feature', False))
        self.full_use_pair = bool(getattr(cfg.MODEL, 'full_use_pair', True))
        self.full_gain = float(getattr(cfg.MODEL, 'full_gain', 1.0))
        if not self.full_enabled:
            return
        if not self.kcc_text or self.word_dim != 300:
            raise ValueError('Full head requires the audited Text K / 300d CPF')
        if cfg.TRAIN.max_epoch >= cfg.TRAIN.aug_epoch:
            raise ValueError('This first full release does not cover CPF minority augmentation')
        cache = Path(cfg.MODEL.kcc_cache)
        manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
        blocks = []
        for i in range(self.num_objs):
            name = 'W_ao_K/obj_{:05d}.pt'.format(i)
            if file_sha256(cache / name) != manifest['files_sha256'][name]:
                raise ValueError('Pair block checksum mismatch: ' + name)
            block = torch.load(cache / name, map_location='cpu', weights_only=True).float()
            if tuple(block.shape) != (self.num_attrs, self.word_dim) or not torch.isfinite(block).all():
                raise ValueError('Invalid pair block: ' + name)
            if not torch.allclose(block.norm(dim=-1), torch.ones(self.num_attrs), atol=1e-5):
                raise ValueError('Pair block is not unit normalized')
            blocks.append(block)
        self.register_buffer('full_pair_embeddings', torch.stack(blocks), persistent=False)
        # Preserve the CPF initialization and subsequent data-loader RNG stream.
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(600000 + int(cfg.TRAIN.seed))
            self.full_head = FactorizedKnowledgeConditionedHead(hidden_dim=512)
        print('FULL_FEATURE_READY hidden=512 pair={} gain={} parameters={}'.format(
            self.full_use_pair, self.full_gain, sum(p.numel() for p in self.full_head.parameters())), flush=True)

    def _full_grid(self, visual_obj, visual_attr, scale):
        pair = self.full_pair_embeddings if self.full_use_pair else None
        z = self.full_head(visual_attr, visual_obj, self.W_a_K, self.W_o_K, pair)
        return self.full_gain * z / self.classifier.temp if scale else self.full_gain * z

    def _full_train(self, visual_obj, visual_attr, obj_logits, attr_logits, comp_logits, obj_labels, attr_labels):
        zero = comp_logits.new_zeros(())
        info = {'full_delta_rms': zero, 'loss_conditional_attr': zero}
        if not self.full_enabled:
            return comp_logits, attr_logits, info
        delta = self._full_grid(visual_obj, visual_attr, scale=True)
        comp_logits = comp_logits + delta[:, self.train_objs, self.train_attrs]
        rows = torch.arange(obj_labels.numel(), device=obj_labels.device)
        conditional = attr_logits + delta[rows, obj_labels, :]
        info = {'full_delta_rms': delta.square().mean().sqrt(),
                'loss_conditional_attr': F.cross_entropy(conditional, attr_labels)}
        return comp_logits, conditional, info

    def _full_eval(self, pred, visual_obj, visual_attr):
        if not self.full_enabled:
            return pred
        delta = self._full_grid(visual_obj, visual_attr, scale=False)
        return pred + delta[:, self.val_objs, self.val_attrs]

    def _full_logic_attr(self, comp_logits, native_attr_logits):
        # B/F retain their audited native adapter; H follows section 7's marginal adapter.
        if not self.full_enabled:
            return native_attr_logits
        return attribute_marginal(comp_logits, self.train_attrs, self.num_attrs)
