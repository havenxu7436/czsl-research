"""Unified corrected 0601 follow-up, not a literal original-paper replica."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .formula_model import FormulaMixin
from .revision_math import (CoupledHead, joint_log_probability, make_prior,
                            js_divergence, descriptor_logprob, beta_at)


class RevisionMixin(FormulaMixin):
    def __init__(self, dset, cfg):
        super().__init__(dset, cfg)
        self.rev_epoch = 1
        self.rev_enabled = bool(getattr(cfg.MODEL, 'rev_enabled', False))
        if not self.rev_enabled:
            if self.formula_joint or self.full_enabled or self.bf_logic.enabled or self.kcc_text or self.kcc_calib:
                raise ValueError('This release permits rev_enabled=False ONLY for the unchanged native CPF baseline; old broken formula configs must not run silently')
            return
        if not self.formula_joint:
            raise ValueError('Revised probability model requires formula_joint')
        cache = Path(cfg.MODEL.kcc_cache)
        manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
        blob = (cache / 'revision_bank.pt').read_bytes()
        if hashlib.sha256(blob).hexdigest() != manifest['files_sha256']['revision_bank.pt']:
            raise ValueError('Revision bank checksum mismatch')
        bank = torch.load(cache / 'revision_bank.pt', map_location='cpu', weights_only=True)
        for name in ('oe', 'ae', 'oe_mask', 'ae_mask', 'parent', 'parent_ids', 'parent_mask', 'K'):
            self.register_buffer('rev_' + name, bank[name], persistent=False)
        if tuple(bank['K'].shape) != (self.num_objs, self.num_attrs):
            raise ValueError('Revision bank label axes differ')
        for name, n in (('oe', self.num_objs), ('ae', self.num_attrs)):
            value, mask = bank[name], bank[name + '_mask']
            if value.shape[0] != n or value.shape[-1] != 300 or mask.shape != value.shape[:2] or not mask.any(-1).all():
                raise ValueError('Invalid description bank/mask')
            if mask.dtype != torch.bool or not torch.isfinite(value).all() or not torch.allclose(value.norm(dim=-1),torch.ones(value.shape[:2]),atol=1e-5):
                raise ValueError('Descriptions must be finite unit vectors with boolean masks')
        for level in range(len(bank['parent'])):
            valid=bank['parent_mask'][level]
            count=int(valid.sum())
            ids=bank['parent_ids'][level]
            if count<2 or not torch.equal(valid,torch.arange(len(valid))<count) or ids.shape!=(self.num_objs,) or set(ids.tolist())!=set(range(count)):
                raise ValueError('Hierarchy must contain packed, exhaustive, nontrivial label partitions')
            vectors=bank['parent'][level,valid]
            if not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=-1),torch.ones(count),atol=1e-5):
                raise ValueError('Invalid parent embeddings')
        self.register_buffer('rev_prior', make_prior(bank['K'], cfg.MODEL.rev_prior_mode,
                             floor=float(cfg.MODEL.rev_prior_floor)), persistent=False)
        self.rev_attr_mode = cfg.MODEL.rev_attr_mode
        if self.rev_attr_mode not in ('gt', 'marginal'):
            raise ValueError('Unknown attribute loss')
        self.rev_gain = float(cfg.MODEL.rev_gain)
        if not math.isfinite(self.rev_gain) or self.rev_gain <= 0:
            raise ValueError('Residual gain must be positive and finite')
        self.rev_desc_temp = float(cfg.MODEL.rev_desc_temperature)
        self.rev_anchor = float(cfg.MODEL.rev_anchor)
        self.rev_delay, self.rev_ramp = int(cfg.MODEL.rev_beta_delay), int(cfg.MODEL.rev_beta_ramp)
        self.rev_depth = int(cfg.MODEL.rev_oi_depth)
        if not 1 <= self.rev_depth <= bank['parent'].size(0):
            raise ValueError('Missing meaningful hierarchy levels')
        if self.rev_desc_temp <= 0 or not math.isfinite(self.rev_desc_temp) or self.rev_anchor < 0 or not math.isfinite(self.rev_anchor) or self.rev_delay<0 or self.rev_ramp<0:
            raise ValueError('Invalid semantic constraint settings')
        self.rev_attention = cfg.MODEL.rev_visual_attention
        self.rev_attention_temp = float(cfg.MODEL.rev_attention_temperature)
        if self.rev_attention not in ('cosine', 'native') or self.rev_attention_temp <= 0:
            raise ValueError('Invalid visual attention')
        if bool(cfg.MODEL.rev_learn_lambda):
            if not self.kcc_calib or not 0 < self.kcc_lambda < 1:
                raise ValueError('Learnable lambda requires active prior and initial lambda in (0,1)')
            self.rev_lambda_raw = nn.Parameter(torch.tensor(math.log(self.kcc_lambda / (1-self.kcc_lambda))))
        if self.full_enabled:
            # New head replaces unused old head. Preserve native RNG/sampler state.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(700000 + int(cfg.TRAIN.seed))
                self.full_head = CoupledHead(rank=int(cfg.MODEL.rev_rank), chunk=int(cfg.MODEL.rev_object_chunk),
                                            static=bool(cfg.MODEL.rev_static_head), temperature=self.classifier.temp)
        print('REVISION_READY normalized_joint=True gain={} lambda={} beta={} desc_temperature={} depth={}'.format(
            self.rev_gain, self.kcc_lambda, self.bf_logic.beta, self.rev_desc_temp, self.rev_depth), flush=True)

    def set_epoch(self, epoch):
        self.rev_epoch = int(epoch)

    def effective_parameters(self):
        weight = self.rev_lambda_raw.sigmoid() if hasattr(self, 'rev_lambda_raw') else self.kcc_lambda
        return {'lambda_effective': float(weight), 'revision': 'unified_20261013',
                'score_is_log_joint_probability': self.rev_enabled,
                'prior_mode': getattr(self.cfg.MODEL, 'rev_prior_mode', None)}

    def formula_features(self, batch):
        if not self.rev_enabled or self.rev_attention == 'native':
            return super().formula_features(batch)
        l3, l6, l9, l12, cls = self.feat_extractor(batch['img'])
        eo, ea = self._kcc_obj_embeddings(self.tot_objs), self._kcc_attr_embeddings(self.tot_attrs)
        g = self.objc_embedder(cls)
        def attend(query, keys):
            return (torch.einsum('bd,bnd->bn', F.normalize(query, dim=-1),
                                  F.normalize(keys, dim=-1)) / self.rev_attention_temp).softmax(-1)
        text_weights = (F.normalize(g, dim=-1) @ F.normalize(eo, dim=-1).t() / self.rev_attention_temp).softmax(-1)
        guide = text_weights @ eo
        vo = g + self.objc_embedder((l12 * attend(guide, self.objc_embedder(l12))[:, :, None]).sum(1))
        parts = [self.objc_embedder((layer * attend(vo, self.objc_embedder(layer))[:, :, None]).sum(1))
                 for layer in (l9, l6, l3)]
        return vo, self.attrc_embedder(torch.cat(parts, -1)), g, eo, ea

    def formula_logits(self, batch, objs, attrs):
        if not self.rev_enabled:
            return super().formula_logits(batch, objs, attrs)
        go, ga = self.tot_objs.repeat_interleave(self.num_attrs), self.tot_attrs.repeat(self.num_objs)
        # Same native composition branch retained for every revised group.
        words = self.compose(torch.cat((self._kcc_obj_embeddings(go), self._kcc_attr_embeddings(ga)), -1))
        vo, va, g, eo, ea = self.formula_features(batch)
        obj = self.classifier(vo, eo, scale=True)
        attr = self.classifier(va, ea, scale=True)
        comp = self.classifier(self.compose_visual(vo, va), words, scale=True)
        base = comp.reshape(-1, self.num_objs, self.num_attrs) + attr[:, None]
        delta = torch.zeros_like(base)
        if self.full_enabled:
            pair = self.full_pair_embeddings if self.full_use_pair else None
            delta = self.rev_gain * self.full_head(va, vo, ea, eo, pair)
        u = base + delta
        weight = self.rev_lambda_raw.sigmoid() if hasattr(self, 'rev_lambda_raw') else self.kcc_lambda
        score, lo, lc = joint_log_probability(obj, u, self.rev_prior if self.kcc_calib else None, weight)
        return {'obj': obj, 'attr': attr, 'comp': comp, 'conditional': u,
                'score': score, 'log_obj': lo, 'log_conditional': lc, 'delta': delta, 'image_feat': g}

    def revision_logic(self, logits, batch):
        lo = logits['log_obj']
        la = torch.logsumexp(logits['score'], dim=1)
        terms = []
        for level in range(self.rev_depth):
            ids, valid = self.rev_parent_ids[level], self.rev_parent_mask[level]
            ph = F.log_softmax(F.normalize(logits['image_feat'], dim=-1) @
                              F.normalize(self.rev_parent[level, valid], dim=-1).t() / self.rev_desc_temp, -1)
            # Categorical object labels are disjoint events: P(parent)=SUM children.
            child = torch.stack([torch.logsumexp(lo[:, ids == j], -1) for j in range(int(valid.sum()))], -1)
            terms.append(js_divergence(child, ph))
        oe = descriptor_logprob(logits['image_feat'], self.rev_oe, self.rev_oe_mask, self.rev_desc_temp)
        ae = descriptor_logprob(logits['image_feat'], self.rev_ae, self.rev_ae_mask, self.rev_desc_temp)
        oi = torch.stack(terms).mean()
        le = js_divergence(lo, oe) + self.rev_anchor * F.nll_loss(oe, batch['obj'])
        lae = js_divergence(la, ae) + self.rev_anchor * F.nll_loss(ae, batch['attr'])
        raw = oi + le + lae
        weight = beta_at(self.rev_epoch, self.bf_logic.beta, self.rev_delay, self.rev_ramp)
        return {'loss_oi': oi, 'loss_oe': le, 'loss_ae': lae, 'loss_logic_raw': raw,
                'loss_logic_weighted': weight * raw}, {'object': lo.exp(), 'attribute': la.exp(),
                'pair': logits['score'].exp(), 'oe': oe.exp(), 'ae': ae.exp()}

    def formula_train_terms(self, batch, return_details=False):
        if not self.rev_enabled:
            return super().formula_train_terms(batch, return_details)
        logits = self.formula_logits(batch, self.train_objs, self.train_attrs)
        selected = logits['score'][:, self.train_objs, self.train_attrs]
        rows = torch.arange(len(batch['obj']), device=selected.device)
        attribute = logits['log_conditional'][rows, batch['obj']] if self.rev_attr_mode == 'gt' else torch.logsumexp(logits['score'], 1)
        lp, lo, la = F.cross_entropy(selected, batch['pair']), F.nll_loss(logits['log_obj'], batch['obj']), F.nll_loss(attribute, batch['attr'])
        task = lp + 0.4 * lo + 0.6 * la
        zero = task.new_zeros(())
        terms = {k: zero for k in ('loss_oi','loss_oe','loss_ae','loss_logic_raw','loss_logic_weighted')}
        nodes = None
        if self.bf_logic.enabled:
            terms, nodes = self.revision_logic(logits, batch)
        out = dict(loss_total=task+terms['loss_logic_weighted'], loss_cpf=task, loss_composition=lp,
                   loss_object=lo, loss_conditional_attr=la, full_delta_rms=logits['delta'].square().mean().sqrt(),
                   acc_pair=(selected.argmax(-1)==batch['pair']).float().mean(),
                   acc_obj=(logits['obj'].argmax(-1)==batch['obj']).float().mean(),
                   acc_attr=(attribute.argmax(-1)==batch['attr']).float().mean())
        out.update(terms)
        return (out, logits, nodes) if return_details else out
