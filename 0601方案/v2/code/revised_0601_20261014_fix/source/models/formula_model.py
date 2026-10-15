"""0601 sections 5/6/7/8/9, with explicit, frozen CPF interfaces.

G/H: head output is the conditional logit itself, NEVER a CPF residual.
Object classifier: native cosine / T in BOTH train and inference.
Head: exactly the raw section 6.1 output, no undocumented temperature.
MVP conditional interface (not specified by the document): native pair
composition logit + broadcast native attribute logit. Recorded in protocol.
Description interface: native 768->300 projected global CLS, raw dot, sigmoid.
"""
import torch
from torch.nn import functional as F
from .ddm_utils import choose_texts, choose_visual_tokens
from .bf_logic_loss import pmean_violation


def centered_prior(k, eps=1e-8):
    return (k - k.mean()) / (k.std() + eps)


def compatibility(e_obj, e_attr, e_pair):
    proto = F.normalize((e_obj[:, None] + e_attr[None, :]) / 2, dim=-1)
    raw = F.cosine_similarity(e_pair, proto, dim=-1)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)


def joint_score(obj_logits, conditional_logits, prior=None, weight=0.0):
    score = obj_logits[:, :, None] + conditional_logits
    return score if prior is None else score + weight * prior[None, :, :]


def concept_nodes(obj_logits, masked_joint, image_feat, bank):
    """Literal section 7.2 activations. No second softmax inside rule loss."""
    return {
        'object': F.softmax(obj_logits, dim=-1),
        'attribute': F.softmax(torch.logsumexp(masked_joint, dim=1), dim=-1),
        'pair': torch.sigmoid(masked_joint),
        'parent': F.softmax(image_feat @ bank.parent.t(), dim=-1),
        'oe': torch.sigmoid(image_feat @ bank.oe.reshape(-1, 300).t()).reshape(-1, bank.oe.size(0), 3),
        'ae': torch.sigmoid(image_feat @ bank.ae.reshape(-1, 300).t()).reshape(-1, bank.ae.size(0), 3),
    }


def logic_from_nodes(nodes, bank):
    """Keep existing reconstructed OI/OE/AE operators, replace node grounding.

    Section 7.3 does not prescribe a separate pair rule: pair nodes are
    exposed and tested, not silently turned into a new fourth loss.
    """
    obj, parent, attr = nodes['object'], nodes['parent'], nodes['attribute']
    ids, q = bank.rules.object_parent, bank.rules.q
    oi_up = pmean_violation(obj * (1 - parent[:, ids]), q).mean()
    children = torch.stack([obj[:, ids == i].max(dim=1).values
                            for i in range(bank.rules.n_parents)], dim=1)
    oi_down = pmean_violation(parent * (1 - children), q).mean()
    oe = pmean_violation(obj.unsqueeze(-1) * (1 - nodes['oe']), q).mean()
    ae = pmean_violation(attr.unsqueeze(-1) * (1 - nodes['ae']), q).mean()
    oi = oi_up + oi_down
    raw = oi + oe + ae
    return {'loss_oi': oi, 'loss_oe': oe, 'loss_ae': ae,
            'loss_logic_raw': raw, 'loss_logic_weighted': bank.beta * raw}


class FormulaMixin:
    def __init__(self, dset, cfg):
        super().__init__(dset, cfg)
        self.formula_joint = bool(getattr(cfg.MODEL, 'formula_joint', False))
        if cfg.TRAIN.max_epoch >= cfg.TRAIN.aug_epoch:
            raise ValueError('This release requires epochs < aug_epoch for ALL groups; augmentation is not part of section 9')
        if self.full_enabled and not self.formula_joint:
            raise ValueError('Feature K must use section 9 joint scoring')
        if not bool(getattr(cfg.MODEL, 'rev_enabled', False)):
            print('LEGACY_FORMULA_0601_READY joint={} feature={} logic={} obj_temperature={} head_temperature=1'.format(
                self.formula_joint, self.full_enabled, self.bf_logic.enabled, self.classifier.temp), flush=True)

    def formula_features(self, batch):
        l3, l6, l9, l12, cls = self.feat_extractor(batch['img'])
        ew_o = self._kcc_obj_embeddings(self.tot_objs)
        ew_a = self._kcc_attr_embeddings(self.tot_attrs)
        # Existing CPF attention functions and layer fusion, not the recap's
        # underspecified replacement. Shared vectors are valid broadcast inputs.
        global_feat = self.objc_embedder(cls)
        guide = choose_texts(global_feat, ew_o) @ ew_o
        weights = choose_visual_tokens(guide, self.objc_embedder(l12))
        vo = global_feat + self.objc_embedder((l12 * weights.unsqueeze(-1)).sum(dim=1))
        parts = []
        for layer in (l9, l6, l3):
            weights = choose_visual_tokens(vo, self.objc_embedder(layer))
            parts.append(self.objc_embedder((layer * weights.unsqueeze(-1)).sum(dim=1)))
        va = self.attrc_embedder(torch.cat(parts, dim=1))
        return vo, va, global_feat, ew_o, ew_a

    def formula_logits(self, batch, objs, attrs):
        words = None
        if not self.full_enabled:
            if self.formula_joint:
                grid_o = self.tot_objs.repeat_interleave(self.num_attrs)
                grid_a = self.tot_attrs.repeat(self.num_objs)
            else:
                grid_o, grid_a = objs, attrs
            # Preserve B's native word-composer-before-backbone RNG ordering.
            words = self.compose(torch.cat((self._kcc_obj_embeddings(grid_o),
                                            self._kcc_attr_embeddings(grid_a)), dim=-1))
        vo, va, global_feat, ew_o, ew_a = self.formula_features(batch)
        obj = self.classifier(vo, ew_o, scale=True)
        attr = self.classifier(va, ew_a, scale=True)
        comp = None
        if self.full_enabled:
            pair = self.full_pair_embeddings if self.full_use_pair else None
            conditional = self.full_head(va, vo, self.W_a_K, self.W_o_K, pair)
        else:
            # Joint MVP needs all object x attribute logits for GT-object CE.
            # B only needs native train candidates for its unchanged task CE.
            comp = self.classifier(self.compose_visual(vo, va), words, scale=True)
            if self.formula_joint:
                conditional = attr[:, None, :] + comp.reshape(-1, self.num_objs, self.num_attrs)
            else:
                conditional = attr[:, None, :].expand(-1, self.num_objs, -1).clone()
                conditional[:, objs, attrs] = conditional[:, objs, attrs] + comp
        prior = self.K_oa_prior if self.kcc_calib else None
        score = joint_score(obj, conditional, prior, self.kcc_lambda)
        return {'obj': obj, 'attr': attr, 'comp': comp, 'conditional': conditional,
                'score': score, 'image_feat': global_feat}

    def formula_train_terms(self, batch, return_details=False):
        logits = self.formula_logits(batch, self.train_objs, self.train_attrs)
        selected = logits['score'][:, self.train_objs, self.train_attrs]
        mask = torch.zeros(self.num_objs, self.num_attrs, dtype=torch.bool, device=selected.device)
        mask[self.train_objs, self.train_attrs] = True
        masked = logits['score'].masked_fill(~mask[None], -torch.inf)
        if self.formula_joint:
            rows = torch.arange(batch['obj'].numel(), device=selected.device)
            conditioned = logits['conditional'][rows, batch['obj'], :]
            comp_loss = F.cross_entropy(selected, batch['pair'])
            attr_loss = F.cross_entropy(conditioned, batch['attr'])
        else:
            # B is external-only: CPF task CE stays native, not section 9.
            conditioned = logits['attr']
            comp_loss = F.cross_entropy(logits['comp'], batch['pair'])
            attr_loss = F.cross_entropy(conditioned, batch['attr'])
        obj_loss = F.cross_entropy(logits['obj'], batch['obj'])
        task = comp_loss + 0.6 * attr_loss + 0.4 * obj_loss
        zero = task.new_zeros(())
        terms = {k: zero for k in ('loss_oi', 'loss_oe', 'loss_ae', 'loss_logic_raw', 'loss_logic_weighted')}
        nodes = None
        if self.bf_logic.enabled and self.bf_logic.beta > 0:
            nodes = concept_nodes(logits['obj'], masked, logits['image_feat'], self.bf_logic)
            terms = logic_from_nodes(nodes, self.bf_logic)
        out = {'loss_total': task + terms['loss_logic_weighted'], 'loss_cpf': task,
               'loss_composition': comp_loss, 'loss_object': obj_loss,
               'loss_conditional_attr': attr_loss,
               'full_delta_rms': logits['conditional'].square().mean().sqrt() if self.full_enabled else zero,
               'acc_pair': (selected.argmax(1) == batch['pair']).float().mean() if self.formula_joint
                           else (logits['comp'].argmax(1) == batch['pair']).float().mean(),
               'acc_attr': (conditioned.argmax(1) == batch['attr']).float().mean(),
               'acc_obj': (logits['obj'].argmax(1) == batch['obj']).float().mean()}
        out.update(terms)
        return (out, logits, nodes) if return_details else out

    def train_forward(self, batch):
        if not self.formula_joint and not self.bf_logic.enabled:
            return super().train_forward(batch)  # exact original A/C branch
        return self.formula_train_terms(batch)

    def train_forward_augment(self, batch, lam=0.5):
        raise RuntimeError('Formula protocol excludes minority augmentation; never silently bypass K')

    def val_forward(self, batch):
        if not self.formula_joint:
            return super().val_forward(batch)  # A/B/C: original CPF inference
        logits = self.formula_logits(batch, self.val_objs, self.val_attrs)
        pred = logits['score'][:, self.val_objs, self.val_attrs]
        if pred.size(1) != len(self.val_pairs):
            raise ValueError('Evaluation candidate order/count mismatch')
        return {'pred': pred, 'scores': {p: pred[:, i] for i, p in enumerate(self.val_pairs)}}
