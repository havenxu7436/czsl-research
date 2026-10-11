# Generated from this project's original models/ddm.py.
# A/C/D/E GloVe MVP. Original baseline layers/loss weights are inherited.
from .ddm import *
from .ddm import DDM as BaselineDDM
from .kcc_cache import KCCCacheMixin
from .bf_logic import CPFLogicAdapter


class DDM(KCCCacheMixin, BaselineDDM):
    def __init__(self, dset, cfg):
        super().__init__(dset, cfg)
        self._setup_kcc(dset, cfg)
        self.bf_logic = CPFLogicAdapter(dset, cfg)

    def compose_word_embeddings(self, mode='train'):
        if mode == 'train':
            attr_emb = self._kcc_attr_embeddings(self.train_attrs)  # [n_pairs, word_dim].
            obj_emb = self._kcc_obj_embeddings(self.train_objs)  # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self._kcc_attr_embeddings(self.all_attrs)  # [n_pairs, word_dim].
            obj_emb = self._kcc_obj_embeddings(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self._kcc_attr_embeddings(self.unseen_pair_attrs)  # [n_pairs, word_dim].
            obj_emb = self._kcc_obj_embeddings(self.unseen_pair_objs)
        else:
            attr_emb = self._kcc_attr_embeddings(self.val_attrs)  # [n_pairs, word_dim].
            obj_emb = self._kcc_obj_embeddings(self.val_objs)  # # [n_pairs, word_dim].

        '''add'''
        # attr_emb = self.text_finetune(attr_emb)
        # obj_emb = self.text_finetune(obj_emb)
        concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        concept_emb = self.compose(concept_emb)

        # '''modify'''
        # concept_emb = obj_emb * attr_emb

        return concept_emb

    def train_forward(self, batch):
        img1 = batch['img']  # 64, 3, 224, 224

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']

        bs = img1.shape[0]

        # text_o + text_a
        '''modify'''
        # composition text features ->
        concept = self.compose_word_embeddings(mode='train')  # (n_pairs, emb_dim)

        # 64, 196, 768
        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)  # cls_token from img_feat_l12

        obj_weight = self._kcc_obj_embeddings(self.tot_objs)   # n * d
        attr_weight = self._kcc_attr_embeddings(self.tot_attrs)

        # print(obj_weight.shape)
        # print(attr_weight.shape)


        '''add'''
        # obj_weight = self.text_finetune(obj_weight)
        # attr_weight = self.text_finetune(attr_weight)

        vis_obj = self.objc_embedder(cls_token)  # 768 -> 300

        '''choose text'''
        cls_text_scores = choose_texts(vis_obj, obj_weight)    # b * n
        obj_guide_texts = cls_text_scores @ obj_weight  # b, d

        cls_text_scores = choose_texts(vis_obj, attr_weight)  # b * n
        attr_guide_texts = cls_text_scores @ attr_weight  # b, d

        # obj_guide_texts = self.text_linear(obj_guide_texts)

        '''choose visual tokens'''
        img_feat_l12_copy = self.objc_embedder(img_feat_l12)
        text_visual_scores = choose_visual_tokens(obj_guide_texts, img_feat_l12_copy)
        cls_token2 = torch.sum(img_feat_l12 * text_visual_scores.unsqueeze(-1), dim=1)
        text_guided_vis_obj = self.objc_embedder(cls_token2)
        vis_obj_new = vis_obj + text_guided_vis_obj
        # vis_obj_new = text_guided_vis_obj

        # '''add'''
        # # obj_guide = self.obj_guide_embedder(vis_obj)    # 64, 768
        # obj_guide = self.obj_guide_embedder(vis_obj_new)
        # vis_obj2 = self.objc_embedder_2(obj_guide)
        # vis_obj_all = vis_obj_new + vis_obj2

        # cls_l3 = self.objc_embedder(cls_token_l3)
        # cls_text_scores_l3 = choose_texts(cls_l3, obj_weight)
        # obj_guide_texts_l3 = cls_text_scores_l3 @ obj_weight

        img_feat_l3_copy = self.objc_embedder(img_feat_l3)
        text_visual_scores_l3 = choose_visual_tokens(vis_obj_new, img_feat_l3_copy)
        cls_token_l3 = torch.sum(img_feat_l3 * text_visual_scores_l3.unsqueeze(-1), dim=1)
        text_guided_vis_obj_l3 = self.objc_embedder(cls_token_l3)
        vis_obj_new_l3 = text_guided_vis_obj_l3

        # cls_l6 = self.objc_embedder(cls_token_l6)
        # cls_text_scores_l6 = choose_texts(cls_l6, obj_weight)
        # obj_guide_texts_l6 = cls_text_scores_l6 @ obj_weight

        img_feat_l6_copy = self.objc_embedder(img_feat_l6)
        text_visual_scores_l6 = choose_visual_tokens(vis_obj_new, img_feat_l6_copy)
        cls_token_l6 = torch.sum(img_feat_l6 * text_visual_scores_l6.unsqueeze(-1), dim=1)
        text_guided_vis_obj_l6 = self.objc_embedder(cls_token_l6)
        vis_obj_new_l6 = text_guided_vis_obj_l6

        # cls_l9 = self.objc_embedder(cls_token_l9)
        # cls_text_scores_l9 = choose_texts(cls_l9, obj_weight)
        # obj_guide_texts_l9 = cls_text_scores_l9 @ obj_weight

        img_feat_l9_copy = self.objc_embedder(img_feat_l9)
        text_visual_scores_l9 = choose_visual_tokens(vis_obj_new, img_feat_l9_copy)
        cls_token_l9 = torch.sum(img_feat_l9 * text_visual_scores_l9.unsqueeze(-1), dim=1)
        text_guided_vis_obj_l9 = self.objc_embedder(cls_token_l9)
        vis_obj_new_l9 = text_guided_vis_obj_l9

        # img_feat_l3_scores = image_soft_weights(obj_guide, img_feat_l3)
        # img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768

        # img_feat_l6_scores = image_soft_weights(obj_guide, img_feat_l6)
        # img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)

        # img_feat_l9_scores = image_soft_weights(obj_guide, img_feat_l9)
        # img_feat_l9 = torch.sum(img_feat_l9 * img_feat_l9_scores.unsqueeze(-1), dim=1)

        # fused_vis = torch.cat([img_feat_l9, img_feat_l6, img_feat_l3], dim=1)
        fused_vis = torch.cat([vis_obj_new_l9, vis_obj_new_l6, vis_obj_new_l3], dim=1)
        # fused_vis = torch.cat(
        #     [torch.mean(img_feat_l9, dim=1), torch.mean(img_feat_l6, dim=1), torch.mean(img_feat_l3, dim=1)], dim=1)
        attr_emb = self.attrc_embedder(fused_vis)

        # obj prediction
        obj_pred = self.classifier(vis_obj_new, obj_weight)
        # obj_pred = self.classifier(vis_obj, obj_weight)

        # attr prediction
        attr_pred = self.classifier(attr_emb, attr_weight)

        # composition prediction
        vis_comp = self.compose_visual(vis_obj_new, attr_emb)
        comp_pred = self.classifier(vis_comp, concept)
        comp_pred = self._kcc_calibrate(comp_pred, self.train_objs, self.train_attrs)

        logic_terms = self.bf_logic(vis_obj_new, attr_emb, obj_pred, attr_pred, self.classifier)

        # obj loss and accuracy
        obj_loss = F.cross_entropy(obj_pred, batch['obj'])  # object classification
        obj_pred = torch.max(obj_pred, dim=1)[1]

        # attr loss and accuracy
        attr_loss = F.cross_entropy(attr_pred, batch['attr'])  # object classification
        attr_pred = torch.max(attr_pred, dim=1)[1]

        # composition loss and accuracy
        # comp_pred = comp_pred + obj_pred + attr_pred
        comp_loss = F.cross_entropy(comp_pred, pair_labels)
        comp_pred = torch.max(comp_pred, dim=1)[1]

        correct_obj = (obj_pred == obj_labels)
        correct_comp = (comp_pred == pair_labels)
        correct_attr = (attr_pred == attr_labels)

        loss = comp_loss + 0.4 * obj_loss + 0.6 * attr_loss

        loss_cpf = loss
        loss = loss + logic_terms['loss_logic_weighted']
        out = {
            'loss_total': loss,
            'loss_cpf': loss_cpf,
            'acc_attr': torch.div(correct_attr.sum(), float(bs)),
            'acc_obj': torch.div(correct_obj.sum(), float(bs)),
            'acc_pair': torch.div(correct_comp.sum(), float(bs))
        }

        out.update(logic_terms)
        return out

    def train_forward_augment(self, batch, lam=0.5):
        img1 = batch['img']
        img2_o = batch['img1_o']  # Image that shares the same object

        # Labels of 1st image.
        attr_labels_a = batch['attr']
        attr_labels_b = batch['attr1_o']
        obj_labels_a = batch['obj']
        obj_labels_b = batch['obj1_o']
        pair_labels_a = batch['pair']
        pair_labels_b = batch['idx1_o']

        # concat image & label
        img1 = torch.cat((img1, img2_o))
        pair_labels = torch.cat((pair_labels_a, pair_labels_b))
        obj_labels = torch.cat((obj_labels_a, obj_labels_b))
        attr_labels = torch.cat((attr_labels_a, attr_labels_b))

        # generate word embedding
        # obj_weight = self._kcc_obj_embeddings(self.tot_objs)
        # attr_weight = self._kcc_attr_embeddings(self.tot_attrs) # 440 300

        bs = img1.shape[0]
        concept = self.compose_word_embeddings(mode='train')  # (n_pairs, emb_dim)

        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)

        obj_weight = self._kcc_obj_embeddings(self.tot_objs)
        attr_weight = self._kcc_attr_embeddings(self.tot_attrs)

        # '''add'''
        # obj_weight = self.text_finetune(obj_weight)
        # attr_weight = self.text_finetune(attr_weight)

        vis_obj = self.objc_embedder(cls_token)

        '''add'''
        obj_guide = self.obj_guide_embedder(vis_obj)  # 64, 768
        vis_obj2 = self.objc_embedder_2(obj_guide)
        vis_obj_all = vis_obj + vis_obj2

        img_feat_l3_scores = image_soft_weights(obj_guide, img_feat_l3)
        img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768

        img_feat_l6_scores = image_soft_weights(obj_guide, img_feat_l6)
        img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)

        img_feat_l9_scores = image_soft_weights(obj_guide, img_feat_l9)
        img_feat_l9 = torch.sum(img_feat_l9 * img_feat_l9_scores.unsqueeze(-1), dim=1)

        fused_vis = torch.cat([img_feat_l9, img_feat_l6, img_feat_l3], dim=1)
        attr_emb = self.attrc_embedder(fused_vis)

        # obj prediction
        obj_pred = self.classifier(vis_obj_all, obj_weight)

        # attr prediction
        attr_pred = self.classifier(attr_emb, attr_weight)

        # composition prediction
        vis_comp = self.compose_visual(vis_obj_all, attr_emb)
        comp_pred = self.classifier(vis_comp, concept)
        comp_pred = self._kcc_calibrate(comp_pred, self.train_objs, self.train_attrs)

        logic_terms = self.bf_logic(vis_obj_all, attr_emb, obj_pred, attr_pred, self.classifier)

        # obj loss and accuracy
        obj_loss = F.cross_entropy(obj_pred, obj_labels)  # object classification
        obj_pred = torch.max(obj_pred, dim=1)[1]

        # attr loss and accuracy
        attr_loss = F.cross_entropy(attr_pred, attr_labels)  # object classification
        attr_pred = torch.max(attr_pred, dim=1)[1]

        # composition loss and accuracy
        comp_loss = F.cross_entropy(comp_pred, pair_labels)
        comp_pred = torch.max(comp_pred, dim=1)[1]

        # mma augmentation

        lam = np.random.beta(1, 1)

        # mixing attribute vectors
        new_attr = attr_weight[attr_labels_a] * lam + attr_weight[attr_labels_b] * (1 - lam)  # 64 300
        new_obj = obj_weight[obj_labels_a]

        # virtual ground class
        concept_gt = torch.cat((new_obj, new_attr), dim=-1)
        concept_gt = self.compose(concept_gt)  # 64 300

        # mixing visual vectors
        vis_comp_a, vis_comp_b = vis_comp.chunk(2, dim=0)
        vis_mix = lam * vis_comp_a + (1 - lam) * vis_comp_b

        # vector normalization
        vis_mix = F.normalize(vis_mix, dim=-1)  # 32 300
        concept_gt = F.normalize(concept_gt, dim=-1)  # 32, 300
        concept_all = F.normalize(concept, dim=-1)

        cpositive = torch.sum(vis_mix * concept_gt, dim=1, keepdim=True)  # 32, 1
        cnegative = vis_mix @ concept_all.T  # 32, 5592

        clogits = torch.cat([cpositive, cnegative], dim=1)
        clabels = torch.zeros(len(clogits), dtype=torch.long, device=cpositive.device)
        mix_loss = F.cross_entropy(clogits / 0.1, clabels)

        correct_obj = (obj_pred == obj_labels)
        correct_comp = (comp_pred == pair_labels)
        correct_attr = (attr_pred == attr_labels)

        loss = comp_loss + 0.4 * obj_loss + 0.6 * attr_loss + mix_loss

        loss_cpf = loss
        loss = loss + logic_terms['loss_logic_weighted']
        out = {
            'loss_total': loss,
            'loss_cpf': loss_cpf,
            'acc_attr': torch.div(correct_attr.sum(), float(bs)),
            'acc_obj': torch.div(correct_obj.sum(), float(bs)),
            'acc_pair': torch.div(correct_comp.sum(), float(bs))
        }

        out.update(logic_terms)
        return out

    def val_forward(self, batch):
        img = batch['img']

        '''modify'''
        concept = self.compose_word_embeddings(mode='val')  # [n_pairs, emb_dim].

        # img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img)
        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(
            img)

        obj_weight = self._kcc_obj_embeddings(self.tot_objs)
        attr_weight = self._kcc_attr_embeddings(self.tot_attrs)

        # print(obj_weight.shape)
        # print(attr_weight.shape)
        # sys.exit(0)

        '''add'''
        # obj_weight = self.text_finetune(obj_weight)
        # attr_weight = self.text_finetune(attr_weight)

        vis_obj = self.objc_embedder(cls_token)

        '''choose text'''
        cls_text_scores = choose_texts(vis_obj, obj_weight)  # b * n
        obj_guide_texts = cls_text_scores @ obj_weight  # b, d

        cls_text_scores = choose_texts(vis_obj, attr_weight)  # b * n
        attr_guide_texts = cls_text_scores @ attr_weight  # b, d

        # obj_guide_texts = self.text_linear(obj_guide_texts)

        '''choose visual tokens'''
        img_feat_l12_copy = self.objc_embedder(img_feat_l12)
        text_visual_scores = choose_visual_tokens(obj_guide_texts, img_feat_l12_copy)
        cls_token2 = torch.sum(img_feat_l12 * text_visual_scores.unsqueeze(-1), dim=1)
        text_guided_vis_obj = self.objc_embedder(cls_token2)
        vis_obj_new = vis_obj + text_guided_vis_obj
        # vis_obj_new = text_guided_vis_obj

        # cls_l3 = self.objc_embedder(cls_token_l3)
        # cls_text_scores_l3 = choose_texts(cls_l3, obj_weight)
        # obj_guide_texts_l3 = cls_text_scores_l3 @ obj_weight

        img_feat_l3_copy = self.objc_embedder(img_feat_l3)
        text_visual_scores_l3 = choose_visual_tokens(vis_obj_new, img_feat_l3_copy)
        cls_token_l3 = torch.sum(img_feat_l3 * text_visual_scores_l3.unsqueeze(-1), dim=1)
        text_guided_vis_obj_l3 = self.objc_embedder(cls_token_l3)
        vis_obj_new_l3 = text_guided_vis_obj_l3

        # cls_l6 = self.objc_embedder(cls_token_l6)
        # cls_text_scores_l6 = choose_texts(cls_l6, obj_weight)
        # obj_guide_texts_l6 = cls_text_scores_l6 @ obj_weight

        img_feat_l6_copy = self.objc_embedder(img_feat_l6)
        text_visual_scores_l6 = choose_visual_tokens(vis_obj_new, img_feat_l6_copy)
        cls_token_l6 = torch.sum(img_feat_l6 * text_visual_scores_l6.unsqueeze(-1), dim=1)
        text_guided_vis_obj_l6 = self.objc_embedder(cls_token_l6)
        vis_obj_new_l6 = text_guided_vis_obj_l6

        # cls_l9 = self.objc_embedder(cls_token_l9)
        # cls_text_scores_l9 = choose_texts(cls_l9, obj_weight)
        # obj_guide_texts_l9 = cls_text_scores_l9 @ obj_weight

        img_feat_l9_copy = self.objc_embedder(img_feat_l9)
        text_visual_scores_l9 = choose_visual_tokens(vis_obj_new, img_feat_l9_copy)
        cls_token_l9 = torch.sum(img_feat_l9 * text_visual_scores_l9.unsqueeze(-1), dim=1)
        text_guided_vis_obj_l9 = self.objc_embedder(cls_token_l9)
        vis_obj_new_l9 = text_guided_vis_obj_l9

        '''add'''
        # obj_guide = self.obj_guide_embedder(vis_obj_new)  # 64, 768
        # vis_obj2 = self.objc_embedder_2(obj_guide)
        # vis_obj_all = vis_obj_new + vis_obj2
        #
        # img_feat_l3_scores = image_soft_weights(obj_guide, img_feat_l3)
        # img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768
        #
        # img_feat_l6_scores = image_soft_weights(obj_guide, img_feat_l6)
        # img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)
        #
        # img_feat_l9_scores = image_soft_weights(obj_guide, img_feat_l9)
        # img_feat_l9 = torch.sum(img_feat_l9 * img_feat_l9_scores.unsqueeze(-1), dim=1)
        #
        # fused_vis = torch.cat([img_feat_l9, img_feat_l6, img_feat_l3], dim=1)
        # fused_vis = torch.cat([torch.mean(img_feat_l9, dim=1), torch.mean(img_feat_l6, dim=1), torch.mean(img_feat_l3, dim=1)], dim=1)
        fused_vis = torch.cat([vis_obj_new_l9, vis_obj_new_l6, vis_obj_new_l3], dim=1)
        attr_emb = self.attrc_embedder(fused_vis)

        # obj prediction
        obj_pred = self.classifier(vis_obj_new, obj_weight, scale=False)
        obj_pred = obj_pred.index_select(1, self.val_objs)

        # attr prediction
        attr_pred = self.classifier(attr_emb, attr_weight, scale=False)
        attr_pred = attr_pred.index_select(1, self.val_attrs)

        # composition prediction
        vis_comp = self.compose_visual(vis_obj_new, attr_emb)
        comp_pred = self.classifier(vis_comp, concept, scale=False)

        # print(comp_pred.shape)
        # print(attr_pred.shape)

        pred = comp_pred + attr_pred + obj_pred
        pred = self._kcc_calibrate(pred, self.val_objs, self.val_attrs)
        if pred.size(1) != len(self.val_pairs):
            raise ValueError("evaluation candidate count mismatch")
        return {'pred': pred, 'scores': {
            pair: pred[:, index] for index, pair in enumerate(self.val_pairs)
        }}
