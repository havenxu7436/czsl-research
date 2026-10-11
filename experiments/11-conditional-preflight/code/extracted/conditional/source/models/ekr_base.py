from .kcc_cpf import *
from .semantic_control import DDM as SemanticDDM

class DDM(SemanticDDM):
    def raw_forward(self, batch):
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
        return {'pred': pred, 'attr_feature': attr_emb}
