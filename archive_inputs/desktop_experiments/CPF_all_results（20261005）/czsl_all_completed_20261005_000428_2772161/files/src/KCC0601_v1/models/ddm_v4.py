import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .backbone import ViT
from .get_resnet import get_image_extractor
from .basic_layers import MLP, BasicConv
from .word_embedding_utils import initialize_wordembedding_matrix
from .ddm_utils import *



def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)


class DDM(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """

    def __init__(self, dset, cfg):
        super(DDM, self).__init__()
        self.cfg = cfg
        self.dset = dset
        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.pairs = torch.LongTensor(pairs).cuda()

        # Set training pairs.
        train_attrs, train_objs = zip(*dset.train_pairs)
        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()
        self.tot_attrs = torch.LongTensor(list(range(self.num_attrs))).cuda()   # 413
        self.tot_objs = torch.LongTensor(list(range(self.num_objs))).cuda()     # 674
        self.emb_dim = cfg.MODEL.emb_dim  # 300
        # print(self.num_attrs)
        # print(self.num_objs)
        # print(len(train_attrs))
        # print(len(train_objs))


        # Setup layers for word embedding composer.
        self._setup_word_composer(dset, cfg)  # g()

        # if not cfg.TRAIN.use_precomputed_features and not cfg.TRAIN.comb_features:
        #     self.feat_extractor = ViT('B_16', pretrained=True)
        self.feat_extractor = get_image_extractor('resnet18_conv')

        '''modify for resnet18'''
        self.last_avg = nn.AvgPool2d(7, stride=1)  # 512
        self.last_linear = nn.Linear(512, self.emb_dim)  # 300

        self.last2first = nn.Linear(512, 64)
        self.last2second = nn.Linear(512, 128)
        self.last2third = nn.Linear(512, 256)

        self.first_avg = nn.AvgPool2d(4, stride=4)
        self.second_avg = nn.AvgPool2d(2, stride=2)

        feat_dim = cfg.MODEL.img_emb_dim  # 768

        obj_emb_modules = [
            nn.Linear(feat_dim, self.emb_dim)  # 768->300
        ]
        attr_emb_modules = [
            nn.Linear(64+128+256, self.emb_dim),  # 768*3->300
        ]

        '''add'''
        obj_guide_modules = [
            nn.Linear(self.emb_dim, feat_dim)
        ]
        obj_emb_modules_2 = [
            nn.Linear(feat_dim, self.emb_dim)
        ]

        self.objc_embedder = nn.Sequential(*obj_emb_modules)
        self.attrc_embedder = nn.Sequential(*attr_emb_modules)

        '''add'''
        self.obj_guide_embedder = nn.Sequential(*obj_guide_modules)
        self.objc_embedder_2 = nn.Sequential(*obj_emb_modules_2)

        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.attr_classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.obj_classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)

        self.crossentropy = CrossEntropyLoss()

        # visual
        self.projection_1 = MLP(
            self.word_dim * 2, self.emb_dim, self.emb_dim, 2, batchnorm=False,
            drop_input=cfg.MODEL.wordemb_compose_dropout
        )

    def compose_visual(self, obj_feats, att_feats):
        inputs = torch.cat([obj_feats, att_feats], 1)
        output = self.projection_1(inputs)
        return output

    def _setup_word_composer(self, dset, cfg):
        attr_wordemb, self.word_dim = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.attrs, cfg)  # glove
        obj_wordemb, _ = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.objs, cfg)

        self.attr_embedder = nn.Embedding(self.num_attrs, self.word_dim)
        self.obj_embedder = nn.Embedding(self.num_objs, self.word_dim)
        self.attr_embedder.weight.data.copy_(attr_wordemb)
        self.obj_embedder.weight.data.copy_(obj_wordemb)
        '''modify'''
        self.attr_embedder.requires_grad_(requires_grad=False)
        self.obj_embedder.requires_grad_(requires_grad=False)

        self.wordemb_compose = cfg.MODEL.wordemb_compose

        self.compose = nn.Sequential(
            nn.Linear(self.word_dim * 2, self.word_dim * 3),
            nn.BatchNorm1d(self.word_dim * 3),
            nn.ReLU(0.1),
            nn.Linear(self.word_dim * 3, self.word_dim * 2),
            nn.BatchNorm1d(self.word_dim * 2),
            nn.ReLU(0.1),
            nn.Linear(self.word_dim * 2, self.emb_dim)
        )

    def compose_word_embeddings(self, mode='train'):
        if mode == 'train':
            attr_emb = self.attr_embedder(self.train_attrs)  # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.train_objs)  # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self.attr_embedder(self.all_attrs)  # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self.attr_embedder(self.unseen_pair_attrs)  # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.unseen_pair_objs)
        else:
            attr_emb = self.attr_embedder(self.val_attrs)  # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.val_objs)  # # [n_pairs, word_dim].

        '''add'''
        # attr_emb = self.text_finetune(attr_emb)
        # obj_emb = self.text_finetune(obj_emb)
        concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        concept_emb = self.compose(concept_emb)

        # '''modify'''
        # concept_emb = obj_emb * attr_emb

        return concept_emb

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
        # obj_weight = self.obj_embedder(self.tot_objs)
        # attr_weight = self.attr_embedder(self.tot_attrs) # 440 300

        bs = img1.shape[0]
        concept = self.compose_word_embeddings(mode='train')  # (n_pairs, emb_dim)

        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)

        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs)

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

        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(), float(bs)),
            'acc_obj': torch.div(correct_obj.sum(), float(bs)),
            'acc_pair': torch.div(correct_comp.sum(), float(bs))
        }

        return out

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

        obj_weight = self.obj_embedder(self.tot_objs)  # n * d
        attr_weight = self.attr_embedder(self.tot_attrs)

        # 64, 196, 768
        # img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)  # cls_token from img_feat_l12

        first_feature, second_feature, third_feature, last_feature = self.feat_extractor(img1)
        cls_token = self.last_avg(last_feature).squeeze()  # 512
        vis_obj = self.last_linear(cls_token)   # 300

        img_feat_l3 = self.first_avg(first_feature).reshape(bs, 64, -1).permute(0, 2, 1)
        img_feat_l6 = self.second_avg(second_feature).reshape(bs, 128, -1).permute(0, 2, 1)
        img_feat_l9 = third_feature.reshape(bs, 256, -1).permute(0, 2, 1)

        # '''add'''
        # # obj_weight = self.text_finetune(obj_weight)
        # # attr_weight = self.text_finetune(attr_weight)
        #
        # # vis_obj = self.objc_embedder(cls_token)  # 768 -> 300
        #
        # '''choose text'''
        # cls_text_scores = choose_texts(vis_obj, obj_weight)    # b * n
        # obj_guide_texts = cls_text_scores @ obj_weight  # b, d
        #
        # # obj_guide_texts = self.text_linear(obj_guide_texts)
        #
        # '''choose visual tokens'''
        # img_feat_l12_copy = self.objc_embedder(img_feat_l12)
        # text_visual_scores = choose_visual_tokens(obj_guide_texts, img_feat_l12_copy)
        # cls_token2 = torch.sum(img_feat_l12 * text_visual_scores.unsqueeze(-1), dim=1)
        # text_guided_vis_obj = self.objc_embedder(cls_token2)
        # vis_obj_new = vis_obj + text_guided_vis_obj
        # # vis_obj_new = text_guided_vis_obj

        '''add'''
        # obj_guide = self.obj_guide_embedder(vis_obj)    # 64, 768
        obj_guide = self.obj_guide_embedder(vis_obj)
        vis_obj2 = self.objc_embedder_2(obj_guide)
        vis_obj_all = vis_obj + vis_obj2   # 300

        img_feat_l3_scores = image_soft_weights(self.last2first(obj_guide), img_feat_l3)
        img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768

        img_feat_l6_scores = image_soft_weights(self.last2second(obj_guide), img_feat_l6)
        img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)

        img_feat_l9_scores = image_soft_weights(self.last2third(obj_guide), img_feat_l9)
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
        # 0.4 0.6
        loss = comp_loss + 0.7 * obj_loss + 0.3 * attr_loss

        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(), float(bs)),
            'acc_obj': torch.div(correct_obj.sum(), float(bs)),
            'acc_pair': torch.div(correct_comp.sum(), float(bs))
        }

        return out

    def val_forward(self, batch):
        img1 = batch['img']
        bs = img1.shape[0]

        '''modify'''
        '''modify'''
        # composition text features ->
        concept = self.compose_word_embeddings(mode='val')  # (n_pairs, emb_dim)

        obj_weight = self.obj_embedder(self.tot_objs)  # n * d
        attr_weight = self.attr_embedder(self.tot_attrs)

        # 64, 196, 768
        # img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)  # cls_token from img_feat_l12

        first_feature, second_feature, third_feature, last_feature = self.feat_extractor(img1)
        cls_token = self.last_avg(last_feature).squeeze()  # 512
        vis_obj = self.last_linear(cls_token)  # 300

        img_feat_l3 = self.first_avg(first_feature).reshape(bs, 64, -1).permute(0, 2, 1)
        img_feat_l6 = self.second_avg(second_feature).reshape(bs, 128, -1).permute(0, 2, 1)
        img_feat_l9 = third_feature.reshape(bs, 256, -1).permute(0, 2, 1)

        # '''add'''
        # # obj_weight = self.text_finetune(obj_weight)
        # # attr_weight = self.text_finetune(attr_weight)
        #
        # # vis_obj = self.objc_embedder(cls_token)  # 768 -> 300
        #
        # '''choose text'''
        # cls_text_scores = choose_texts(vis_obj, obj_weight)    # b * n
        # obj_guide_texts = cls_text_scores @ obj_weight  # b, d
        #
        # # obj_guide_texts = self.text_linear(obj_guide_texts)
        #
        # '''choose visual tokens'''
        # img_feat_l12_copy = self.objc_embedder(img_feat_l12)
        # text_visual_scores = choose_visual_tokens(obj_guide_texts, img_feat_l12_copy)
        # cls_token2 = torch.sum(img_feat_l12 * text_visual_scores.unsqueeze(-1), dim=1)
        # text_guided_vis_obj = self.objc_embedder(cls_token2)
        # vis_obj_new = vis_obj + text_guided_vis_obj
        # # vis_obj_new = text_guided_vis_obj

        '''add'''
        # obj_guide = self.obj_guide_embedder(vis_obj)    # 64, 768
        obj_guide = self.obj_guide_embedder(vis_obj)
        vis_obj2 = self.objc_embedder_2(obj_guide)
        vis_obj_all = vis_obj + vis_obj2  # 300

        img_feat_l3_scores = image_soft_weights(self.last2first(obj_guide), img_feat_l3)
        img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768

        img_feat_l6_scores = image_soft_weights(self.last2second(obj_guide), img_feat_l6)
        img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)

        img_feat_l9_scores = image_soft_weights(self.last2third(obj_guide), img_feat_l9)
        img_feat_l9 = torch.sum(img_feat_l9 * img_feat_l9_scores.unsqueeze(-1), dim=1)

        # object_guided_mask = torch.mean(torch.stack([img_feat_l3_scores, img_feat_l6_scores, img_feat_l9_scores], dim=0), dim=0)

        fused_vis = torch.cat([img_feat_l9, img_feat_l6, img_feat_l3], dim=1)
        attr_emb = self.attrc_embedder(fused_vis)

        # obj prediction
        obj_pred = self.classifier(vis_obj_all, obj_weight, scale=False)
        # print(obj_weight.shape)
        # print(obj_pred.shape)
        obj_pred = obj_pred.index_select(1, self.pairs[:, 1])

        # attr prediction
        attr_pred = self.classifier(attr_emb, attr_weight, scale=False)
        # print(attr_weight.shape)
        # print(attr_pred.shape)
        attr_pred = attr_pred.index_select(1, self.pairs[:, 0])

        # composition prediction
        vis_comp = self.compose_visual(vis_obj_all, attr_emb)
        comp_pred = self.classifier(vis_comp, concept, scale=False)
        # print(concept.shape)

        # print(comp_pred.shape)
        # print(attr_pred.shape)
        # print(obj_pred.shape)
        # sys.exit(0)
        pred = comp_pred + attr_pred + obj_pred
        out = {'pred': pred, 'scores': {}}

        for _, pair in enumerate(self.val_pairs):
            out['scores'][pair] = pred[:, self.pair2idx[pair]]

        # modify to return mask attention map , text_conditional_mask, object_guided_mask
        return out

    def forward(self, x, flag=False):
        if self.training:
            if flag:
                out = self.train_forward_augment(x)
            else:
                out = self.train_forward(x)
        else:
            with torch.no_grad():
                # modify to get mask attention maps , text_conditional_mask, object_guided_mask
                out = self.val_forward(x)
        return out


class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred
