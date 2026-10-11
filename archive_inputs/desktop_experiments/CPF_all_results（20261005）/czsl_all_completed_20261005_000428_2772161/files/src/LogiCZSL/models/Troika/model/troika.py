import math

import sys

import torch
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *
from model.logicloss import LogicLoss
from .ddm_utils import *

class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Disentangler(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1_fc = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q


class Troika(nn.Module):
    
    def __init__(self, config, dataset, attributes, classes, offset):  # hc_parents, oda_children , ada_children
        super().__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.cross_attn_dropout = config.cross_attn_dropout if hasattr(config, 'cross_attn_dropout') else 0.1
        self.prim_loss_weight = config.prim_loss_weight if hasattr(config, 'prim_loss_weight') else 1

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()   #, oe_ctx_vectors

        self.enable_pos_emb = True

        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # # """Logic"""
        # self.dataset = dataset
        # self.logicloss = LogicLoss(dataset=dataset)
        # # HC
        # self.hc_parents = hc_parents
        # self.hc_parents_embeddings = self.hc_knowledge_embeddings()
        # # ODA
        # self.oda_children = oda_children
        # self.oda_children_embeddings = self.oda_knowledge_embeddings()  # [len(obj), 3, dim]
        # # ADA
        # self.ada_children = ada_children
        # self.ada_children_embeddings = self.ada_knowledge_embeddings()

        # """Logic"""

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.clip.visual.output_dim        # 768

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.attr_disentangler = Disentangler(output_dim, output_dim)
        self.attr_fuse = Disentangler(output_dim * 3, output_dim)
        self.obj_disentangler = Disentangler(output_dim, output_dim)

        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout) for _ in range(config.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

        # Keep CLIP and newly created modules in the same dtype.
        self.float()
        self.dtype = torch.float32
        self.text_encoder.dtype = torch.float32

        # self.compose_visual = Disentangler(output_dim*2, output_dim)
        # attr_emb_modules = [
        #     nn.Linear(768 * 3, 768)
        # ]
        #
        # self.attrc_embedder = nn.Sequential(*attr_emb_modules)

    #     # visual
    #     self.projection_1 = MLP(768 * 2, 768, 768, 2)
    #
    # def compose_visual(self, obj_feats, att_feats):
    #     inputs = torch.cat([obj_feats, att_feats], 1)
    #     output = self.projection_1(inputs)
    #     return output

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, bottleneck=self.config.adapter_dim, dropout=self.config.adapter_dropout) for _ in range(adapter_num)])
        return params

    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        feature_index = [5, 11, 17, 23]
        feature = []
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual
            # x = x + residual
            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            # x = x + residual

            if i_block in feature_index:
                img_feature = x.permute(1, 0, 2)  # LND -> NLD
                img_feature = self.clip.visual.ln_post(img_feature)
                if self.clip.visual.proj is not None:
                    img_feature = img_feature @ self.clip.visual.proj
                feature.append(img_feature[:, 1:, :])
                if i_block == 23:
                    feature.append(img_feature[:, 0, :])
        return feature # img_feature[:, 0, :], img_feature

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    # """Logic"""
    # def hc_knowledge_embeddings(self):
    #     hc_parents_tokenized = torch.cat([self.tokenizer(tok, context_length=self.config.context_length) for tok in self.hc_parents])   # [4, 8]  "a photo of " + tok
    #     embeddings = self.clip.token_embedding(hc_parents_tokenized.cuda())  # [4, 8, 768]
    #
    #     hc_parents_token_embedding = torch.zeros((len(self.hc_parents), embeddings.size(-1)), )
    #     for idx, rep in enumerate(embeddings):
    #         eos_idx = hc_parents_tokenized[idx].argmax()
    #         hc_parents_token_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    #
    #     return hc_parents_token_embedding
    #
    # def oda_knowledge_embeddings(self):
    #     oda_children_tokenized = torch.cat([self.tokenizer(tok, context_length=self.config.context_length) for tok in self.oda_children])   # [*3, 8]
    #     embeddings = self.clip.token_embedding(oda_children_tokenized.cuda())  # [*3, 8, 768]
    #
    #     oda_children_token_embedding = torch.zeros((len(self.oda_children), embeddings.size(-1)), )
    #     for idx, rep in enumerate(embeddings):
    #         eos_idx = oda_children_tokenized[idx].argmax()
    #         oda_children_token_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    #
    #     return oda_children_token_embedding
    #
    # def ada_knowledge_embeddings(self):
    #     ada_children_tokenized = torch.cat([self.tokenizer(tok, context_length=self.config.context_length) for tok in self.ada_children])
    #     embeddings = self.clip.token_embedding(ada_children_tokenized.cuda())
    #
    #     ada_children_token_embedding = torch.zeros((len(self.ada_children), embeddings.size(-1)), )
    #     for idx, rep in enumerate(embeddings):
    #         eos_idx = ada_children_tokenized[idx].argmax()
    #         ada_children_token_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    #
    #     return ada_children_token_embedding
    #
    # """Logic"""

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template, context_length=self.config.context_length).cuda()

        tokenized = torch.cat([self.tokenizer(tok, context_length=self.config.context_length) for tok in self.attributes + self.classes])
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        soft_att_obj = torch.zeros((len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),)
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)
        
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        # Logic
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]    # len(self.hc_parents), len(self.oda_children), len(self.ada_children)
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(class_token_ids.cuda()).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1: len(self.comp_ctx_vectors) + 1, :] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1: len(self.attr_ctx_vectors) + 1, :] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1: len(self.obj_ctx_vectors) + 1, :] = self.obj_ctx_vectors.type(self.clip.dtype)
        # # hc
        # token_tensor[3][:, eos_idx[3] - 1, :] = self.hc_parents_embeddings[:].type(self.clip.dtype)
        # # oda
        # token_tensor[4][:, eos_idx[3] - 1, :] = self.oda_children_embeddings[:].type(self.clip.dtype)
        # # ada
        # token_tensor[5][:, eos_idx[3] - 1, :] = self.ada_children_embeddings[:].type(self.clip.dtype)

        return token_tensor

    def loss_calu(self, predict, target, step):
        # if step < 80:
        #     adaptive_weight = 0.    # control all logic elements
        # else:
        #     adaptive_weight = 1.

        loss_fn = CrossEntropyLoss()

        _, batch_attr, batch_obj, batch_target = target   # batch_hc_parent , batch_oda_child , batch_ada_child
        comp_logits, attr_logits, obj_logits = predict  # hc_parent_logits , oda_child_logits , ada_child_logits
        # oda_child_logits = oda_child_logits.reshape(oda_child_logits.shape[0], -1, 3)
        # ada_child_logits = ada_child_logits.reshape(ada_child_logits.shape[0], -1, 3)

        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        # batch_hc_parent = batch_hc_parent.cuda()
        # batch_oda_child = batch_oda_child.cuda()
        # batch_ada_child = batch_ada_child.cuda()
        loss_comp = loss_fn(comp_logits, batch_target) * self.config.pair_loss_weight
        loss_attr = loss_fn(attr_logits, batch_attr) * self.config.attr_loss_weight
        loss_obj = loss_fn(obj_logits, batch_obj) * self.config.obj_loss_weight
        # loss_hc = loss_fn(hc_parent_logits, batch_hc_parent) * adaptive_weight
        # loss_oda = (loss_fn(oda_child_logits[:, :, 0], batch_oda_child) +
        #             loss_fn(oda_child_logits[:, :, 1], batch_oda_child) +
        #             loss_fn(oda_child_logits[:, :, 2], batch_oda_child)) * adaptive_weight
        # loss_ada = (loss_fn(ada_child_logits[:, :, 0], batch_ada_child) +
        #             loss_fn(ada_child_logits[:, :, 1], batch_ada_child) +
        #             loss_fn(ada_child_logits[:, :, 2], batch_ada_child)) * adaptive_weight

        loss_ce = loss_comp + loss_attr + loss_obj #+ (loss_hc + loss_oda + loss_ada) * 0.01

        # loss_logic = self.logicloss([obj_logits, hc_parent_logits, oda_child_logits, attr_logits, ada_child_logits]) * adaptive_weight  # hc_parent_logits, oda_child_logits, ada_child_logits
        # loss_logic = loss_ce

        # loss = loss_ce + loss_logic

        return [loss_ce, loss_comp, loss_attr, loss_obj]

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        # comp_pred = F.softmax(comp_logits, dim=-1)
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)

        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = comp_logits[:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
            # weighted_attr_pred = attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            # weighted_obj_pred = obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            # comp_pred[:, i_comp] = comp_pred[:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred

        return comp_logits

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    def forward_for_open(self, batch, text_feats):
        # batch_img = batch[0].cuda()
        # b = batch_img.shape[0]
        # # l, _ = idx.shape
        # batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        # batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        # normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]
        #
        # logits = list()
        # for i_element in range(self.token_ids.shape[0]):
        #     idx_text_features = text_feats[i_element]
        #
        #     # CMT
        #     cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
        #     batch_patch = self.patch_norm(batch_patch)
        #     for layer in self.cmt:
        #         cmt_text_features = layer(cmt_text_features, batch_patch)
        #     cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)
        #
        #     cmt_text_features = cmt_text_features / cmt_text_features.norm(
        #         dim=-1, keepdim=True
        #     )
        #
        #     logits.append(torch.einsum("bd, bkd->bk", normalized_img_features[i_element],
        #             cmt_text_features * self.clip.logit_scale.exp()
        #     ))
        #     # logits.append(self.clip.logit_scale.exp() * normalized_img_features[i_element] @ idx_text_features.t())
        #
        # return logits

        logits = list()
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        # l, _ = idx.shape
        # new modified
        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.encode_image(batch_img.type(self.clip.dtype))

        # token_tensors = self.construct_token_tensors(idx)

        _concept = text_feats[0]
        concept = _concept / _concept.norm(dim=-1, keepdim=True)

        _attr_weight = text_feats[1]
        attr_weight = _attr_weight / _attr_weight.norm(dim=-1, keepdim=True)

        _obj_weight = text_feats[2]
        obj_weight = _obj_weight / _obj_weight.norm(dim=-1, keepdim=True)

        vis_obj = self.obj_disentangler(cls_token)

        _vis_obj = vis_obj / vis_obj.norm(dim=-1, keepdim=True)
        '''choose text'''
        cls_text_scores = choose_texts(_vis_obj, obj_weight)  # b * n
        obj_guide_texts = cls_text_scores @ obj_weight  # b, d

        '''choose visual tokens'''
        # img_feat_l12_copy = img_feat_l12
        text_visual_scores = choose_visual_tokens(obj_guide_texts, img_feat_l12)
        text_guided_vis_obj = torch.sum(img_feat_l12 * text_visual_scores.unsqueeze(-1), dim=1)
        # text_guided_vis_obj = cls_token2
        vis_obj_new = vis_obj + self.obj_disentangler(text_guided_vis_obj) # * 0.1

        '''add'''
        # obj_guide = self.obj_guide_embedder(vis_obj)    # 64, 768
        # obj_guide = self.obj_guide_embedder(vis_obj_new)
        # vis_obj2 = self.objc_embedder_2(obj_guide)
        # vis_obj_all = vis_obj_new + vis_obj2

        img_feat_l3_scores = image_soft_weights(vis_obj_new, img_feat_l3)
        img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768

        img_feat_l6_scores = image_soft_weights(vis_obj_new, img_feat_l6)
        img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)

        img_feat_l9_scores = image_soft_weights(vis_obj_new, img_feat_l9)
        img_feat_l9 = torch.sum(img_feat_l9 * img_feat_l9_scores.unsqueeze(-1), dim=1)

        fused_vis = torch.cat([img_feat_l9, img_feat_l6, img_feat_l3], dim=1)
        attr_emb = self.attr_disentangler(cls_token) + self.attr_fuse(fused_vis) # * 0.1

        # com_feature = self.compose_visual(torch.cat([vis_obj_new, attr_emb], dim=1)) + cls_token
        com_feature = cls_token / cls_token.norm(dim=-1, keepdim=True)
        att_feature = attr_emb / attr_emb.norm(dim=-1, keepdim=True)
        obj_feature = vis_obj_new / vis_obj_new.norm(dim=-1, keepdim=True)

        logits.append(self.clip.logit_scale.exp() * com_feature @ concept.t())
        logits.append(self.clip.logit_scale.exp() * att_feature @ attr_weight.t())
        logits.append(self.clip.logit_scale.exp() * obj_feature @ obj_weight.t())
        # attr_image = self.attr_disentangler(batch_img)

        # batch_img_features = [batch_img, attr_image, obj_image]
        # normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        # logits.append(self.clip.logit_scale.exp() * normalized_img_features[i_element] @ idx_text_features.t())

        return logits

    def forward(self, batch, idx, train=True):
        logits = list()
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        l, _ = idx.shape
        # new modified
        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.encode_image(batch_img.type(self.clip.dtype))

        token_tensors = self.construct_token_tensors(idx)

        _concept, _ = self.encode_text(self.token_ids[0], token_tensors[0], enable_pos_emb=self.enable_pos_emb,)
        concept = _concept / _concept.norm(dim=-1, keepdim=True)

        _attr_weight, _ = self.encode_text(self.token_ids[1], token_tensors[1], enable_pos_emb=self.enable_pos_emb, )
        attr_weight = _attr_weight / _attr_weight.norm(dim=-1, keepdim=True)

        _obj_weight, _ = self.encode_text(self.token_ids[2], token_tensors[2], enable_pos_emb=self.enable_pos_emb, )
        obj_weight = _obj_weight / _obj_weight.norm(dim=-1, keepdim=True)

        # normal_text_features = [concept, attr_weight, obj_weight]

        vis_obj = self.obj_disentangler(cls_token)

        _vis_obj = vis_obj / vis_obj.norm(dim=-1, keepdim=True)
        '''choose text'''
        cls_text_scores = choose_texts(_vis_obj, obj_weight)  # b * n
        obj_guide_texts = cls_text_scores @ obj_weight  # b, d

        '''choose visual tokens'''
        # img_feat_l12_copy = img_feat_l12
        text_visual_scores = choose_visual_tokens(obj_guide_texts, img_feat_l12)
        text_guided_vis_obj = torch.sum(img_feat_l12 * text_visual_scores.unsqueeze(-1), dim=1)
        # text_guided_vis_obj = cls_token2
        vis_obj_new = vis_obj # + self.obj_disentangler(text_guided_vis_obj) # * 0.1

        '''add'''
        # obj_guide = self.obj_guide_embedder(vis_obj)    # 64, 768
        # obj_guide = self.obj_guide_embedder(vis_obj_new)
        # vis_obj2 = self.objc_embedder_2(obj_guide)
        # vis_obj_all = vis_obj_new + vis_obj2

        img_feat_l3_scores = image_soft_weights(vis_obj_new, img_feat_l3)
        img_feat_l3 = torch.sum(img_feat_l3 * img_feat_l3_scores.unsqueeze(-1), dim=1)  # 64, 768

        img_feat_l6_scores = image_soft_weights(vis_obj_new, img_feat_l6)
        img_feat_l6 = torch.sum(img_feat_l6 * img_feat_l6_scores.unsqueeze(-1), dim=1)

        img_feat_l9_scores = image_soft_weights(vis_obj_new, img_feat_l9)
        img_feat_l9 = torch.sum(img_feat_l9 * img_feat_l9_scores.unsqueeze(-1), dim=1)

        fused_vis = torch.cat([img_feat_l9, img_feat_l6, img_feat_l3], dim=1)
        attr_emb = self.attr_disentangler(cls_token) + self.attr_fuse(fused_vis) * 0.1

        # com_feature = self.compose_visual(torch.cat([vis_obj_new, attr_emb], dim=1)) + cls_token
        com_feature = cls_token / cls_token.norm(dim=-1, keepdim=True)
        att_feature = attr_emb / attr_emb.norm(dim=-1, keepdim=True)
        obj_feature = vis_obj_new / vis_obj_new.norm(dim=-1, keepdim=True)

        # normalized_img_features = [com_feature, att_feature, obj_feature]

        logits.append(self.clip.logit_scale.exp() * com_feature @ concept.t())
        logits.append(self.clip.logit_scale.exp() * att_feature @ attr_weight.t())
        logits.append(self.clip.logit_scale.exp() * obj_feature @ obj_weight.t())


        # attr_image = self.attr_disentangler(batch_img)

        # batch_img_features = [batch_img, attr_image, obj_image]
        # normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        # logits.append(self.clip.logit_scale.exp() * normalized_img_features[i_element] @ idx_text_features.t())

        return logits
