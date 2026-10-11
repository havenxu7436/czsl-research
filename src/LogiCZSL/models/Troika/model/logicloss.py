# Logit Loss
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
loss_fn = CrossEntropyLoss()

"""Logit HC"""
def hc_compose(obj_logits, hc_parents_logits, mapping, p=5.0):
    OBJ = F.softmax(obj_logits.float(), dim=-1)             # [b, num_objs]->[128, 12]
    PARENT = F.softmax(hc_parents_logits.float(), dim=-1)   # [b, num_parents]->[128,4]

    predicate = OBJ.clone()
    for k in mapping.keys():
        indices = mapping[k]
        predicate[:, indices] = OBJ[:, indices] - OBJ[:, indices] * PARENT[:, k:k+1]

    loss_compose = torch.pow(torch.mean(torch.pow(predicate, p), dim=0), 1.0 / p).mean()

    return loss_compose

def hc_decompose(obj_logits, hc_parents_logits, mapping, p=5.0):
    OBJ = F.softmax(obj_logits.float(), dim=-1)  # [b, num_objs]->[128, 12]
    PARENT = F.softmax(hc_parents_logits.float(), dim=-1)  # [b, num_parents]->[128,4]

    predicate = PARENT.clone()
    for k in mapping.keys():
        indices = mapping[k]
        predicate[:, k:k+1] = PARENT[:, k:k+1] - PARENT[:, k:k+1] * OBJ[:, indices].max(dim=1, keepdim=True)[0]

    loss_decompose = torch.pow(torch.mean(torch.pow(predicate, p), dim=0), 1.0/p).mean()

    return loss_decompose


def logit_hc(obj_logits, hc_parents_logits, dataset, p):
    loss_compose = hc_compose(obj_logits, hc_parents_logits, mapping=dataset.hc_dict2idx, p=p)
    loss_decompose = hc_decompose(obj_logits, hc_parents_logits, mapping=dataset.hc_dict2idx, p=p)
    hc_loss = loss_compose + loss_decompose

    return hc_loss


"""Logit ODA"""
def oda_decompose(obj_logits, oda_logits, mapping, p=5.0):
    OBJ = F.softmax(obj_logits.float(), dim=-1)     # 128, 12
    OBJ = OBJ.repeat(3, 1, 1)                # 3, 128, 12
    CHILD = F.softmax(oda_logits.float(), dim=-1)   # 3, 128, 12

    predicate = OBJ.clone()

    predicate[:, :, :] = OBJ[:, :, :] - OBJ[:, :, :] * CHILD[:, :, :]   # 3, 128, 12

    loss_decompose = torch.pow(torch.mean(torch.pow(predicate, p), dim=1), 1.0 / p).mean()

    return loss_decompose

def logit_oda(obj_logits, oda_logits, dataset, p):
    loss_decompose = oda_decompose(obj_logits, oda_logits, mapping=dataset, p=p)
    oda_loss = loss_decompose

    return oda_loss


"""Logit ADA"""
def ada_decompose(att_logits, ada_logits, mapping, p=5.0):
    ATT = F.softmax(att_logits.float(), dim=-1)  # 128, 12
    ATT = ATT.repeat(3, 1, 1)
    CHILD = F.softmax(ada_logits.float(), dim=-1)  # 3, 128, 12

    predicate = ATT.clone()

    predicate[:, :, :] = ATT[:, :, :] - ATT[:, :, :] * CHILD[:, :, :]  # 3, 128, 12

    loss_decompose = torch.pow(torch.mean(torch.pow(predicate, p), dim=1), 1.0 / p).mean()

    return loss_decompose

def logit_ada(att_logits, ada_logits, dataset, p):
    loss_decompose = ada_decompose(att_logits, ada_logits, mapping=dataset, p=p)
    ada_loss = loss_decompose

    return ada_loss


# """Logit IA"""
# def ia_decompose(com_logits, att_logits, obj_logits, mapping, p=5.0):
#     COM = F.softmax(com_logits.float(), dim=-1)  # batch_size, len(pair)
#     ATT = F.softmax(att_logits.float(), dim=-1)  # batch_size, len(attr)
#     OBJ = F.softmax(obj_logits.float(), dim=-1)  # batch_size, len(obj)
#
#     predicate_att = COM.clone()
#     predicate_obj = COM.clone()
#
#     for k in mapping.keys():
#         index_att, index_obj = mapping[k]
#         predicate_att[:, k:k+1] = COM[:, k:k+1] - COM[:, k:k+1] * ATT[:, index_att:index_att+1]
#         predicate_obj[:, k:k+1] = COM[:, k:k+1] - COM[:, k:k+1] * OBJ[:, index_obj:index_obj+1]
#
#     predicate = torch.cat((predicate_att[None, :, :], predicate_obj[None, :, :]), dim=0)
#
#     loss_decompose = torch.pow(torch.mean(torch.pow(predicate, p), dim=1), 1.0 / p).mean()
#     return loss_decompose
#
# def ia_attr_compose(com_logits, att_logits, mapping, p=5.0):
#     COM = F.softmax(com_logits.float(), dim=-1)  # batch_size, len(pair)
#     ATT = F.softmax(att_logits.float(), dim=-1)  # batch_size, len(attr)
#
#     predicate = ATT.clone()
#     for k in mapping.keys():
#         indices = mapping[k]
#         predicate[:, k:k+1] = ATT[:, k:k+1] - ATT[:, k:k+1] * COM[:, indices].max(dim=1, keepdim=True)[0]
#
#     loss_attr_compose = torch.pow(torch.mean(torch.pow(predicate, p), dim=0), 1.0/p).mean()
#     return loss_attr_compose
#
# def ia_obj_compose(com_logits, obj_logits, mapping, p=5.0):
#     COM = F.softmax(com_logits.float(), dim=-1)
#     OBJ = F.softmax(obj_logits.float(), dim=-1)
#
#     predicate = OBJ.clone()
#     for k in mapping.keys():
#         indices = mapping[k]
#         predicate[:, k:k+1] = OBJ[:, k:k+1] - OBJ[:, k:k+1] * COM[:, indices].max(dim=1, keepdim=True)[0]
#     loss_obj_compose = torch.pow(torch.mean(torch.pow(predicate, p), dim=0), 1.0/p).mean()
#
#     return loss_obj_compose
#
# def logit_ia(com_logits, att_logits, obj_logits, dataset):
#     # loss_decompose = ia_decompose(com_logits, att_logits, obj_logits, mapping=dataset.com_att_obj)
#     loss_attr_compose = ia_attr_compose(com_logits, att_logits, mapping=dataset.attr_com)
#     loss_obj_compose = ia_obj_compose(com_logits, obj_logits, mapping=dataset.obj_com)
#     loss = loss_attr_compose + loss_obj_compose
#
#     return loss


"""Class LogicLoss"""
class LogicLoss(nn.Module):
    def __init__(self, dataset):
        super(LogicLoss, self).__init__()
        self.dataset = dataset
        self.data_name = dataset.data_name
        self.p = 3.0  # 1.0, 3.0, 5.0, 8.0, 10.0 

    def forward(self, predict):
        obj_logits, hc_parents_logits, oda_child_logits, attr_logits,ada_child_logits = predict  # hc_parents_logits , oda_child_logits, ada_child_logits comp_logits, attr_logits,
        # HC
        loss_hc = logit_hc(obj_logits, hc_parents_logits, self.dataset, p=self.p)
        # # ODA
        loss_oda = logit_oda(obj_logits, oda_child_logits.permute(2, 0, 1), self.dataset, p=self.p)
        # ADA
        loss_ada = logit_ada(attr_logits, ada_child_logits.permute(2, 0, 1), self.dataset, p=self.p)

        loss = (loss_hc + loss_oda + loss_ada) * 0.01

        return loss
