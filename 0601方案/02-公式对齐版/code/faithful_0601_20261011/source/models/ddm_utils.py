# help to build ddm
import torch
import torch.nn.functional as F


# attention weights
def image_soft_weights(query, key):
    """
    query: [B, D]
    key:   [B, L, D]
    return: [B, L]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = torch.einsum("bld,bd->bl", key, query)
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)



def choose_texts(query, key):
    """
    query: [B, D]
    key:   [N, D]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = query @ key.t()
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)



def choose_visual_tokens(query, key):
    """
    query: [B, D]
    key:   [B, L, D]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = torch.einsum("bld,bd->bl", key, query)
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)



def image_learnable_soft_weights(query, key):
    '''
    Input:
        query: b * l * d
        key: b * l * d
    Output:
        weights: similarity weights
    '''


def text_learnable_soft_weights(query, key):
    '''
    Input:
        query: b * l * d
        key: b * l * d
    Output:
        weights: similarity weights
    '''
