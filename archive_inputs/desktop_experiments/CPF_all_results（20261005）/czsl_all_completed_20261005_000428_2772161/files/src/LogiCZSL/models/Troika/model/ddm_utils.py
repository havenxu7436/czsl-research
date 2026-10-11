import torch.nn.functional as F


def image_soft_weights(query, key):
    query = query.unsqueeze(-1)
    score = key @ query
    score = score.squeeze()
    score = F.softmax(score, dim=-1)
    return score


def choose_texts(query, key):
    score = query @ key.t()
    score = F.softmax(score, dim=-1)
    return score


def choose_visual_tokens(query, key):
    query = query.unsqueeze(-1)
    score = key @ query
    score = score.squeeze()
    score = F.softmax(score, dim=-1)
    return score


def image_learnable_soft_weights(query, key):
    pass


def text_learnable_soft_weights(query, key):
    pass
