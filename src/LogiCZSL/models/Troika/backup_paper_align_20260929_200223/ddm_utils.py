import torch
import torch.nn.functional as F


def choose_texts(query, key):
    """
    Args:
        query: tensor with shape [B, D]
        key: tensor with shape [N, D]

    Returns:
        Attention weights with shape [B, N]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    score = query @ key.t()
    score = score / (query.size(-1) ** 0.5)

    return F.softmax(score, dim=-1)


def choose_visual_tokens(query, key):
    """
    Args:
        query: tensor with shape [B, D]
        key: tensor with shape [B, L, D]

    Returns:
        Attention weights with shape [B, L]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    score = torch.einsum("bld,bd->bl", key, query)
    score = score / (query.size(-1) ** 0.5)

    return F.softmax(score, dim=-1)


def image_soft_weights(query, key):
    """
    Args:
        query: tensor with shape [B, D]
        key: tensor with shape [B, L, D]

    Returns:
        Attention weights with shape [B, L]
    """
    return choose_visual_tokens(query, key)
