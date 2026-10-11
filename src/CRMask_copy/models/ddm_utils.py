# help to build ddm
import torch
import torch.nn.functional as F


# attention weights
def image_soft_weights(query, key):
    '''
    Input:
        query: b * d
        key: b * l * d
    Output:
        weights: b * l (similarity weights)
    '''
    query = torch.ones(size=query.size()).cuda()
    key = torch.ones(size=key.size()).cuda()

    query = query.unsqueeze(-1)
    score = key @ query
    score = score.squeeze()
    score = F.softmax(score, dim=-1)
    return score


def choose_texts(query, key):
    '''
    Input:
        query: b * d
        key: n * d
    Output:
        weights: similarity weights
    '''
    score = query @ key.t()  # b * n
    score = F.softmax(score, dim=-1)

    return score


def choose_visual_tokens(query, key):
    '''
        Input:
            query: b * d
            key: b * l * d
        Output:
            weights: similarity weights
    '''
    # query = torch.ones(size=query.size()).cuda()
    # key = torch.ones(size=key.size()).cuda()

    query = query.unsqueeze(-1)
    score = key @ query
    score = score.squeeze()
    score = F.softmax(score, dim=-1)
    return score


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
