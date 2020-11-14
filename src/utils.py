"""
Torch-based K-Means
by Ali Hassani
"""
import torch
from torch.nn import functional as F


def distance_clamp(x):
    return torch.clamp(x, 0.0, float("Inf"))


def similarity_clamp(x):
    return torch.clamp(x, 0.0, 1.0)


def squared_norm(x):
    return (x ** 2).sum(1).view(-1, 1)


def distance_matrix(x, y, x_norm=None, y_norm=None):
    x_norm = squared_norm(x) if x_norm is None else x_norm
    y_norm = squared_norm(y) if y_norm is None else y_norm.T
    mat = x_norm + y_norm - 2.0 * torch.mm(x, y.T)
    return distance_clamp(mat)


def self_distance_matrix(x):
    return distance_clamp(((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(2))


def similarity_matrix(x, y, pre_normalized=False):
    if pre_normalized:
        return similarity_clamp(x.matmul(y.T))
    return similarity_clamp(F.normalize(x, p=2, dim=1).matmul(F.normalize(y, p=2, dim=1).T))


def self_similarity_matrix(x, pre_normalized=False):
    if pre_normalized:
        return similarity_clamp(x.matmul(x.T))
    x_normalized = F.normalize(x, p=2, dim=1)
    return similarity_clamp(x_normalized.matmul(x_normalized.T))
