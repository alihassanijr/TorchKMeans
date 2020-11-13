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


def distance_matrix(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    mat = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return distance_clamp(mat)


def self_distance_matrix(x):
    return distance_clamp(((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(2))


def similarity_matrix(x, y):
    return similarity_clamp(F.normalize(x, p=2, dim=1).matmul(F.normalize(y, p=2, dim=1).T))


def self_similarity_matrix(x):
    x_normalized = F.normalize(x, p=2, dim=1)
    return similarity_clamp(x_normalized.matmul(x_normalized.T))
