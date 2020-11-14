"""
Torch-based K-Means
by Ali Hassani

Common util functions
"""
import torch
from torch.nn import functional as F


def distance_clamp(x):
    """
    Clamps the distance matrix to prevent invalid values.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    x_out : torch.Tensor
    """
    return torch.clamp_min(x, 0.0)


def similarity_clamp(x):
    """
    Clamps the similarity matrix to prevent invalid values.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    x_out : torch.Tensor
    """
    return torch.clamp(x, 0.0, 1.0)


def squared_norm(x):
    """
    Computes and returns the squared norm of the input 2d tensor on dimension 1.
    Useful for computing euclidean distance matrix.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    x_squared_norm : torch.Tensor of shape (n, )
    """
    return (x ** 2).sum(1).view(-1, 1)


def row_norm(x):
    """
    Computes and returns the row-normalized version of the input 2d tensor.
    Useful for computing cosine similarity matrix.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    x_normalized : torch.Tensor of shape (n, m)
    """
    return F.normalize(x, p=2, dim=1)


def distance_matrix(x, y, x_norm=None, y_norm=None):
    """
    Returns the pairwise distance matrix between the two input 2d tensors.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)
    y : torch.Tensor of shape (p, m)
    x_norm : torch.Tensor of shape (n, ) or NoneType
    y_norm : torch.Tensor of shape (p, ) or NoneType

    Returns
    -------
    distance_matrix : torch.Tensor of shape (n, p)
    """
    x_norm = squared_norm(x) if x_norm is None else x_norm
    y_norm = squared_norm(y) if y_norm is None else y_norm.T
    mat = x_norm + y_norm - 2.0 * torch.mm(x, y.T)
    return distance_clamp(mat)


def self_distance_matrix(x):
    """
    Returns the self distance matrix of the input 2d tensor.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    distance_matrix : torch.Tensor of shape (n, n)
    """
    return distance_clamp(((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(2))


def similarity_matrix(x, y, pre_normalized=False):
    """
    Returns the pairwise similarity matrix between the two input 2d tensors.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)
    y : torch.Tensor of shape (p, m)
    pre_normalized : bool, default=False
        Whether the inputs are already row-normalized

    Returns
    -------
    similarity_matrix : torch.Tensor of shape (n, p)
    """
    if pre_normalized:
        return similarity_clamp(x.matmul(y.T))
    return similarity_clamp(row_norm(x).matmul(row_norm(y).T))


def self_similarity_matrix(x, pre_normalized=False):
    """
    Returns the self similarity matrix of the input 2d tensor.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)
    pre_normalized : bool, default=False
        Whether the input is already row-normalized

    Returns
    -------
    similarity_matrix : torch.Tensor of shape (n, n)
    """
    if pre_normalized:
        return similarity_clamp(x.matmul(x.T))
    x_normalized = row_norm(x)
    return similarity_clamp(x_normalized.matmul(x_normalized.T))
