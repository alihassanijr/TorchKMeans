"""
Torch-based K-Means
by Ali Hassani
"""
import numpy as np
import torch
from .utils import distance_matrix, similarity_matrix


class KMeans:
    """
    K-Means
    Requires the number of clusters and the initial centroids

    Parameters
    ----------
    n_clusters : int
        The number of clusters or `K`

    init : torch.Tensor of shape (n_clusters, n_features)
        Initial centroid coordinates

    max_iter : int, default=200
        Maximum K-Means iterations

    spherical : bool, default=False
        Whether to use cosine similarity as the assignment metric or not

    eps : float, default=1e-6
        Threshold for early stopping

    Attributes
    ----------
    labels_ : torch.Tensor of shape (n_training_samples,)
        Training cluster assignments

    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Final centroid coordinates

    inertia_ : float
        Sum of squared errors when not spherical and sum of similarities when spherical

    n_iter_ : int
        The number of training iterations
    """
    def __init__(self, n_clusters, init, max_iter=200, spherical=False, eps=1e-6):
        self.n_clusters = n_clusters
        self.cluster_centers_ = init
        self.max_iter = max_iter
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        self.spherical = spherical
        self.eps = eps

    def _assignment(self, x):
        """
        Takes a set of samples and assigns them to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        if self.spherical:
            return self._spherical_assignment(x)
        return self._euclidean_assignment(x)

    def _euclidean_assignment(self, x):
        """
        Takes a set of samples and assigns them using L2 norm to the clusters w.r.t the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        dist = distance_matrix(x, self.cluster_centers_)
        return torch.argmin(dist, dim=1), torch.sum(torch.min(dist, dim=1).values)

    def _spherical_assignment(self, x):
        """
        Takes a set of samples and assigns them using cosine similarity to the clusters w.r.t the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        dist = similarity_matrix(x, self.cluster_centers_)
        return torch.argmax(dist, dim=1), torch.sum(torch.max(dist, dim=1).values)
