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

    def _assign(self, x):
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

    def fit(self, x):
        """
        Fits the centroids using the samples given w.r.t the metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        for itr in range(self.max_iter):
            # TODO: Cleaner and faster K-Means implementation
            self.n_iter_ = itr
            labels, inertia = self._assign(x)
            if self.inertia_ is not None and abs(self.inertia_ - inertia) < self.eps:
                break
            self.labels_ = labels
            self.inertia_ = inertia
            cluster_centers = torch.zeros(self.cluster_centers_.shape, dtype=self.cluster_centers_.dtype,
                                          device=self.cluster_centers_.device)
            cluster_count = np.zeros(self.n_clusters)
            for i in range(x.size(0)):
                cluster_centers[self.labels_[i], :] += x[i, :]
                cluster_count[self.labels_[i]] += 1

            for c in range(self.n_clusters):
                cnt = cluster_count[c]
                if cnt > 0:
                    cluster_centers[c, :] /= cnt

            self.cluster_centers_ = cluster_centers
        return self

    def transform(self, x):
        """
        Assigns the samples given to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        return self._assign(x)

    def fit_transform(self, x):
        """
        Fits the centroids using the samples given w.r.t the metric, returns the final assignments.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        self.fit(x)
        return self.labels_
