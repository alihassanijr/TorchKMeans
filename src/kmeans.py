"""
Torch-based K-Means
by Ali Hassani
"""
import random

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

    init : 'random', 'k-means++' or torch.Tensor of shape (n_clusters, n_features)
        Initial centroid coordinates

    n_init : int, default=10
        Number of initializations, ignored if init is torch.Tensor.

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
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=200, spherical=False, eps=1e-6):
        self.n_clusters = n_clusters
        self.init_method = 'k-means++' if type(init) is not str else init
        self.cluster_centers_ = init if type(init) is torch.Tensor else None
        self.n_init = max(1, int(n_init))
        self.max_iter = max_iter
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        self.spherical = spherical
        self.eps = eps

    def _initialize(self, x):
        """
        Initializes the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        if self.cluster_centers_ == 'k-means++':
            return self._initialize_kpp(x)
        elif self.cluster_centers_ == 'random':
            return self._initialize_random(x)
        else:
            raise NotImplementedError("Initialization `{}` not supported.".format(self.cluster_centers_))

    def _initialize_kpp(self, x):
        """
        Initializes the centroid coordinates using K-Means++.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        # TODO: Implement K-Means++
        raise NotImplementedError("K-Means++ not implemented yet.")

    def _initialize_random(self, x):
        """
        Initializes the centroid coordinates by randomly selecting from the training samples.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.cluster_centers_ = x[random.sample(range(x.size(0)), self.n_clusters), :]
        return self

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
        if self.cluster_centers_ is None:
            # TODO: Cleaner and faster multi-init implementation
            self._initialize(x)
            inertia_list = np.zeros(self.n_init, dtype=float)
            n_iter_list = np.zeros(self.n_init, dtype=int)
            centroid_list = []
            label_list = []
            for run in range(self.n_init):
                self.fit(x)
                inertia_list[run] = self.inertia_
                n_iter_list[run] = self.n_iter_
                centroid_list.append(self.cluster_centers_)
                label_list.append(self.labels_)
            best_idx = int(np.argmax(inertia_list) if self.spherical else np.argmin(inertia_list))
            self.cluster_centers_ = centroid_list[best_idx]
            self.n_iter_ = n_iter_list[best_idx]
            self.inertia_ = inertia_list[best_idx]
            return self

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
            cluster_count = np.zeros(self.n_clusters, dtype=int)
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
