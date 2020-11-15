"""
Torch-based K-Means
by Ali Hassani

K-Means
"""

import random
import numpy as np
import torch
from .utils import distance_matrix, similarity_matrix, squared_norm, row_norm
from ._kmeanspp import k_means_pp
from ._discern import discern


class KMeans:
    """
    K-Means
    Requires the number of clusters and the initial centroids

    Parameters
    ----------
    n_clusters : int or NoneType
        The number of clusters or `K`. Set to None ONLY when init = 'discern'.

    init : 'random', 'k-means++', 'discern' or torch.Tensor of shape (n_clusters, n_features)
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
    def __init__(self, n_clusters=None, init='k-means++', n_init=10, max_iter=200, spherical=False, eps=1e-6):
        self.n_clusters = n_clusters
        self.init_method = 'k-means++' if type(init) is not str else init
        self.cluster_centers_ = init if type(init) is torch.Tensor else None
        self.n_init = max(1, int(n_init))
        self.max_iter = max_iter
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        self.spherical = spherical
        self.center_norm = None
        self.eps = eps

    def _normalize(self, x):
        return row_norm(x) if self.spherical else squared_norm(x)

    def _initialize(self, x, x_norm):
        """
        Initializes the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        if self.init_method == 'k-means++':
            self._initialize_kpp(x, x_norm)
        elif self.init_method == 'random':
            self._initialize_random(x)
        else:
            raise NotImplementedError("Initialization `{}` not supported.".format(self.cluster_centers_))
        self.center_norm = self._normalize(self.cluster_centers_)
        return self

    def _initialize_kpp(self, x, x_norm):
        """
        Initializes the centroid coordinates using K-Means++.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        if type(self.n_clusters) is not int:
            raise NotImplementedError("K-Means++ expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        self.cluster_centers_ = k_means_pp(x, n_clusters=self.n_clusters, x_norm=x_norm if not self.spherical else None)
        return self

    def _initialize_discern(self, x, x_norm):
        """
        Initializes the centroid coordinates using DISCERN.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        self.cluster_centers_ = discern(x, n_clusters=self.n_clusters, x_norm=x_norm if self.spherical else None)
        return self

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
        if type(self.n_clusters) is not int:
            raise NotImplementedError("Randomized K-Means expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        self.cluster_centers_ = x[random.sample(range(x.size(0)), self.n_clusters), :]
        return self

    def _assign(self, x, x_norm=None):
        """
        Takes a set of samples and assigns them to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        if self.spherical:
            return self._spherical_assignment(x_norm)
        return self._euclidean_assignment(x, x_norm)

    def _euclidean_assignment(self, x, x_norm=None):
        """
        Takes a set of samples and assigns them using L2 norm to the clusters w.r.t the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or NoneType

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        dist = distance_matrix(x, self.cluster_centers_, x_norm=x_norm, y_norm=self.center_norm)
        return torch.argmin(dist, dim=1), torch.sum(torch.min(dist, dim=1).values)

    def _spherical_assignment(self, x_norm):
        """
        Takes a set of samples and assigns them using cosine similarity to the clusters w.r.t the centroid coordinates.

        Parameters
        ----------
        x_norm : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        dist = similarity_matrix(x_norm, self.center_norm, pre_normalized=True)
        return torch.argmax(dist, dim=1), torch.sum(torch.max(dist, dim=1).values)

    def fit(self, x, x_norm=None):
        """
        Fits the centroids using the samples given w.r.t the metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        x_norm = x_norm if x_norm is not None else self._normalize(x)
        if self.cluster_centers_ is None:
            inertia_list = np.zeros(self.n_init, dtype=float)
            n_iter_list = np.zeros(self.n_init, dtype=int)
            centroid_list = []
            label_list = []
            for run in range(self.n_init):
                self._initialize(x, x_norm)
                self.fit(x, x_norm)
                inertia_list[run] = self.inertia_
                n_iter_list[run] = self.n_iter_
                centroid_list.append(self.cluster_centers_)
                label_list.append(self.labels_)
            best_idx = int(np.argmax(inertia_list) if self.spherical else np.argmin(inertia_list))
            self.cluster_centers_ = centroid_list[best_idx]
            self.center_norm = self._normalize(self.cluster_centers_)
            self.n_iter_ = n_iter_list[best_idx]
            self.inertia_ = inertia_list[best_idx]
            return self

        for itr in range(self.max_iter):
            self.n_iter_ = itr
            labels, inertia = self._assign(x, x_norm)
            if self.inertia_ is not None and abs(self.inertia_ - inertia) < self.eps:
                self.labels_ = labels
                self.inertia_ = inertia
                break
            self.labels_ = labels
            self.inertia_ = inertia
            for c in range(self.n_clusters):
                idx = torch.where(labels == c)[0]
                self.cluster_centers_[c, :] = torch.mean(torch.index_select(x, 0, idx), dim=0)
            self.center_norm = self._normalize(self.cluster_centers_)
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
        return self._assign(x, self._normalize(x))

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
