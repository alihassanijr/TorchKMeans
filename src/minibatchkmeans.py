"""
Torch-based K-Means
by Ali Hassani

K-Means
"""
import numpy as np
import torch
from .kmeans import _BaseKMeans


class MiniBatchKMeans(_BaseKMeans):
    """
    Mini Batch K-Means

    Parameters
    ----------
    n_clusters : int
        The number of clusters or `K`.

    init : 'random' or torch.Tensor of shape (n_clusters, n_features)
        Tensor of the initial centroid coordinates, one of the pre-defined methods {'random'}.

    n_init : int, default=10
        Number of initializations.

    max_iter : int, default=200
        Maximum K-Means iterations.

    metric : 'default' or callable, default='default'
        Distance metric when similarity_based=False and similarity metric otherwise. Default is 'default'
        which uses L2 distance and cosine similarity as the distance and similarity metrics respectively.
        The callable metrics should take in two tensors of shapes (n, d) and (m, d) and return a tensor of
        shape (n, m).

    similarity_based : bool, default=False
        Whether the metric is a similarity metric or not.

    eps : float, default=1e-6
        Threshold for early stopping.

    Attributes
    ----------
    labels_ : torch.Tensor of shape (n_training_samples,)
        Training cluster assignments

    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Final centroid coordinates

    inertia_ : float
        Sum of squared errors when not similarity_based and sum of similarities when similarity_based

    n_iter_ : int
        The number of training iterations
    """
    def __init__(self, n_clusters=None, init='random', n_init=10, max_iter=200, metric='default',
                 similarity_based=False, eps=1e-6):
        init = init if type(init) is torch.Tensor else 'random'
        super(MiniBatchKMeans, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
                                              metric=metric, similarity_based=similarity_based, eps=eps)

    def _initialize(self, dataloader):
        """
        Initializes the centroid coordinates.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        self
        """
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        if self.init_method == 'random':
            self._initialize_random(dataloader)
        else:
            raise NotImplementedError("Initialization `{}` not supported.".format(self.cluster_centers_))
        return self

    def _initialize_random(self, dataloader):
        """
        Initializes the centroid coordinates by randomly selecting from the training samples.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        self
        """
        if type(self.n_clusters) is not int:
            raise NotImplementedError("Randomized K-Means expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        # TODO: Better implementation of random initialization
        self.cluster_centers_, self.center_norm = dataloader.dataset.random_sample(self.n_clusters)
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
        # TODO: Implement MiniBatch K++
        raise NotImplementedError

    def fit(self, dataloader):
        """
        Initializes and fits the centroids using the samples given w.r.t the metric.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        self
        """
        inertia_list = np.zeros(self.n_init, dtype=float)
        n_iter_list = np.zeros(self.n_init, dtype=int)
        centroid_list = []
        label_list = []
        for run in range(self.n_init):
            self._initialize(dataloader)
            self._fit(dataloader)
            if self.n_init < 2:
                return self
            inertia_list[run] = self.inertia_
            n_iter_list[run] = self.n_iter_
            centroid_list.append(self.cluster_centers_)
            label_list.append(self.labels_)
        best_idx = int(np.argmax(inertia_list) if self.similarity_based else np.argmin(inertia_list))
        self.cluster_centers_ = centroid_list[best_idx]
        self.center_norm = self._normalize(self.cluster_centers_)
        self.n_iter_ = n_iter_list[best_idx]
        self.inertia_ = inertia_list[best_idx]
        return self

    def _fit(self, dataloader):
        """
        Fits the centroids using the samples given w.r.t the metric.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        self
        """
        self.inertia_ = None
        self.cluster_counts = np.zeros(self.n_clusters, dtype=int)
        for itr in range(self.max_iter):
            inertia = self._fit_epoch(dataloader)
            if self.inertia_ is not None and abs(self.inertia_ - inertia) < self.eps:
                self.inertia_ = inertia
                break
        self.n_iter_ = itr + 1
        return self

    def _fit_epoch(self, dataloader):
        # TODO: Cleaner and faster implementation
        inertia_ = 0
        for x, x_norm in dataloader:
            labels, inertia = self._assign(x, x_norm)
            inertia_ += inertia
            for c in range(self.n_clusters):
                idx = torch.where(labels == c)[0]
                self.cluster_counts[c] += len(idx)
                if len(idx) > 0:
                    lr = 1 / self.cluster_counts[c]
                    self.cluster_centers_[c, :] = ((1 - lr) * self.cluster_centers_[c, :]) + \
                                                  (lr * torch.sum(torch.index_select(x, 0, idx), dim=0))
            self.center_norm = self._normalize(self.cluster_centers_)
        return inertia_

    def transform(self, dataloader):
        """
        Assigns the samples given to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        label_list = []
        for x, x_norm in dataloader:
            labels, _ = self._assign(x, x_norm)
            label_list.append(labels)
        return torch.cat(label_list)

    def transform_tensor(self, x):
        """
        Assigns the samples given to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        labels, _ = self._assign(x)
        return labels

    def fit_transform(self, dataloader):
        """
        Fits the centroids using the samples given w.r.t the metric, returns the final assignments.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        self.fit(dataloader)
        return self.labels_
