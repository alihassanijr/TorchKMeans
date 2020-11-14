"""
Torch-based K-Means
by Ali Hassani

K-Means++ initializer
"""

import numpy as np
import torch
from .utils import distance_matrix, squared_norm


def k_means_pp(x, n_clusters, x_norm=None):
    """
    K-Means++ initialization

    Based on Scikit-Learn's implementation

    Parameters
    ----------
    x : torch.Tensor of shape (n_training_samples, n_features)
    n_clusters : int
    x_norm : torch.Tensor of shape (n_training_samples, )

    Returns
    -------
    centroids : torch.Tensor of shape (n_clusters, n_features)
    """
    if x_norm is None:
        x_norm = squared_norm(x)
    n_samples, n_features = x.shape

    centroids = torch.zeros((n_clusters, n_features))

    n_local_trials = 2 + int(np.log(n_clusters))

    initial_centroid_idx = torch.randint(low=0, high=n_samples, size=(1,))[0]
    centroids[0, :] = x[initial_centroid_idx, :]

    dist_mat = distance_matrix(x=centroids[0, :].unsqueeze(0), y=x,
                               x_norm=x_norm[initial_centroid_idx, :].unsqueeze(0), y_norm=x_norm)
    current_potential = dist_mat.sum(1)

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_potential
        candidate_ids = torch.searchsorted(torch.cumsum(dist_mat.squeeze(0), dim=0), rand_vals)
        torch.clamp_max(candidate_ids, dist_mat.size(1) - 1, out=candidate_ids)

        distance_to_candidates = distance_matrix(x=x[candidate_ids, :], y=x,
                                                 x_norm=x_norm[candidate_ids, :], y_norm=x_norm)

        # TODO: torch implementation
        distance_to_candidates = torch.from_numpy(np.minimum(dist_mat.numpy(), distance_to_candidates.numpy()))

        candidates_potential = distance_to_candidates.sum(1)

        best_candidate = torch.argmin(candidates_potential)
        current_potential = candidates_potential[best_candidate]
        dist_mat = distance_to_candidates[best_candidate].unsqueeze(0)
        best_candidate = candidate_ids[best_candidate]

        centroids[c, :] = x[best_candidate, :]

    return centroids
