"""
Torch-based K-Means++
by Ali Hassani
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
    """
    if x_norm is None:
        x_norm = squared_norm(x)
    n_samples, n_features = x.shape

    centers = torch.zeros((n_clusters, n_features))

    n_local_trials = 2 + int(np.log(n_clusters))

    initial_centroid_idx = torch.randint(low=0, high=n_samples, size=(1,))[0]
    centers[0, :] = x[initial_centroid_idx, :]

    dist_mat = distance_matrix(x=centers[0, :].unsqueeze(0), y=x, y_norm=x_norm)
    current_pot = dist_mat.sum(1)

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_pot
        candidate_ids = torch.searchsorted(torch.cumsum(dist_mat.squeeze(0), dim=0), rand_vals)
        torch.clamp_max(candidate_ids, dist_mat.size(1) - 1, out=candidate_ids)

        distance_to_candidates = distance_matrix(x=x[candidate_ids, :], y=x, y_norm=x_norm)

        # TODO: torch implementation
        distance_to_candidates = torch.from_numpy(np.minimum(dist_mat.numpy(), distance_to_candidates.numpy()))
        candidates_pot = distance_to_candidates.sum(1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        dist_mat = distance_to_candidates[best_candidate].unsqueeze(0)
        best_candidate = candidate_ids[best_candidate]

        centers[c, :] = x[best_candidate, :]

    return centers
