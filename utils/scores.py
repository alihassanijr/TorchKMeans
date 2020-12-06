import torch


def purity_score(y_true, y_pred):
    """
    Computest the purity score (clustering accuracy)

    Parameters
    ----------
    y_true : torch.Tensor[int] or torch.Tensor[long] and of shape (n_samples, )
        Ground truth labels

    y_pred : torch.Tensor[int] or torch.Tensor[long] and of shape (n_samples, )
        Predicted labels

    Returns
    -------
    accuracy : float
    """
    n = y_true.size(0)
    unique_classes = y_true.unique()
    unique_clusters = y_pred.unique()
    num_classes = len(unique_classes)
    num_clusters = len(unique_clusters)
    class_to_idx = {int(unique_classes[i]): i for i in range(num_classes)}
    cluster_to_idx = {int(unique_clusters[i]): i for i in range(num_clusters)}

    scores = torch.zeros((num_classes, num_clusters), dtype=torch.int16, device=y_true.device)
    for i in range(n):
        scores[class_to_idx[int(y_true[i])], cluster_to_idx[int(y_pred[i])]] += 1
    return scores.max(0).values.sum().item() / n
