import numpy as np
from sklearn import preprocessing


def purity_score(y_true, y_pred):
    """
    Computest the purity score (clustering accuracy)

    Parameters
    ----------
    y_true : Tensor of type Int or Long and of shape (n_samples, )
        Ground truth labels

    y_pred : Tensor of type Int or Long and of shape (n_samples, )
        Predicted labels

    Returns
    -------
    accuracy : float
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    # Encoding the true labels, just to be on the safe side
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_true_np)

    # Calculate purity score
    num_class = len(np.unique(y_true_np))
    num_clusters = len(np.unique(y_pred_np))
    lbl = np.unique(y_pred_np)
    scores = np.zeros((num_class, num_clusters))
    for i in range(0, len(y)):
        scores[y[i], np.where(lbl == y_pred_np[i])[0]] += 1
    acc = np.sum(np.max(scores, axis=0))
    acc /= len(y)
    return acc
