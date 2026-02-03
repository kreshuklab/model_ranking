import numpy as np
from numpy.typing import NDArray

from typing import Any


def check_number_of_labels(n_labels: int, n_samples: int):
    """Check that number of labels are valid.
    Parameters
    ----------
    n_labels : int
        Number of labels.
    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )


def dispersion(x: NDArray[Any], labels: NDArray[Any]) -> float:
    """Calculate dispersion score

    Args:
        X (NDArray[Any]): 2D N x D array where N is number of samples and D is number of features
        labels (NDArray[Any]): 1D Array of labels corresponding to each sample

    Returns:
        float: Dispersion score
    """

    n_samples, _ = x.shape
    n_labels = len(np.unique(labels))

    check_number_of_labels(n_labels, n_samples)

    extra_disp = 0.0
    mean = np.mean(x, axis=0)
    for k in range(n_labels):
        cluster_k = x[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
    return np.log(extra_disp / (n_labels - 1.0))
