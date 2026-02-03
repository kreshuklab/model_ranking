import numpy as np
from numpy.typing import NDArray
from typing import Any


def get_nuno(test_probs: NDArray[Any]) -> float:
    """Calculate the Normalized Nuclear Norm (NUNO) of the test probabilities.

    Args:
        test_probs (NDArray[Any]): N x k array where N is number of samples and k is number of classes,
        softmaxed probabilities for each sample and class.

    Returns:
        float: NUNO score
    """
    singular_values = np.linalg.svd(test_probs, compute_uv=False)
    nuclear_norm = np.sum(singular_values)
    min_dim = min(test_probs.shape[0], test_probs.shape[1])

    return nuclear_norm / np.sqrt(min_dim * test_probs.shape[0])
