import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf
from typing import Any

from model_ranking.utils import is_ndarray


def h_score(features: NDArray[Any], labels: NDArray[Any]) -> float:
    r"""
    H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019)
    <http://yangli-feasibility.com/home/media/icip-19.pdf>`_.

    The H-Score :math:`\mathcal{H}` can be described as:

    .. math::
        \mathcal{H}=\operatorname{tr}\left(\operatorname{cov}(f)^{-1} \operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)

    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector

    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.

    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features
    y = labels

    covf = np.cov(f, rowvar=False)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score


def regularized_h_score(features: NDArray[Any], labels: NDArray[Any]) -> float:
    r"""
    Regularized H-score in `Newer is not always better: Rethinking transferability metrics, their peculiarities, stability and performance (NeurIPS 2021)
    <https://openreview.net/pdf?id=iz_Wwmfquno>`_.

    The  regularized H-Score :math:`\mathcal{H}_{\alpha}` can be described as:
    .. math::
        \mathcal{H}_{\alpha}=\operatorname{tr}\left(\operatorname{cov}_{\alpha}(f)^{-1}\left(1-\alpha \right)\operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)

    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector and :math:`\operatorname{cov}_{\alpha}` the  Ledoit-Wolf
    covariance estimator with shrinkage parameter :math:`\alpha`
    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.
    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features.astype("float64")
    f = f - np.mean(
        f, axis=0, keepdims=True
    )  # Center the features for correct Ledoit-Wolf Estimation
    y = labels

    C = int(y.max() + 1)
    g = np.zeros_like(f)

    cov = LedoitWolf(assume_centered=False).fit(f)
    alpha = cov.shrinkage_
    covf_alpha = cov.covariance_  # pyright: ignore[reportUnknownVariableType]

    assert is_ndarray(covf_alpha), f"Expected covf_alpha to be a numpy array"

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)  # pyright: ignore[reportUnknownVariableType]
    score = np.trace(
        np.dot(
            np.linalg.pinv(covf_alpha, rcond=1e-15),
            (1 - alpha) * covg,  # pyright: ignore[reportUnknownArgumentType]
        )
    )

    return score
