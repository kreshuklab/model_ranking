import numpy as np
from numpy.typing import NDArray
from typing import Any, List
from numba import njit  # pyright: ignore


def log_maximum_evidence(  # pyright: ignore
    features: NDArray[Any],
    targets: NDArray[Any],
    regression: bool = False,
    return_weights: bool = False,
):
    r"""
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
    for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`_.

    Args:
        features (np.ndarray): feature matrix from pre-trained model.
        targets (np.ndarray): targets labels/values.
        regression (bool, optional): whether to apply in regression setting. (Default: False)
        return_weights (bool, optional): whether to return bayesian weight. (Default: False)

    Shape:
        - features: (N, F) with element in [0, :math:`C_t`) and feature dimension F, where :math:`C_t` denotes the number of target class
        - targets: (N, ) or (N, C), with C regression-labels.
        - weights: (F, :math:`C_t`).
        - score: scalar.
    """
    f = features.astype(np.float64)
    y = targets
    if regression:
        y = targets.astype(np.float64)

    fh = f
    f = f.transpose()
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

    evidences: List[float] = []
    weights = []
    if regression:
        C = y.shape[1]
        for i in range(C):
            y_ = y[:, i]
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)
    else:
        C = int(y.max() + 1)  # pyright: ignore
        for i in range(C):
            y_ = (y == i).astype(np.float64)
            evidence, weight = (  # pyright: ignore[reportUnknownVariableType]
                each_evidence(y_, f, fh, v, s, vh, N, D)
            )
            # print(evidence)
            evidences.append(evidence)  # pyright: ignore[reportUnknownArgumentType]
            weights.append(weight)  # pyright: ignore[reportUnknownArgumentType]

    score = np.mean(evidences)
    weights = np.vstack(  # pyright: ignore[reportUnknownVariableType]
        weights  # pyright: ignore[reportUnknownArgumentType]
    )

    if return_weights:
        return score, weights  # pyright: ignore[reportUnknownVariableType]
    else:
        return score


@njit
def each_evidence(
    y_: NDArray[Any],
    f: NDArray[Any],
    fh: NDArray[Any],
    v: NDArray[Any],
    s: NDArray[Any],
    vh: NDArray[Any],
    N: int,
    D: int,
):
    """
    compute the maximum evidence for each class
    """
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = vh @ (f @ y_)
    EPS = 1e-7

    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + EPS)
        # print(f"alpha: {alpha}")
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + EPS)
        # print(f"beta: {beta}")
        new_lam = alpha / (beta + EPS)
        # print(f"new_lam: {new_lam}")
        if np.abs(new_lam - lam) / (lam + EPS) < 0.01:
            break
        lam = new_lam

    evidence = (
        D / 2.0 * np.log(alpha)
        + N / 2.0 * np.log(beta)
        - 0.5 * np.sum(np.log(alpha + beta * s))
        - beta / 2.0 * beta_de  # pyright: ignore[reportPossiblyUnboundVariable]
        - alpha / 2.0 * alpha_de  # pyright: ignore[reportPossiblyUnboundVariable]
        - N / 2.0 * np.log(2 * np.pi)
    )

    return evidence / N, m  # pyright: ignore[reportPossiblyUnboundVariable]
