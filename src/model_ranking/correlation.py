import numpy as np
from numpy.typing import NDArray
from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
    kendalltau,  # pyright: ignore[reportUnknownVariableType]
    spearmanr,  # pyright: ignore[reportUnknownVariableType]
    permutation_test,  # pyright: ignore[reportUnknownVariableType]
)
from typing import Any, Tuple

from model_ranking.utils import is_ndarray


def scores_to_rank(scores: NDArray[Any], ascending: bool = True, tolerance: float = 0):
    # intial rank array produce ranking based on the array if ascending lowest value will have rank 1
    # if descending highest value will have rank 1
    scores = np.asarray(scores)
    rank = np.arange(1, len(scores) + 1)
    sorted_indices = np.argsort(scores if ascending else -scores)
    rank[sorted_indices] = np.arange(len(scores))
    rank = rank + 1

    count = 0
    scores_ranking = scores.copy()
    while len(np.unique(rank)) > 1:
        # find pairwise distance between all elements of the array only needs to be a lower triangular matrix as symmetrical
        pairwise_distance = np.tril(np.abs(scores_ranking[:, None] - scores_ranking))
        pairwise_distance = np.abs(scores_ranking[:, None] - scores_ranking)
        pairwise_distance[np.triu_indices_from(pairwise_distance, k=1)] = np.inf
        np.fill_diagonal(pairwise_distance, np.inf)
        if count > 0:
            # set ids where pairwise distance == zero  to inf to avoid merging the same ranks
            pairwise_distance[pairwise_distance == 0] = np.inf
            # for index in merged_indicies:
            #    pairwise_distance[index] = np.inf

        # if min_distance < tolerance, merge the ranks
        if np.min(pairwise_distance) <= tolerance + 1e-6:
            # find the indices of the minimum distance
            min_distance_indices = np.argwhere(
                pairwise_distance == np.min(pairwise_distance)
            )
            # get rank value of the two indices
            rank1 = rank[min_distance_indices[0][0]]
            rank2 = rank[min_distance_indices[0][1]]
            # find ids in rank equal to either of the two ranks
            ids = np.argwhere((rank == rank1) | (rank == rank2)).flatten()
            # set ranks at ids to be equal to the lowest rank of the two
            rank[ids] = np.min([rank1, rank2])
            # reassign ranking to make contiguous
            rank = np.unique(rank, return_inverse=True)[1] + 1
            # average scores of the merged ranks
            mean_merged_scores = np.mean(scores[ids])
            # set scores ranking at location in ids to mean_merged_scores
            scores_ranking[ids] = mean_merged_scores
            count += 1
        else:
            break

    return rank


def permutation_test_spearman_rho(
    x: NDArray[Any],
    y: NDArray[Any],
    n_permutations: int = 1000,
) -> Tuple[float, float]:
    """
    Perform a permutation test for Spearman's rank correlation coefficient using scipy's permutation_test.

    Parameters:
    - x, y: arrays of the same length, representing paired data.
    - n_permutations: Number of permutations for the test.
    - random_state: Seed for reproducibility.

    Returns:
    - observed_rho: Spearman's rho for the original data.
    - p_value: The p-value from the permutation test.
    """

    # Define the statistic function for Spearman's rho
    def statistic(x: NDArray[Any], y: NDArray[Any]):
        rho, _ = spearmanr(x, y)
        return rho

    # Perform the permutation test
    result = permutation_test(
        (x, y),
        statistic,
        permutation_type="pairings",  # Permute pairings of x and y
        n_resamples=n_permutations,
        alternative="two-sided",
    )

    # Return the observed statistic and p-value
    sp = result.statistic  # pyright: ignore[reportUnknownVariableType]
    pval = result.pvalue  # pyright: ignore[reportUnknownVariableType]
    assert not is_ndarray(sp), "result.statistic is not a numpy array"
    assert not is_ndarray(pval), "result.pvalue is not a numpy array"

    return sp, pval  # pyright: ignore[reportUnknownVariableType, reportReturnType]


def permutation_test_kendall_tau(
    x: NDArray[Any], y: NDArray[Any], n_permutations: int = 1000
) -> Tuple[float, float]:
    """
    Perform a permutation test for Kendall's tau with ties using scipy's permutation_test.

    Parameters:
    - x, y: arrays of the same length, representing paired data.
    - n_permutations: Number of permutations for the test.
    - random_state: Seed for reproducibility.

    Returns:
    - observed_tau: Kendall's tau for the original data.
    - p_value: The p-value from the permutation test.
    """

    # Define the statistic function for Kendall's tau
    def statistic(x: NDArray[Any], y: NDArray[Any]):
        tau, _ = kendalltau(x, y)
        return tau

    # Perform the permutation test
    result = permutation_test(
        (x, y),
        statistic,
        permutation_type="pairings",  # Permute pairings of x and y
        n_resamples=n_permutations,
        alternative="two-sided",
    )

    # Return the observed statistic and p-value
    return (
        result.statistic,
        result.pvalue,
    )  # pyright: ignore[reportUnknownVariableType, reportReturnType]
