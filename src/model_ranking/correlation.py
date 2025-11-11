import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
    kendalltau,  # pyright: ignore[reportUnknownVariableType]
    spearmanr,  # pyright: ignore[reportUnknownVariableType]
    permutation_test,  # pyright: ignore[reportUnknownVariableType]
    pearsonr,  # pyright: ignore[reportUnknownVariableType]
)
from typing import Any, Dict, Sequence, Tuple

from model_ranking.utils import is_ndarray
from model_ranking.results import transfer_results_to_arrays, results_to_arrays


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


def calculate_correlation_statistics(
    transfer_scores: NDArray[Any],
    perf_scores: NDArray[Any],
    rank_ascending: bool = False,
    perf_tolerance: float = 0,
    consis_tolerance: float = 0,
):
    """
    Calculate correlation statistics for the given consistency and performance results.
    """

    ranked_perf = scores_to_rank(
        perf_scores, ascending=rank_ascending, tolerance=perf_tolerance
    )

    pearson_scores = np.zeros((transfer_scores.shape[1], 2))
    kendall_tau_scores = np.zeros((transfer_scores.shape[1], 2))
    spearman_scores = np.zeros((transfer_scores.shape[1], 2))

    for i in range(transfer_scores.shape[1]):
        ranked_consis = scores_to_rank(
            transfer_scores[:, i], ascending=rank_ascending, tolerance=consis_tolerance
        )
        pearson_scores[i, 0], pearson_scores[i, 1] = pearsonr(
            perf_scores, transfer_scores[:, i]
        )
        kendall_tau_scores[i, 0], kendall_tau_scores[i, 1] = (
            permutation_test_kendall_tau(ranked_perf, ranked_consis)
        )
        spearman_scores[i, 0], spearman_scores[i, 1] = permutation_test_spearman_rho(
            ranked_perf, ranked_consis
        )

    return kendall_tau_scores, spearman_scores, pearson_scores


def to_target_transfer_correlations(
    targets: Sequence[str],
    transfer_metric_per_target: Dict[str, Dict[str, float]],
    performance_per_target: Dict[str, Dict[str, float]],
    invert_transfer_metric: bool = False,
    invert_perf_metric: bool = False,
):
    per_target_KT = np.zeros((len(targets), 1, 2))
    per_target_SP = np.zeros((len(targets), 1, 2))
    per_target_PE = np.zeros((len(targets), 1, 2))
    for i, target in enumerate(targets):
        transfer_scores, NA_perf_scores = transfer_results_to_arrays(
            transfer_metric_per_target[target],
            performance_per_target[target],
        )
        if invert_transfer_metric:
            transfer_scores = 1 - transfer_scores
        if invert_perf_metric:
            NA_perf_scores = 1 - NA_perf_scores

        (per_target_KT[i], per_target_SP[i], per_target_PE[i]) = (
            calculate_correlation_statistics(
                transfer_scores,
                NA_perf_scores,
            )
        )
    return per_target_KT, per_target_SP, per_target_PE


def to_target_transfer_correlations_with_norm(
    targets: Sequence[str],
    transfer_metric_per_target: Dict[
        str, Dict[str, Dict[str, Dict[str, NDArray[Any]]]]
    ],
    performance_per_target: Dict[str, Dict[str, Dict[str, float]]],
    num_aug_alphas: int,
    perturbation_key: str = "gauss",
    invert_consis_score: bool = False,
    invert_perf_score: bool = False,
):
    per_target_KT = np.zeros((len(targets), num_aug_alphas, 2))
    per_target_SP = np.zeros((len(targets), num_aug_alphas, 2))
    per_target_PE = np.zeros((len(targets), num_aug_alphas, 2))
    for i, target in enumerate(targets):
        transfer_scores, NA_perf_scores = results_to_arrays(
            transfer_metric_per_target[target],
            performance_per_target[target],
            perturbation_key=perturbation_key,
            num_alphas=num_aug_alphas,
        )
        if invert_consis_score:
            transfer_scores = 1 - transfer_scores
        if invert_perf_score:
            NA_perf_scores = 1 - NA_perf_scores

        (per_target_KT[i], per_target_SP[i], per_target_PE[i]) = (
            calculate_correlation_statistics(
                transfer_scores,
                NA_perf_scores,
            )
        )
    return per_target_KT, per_target_SP, per_target_PE


def correlation_table(
    KT_per_target: NDArray[Any],
    SP_per_target: NDArray[Any],
    PE_per_target: NDArray[Any],
    targets: Sequence[str],
    task: str = "Mito",
    idx: int = 0,
    num_sig_fig: int = 2,
):
    df = pd.DataFrame(
        {
            "kt": KT_per_target[:, idx, 0],
            "kt pval": KT_per_target[:, idx, 1],
            "s rho": SP_per_target[:, idx, 0],
            "s rho pval": SP_per_target[:, idx, 1],
            "pr": PE_per_target[:, idx, 0],
            "pr pval": PE_per_target[:, idx, 1],
        },
    )
    df["targets"] = targets
    df = df.set_index("targets")
    df.index = pd.MultiIndex.from_product([[task], df.index], names=["Task", "targets"])
    # Round all float columns to appropriate significant figures
    df = df.map(
        lambda x: (  # pyright: ignore
            round(x, num_sig_fig) if isinstance(x, float) else x
        )
    )
    return df


def avg_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean and standard deviation of correlation scores across target datasets.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with targets as rows and correlation scores (kt, s rho, pr) as columns.
        Expected to have a MultiIndex with 'Task' and 'targets' levels.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with one row containing KT_avg, KT_std, SR_avg, SR_std, PR_avg, PR_std
        The Task index value is preserved from the input dataframe.
    """
    import pandas as pd

    # Get the Task value from the first level of the MultiIndex
    task_value = df.index.get_level_values("Task")[0]

    # Calculate mean and standard deviation for kt, s rho, and pr columns
    kt_avg = df["kt"].mean()
    kt_std = df["kt"].std()
    sr_avg = df["s rho"].mean()
    sr_std = df["s rho"].std()
    pr_avg = df["pr"].mean()
    pr_std = df["pr"].std()

    # Create result dataframe
    result_df = pd.DataFrame(
        {
            "KT_avg": [kt_avg],
            "KT_std": [kt_std],
            "SR_avg": [sr_avg],
            "SR_std": [sr_std],
            "PR_avg": [pr_avg],
            "PR_std": [pr_std],
        },
        index=pd.Index([task_value], name="Task"),
    )

    return result_df
