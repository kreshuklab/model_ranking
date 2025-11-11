import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from typing import Any, Tuple
from numpy.typing import NDArray
from numpy.random import uniform
from random import sample
from typing import Optional, Any
import math


def ensure_even_feature_sampling(
    features: NDArray[Any],
    labels: NDArray[Any],
    predictions: NDArray[Any],
    n_samples_per_class: Optional[int] = None,
):
    # Sample equal number of class 0 and 1 labels
    class_0_indices = np.where(labels.flatten() == 0)[0]
    class_1_indices = np.where(labels.flatten() == 1)[0]

    # Determine the minimum count between the two classes
    min_count = min(len(class_0_indices), len(class_1_indices))

    if n_samples_per_class is None:
        n_samples = min_count

    else:
        n_samples = min(n_samples_per_class, min_count)

    # Randomly sample equal numbers from each class
    np.random.seed(42)  # for reproducibility
    sampled_class_0 = np.random.choice(class_0_indices, size=n_samples, replace=False)
    sampled_class_1 = np.random.choice(class_1_indices, size=n_samples, replace=False)

    # Combine the indices
    balanced_indices = np.concatenate([sampled_class_0, sampled_class_1])

    # Create balanced datasets
    features_balanced = features.reshape(-1, features.shape[-1])[balanced_indices]
    labels_balanced = labels.flatten()[balanced_indices]
    predictions_balanced = predictions.flatten()[balanced_indices]

    print(f"Original dataset: {len(labels.flatten())} samples")
    print(f"Balanced dataset: {len(labels_balanced)} samples")
    return features_balanced, labels_balanced, predictions_balanced


def calculate_transfer_metric(
    features: NDArray[Any],
    predictions: NDArray[Any],
    weights: Optional[torch.Tensor],
    num_classes: int = 2,
):
    H_stat = hopkins_statistic(features)

    if weights is None:
        unf = 0.0
    else:
        if (weights.shape[0] == 1) and (num_classes == 2):
            weights = torch.cat([-weights, weights], dim=0)

        unf = uniformity(weights)

    if (predictions.ndim == 1) and (num_classes == 2):
        predictions = np.stack([1 - predictions, predictions], axis=1)

    mi_score = mutual_information_from_probs_np(predictions)

    transfer_metric = H_stat - mi_score / math.log(2) - unf

    return transfer_metric, H_stat, mi_score, unf


def run_transfer_metric_calc(
    features: NDArray[Any],
    predictions: NDArray[Any],
    weights: Optional[torch.Tensor],
    labels: NDArray[Any],
    num_classes: int = 2,
    n_samples_per_class: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    features_balanced, _, predictions_balanced = ensure_even_feature_sampling(
        features=features,
        labels=labels,
        predictions=predictions,
        n_samples_per_class=n_samples_per_class,
    )
    transfer_metric = calculate_transfer_metric(
        features_balanced, predictions_balanced, weights, num_classes=num_classes
    )
    return transfer_metric


def entropy(input_: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def entropy_np(input_: NDArray[Any]) -> NDArray[Any]:
    epsilon = 1e-5
    entropy = -input_ * np.log(input_ + epsilon)
    entropy = np.sum(entropy, axis=1)
    return entropy


def mutual_information_from_probs_torch(pred: torch.Tensor) -> float:
    """

    Args:
        pred (torch.Tensor): (m, K) predicted probabilities for sampled pixels

    Returns:
        float: _description_
    """
    entropy_loss = torch.mean(entropy(pred))
    mean_P = pred.mean(dim=0)
    gentropy_loss = torch.sum(-mean_P * torch.log(mean_P + 1e-6))
    entropy_loss -= gentropy_loss
    return entropy_loss.item()


def mutual_information_from_probs_np(pred: NDArray[Any]) -> float:
    entropy_loss = np.mean(entropy_np(pred))
    mean_P = pred.mean(axis=0)
    gentropy_loss = np.sum(-mean_P * np.log(mean_P + 1e-6))
    entropy_loss -= gentropy_loss
    return entropy_loss.item()


def hopkins_statistic(features: NDArray[Any]) -> float:
    """calculates the hopkins statistic for a set of n d-dimensional features

    Args:
        features (NDArray[Any]): (n_samples x d)

    Returns:
        float: hopkins statistic
    """
    sample_size = int(
        features.shape[0] * 0.05
    )  # 0.05 (5%) based on paper by Lawson and Jures
    # a uniform random sample in the original data space
    X_uniform_random_sample = uniform(
        features.min(axis=0), features.max(axis=0), (sample_size, features.shape[1])
    )
    random_indices = sample(range(0, features.shape[0], 1), sample_size)
    X_sample = features[random_indices]

    # initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)

    u_distances, _ = nbrs.kneighbors(  # pyright: ignore
        X_uniform_random_sample, n_neighbors=2
    )
    u_distances = u_distances[:, 0]  # pyright: ignore

    w_distances, _ = nbrs.kneighbors(X_sample, n_neighbors=2)  # pyright: ignore
    w_distances = w_distances[:, 1]  # pyright: ignore

    u_sum = np.sum(u_distances)  # pyright: ignore
    w_sum = np.sum(w_distances)  # pyright: ignore
    hopkins_statistic = u_sum / (u_sum + w_sum)  # pyright: ignore
    return hopkins_statistic  # pyright: ignore


def uniformity(weight: torch.Tensor) -> float:
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())

    n = cosine.size(0)

    cosine = cosine.flatten()[:-1].view(n - 1, n + 1)
    cosine = cosine[:, 1:]
    theta = torch.acos(cosine)

    theta0 = torch.acos(torch.tensor(-1 / (n - 1)))

    dif = theta - theta0

    u1 = dif.abs()
    u2 = dif.mul(dif)

    _ = torch.mean(u1)
    u2_loss = torch.mean(u2)
    return u2_loss.item()
