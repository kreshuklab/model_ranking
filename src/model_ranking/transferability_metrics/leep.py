import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Any

from model_ranking.utils import is_ndarray


def log_expected_empirical_prediction(predictions: NDArray[Any], labels: NDArray[Any]):
    r"""
    Log Expected Empirical Prediction in `LEEP: A New Measure to
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>`_.

    The LEEP :math:`\mathcal{T}` can be described as:

    .. math::
        \mathcal{T}=\mathbb{E}\log \left(\sum_{z \in \mathcal{C}_s} \hat{P}\left(y \mid z\right) \theta\left(y \right)_{z}\right)

    where :math:`\theta\left(y\right)_{z}` is the predictions of pre-trained model on source category, :math:`\hat{P}\left(y \mid z\right)` is the empirical conditional distribution estimated by prediction and ground-truth label.

    Args:
        predictions (np.ndarray): predictions of pre-trained model.
        labels (np.ndarray): groud-truth labels.

    Shape:
        - predictions: (N, :math:`C_s`), with number of samples N and source class number :math:`C_s`.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar
    """
    N, C_s = predictions.shape
    labels = labels.reshape(-1)
    C_t = int(np.max(labels) + 1)

    normalized_prob = predictions / float(N)
    joint = np.zeros(
        (C_t, C_s), dtype=float
    )  # placeholder for joint distribution over (y, z)

    for i in range(C_t):
        this_class = normalized_prob[labels == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = predictions @ p_target_given_source
    empirical_prob = np.array(
        [predict[label] for predict, label in zip(empirical_prediction, labels)]
    )
    score = np.mean(np.log(empirical_prob))

    return score


def gaussian_log_expected_empirical_prediction(
    features: NDArray[Any], labels: NDArray[Any]
):
    r"""
    Log Expected Empirical Prediction in `LEEP: A New Measure to
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>`_.

    The LEEP :math:`\mathcal{T}` can be described as:

    .. math::
        \mathcal{T}=\mathbb{E}\log \left(\sum_{z \in \mathcal{C}_s} \hat{P}\left(y \mid z\right) \theta\left(y \right)_{z}\right)

    where :math:`\theta\left(y\right)_{z}` is the predictions of pre-trained model on source category, :math:`\hat{P}\left(y \mid z\right)` is the empirical conditional distribution estimated by prediction and ground-truth label.

    Args:
        predictions (np.ndarray): predictions of pre-trained model.
        labels (np.ndarray): groud-truth labels.

    Shape:
        - predictions: (N, :math:`C_s`), with number of samples N and source class number :math:`C_s`.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar
    """
    labels = labels.reshape(-1)
    num_classes = int(np.max(labels) + 1)

    # first calculate pca retaining 80% of the variance
    # For large datasets, we might want to limit PCA dimensions more aggressively
    features_pca = PCA(  # pyright: ignore
        n_components=0.8, random_state=42
    ).fit_transform(features)

    assert is_ndarray(features_pca), f"Expected features_pca to be a numpy array"

    # Determine appropriate number of GMM components
    _, n_features_pca = features_pca.shape

    desired_components = 5 * num_classes

    # Limit based on PCA dimensions - use fewer components for high-dim spaces
    if n_features_pca > 50:
        max_components_by_features = min(
            desired_components, max(2, n_features_pca // 5)
        )
    else:
        max_components_by_features = desired_components

    n_components = min(desired_components, max_components_by_features)

    # print(f"Data shape: {features.shape} -> PCA shape: {features_pca.shape}")
    # print(f"Using {n_components} GMM components (desired: {desired_components})")
    # print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # then calculate gmm as density estimator and calculate leep
    # For large datasets, we can use more iterations and better initialization

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42,
        max_iter=300,  # More iterations for large datasets
        tol=1e-4,  # Tighter tolerance
        reg_covar=1e-6,
        init_params="k-means++",
        n_init=5,  # More initializations for better convergence
        verbose=1,  # Show convergence progress
    ).fit(features_pca)

    if not gmm.converged_:
        print("Warning: GMM did not converge, but continuing anyway")

    gmm_predictions = gmm.predict_proba(  # pyright: ignore[reportUnknownVariableType]
        features_pca
    )

    assert is_ndarray(gmm_predictions), f"Expected gmm_predictions to be a numpy array"

    return log_expected_empirical_prediction(gmm_predictions, labels)
