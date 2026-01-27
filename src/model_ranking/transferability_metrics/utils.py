import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional


def ensure_even_label_sampling(
    features: Optional[NDArray[Any]],
    labels: NDArray[Any],
    predictions: Optional[NDArray[Any]],
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
    if features is not None:
        features_balanced = features.reshape(-1, features.shape[-1])[balanced_indices]
    else:
        features_balanced = None
    labels_balanced = labels.flatten()[balanced_indices]
    if predictions is not None:
        predictions_balanced = predictions.flatten()[balanced_indices]
    else:
        predictions_balanced = None

    print(f"Original dataset: {len(labels.flatten())} samples")
    print(f"Balanced dataset: {len(labels_balanced)} samples")
    return features_balanced, labels_balanced, predictions_balanced
