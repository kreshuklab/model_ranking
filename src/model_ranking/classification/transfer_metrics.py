import numpy as np

from .utils import (
    get_classification_pred_path,
)

from model_ranking.dataclass import (
    PrecomputedFeatureConfig,
    transferability_metric_names,
    PrecomputedClassificationPerformanceConfig,
)
from model_ranking.utils import load_h5


def get_transfer_data_classification(
    model_name: str,
    target: str,
    feature_config: PrecomputedFeatureConfig,
    transferability_metric: transferability_metric_names,
    performance_config: PrecomputedClassificationPerformanceConfig,
):
    feature_ids = list(feature_config.layer_keys.keys())
    model_identifier = model_name.split("_")[-1][:-1]
    if model_identifier in feature_ids:
        key = feature_config.layer_keys[model_identifier]
    else:
        key = "ResNet.avgpool"

    pred_path = get_classification_pred_path(
        model_name,
        target,
        feature_config.base_path,
    )
    if str(transferability_metric) not in ["LEEP"]:
        features = load_h5(pred_path, key)
        predictions = None
    else:
        features = None
        predictions = load_h5(pred_path, f"predictions")

    labels = load_h5(pred_path, "labels")

    performance = load_h5(pred_path, performance_config.key)

    if performance_config.invert_score == True:
        performance = 1 - performance

    assert (
        features is not None or predictions is not None
    ), "Either features or predictions must be provided for transferability metric calculation."

    if predictions is not None:
        predictions = np.column_stack([1 - predictions, predictions])

    labels = labels.astype(int)

    return features, predictions, labels, performance
