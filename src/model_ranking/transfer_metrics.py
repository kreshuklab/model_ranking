import numpy as np
from typing import Dict, Any
from tqdm import tqdm
import os
from numpy.typing import NDArray
from typing import Any, Optional

from model_ranking.GBC import bhattacharyya_coefficient
from model_ranking.hscore import h_score, regularized_h_score
from model_ranking.leep import (
    log_expected_empirical_prediction,
    gaussian_log_expected_empirical_prediction,
)
from model_ranking.logme import log_maximum_evidence
from model_ranking.feature_ranking import get_precomputed_feature_path
from model_ranking.plots import plot_performance_vs_transfer_metric
from model_ranking.results import (
    get_NA_prediction_path,
    save_transfer_metric_results,
)
from model_ranking.utils import load_h5
from model_ranking.dataclass import (
    TransferabilityMetricConfig,
    transferability_metrics,
)


def calculate_transfer_metric(  # pyright: ignore
    metric_name: transferability_metrics,
    features: Optional[NDArray[Any]],
    labels: NDArray[Any],
    predictions: Optional[NDArray[Any]],
):
    if metric_name == "GBC":
        assert features is not None, "Features must be provided for GBC metric."
        return bhattacharyya_coefficient(features, labels)
    elif metric_name == "LEEP":
        assert predictions is not None, "Predictions must be provided for LEEP metric."
        return log_expected_empirical_prediction(predictions, labels)
    elif metric_name == "Gaussian_LEEP":
        assert (
            features is not None
        ), "Features must be provided for Gaussian_LEEP metric."
        return gaussian_log_expected_empirical_prediction(features, labels)
    elif metric_name == "Hscore":
        assert features is not None, "Features must be provided for Hscore metric."
        return h_score(features, labels)
    elif metric_name == "Regularized_Hscore":
        assert (
            features is not None
        ), "Features must be provided for Regularized_Hscore metric."
        return regularized_h_score(features, labels)
    elif metric_name == "LogME":
        assert features is not None, "Features must be provided for LogME metric."
        return log_maximum_evidence(features, labels)  # pyright: ignore
    else:
        raise ValueError(f"Unknown transfer metric: {metric_name}")


def transfer_sweep_transferability_metric(config: TransferabilityMetricConfig):
    feature_cfg = config.feature_config
    performance_cfg = config.performance_config
    output_cfg = config.output_config
    transfer_metric_per_target: Dict[str, Dict[str, float]] = {}
    performance_per_target: Dict[str, Dict[str, float]] = {}
    feature_ids = list(feature_cfg.layer_keys.keys())
    for target in config.targets:
        print(f"Processing target: {target}")
        performance_per_model: Dict[str, Any] = {}
        transfer_metric_per_model: Dict[str, float] = {}
        for model_name in tqdm(config.source_models):
            if ("V" in model_name) and (target == "VNC"):
                continue
            else:
                model_identifier = model_name.split("_")[-1][:-1]
                if model_identifier in feature_ids:
                    key = feature_cfg.layer_keys[model_identifier]
                else:
                    key = "decoders.2"

                feature_path = get_precomputed_feature_path(
                    model_name,
                    target,
                    feature_cfg.base_path,
                    filetype=feature_cfg.file_type,
                )
                if str(config.transferability_metric) not in ["LEEP"]:
                    features = load_h5(feature_path, f"{key}_features")
                    predictions = None
                else:
                    features = None
                    predictions = load_h5(feature_path, f"{key}_predictions")
                    # Threshold predictions
                    # predictions = (predictions > performance_cfg.threshold).astype(int)

                labels = load_h5(feature_path, f"{key}_labels")

                performance_path = get_NA_prediction_path(
                    model_name,
                    target,
                    performance_cfg.base_path,
                    approach=performance_cfg.approach,
                    run_id=performance_cfg.run_id,
                )

                performance_score = load_h5(performance_path, performance_cfg.key)

                performance_per_model[model_name] = np.median(performance_score[:, 1])

                non_zero_patch_ids = np.where(~np.all(labels == 0, axis=1))[0]

                assert (
                    features is not None or predictions is not None
                ), "Either features or predictions must be provided for transferability metric calculation."

                if features is not None:
                    features = features[non_zero_patch_ids]
                    features_flat = features.reshape(-1, features.shape[-1])
                    predictions_flat = None
                else:
                    assert predictions is not None, "Predictions must be provided."
                    predictions = predictions[non_zero_patch_ids]
                    predictions_flat = predictions.reshape(-1)  # Shape: (n * 1000,)
                    predictions_flat = np.column_stack(
                        [1 - predictions_flat, predictions_flat]
                    )
                    features_flat = None

                labels = labels[non_zero_patch_ids]

                labels_flat = labels.reshape(-1).astype(int)

                transfer_metric = calculate_transfer_metric(  # pyright: ignore[reportUnknownVariableType]
                    metric_name=config.transferability_metric,
                    features=features_flat,
                    predictions=predictions_flat,
                    labels=labels_flat,
                )
                transfer_metric_per_model[model_name] = (  # pyright: ignore
                    transfer_metric
                )

        transfer_metric_per_target[target] = transfer_metric_per_model
        performance_per_target[target] = performance_per_model
        if output_cfg.save_plot:
            assert (
                output_cfg.save_base_path is not None
            ), "Save base path must be provided for plotting."
            assert (
                output_cfg.save_name is not None
            ), "Save name must be provided for plotting."
            save_path = (
                f"{output_cfg.save_base_path}/figs/{target}_{output_cfg.save_name}.png"
            )
            os.makedirs(output_cfg.save_base_path, exist_ok=True)
            plot_performance_vs_transfer_metric(
                performance_per_model,
                transfer_metric_per_model,
                metric_name=config.transferability_metric,
                save_path=save_path,
            )

    if output_cfg.save_base_path is not None:
        assert output_cfg.save_name is not None, "Save name must be provided."
        os.makedirs(output_cfg.save_base_path, exist_ok=True)
        save_transfer_metric_results(
            transfer_metric_per_target,
            performance_per_target,
            save_dir=output_cfg.save_base_path,
            experiment_name=output_cfg.save_name,
        )
    return transfer_metric_per_target, performance_per_target
