from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple, Union
from tqdm import tqdm
import os
from numpy.typing import NDArray
from typing import Any, Optional

from model_ranking.utils import is_ndarray

from model_ranking.classification.transfer_metrics import (
    get_transfer_data_classification,
)
from model_ranking.classification.utils import get_source_from_classification_model_name

from model_ranking.transferability_metrics import (
    bhattacharyya_coefficient,
    h_score,
    regularized_h_score,
    log_expected_empirical_prediction,
    gaussian_log_expected_empirical_prediction,
    log_maximum_evidence,
    NCTI_Score,
    process_NCTI_scores,
    run_transfer_metric_calc,
    dispersion,
    get_nuno,
    ensure_even_label_sampling,
)
from model_ranking.feature_ranking import get_precomputed_feature_path
from model_ranking.utils import load_h5, get_source_from_model_name
from model_ranking.plots import plot_performance_vs_transfer_metric
from model_ranking.results import (
    get_NA_prediction_path,
    save_transfer_metric_results,
    get_finetuned_result_path,
)
from model_ranking.dataclass import (
    TransferabilityMetricConfig,
    transferability_metric_names,
    PrecomputedFeatureConfig,
    segmentation_performance_type,
)
from model_ranking.correlation import (
    to_target_transfer_correlations,
)


def calculate_transfer_metric(  # pyright: ignore
    metric_name: transferability_metric_names,
    features: Optional[NDArray[Any]],
    labels: NDArray[Any],
    predictions: Optional[NDArray[Any]],
    n_PCA_components: Optional[int] = None,
):
    if metric_name == "GBC":
        assert features is not None, "Features must be provided for GBC metric."
        # assert (
        #     n_PCA_components is not None
        # ), "n_PCA_components must be provided for GBC metric."
        return bhattacharyya_coefficient(
            features, labels, n_feature_components=n_PCA_components
        )
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
    elif metric_name == "NCTI":
        assert features is not None, "Features must be provided for NCTI metric."
        assert (
            n_PCA_components is not None
        ), "n_PCA_components must be provided for NCTI metric."
        return NCTI_Score(features, labels, PCA_components=n_PCA_components)
    elif metric_name == "Transfer_Score":
        assert (
            features is not None
        ), "Features must be provided for Transfer_Score metric."
        assert (
            predictions is not None
        ), "Predictions must be provided for Transfer_Score metric."
        return run_transfer_metric_calc(
            features,
            predictions,
            weights=None,
            labels=labels,
            n_samples_per_class=150000,
        )
    elif metric_name == "Dispersion":
        assert (
            features is not None
        ), "Features must be provided for Transfer_Score metric."
        assert (
            predictions is not None
        ), "Predictions must be provided for Transfer_Score metric."
        features_balanced, _, predictions_balanced = ensure_even_label_sampling(
            features=features,
            labels=labels,
            predictions=predictions,
            n_samples_per_class=150000,
        )
        assert is_ndarray(features_balanced), "Features should be numpy array"
        assert is_ndarray(predictions_balanced), "Predictions should be numpy array"
        return dispersion(
            features_balanced, (predictions_balanced > 0.5).astype(np.int8)
        )

    elif metric_name == "NuNo":
        assert (
            predictions is not None
        ), "Predictions must be provided for Transfer_Score metric."
        _, _, predictions_balanced = ensure_even_label_sampling(
            features=None,
            labels=labels,
            predictions=predictions,
            n_samples_per_class=150000,
        )
        assert is_ndarray(predictions_balanced), "Predictions should be numpy array"
        if predictions_balanced.ndim == 1:
            predictions_balanced = np.stack(
                [1 - predictions_balanced, predictions_balanced], axis=1
            )
        return get_nuno(predictions_balanced)
    else:
        raise ValueError(f"Unknown transfer metric: {metric_name}")


def get_transfer_data_segmentation(
    model_name: str,
    target: str,
    epoch: str,
    feature_config: PrecomputedFeatureConfig,
    transferability_metric: transferability_metric_names,
    performance_config: segmentation_performance_type,
):
    feature_ids = list(feature_config.layer_keys.keys())
    model_identifier = model_name.split("_")[-1][:-1]
    if model_identifier in feature_ids:
        key = feature_config.layer_keys[model_identifier]
    elif any(fid in model_name for fid in feature_ids):
        matched_id = next(fid for fid in feature_ids if fid in model_name)
        key = feature_config.layer_keys[matched_id]
    else:
        key = "decoders.2"

    feature_path = get_precomputed_feature_path(
        model_name,
        target,
        feature_config.base_path,
        filetype=feature_config.file_type,
    )
    if str(transferability_metric) == "LEEP":
        features = None
        predictions = load_h5(feature_path, f"{key}_predictions")
    elif str(transferability_metric) == "Transfer_Score":
        features = load_h5(feature_path, f"{key}_features")
        predictions = load_h5(feature_path, f"{key}_predictions")
    else:
        features = load_h5(feature_path, f"{key}_features")
        predictions = None

    labels = load_h5(feature_path, f"{key}_labels")

    if performance_config.name == "direct_performance":
        performance_path = get_NA_prediction_path(
            model_name,
            target,
            performance_config.base_path,
            approach=performance_config.approach,
            run_id=performance_config.run_id,
            metric_summary=performance_config.metric_summary,
        )
    else:
        performance_path = get_finetuned_result_path(
            model_name,
            finetuning_approach=performance_config.finetuning_approach,
            epoch=epoch,
            base_path=performance_config.base_path,
            result_type=performance_config.result_type,
        )

    if performance_config.metric_summary == True:
        performance_score = load_h5(
            performance_path, f"{performance_config.key}_median"
        )[1]
    else:
        performance_score = load_h5(performance_path, performance_config.key)
        performance_score = np.median(performance_score[:, 1])

    if performance_config.invert_score == True:
        performance_score = 1 - performance_score

    non_zero_patch_ids = np.where(~np.all(labels == 0, axis=1))[0]

    assert (
        features is not None or predictions is not None
    ), "Either features or predictions must be provided for transferability metric calculation."

    if features is not None:
        features = features[non_zero_patch_ids]
        features_flat = features.reshape(-1, features.shape[-1])
    else:
        features_flat = None

    if predictions is not None:
        predictions = predictions[non_zero_patch_ids]
        predictions_flat = predictions.reshape(-1)  # Shape: (n * 1000,)
        predictions_flat = np.column_stack([1 - predictions_flat, predictions_flat])
    else:
        predictions_flat = None

    labels = labels[non_zero_patch_ids]

    labels_flat = labels.reshape(-1).astype(int)

    return (
        features_flat,
        predictions_flat,
        labels_flat,
        performance_score,
    )


def transfer_sweep_transferability_metric(config: TransferabilityMetricConfig):
    feature_cfg = config.feature_config
    performance_cfg = config.performance_config
    output_cfg = config.output_config
    transfer_metric_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    performance_per_target: Dict[str, Dict[str, float]] = {}
    for transferability_metric in config.transferability_metrics:
        print(f"Calculating transferability metric: {transferability_metric}")
        transfer_metric_per_target: Dict[str, Dict[str, float]] = {}
        component_scores_per_target: Dict[str, Dict[str, Dict[str, float]]] = {}
        for target in config.targets:
            print(f"Processing target: {target}")
            performance_per_model: Dict[str, Any] = {}
            transfer_metric_per_model: Union[
                Dict[str, float], Dict[str, Tuple[float, float, float]]
            ] = {}
            for model_name, epoch in tqdm(config.source_models.items()):
                if performance_cfg.name == "classification_performance":
                    source = get_source_from_classification_model_name(model_name)
                else:
                    source = get_source_from_model_name(model_name)

                if (source == "VNC") and (target == "VNC"):
                    continue
                if (source == "S_BIAD895") and (target == "S_BIAD895"):
                    continue
                else:
                    if performance_cfg.name == "classification_performance":
                        (
                            features,
                            predictions,
                            labels,
                            performance_score,
                        ) = get_transfer_data_classification(
                            model_name,
                            target,
                            feature_cfg,
                            transferability_metric,
                            performance_cfg,
                        )
                    else:
                        (
                            features,
                            predictions,
                            labels,
                            performance_score,
                        ) = get_transfer_data_segmentation(
                            model_name,
                            target,
                            epoch,
                            feature_cfg,
                            transferability_metric,
                            performance_cfg,
                        )

                    performance_per_model[model_name] = performance_score

                    transfer_metric = calculate_transfer_metric(  # pyright: ignore[reportUnknownVariableType]
                        metric_name=transferability_metric,
                        features=features,
                        predictions=predictions,
                        labels=labels,
                        n_PCA_components=feature_cfg.n_PCA_components,
                    )
                    transfer_metric_per_model[model_name] = (  # pyright: ignore
                        transfer_metric
                    )

            if transferability_metric == "Transfer_Score":
                transfer_score_per_model: Dict[str, float] = {}
                for model_name, transfer_metric in transfer_metric_per_model.items():
                    assert isinstance(
                        transfer_metric, tuple
                    ), "Transfer_Score should return a tuple of scores."
                    component_scores_per_target[target] = {
                        "Hopkins": transfer_metric[1],
                        "MI": transfer_metric[2],
                        "UNF": transfer_metric[3],
                    }
                    transfer_score_per_model[model_name] = transfer_metric[0]
                transfer_metric_per_model = transfer_score_per_model

            elif transferability_metric == "NCTI":
                # Ensure transfer_metric_per_model is not a Dict[str, float] before unpacking
                transfer_metric_per_model, seli_scores, ncc_scores, vc_scores = (
                    process_NCTI_scores(
                        transfer_metric_per_model  # pyright: ignore[reportArgumentType]
                    )
                )

                component_scores_per_target[target] = {
                    "SELI": seli_scores,
                    "NCC": ncc_scores,
                    "VC": vc_scores,
                }

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
                    Path(output_cfg.save_base_path)
                    / "figs"
                    / f"{target}_{output_cfg.save_name}_{transferability_metric}.png"
                )

                os.makedirs(save_path.parent, exist_ok=True)

                _ = plot_performance_vs_transfer_metric(
                    performance_per_model,
                    transfer_metric_per_model,
                    target=target,
                    metric_name=transferability_metric,
                    performance_metric_name=performance_cfg.key,
                    show_plot=False,
                    save_path=str(save_path),
                )

        correlation_scores: Dict[str, NDArray[Any]] = {}

        correlation_scores["KT"], correlation_scores["SP"], correlation_scores["PE"] = (
            to_target_transfer_correlations(
                config.targets,
                transfer_metric_per_target,
                performance_per_target,
            )
        )

        if output_cfg.save_base_path is not None:
            assert output_cfg.save_name is not None, "Save name must be provided."
            os.makedirs(output_cfg.save_base_path, exist_ok=True)

            if len(component_scores_per_target) > 0:
                cmp_scores_per_target = component_scores_per_target
            else:
                cmp_scores_per_target = None

            save_transfer_metric_results(
                transfer_metric_per_target,
                performance_per_target,
                correlation_scores=correlation_scores,
                save_dir=output_cfg.save_base_path,
                experiment_name=f"{output_cfg.save_name}_{transferability_metric}",
                component_transfer_scores_per_target=cmp_scores_per_target,
            )
    return transfer_metric_results, performance_per_target
