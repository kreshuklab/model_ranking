from typing import Dict, List, Mapping, Optional, Union, Any, Sequence, Tuple, Set
from pathlib import Path
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray
import imageio.v2 as imageio
from tqdm import tqdm
import json
import os
from datetime import datetime

from model_ranking.consistency import calculate_per_patch_consistency
from model_ranking.dataclass import (
    ForegroundFilterConfig,
    SummaryResultsConfig,
    # transferability_metrics,
)
from model_ranking.utils import (
    find_transfer_from_pred_path,
    load_h5,
    threshold_patch_foreground_ratio,
    save_h5,
    extract_filename,
    get_roi_slice,
    is_ndarray,
    load_select_prediction_scores,
    create_h5_dataset,
    get_output_dir,
)
from model_ranking.yaml_generators import (
    MODEL_ABBREVIATIONS_TO_DATASET,
)

per_source_consis_result_type = Dict[str, Dict[str, Dict[str, NDArray[Any]]]]
per_source_performance_result_type = Dict[str, Dict[str, float]]
per_target_consis_result_type = Dict[str, per_source_consis_result_type]
per_target_performance_result_type = Dict[str, per_source_performance_result_type]

DATASET_ABBREVIATIONS = {
    "BBBC039": "BC",
    "DSB2018": "DSB",
    "Go-Nuclear": "GN",
    "HeLaNuc": "HN",
    "Hoechst": "Hst",
    "S_BIAD634": "634",
    "S_BIAD895": "895",
    "S_BIAD1196": "1196",
    "S_BIAD1410": "1410",
    "FlyWing": "fw",
    "Ovules": "ov",
    "PNAS": "p",
    "EPFL": "E",
    "Hmito": "Hm",
    "Rmito": "Rm",
    "VNC": "V",
    "affable-shark": "AS",
    "cp_nuclei": "CN",
    "cp_cyto3": "C3",
    "root_nuclei_ds1x": "RN",
    "laid-back-lobster": "LL",
    "pioneering-rhino": "PR",
}


def run_foreground_patch_selection(
    config: SummaryResultsConfig,
) -> Dict[str, NDArray[Any]]:
    filter_cfg = config.filter_patches
    assert isinstance(filter_cfg, ForegroundFilterConfig)
    pred_paths = sorted(Path(config.output_path).glob("*.h5"))
    selected_patches: Dict[str, NDArray[Any]] = {}
    for pred_path in pred_paths:
        if "metric_summary" in pred_path.name:
            continue
        transfer = find_transfer_from_pred_path(str(pred_path))
        target = transfer.split("_to_")[-1]
        filename = extract_filename(pred_path)
        assert filename is not None, "filename not found in pred_path"
        if target == "S_BIAD1410":
            gt_path = list(
                Path(filter_cfg.gt_dir_path).rglob(f"{filename}/*{filename}*mask.tif")
            )
        else:
            gt_path = list(Path(filter_cfg.gt_dir_path).glob(f"*{filename}*.h5"))
        assert len(gt_path) == 1, f"Found {len(gt_path)} files for {filename}"
        gt_path = gt_path[0]
        select_patches, _ = select_foreground_patches(
            pred_path=pred_path,
            gt_path=gt_path,
            roi=filter_cfg.roi,
            gt_key=filter_cfg.gt_key,
            gt_threshold=filter_cfg.foreground_threshold,
        )
        if filter_cfg.save_selection:
            save_h5(
                pred_path,
                f"foreground_patches_th{str(filter_cfg.foreground_threshold).replace('.', '')}",
                select_patches,
                overwrite=filter_cfg.overwrite,
            )
        selected_patches[filename] = select_patches
    return selected_patches


def select_foreground_patches(
    pred_path: Union[str, Path],
    gt_path: Union[str, Path],
    roi: Optional[Sequence[Sequence[int]]] = None,
    gt_key: Optional[str] = "label",
    patch_key: str = "patch_index",
    gt_threshold: float = 0.05,
):
    if isinstance(gt_path, str):
        gt_path = Path(gt_path)
    if isinstance(pred_path, str):
        pred_path = Path(pred_path)
    if gt_path.suffix == ".h5":
        assert gt_key is not None, "gt_key must be provided for h5 files"
        gt = load_h5(gt_path, gt_key)
    else:
        gt = imageio.volread(gt_path)
        assert is_ndarray(gt), "gt must be a numpy array"
    if roi is not None:
        gt = gt[get_roi_slice(roi)]
    patches = load_h5(pred_path, patch_key)
    selected_ids, rejected_ids = threshold_patch_foreground_ratio(
        patches, gt, gt_threshold
    )
    return np.array(selected_ids), np.array(rejected_ids)


def save_summary_metrics(
    config: SummaryResultsConfig,
):
    pred_paths = sorted(Path(config.output_path).glob("*.h5"))
    assert len(pred_paths) > 0, f"No prediction files found in {config.output_path}"
    perf_scores: List[NDArray[Any]] = []
    consis_scores: List[NDArray[Any]] = []
    select_vol_patches_per_pred: Dict[str, Optional[NDArray[Any]]] = {}
    sp_key = ""  # Initialize sp_key with a default value
    for pred_path in pred_paths:
        if "metric_summary" in pred_path.name:
            continue
        filename = extract_filename(pred_path)
        assert filename is not None, "filename not fround in pred_path"
        if isinstance(config.filter_patches, ForegroundFilterConfig):
            sp_key = f"foreground_patches_th{str(config.filter_patches.foreground_threshold).replace('.', '')}"
            select_patches = load_h5(pred_path, sp_key)
        else:
            select_patches = None

        select_vol_patches_per_pred[filename] = select_patches

        if config.eval_key is not None:
            perf_score = load_select_prediction_scores(
                pred_path, config.eval_key, select_patches
            )
            perf_scores.append(perf_score)

        if "none" not in str(pred_path):
            if config.consis_key is not None:
                consis_score = load_select_prediction_scores(
                    pred_path, config.consis_key, select_patches
                )
                if consis_score.ndim >= 2:
                    consis_score_PP = calculate_per_patch_consistency(
                        consis_score, config.consis_key
                    )
                else:
                    consis_score_PP = consis_score

                assert is_ndarray(
                    consis_score_PP
                ), "consis_score_PP must be a numpy array"
                consis_scores.append(consis_score_PP)

    if len(perf_scores) > 0:
        if (perf_scores[0].ndim == 0) or (
            (perf_scores[0].ndim == 1) and (len(np.array(perf_scores[0])) == 1)
        ):
            performance_scores = np.hstack(perf_scores).squeeze()
        else:
            performance_scores = np.vstack(perf_scores).squeeze()

        mean_perf_score = np.array(np.nanmean(performance_scores, axis=0))
        median_perf_scores = np.array(np.nanmedian(performance_scores, axis=0))
        std_perf_scores = np.array(np.nanstd(performance_scores, axis=0))
    else:
        performance_scores = None
        mean_perf_score = None
        median_perf_scores = None
        std_perf_scores = None
    if len(consis_scores) > 0:
        if (consis_scores[0].ndim == 0) or (
            (consis_scores[0].ndim == 1) and (len(np.array(consis_scores[0])) == 1)
        ):
            consis_PP = np.hstack(consis_scores).squeeze()
        else:
            consis_PP = np.vstack(consis_scores).squeeze()
        consis_mean = np.array(np.nanmean(consis_PP, axis=0))
        consis_median = np.array(np.nanmedian(consis_PP, axis=0))
        consis_std = np.array(np.nanstd(consis_PP, axis=0))
    else:
        consis_PP = None
        consis_mean = None
        consis_median = None
        consis_std = None

    # save scores in h5 file in parent directory
    # save_path = Path(config.output_path).parent / "metric_summary.h5"
    save_path = (
        Path(config.output_path) / f"metric_summary{config.save_name_postfix}.h5"
    )

    with h5py.File(save_path, "a") as f:
        # check if key already exists
        if config.eval_key is not None:
            for eval_score, save_postfix in zip(
                [
                    performance_scores,
                    mean_perf_score,
                    median_perf_scores,
                    std_perf_scores,
                ],
                ["", "_mean", "_median", "_std"],
            ):
                assert is_ndarray(
                    eval_score
                ), f"performance_score{save_postfix} must be a numpy array"

                create_h5_dataset(
                    f,
                    f"{config.eval_key}{save_postfix}",
                    eval_score,
                    config.overwrite_scores,
                )

        if len(consis_scores) > 0:
            for consis_score, save_postfix in zip(
                [
                    consis_PP,
                    consis_mean,
                    consis_median,
                    consis_std,
                ],
                ["", "_mean", "_median", "_std"],
            ):
                assert is_ndarray(
                    consis_score
                ), f"consistency_score{save_postfix} must be a numpy array"
                create_h5_dataset(
                    f,
                    f"{config.consis_key}{save_postfix}",
                    consis_score,
                    config.overwrite_scores,
                )


def get_summary_results(
    source_data: List[str],
    target_data: List[str],
    selected_augmentations: Optional[Dict[str, List[str]]],
    consis_keys: Optional[Dict[str, str]],
    result_folders: Dict[str, str],
    source_models: Dict[str, str] = {
        "BBBC039": "BC_model4",
        "DSB2018": "DSB_model4",
        "Go-Nuclear": "GN_model4",
        "HeLaNuc": "HN_model1",
        "Hoechst": "Hst_model5",
        "S_BIAD634": "634_model1",
        "S_BIAD895": "895_model2",
        "S_BIAD1196": "1196_model3",
        "S_BIAD1410": "1410_model2",
        "FlyWing": "fw_model8",
        "Ovules": "ov_model8",
        "PNAS": "p_model5",
        "EPFL": "E_model4",
        "Hmito": "Hm_model3",
        "Rmito": "Rm_model3",
    },
    selected_norms: Mapping[str, Union[List[Tuple[float, float]], List[None]]] = {
        "BBBC039": [(5.0, 98.0)],
        "DSB2018": [(5.0, 98.0)],
        "Go-Nuclear": [(0, 99.8)],
        "HeLaNuc": [(5.0, 99.6)],
        "Hoechst": [(5.0, 98.0)],
        "S_BIAD634": [(5.0, 98.0)],
        "S_BIAD895": [(5.0, 98.0)],
        "S_BIAD1196": [(5.0, 98.0)],
        "S_BIAD1410": [(5.0, 98.0)],
        "FlyWing": [(5.0, 95.0)],
        "Ovules": [(5.0, 95.0)],
        "PNAS": [(5.0, 95.0)],
        "EPFL": [None],
        "Hmito": [None],
        "Rmito": [None],
        "VNC": [None],
    },
    perf_key: str = "hard_f1",
    select_results_by_source: bool = False,
    approach: str = "consistency",
    per_transfer_norms: bool = False,
    per_target_norms: bool = True,
    consis_postfix: str = "mean",
    perf_postfix: str = "mean",
    summary_results_postfix: str = "",
    base_seg_dir: str = "/g/kreshuk/talks/domain_gap/experiments/patch_segmentation",
):
    consis_PT_PA_strength: Dict[str, Dict[str, Dict[str, NDArray[Any]]]] = {}
    perf_PT_PA_strength: Dict[str, Dict[str, Dict[str, NDArray[Any]]]] = {}
    no_aug_perf_scores: Dict[str, Dict[str, float]] = {}
    for source in source_data:
        print(f"Source: {source}")
        for target in tqdm(target_data):
            transfer = (
                DATASET_ABBREVIATIONS[source] + "_to_" + DATASET_ABBREVIATIONS[target]
            )
            if select_results_by_source:
                result_folder = result_folders[source]
            else:
                result_folder = result_folders[target]
            output_dir = get_output_dir(
                source=source,
                target=target,
                model_name=source_models[source],
                output_folder=None,
                result_type=result_folder,
                approach=approach,
                base_seg_folder=base_seg_dir,
            )
            consis_PT_PN_PA_strength: Dict[str, Dict[str, NDArray[Any]]] = {}
            perf_PT_PN_PA_strength: Dict[str, Dict[str, NDArray[Any]]] = {}
            no_aug_PN_perf_scores: Dict[str, float] = {}
            if per_transfer_norms:
                norms = selected_norms[transfer]
            elif per_target_norms:
                norms = selected_norms[target]
            else:
                norms = selected_norms
            for norm in norms:
                if norm == None:
                    norm_foldername = "norm_Normalize"
                else:
                    norm_foldername = f"norm_{str(norm[0]).replace('.', '')}_{str(norm[1]).replace('.', '')}"
                norm_dir_path = Path(output_dir) / norm_foldername

                consis_per_aug_strength: Dict[str, NDArray[Any]] = {}
                perf_per_aug_strength: Dict[str, NDArray[Any]] = {}

                if selected_augmentations is None:
                    metric_filepath = list(
                        Path(norm_dir_path).rglob(
                            f"**/metric_summary{summary_results_postfix}.h5"
                        )
                    )[0]
                    perf_score = load_summary_metric(
                        metric_filepath, perf_key, perf_postfix
                    )
                    no_aug_PN_perf_scores[norm_foldername] = float(perf_score)
                else:
                    for aug, alphas in selected_augmentations.items():
                        if aug == "none":
                            metric_filepath = list(
                                (Path(norm_dir_path) / f"{aug}").rglob(
                                    f"**/metric_summary{summary_results_postfix}.h5"
                                )
                            )[0]
                            perf_score = load_summary_metric(
                                metric_filepath, perf_key, perf_postfix
                            )
                            no_aug_PN_perf_scores[norm_foldername] = float(perf_score)
                        else:
                            assert (
                                consis_keys is not None
                            ), "consis_keys must be provided"
                            consis_per_alpha = np.zeros(len(alphas))
                            perf_per_alpha = np.zeros(len(alphas))
                            for i, alpha in enumerate(alphas):
                                metric_filepath = list(
                                    (Path(norm_dir_path) / f"{aug}_{alpha}").rglob(
                                        f"**/metric_summary{summary_results_postfix}.h5"
                                    )
                                )[0]
                                consis_score = load_summary_metric(
                                    metric_filepath, consis_keys[target], consis_postfix
                                )
                                consis_per_alpha[i] = consis_score
                                perf_score = load_summary_metric(
                                    metric_filepath, perf_key, perf_postfix
                                )
                                perf_per_alpha[i] = perf_score

                            consis_per_aug_strength[aug] = consis_per_alpha
                            perf_per_aug_strength[aug] = perf_per_alpha
                consis_PT_PN_PA_strength[norm_foldername] = consis_per_aug_strength
                perf_PT_PN_PA_strength[norm_foldername] = perf_per_aug_strength
            consis_PT_PA_strength[transfer] = consis_PT_PN_PA_strength
            perf_PT_PA_strength[transfer] = perf_PT_PN_PA_strength
            no_aug_perf_scores[transfer] = no_aug_PN_perf_scores

    return (
        consis_PT_PA_strength,
        perf_PT_PA_strength,
        no_aug_perf_scores,
    )


def load_summary_metric(
    filepath: Union[Path, str], metric_key: str, metric_postfix: str
):
    with h5py.File(filepath, "r") as f:
        ds = f[f"{metric_key}_{metric_postfix}"]
        assert isinstance(
            ds, h5py.Dataset
        ), f"{metric_key}_{metric_postfix} must be a h5py.Dataset"

        score = ds[...]  # pyright: ignore[reportUnknownVariableType]

        assert is_ndarray(
            score
        ), f"{metric_key}_{metric_postfix} score must be a numpy array"

        if score.size == 2:
            score = score[1]
        elif score.size == 3:
            score = score[0]
        elif (len(score.shape) == 1) and (score.size == 0):
            score = score[0]
        elif (len(score.shape) == 0) and (score.size == 1):
            score = score
        else:
            raise ValueError(
                f"{metric_key}_{metric_postfix} score has unexpected shape {score.shape}"
            )
    return float(score)


def results_to_arrays(
    score_per_transfer: Dict[str, Dict[str, Dict[str, NDArray[Any]]]],
    no_aug_score: Dict[str, Dict[str, float]],
    perturbation_key: str,
    num_alphas: int,
):
    consis_array = np.zeros((len(score_per_transfer), num_alphas))
    no_aug_eval_array = np.zeros(len(score_per_transfer))
    for i, (transfer, per_norm_consis) in enumerate(score_per_transfer.items()):
        per_norm_NA_eval = no_aug_score[transfer]
        norm = list(per_norm_consis.keys())[0]
        no_aug_eval_array[i] = per_norm_NA_eval[norm]
        consis_array[i, :] = per_norm_consis[norm][perturbation_key]
    return consis_array, no_aug_eval_array


def perf_results_to_array(
    perf_scores: Dict[str, Dict[str, float]],
):
    perf_array = np.zeros(len(perf_scores))
    for i, (_, per_norm_perf) in enumerate(perf_scores.items()):
        norm = list(per_norm_perf.keys())[0]
        perf_array[i] = per_norm_perf[norm]
    return perf_array


def transfer_results_to_arrays(
    transfer_score: Dict[str, float],
    performance_score: Dict[str, float],
):
    transfer_array = np.zeros((len(transfer_score), 1))
    performance_array = np.zeros(len(performance_score))
    for i, (model_name, t_score) in enumerate(transfer_score.items()):
        p_score = performance_score[model_name]
        performance_array[i] = p_score
        transfer_array[i, :] = t_score
    return transfer_array, performance_array


def get_ckpt_eval_scores(
    path: Union[str, Path], checkpoint_ids: List[int], eval_key: str = "F1_eval"
) -> Tuple[List[float], List[float]]:
    mean_eval_scores: List[float] = []
    median_eval_scores: List[float] = []
    if isinstance(path, str):
        path = Path(path)
    for id in checkpoint_ids:
        checkpoint_name = f"epoch-{int(id)}"
        summary_path = path / checkpoint_name / "predictions" / "metric_summary.h5"
        eval_score_pp = load_h5(summary_path, eval_key)
        eval_mean = load_h5(summary_path, f"{eval_key}_mean")
        eval_median = load_h5(summary_path, f"{eval_key}_median")
        mean_eval_scores.append(eval_mean[1])
        median_eval_scores.append(eval_median[1])
        assert np.all(
            np.equal(np.mean(eval_score_pp, axis=0), eval_mean)
        ), f"Eval mean mismatch for {checkpoint_name} {eval_mean} vs {np.mean(eval_score_pp, axis=0)}"
    return mean_eval_scores, median_eval_scores


def per_source_model_results(
    results: Union[
        Dict[str, Dict[str, Dict[str, NDArray[Any]]]], Dict[str, Dict[str, float]]
    ],
    source_models: Dict[str, str],
    source_model_mapping: Dict[str, str] = {
        "E": "EPFL",
        "Hm": "Hmito",
        "Rm": "Rmito",
        "V": "VNC",
    },
) -> Mapping[str, Union[Mapping[str, Dict[str, NDArray[Any]]], Mapping[str, float]]]:
    model_results: Mapping[
        str, Union[Mapping[str, Dict[str, NDArray[Any]]], Mapping[str, float]]
    ] = {}
    for transfer, result in results.items():
        source = source_model_mapping[transfer.split("_")[0]]
        source_model = source_models[source]
        model_results[source_model] = result
    return model_results


def cmb_consistency_score_weighted_average(
    foreground_consistency: per_target_consis_result_type,
    background_consistency: per_target_consis_result_type,
    w_fg: float = 0.5,
    w_bg: float = 0.5,
    perturbation_key: str = "DO",
):
    per_target_cmb_consistency: per_target_consis_result_type = {}
    for target in foreground_consistency.keys():
        per_model_cmb_consistency = {}
        for model in foreground_consistency[target].keys():
            bckg_consis = background_consistency[target][model]["norm_Normalize"][
                perturbation_key
            ]
            forg_consis = foreground_consistency[target][model]["norm_Normalize"][
                perturbation_key
            ]
            cmb_consis = (w_fg * forg_consis + w_bg * bckg_consis) / (w_fg + w_bg)
            per_model_cmb_consistency[model] = {
                "norm_Normalize": {perturbation_key: cmb_consis}
            }
        per_target_cmb_consistency[target] = per_model_cmb_consistency
    return per_target_cmb_consistency


def get_NA_performance_score(
    model_name: str,
    target: str,
    base_path: Union[str, Path],
    performance_key: str = "hard_f1",
    approach: str = "consistency",
    run_id: str = "P_full",
):
    path = get_NA_prediction_path(
        model_name=model_name,
        target=target,
        base_path=base_path,
        approach=approach,
        run_id=run_id,
    )
    performance_score = load_h5(path, performance_key)
    return performance_score


def get_NA_prediction_path(
    model_name: str,
    target: str,
    base_path: Union[str, Path],
    approach: str = "consistency",
    run_id: str = "P_full",
):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    source = MODEL_ABBREVIATIONS_TO_DATASET[model_name.split("_")[0]]
    paths = list(
        base_path.rglob(
            f"{source}_to_{target}_gap/{approach}/{run_id}/{model_name}/*/none/predictions/*predictions.h5"
        )
    )
    assert (
        len(paths) == 1
    ), f"Expected exactly one path for {model_name} to {target}, found {len(paths)}"
    return paths[0]


# Convert numpy types to Python native types for JSON serialization
def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {
            str(key): convert_numpy_types(value)  # pyright: ignore
            for key, value in obj.items()  # pyright: ignore
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]  # pyright: ignore
    elif isinstance(obj, np.ndarray):
        # Handle numpy arrays by converting to list
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
        # Handle numpy scalar types explicitly
        return obj.item()
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        # Fallback for other numpy scalar types
        return obj.item()
    else:
        return obj


def save_transfer_metric_results(
    transfer_metric_per_target: Dict[str, Dict[str, float]],
    performance_per_target: Dict[str, Dict[str, float]],
    correlation_scores: Dict[str, NDArray[Any]],
    component_transfer_scores_per_target: Optional[
        Dict[str, Dict[str, Dict[str, float]]]
    ] = None,
    save_dir: str = "./gbc_results",
    experiment_name: str = "mitochondria_gbc",
    add_metadata: bool = True,
) -> None:
    """
    Save Transfer metric results in a structured format that allows for easy extension.

    Parameters:
    -----------
    transfer_metric_per_target : Dict[str, Dict[str, float]]
        Dictionary where keys are target datasets and values are dictionaries
        mapping source model names to their transfer metric scores
    performance_per_target : Dict[str, Dict[str, float]]
        Dictionary where keys are target datasets and values are dictionaries
        mapping source model names to their performance scores
    save_dir : str
        Directory to save the results
    experiment_name : str
        Name of the experiment (used in filename)
    add_metadata : bool
        Whether to include metadata about the experiment
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Convert the data
    transfer_score_converted = convert_numpy_types(transfer_metric_per_target)
    performance_converted = convert_numpy_types(performance_per_target)

    # Prepare the data structure
    results: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "transfer_scores": transfer_score_converted,
        "performance_scores": performance_converted,
        "correlation_scores": convert_numpy_types(correlation_scores),
    }

    if component_transfer_scores_per_target is not None:
        # Add component transfer scores if provided
        results["component_transfer_scores"] = convert_numpy_types(
            component_transfer_scores_per_target
        )

    if add_metadata:
        # Add metadata about the source models and targets
        all_source_models: Set[str] = set()
        for target_results in transfer_metric_per_target.values():
            all_source_models.update(target_results.keys())

        results["metadata"] = {
            "num_targets": len(transfer_metric_per_target),
            "targets": list(transfer_metric_per_target.keys()),
            "num_source_models": len(all_source_models),
            "source_models": sorted(list(all_source_models)),
            "total_transfers": sum(
                len(models) for models in transfer_metric_per_target.values()
            ),
        }

    # Save to JSON file
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Transfer metric results saved to: {filepath}")


def load_transfer_metric_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load Transfer metric results from a JSON file.

    Parameters:
    -----------
    filepath : str
        Path to the JSON file containing Transfer metric results

    Returns:
    --------
    Dict containing the loaded results
    """
    with open(filepath, "r") as f:
        results = json.load(f)

    print(f"Loaded Transfer metric results from: {filepath}")
    if "metadata" in results:
        metadata = results["metadata"]
        print(f"Experiment: {results['experiment_name']}")
        print(f"Targets: {metadata['num_targets']} ({', '.join(metadata['targets'])})")
        print(f"Source models: {metadata['num_source_models']}")
        print(f"Total transfers: {metadata['total_transfers']}")

    return results


def merge_transfer_metric_results(
    existing_results: Dict[str, Any],
    new_transfer_per_target: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Merge new transfer results with existing ones, updating source models as needed.

    Parameters:
    -----------
    existing_results : Dict[str, Any]
        Previously saved results loaded from JSON
    new_transfer_per_target : Dict[str, Dict[str, float]]
        New transfer results to merge

    Returns:
    --------
    Merged results dictionary
    """
    merged_results = existing_results.copy()
    existing_transfer = merged_results["transfer_scores"]

    for target, source_models in new_transfer_per_target.items():
        if target in existing_transfer:
            # Update existing target with new source models
            existing_transfer[target].update(source_models)
        else:
            # Add new target
            existing_transfer[target] = source_models

    # Update metadata
    if "metadata" in merged_results:
        all_source_models: set[str] = set()
        for target_results in existing_transfer.values():
            all_source_models.update(target_results.keys())

        merged_results["metadata"].update(
            {
                "num_targets": len(existing_transfer),
                "targets": list(existing_transfer.keys()),
                "num_source_models": len(all_source_models),
                "source_models": sorted(list(all_source_models)),
                "total_transfers": sum(
                    len(models) for models in existing_transfer.values()
                ),
                "last_updated": datetime.now().isoformat(),
            }
        )

    return merged_results


def find_transferability_results_path(
    base_path: Union[str, Path],
    transfer_metric: str,
    data_task: str = "mitochondria",
):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    path = base_path / f"{data_task}_{transfer_metric}.json"
    return path
