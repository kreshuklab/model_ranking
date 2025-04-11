from typing import Dict, List, Mapping, Optional, Union, Any, Sequence, Tuple
from pathlib import Path
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray
import imageio.v2 as imageio
from tqdm import tqdm

from model_ranking.dataclass import (
    ForegroundFilterConfig,
    SummaryResultsConfig,
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
                overwrite=True,
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
                if "HD" not in config.consis_key:
                    if consis_score.ndim == 2:
                        consis_score_PP = np.array(np.nanmean(consis_score))

                    else:
                        consis_score_PP = np.array(
                            np.nanmean(
                                consis_score, axis=tuple(range(1, consis_score.ndim))
                            )
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
            performance_scores = np.hstack(perf_scores)
        else:
            performance_scores = np.vstack(perf_scores)

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
            consis_PP = np.hstack(consis_scores)
        else:
            consis_PP = np.vstack(consis_scores)
        consis_mean = np.array(np.nanmean(consis_PP, axis=0))
        consis_median = np.array(np.nanmedian(consis_PP, axis=0))
        consis_std = np.array(np.nanstd(consis_PP, axis=0))
    else:
        consis_PP = None
        consis_mean = None
        consis_median = None
        consis_std = None

    # save scores in h5 file in parent directory
    save_path = Path(config.output_path).parent / "metric_summary.h5"

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
    selected_augmentations: Dict[str, List[str]],
    consis_keys: Dict[str, str],
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

                for aug, alphas in selected_augmentations.items():
                    if aug == "none":
                        metric_filepath = (
                            Path(norm_dir_path) / f"{aug}" / "metric_summary.h5"
                        )
                        perf_score = load_summary_metric(
                            metric_filepath, perf_key, perf_postfix
                        )
                        no_aug_PN_perf_scores[norm_foldername] = float(perf_score)
                    else:
                        consis_per_alpha = np.zeros(len(alphas))
                        perf_per_alpha = np.zeros(len(alphas))
                        for i, alpha in enumerate(alphas):
                            metric_filepath = (
                                Path(norm_dir_path)
                                / f"{aug}_{alpha}"
                                / "metric_summary.h5"
                            )
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
