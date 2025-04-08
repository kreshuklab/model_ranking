from typing import Dict, List, Optional, Union, Any, Sequence
from pathlib import Path
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray
import imageio.v2 as imageio

from model_ranking.dataclass import (
    ForegroundFilterConfig,
    SaveResultsConfig,
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
)


def run_foreground_patch_selection(
    config: SaveResultsConfig,
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
    config: SaveResultsConfig,
):
    pred_paths = sorted(Path(config.output_path).glob("*.h5"))
    assert len(pred_paths) > 0, f"No prediction files found in {config.output_path}"
    perf_scores: List[NDArray[Any]] = []
    consis_scores: List[NDArray[Any]] = []
    select_vol_patches_per_pred: Dict[str, NDArray[Any]] = {}
    sp_key = ""  # Initialize sp_key with a default value
    for pred_path in pred_paths:
        if isinstance(config.filter_patches, ForegroundFilterConfig):
            sp_key = f"foreground_patches_th{str(config.filter_patches.foreground_threshold).replace('.', '')}"
            select_patches = load_h5(pred_path, sp_key)
            filename = extract_filename(pred_path)
            assert filename is not None, "filename not fround in pred_path"
            select_vol_patches_per_pred[filename] = select_patches
        else:
            select_patches = None

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
                        consis_score_PP = np.nanmean(consis_score)

                    else:
                        consis_score_PP = np.nanmean(
                            consis_score, axis=tuple(range(1, consis_score.ndim))
                        )

                else:
                    consis_score_PP = consis_score
                assert is_ndarray(
                    consis_score_PP
                ), "consis_score_PP must be a numpy array"
                consis_scores.append(consis_score_PP)

    if len(perf_scores) > 0:
        if (
            (perf_scores[0].ndim == 0)
            or (perf_scores[0].ndim == 1)
            and (len(np.array(perf_scores[0])) != 3)
        ):
            performance_scores = np.hstack(perf_scores)
        else:
            performance_scores = np.vstack(perf_scores)

        mean_perf_score = np.nanmean(performance_scores, axis=0)
        median_perf_scores = np.nanmedian(performance_scores, axis=0)
        std_perf_scores = np.nanstd(performance_scores, axis=0)
    else:
        performance_scores = None
        mean_perf_score = None
        median_perf_scores = None
        std_perf_scores = None
    if len(consis_scores) > 0:
        if (
            (consis_scores[0].ndim == 0)
            or (consis_scores[0].ndim == 1)
            and (len(np.array(consis_scores[0])) != 3)
        ):
            consis_PP = np.hstack(consis_scores)
        else:
            consis_PP = np.vstack(consis_scores)
        consis_per_alpha = np.nanmean(consis_PP, axis=0)
        consis_median_per_alpha = np.nanmedian(consis_PP, axis=0)
        consis_std_per_alpha = np.nanstd(consis_PP, axis=0)
    else:
        consis_PP = None
        consis_per_alpha = None
        consis_median_per_alpha = None
        consis_std_per_alpha = None

    # save scores in h5 file in parent directory
    save_path = Path(config.output_path).parent / "metric_summary.h5"

    with h5py.File(save_path, "a") as f:
        # check if key already exists
        if config.eval_key is not None:
            assert is_ndarray(
                performance_scores
            ), "performance_scores must be a numpy array"
            assert is_ndarray(mean_perf_score), "mean_perf_score must be a numpy array"
            assert is_ndarray(
                median_perf_scores
            ), "median_perf_scores must be a numpy array"
            assert is_ndarray(std_perf_scores), "std_perf_scores must be a numpy array"
            create_h5_dataset(
                f, config.eval_key, performance_scores, config.overwrite_scores
            )
            create_h5_dataset(
                f, f"{config.eval_key}_mean", mean_perf_score, config.overwrite_scores
            )
            create_h5_dataset(
                f,
                f"{config.eval_key}_median",
                median_perf_scores,
                config.overwrite_scores,
            )
            create_h5_dataset(
                f, f"{config.eval_key}_std", std_perf_scores, config.overwrite_scores
            )

        if config.save_select_patches is True:
            assert isinstance(config.filter_patches, ForegroundFilterConfig)
            for key, patches in select_vol_patches_per_pred.items():
                create_h5_dataset(
                    f, f"{key}_{sp_key}", np.array(patches), config.overwrite_scores
                )
        if len(consis_scores) > 0:
            assert is_ndarray(consis_PP), "consis_PP must be a numpy array"
            assert is_ndarray(
                consis_per_alpha
            ), "consis_per_alpha must be a numpy array"
            assert is_ndarray(
                consis_median_per_alpha
            ), "consis_median_per_alpha must be a numpy array"
            assert is_ndarray(
                consis_std_per_alpha
            ), "consis_std_per_alpha must be a numpy array"
            create_h5_dataset(
                f, f"{config.consis_key}_per_patch", consis_PP, config.overwrite_scores
            )
            create_h5_dataset(
                f,
                f"{config.consis_key}_per_alpha",
                consis_per_alpha,
                config.overwrite_scores,
            )
            create_h5_dataset(
                f,
                f"{config.consis_key}_median_per_alpha",
                consis_median_per_alpha,
                config.overwrite_scores,
            )
            create_h5_dataset(
                f,
                f"{config.consis_key}_std_per_alpha",
                consis_std_per_alpha,
                config.overwrite_scores,
            )
