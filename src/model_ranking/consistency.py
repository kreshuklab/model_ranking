import h5py  # pyright: ignore[reportMissingTypeStubs]
from typing import List, Any, Union, Dict, Literal, Optional, Tuple
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
    entropy,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.spatial.distance import hamming
from sklearn.metrics import adjusted_rand_score
from skimage.metrics import (
    adapted_rand_error,  # pyright: ignore[reportUnknownVariableType]
)
from torch.utils.data import DataLoader

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.datasets.dsb import S_BIAD1410_Dataset
from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.unet3d.utils import (
    remove_background_seg,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.augment.transforms import Relabel


from plantseg.dataprocessing import (  # pyright: ignore[reportMissingTypeStubs]
    set_background_to_value,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.dataclass import (
    ConsistencyMetricConfig,
    EvaluateConfig,
    EvalDataloaderConfig,
    TIFEvalDatasetConfig,
)
from model_ranking.datasets import StandardEvalDataset
from model_ranking.evaluation import (
    assign_unique_ids_to_value,
    get_border_mask,
    get_mask,
)
from model_ranking.metrics import (
    AdaptedRandErrorEval,
    HammingDistanceEval,
)
from model_ranking.utils import (
    extract_filename,
    load_h5,
    save_h5,
    is_ndarray,
    loader_classes,
)


def get_consistency_loaders(config: EvalDataloaderConfig):
    if config.eval_dataset.name == "StandardEvalDataset":
        dataset_class = loader_classes(config.eval_dataset.name)
        eval_datasets = dataset_class.create_datasets(config.eval_dataset)
    else:
        dataset_class = loader_classes(config.eval_dataset.name)
        eval_datasets = dataset_class.create_datasets(
            config.eval_dataset.model_dump(), phase="eval"
        )

    for dataset in eval_datasets:
        yield DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )


def calc_consistency_score(
    dataloader: DataLoader[Any],
    config: EvaluateConfig,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    metric_cfg = config.consistency_metric
    consis_cfg = config.consistency_settings
    if metric_cfg.name == "AdaptedRandError":
        metric = metric_cfg.initialise_metric(
            dataset_name=dataloader.dataset.__class__.__name__
        )
        scores = metric_cfg.initialise_score_array(
            dataloader.dataset.__len__()  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
        )

    else:
        metric = metric_cfg.initialise_metric()

        if metric_cfg.name == "Hamming-Distance":
            scores = metric_cfg.initialise_score(
                dataloader.dataset.__len__()  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
            )
        else:
            scores = metric_cfg.initialise_score(
                dataloader.dataset.__len__(),  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
                dataloader.dataset[0][0].shape,
            )
    consis_mask = np.zeros(
        (
            dataloader.dataset.__len__(),  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
            dataloader.dataset[0][0].shape,
        ),
        dtype=bool,
    )
    assert isinstance(
        dataloader.batch_size, int
    ), "dataloader.batch_size must be provided"
    for i, (perturbed_pred, unperturbed_pred) in enumerate(tqdm(dataloader)):
        if isinstance(metric, AdaptedRandErrorEval):
            batch_scores, batch_consis_mask = metric(perturbed_pred, unperturbed_pred)

        else:
            batch_consis_mask = get_mask(
                unperturbed_pred, perturbed_pred, consis_cfg.mask_threshold
            )
            if isinstance(metric, HammingDistanceEval):
                batch_scores = metric(
                    perturbed_pred, unperturbed_pred, batch_consis_mask
                )
            else:
                batch_scores = metric(perturbed_pred, unperturbed_pred)
                mask_inverted = np.logical_not(batch_consis_mask)
                batch_scores[mask_inverted] = None

        scores[
            i * dataloader.batch_size : i * dataloader.batch_size
            + perturbed_pred.shape[0]
        ] = batch_scores
        consis_mask[
            i * dataloader.batch_size : i * dataloader.batch_size
            + perturbed_pred.shape[0]
        ] = batch_consis_mask
    return scores, consis_mask


def run_consistency_evaluation(
    config_data: EvaluateConfig,
):
    paths: List[Union[Path, str]] = []
    consis_scores: List[NDArray[Any]] = []
    consis_masks: List[NDArray[Any]] = []
    for dataloader in get_consistency_loaders(config_data.consistency_dataloader):
        scores, masks = calc_consistency_score(dataloader, config_data)
        consis_scores.append(scores)
        consis_masks.append(masks)

        if isinstance(dataloader.dataset, S_BIAD1410_Dataset) | isinstance(
            dataloader.dataset, StandardHDF5Dataset
        ):
            paths.append(
                dataloader.dataset.file_path  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
            )

        elif isinstance(dataloader.dataset, StandardEvalDataset):
            paths.append(dataloader.dataset.pred_path)

    if isinstance(config_data.eval_dataloader.eval_dataset, TIFEvalDatasetConfig):
        pred_dir = config_data.eval_dataloader.eval_dataset.eval.image_dir[0]
        pred_paths = sorted(list(Path(pred_dir).glob("*.h5")))
        assert consis_scores[0] is not None, "Scores are not available"
        assert len(pred_paths) == len(
            consis_scores[0]
        ), "Number of predictions and scores differ"
        for i, pred_path in enumerate(pred_paths):
            # save scores in pred_file
            print(f"saving scores to {pred_path}")
            save_h5(
                pred_path,
                config_data.consistency_settings.save_key,
                consis_scores[0][i],
                overwrite=config_data.consistency_settings.overwrite_score,
            )

    else:
        for path, scores in zip(paths, consis_scores):
            print(f"saving scores to {path}")
            save_h5(
                path,
                config_data.eval_save_key,
                scores,
                overwrite=config_data.consistency_settings.overwrite_score,
            )

    return consis_scores


def calc_segmentation_model_consistency(paths: List[Path]):
    # find none augmentation path
    no_aug_dirs = [path.parent for path in paths if "none" in path.stem]
    for no_aug_dir in tqdm(no_aug_dirs):
        parent_dir = no_aug_dir.parent
        # find all paths in paths that contain parent_dir and not none
        aug_dirs = [
            path.parent
            for path in paths
            if parent_dir in path.parents and "none" not in str(path)
        ]
        # aug_dirs = list(parent_dir.glob("*/"))
        # aug_dirs = [path for path in aug_dirs if "none" not in str(path)]
        for aug_dir in aug_dirs:
            yaml_paths = list(aug_dir.glob("*.yml"))
            assert len(yaml_paths) == 1, (
                f"found {len(yaml_paths)} yaml files in {aug_dir} ambibiguous"
                f"which yaml to use for evaluation"
            )
            yaml_path = yaml_paths[0]
            config, _ = load_config_direct(yaml_path)
            consis_config = ConsistencyMetricConfig.model_validate(
                config["evaluation"]["consistency"]
            )
            for no_aug_path in tqdm(list(no_aug_dir.rglob("predictions/*.h5"))):
                # filename = "_".join(no_aug_path.stem.split("_")[:-1])
                filename = extract_filename(no_aug_path)
                aug_path = list(aug_dir.rglob(f"{filename}_*.h5"))
                # aug_path = list(aug_dir.rglob(f"{filename}*"))
                assert len(aug_path) == 1, (
                    f"found {len(aug_path)} files with name {filename} in {aug_dir}"
                    f"ambiguous which file to use for evaluation"
                )
                aug_path = aug_path[0]
                # check if dataset with name save_key already exists in the path
                file = h5py.File(aug_path, "r")
                if consis_config.save_key in file:
                    file.close()
                    print(f"Consistency metric already exists in {aug_path}")
                else:
                    file.close()
                    _ = run_consistency_metric(consis_config, aug_path, no_aug_path)


def run_consistency_metric(
    config: ConsistencyMetricConfig,
    aug_path: Union[Path, str],
    no_aug_path: Union[Path, str],
) -> tuple[Dict[str, NDArray[Any]], Dict[str, NDArray[Any]]]:

    if getattr(config, "ignore_path", None) is not None:
        assert config.ignore_key is not None, "ignore_key must be provided"
        assert config.ignore_path is not None, "ignore_path must be provided"
        ignore_mask = load_h5(config.ignore_path, config.ignore_key)
    else:
        ignore_mask = None
    aug_pred = load_h5(aug_path, config.pred_key)
    no_aug_pred = load_h5(no_aug_path, config.pred_key)
    if ignore_mask is not None:
        aug_pred[ignore_mask == 1] = 0
        no_aug_pred[ignore_mask == 1] = 0
    if (config.remove_background) or (config.zero_largest_instance):
        if aug_pred.ndim == 2:
            if config.zero_largest_instance:
                aug_pred = set_background_to_value(  # pyright: ignore[reportUnknownVariableType]
                    aug_pred, 0
                )
                aug_pred = Relabel()(  # pyright: ignore[reportUnknownVariableType]
                    aug_pred
                )
            else:
                aug_pred = (  # pyright: ignore[reportUnknownVariableType]
                    remove_background_seg(aug_pred)
                )
            assert is_ndarray(aug_pred), "aug_pred is not an ndarray"
        else:
            for i in range(len(aug_pred)):
                if config.zero_largest_instance:
                    pred = set_background_to_value(  # pyright: ignore[reportUnknownVariableType]
                        aug_pred[i], 0
                    )
                    pred = Relabel()(  # pyright: ignore[reportUnknownVariableType]
                        pred[0]
                    )
                    assert is_ndarray(pred), "pred is not an ndarray"
                    aug_pred[i] = np.expand_dims(pred, axis=0)
                else:
                    aug_pred[i] = remove_background_seg(aug_pred[i])

        if no_aug_pred.ndim == 2:
            if config.zero_largest_instance:
                no_aug_pred = set_background_to_value(  # pyright: ignore[reportUnknownVariableType]
                    no_aug_pred, 0
                )
                no_aug_pred = Relabel()(  # pyright: ignore[reportUnknownVariableType]
                    no_aug_pred
                )
            else:
                no_aug_pred = (  # pyright: ignore[reportUnknownVariableType]
                    remove_background_seg(no_aug_pred)
                )
            assert is_ndarray(no_aug_pred), "no_aug_pred is not an ndarray"
        else:
            for i in range(len(no_aug_pred)):
                if config.zero_largest_instance:
                    pred = set_background_to_value(  # pyright: ignore[reportUnknownVariableType]
                        no_aug_pred[i], 0
                    )
                    pred = Relabel()(  # pyright: ignore[reportUnknownVariableType]
                        pred[0]
                    )
                    assert is_ndarray(pred), "pred is not an ndarray"
                    no_aug_pred[i] = np.expand_dims(pred, axis=0)
                else:
                    no_aug_pred[i] = remove_background_seg(no_aug_pred[i])

    consis_metrics: Dict[str, NDArray[Any]] = {}
    consis_masks: Dict[str, NDArray[Any]] = {}
    for consis_threshold in config.threshold:
        save_key = f"{config.save_key}{str(consis_threshold).replace('.', '')}"
        # check if dataset with name save_key already exists in the path
        file = h5py.File(aug_path, "r")
        if save_key in file.keys():
            file.close()
            print(f"Consistency metric already exists in {aug_path}")
        else:
            file.close()
            consis_metric, mask = calc_seg_consistency_metric(
                aug_pred=aug_pred,
                no_aug_pred=no_aug_pred,
                metric=config.metric,
                threshold=consis_threshold,
                diff_alpha=config.diff_alpha,
                entr_base=config.entr_base,
                ignore_mask=ignore_mask,
                border_params=getattr(
                    config, "border_parameters", {"num_dilations": 1, "num_erosions": 1}
                ),
            )
            save_h5(aug_path, save_key, consis_metric)
            if config.save_mask:
                save_h5(aug_path, f"consistency_mask_{save_key}", mask)
            consis_metrics[save_key] = consis_metric
            consis_masks[save_key] = mask
    return consis_metrics, consis_masks


def calc_seg_consistency_metric(
    aug_pred: NDArray[Any],
    no_aug_pred: NDArray[Any],
    metric: Literal[
        "Diff",
        "EI",
        "Entropy",
        "Cross-Entropy",
        "KL-Divergence",
        "Hamming-Distance",
        "Rand-Index",
        "Adapted-Rand-Error",
        "AdaRand-Error",
    ],
    ignore_mask: Optional[NDArray[Any]] = None,
    threshold: float = 0.5,
    diff_alpha: Optional[float] = 1,
    entr_base: Optional[int] = 2,
    eps: float = 1e-7,
    border_params: Optional[Dict[str, int]] = {"num_dilations": 1, "num_erosions": 1},
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Calculate consistency metric between single aug pred array and non augmented pred array.
    The metric is calculated for pixels above threshold and set to None for non selected pixels.
    The output has the same shape as the input arrays with singleton dimensions removed.

    Args:
        aug_pred (np.ndarray): test time augmented prediction array
        no_aug_pred (np.ndarray): non augmented prediction array
        metric (Literal["Diff";, "EI", "Entropy"]): consistency metric type
        threshold (float, optional): prediction threshold to calculate consistency on. Defaults to 0.5.

    Returns:
        np.ndarray: consistency metric array with same shape as input arrays with singleton
        dimensions removed.
    """
    # Identify selected pixels (Union of aug and non
    # aug pixel predictions above threshold)
    mask = get_mask(no_aug_pred, aug_pred, threshold)
    if ignore_mask is not None:
        mask = np.logical_and(mask, ~ignore_mask.astype(bool))

    if metric == "Diff":
        metric_result = np.abs(aug_pred**diff_alpha - no_aug_pred**diff_alpha)
    elif metric == "EI":
        cmb_pred = np.stack([no_aug_pred, aug_pred], axis=0)
        hard_pred = cmb_pred > threshold
        metric_result, _, _, _ = calculate_EI_binary(hard_pred, cmb_pred)
    elif metric == "Entropy":
        cmb_pred = np.stack([no_aug_pred, aug_pred], axis=0)
        mean_pred = np.mean(cmb_pred, axis=0)
        probs = np.stack([1 - mean_pred, mean_pred], axis=0)
        metric_result = entropy(  # pyright: ignore[reportUnknownVariableType]
            probs, base=entr_base
        )
    elif metric == "KL-Divergence":
        probs_NA = np.clip(
            np.stack([1 - no_aug_pred, no_aug_pred], axis=0), eps, 1 - eps
        )
        probs_A = np.clip(np.stack([1 - aug_pred, aug_pred], axis=0), eps, 1 - eps)
        metric_result = entropy(  # pyright: ignore[reportUnknownVariableType]
            probs_NA, probs_A, base=entr_base
        )
    elif metric == "Cross-Entropy":
        probs_NA = np.clip(
            np.stack([1 - no_aug_pred, no_aug_pred], axis=0), eps, 1 - eps
        )
        probs_A = np.clip(np.stack([1 - aug_pred, aug_pred], axis=0), eps, 1 - eps)
        metric_result = entropy(probs_NA, base=entr_base) + entropy(
            probs_NA, probs_A, base=entr_base
        )
    elif metric == "Hamming-Distance":
        if aug_pred.ndim == 2:
            if np.sum(mask) == 0:
                metric_result = np.array([np.nan])
            else:
                metric_result = np.array(
                    hamming(aug_pred[mask] > threshold, no_aug_pred[mask] > threshold)
                )
        else:
            metric_result = np.zeros(len(aug_pred))
            for i in range(len(aug_pred)):
                # if mask empty set to None
                if np.sum(mask[i]) == 0:
                    metric_result[i] = np.array([np.nan])
                else:
                    metric_result[i] = hamming(
                        (aug_pred[i][mask[i]] > threshold),
                        (no_aug_pred[i][mask[i]] > threshold),
                    )
        return metric_result, mask

    elif metric == "Rand-Index":
        if aug_pred.ndim == 2:
            if np.sum(mask) == 0:
                metric_result = np.array([np.nan])
            else:
                metric_result = np.array(
                    adjusted_rand_score(no_aug_pred[mask], aug_pred[mask])
                )
        else:
            metric_result = np.zeros(len(aug_pred))
            for i in range(len(aug_pred)):
                # if mask empty set to None
                if np.sum(mask[i]) == 0:
                    metric_result[i] = np.array([np.nan])
                else:
                    metric_result[i] = adjusted_rand_score(
                        no_aug_pred[i][mask[i]], aug_pred[i][mask[i]]
                    )
        return metric_result, mask

    elif metric == "Adapted-Rand-Error":
        if aug_pred.ndim == 2:
            if np.sum(mask) == 0:
                metric_result = np.array([np.nan])
            else:
                score_are, _, _ = (  # pyright: ignore[reportUnknownVariableType]
                    adapted_rand_error(
                        no_aug_pred[mask], aug_pred[mask], ignore_labels=None
                    )
                )
                assert isinstance(score_are, float), "score_are is not a float"
                metric_result = np.array(score_are)
        else:
            metric_result = np.zeros(len(aug_pred))
            for i in range(len(aug_pred)):
                # if mask empty set to None
                if np.sum(mask[i]) == 0:
                    metric_result[i] = np.array([np.nan])
                else:
                    metric_result[i] = adapted_rand_error(
                        no_aug_pred[i][mask[i]],
                        aug_pred[i][mask[i]],
                        ignore_labels=None,
                    )[0]
        return metric_result, mask

    elif metric == "AdaRand-Error":
        assert border_params is not None, "border_params must be provided"
        metric_result, mask = adapted_rand_consis_metric(
            aug_pred, no_aug_pred, mask, border_params
        )
        return metric_result, mask

    # Remove singleton dimensions
    assert is_ndarray(metric_result), "metric_result is not an ndarray"
    metric_result = np.squeeze(metric_result)
    mask = np.squeeze(mask)
    # invert mask to get non selected pixels
    mask_inverted = np.logical_not(mask)
    # set non selected pixels to None
    metric_result[mask_inverted] = None
    return metric_result, mask


def adapted_rand_consis_metric(
    aug_pred: NDArray[Any],
    no_aug_pred: NDArray[Any],
    mask: NDArray[Any],
    border_params: Dict[str, int] = {"num_dilations": 1, "num_erosions": 1},
):
    if aug_pred.ndim == 2:
        if np.sum(mask) == 0:
            metric_result = np.array([np.nan])
            consis_mask = mask
        else:
            if np.sum(no_aug_pred) != 0:
                border_mask = get_border_mask(no_aug_pred, **border_params)
                consis_mask = np.logical_and(mask, ~border_mask)
            else:
                consis_mask = mask
            if np.sum(consis_mask) == 0:
                metric_result = np.array([np.nan])
            else:
                score_are, _, _ = (  # pyright: ignore[reportUnknownVariableType]
                    adapted_rand_error(
                        assign_unique_ids_to_value(no_aug_pred[consis_mask]),
                        assign_unique_ids_to_value(aug_pred[consis_mask]),
                        ignore_labels=None,
                    )
                )
                assert isinstance(score_are, float), "score_are is not a float"
                metric_result = np.array(score_are)
    else:
        metric_result = np.zeros(len(aug_pred))
        consis_masks = np.zeros_like(mask)
        for i in range(len(aug_pred)):
            # if mask empty set to None
            if np.sum(mask[i]) == 0:
                metric_result[i] = np.array([np.nan])
                consis_masks[i] = mask[i]
            else:
                if np.sum(no_aug_pred[i]) != 0:
                    border_mask = get_border_mask(no_aug_pred[i])
                    consis_mask = np.logical_and(mask[i], ~border_mask)
                else:
                    consis_mask = mask[i]
                if np.sum(consis_mask) == 0:
                    metric_result[i] = np.array([np.nan])
                else:
                    metric_result[i] = adapted_rand_error(
                        assign_unique_ids_to_value(no_aug_pred[i][consis_mask]),
                        assign_unique_ids_to_value(aug_pred[i][consis_mask]),
                        ignore_labels=None,
                    )[0]
                consis_masks[i] = consis_mask
        consis_mask = consis_masks
    return metric_result, consis_mask


def calculate_EI_binary(
    preds: NDArray[Any],
    soft_preds: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    # remove single dimensions
    # preds = np.squeeze(preds)
    # soft_preds = np.squeeze(soft_preds)
    # mask for change in classification prediction relative to No Aug
    mask_same_pred = preds[1:] == preds[0]
    mask0 = preds[1:] == 0
    mask1 = preds[1:] == 1
    mask0 = np.logical_and(mask0, mask_same_pred)
    mask1 = np.logical_and(mask1, mask_same_pred)
    EI_result = np.zeros_like(soft_preds[1:])
    zero_inverted_None_pred = np.where(preds[0] == 0, 1 - soft_preds[0], soft_preds[0])
    EI_result[mask1] = soft_preds[1:][mask1]
    EI_result[mask0] = 1 - soft_preds[1:][mask0]
    EI_result = np.sqrt(zero_inverted_None_pred[np.newaxis, :] * EI_result)
    return EI_result, EI_result.mean(axis=1), mask0, mask1
