from typing import List, Any, Union, Tuple
from natsort import natsorted
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch3dunet.datasets.dsb import S_BIAD1410_Dataset
from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset

from model_ranking.dataclass import (
    ConsistencyConfig,
    ConsistencyPatchedTransformerConfig,
    EvalDataloaderConfig,
    TIFEvalDatasetConfig,
)
from model_ranking.datasets import StandardEvalDataset
from model_ranking.metrics import (
    AdaptedRandErrorEval,
    get_mask,
)
from model_ranking.utils import (
    save_h5,
    loader_classes,
    is_ndarray,
    load_predictions_transformers,
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
    config: ConsistencyConfig,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    metric_cfg = config.consistency_metric
    # consis_cfg = config.consistency_settings
    if metric_cfg.name == "AdaptedRandError":
        if dataloader.dataset.__class__.__name__ == "S_BIAD1410_Dataset":
            metric = metric_cfg.initialise_metric(incomplete_gt=True)
        else:
            metric = metric_cfg.initialise_metric(incomplete_gt=False)
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
                tuple(dataloader.dataset[0][0].shape),
            )
    consis_mask = np.zeros(
        (
            (
                dataloader.dataset.__len__(),  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
            )
            + tuple(dataloader.dataset[0][0].shape)
        ),
        dtype=bool,
    )
    assert isinstance(
        dataloader.batch_size, int
    ), "dataloader.batch_size must be provided"
    for i, (perturbed_pred, unperturbed_pred) in enumerate(tqdm(dataloader)):

        if isinstance(perturbed_pred, torch.Tensor):
            assert isinstance(
                unperturbed_pred, torch.Tensor
            ), "unperturbed_pred is not a torch.Tensor"
            perturbed_pred = (  # pyright: ignore[reportUnknownVariableType]
                perturbed_pred.numpy()
            )
            unperturbed_pred = (  # pyright: ignore[reportUnknownVariableType]
                unperturbed_pred.numpy()
            )
            assert is_ndarray(perturbed_pred), "perturbed_pred is not a numpy array"
            assert is_ndarray(unperturbed_pred), "unperturbed_pred is not a numpy array"
        if isinstance(metric, AdaptedRandErrorEval):
            batch_scores, batch_consis_mask = metric(perturbed_pred, unperturbed_pred)

        else:
            if metric_cfg.bckg_consistency:
                unperturbed_pred = 1 - unperturbed_pred
                perturbed_pred = 1 - perturbed_pred
                threshold = 1 - metric_cfg.mask_threshold
            else:
                threshold = metric_cfg.mask_threshold
            batch_consis_mask = get_mask(
                unperturbed_pred,
                perturbed_pred,
                threshold,
            )
            batch_scores = metric(perturbed_pred, unperturbed_pred, batch_consis_mask)

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
    config_data: ConsistencyConfig,
):
    paths: List[Union[Path, str]] = []
    consis_scores: List[NDArray[Any]] = []
    consis_masks: List[NDArray[Any]] = []
    metric_cfg = config_data.consistency_metric
    loader_cfg = config_data.consistency_dataloader
    for dataloader in get_consistency_loaders(loader_cfg):
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

    if metric_cfg.save_key is not None:
        assert (
            metric_cfg.overwrite_score is not None
        ), "boolean overwrite_score must be provided if save_key is set"
        if isinstance(loader_cfg.eval_dataset, TIFEvalDatasetConfig):
            pred_dir = loader_cfg.eval_dataset.eval.image_dir[0]
            pred_paths = natsorted(list(Path(pred_dir).glob("*.h5")))
            assert consis_scores[0] is not None, "Scores are not available"
            assert len(pred_paths) == len(
                consis_scores[0]
            ), "Number of predictions and scores differ"
            for i, pred_path in enumerate(pred_paths):
                if "metric_summary" in str(pred_path.stem):
                    continue
                # save scores in pred_file
                consis_PP = calculate_per_patch_consistency(
                    consis_scores[0][i].squeeze(), metric_cfg.name
                )
                print(f"saving scores to {pred_path}")
                save_h5(
                    pred_path,
                    metric_cfg.save_key,
                    consis_PP,
                    # consis_scores[0][i].squeeze(),
                    overwrite=metric_cfg.overwrite_score,
                )
                if config_data.consistency_metric.save_mask:
                    save_h5(
                        pred_path,
                        f"consistency_mask_{config_data.consistency_metric.save_key}",
                        consis_masks[0][i].squeeze(),
                        overwrite=metric_cfg.overwrite_score,
                    )

        else:
            for path, scores, mask in zip(paths, consis_scores, consis_masks):
                print(f"saving scores to {path}")
                consis_PP = calculate_per_patch_consistency(
                    scores.squeeze(), metric_cfg.save_key
                )
                save_h5(
                    path,
                    metric_cfg.save_key,
                    # scores.squeeze(),
                    consis_PP,
                    overwrite=metric_cfg.overwrite_score,
                )
                if metric_cfg.save_mask:
                    save_h5(
                        path,
                        f"consistency_mask_{metric_cfg.save_key}",
                        mask.squeeze(),
                        overwrite=metric_cfg.overwrite_score,
                    )

    return consis_scores, consis_masks


def calculate_per_patch_consistency(consis_score: NDArray[Any], consis_name: str):
    if ("Hamming-Distance" in consis_name) or ("AdaptedRandError" in consis_name):
        consis_score_PP = consis_score

    else:
        if consis_score.ndim == 2:
            consis_score_PP = np.array(np.nanmean(consis_score))

        else:
            consis_score_PP = np.array(
                np.nanmean(consis_score, axis=tuple(range(1, consis_score.ndim)))
            )

    return consis_score_PP


def run_patched_transformer_consistency(
    config: ConsistencyPatchedTransformerConfig,
):
    perturbed_cfg = config.predictions_perturbed
    unperturbed_cfg = config.predictions_unperturbed
    metric_cfg = config.metric_config

    perturbed_preds, perturbed_path = load_predictions_transformers(
        model_name=perturbed_cfg.model_name,
        TTA_aug=perturbed_cfg.TTA_key,
        base_dir_path=perturbed_cfg.base_dir_path,
        file_identifier=perturbed_cfg.file_identifier,
    )
    assert len(perturbed_path) == 1, "Only one prediction file supported"
    unperturbed_preds, unperturbed_path = load_predictions_transformers(
        model_name=unperturbed_cfg.model_name,
        TTA_aug=unperturbed_cfg.TTA_key,
        base_dir_path=unperturbed_cfg.base_dir_path,
        file_identifier=unperturbed_cfg.file_identifier,
    )
    assert len(unperturbed_path) == 1, "Only one prediction file supported"

    assert perturbed_preds.shape == unperturbed_preds.shape, (
        f"Shape mismatch between perturbed {perturbed_preds.shape} and"
        + f" unperturbed {unperturbed_preds.shape} predictions"
    )

    metric = metric_cfg.initialise_metric(incomplete_gt=False)
    scores = metric_cfg.initialise_score_array(len(perturbed_preds))
    consis_masks = np.zeros_like(perturbed_preds, dtype=bool)

    for i, (p_pred, unp_pred) in enumerate(zip(perturbed_preds, unperturbed_preds)):
        scores[i], consis_masks[i] = metric(p_pred, unp_pred)

    if metric_cfg.save_key is not None:
        save_h5(
            perturbed_path[0],
            metric_cfg.save_key,
            scores,
            overwrite=config.overwrite_scores,
        )
        if metric_cfg.save_mask:
            save_h5(
                perturbed_path[0],
                metric_cfg.save_key + "_mask",
                consis_masks,
                overwrite=config.overwrite_scores,
            )
