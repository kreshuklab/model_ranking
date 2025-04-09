from typing import List, Any, Union, Tuple
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
    EvalDataloaderConfig,
    TIFEvalDatasetConfig,
)
from model_ranking.datasets import StandardEvalDataset

from model_ranking.metrics import (
    AdaptedRandErrorEval,
    HammingDistanceEval,
    get_mask,
)
from model_ranking.utils import (
    save_h5,
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
    config: ConsistencyConfig,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    metric_cfg = config.consistency_metric
    # consis_cfg = config.consistency_settings
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
            perturbed_pred: NDArray[Any] = perturbed_pred.numpy()
            unperturbed_pred: NDArray[Any] = unperturbed_pred.numpy()
        if isinstance(metric, AdaptedRandErrorEval):
            batch_scores, batch_consis_mask = metric(perturbed_pred, unperturbed_pred)

        else:
            batch_consis_mask = get_mask(
                unperturbed_pred, perturbed_pred, metric_cfg.mask_threshold
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
    config_data: ConsistencyConfig,
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

    if isinstance(
        config_data.consistency_dataloader.eval_dataset, TIFEvalDatasetConfig
    ):
        pred_dir = config_data.consistency_dataloader.eval_dataset.eval.image_dir[0]
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
                config_data.consistency_metric.save_key,
                consis_scores[0][i].squeeze(),
                overwrite=config_data.consistency_metric.overwrite_score,
            )
            if config_data.consistency_metric.save_mask:
                save_h5(
                    pred_path,
                    f"consistency_mask_{config_data.consistency_metric.save_key}",
                    consis_masks[0][i].squeeze(),
                    overwrite=config_data.consistency_metric.overwrite_score,
                )

    else:
        for path, scores, mask in zip(paths, consis_scores, consis_masks):
            print(f"saving scores to {path}")
            save_h5(
                path,
                config_data.consistency_metric.save_key,
                scores.squeeze(),
                overwrite=config_data.consistency_metric.overwrite_score,
            )
            if config_data.consistency_metric.save_mask:
                save_h5(
                    path,
                    f"consistency_mask_{config_data.consistency_metric.save_key}",
                    mask.squeeze(),
                    overwrite=config_data.consistency_metric.overwrite_score,
                )

    return consis_scores, consis_masks
