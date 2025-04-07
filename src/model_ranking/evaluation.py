import torch
import numpy as np
from typing import Any, List, Union
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.datasets.dsb import (
    S_BIAD1410_Dataset,
)

from model_ranking.dataclass import (
    EvalDataloaderConfig,
    EvaluateConfig,
    TIFEvalDatasetConfig,
)
from model_ranking.datasets import StandardEvalDataset
from model_ranking.metrics import AdaptedRandErrorEval
from model_ranking.utils import save_h5, loader_classes


def get_evaluation_loaders(config: EvalDataloaderConfig):
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


def run_performance_evaluation(
    config_data: EvaluateConfig,
) -> List[NDArray[Any]]:
    # dataloader = get_evaluation_loaders(config_data.eval_dataloader)
    # check is cuda available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # metric = get_evaluation_metric_dataclass(config_data.eval_metric)
    paths: List[Union[Path, str]] = []
    eval_scores: List[NDArray[Any]] = []
    for dataloader in get_evaluation_loaders(config_data.eval_dataloader):
        eval_scores.append(
            calc_evaluation_score(
                dataloader=dataloader,
                device=device,
                config=config_data,
            )
        )

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
        assert eval_scores[0] is not None, "Scores are not available"
        assert len(pred_paths) == len(
            eval_scores[0]
        ), "Number of predictions and scores differ"
        for i, pred_path in enumerate(pred_paths):
            # save scores in pred_file
            print(f"saving scores to {pred_path}")
            save_h5(
                pred_path,
                config_data.eval_metric.eval_save_key,
                eval_scores[0][i],
                overwrite=config_data.eval_metric.overwrite_score,
            )

    else:
        for path, scores in zip(paths, eval_scores):
            print(f"saving scores to {path}")
            save_h5(
                path,
                config_data.eval_metric.eval_save_key,
                scores,
                overwrite=config_data.eval_metric.overwrite_score,
            )

    return eval_scores


def calc_evaluation_score(
    dataloader: DataLoader[Any],
    device: str,
    config: EvaluateConfig,
) -> NDArray[Any]:
    metric_cfg = config.eval_metric
    if metric_cfg.name == "AdaptedRandError":
        metric = metric_cfg.initialise_metric(
            dataset_name=dataloader.dataset.__class__.__name__
        )
    else:
        metric = metric_cfg.initialise_metric()

    # Intialise Tensor to score eval scores
    scores = metric_cfg.initialise_score(
        dataloader.dataset.__len__()  # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue]
    )

    eval_scores: NDArray[Any] = np.array([])
    assert isinstance(dataloader.batch_size, int), "batch_size must be provided"
    for i, (pred, gt) in enumerate(tqdm(dataloader)):
        pred = pred.to(device)
        gt = gt.to(device)
        if isinstance(metric, AdaptedRandErrorEval):
            metric_scores, _ = metric(pred.cpu().numpy(), gt.cpu().numpy())
            metric_scores = torch.tensor(metric_scores, device=device)
        else:
            metric_scores = metric(pred, gt)
        scores[
            i * dataloader.batch_size : i * dataloader.batch_size + pred.shape[0]
        ] = metric_scores
    eval_scores = scores.cpu().numpy()
    return eval_scores
