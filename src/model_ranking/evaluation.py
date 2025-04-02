import torch
import numpy as np
from typing import Any, Optional, List, Union
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score, multiclass_f1_score
from skimage.metrics import (
    adapted_rand_error,  # pyright: ignore[reportUnknownVariableType]
)
from skimage.morphology import (
    erosion,  # pyright: ignore[reportUnknownVariableType]
    dilation,  # pyright: ignore[reportUnknownVariableType]
)

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.datasets.dsb import (
    S_BIAD1410_Dataset,
)

from pytorch3dunet.unet3d.metrics import (
    DiceCoefficient,
)

from model_ranking.dataclass import (
    EvalDataloaderConfig,
    EvaluateConfig,
    TIFEvalDatasetConfig,
)
from model_ranking.datasets import StandardEvalDataset
from model_ranking.metrics import AdaptedRandErrorEval
from model_ranking.utils import save_h5, is_ndarray, loader_classes


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


def run_evaluation(
    config_data: EvaluateConfig, overwrite_score: bool = False
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
                config_data.eval_save_key,
                eval_scores[0][i],
                overwrite=overwrite_score,
            )

    else:
        for path, scores in zip(paths, eval_scores):
            print(f"saving scores to {path}")
            save_h5(
                path,
                config_data.eval_save_key,
                scores,
                overwrite=overwrite_score,
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


def adaRandError_eval(
    pred: NDArray[Any],
    gt: NDArray[Any],
    dataset_name: str,
    num_dilations: Optional[int] = 1,
    num_erosions: Optional[int] = 1,
    # border_params: Optional[Dict[str, int]] = {"num_dilations": 1, "num_erosions": 1},
):
    # check that either both or neither num_dilations and num_erosions are provided
    assert (num_dilations is not None and num_erosions is not None) or (
        num_dilations is None and num_erosions is None
    ), "Either both num_dilations and num_erosions must be provided or neither"
    batch_scores = torch.zeros((pred.shape[0], 3))
    for j in range(len(pred)):
        if gt[j].sum() == 0:
            # Prevent warning from empty GT patches
            are = float("nan")
            prec = float("nan")
            rec = float("nan")
        else:

            if dataset_name == "S_BIAD1410_Dataset":
                mask = get_mask_incomplete_gt(gt[j], pred[j])
            else:
                mask = get_mask(gt[j], pred[j], 0)
            if (num_dilations is not None) and (num_erosions is not None):
                # border_mask = get_border_mask(img=gt[j], **border_params)
                border_mask = get_border_mask(
                    img=gt[j], num_dilations=num_dilations, num_erosions=num_erosions
                )
                mask = np.logical_and(mask, ~border_mask)
            if np.sum(mask) == 0:
                are = float("nan")
                prec = float("nan")
                rec = float("nan")
            else:
                are, prec, rec = (  # pyright: ignore[reportUnknownVariableType]
                    adapted_rand_error(
                        assign_unique_ids_to_value(gt[j][mask]),
                        assign_unique_ids_to_value(pred[j][mask]),
                        ignore_labels=None,
                    )
                )
                assert isinstance(are, float), f"are is not a float: {are}"
                assert isinstance(prec, float), f"prec is not a float: {prec}"
                assert isinstance(rec, float), f"rec is not a float: {rec}"
        batch_scores[j, 0] = are
        batch_scores[j, 1] = prec
        batch_scores[j, 2] = rec
    return batch_scores


def get_mask_incomplete_gt(
    gt: NDArray[Any], pred: NDArray[Any], value: int = 0
) -> NDArray[Any]:
    gt_mask = gt > value
    # find ids in masked region of prediction
    pred_ids = np.unique(pred[gt_mask])
    # remove 0 from pred_ids
    pred_ids = pred_ids[pred_ids != 0]
    # find mask of region containing pred_ids
    mask = np.zeros_like(pred, dtype=bool)
    for pred_id in pred_ids:
        mask = np.logical_or(mask, pred == pred_id)
    mask = np.logical_or(mask, gt_mask)
    return mask


def get_mask(
    pred_none: NDArray[Any], pred_aug: NDArray[Any], threshold: float = 0.5
) -> NDArray[Any]:
    masks = np.zeros((2, *pred_none.shape))
    masks[0] = pred_none > threshold
    masks[1] = pred_aug > threshold
    # Combine masks across augmentations (Union)
    combined_mask = np.logical_or.reduce(masks, axis=0)
    assert is_ndarray(combined_mask), f"Data is not a numpy array: {combined_mask}"
    return combined_mask


def get_border_mask(
    img: NDArray[Any], num_dilations: int = 1, num_erosions: int = 1
) -> NDArray[Any]:
    dilated = img.copy()
    eroded = img.copy()
    for _ in range(num_dilations):
        dilated = dilation(dilated)  # pyright: ignore[reportUnknownVariableType]
        assert is_ndarray(dilated), f"Data is not a numpy array: {dilated}"
    for _ in range(num_erosions):
        eroded = erosion(eroded)  # pyright: ignore[reportUnknownVariableType]
        assert is_ndarray(eroded), f"Data is not a numpy array: {eroded}"
    # return np.logical_and(dilated != eroded, img == 0)
    return (dilated - eroded) > 0


def assign_unique_ids_to_value(data: NDArray[Any], value: int = 0):
    data = data.copy()
    max_val = np.max(data)
    max_id_assigned = max_val + np.sum(data == value) + 1
    # check for overflow error
    if np.iinfo(data.dtype).max < max_id_assigned:
        for dtype in [np.uint16, np.uint32, np.uint64]:
            if np.iinfo(dtype).max >= max_id_assigned:
                # incease dtype size by one
                data = data.astype(dtype)
                break
        assert np.iinfo(data.dtype).max >= max_id_assigned, "Overflow error"
    data[data == value] = np.arange(max_val + 1, max_id_assigned)
    return data


class MultiClassF1:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros((pred.shape[0], 2))
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            pred_th = (pred[j] > self.threshold).type(torch.int64)
            if gt[j].sum() == 0:
                # Prevent warning from empty GT patches
                batch_perf_scores[j, 1] = 0
                batch_perf_scores[j, 0] = binary_f1_score(
                    (1 - pred[j]).flatten(),
                    (1 - gt[j].type(torch.int64)).flatten(),
                    threshold=self.threshold,
                )
            else:
                batch_perf_scores[j] = multiclass_f1_score(
                    pred_th.flatten(),
                    gt[j].flatten().type(torch.int64),
                    num_classes=2,
                    average=None,
                )
        return batch_perf_scores


class BinaryF1:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros(pred.shape[0])
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            if gt[j].sum() == 0:
                # Prevent warning from empty GT patches
                batch_perf_scores[j] = 0
            else:
                batch_perf_scores[j] = binary_f1_score(
                    pred[j].flatten(), gt[j].flatten(), threshold=self.threshold
                )
        return batch_perf_scores


class SoftF1:
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros(pred.shape[0])
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            # add channel dimension to gt if needed
            if len(gt.shape) == 4:
                gt_patch = torch.unsqueeze(gt[j : j + 1], 1)
            else:
                gt_patch = gt[j : j + 1]
            batch_perf_scores[j] = DiceCoefficient()(pred[j : j + 1], gt_patch)
        return batch_perf_scores


class AdaptedRandError:
    def __init__(
        self,
        dataset_name: str,
        num_dilations: Optional[int] = 1,
        num_erosions: Optional[int] = 1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_dilations = num_dilations
        self.num_erosions = num_erosions

    def __call__(self, pred: NDArray[Any], gt: NDArray[Any]) -> torch.Tensor:
        return adaRandError_eval(
            pred,
            gt,
            self.dataset_name,
            num_dilations=self.num_dilations,
            num_erosions=self.num_erosions,
        )
