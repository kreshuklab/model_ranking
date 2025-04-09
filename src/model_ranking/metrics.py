import torch
from typing import Optional, Any, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from torcheval.metrics.functional import binary_f1_score, multiclass_f1_score
from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
    entropy,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.spatial.distance import hamming
from skimage.morphology import (
    erosion,  # pyright: ignore[reportUnknownVariableType]
    dilation,  # pyright: ignore[reportUnknownVariableType]
)
from skimage.metrics import (
    adapted_rand_error,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.metrics import (
    DiceCoefficient,
)
from model_ranking.utils import is_ndarray, avoid_int_overflow


class MultiClassF1Eval:
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


class BinaryF1Eval:
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


class SoftF1Eval:
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


class AdaptedRandErrorEval:
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

    def __call__(
        self, pred: NDArray[Any], gt: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        pred_converted = avoid_int_overflow(pred.astype(np.uint16), np.max(pred))
        gt_converted = avoid_int_overflow(gt.astype(np.uint16), np.max(gt))
        # pred_converted = pred.cpu().numpy().astype("uint16")
        # gt_converted = gt.cpu().numpy().astype("uint16")
        metric_result, mask = adaRandError_eval(
            pred_converted,
            gt_converted,
            self.dataset_name,
            num_dilations=self.num_dilations,
            num_erosions=self.num_erosions,
        )
        return (metric_result, mask)
        # return (
        #    torch.from_numpy(metric_result).to(pred.device).float(),
        #    torch.from_numpy(mask).to(pred.device).float(),
        # )


class DifferenceImageEval:
    def __init__(self, diff_alpha: float = 0.5):

        super().__init__()
        self.diff_alpha = diff_alpha

    def __call__(self, pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
        return np.abs(pred**self.diff_alpha - gt**self.diff_alpha)


class EffectiveInvarianceEval:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
        # cmb_pred = np.stack([gt, pred], axis=0)
        cmb_pred = np.vstack([gt, pred])
        hard_pred = cmb_pred > self.threshold
        metric_result, _, _, _ = calculate_EI_binary(hard_pred, cmb_pred)
        return metric_result


class EntropyEval:
    def __init__(self, entr_base: int = 2):
        super().__init__()
        self.entr_base = entr_base

    def __call__(self, pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("float32")
        # gt_converted = gt.cpu().numpy().astype("float32")
        # cmb_pred = np.stack([gt_converted, pred_converted], axis=0)
        cmb_pred = np.stack([gt, pred], axis=0)
        mean_pred = np.mean(cmb_pred, axis=0)
        probs = np.stack([1 - mean_pred, mean_pred], axis=0)
        metric_result = entropy(  # pyright: ignore[reportUnknownVariableType]
            probs, base=self.entr_base
        )
        assert is_ndarray(metric_result), f"Data is not a numpy array: {metric_result}"
        return metric_result
        # return torch.from_numpy(metric_result).to(pred.device).float()


class KLDivergenceEval:
    def __init__(self, eps: float = 1e-7, entr_base: int = 2):
        super().__init__()
        self.eps = eps
        self.entr_base = entr_base

    def __call__(self, pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("float32")
        # gt_converted = gt.cpu().numpy().astype("float32")
        probs_NA = np.clip(np.stack([1 - gt, gt], axis=0), self.eps, 1 - self.eps)
        probs_A = np.clip(
            np.stack([1 - pred, pred], axis=0),
            self.eps,
            1 - self.eps,
        )
        metric_result = entropy(  # pyright: ignore[reportUnknownVariableType]
            probs_NA, probs_A, base=self.entr_base
        )
        assert is_ndarray(metric_result), f"Data is not a numpy array: {metric_result}"
        # return torch.from_numpy(metric_result).to(pred.device).float()
        return metric_result


class CrossEntropyEval:
    def __init__(self, eps: float = 1e-7, entr_base: float = 2.0):
        super().__init__()
        self.eps = eps
        self.entr_base = entr_base

    def __call__(self, pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("float32")
        # gt_converted = gt.cpu().numpy().astype("float32")
        probs_NA = np.clip(np.stack([1 - gt, gt], axis=0), self.eps, 1 - self.eps)
        probs_A = np.clip(
            np.stack([1 - pred, pred], axis=0),
            self.eps,
            1 - self.eps,
        )
        metric_result = entropy(probs_NA, base=self.entr_base) + entropy(
            probs_NA, probs_A, base=self.entr_base
        )
        assert is_ndarray(metric_result), f"Data is not a numpy array: {metric_result}"
        # return torch.from_numpy(metric_result).to(pred.device).float()
        return metric_result


class HammingDistanceEval:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(
        self, pred: NDArray[Any], gt: NDArray[Any], mask: NDArray[Any]
    ) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("uint16")
        # gt_converted = gt.cpu().numpy().astype("uint16")
        # mask_converted = mask.cpu().numpy().astype("bool")
        if pred.ndim == 2:
            if np.sum(mask) == 0:
                metric_result = np.array([np.nan])
            else:
                metric_result = np.array(
                    hamming(
                        pred[mask] > self.threshold,
                        gt[mask] > self.threshold,
                    )
                )
        else:
            metric_result = np.zeros(len(pred))
            for i in range(len(pred)):
                # if mask empty set to None
                if np.sum(mask[i]) == 0:
                    metric_result[i] = np.array([np.nan])
                else:
                    metric_result[i] = hamming(
                        (pred[i][mask[i]] > self.threshold),
                        (gt[i][mask[i]] > self.threshold),
                    )
        assert is_ndarray(metric_result), f"Data is not a numpy array: {metric_result}"
        # return torch.from_numpy(metric_result).to(pred.device).float()
        return metric_result


def adaRandError_eval(
    pred: NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]],
    gt: NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]],
    dataset_name: str,
    num_dilations: Optional[int] = 1,
    num_erosions: Optional[int] = 1,
    # border_params: Optional[Dict[str, int]] = {"num_dilations": 1, "num_erosions": 1},
) -> Tuple[NDArray[Any], NDArray[Any]]:
    # check that either both or neither num_dilations and num_erosions are provided
    assert (num_dilations is not None and num_erosions is not None) or (
        num_dilations is None and num_erosions is None
    ), "Either both num_dilations and num_erosions must be provided or neither"
    batch_scores = np.zeros((pred.shape[0], 3), dtype=np.float32)
    consis_mask = np.zeros_like(pred)
    if pred.ndim == 2:
        pred = np.expand_dims(pred, axis=0)
        gt = np.expand_dims(gt, axis=0)
    # check that pred and gt have the same shape
    assert (
        pred.shape == gt.shape
    ), f"pred and gt have different shapes: {pred.shape} {gt.shape}"
    for j in range(len(pred)):
        if gt[j].sum() == 0:
            # Prevent warning from empty GT patches
            are = float("nan")
            prec = float("nan")
            rec = float("nan")
            consis_mask[j] = get_mask(gt[j], pred[j], 0)

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
            consis_mask[j] = mask
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
    return batch_scores, consis_mask


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
    data = avoid_int_overflow(data, max_id_assigned)
    data[data == value] = np.arange(max_val + 1, max_id_assigned)
    return data


def calculate_EI_binary_tensors(
    preds: torch.Tensor,
    soft_preds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Mask for change in classification prediction relative to No Aug
    mask_same_pred = preds[1:] == preds[0]
    mask0 = preds[1:] == 0
    mask1 = preds[1:] == 1
    mask0 = torch.logical_and(mask0, mask_same_pred)
    mask1 = torch.logical_and(mask1, mask_same_pred)

    EI_result = torch.zeros_like(soft_preds[1:])
    zero_inverted_None_pred = torch.where(
        preds[0] == 0, 1 - soft_preds[0], soft_preds[0]
    )
    EI_result[mask1] = soft_preds[1:][mask1]
    EI_result[mask0] = 1 - soft_preds[1:][mask0]
    EI_result = torch.sqrt(zero_inverted_None_pred.unsqueeze(0) * EI_result)

    return EI_result, EI_result.mean(dim=1), mask0, mask1


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
