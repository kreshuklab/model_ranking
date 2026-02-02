import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Sequence, Tuple, Union, Literal
import numpy as np
from numpy.typing import NDArray
from torcheval.metrics.functional import (
    binary_f1_score,
    multiclass_f1_score,
    binary_accuracy,
)
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
from tqdm import tqdm
from torch_em.loss.dice import (
    dice_score,
)
from collections import Counter

from pytorch3dunet.unet3d.metrics import (
    DiceCoefficient,
)
from model_ranking.utils import is_ndarray, avoid_int_overflow, is_torch_tensor


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


class BinaryAccuracyEval:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros(pred.shape[0])
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            batch_perf_scores[j] = binary_accuracy(
                pred[j].flatten(), gt[j].long().flatten(), threshold=self.threshold
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
        incomplete_gt: bool,
        num_dilations: Optional[int] = 1,
        num_erosions: Optional[int] = 1,
    ):
        super().__init__()
        self.incomplete_gt = incomplete_gt
        self.num_dilations = num_dilations
        self.num_erosions = num_erosions

    def __call__(
        self, pred: NDArray[Any], gt: NDArray[Any], bckg: bool = False
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        pred_converted = avoid_int_overflow(pred.astype(np.uint16), np.max(pred))
        gt_converted = avoid_int_overflow(gt.astype(np.uint16), np.max(gt))
        # pred_converted = pred.cpu().numpy().astype("uint16")
        # gt_converted = gt.cpu().numpy().astype("uint16")
        metric_result, mask = adaRandError_eval(
            pred_converted,
            gt_converted,
            self.incomplete_gt,
            num_dilations=self.num_dilations,
            num_erosions=self.num_erosions,
            bckg=bckg,
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

    def __call__(
        self,
        pred: NDArray[Any],
        gt: NDArray[Any],
        consis_mask: Optional[NDArray[Any]] = None,
    ) -> NDArray[Any]:
        metric_result = np.abs(pred**self.diff_alpha - gt**self.diff_alpha)

        if consis_mask is not None:
            mask_inverted = np.logical_not(consis_mask)
            metric_result[mask_inverted] = None

        return metric_result


class EffectiveInvarianceEval:
    def __init__(self, threshold: float = 0.5, invert_threshold: bool = False):
        super().__init__()
        if invert_threshold:
            self.threshold = 1 - threshold
        else:
            self.threshold = threshold

    def __call__(
        self,
        pred: NDArray[Any],
        gt: NDArray[Any],
        consis_mask: Optional[NDArray[Any]] = None,
    ) -> NDArray[Any]:
        # cmb_pred = np.stack([gt, pred], axis=0)
        if pred.ndim == 2:
            flag_2d = True
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)
            if consis_mask is not None:
                consis_mask = np.expand_dims(consis_mask, axis=0)
        else:
            flag_2d = False
        metric_result = np.zeros_like(pred)
        for i in range(len(pred)):
            # pred_converted = pred.cpu().numpy().astype("float32")
            cmb_pred = np.stack([gt[i], pred[i]])
            hard_pred = cmb_pred > self.threshold
            metric_result_pp, _, _, _ = calculate_EI_binary(hard_pred, cmb_pred)
            if consis_mask is not None:
                mask_inverted = np.logical_not(consis_mask[i : i + 1])
                metric_result_pp[mask_inverted] = None
            metric_result[i] = metric_result_pp
        if flag_2d:
            metric_result = np.squeeze(metric_result)
        return metric_result


class EntropyEval:
    def __init__(self, entr_base: int = 2):
        super().__init__()
        self.entr_base = entr_base

    def __call__(
        self,
        pred: NDArray[Any],
        gt: NDArray[Any],
        consis_mask: Optional[NDArray[Any]] = None,
    ) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("float32")
        # gt_converted = gt.cpu().numpy().astype("float32")
        # cmb_pred = np.stack([gt_converted, pred_converted], axis=0)
        if pred.ndim == 2:
            flag_2d = True
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)
            if consis_mask is not None:
                consis_mask = np.expand_dims(consis_mask, axis=0)
        else:
            flag_2d = False

        metric_result = np.zeros_like(pred)
        for i in range(len(pred)):
            cmb_pred = np.stack([gt[i], pred[i]], axis=0)
            mean_pred = np.mean(cmb_pred, axis=0)
            probs = np.stack([1 - mean_pred, mean_pred], axis=0)
            metric_result_pp = entropy(  # pyright: ignore[reportUnknownVariableType]
                probs, base=self.entr_base
            )
            assert is_ndarray(
                metric_result_pp
            ), f"Data is not a numpy array: {metric_result_pp}"
            if consis_mask is not None:
                mask_inverted = np.logical_not(consis_mask[i])
                metric_result_pp[mask_inverted] = None
            metric_result[i] = metric_result_pp
        if flag_2d:
            metric_result = np.squeeze(metric_result)
        return metric_result
        # return torch.from_numpy(metric_result).to(pred.device).float()


class KLDivergenceEval:
    def __init__(self, eps: float = 1e-7, entr_base: int = 2):
        super().__init__()
        self.eps = eps
        self.entr_base = entr_base

    def __call__(
        self,
        pred: NDArray[Any],
        gt: NDArray[Any],
        consis_mask: Optional[NDArray[Any]] = None,
    ) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("float32")
        # gt_converted = gt.cpu().numpy().astype("float32")
        if pred.ndim == 2:
            flag_2d = True
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)
            if consis_mask is not None:
                consis_mask = np.expand_dims(consis_mask, axis=0)
        else:
            flag_2d = False
        metric_result = np.zeros_like(pred)

        for i in range(len(pred)):
            probs_NA = np.clip(
                np.stack([1 - gt[i], gt[i]], axis=0), self.eps, 1 - self.eps
            )
            probs_A = np.clip(
                np.stack([1 - pred[i], pred[i]], axis=0),
                self.eps,
                1 - self.eps,
            )
            metric_result_pp = entropy(  # pyright: ignore[reportUnknownVariableType]
                probs_NA, probs_A, base=self.entr_base
            )
            assert is_ndarray(
                metric_result_pp
            ), f"Data is not a numpy array: {metric_result_pp}"
            # return torch.from_numpy(metric_result).to(pred.device).float()
            if consis_mask is not None:
                mask_inverted = np.logical_not(consis_mask[i])
                metric_result_pp[mask_inverted] = None
            metric_result[i] = metric_result_pp
        if flag_2d:
            metric_result = np.squeeze(metric_result)

        return metric_result


class CrossEntropyEval:
    def __init__(self, eps: float = 1e-7, entr_base: float = 2.0):
        super().__init__()
        self.eps = eps
        self.entr_base = entr_base

    def __call__(
        self,
        pred: NDArray[Any],
        gt: NDArray[Any],
        consis_mask: Optional[NDArray[Any]] = None,
    ) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("float32")
        # gt_converted = gt.cpu().numpy().astype("float32")
        if pred.ndim == 2:
            flag_2d = True
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)
            if consis_mask is not None:
                consis_mask = np.expand_dims(consis_mask, axis=0)
        else:
            flag_2d = False
        metric_result = np.zeros_like(pred)
        for i in range(len(pred)):
            probs_NA = np.clip(
                np.stack([1 - gt[i], gt[i]], axis=0), self.eps, 1 - self.eps
            )
            probs_A = np.clip(
                np.stack([1 - pred[i], pred[i]], axis=0),
                self.eps,
                1 - self.eps,
            )
            metric_result_pp = entropy(probs_NA, base=self.entr_base) + entropy(
                probs_NA, probs_A, base=self.entr_base
            )
            assert is_ndarray(
                metric_result_pp
            ), f"Data is not a numpy array: {metric_result_pp}"
            # return torch.from_numpy(metric_result).to(pred.device).float()
            if consis_mask is not None:
                mask_inverted = np.logical_not(consis_mask[i])
                metric_result_pp[mask_inverted] = None
            metric_result[i] = metric_result_pp
        if flag_2d:
            metric_result = np.squeeze(metric_result)
        return metric_result


class HammingDistanceEval:
    def __init__(self, threshold: float = 0.5, invert_threshold: bool = False):
        super().__init__()
        if invert_threshold:
            self.threshold = 1 - threshold
        else:
            self.threshold = threshold

    def __call__(
        self, pred: NDArray[Any], gt: NDArray[Any], mask: NDArray[Any]
    ) -> NDArray[Any]:
        # pred_converted = pred.cpu().numpy().astype("uint16")
        # gt_converted = gt.cpu().numpy().astype("uint16")
        # mask_converted = mask.cpu().numpy().astype("bool")
        if pred.ndim == 2:
            if np.sum(mask) == 0:
                metric_result = np.nan
            else:
                pred_th = pred[mask] > self.threshold
                gt_th = gt[mask] > self.threshold
                metric_result = hamming(pred_th, gt_th)

        else:
            metric_result = np.zeros(len(pred))
            for i in range(len(pred)):
                # if mask empty set to None
                if np.sum(mask[i]) == 0:
                    metric_result[i] = np.nan
                else:
                    pred_th = pred[i][mask[i]] > self.threshold
                    gt_th = gt[i][mask[i]] > self.threshold
                    metric_result[i] = hamming(pred_th, gt_th)
        # Convert to numpy array to ensure consistent return type
        if not isinstance(metric_result, np.ndarray):
            metric_result = np.array(metric_result)
        assert is_ndarray(metric_result), f"Data is not a numpy array: {metric_result}"
        # return torch.from_numpy(metric_result).to(pred.device).float()
        return metric_result


def adaRandError_eval(
    pred: NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]],
    gt: NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]],
    incomplete_gt: bool,
    num_dilations: Optional[int] = 1,
    num_erosions: Optional[int] = 1,
    bckg: bool = False,
    # border_params: Optional[Dict[str, int]] = {"num_dilations": 1, "num_erosions": 1},
) -> Tuple[NDArray[Any], NDArray[Any]]:
    # check that either both or neither num_dilations and num_erosions are provided
    assert (num_dilations is not None and num_erosions is not None) or (
        num_dilations is None and num_erosions is None
    ), "Either both num_dilations and num_erosions must be provided or neither"
    if pred.ndim == 2:
        pred = np.expand_dims(pred, axis=0)
        gt = np.expand_dims(gt, axis=0)
    # check that pred and gt have the same shape
    assert (
        pred.shape == gt.shape
    ), f"pred and gt have different shapes: {pred.shape} {gt.shape}"
    batch_scores = np.zeros((pred.shape[0], 3), dtype=np.float32)
    consis_mask = np.zeros_like(pred)
    if bckg == True:
        assert (
            num_dilations is None and num_erosions is None
        ), "num_dilations and num_erosions must be None when bckg is True"
    for j in range(len(pred)):
        if (gt[j].sum() == 0) and (bckg == False):
            # Prevent warning from empty GT patches
            are = float("nan")
            prec = float("nan")
            rec = float("nan")
            consis_mask[j] = get_mask(gt[j], pred[j], 0)

        else:
            if incomplete_gt:
                mask = get_mask_incomplete_gt(gt[j], pred[j])
            else:
                if bckg == True:
                    mask = get_mask(
                        -gt[j].astype(np.int16), -pred[j].astype(np.int16), -1
                    )
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

                if bckg == True:
                    gt_img = gt[j][mask]
                    pred_img = pred[j][mask]
                    assert is_ndarray(gt_img), f"Data is not a numpy array: {gt_img}"
                    assert is_ndarray(
                        pred_img
                    ), f"Data is not a numpy array: {pred_img}"
                    are, prec, rec = (  # pyright: ignore[reportUnknownVariableType]
                        adapted_rand_error(
                            assign_unique_ids_to_value(
                                gt_img, list(np.unique(gt_img)[1:])
                            ),
                            assign_unique_ids_to_value(
                                pred_img, list(np.unique(pred_img)[1:])
                            ),
                            ignore_labels=None,
                        )
                    )
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
    pred_none: NDArray[Any],
    pred_aug: NDArray[Any],
    threshold: float = 0.5,
) -> NDArray[Any]:
    masks = np.zeros((2, *pred_none.shape))

    masks[0] = pred_none > threshold
    masks[1] = pred_aug > threshold

    # Combine masks across augmentations (Union)
    combined_mask = np.logical_or.reduce(masks, axis=0)
    assert is_ndarray(combined_mask), f"Data is not a numpy array: {combined_mask}"
    return combined_mask


def get_segmentation_mask(
    pred_none: NDArray[Any],
    pred_aug: NDArray[Any],
    id: int,
) -> NDArray[Any]:
    masks = np.zeros((2, *pred_none.shape))

    masks[0] = pred_none == id
    masks[1] = pred_aug == id

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


def assign_unique_ids_to_value(data: NDArray[Any], value: List[int] = [0]):
    data = data.copy()
    for v in value:
        max_val = np.max(data)
        max_id_assigned = max_val + np.sum(data == v) + 1
        # check for overflow error
        data = avoid_int_overflow(data, max_id_assigned)
        data[data == v] = np.arange(max_val + 1, max_id_assigned)

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


class DiceMetric(nn.Module):
    """Metric computed based on the dice error between a binary input and binary target.

    Args:
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
    """

    def __init__(
        self,
        channelwise: bool = True,
        eps: float = 1e-7,
        reduce_channel: Optional[str] = "sum",
        threshold: Optional[float] = None,
        final_activation: Optional[Literal["sigmoid", "softmax"]] = None,
    ):
        if reduce_channel not in ("sum", "mean", "max", "min", None):
            raise ValueError(f"Unsupported channel reduction {reduce_channel}")

        super().__init__()
        self.channelwise = channelwise
        self.eps = eps
        self.reduce_channel = reduce_channel
        self.threshold = threshold
        self.final_activation = final_activation

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {
            "channelwise": channelwise,
            "eps": self.eps,
            "reduce_channel": self.reduce_channel,
        }

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_: The input, logits, probabilities or binary.
            target: The target, logits, probablities or binary.

        Returns:
            The dice score.
        """
        if self.final_activation is not None:
            if self.final_activation == "sigmoid":
                input_ = nn.functional.sigmoid(input_)
            elif self.final_activation == "softmax":
                input_ = nn.functional.softmax(input_, dim=1)
            else:
                raise ValueError(
                    f"Unsupported final activation {self.final_activation}"
                )

        if self.threshold is not None:
            input_ = ensure_binary(input_, threshold=self.threshold)
            target = ensure_binary(target, threshold=self.threshold)

        return dice_score(
            input_=input_,
            target=target,
            invert=False,
            channelwise=self.channelwise,
            eps=self.eps,
            reduce_channel=self.reduce_channel,
        )


def ensure_binary(
    input: torch.Tensor, threshold: Optional[float] = None
) -> torch.Tensor:
    # Check if input is in [0, 1]
    if input.min() < 0 or input.max() > 1:
        raise ValueError(
            f"Input tensor values must be in [0, 1], but got min={input.min().item()}, max={input.max().item()}"
        )
    unique_vals = torch.unique(input)  # pyright: ignore[reportUnknownVariableType]
    assert is_torch_tensor(unique_vals), f"Data is not a torch tensor: {unique_vals}"
    if not torch.all((unique_vals == 0) | (unique_vals == 1)):
        assert threshold is not None, "Input must be binary or threshold must be set."
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")
        input = (input > threshold).float()
    return input


def jaccard_index(prediction: NDArray[Any], target: NDArray[Any]) -> float:
    """
    Computes IoU for a given target and prediction numpy arrays
    """
    intersection = np.sum(prediction & target)
    union = np.sum(prediction | target)
    return intersection / np.maximum(union, 1e-8)


def per_class_iou_consistency(
    prediction: NDArray[Any], target: NDArray[Any]
) -> Dict[int, float]:
    classes = np.unique(target)
    iou_scores: Dict[int, float] = {}
    for cls in tqdm(classes):
        mask_pred = prediction == cls
        mask_target = target == cls
        iou = jaccard_index(mask_pred, mask_target)
        iou_scores[int(cls)] = iou
    return iou_scores


def per_class_NHD_consistency(
    prediction: NDArray[Any], target: NDArray[Any]
) -> Dict[Any, float]:
    classes = np.unique(target)
    nhd_scores: Dict[int, float] = {}
    for cls in tqdm(classes):
        mask = get_segmentation_mask(target, prediction, id=cls)
        HD_score = hamming(target[mask], prediction[mask])
        nhd_scores[cls] = 1 - HD_score  # pyright: ignore[reportArgumentType]
    return nhd_scores


def calculate_instance_rand_indices(
    arr_a: NDArray[Any],
    arr_b: NDArray[Any],
    instance_ids: Optional[Sequence[int]] = None,
):
    """
    Ultra-fast implementation using the mathematical formulation of Rand Index.

    For a given instance, we can calculate the Rand Index using:
    RI = (TP + TN) / (TP + TN + FP + FN)

    Where for pairs involving the target instance:
    - TP: pairs that agree in both segmentations
    - TN: pairs that disagree in both segmentations
    - FP + FN: pairs that disagree between segmentations
    """
    flat_a = arr_a.flatten()
    flat_b = arr_b.flatten()

    if instance_ids is None:
        instance_ids = np.unique(flat_a)  # pyright: ignore[reportAssignmentType]

    results: Dict[int, float] = {}

    assert instance_ids is not None, "instance_ids should not be None here"

    for instance_id in instance_ids:
        instance_mask = flat_a == instance_id
        n_instance = np.sum(instance_mask)

        if n_instance < 2:
            results[instance_id] = float("nan")
            continue

        # Get the labels in arr_b for instance pixels
        instance_labels_b = flat_b[instance_mask]

        # Calculate pairs within instance that agree in arr_b
        unique_labels, counts = np.unique(instance_labels_b, return_counts=True)
        within_agreements = np.sum(counts * (counts - 1)) // 2
        total_within_pairs = n_instance * (n_instance - 1) // 2

        # For cross-instance pairs: instance pixels vs non-instance pixels
        non_instance_mask = ~instance_mask
        n_non_instance = np.sum(non_instance_mask)

        if n_non_instance == 0:
            # Only within-instance pairs exist
            rand_index = (
                within_agreements / total_within_pairs
                if total_within_pairs > 0
                else float("nan")
            )
        else:
            non_instance_labels_b = flat_b[non_instance_mask]

            # Count cross-agreements (instance pixel has same label as non-instance pixel in arr_b)
            cross_agreements = 0
            for label in unique_labels:
                n_instance_with_label = np.sum(instance_labels_b == label)
                n_non_instance_with_label = np.sum(non_instance_labels_b == label)
                cross_agreements += n_instance_with_label * n_non_instance_with_label

            total_cross_pairs = n_instance * n_non_instance
            cross_disagreements = total_cross_pairs - cross_agreements

            # Rand Index calculation
            consistent_pairs = within_agreements + cross_disagreements
            total_pairs = total_within_pairs + total_cross_pairs
            rand_index = (
                consistent_pairs / total_pairs if total_pairs > 0 else float("nan")
            )

        results[instance_id] = rand_index

    return results


def per_object_adaRandError_eval(
    pred_unperturbed: NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]],
    pred_perturbed: NDArray[Union[np.uint8, np.uint16, np.uint32, np.uint64]],
    num_dilations: Optional[int] = 1,
    num_erosions: Optional[int] = 1,
) -> Tuple[
    Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]]
]:
    """
    Calculate per-object foreground restricted Adapted Rand Error between
    unperturbed and perturbed instance segmentation predictions.

    For each instance in both images, calculates the consistency by treating
    that instance as the "ground truth" and the corresponding region in the
    other image as the "prediction".

    Args:
        pred_unperturbed: Unperturbed instance segmentation prediction
        pred_perturbed: Perturbed instance segmentation prediction
        num_dilations: Number of dilations for border mask (optional)
        num_erosions: Number of erosions for border mask (optional)

    Returns:
        Tuple of two dictionaries:
        - First dict: Per-object scores using unperturbed instances as reference
        - Second dict: Per-object scores using perturbed instances as reference
        Each dict maps instance_id -> (are, precision, recall)
    """
    assert (
        pred_unperturbed.shape == pred_perturbed.shape
    ), f"Predictions have different shapes: {pred_unperturbed.shape} vs {pred_perturbed.shape}"

    # Handle 2D case
    unperturbed_scores = {}
    perturbed_scores = {}

    # Get unique instance IDs (excluding background=0)
    unp_instances = np.unique(pred_unperturbed)
    unp_instances = unp_instances[unp_instances != 0]

    pert_instances = np.unique(pred_perturbed)
    pert_instances = pert_instances[pert_instances != 0]

    # Calculate scores using unperturbed instances as reference
    for instance_id in unp_instances:
        are, prec, rec = _calculate_per_object_consistency(
            pred_unperturbed, pred_perturbed, instance_id, num_dilations, num_erosions
        )
        unperturbed_scores[instance_id] = (are, prec, rec)

    # Calculate scores using perturbed instances as reference
    for instance_id in pert_instances:
        are, prec, rec = _calculate_per_object_consistency(
            pred_perturbed, pred_unperturbed, instance_id, num_dilations, num_erosions
        )
        perturbed_scores[instance_id] = (are, prec, rec)

    return unperturbed_scores, perturbed_scores  # pyright: ignore


def _calculate_per_object_consistency(
    reference_img: NDArray[Any],
    comparison_img: NDArray[Any],
    instance_id: int,
    num_dilations: Optional[int] = 1,
    num_erosions: Optional[int] = 1,
) -> Tuple[float, float, float]:
    """
    Helper function to calculate consistency for a single object instance.

    Args:
        reference_img: Image containing the reference instance
        comparison_img: Image to compare against
        instance_id: ID of the instance to analyze
        num_dilations: Number of dilations for border mask
        num_erosions: Number of erosions for border mask

    Returns:
        Tuple of (adapted_rand_error, precision, recall)
    """
    # Create mask for the specific instance
    instance_mask = reference_img == instance_id

    if not np.any(instance_mask):
        return float("nan"), float("nan"), float("nan")

    # Create foreground mask (union of instance regions in both images)
    # foreground_mask = get_mask(reference_img, comparison_img, threshold=0)

    # Combine instance mask with foreground mask
    # eval_mask = np.logical_and(instance_mask, foreground_mask)

    # Apply border mask if specified
    if num_dilations is not None and num_erosions is not None:
        border_mask = get_border_mask(
            img=reference_img, num_dilations=num_dilations, num_erosions=num_erosions
        )
        instance_mask = np.logical_and(instance_mask, ~border_mask)

    if not np.any(instance_mask):
        return float("nan"), float("nan"), float("nan")

    # Extract regions within evaluation mask
    ref_region = reference_img[instance_mask]
    comp_region = comparison_img[instance_mask]

    # For foreground restricted evaluation, assign unique IDs to background pixels
    # in the comparison region that fall within our evaluation mask
    ref_region_processed = assign_unique_ids_to_value(ref_region, value=[0])
    comp_region_processed = assign_unique_ids_to_value(comp_region, value=[0])

    try:
        are, prec, rec = adapted_rand_error(  # pyright: ignore
            ref_region_processed,
            comp_region_processed,
            ignore_labels=None,
        )

        # Ensure return types are floats
        return float(are), float(prec), float(rec)  # pyright: ignore

    except Exception as e:
        print(f"Error calculating adapted rand error for instance {instance_id}: {e}")
        return float("nan"), float("nan"), float("nan")


def adapted_rand_error_object(
    true_seg: NDArray[Any], pred_seg: NDArray[Any], object_id: int, alpha: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate adapted Rand error for a single object instance, considering only
    pairs of pixels within the object mask in the ground truth.

    Parameters
    ----------
    true_seg : ndarray
        Ground truth segmentation. Shape (H, W) or (N,).
    pred_seg : ndarray
        Predicted segmentation, same shape as true_seg.
    object_id : int
        The label of the object in the ground truth to evaluate.
    alpha : float
        Weight between precision and recall in the F-score (default 0.5).

    Returns
    -------
    are : float
        The adapted Rand error for the object.
    precision : float
        Precision score for the object.
    recall : float
        Recall score for the object.
    """

    # Mask for the object in the ground truth
    mask = true_seg == object_id
    if np.sum(mask) < 2:
        # Not enough pixels for evaluation
        return np.nan, np.nan, np.nan

    # Extract pixels within the object mask
    true_labels = true_seg[mask]
    pred_labels = pred_seg[mask]

    # Relabel for sequential labels (start from 1, avoid 0 confusion)
    true_unique, true_labels = np.unique(  # pyright: ignore
        true_labels, return_inverse=True
    )
    pred_unique, pred_labels = np.unique(  # pyright: ignore
        pred_labels, return_inverse=True
    )

    n = len(true_labels)
    if n < 2:
        return np.nan, np.nan, np.nan

    # Build contingency table
    # Count occurrences of (true_label, pred_label) pairs
    pair_counter = Counter(zip(true_labels, pred_labels))
    # Count occurrences for marginal sums
    true_counter = Counter(true_labels)
    pred_counter = Counter(pred_labels)

    # Number of pairs
    total_pairs = n * (n - 1) // 2  # pyright: ignore

    # Sums for precision and recall
    sum_nij = sum(v * (v - 1) // 2 for v in pair_counter.values())
    sum_ai = sum(v * (v - 1) // 2 for v in true_counter.values())
    sum_bj = sum(v * (v - 1) // 2 for v in pred_counter.values())

    # Precision and recall
    if sum_bj == 0 or sum_ai == 0:
        return np.nan, np.nan, np.nan

    precision = sum_nij / sum_bj
    recall = sum_nij / sum_ai

    # F-score
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    adapted_rand_error = 1 - fscore

    return adapted_rand_error, precision, recall


def foreground_restricted_AdaRandError_consistency(
    prediction: NDArray[Any],
    target: NDArray[Any],
    num_dilations: Optional[int] = None,
    num_erosions: Optional[int] = None,
) -> Tuple[float, float, float]:
    instance_mask = get_mask(target, prediction, threshold=0)

    if num_dilations is not None and num_erosions is not None:
        border_mask = get_border_mask(
            img=target, num_dilations=num_dilations, num_erosions=num_erosions
        )
        instance_mask = np.logical_and(instance_mask, ~border_mask)

    are, prec, rec = adapted_rand_error(  # pyright: ignore[reportUnknownVariableType]
        assign_unique_ids_to_value(target[instance_mask], value=[0]),
        assign_unique_ids_to_value(prediction[instance_mask], value=[0]),
    )
    assert isinstance(are, float), f"are is not a float: {are}"
    assert isinstance(prec, float), f"prec is not a float: {prec}"
    assert isinstance(rec, float), f"rec is not a float: {rec}"
    return are, prec, rec
