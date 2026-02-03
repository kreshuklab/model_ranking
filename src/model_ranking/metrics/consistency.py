import torch
from typing import Dict, Optional, Any, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
    entropy,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.spatial.distance import hamming
from skimage.metrics import (
    adapted_rand_error,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from ._utils import (
    get_mask,
    get_mask_incomplete_gt,
    get_segmentation_mask,
    get_border_mask,
    assign_unique_ids_to_value,
)

from model_ranking.utils import is_ndarray, avoid_int_overflow


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
