import torch
from typing import List, Optional, Any
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import (
    erosion,  # pyright: ignore[reportUnknownVariableType]
    dilation,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.utils import is_ndarray, avoid_int_overflow, is_torch_tensor


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
