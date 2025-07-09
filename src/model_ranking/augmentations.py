from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray
import torch

from torch_em.transform.raw import (
    normalize,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.augment.transforms import Transformer

DEFAULT_TRANSFORMS: Dict[str, Any] = {
    "raw": [
        {"name": "RandomFlip"},
        {"name": "RandomRotate90"},
        {
            "name": "RandomRotate",
            "axes": [[2, 1]],
            "angle_spectrum": 45,
            "mode": "reflect",
        },
        {"name": "ElasticDeformation", "spline_order": 3, "execution_probability": 0.2},
    ],
}

DEFAULT_TRAIN_AUGMENTATIONS: Dict[str, Any] = {
    "raw": [
        {"name": "Normalize"},
        {
            "name": "RandomContrast",
            "execution_probability": 0.3,
            "alpha": [0.5, 2],
            "clip_kwargs": None,
        },
        {
            "name": "RandomBrightness",
            "execution_probability": 0.3,
            "alpha": [0.0, 0.3],
            "clip_kwargs": None,
        },
        {
            "name": "RandomGamma",
            "execution_probability": 0.3,
            "gamma": [0.2, 2],
            "clip_kwargs": None,
        },
        {
            "name": "AdditiveGaussianNoise",
            "execution_probability": 0.3,
            "scale": [0.0, 0.3],
        },
        # {"name": "ToTensor", "expand_dims": True},
    ],
    "label": [
        # {"name": "ToTensor", "expand_dims": True},
    ],
}

DEFAULT_VAL_AUGMENTATIONS: Dict[str, Any] = {
    "raw": [{"name": "Normalize"}, {"name": "ToTensor", "expand_dims": True}],
    "label": [{"name": "ToTensor", "expand_dims": True}],
}


def get_default_augmentations(config: Dict[str, Any], stats: Dict[str, Any]):
    """Get the default augmentations based on the provided configuration and statistics.

    Args:
        config (Dict[str, Any]): transform configuration
        stats (Dict[str, Any]): global statistics for the dataset, e.g., mean and std

    Returns:
        _type_: _description_
    """
    transformer = Transformer(config, stats)
    raw_transform = transformer.raw_transform()
    label_transform = transformer.label_transform()
    return raw_transform, label_transform


def normalize_specify_range(
    raw: Union[torch.Tensor, NDArray[Any]],
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-7,
    norm01: bool = False,
) -> Union[NDArray[Any], torch.Tensor]:
    """Normalize the input data so that it is in range [0, 1].

    Args:
        raw: The input data.
        minval: The minimum data value. If None, it will be computed from the data.
        maxval: The maximum data value. If None, it will be computed from the data.
        axis: The axis along which to compute the min and max value.
        eps: The epsilon value for numerical stability.

    Returns:
        The normalized input data.
    """
    normalised_raw = normalize(
        raw,
        minval=minval,
        maxval=maxval,
        axis=axis,
        eps=eps,
    )
    if norm01:
        return (normalised_raw,)
    else:
        return 2 * normalised_raw - 1
