from typing import Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
import torch
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]

from torch_em.transform.raw import (
    normalize,  # pyright: ignore[reportUnknownVariableType]
    GaussianBlur,
    AdditiveGaussianNoise,
    ToTensorDtype,
)
from torch_em.transform import (
    get_raw_transform,  # pyright: ignore[reportUnknownVariableType]
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
    norm01: Optional[bool] = None,
) -> Union[NDArray[Any], torch.Tensor]:
    """Normalize the input data using min/max normalisation, if norm01 is True clip to range [0, 1]
    else [-1,1] or no clipping if norm01 is None.

    Args:
        raw: The input data.
        minval: The minimum data value. If None, it will be computed from the data.
        maxval: The maximum data value. If None, it will be computed from the data.
        axis: The axis along which to compute the min and max value.
        eps: The epsilon value for numerical stability.
        norm01: If True, normalizes to [0, 1]. If False, normalizes to [-1, 1].

    Returns:
        The normalized input data.
    """
    normalised_raw: Union[NDArray[Any], torch.Tensor] = normalize(  # pyright: ignore
        raw,
        minval=minval,
        maxval=maxval,
        axis=axis,
        eps=eps,
    )
    if norm01 is None:
        return normalised_raw
    else:
        if norm01 == True:
            return clip_array(normalised_raw, 0.0, 1.0)
        else:
            return clip_array(2 * normalised_raw - 1, -1.0, 1.0)


def clip_array(x: Union[NDArray[Any], torch.Tensor], min_val: float, max_val: float):
    if isinstance(x, np.ndarray):
        return np.clip(x, min_val, max_val)
    elif isinstance(x, torch.Tensor):
        return torch.clamp(x, min=min_val, max=max_val)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def identity(
    raw: Union[torch.Tensor, NDArray[Any]],
) -> Union[torch.Tensor, NDArray[Any]]:
    """Identity function for empty callabel to replace normalisation when not required."""
    return raw


def weak_augmentations(  # pyright: ignore[reportUnknownParameterType]
    p: float = 0.75,
    norm: Callable[[Any], torch.Tensor | NDArray[Any]] = identity,
    dtype: torch.dtype = torch.float32,
):
    assert isinstance(norm, Callable)
    aug = transforms.Compose(
        [
            transforms.RandomApply([GaussianBlur(sigma=(0, 2.5))], p=p),
            transforms.RandomApply(
                [
                    AdditiveGaussianNoise(
                        scale=(0, 0.15),
                        clip_kwargs=False,  # pyright: ignore[reportArgumentType]
                    )
                ],
                p=p,
            ),
            ToTensorDtype(dtype=dtype),  # pyright: ignore[reportUnknownVariableType
        ]
    )
    return get_raw_transform(
        normalizer=norm, augmentation1=aug
    )  # pyright: ignore[reportUnknownVariableType]
