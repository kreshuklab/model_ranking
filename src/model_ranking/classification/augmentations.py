import numpy as np
from torch_em.segmentation import (
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform.raw import (
    get_single_TTA_raw_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from typing import Any, Sequence


from .dataclass import AugmentationsConfig

AUGMENTATION_ABBREVIATIONS = {
    "None": "None",
    "RandomHorizontalFlip": "HFlip",
    "RandomVerticalFlip": "VFlip",
    "RandomRotation": "Rot",
    "RandomAffine": "Aff",
    "RandomContrast": "Ctr",
    "RandomBrightness": "Brt",
    "RandomGamma": "Gamma",
    "AdditiveGaussianNoise": "Gauss",
}


def classification_raw_TTAs(config: AugmentationsConfig, raw_transform: Sequence[Any]):
    assert config.n_samples is not None, "Number of samples must be specified"
    assert config.aug_rnd_seed is not None, "Random seed must be specified"
    TTA_alphas = get_TTA_aug_alphas(
        raw_transform, config.n_samples, config.aug_rnd_seed
    )
    TT_raw_transform = get_single_TTA_raw_augmentations(raw_transform)
    return TTA_alphas, TT_raw_transform


def classification_geometric_TTAs(
    config: AugmentationsConfig, transform: Sequence[Any]
):
    TT_transform = get_augmentations(config.ndim, transform)
    return TT_transform


def get_TTA_aug_alphas(aug: Sequence[Any], n_samples: int, random_seed: int):
    alpha = aug[1]["alpha"]
    r = np.random.RandomState(random_seed)
    alphas = r.uniform(alpha[0], alpha[1], n_samples)
    return alphas
