from torch_em.segmentation import (
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from typing import Any, Dict, Literal


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


augmentation_type = Literal[
    "None", "HFlip", "VFlip", "Rot", "Aff", "Ctr", "Brt", "Gamma", "Gauss"
]


def classification_geometric_TTAs(
    config: AugmentationsConfig, transform: Dict[str, Any]
):
    TT_transform = get_augmentations(config.ndim, transform)
    return TT_transform
