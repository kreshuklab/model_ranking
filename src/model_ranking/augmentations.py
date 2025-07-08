from typing import Dict, Any


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
