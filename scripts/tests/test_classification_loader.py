from typing import Dict, Any, List
import numpy as np
from numpy.typing import NDArray
import torch
from model_ranking import (
    get_classification_dataloader,
    get_classification_TTA_loaders,
    ResNet,
    load_checkpoint_resnet,
    ClassificationModelConfig,
    ClassificationLoaderConfig,
)

dataloader_config: Dict[str, Any] = {
    "patch_position": {
        "patch_pos_path": (
            "/g/kreshuk/talks/data/epfl/classification_patches/epfl_resized_01_80x80.h5"
        ),
        "patch_pos_key": "validation",
        "roi": None,
        "rnd_seed": None,
        "n_unique_patches": None,
    },
    "aug_config": {
        "aug_rnd_seed": None,
        "transform_params": None,
        "raw_transform_params": None,
        "ndim": 2,
        "n_samples": None,
    },
    "dataset": {
        "raw_path": "/scratch/talks/data/EPFL/val.h5",
        "raw_key": "raw",
        "mask_path": "/scratch/talks/data/EPFL/val.h5",
        "mask_key": "labels",
        "patch_shape": (1, 80, 80),
        "repeat_patches": False,
        "sample_patches": True,
        "patch_rnd_seed": 1,
        "mask_return_mode": False,
        "patch_return_mode": False,
    },
    "n_samples": 3,
    "ndim": 2,
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 1,
}
dataloader_cfg = ClassificationLoaderConfig.model_validate(dataloader_config)

dataloader = get_classification_dataloader(dataloader_cfg)

raw_data: List[NDArray[Any]] = []
for i, (raw, label) in enumerate(dataloader):
    print(i, raw.shape, label.shape)
    raw_data.append(raw.numpy())
    if i > 3:
        break

print(np.all(np.equal(raw_data[0], raw_data[1])))
