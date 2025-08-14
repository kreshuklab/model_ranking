from typing import Dict, Any, List
from numpy.typing import NDArray
import torch

from model_ranking.datasets import calculate_global_stats
from model_ranking import load_h5

# from model_ranking import get_supervised_loader
from model_ranking import get_DummySelfTraining_loader

"""
data_supervised_cfg: Dict[str, Any] = {
    "patch_shape": [1, 256, 256],
    "supervised_train_paths": ["/scratch/talks/data/EPFL/train.h5"],
    "supervised_val_paths": ["/scratch/talks/data/EPFL/val.h5"],
    "raw_key": "raw",
    "label_key": "labels",
    "batch_size": 2,
    "num_workers": 1,
    "n_samples_train": None,
    "n_samples_val": None,
    "roi_supervised_train": None,
    "roi_supervised_val": None,
}

data_cfg: Dict[str, Any] = {
    "unsupervised_train_paths": ["/scratch/talks/data/EPFL/train.h5"],
    "unsupervised_val_paths": ["/scratch/talks/data/EPFL/val.h5"],
    "patch_shape": [1, 256, 256],
    "supervised_train_paths": None,
    "supervised_val_paths": None,
    "raw_key": "raw",
    "raw_key_supervised": None,
    "label_key": "labels",
    "batch_size": 32,
    "num_workers": 1,
    "n_samples_train": 640,
    "n_samples_val": 512,
}


global_normalisation = True

if global_normalisation:
    raw_train: List[NDArray[Any]] = []
    for train_path in data_cfg["unsupervised_train_paths"]:
        raw_train.append(load_h5(train_path, data_cfg["raw_key"]))
    raw_stats = calculate_global_stats(raw_train)
else:
    raw_stats = None

loader = get_DummySelfTraining_loader(
    paths=data_cfg["unsupervised_train_paths"],
    raw_key=data_cfg["raw_key"],
    label_key=data_cfg["label_key"],
    patch_shape=data_cfg["patch_shape"],
    batch_size=data_cfg["batch_size"],
    num_workers=data_cfg["num_workers"],
    n_samples=data_cfg["n_samples_train"],
    global_stats=raw_stats,
    norm01=False,
)

"""
# supervised_loader = get_supervised_loader(
#     data_supervised_cfg["supervised_train_paths"],
#     data_supervised_cfg["raw_key"],
#     data_supervised_cfg["label_key"],
#     data_supervised_cfg["patch_shape"],
#     data_supervised_cfg["batch_size"],
#     "/g/kreshuk/talks/model_ranking/notebooks/checks",
#     num_workers=data_supervised_cfg["num_workers"],
#     n_samples=data_supervised_cfg["n_samples_train"],
#     rois=None,
# )

# raw, label = next(iter(supervised_loader))

# raw1_label, raw2 = next(iter(loader))
# print(f"Raw1_label shape: {raw1_label.shape}, raw2 shape: {raw2.shape}")

from model_ranking.datasets import get_loaders
from model_ranking import (
    # EPFLTargetConfig,
    # HmitoTargetConfig,
    VNCTargetConfig,
    # OvulesTargetConfig,
)

# target_config = OvulesTargetConfig()
# target_config = EPFLTargetConfig()
# target_config = HmitoTargetConfig()
target_config = VNCTargetConfig()

with torch.no_grad():
    for loader in get_loaders(
        target_config,
        phase="train",
        output_path=None,
        shuffle=False,
    ):
        raw, label = next(iter(loader))
