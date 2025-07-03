from typing import Dict, Any

from model_ranking.supervised_training import get_supervised_loader

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

supervised_loader = get_supervised_loader(
    data_supervised_cfg["supervised_train_paths"],
    data_supervised_cfg["raw_key"],
    data_supervised_cfg["label_key"],
    data_supervised_cfg["patch_shape"],
    data_supervised_cfg["batch_size"],
    "/g/kreshuk/talks/model_ranking/notebooks/checks",
    num_workers=data_supervised_cfg["num_workers"],
    n_samples=data_supervised_cfg["n_samples_train"],
    rois=None,
)

raw, label = next(iter(supervised_loader))
