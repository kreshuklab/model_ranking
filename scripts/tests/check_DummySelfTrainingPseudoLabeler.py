from typing import Dict, Any
import torch
from torch_em.segmentation import (  # pyright: ignore[reportMissingTypeStubs]
    get_data_loader,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.self_training import weak_augmentations
from model_ranking.pseudo_labeling import DummyDirectEvalPseudoLabeler
from model_ranking.datasets import DummySelfTrainingDataset

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

data_cfg: Dict[str, Any] = {
    "unsupervised_train_paths": "/scratch/talks/data/EPFL/train.h5",
    "unsupervised_val_paths": "/scratch/talks/data/EPFL/val.h5",
    "patch_shape": [1, 256, 256],
    "supervised_train_paths": None,
    "supervised_val_paths": None,
    "raw_key": "raw",
    "raw_key_supervised": None,
    "label_key": "labels",
    "batch_size": 32,
    "num_workers": 32,
    "n_samples_train": None,
    "n_samples_val": None,
}

model_cfg: Dict[str, Any] = {
    "name": "UNet2D",
    "in_channels": 1,
    "out_channels": 1,
    "layer_order": "bcr",
    "f_maps": 32,
    "final_sigmoid": True,
    "feature_return": False,
    "is_segmentation": True,
    "feature_perturbation": None,
}

augmentations = (  # pyright: ignore[reportUnknownVariableType]
    weak_augmentations(),
    weak_augmentations(),
)

dataset = DummySelfTrainingDataset(
    raw_path=data_cfg["unsupervised_train_paths"],
    raw_key=data_cfg["raw_key"],
    label_path=data_cfg["unsupervised_train_paths"],
    label_key=data_cfg["label_key"],
    patch_shape=data_cfg["patch_shape"],
    n_samples=20,
    augmentations=augmentations,  # pyright: ignore[reportUnknownArgumentType]
)

loader = get_data_loader(  # pyright: ignore[reportUnknownVariableType]
    dataset,
    batch_size=data_cfg["batch_size"],
    num_workers=data_cfg["num_workers"],
    shuffle=True,
)

pseudo_labeler = DummyDirectEvalPseudoLabeler(
    score_threshold=0.8,
)

model_path = "/g/kreshuk/talks/segmentation_ModelSelection/experiments/Hmito/BatchNorm/Hm_model3/best_checkpoint.pytorch"
model = get_model(model_cfg)
load_checkpoint(model_path, model)
model = model.cuda()
model = model.eval()

with torch.no_grad():
    img, _ = next(iter(loader))  # type: ignore[reportUnknownVariableType]
    print(img.shape)
    img = torch.squeeze(img, dim=-3)
    img = img.cuda()
    pseudo_labels, label_mask = pseudo_labeler(model, img)
    print(f"label shape: {pseudo_labels.shape}")
    if label_mask is not None:
        print(f"label mask shape: {label_mask.shape}")
