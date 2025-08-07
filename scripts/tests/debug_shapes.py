import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

from model_ranking.feature_ranking import TransferFeatureExtraction
from model_ranking.dataclass import TransferFeatureExtractionConfig

from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

print("Setting up config...")

config: Dict[str, Any] = {
    "target_datasets": [{"name": "EPFL"}],
    "source_models": [
        {"source_name": "EPFL", "model_name": "E_model5", "model_type": "UNet2D"}
    ],
    "source_model_base_path": (
        "/g/kreshuk/talks/segmentation_ModelSelection/experiments"
    ),
    "data_base_path": "/scratch/talks/data",
    "feature_cfg": {
        "layers": ["decoders.2"],
        "sampling_seed": 42,
        "num_samples": 1000,
        "output_dir_path": None,
    },
}

feature_ranking_cfg = TransferFeatureExtractionConfig.model_validate(config)
feature_ranking = TransferFeatureExtraction(config=feature_ranking_cfg)

print("Setting up model...")

model_cfg = feature_ranking.source_model_cfgs[0]
model_cfg = model_cfg.create_config(feature_perturbation=None)
model_cfg = model_cfg.model_copy(update={"feature_return": True})

source_model_path = "/g/kreshuk/talks/segmentation_ModelSelection/experiments/EPFL/BatchNorm/E_model5/best_checkpoint.pytorch"

model = UNet2D(**model_cfg.model_dump())
load_checkpoint(source_model_path, model, model_key="model_state_dict")
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.eval()

target_dataset = feature_ranking.target_datasets["EPFL"]

print(f"Dataset length: {len(target_dataset)}")

# Test with a single image first
print("Testing with first image...")
image, label = next(iter(target_dataset))
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")

# Test feature extraction
image_gpu = image.to("cuda:0")
pred, feats = model(image_gpu)
print(f"Last layer features shape: {feats[-1].shape}")

# Check the shape transformation
last_layer_feats = feats[-1].cpu().detach().numpy()
print(f"Feature shape before flatten: {last_layer_feats.shape}")
feat_flat = np.reshape(last_layer_feats, [-1, last_layer_feats.shape[1]])
print(f"Feature shape after flatten: {feat_flat.shape}")

print("Test completed!")
