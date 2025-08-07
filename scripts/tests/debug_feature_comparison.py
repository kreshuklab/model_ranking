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

from CCFV.utils.sliding_window_sampling import FeatureExtractor

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

print("Testing with first image...")
image, label = next(iter(target_dataset))
print(f"Image shape: {image.shape}")

# Method 1: Direct model call
image_gpu = image.to("cuda:0")
# Note: Don't squeeze here - keep original shape like in the test script
pred, feats = model(image_gpu)
print(f"Direct model call - feats[-1] shape: {feats[-1].shape}")

# Method 2: FeatureExtractor
# Use the same shape as direct method
feature_extractor = FeatureExtractor(model, layers=["decoders.2"])
with torch.no_grad():
    extracted_features = feature_extractor(image_gpu)  # Same shape as direct method
    print(
        f"FeatureExtractor - decoders.2 shape: {extracted_features['decoders.2'].shape}"
    )

feature_extractor.remove_handler()

# Compare if they're the same
direct_features = feats[-1].detach().cpu().numpy()
extracted_features_np = extracted_features["decoders.2"].detach().cpu().numpy()

print(f"Are features identical? {np.allclose(direct_features, extracted_features_np)}")
print(f"Max difference: {np.max(np.abs(direct_features - extracted_features_np))}")

print("Test completed!")
