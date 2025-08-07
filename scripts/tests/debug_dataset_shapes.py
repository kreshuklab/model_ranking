import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

from model_ranking.feature_ranking import TransferFeatureExtraction
from model_ranking.dataclass import TransferFeatureExtractionConfig

print("Creating feature ranking config...")

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

target_dataset = feature_ranking.target_datasets["EPFL"]

print(f"Dataset length: {len(target_dataset)}")

# Check the shape of each image in the dataset
print("Checking image shapes in dataset:")
for i, (image, label) in enumerate(target_dataset):
    print(f"Image {i}: image shape = {image.shape}, label shape = {label.shape}")
    if i >= 5:  # Just check first few
        break

print("Done!")
