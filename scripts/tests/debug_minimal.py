import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

from model_ranking import TransferFeatureExtraction
from model_ranking import TransferFeatureExtractionConfig
from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.utils import load_checkpoint

print("Creating minimal test...")

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
        "num_samples": 10,  # Small number for debugging
        "output_dir_path": None,
    },
}

feature_ranking_cfg = TransferFeatureExtractionConfig.model_validate(config)
feature_ranking = TransferFeatureExtraction(config=feature_ranking_cfg)

# Setup model
model_cfg = feature_ranking.source_model_cfgs[0]
model_cfg = model_cfg.create_config(feature_perturbation=None)
model_cfg = model_cfg.model_copy(update={"feature_return": True})

source_model_path = "/g/kreshuk/talks/segmentation_ModelSelection/experiments/EPFL/BatchNorm/E_model5/best_checkpoint.pytorch"

model = UNet2D(**model_cfg.model_dump())
load_checkpoint(source_model_path, model, model_key="model_state_dict")
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.eval()

target_dataset = feature_ranking.target_datasets["EPFL"]

print("Testing just first image...")
try:
    # Process just one image to debug
    features2, labels2, indices2 = feature_ranking.extract_features_sampled_efficient(
        model,
        target="EPFL",
        target_dataset=target_dataset,
        batch_size=1,  # Single image batches
    )
    print("Success! Features extracted.")
    print(f"Feature shape: {features2['decoders.2'].shape}")
    print(f"Labels shape: {labels2['decoders.2'].shape}")
    print(f"Indices shape: {indices2['decoders.2'].shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
