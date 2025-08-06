import numpy as np
import torch

# from tqdm import tqdm
from typing import Dict, Any

from model_ranking.feature_ranking import FeatureBasedTransferRanking
from model_ranking.dataclass import FeatureBasedTransferRankingConfig

from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

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
        "output_dir_path": "/g/kreshuk/talks/model_ranking/notebooks/checks",
        # "output_dir_path": None,  # Set to None for testing
    },
}


feature_ranking_cfg = FeatureBasedTransferRankingConfig.model_validate(config)

feature_ranking = FeatureBasedTransferRanking(
    config=feature_ranking_cfg,
)

model_cfg = feature_ranking.source_model_cfgs[0]
model_cfg = model_cfg.create_config(feature_perturbation=None)
model_cfg = model_cfg.model_copy(update={"feature_return": True})

source_model_path = "/g/kreshuk/talks/segmentation_ModelSelection/experiments/EPFL/BatchNorm/E_model5/best_checkpoint.pytorch"

model = UNet2D(**model_cfg.model_dump())
load_checkpoint(source_model_path, model, model_key="model_state_dict")
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.eval()


target_dataset = feature_ranking.target_datasets["EPFL"]
target_dataloader = feature_ranking.target_dataloaders["EPFL"]

indices_loaded = feature_ranking.feature_indices["EPFL"]
assert indices_loaded is not None, "Feature indices for target dataset must be defined."
features, labels, indices = feature_ranking.extract_features_sampled_batched(
    model,
    target="EPFL",
    target_dataloader=target_dataloader,  # Use the dataloader directly
)

# feature_ranking.run_transfer_ranking_batched()

for layer, feature in features.items():
    print(f"Layer: {layer}, Feature shape: {feature.shape}")
    print(f"Labels shape: {labels[layer].shape}")
    print(f"Indices shape: {indices[layer].shape}")

precomputed_path = "/g/kreshuk/talks/model_ranking/notebooks/checks/EPFL_to_EPFL/E_model5_to_EPFL_features.npz"

data = np.load(precomputed_path)
print(f"Precomputed features shape: {data['decoders.2_features'].shape}")
print(f"Precomputed labels shape: {data['decoders.2_labels'].shape}")
print(f"Precomputed indices shape: {data['decoders.2_indices'].shape}")

# Compare with precomputed features
# for layer in features:
for layer in indices:
    print(f"Comparing layer: {layer}")
    np.testing.assert_array_equal(features[layer], data[f"{layer}_features"])
    np.testing.assert_array_equal(labels[layer], data[f"{layer}_labels"])
    np.testing.assert_array_equal(indices[layer], data[f"{layer}_indices"])
    print(f"✓ {layer} features match precomputed values.")
