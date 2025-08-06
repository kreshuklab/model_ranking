import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

from model_ranking.feature_ranking import FeatureBasedTransferRanking
from model_ranking.dataclass import FeatureBasedTransferRankingConfig

from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.utils import load_checkpoint

print("Creating detailed comparison test...")

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

feature_ranking_cfg = FeatureBasedTransferRankingConfig.model_validate(config)
feature_ranking = FeatureBasedTransferRanking(config=feature_ranking_cfg)

model_cfg = feature_ranking.source_model_cfgs[0]
model_cfg = model_cfg.create_config(feature_perturbation=None)
model_cfg = model_cfg.model_copy(update={"feature_return": True})

source_model_path = "/g/kreshuk/talks/segmentation_ModelSelection/experiments/EPFL/BatchNorm/E_model5/best_checkpoint.pytorch"

model = UNet2D(**model_cfg.model_dump())
load_checkpoint(source_model_path, model, model_key="model_state_dict")
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.eval()

target_dataset = feature_ranking.target_datasets["EPFL"]

print("Extracting features with efficient method...")
features2, labels2, indices2 = feature_ranking.extract_features_sampled_efficient(
    model,
    target="EPFL",
    target_dataset=target_dataset,
    batch_size=4,  # Same as original script
)

print("Extracting features with direct method...")
with torch.no_grad():
    features_per_image = np.zeros((len(target_dataset), 1000, 32))
    for i, (image, label) in enumerate(tqdm(target_dataset)):
        image = image.to("cuda:0")
        pred, feats = model(image)

        last_layer_feats = feats[-1].cpu().detach().numpy()
        feat_flat = np.reshape(last_layer_feats, [-1, last_layer_feats.shape[1]])
        features_per_image[i] = feat_flat[indices2["decoders.2"][i]]

print("Comparing results...")
efficient_features = features2["decoders.2"]
direct_features = features_per_image

print(f"Efficient features shape: {efficient_features.shape}")
print(f"Direct features shape: {direct_features.shape}")
print(f"Features match: {np.allclose(efficient_features, direct_features, atol=1e-6)}")

# More detailed comparison
diff = np.abs(efficient_features - direct_features)
print(f"Max absolute difference: {np.max(diff)}")
print(f"Mean absolute difference: {np.mean(diff)}")
print(f"Number of non-matching elements: {np.sum(diff > 1e-6)}")

# Check if the issue is with specific images
for i in range(min(3, len(target_dataset))):
    img_diff = np.abs(efficient_features[i] - direct_features[i])
    print(
        f"Image {i} - Max diff: {np.max(img_diff):.8f}, Mean diff: {np.mean(img_diff):.8f}"
    )

    # Check if the features are at least in the same range
    print(
        f"  Efficient range: [{np.min(efficient_features[i]):.6f}, {np.max(efficient_features[i]):.6f}]"
    )
    print(
        f"  Direct range: [{np.min(direct_features[i]):.6f}, {np.max(direct_features[i]):.6f}]"
    )

print("Done!")
