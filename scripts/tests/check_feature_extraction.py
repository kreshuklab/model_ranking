import numpy as np
import torch
import time
from tqdm import tqdm
from typing import Dict, Any

from model_ranking import TransferFeatureExtraction
from model_ranking import TransferFeatureExtractionConfig

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
        "output_dir_path": None,
    },
}

feature_ranking_cfg = TransferFeatureExtractionConfig.model_validate(config)

feature_ranking = TransferFeatureExtraction(
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

# model = feature_ranking.initialise_model(
#     model_cfg,
#     model_dir_path=feature_ranking.source_model_base_path,
# )

target_dataset = feature_ranking.target_datasets["EPFL"]
target_dataloader = feature_ranking.target_dataloaders["EPFL"]

print("Starting feature extraction timing comparison...")
print("=" * 60)

# Time the original extract_features_sampled method
print("Running extract_features_sampled()...")
start_time = time.time()
features, labels, indices = feature_ranking.extract_features_sampled(
    model,
    target="EPFL",
    target_dataset=target_dataset,
)
original_time = time.time() - start_time
print(f"✓ extract_features_sampled() completed in {original_time:.4f} seconds")

# Time the efficient extract_features_sampled_efficient method
print("\nRunning extract_features_sampled_efficient()...")
start_time = time.time()
features2, labels2, indices2 = feature_ranking.extract_features_sampled_efficient(
    model,
    target="EPFL",
    target_dataloader=target_dataloader,  # Use the dataloader directly
)
efficient_time = time.time() - start_time
print(
    f"✓ extract_features_sampled_efficient() completed in {efficient_time:.4f} seconds"
)

# Calculate and display performance comparison
speedup = original_time / efficient_time if efficient_time > 0 else float("inf")
time_saved = original_time - efficient_time

print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON RESULTS:")
print("=" * 60)
print(f"Original method time:   {original_time:.4f} seconds")
print(f"Efficient method time:  {efficient_time:.4f} seconds")
print(
    f"Time saved:             {time_saved:.4f} seconds ({time_saved/original_time*100:.1f}%)"
)
print(f"Speedup factor:         {speedup:.2f}x")
print("=" * 60)

for layer, feature in features.items():
    print(f"Layer: {layer}, Feature shape: {feature.shape}")
    print(f"Labels shape: {labels[layer].shape}")
    print(f"Indices shape: {indices[layer].shape}")

for layer, feature in features2.items():
    print(f"Layer: {layer}, Feature shape: {feature.shape}")
    print(f"Labels shape: {labels2[layer].shape}")
    print(f"Indices shape: {indices2[layer].shape}")


layer = "decoders.2"
indices_equal = np.array_equal(indices[layer], indices2[layer])
print(f"indices for layer '{layer}' are equal:", indices_equal)
# if not indices_equal:
#     diff_indices = np.where(indices[layer] != indices2[layer])
#     print(f"Indices differ at positions: {diff_indices}")
#     print("Sample differing values:")
#     print("indices[layer]:", indices[layer][diff_indices])
#     print("indices2[layer]:", indices2[layer][diff_indices])

print("\nRunning manual feature extraction for comparison...")
start_time = time.time()
with torch.no_grad():
    features_per_image = np.zeros((len(target_dataset), 1000, 32))
    for i, (image, label) in enumerate(tqdm(iter(target_dataset))):
        image = image.to("cuda:0")
        pred, feats = model(image)

        last_layer_feats = feats[-1].cpu().detach().numpy()
        feat_flat = np.reshape(last_layer_feats, [-1, last_layer_feats.shape[1]])
        features_per_image[i] = feat_flat[indices2["decoders.2"][i]]
manual_time = time.time() - start_time
print(f"✓ Manual feature extraction completed in {manual_time:.4f} seconds")

print("\n" + "=" * 60)
print("EXTENDED PERFORMANCE COMPARISON:")
print("=" * 60)
print(f"Original method time:   {original_time:.4f} seconds")
print(f"Efficient method time:  {efficient_time:.4f} seconds")
print(f"Manual extraction time: {manual_time:.4f} seconds")
print(f"Efficient vs Original:  {original_time/efficient_time:.2f}x speedup")
print(f"Efficient vs Manual:    {manual_time/efficient_time:.2f}x speedup")
print("=" * 60)

print("Features per image shape:", features_per_image.shape)

print("Features per image shape:", features_per_image.shape)

# Check with more reasonable tolerance for floating point differences
print(
    "features match:",
    np.allclose(features2["decoders.2"], features_per_image, atol=1e-4),
)

print(
    "features_match (original vs efficient):",
    np.allclose(features["decoders.2"], features2["decoders.2"], atol=1e-4),
)

# Show the actual differences for debugging
diff = np.abs(features2["decoders.2"] - features_per_image)
print(f"Max absolute difference: {np.max(diff):.2e}")
print(f"Mean absolute difference: {np.mean(diff):.2e}")
print(f"Number of non-zero differences: {np.sum(diff > 1e-4)}")
