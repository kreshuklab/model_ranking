from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from adabn.utils import compute_bn_stats, replace_bn_stats

from model_ranking import Pytorch3DUnetModelConfig

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.datasets.utils import (
    get_test_loaders,  # pyright: ignore[reportUnknownVariableType]
)


def print_batchnorm_stats(model: nn.Module, prefix: str = "") -> None:
    """Print statistics of all BatchNorm layers in the model."""
    print(f"\n{prefix}BatchNorm Layer Statistics:")
    print("=" * 50)

    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append((name, module))

    if not bn_layers:
        print("No BatchNorm layers found in the model.")
        return

    for name, bn_layer in bn_layers:
        print(f"\nLayer: {name}")
        print(f"  Running mean shape: {bn_layer.running_mean.shape}")
        print(
            f"  Running mean (first 5): {bn_layer.running_mean[:5].cpu().detach().numpy()}"
        )
        print(f"  Running var shape: {bn_layer.running_var.shape}")
        print(
            f"  Running var (first 5): {bn_layer.running_var[:5].cpu().detach().numpy()}"
        )
        print(f"  Num batches tracked: {bn_layer.num_batches_tracked}")

        if hasattr(bn_layer, "weight") and bn_layer.weight is not None:
            print(f"  Weight (first 5): {bn_layer.weight[:5].cpu().detach().numpy()}")
        if hasattr(bn_layer, "bias") and bn_layer.bias is not None:
            print(f"  Bias (first 5): {bn_layer.bias[:5].cpu().detach().numpy()}")

    print("=" * 50)


model_config: Dict[str, Any] = {
    "model": {
        "name": "UNet2d_as3d",
        "in_channels": 1,
        "out_channels": 1,
        "layer_order": "bcr",
        "f_maps": 32,
        "final_sigmoid": False,
        "feature_return": False,
        "is_segmentation": False,
        "feature_perturbation": None,
    },
    "source_checkpoint": (
        "/g/kreshuk/talks/segmentation_ModelSelection/experiments/EPFL/BatchNorm/E_model_NA2/best_checkpoint.pytorch"
    ),
}

supervised_loader_config: Dict[str, Any] = {
    "device": "cuda:0",
    "loaders": {
        "dataset": "StandardHDF5Dataset",
        "batch_size": 32,
        "num_workers": 8,
        "raw_internal_path": "raw",
        "label_internal_path": "labels",
        "global_normalization": True,
        "global_percentiles": None,
        "output_dir": (
            "/g/kreshuk/talks/model_ranking/notebooks/self_training/AdaptiveBatchNorm"
        ),
        "test": {
            "file_paths": ["/scratch/talks/data/EPFL/test.h5"],
            "slice_builder": {
                "name": "SliceBuilder",
                "patch_shape": [1, 256, 256],
                "stride_shape": [1, 256, 256],
                "halo_shape": [0, 32, 32],
            },
            "transformer": {
                "raw": [
                    {"name": "Normalize"},
                    {"name": "ToTensor", "expand_dims": True},
                ]
            },
            "roi": None,
        },
    },
}

model_cfg = Pytorch3DUnetModelConfig.model_validate(model_config["model"])

model = get_model(model_cfg.model_dump())
source_checkpoint = model_config["source_checkpoint"]

print("Mean teacher training initialized from source model:", source_checkpoint)
if Path(source_checkpoint).suffix == ".pt":
    model_key = "model_state"
else:
    model_key = "model_state_dict"
_ = load_checkpoint(source_checkpoint, model, model_key=model_key)


test_loaders = get_test_loaders(supervised_loader_config)
test_loader = next(test_loaders)

# Move model to the appropriate device
device = supervised_loader_config["device"]
model = model.to(device)

# Set model to eval mode
_ = model.eval()

# Print initial BatchNorm statistics
print_batchnorm_stats(model, "BEFORE AdaBN - ")

# Compute target domain statistics
bn_stats = compute_bn_stats(model, test_loader)

# Apply AdaBN
replace_bn_stats(model, bn_stats)

# Print updated BatchNorm statistics
print_batchnorm_stats(model, "AFTER AdaBN - ")
