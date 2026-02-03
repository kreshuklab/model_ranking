from collections import OrderedDict
import numpy as np
import os
from pathlib import Path
import shutil
import torch
import torchvision.utils as vutils  # pyright: ignore[reportMissingTypeStubs]
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union
import wandb

from model_ranking.utils import load_h5
from model_ranking.data_structures import (
    augmentation_type,
    ClassificationPatchPositionConfig,
    WandbConfig,
)

CLASSIFICATION_DATASETS = {
    "epfl": "EPFL",
    "EPFL": "EPFL",
    "Hmito": "Hmito",
    "Rmito": "Rmito",
    "VNC": "VNC",
}


def merge_dicts(dicts: Union[List[Dict[Any, Any]], List[OrderedDict[str, Any]]]):
    if not dicts:
        raise ValueError("dicts must not be empty")
    keys = dicts[0].keys()
    if not all(d.keys() == keys for d in dicts):
        raise ValueError("All dictionaries in dicts must have the same keys")
    merged_dict: Dict[Any, Any] = {}
    for key in keys:
        # Initialize merged_value to None
        merged_value = np.array([])
        # Concatenate the tensors for the current key
        for i in range(len(dicts)):
            if i == 0:
                merged_value = dicts[i][key].to("cpu").numpy()
            else:
                merged_value = np.vstack(
                    (merged_value, dicts[i][key].to("cpu").numpy())
                )
        merged_dict[key] = merged_value.squeeze()
    return merged_dict


def load_from_checkpoint(
    name: str,
    model: torch.nn.Module,
    path: str,
    location: Union[str, torch.device],
    ckpt: str = "best.pt",
    optimizer: Optional[torch.optim.Optimizer] = None,
    key: str = "model_state",
    layer_key: Optional[str] = "ClassNet",
) -> Union[
    torch.nn.Module,
    Tuple[torch.nn.Module, torch.optim.Optimizer, int, float, int, float],
]:
    # load model
    checkpoint_path = os.path.join(path, name, f"{ckpt}")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Cannot find checkpoint {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=location)
    if layer_key is not None:
        new_state_dict: OrderedDict[str, Any] = OrderedDict()
        for k, v in checkpoint[key].items():
            name = f"{layer_key}." + k
            new_state_dict[name] = v
        _ = model.load_state_dict(new_state_dict)
    else:
        _ = model.load_state_dict(checkpoint[key])
    if optimizer is None:
        return model

    else:
        # optimizer.load_state_dict(checkpoint["optimizer"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        current_epoch = checkpoint["current_epoch"]
        current_loss = checkpoint["current_loss"]
        return (
            model,
            optimizer,
            best_epoch,
            best_loss,
            current_epoch,
            current_loss,
        )


def save_checkpoint(
    ckpt: str,
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer,
    best_epoch: int,
    best_loss: float,
    epoch: int,
    loss: float,
):
    """Save model checkpoint.

    Parameters:
    model - the model to be saved
    optimizer - the optimizer to be saved
    epoch - the current epoch
    path - the path to the checkpoint folder
    """
    save_path = os.path.join(path, f"{ckpt}.pt")

    if ckpt == "best" or ckpt == "latest":
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "current_epoch": epoch,
                "current_loss": loss,
            },
            save_path,
        )
    else:
        raise ValueError(f"Invalid checkpoint type: {ckpt}")


def get_patch_positions(config: ClassificationPatchPositionConfig):
    # Load classification patch positions from h5 file
    patch_pos = load_h5(config.patch_pos_path, config.patch_pos_key)
    # If region of interest of orginal data volume specified, only
    # keep patches within this region
    if config.slice_offset:
        patch_pos[:, 0] -= config.slice_offset
    if config.roi is not None:
        roi = np.array(config.roi)
        select_ids = np.all((patch_pos >= roi[:, 0]) & (patch_pos <= roi[:, 1]), axis=1)
        patch_pos = patch_pos[select_ids]

    if config.n_unique_patches:
        if config.n_unique_patches > len(patch_pos):
            print(
                f"Warning: n_unique_patches ({config.n_unique_patches}) is greater than the"
                + f" number of available patches ({len(patch_pos)}) taking full set."
            )
        else:
            if config.rnd_seed is not None:
                rng = np.random.default_rng(config.rnd_seed)
                patch_pos = rng.choice(
                    patch_pos, size=config.n_unique_patches, replace=False
                )
            else:
                patch_pos = np.random.choice(
                    patch_pos, size=config.n_unique_patches, replace=False
                )
    print(f"number of unique patches loaded: {len(patch_pos)}")
    return patch_pos


def copy_classification_config(old_path: Union[str, Path], save_path: Union[str, Path]):
    new_path = Path(save_path) / Path(old_path).name
    _ = shutil.copy2(old_path, new_path)


def get_classification_transfer(
    model_name: str,
    data_path: str,
    dataset_mapping: Mapping[str, str] = CLASSIFICATION_DATASETS,
) -> str:
    source = None
    target = None
    for key in dataset_mapping.keys():
        if key in model_name:
            source = dataset_mapping[key]
        if key in data_path:
            target = dataset_mapping[key]
        if source and target:
            break
    assert isinstance(source, str) and isinstance(target, str)
    return f"{source}_to_{target}"


def get_source_from_classification_model_name(
    model_name: str, dataset_mapping: Mapping[str, str] = CLASSIFICATION_DATASETS
) -> str:
    source = None
    for key in dataset_mapping.keys():
        if key in model_name:
            source = dataset_mapping[key]
            break
    assert isinstance(source, str)
    return source


def get_classification_pred_path(
    model_name: str,
    target: str,
    base_path: Union[str, Path],
    aug: augmentation_type = "None",
    aug_str: Optional[str] = None,
):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    source = get_source_from_classification_model_name(model_name)
    search_path = base_path / f"{source}_to_{target}/{model_name}/{aug}"
    if aug_str is not None:
        search_path = search_path / aug_str
    paths = list(search_path.rglob("predictions.h5"))
    assert (
        len(paths) == 1
    ), f"Expected exactly one path for {model_name} to {target}, found {len(paths)}"
    return paths[0]


def initialise_wandb(
    wandb_config: WandbConfig, config: Optional[Dict[str, Any]] = None
):
    if wandb_config.run_id is not None:
        _ = wandb.init(
            project=wandb_config.project,
            name=wandb_config.name,
            id=wandb_config.run_id,
            config=config,
            resume="must",
            mode=wandb_config.mode,
        )
    else:
        _ = wandb.init(
            project=wandb_config.project,
            name=wandb_config.name,
            config=config,
            mode=wandb_config.mode,
        )


def get_loss_function(name: Literal["BCEWithLogitsLoss"]) -> torch.nn.Module:
    if name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function {name}")


def create_image_grid(batch_tensor: torch.Tensor, max_images: int = 12) -> torch.Tensor:
    """
    Convert a batch of images to a grid for logging.

    Parameters:
    batch_tensor - tensor of shape [batch_size, channels, height, width]
    max_images - maximum number of images to include in the grid (default: 12)

    Returns:
    grid_tensor - tensor representing the image grid
    """
    # Take only the first max_images if batch is larger
    if batch_tensor.size(0) > max_images:
        batch_tensor = batch_tensor[:max_images]

    batch_size = batch_tensor.size(0)

    # Calculate grid dimensions (up to 4x3 = 12 images)
    if batch_size <= 4:
        nrow = batch_size
    elif batch_size <= 8:
        nrow = 4
    else:  # batch_size <= 12
        nrow = 4

    grid = vutils.make_grid(batch_tensor, nrow=nrow, normalize=True, padding=2)

    # Convert from CHW to HWC for wandb (wandb expects HWC format)
    grid = grid.permute(1, 2, 0)

    return grid
