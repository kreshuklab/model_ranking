from collections import OrderedDict
import numpy as np
import os
from pathlib import Path
import shutil
import torch
from typing import Any, Dict, List, Mapping, Union

from .augmentations import augmentation_type
from .dataclass import ClassificationPatchPositionConfig

from model_ranking.utils import load_h5

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


def load_checkpoint_classnet(
    model_name: str,
    model: torch.nn.Module,
    path: str,
    location: str,
    ckpt: str = "best",
    model_key: str = "resnet_18",
):
    # load model
    checkpoint_path = os.path.join(path, model_name, f"{ckpt}.pt")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Cannot find checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=location)
    new_state_dict: OrderedDict[str, Any] = OrderedDict()
    for k, v in checkpoint["model_state"].items():
        name = f"{model_key}." + k
        new_state_dict[name] = v
    _ = model.load_state_dict(new_state_dict)
    return model


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
    new_path = Path(save_path).parent / Path(old_path).name
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
):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    source = get_source_from_classification_model_name(model_name)
    paths = list(
        base_path.rglob(f"{source}_to_{target}/{model_name}/{aug}/predictions.h5")
    )
    assert (
        len(paths) == 1
    ), f"Expected exactly one path for {model_name} to {target}, found {len(paths)}"
    return paths[0]
