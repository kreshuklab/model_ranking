from collections import OrderedDict
import numpy as np
import os
import torch
from typing import Any, Dict, List, Union

from .dataclass import ClassificationPatchPositionConfig

from model_ranking.utils import load_h5


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


def load_checkpoint_resnet(
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
    if config.roi is not None:
        roi = np.array(config.roi)
        select_ids = np.all((patch_pos >= roi[:, 0]) & (patch_pos <= roi[:, 1]), axis=1)
        patch_pos = patch_pos[select_ids]
    return patch_pos
