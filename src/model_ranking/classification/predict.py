from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List
from tqdm import tqdm

from .datasets import ClassificationFilteredDataset
from .utils import merge_dicts


def predict_with_features(
    model: nn.Module,
    loader: DataLoader[ClassificationFilteredDataset],
    device: torch.device,
    patch_pos: bool = False,
    extract_features: bool = False,
):
    model = model.eval()

    num_samples = loader.dataset._len  # pyright: ignore
    assert isinstance(num_samples, int)

    # Use -1 as a sentinel value instead of np.nan for integer arrays
    predictions = np.full((num_samples), -1, dtype=np.float32)
    labels = np.full((num_samples), -1, dtype=np.int8)
    patch_positions = np.full((num_samples, 3), -1, dtype=np.int16)
    layerwise_features: List[OrderedDict[str, Any]] = []

    with torch.no_grad():
        if patch_pos is True:
            for i, (x, y, pp) in enumerate(tqdm(loader)):
                x = x.to(device)
                y = y.to(device)
                if extract_features is True:
                    prediction, x_features = model.features(x)
                    layerwise_features.append(x_features.copy())
                else:
                    prediction = model(x)

                prediction = torch.as_tensor((torch.sigmoid(prediction)))
                # store the predictions and labels
                assert loader.batch_size is not None
                predictions[
                    i * loader.batch_size : i * loader.batch_size + len(prediction)
                ] = (prediction[:, 0].to("cpu").numpy())
                labels[i * loader.batch_size : i * loader.batch_size + len(y)] = (
                    y[:, 0].to("cpu").numpy().astype(np.int8)
                )

                patch_positions[
                    i * loader.batch_size : i * loader.batch_size + len(pp)
                ] = pp.to("cpu").numpy()
        else:
            for i, (x, y) in enumerate(tqdm(loader)):
                x = x.to(device)
                y = y.to(device)
                if extract_features is True:
                    # assert isinstance(model.features, nn.Module)
                    prediction, x_features = model.features(x)
                    layerwise_features.append(x_features.copy())
                else:
                    prediction = model(x)

                prediction = torch.as_tensor((torch.sigmoid(prediction)))
                # store the predictions and labels
                assert loader.batch_size is not None
                predictions[
                    i * loader.batch_size : i * loader.batch_size + len(prediction)
                ] = (prediction[:, 0].to("cpu").numpy())
                labels[i * loader.batch_size : i * loader.batch_size + len(y)] = (
                    y[:, 0].to("cpu").numpy().astype(np.int8)
                )
    output: Dict[str, Any] = {
        "predictions": predictions,
        "labels": labels,
    }
    if patch_pos is True:
        output["patch_positions"] = patch_positions
    if extract_features is True:
        merged_layerwise_features = merge_dicts(layerwise_features)
        output["features"] = merged_layerwise_features

    return output
