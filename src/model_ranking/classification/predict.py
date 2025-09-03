from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from .datasets import ClassificationFilteredDataset
from .utils import merge_dicts

from model_ranking.feature_ranking import FeatureExtractor


def predict_with_features(
    model: nn.Module,
    loader: DataLoader[ClassificationFilteredDataset],
    device: torch.device,
    feature_layers: Optional[List[str]] = None,
    patch_pos: bool = False,
):
    model = model.eval()

    num_samples = loader.dataset._len  # pyright: ignore
    assert isinstance(num_samples, int)

    # Use -1 as a sentinel value instead of np.nan for integer arrays
    predictions = np.full((num_samples), -1, dtype=np.float32)
    labels = np.full((num_samples), -1, dtype=np.int8)
    patch_positions = np.full((num_samples, 3), -1, dtype=np.int16)
    layerwise_features: List[OrderedDict[str, Any]] = []

    if feature_layers:
        feature_extractor = FeatureExtractor(model, layers=feature_layers)
    else:
        feature_extractor = None
    model = model.to(device)

    with torch.no_grad():
        if patch_pos is True:
            for i, (x, y, pp) in enumerate(tqdm(loader)):
                x = x.to(device)
                y = y.to(device)
                if feature_extractor:
                    x_features, _, prediction = feature_extractor(x)
                    layerwise_features.append(x_features)
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
                if feature_extractor:
                    x_features, _, prediction = feature_extractor(x)
                    layerwise_features.append(x_features)
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
    if feature_extractor:
        feature_extractor.remove_handler()
    output: Dict[str, Any] = {
        "predictions": predictions,
        "labels": labels,
    }
    if patch_pos is True:
        output["patch_positions"] = patch_positions
    if len(layerwise_features) > 0:
        merged_layerwise_features = merge_dicts(layerwise_features)
        output["features"] = merged_layerwise_features

    return output


def classification_prediction_evaluation(
    predictions: NDArray[Any], labels: NDArray[Any]
) -> Tuple[float, float, float, float]:
    accuracy_error = 1.0 - metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(  # pyright: ignore[reportUnknownVariableType]
        labels, predictions
    )
    recall = metrics.recall_score(  # pyright: ignore[reportUnknownVariableType]
        labels, predictions
    )
    f1_score = metrics.f1_score(  # pyright: ignore[reportUnknownVariableType]
        labels, predictions
    )
    assert isinstance(accuracy_error, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1_score, float)
    return accuracy_error, precision, recall, f1_score
