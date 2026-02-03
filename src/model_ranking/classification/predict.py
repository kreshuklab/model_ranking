from collections import OrderedDict
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from .data_loader import get_classification_TTA_loaders
from .datasets import ClassificationFilteredDataset
from .model import ClassificationNet
from .utils import (
    merge_dicts,
    get_classification_transfer,
    copy_classification_config,
    load_from_checkpoint,
)

from model_ranking.data_structures import ClassificationPredictConfig
from model_ranking.features import FeatureExtractor

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def predict(
    model: nn.Module,
    loader: DataLoader[ClassificationFilteredDataset],
    device: torch.device,
    feature_layers: Optional[List[str]] = None,
    input_features: bool = False,
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
                    out_features, in_features, prediction = feature_extractor(x)
                    if input_features:
                        layerwise_features.append(in_features)
                    else:
                        layerwise_features.append(out_features)
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
                    out_features, in_features, prediction = feature_extractor(x)
                    if input_features:
                        layerwise_features.append(in_features)
                    else:
                        layerwise_features.append(out_features)
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


def run_classification_prediction(config_path: Union[str, Path]):
    cfg_data, _ = load_config_direct(config_path)
    cfg = ClassificationPredictConfig.model_validate(cfg_data)

    # get TTA dataloaders
    loaders = get_classification_TTA_loaders(cfg.loader)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available, using CPU")
    model = ClassificationNet(cfg.model)
    model = load_from_checkpoint(
        cfg.model.modelname,
        model=model,
        path=str(cfg.model.ckpt_path),
        location=device,
        layer_key=cfg.model.layer_key,
    )
    assert isinstance(model, torch.nn.Module)
    model = model.eval()

    for aug, loader in tqdm(loaders.items()):
        print(f"Applying {aug} Test Time Augmentation")
        if aug == "None":
            aug_path = "None"
        else:
            aug_path = Path(aug.split("_")[0]) / aug.split("_")[-1]
        transfer_title = get_classification_transfer(
            cfg.model.modelname, str(cfg.loader.dataset.raw_path)
        )
        save_dir_path = (
            Path(cfg.output.save_dir_path)
            / transfer_title
            / cfg.model.modelname
            / aug_path
        )
        # check if save path exists if not create it
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True, exist_ok=True)

        save_path = save_dir_path / "predictions.h5"

        copy_classification_config(old_path=config_path, save_path=save_path.parent)

        # Run classification prediction
        model_output = predict(
            model,
            loader=loader,
            device=torch.device(device),
            feature_layers=cfg.model.feature_layers,  # pyright: ignore
            input_features=cfg.model.input_features,
        )

        evaluation_scores = classification_prediction_evaluation(
            (model_output["predictions"] > cfg.output.prediction_threshold).astype(
                np.uint8
            ),
            model_output["labels"],
        )

        with h5py.File(save_path, "w") as f:
            for output_key, value in model_output.items():
                if output_key == "features":
                    for k, v in model_output["features"].items():
                        _ = f.create_dataset(k, data=v)
                else:
                    _ = f.create_dataset(output_key, data=value)

            for metric, score in zip(
                ["accuracy_error", "precision", "recall", "f1_score"],
                evaluation_scores,
            ):
                _ = f.create_dataset(metric, data=score)
