import numpy as np
from numpy.typing import NDArray
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, List, Union

import wandb

from ._utils import create_image_grid


def validate(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: nn.Module,
    device: Union[str, torch.device],
    step: int,
    log_val_images: bool = True,
    log_pred: bool = False,
):
    """
    Validate the model predictions.

    Parameters:
    model - the model to be evaluated
    loader - the loader providing images and labels
    loss_function - the loss function
    device - the device used for prediction (cpu or gpu)
    step - the current training step. we need to know this for logging
    tb_logger - the tensorboard logger. if 'None', logging is disabled
    """
    # set the model to eval mode
    _ = model.eval()
    n_batches = len(loader)

    # we record the loss and the predictions / labels for all samples
    mean_loss = 0
    predictions: List[NDArray[np.int16]] = []
    labels: List[NDArray[np.int16]] = []

    # the model parameters should not be updated during validation
    # torch.no_grad disables gradient updates in its scope
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            # update the loss
            mean_loss += loss_function(prediction[:, 0], y[:, 0].float()).item()

            # compute the most likely class predictions
            prediction = torch.as_tensor(
                (torch.sigmoid(prediction)) > 0.5, dtype=torch.int32
            )

            # store the predictions and labels
            predictions.append(prediction[:, 0].to("cpu").numpy().astype(np.int16))
            labels.append(y[:, 0].to("cpu").numpy().astype(np.int16))

    # predictions and labels to numpy arrays
    pred_cmb = np.concatenate(predictions)
    labels_cmb = np.concatenate(labels)

    # log the validation results if we have a tensorboard
    accuracy_error = 1.0 - metrics.accuracy_score(labels_cmb, pred_cmb)
    mean_loss /= n_batches

    wandb.log(
        {
            "validation-error": accuracy_error,
            "validation-loss": mean_loss,
        },
        step=step,
    )

    if log_val_images:
        # Create image grid from batch (limit to 12 images max)
        image_grid = create_image_grid(x, max_images=12)  # type: ignore

        # Create caption with predictions and labels (for first 12 items)
        batch_size = min(x.size(0), 12)  # type: ignore
        pred_values = prediction[:batch_size, 0].to("cpu").numpy()  # type: ignore
        label_values = y[:batch_size, 0].to("cpu").numpy()  # type: ignore
        caption = f"pred: {pred_values}, label: {label_values}"

        wandb.log(
            {
                "validation-image": wandb.Image(
                    image_grid.cpu().numpy(),
                    caption=caption,
                )
            },
            step=step,
        )

    if log_pred:
        val_pred_table = wandb.Table(columns=["label", "prediction"])
        val_pred_table.add_data(labels_cmb.astype(bool), pred_cmb.astype(bool))
        wandb.log(
            {
                "val_predictions": val_pred_table,
            }
        )

    # return all predictions and labels for further evaluation
    return predictions, labels, mean_loss, accuracy_error
