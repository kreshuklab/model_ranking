import numpy as np
from numpy.typing import NDArray
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, List, Union

import wandb


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
    predictions: List[NDArray[np.int32]] = []
    labels: List[NDArray[np.int32]] = []

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
            predictions.append(prediction[:, 0].to("cpu").numpy().astype(np.int32))
            labels.append(y[:, 0].to("cpu").numpy().astype(np.int32))

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
        if len(x) > 32:  # type: ignore
            x = x[:32]  # type: ignore
            y = y[:32]  # type: ignore
            prediction = prediction[:32]  # type: ignore
        wandb.log(
            {
                "validation-image": [
                    wandb.Image(
                        x,  # type: ignore
                        caption=f"pred: {prediction[:, 0].to('cpu').numpy()}, label: {y[:, 0].to('cpu').numpy()}",  # type: ignore
                    )
                ],
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
