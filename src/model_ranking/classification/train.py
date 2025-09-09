import numpy as np
from numpy.typing import NDArray
import os
import sklearn.metrics as metrics
import sys
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange
from typing import Any, List, Optional
import wandb

from .dataclass import SchedulerConfig
from .utils import load_from_checkpoint, save_checkpoint
from .validate import validate


def train(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    log_image_interval: Optional[int] = None,
    log_pred: bool = False,
):
    """Train model for one epoch.

    Parameters:
    model - the model we are training
    loader - the data loader that provides the training data
        (= pairs of images and labels)
    loss_function - the loss function that will be optimized
    optimizer - the optimizer that is used to update the network parameters
        by backpropagation of the loss
    device - the device used for training. this can either be the cpu or gpu
    epoch - which trainin eppch are we in? we keep track of this for logging
    log_image_interval - how often do we log images
    """

    # set model to train mode
    _ = model.train()
    predictions: List[NDArray[Any]] = []
    labels: List[NDArray[Any]] = []

    print("Epoch %d: lr=%.8f" % (epoch, optimizer.param_groups[0]["lr"]))
    # iterate over the training batches provided by the loader
    n_batches = len(loader)
    step = 0
    for batch_id, (x, y) in enumerate(loader):
        # send data and target tensors to the active device
        x = x.to(device)
        y = y.to(device)
        # set the gradients to zero, to start with "clean" gradients
        # in this training iteration
        optimizer.zero_grad()

        # apply the model to get the prediction
        prediction = model(x)

        loss_value = loss_function(prediction[:, 0], y[:, 0].float())

        loss_value.backward()
        optimizer.step()

        # log the loss value to tensorboard
        step = epoch * n_batches + batch_id
        wandb.log({"train-loss": loss_value.item()}, step=step)

        # check if we log images, and if we do then send the
        # current image to tensorboard
        if log_image_interval is not None and step % log_image_interval == 0:
            wandb.log({"input": [wandb.Image(x)]}, step=step)

        prediction = torch.as_tensor(
            (torch.sigmoid(prediction)) > 0.5, dtype=torch.int32
        )

        # store the predictions and labels
        predictions.append(prediction[:, 0].to("cpu").numpy().astype(np.int32))
        labels.append(y[:, 0].to("cpu").numpy())

    # predictions and labels to numpy arrays
    pred_cmb = np.concatenate(predictions)
    label_cmb = np.concatenate(labels)

    # log the validation results if we have a tensorboard
    accuracy_error = 1.0 - metrics.accuracy_score(label_cmb, pred_cmb)
    assert isinstance(step, int)
    wandb.log({"train-accuracy-error": accuracy_error}, step=step)

    if log_pred:
        train_pred_table = wandb.Table(columns=["label", "prediction"])
        train_pred_table.add_data(label_cmb.astype(bool), pred_cmb.astype(bool))
        wandb.log(
            {
                "train_predictions": train_pred_table,
            }
        )


def run_training(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    log_image_interval: int,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    name: str,
    learning_rate: float = 1e-4,
    scheduler_kwargs: SchedulerConfig = SchedulerConfig(),
    ckpt_name: str = "latest.pt",
    path_to_run_folder: str = ".",
    **kwargs: Any,
):
    _ = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = os.path.join(path_to_run_folder, name)
    assert wandb.run is not None
    if (
        os.path.exists(os.path.join(checkpoint_path, ckpt_name + ".pt"))
        and wandb.run.resumed
    ):
        result = load_from_checkpoint(
            name=name,
            model=model,
            path=path_to_run_folder,
            location=device,
            ckpt=ckpt_name,
            optimizer=optimizer,
        )
        # Since optimizer is provided, we know this returns the tuple form
        assert isinstance(result, tuple) and len(result) == 6
        (
            model,
            optimizer,
            best_epoch,
            best_loss,
            starting_epoch,
            current_loss,
        ) = result

        starting_epoch = starting_epoch + 1

    else:
        best_loss = np.inf
        best_epoch = 0
        starting_epoch = 0
        os.makedirs(checkpoint_path, exist_ok=False)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **scheduler_kwargs.model_dump()
    )
    _ = loss_function.to(device)

    for epoch in trange(num_epochs, file=sys.stdout):
        current_epoch = epoch + starting_epoch
        train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            epoch=current_epoch,
            log_image_interval=log_image_interval,
            log_pred=kwargs.get("log_pred", False),
        )

        step = (current_epoch + 1) * len(train_loader)

        _, _, current_loss, _ = validate(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
            step=step,
            **kwargs,
        )

        lr_scheduler.step(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = current_epoch
            save_checkpoint(
                "best",
                model,
                checkpoint_path,
                optimizer,
                best_epoch,
                best_loss,
                current_epoch,
                current_loss,
            )

        save_checkpoint(
            "latest",
            model,
            checkpoint_path,
            optimizer,
            best_epoch,
            best_loss,
            current_epoch,
            current_loss,
        )

    print(f"Training finished - Best loss: {best_loss} at epoch {best_epoch}")
