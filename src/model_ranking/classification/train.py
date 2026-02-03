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

from .model import ClassificationNet
from .utils import (
    create_image_grid,
    load_from_checkpoint,
    save_checkpoint,
    get_loss_function,
)
from .validate import validate
from model_ranking import TrainingSettingsConfig, ClassificationModelConfig


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
            # Create image grid from batch
            image_grid = create_image_grid(x, max_images=12)
            wandb.log({"input": wandb.Image(image_grid.cpu().numpy())}, step=step)

        prediction = torch.as_tensor(
            (torch.sigmoid(prediction)) > 0.5, dtype=torch.int16
        )

        # store the predictions and labels
        predictions.append(prediction[:, 0].to("cpu").numpy().astype(np.int16))
        labels.append(y[:, 0].to("cpu").numpy().astype(np.int16))

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
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    device: torch.device,
    model_cfg: ClassificationModelConfig,
    config: TrainingSettingsConfig,
):
    # set up backbone model
    print("Initialize model")
    model = ClassificationNet(model_cfg)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    save_path = os.path.join(
        config.save_path, model_cfg.conv1.name, model_cfg.modelname
    )
    assert wandb.run is not None
    if model_cfg.ckpt_path is not None:
        assert wandb.run.resumed
        assert (
            model_cfg.ckpt_key is not None
        ), "When resuming a run, a checkpoint key must be provided"
        result = load_from_checkpoint(
            name=model_cfg.modelname,
            model=model,
            path=model_cfg.ckpt_path,
            location=device,
            key=model_cfg.ckpt_key,
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
        os.makedirs(save_path, exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **config.scheduler_kwargs.model_dump()
    )

    loss_function = get_loss_function(config.loss_function).to(device)

    for epoch in trange(config.num_epochs, file=sys.stdout):
        current_epoch = epoch + starting_epoch
        train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            epoch=current_epoch,
            log_image_interval=config.logging.log_image_interval,
            log_pred=config.logging.log_pred,
        )

        step = (current_epoch + 1) * len(train_loader)

        _, _, current_loss, _ = validate(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
            step=step,
            log_val_images=config.logging.log_val_images,
            log_pred=config.logging.log_pred,
        )

        lr_scheduler.step(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = current_epoch
            save_checkpoint(
                "best",
                model,
                save_path,
                optimizer,
                best_epoch,
                best_loss,
                current_epoch,
                current_loss,
            )

        save_checkpoint(
            "latest",
            model,
            save_path,
            optimizer,
            best_epoch,
            best_loss,
            current_epoch,
            current_loss,
        )

    print(f"Training finished - Best loss: {best_loss} at epoch {best_epoch}")
