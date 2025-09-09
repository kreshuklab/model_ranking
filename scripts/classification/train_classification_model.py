# type: ignore
import wandb
import torch.nn as nn
import typer
import torch
from pathlib import Path
from torchvision import models
from torch.nn.modules.loss import BCEWithLogitsLoss


from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking import (
    ClassificationTrainConfig,
    initialise_wandb,
    get_classification_dataloader,
    ClassificationNet,
    run_training,
)


def main(config: Path = None):
    config_data, _ = load_config_direct(config=config)
    cfg = ClassificationTrainConfig.model_validate(config_data)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available, using CPU")
    else:
        assert torch.cuda.device_count() == 1

    initialise_wandb(cfg.wandb, config_data)

    # set up dataloaders
    train_loader = get_classification_dataloader(cfg.train_loader)
    val_loader = get_classification_dataloader(cfg.val_loader)

    # set up backbone model
    print("Initialize model")
    model = ClassificationNet(cfg.model)

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg.training_config,
    )

    # if cfg.test_loader is not None:
    #     test_loader = get_classification_dataloader(cfg.test_loader)
    #     # test the model
    #     print("Test the model")
    #     run_prediction(
    #         model_name=config_data["model_name"],
    #         model=model,
    #         data_loader=test_loader,
    #         device=device,
    #         ckpt="best",
    #         model_path=config_data["model_path"],
    #         pred_save_path=config_data["pred_save_path"],
    #     )


if __name__ == "__main__":
    typer.run(main)
