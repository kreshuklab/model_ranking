import os
import typer
import torch
from pathlib import Path

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.data_structures import (
    ClassificationTrainConfig,
)
from model_ranking.classification import (
    copy_classification_config,
    get_classification_dataloader,
    initialise_wandb,
    run_training,
)


def main(config: Path = typer.Option(..., help="Path to config yaml")):
    config_data, _ = load_config_direct(config)
    cfg = ClassificationTrainConfig.model_validate(config_data)

    output_path = (
        Path(cfg.training_config.save_path)
        / cfg.model_cfg.conv1.name
        / cfg.model_cfg.modelname
    )
    os.makedirs(output_path, exist_ok=True)

    copy_classification_config(old_path=config, save_path=output_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available, using CPU")
    else:
        assert torch.cuda.device_count() == 1

    initialise_wandb(cfg.wandb, config_data)

    # set up dataloaders
    train_loader = get_classification_dataloader(cfg.train_loader)
    val_loader = get_classification_dataloader(cfg.val_loader)

    run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(device),
        config=cfg.training_config,
        model_cfg=cfg.model_cfg,
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
