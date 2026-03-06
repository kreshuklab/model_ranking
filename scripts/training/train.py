import random
import wandb
import torch

from pytorch3dunet.unet3d.config import load_config, copy_config  # pyright: ignore
from pytorch3dunet.unet3d.trainer import create_trainer  # pyright: ignore
from pytorch3dunet.unet3d.utils import get_logger  # pyright: ignore

logger = get_logger("TrainingSetup")  # pyright: ignore


def main():
    # Load and log experiment configuration
    config, config_path = load_config()
    # Setup wandb logger
    _ = wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        mode=config["wandb"]["mode"],
        config=config,
        # sync_tensorboard=True,
    )
    _ = logger.info(config)  # pyright: ignore

    manual_seed = config.get("manual_seed", None)
    if manual_seed is not None:
        _ = logger.info(  # pyright: ignore
            f"Seed the RNG for all devices with {manual_seed}"
        )
        _ = logger.warning(  # pyright: ignore
            "Using CuDNN deterministic setting. This may slow down the training!"
        )
        random.seed(manual_seed)
        _ = torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = create_trainer(config)
    # Copy config file
    copy_config(config, config_path)
    # Start training
    trainer.fit()


if __name__ == "__main__":
    main()
