# type: ignore

from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.datasets.utils import get_train_loaders  # adjust import path to wherever this actually lives
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("DataLoaderTest")


def main():
    config, config_path = load_config()

    logger.info("Building train/val loaders (this will trigger slice building)...")
    loaders = get_train_loaders(config)

    train_loader = loaders['train']
    val_loader = loaders['val']

    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size: {len(val_loader.dataset)}")

    # optional: pull one batch through to fully sanity-check shapes
    batch = next(iter(train_loader))
    if isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            logger.info(f"batch[{i}] shape: {item.shape}")
    else:
        logger.info(f"batch shape: {batch.shape}")


if __name__ == "__main__":
    main()