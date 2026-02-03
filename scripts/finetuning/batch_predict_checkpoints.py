from typing import Annotated
import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.data_structures import (
    MetaConfig,
)
from model_ranking.predict import batch_predict_checkpoints


def main(
    config: Annotated[str, typer.Option(help="Path to the configuration file")],
    checkpoints: Annotated[
        str, typer.Option(help="Comma-separated list of checkpoints")
    ] = "best",
):
    """
    Main function to run predictions for multiple checkpoints.
    """
    checkpoints_list = [ckpt.strip() for ckpt in checkpoints.split(",")]
    meta_cfg, _ = load_config_direct(config)
    meta_cfg = MetaConfig.model_validate(meta_cfg)
    batch_predict_checkpoints(meta_cfg, checkpoints_list)


if __name__ == "__main__":
    typer.run(main)
