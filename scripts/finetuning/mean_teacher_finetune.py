import typer
from typing import Annotated

from model_ranking import MeanTeacherConfig
from model_ranking import self_training_mean_teacher
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to config file", exists=True)],
):
    cfg, _ = load_config_direct(config)
    mt_cfg = MeanTeacherConfig.model_validate(cfg)

    self_training_mean_teacher(mt_cfg, config)


if __name__ == "__main__":
    typer.run(main)
