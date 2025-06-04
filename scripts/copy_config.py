import typer
from typing import Annotated

from model_ranking.config import copy_config
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.dataclass import MeanTeacherConfig


def main(
    config: Annotated[
        str, typer.Option(help="Path to config file to be copied", exists=True)
    ],
):
    cfg, _ = load_config_direct(config)
    mt_cfg = MeanTeacherConfig.model_validate(cfg)

    copy_config(mt_cfg, config)


if __name__ == "__main__":
    typer.run(main)
