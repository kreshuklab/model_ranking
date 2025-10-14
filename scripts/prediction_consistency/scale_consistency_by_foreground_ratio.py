import typer
from typing import Annotated

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking import (
    batch_scale_consis_by_foreground_ratio,
    ScaleConsistencyConfig,
)


def main(
    config: Annotated[str, typer.Option(help="Path to config file", exists=True)],
):
    cfg_values, _ = load_config_direct(config)
    cfg = ScaleConsistencyConfig.model_validate(cfg_values)

    batch_scale_consis_by_foreground_ratio(cfg)


if __name__ == "__main__":
    typer.run(main)
