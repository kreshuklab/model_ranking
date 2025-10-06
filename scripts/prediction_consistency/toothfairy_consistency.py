import typer
from typing import Annotated


from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking import run_toothfairy_consistency, ToothfairyConsistencyConfig


def main(
    config: Annotated[
        str, typer.Option(help="Path to a single config file", exists=True)
    ],
):
    cfg_values, _ = load_config_direct(config)
    cfg = ToothfairyConsistencyConfig.model_validate(cfg_values)

    _ = run_toothfairy_consistency(cfg)


if __name__ == "__main__":
    typer.run(main)
