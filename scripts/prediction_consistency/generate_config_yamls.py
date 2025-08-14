from typing import Annotated
import typer

from model_ranking.yaml_generators import generate_run_yamls
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to Meta config file", exists=True)],
):

    cfg, _ = load_config_direct(config)
    _ = generate_run_yamls(cfg)


if __name__ == "__main__":
    typer.run(main)
