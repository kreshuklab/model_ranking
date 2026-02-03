import typer
from pathlib import Path
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.data_structures import (
    ClassificationConsistencyConfig,
)
from model_ranking.classification import (
    run_classification_consistency,
)

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Path = typer.Option(
        ..., help="Path to config yaml or directory of config yamls"
    ),
):
    cfg_data, _ = load_config_direct(config)
    cfg = ClassificationConsistencyConfig.model_validate(cfg_data)
    run_classification_consistency(cfg)


if __name__ == "__main__":
    typer.run(main)
