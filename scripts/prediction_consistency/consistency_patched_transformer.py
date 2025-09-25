import typer
from typing import Annotated

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking import (
    ConsistencyPatchedTransformerConfig,
    run_patched_transformer_consistency,
)


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    config_values, _ = load_config_direct(config)
    consis_config = ConsistencyPatchedTransformerConfig.model_validate(**config_values)

    run_patched_transformer_consistency(consis_config)


if __name__ == "__main__":
    typer.run(main)
