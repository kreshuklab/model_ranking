import typer
from typing import Annotated

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking import (
    ConsistencyPatchedTransformerConfig,
    run_patched_transformer_consistency,
    save_summary_metrics,
    ForegroundFilterConfig,
    run_foreground_patch_selection,
)


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    config_values, _ = load_config_direct(config)
    consis_config = ConsistencyPatchedTransformerConfig.model_validate(config_values)

    run_patched_transformer_consistency(consis_config)

    summary_config = consis_config.summary
    if isinstance(summary_config.filter_patches, ForegroundFilterConfig):
        _ = run_foreground_patch_selection(summary_config)
    save_summary_metrics(summary_config)


if __name__ == "__main__":
    typer.run(main)
