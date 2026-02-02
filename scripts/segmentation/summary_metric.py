import typer
from typing import Annotated

from model_ranking import (
    save_summary_metrics,
    SummaryResultsConfig,
    ForegroundFilterConfig,
    run_foreground_patch_selection,
)

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    config_values, _ = load_config_direct(config)
    summary_config = SummaryResultsConfig.model_validate(config_values)

    if isinstance(summary_config.filter_patches, ForegroundFilterConfig):
        _ = run_foreground_patch_selection(summary_config)
    save_summary_metrics(summary_config)


if __name__ == "__main__":
    typer.run(main)
