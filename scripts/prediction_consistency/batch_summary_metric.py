from typing import Annotated
import typer
from tqdm import tqdm

from model_ranking import (
    run_foreground_patch_selection,
    save_summary_metrics,
    generate_run_yamls,
    ForegroundFilterConfig,
    SummaryResultsConfig,
)
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    cfg, _ = load_config_direct(config)
    assert (
        cfg["run_mode"] == "summary_results"
    ), f"Current Run mode = {cfg['run_mode']}, should be 'summary_results'"
    run_config_paths = generate_run_yamls(cfg)

    for transfer_title, config_paths in run_config_paths.items():
        print(f"Running transfer {transfer_title}")
        for config_path in tqdm(config_paths):
            cfg, _ = load_config_direct(config_path)

            # save summary metrics
            summary_config = SummaryResultsConfig.model_validate(cfg["summary_results"])
            if isinstance(summary_config.filter_patches, ForegroundFilterConfig):
                _ = run_foreground_patch_selection(summary_config)
            save_summary_metrics(summary_config)


if __name__ == "__main__":
    typer.run(main)
