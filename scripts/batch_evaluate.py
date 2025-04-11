from typing import Annotated
import typer

from model_ranking.results import (
    # run_foreground_patch_selection,
    save_summary_metrics,
)
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.dataclass import EvaluateConfig, SummaryResultsConfig
from model_ranking.evaluation import run_performance_evaluation
from model_ranking.yaml_generators import generate_run_yamls


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    run_config_paths = generate_run_yamls(config)

    for transfer_title, config_paths in run_config_paths.items():
        print(f"Running transfer {transfer_title}")
        for config_path in config_paths:
            cfg, _ = load_config_direct(config_path)
            eval_config = EvaluateConfig.model_validate(cfg["evaluation"])
            _ = run_performance_evaluation(eval_config)
            # save summary metrics
            summary_config = SummaryResultsConfig.model_validate(cfg["summary_results"])
            # _ = run_foreground_patch_selection(summary_config)
            save_summary_metrics(summary_config)


if __name__ == "__main__":
    typer.run(main)
