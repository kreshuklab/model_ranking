from typing import Annotated
import typer
from pathlib import Path

from model_ranking.results import run_foreground_patch_selection, save_summary_metrics
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.consistency import (
    run_consistency_evaluation,
)
from model_ranking.dataclass import (
    EvaluateConfig,
    ConsistencyConfig,
    SummaryResultsConfig,
)
from model_ranking.evaluation import run_performance_evaluation


def main(
    config: Annotated[
        str, typer.Option(help="Path to a single config file", exists=True)
    ],
):
    cfg, _ = load_config_direct(config)
    eval_config = EvaluateConfig.model_validate(cfg["evaluation"])
    _ = run_performance_evaluation(eval_config)

    if "none" in str(Path(config).name):
        print(f"Skipping consistency evaluation for {Path(config).stem}")
    else:
        consis_config = ConsistencyConfig.model_validate(cfg["consistency"])
        _, _ = run_consistency_evaluation(consis_config)

    # save summary metrics
    summary_config = SummaryResultsConfig.model_validate(cfg["summary_results"])
    _ = run_foreground_patch_selection(summary_config)
    save_summary_metrics(summary_config)


if __name__ == "__main__":
    typer.run(main)
