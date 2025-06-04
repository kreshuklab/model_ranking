from typing import Dict, Any

from pytorch3dunet.unet3d.config import (
    load_config,
)
from pytorch3dunet.predict import (
    predict,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.dataclass import (
    EvaluateConfig,
    ForegroundFilterConfig,
    SummaryResultsConfig,
)
from model_ranking.evaluation import run_performance_evaluation
from model_ranking.results import (
    run_foreground_patch_selection,
    save_summary_metrics,
)


def predict_eval(
    config: Dict[str, Any],
):
    predict(config)
    eval_config = EvaluateConfig.model_validate(config["evaluation"])
    _ = run_performance_evaluation(eval_config)

    # save summary metrics
    summary_config = SummaryResultsConfig.model_validate(config["summary_results"])
    if isinstance(summary_config.filter_patches, ForegroundFilterConfig):
        _ = run_foreground_patch_selection(summary_config)
    save_summary_metrics(summary_config)


if __name__ == "__main__":
    cfg, _ = load_config()
    predict_eval(cfg)
