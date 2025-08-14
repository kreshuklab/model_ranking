from typing import List, Dict, Any

from model_ranking.dataclass import (
    MetaConfig,
    EvaluateConfig,
    ForegroundFilterConfig,
    SummaryResultsConfig,
)
from model_ranking.evaluation import run_performance_evaluation
from model_ranking.results import (
    run_foreground_patch_selection,
    save_summary_metrics,
)
from model_ranking.yaml_generators import generate_run_yamls

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.predict import (
    predict,  # pyright: ignore[reportUnknownVariableType]
)


def batch_predict_checkpoints(
    meta_config: MetaConfig,
    checkpoint_names: List[str] = ["best"],  # Comma-separated list of checkpoints
):

    # Run predictions using the specified configuration and checkpoints.

    for checkpoint in checkpoint_names:
        # Update the checkpoint in the configuration
        for source_model in meta_config.source_models:
            source_model.checkpoint_name = checkpoint
        meta_config.output_settings.output_folder = checkpoint
        pred_yaml_paths = generate_run_yamls(meta_config.model_dump())
        for transfer in pred_yaml_paths.keys():
            for pred_yaml_path in pred_yaml_paths[transfer]:
                pred_cfg, _ = load_config_direct(pred_yaml_path)
                print(f"Running prediction with config: {pred_yaml_path}")
                predict_eval(pred_cfg)


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
