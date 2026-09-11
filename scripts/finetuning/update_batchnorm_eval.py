from pathlib import Path
from typing import Annotated, Any, cast

import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.predict import (
    predict,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.data_structures import (
    MetaConfig,
    AdaptiveBatchNormConfig,
    EvaluateConfig,
    ConsistencyConfig,
    ForegroundFilterConfig,
    SummaryResultsConfig,
)
from model_ranking.consistency import run_consistency_evaluation
from model_ranking.configs import (
    generate_run_yamls,
)
from model_ranking.evaluation import run_performance_evaluation
from model_ranking.finetuning import (
    run_adaptive_batchnorm,
    run_sequential_adaptive_batchnorm,
    run_adaptive_batchnorm_momentum,
    copy_checkpoint_with_updated_model,
)
from model_ranking.results import (
    run_foreground_patch_selection,
    save_summary_metrics,
)


def _get_adapted_model_name(adabn_config_path: Path) -> str:
    return adabn_config_path.parents[4].name


def _run_adaptive_batchnorm_config(
    config_path: Path,
    sequential: bool,
    momentum: bool,
) -> None:
    cfg, _ = load_config_direct(config_path) 
    cfg = cast(dict[str, Any], cfg)
    adabn_cfg = AdaptiveBatchNormConfig.model_validate(cfg)

    assert (
        adabn_cfg.model_cfg.source_checkpoint is not None
    ), "Source checkpoint must be specified for adaptive batch norm"

    if sequential:
        print("Running sequential adaptive batch norm...")
        updated_model = run_sequential_adaptive_batchnorm(adabn_cfg)
    elif momentum:
        print("Running per batch momentum adaptive batch norm...")
        updated_model = run_adaptive_batchnorm_momentum(adabn_cfg)
    else:

        updated_model = run_adaptive_batchnorm(adabn_cfg)

    _ = copy_checkpoint_with_updated_model(
        source_checkpoint_path=adabn_cfg.model_cfg.source_checkpoint,
        new_checkpoint_dir_path=adabn_cfg.output_checkpoint_dir_path,
        updated_model=updated_model,
    )


def _run_prediction_and_evaluation_config(
    config_path: Path,
) -> None:
    cfg, _ = load_config_direct(config_path)  
    predict(cfg) 

    eval_config = EvaluateConfig.model_validate(cfg["evaluation"])
    _ = run_performance_evaluation(eval_config)

    if "consistency" in cfg:
        consis_config = ConsistencyConfig.model_validate(cfg["consistency"])
        _, _ = run_consistency_evaluation(consis_config)

    summary_config = SummaryResultsConfig.model_validate(cfg["summary_results"])
    if isinstance(summary_config.filter_patches, ForegroundFilterConfig):
        _ = run_foreground_patch_selection(summary_config)
    save_summary_metrics(summary_config)


def _get_eval_meta_config(
    config_dict: dict[str, Any],
    source_name: str,
    target_name: str,
    adapted_model_name: str,
) -> dict[str, Any]:
    eval_config: dict[str, Any] = config_dict.copy()
    eval_config["run_mode"] = "pred_eval"
    eval_config["model_dir_path"] = config_dict["output_settings"]["base_dir_path"]
    eval_config["target_datasets"] = [
        target
        for target in config_dict["target_datasets"]
        if target["name"] == target_name
    ]
    eval_config["source_models"] = [
        source.copy()
        for source in config_dict["source_models"]
        if source["source_name"] == source_name
    ]
    assert (
        len(eval_config["target_datasets"]) == 1
    ), f"Expected exactly one target dataset named {target_name}"
    assert (
        len(eval_config["source_models"]) == 1
    ), f"Expected exactly one source model for {source_name}"
    eval_config["source_models"][0]["model_name"] = adapted_model_name

    if "eval_output_settings" in config_dict:
        eval_config["output_settings"] = config_dict["eval_output_settings"]

    return eval_config


def main(
    config: Annotated[
        Path, typer.Option(help="Path to the meta configuration file", exists=True)
    ],
    sequential: bool = typer.Option(False, help="Use sequential BN adaptation"),
    momentum: bool = typer.Option(False, help="Use per batch momentum adaptation"),
) -> None:
    assert not (
        sequential and momentum
    ), "Cannot use both --sequential and --momentum at the same time"

    cfg, _ = load_config_direct(config)  
    cfg = cast(dict[str, Any], cfg)

    assert (
        cfg["run_mode"] in ["adaptive_batchnorm", "adabn_eval", "pred_eval"]
    ), f"Current Run mode = {cfg['run_mode']}, should be 'adaptive_batchnorm' or 'adabn_eval'"

    adabn_meta_cfg: dict[str, Any] = cfg.copy()
    adabn_meta_cfg["run_mode"] = "adaptive_batchnorm"
    adabn_meta_cfg["eval_settings"] = None
    adabn_meta_cfg["consistency_settings"] = None

    pred_cfg = MetaConfig.model_validate(adabn_meta_cfg)
    pred_yaml_paths = generate_run_yamls(pred_cfg.model_dump())

    for transfer_title, config_paths in pred_yaml_paths.items():
        print(f"Running transfer {transfer_title}")
        for config_path in config_paths:
            _run_adaptive_batchnorm_config(config_path, sequential, momentum)

    if cfg["run_mode"] not in ["adabn_eval", "pred_eval"]:
        return

    for transfer_title, adabn_config_paths in pred_yaml_paths.items():
        source_name, target_name = transfer_title.split("_to_", maxsplit=1)
        adapted_model_name = _get_adapted_model_name(adabn_config_paths[0])
        eval_meta_cfg = _get_eval_meta_config(
            cfg,
            source_name=source_name,
            target_name=target_name,
            adapted_model_name=adapted_model_name,
        )
        eval_cfg = MetaConfig.model_validate(eval_meta_cfg)
        eval_yaml_paths = generate_run_yamls(eval_cfg.model_dump())

        for eval_transfer_title, config_paths in eval_yaml_paths.items():
            print(f"Evaluating transfer {eval_transfer_title}")
            for config_path in config_paths:
                _run_prediction_and_evaluation_config(config_path)


if __name__ == "__main__":
    typer.run(main)
