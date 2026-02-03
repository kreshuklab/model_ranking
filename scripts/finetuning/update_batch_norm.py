from typing import Annotated
import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.data_structures import (
    MetaConfig,
    AdaptiveBatchNormConfig,
)
from model_ranking.configs import (
    generate_run_yamls,
)
from model_ranking.finetuning import (
    run_adaptive_batchnorm,
    run_sequential_adaptive_batchnorm,
    copy_checkpoint_with_updated_model,
)


def main(
    config: Annotated[
        str, typer.Option(help="Path to the meta configuration file", exists=True)
    ],
    sequential: bool = typer.Option(False, help="Use sequential BN adaptation"),
):
    cfg, _ = load_config_direct(config)

    assert (
        cfg["run_mode"] == "adaptive_batchnorm"
    ), f"Current Run mode = {cfg['run_mode']}, should be 'adaptive_batchnorm'"

    pred_cfg = MetaConfig.model_validate(cfg)

    pred_yaml_paths = generate_run_yamls(pred_cfg.model_dump())

    for transfer_title, config_paths in pred_yaml_paths.items():
        print(f"Running transfer {transfer_title}")
        for config_path in config_paths:
            cfg, _ = load_config_direct(config_path)
            adabn_cfg = AdaptiveBatchNormConfig.model_validate(cfg)

            assert (
                adabn_cfg.model_cfg.source_checkpoint is not None
            ), "Source checkpoint must be specified for adaptive batch norm"

            if sequential == True:
                print("Running sequential adaptive batch norm...")
                updated_model = run_sequential_adaptive_batchnorm(adabn_cfg)
            else:
                print("Running standard adaptive batch norm...")
                updated_model = run_adaptive_batchnorm(adabn_cfg)

            copy_checkpoint_with_updated_model(
                source_checkpoint_path=adabn_cfg.model_cfg.source_checkpoint,
                new_checkpoint_dir_path=adabn_cfg.output_checkpoint_dir_path,
                updated_model=updated_model,
            )


if __name__ == "__main__":
    typer.run(main)
