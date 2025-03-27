from typing import Annotated
import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.predict import predict  # pyright: ignore[reportUnknownVariableType]

from model_ranking.consistency import calc_segmentation_model_consistency
from model_ranking.dataclass import EvaluateConfig
from model_ranking.evaluation import run_evaluation
from model_ranking.utils import check_for_no_aug_configs
from model_ranking.yaml_generators import generate_yaml


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    run_config_paths = generate_yaml(config)

    for transfer_title, config_paths in run_config_paths.items():
        for config_path in config_paths:
            cfg, _ = load_config_direct(config_path)
            predict(cfg)
            eval_config = EvaluateConfig.model_validate(cfg["evaluation"])
            _ = run_evaluation(eval_config)

        config_paths_with_NA = check_for_no_aug_configs(
            source_dataset=transfer_title.split("_")[0],
            target_dataset=transfer_title.split("_")[-1],
            configs=config_paths,
        )
        calc_segmentation_model_consistency(config_paths_with_NA)


if __name__ == "__main__":
    typer.run(main)
