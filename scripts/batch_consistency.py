from typing import Annotated
import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.consistency import (
    run_consistency_evaluation,
)
from model_ranking.dataclass import ConsistencyConfig
from model_ranking.yaml_generators import generate_yaml


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    run_config_paths = generate_yaml(config)

    for transfer_title, config_paths in run_config_paths.items():
        print(f"Running transfer {transfer_title}")
        for config_path in config_paths:
            cfg, _ = load_config_direct(config_path)

            if "none" in str(config_path.stem):
                print(f"Skipping consistency evaluation for {config_path.stem}")
            else:
                consis_config = ConsistencyConfig.model_validate(cfg["consistency"])
                _, _ = run_consistency_evaluation(consis_config)


if __name__ == "__main__":
    typer.run(main)
