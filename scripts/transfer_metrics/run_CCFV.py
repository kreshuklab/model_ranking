import typer
from typing import Annotated
from model_ranking.configs import (
    generate_ccfv_yaml,
)
from model_ranking.baseline_metrics import (
    run_ccfv_evaluation,
)
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to the meta configuration file")],
):
    yaml_paths = generate_ccfv_yaml(config)
    for transfer_title, yaml_path in yaml_paths.items():
        print(f"Running CCFV evaluation for {transfer_title} using config {yaml_path}")
        ccfv_config, _ = load_config_direct(yaml_path)
        run_ccfv_evaluation(ccfv_config)


if __name__ == "__main__":
    typer.run(main)
