from pathlib import Path
import typer


from model_ranking.classification import (
    run_classification_prediction,
)


def main(
    config: Path = typer.Option(
        ..., help="Path to config yaml or directory of config yamls"
    ),
    batch: bool = typer.Option(
        False, help="If set, config is a directory of config yamls"
    ),
):
    if batch:
        config_files = sorted(Path(config).glob("*.yaml"))
        for config_path in config_files:
            print(f"Running prediction for config: {config_path}")
            run_classification_prediction(config_path)
    else:
        run_classification_prediction(config)


if __name__ == "__main__":
    typer.run(main)
