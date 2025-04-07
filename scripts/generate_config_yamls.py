from typing import Annotated
import typer

from model_ranking.yaml_generators import generate_yaml


def main(
    config: Annotated[str, typer.Option(help="Path to Meta config file", exists=True)],
):
    _ = generate_yaml(config)


if __name__ == "__main__":
    typer.run(main)
