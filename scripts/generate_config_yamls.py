from typing import Annotated
import typer

from model_ranking.yaml_generators import generate_run_yamls


def main(
    config: Annotated[str, typer.Option(help="Path to Meta config file", exists=True)],
):
    _ = generate_run_yamls(config)


if __name__ == "__main__":
    typer.run(main)
