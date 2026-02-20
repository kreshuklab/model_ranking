from model_ranking.utils import subsample_data_split
import typer
from typing import Optional

def main(
    input_path: str = typer.Argument(..., help="Path to the .txt file containing filenames"),
    proportion: Optional[float] = typer.Option(None, help="Proportion of files to sample (e.g., 0.5 for 50%)"),
    n_samples: Optional[int] = typer.Option(None, help="Absolute number of files to sample"),
    output_path: Optional[str] = typer.Option(None, help="Output path to write selected filenames"),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    _ = subsample_data_split(
        input_path=input_path,
        proportion=proportion,
        n_samples=n_samples,
        output_path=output_path,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    typer.run(main)
