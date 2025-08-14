import typer
from typing import Any, Dict
from numpy.typing import NDArray
import json

# from model_ranking.dataclass import transferability_metrics
from model_ranking.results import (
    load_transfer_metric_results,
    convert_numpy_types,
    find_transferability_results_path,
)
from model_ranking.correlation import (
    to_target_transfer_correlations,
)


def main(
    base_path: str = typer.Option(help="Path to base results ", exists=True),
    transfer_metric: str = typer.Option(help="Transferability metric to calculate"),
    task: str = typer.Option(default="mitochondria", help="Data task to use"),
    overwrite: bool = typer.Option(
        default=False, help="Overwrite existing correlation scores if they exist"
    ),
):
    path = find_transferability_results_path(
        base_path=base_path,
        transfer_metric=transfer_metric,
        data_task=task,
    )
    results = load_transfer_metric_results(path)
    transferability_scores = results["transfer_scores"]
    performance_scores = results["performance_scores"]

    correlation_scores: Dict[str, NDArray[Any]] = {}

    correlation_scores["KT"], correlation_scores["SP"], correlation_scores["PE"] = (
        to_target_transfer_correlations(
            list(transferability_scores.keys()),
            transferability_scores,
            performance_scores,
        )
    )

    # Check if correlation_scores already exist
    if "correlation_scores" in results and not overwrite:
        print("Correlation scores already exist in the results file.")
        print("Use --overwrite flag to overwrite existing correlation scores.")
        return

    # Add correlation scores to results
    results["correlation_scores"] = convert_numpy_types(correlation_scores)

    # Save updated results back to the file
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Correlation scores added to: {path}")
    print(f"Correlation scores: {list(correlation_scores.keys())}")


if __name__ == "__main__":
    typer.run(main)
