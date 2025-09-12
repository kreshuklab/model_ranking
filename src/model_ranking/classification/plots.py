import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_model_scores(
    results: Dict[str, List[float]], target: str, model_names: List[str]
):
    """
    Plots a multi-bar plot for model scores.

    Args:
        results (Dict[str, List[float]]): Dictionary of score names to lists of values.
        target (str): Target dataset name (used for title).
        model_names (List[str]): List of model names (x-axis).
    """
    score_names = list(results.keys())
    n_scores = len(score_names)
    n_models = len(model_names)
    x = np.arange(n_models)
    bar_width = 0.8 / n_scores

    _, ax = plt.subplots(figsize=(10, 6))

    for i, score in enumerate(score_names):
        values = results[score]
        if score == "accuracy_error":
            values = [1 - v for v in values]
            score = "accuracy"
        _ = ax.bar(x + i * bar_width, values, width=bar_width, label=score)

    _ = ax.set_xticks(x + bar_width * (n_scores - 1) / 2)
    _ = ax.set_xticklabels(model_names, rotation=90)
    _ = ax.set_ylabel("Score Value")
    _ = ax.set_title(f"Model Scores for Target: {target}")
    _ = ax.legend(title="Score Type")
    plt.tight_layout()
    plt.show()
