import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Tuple, Mapping


def plot_perturbation_sweep(
    consis_scores: NDArray[Any],
    perf_scores: NDArray[Any],
    x_label: str,
    y_label: str,
    title: str,
    transfer_labels: List[str],
    perturbation_labels: List[str],
    invert_consis_metric: bool = True,
    invert_perf_metric: bool = False,
    style: str = "default",
    figsize: tuple[int, int] = (12, 9),
    legend_size: int = 22,
    fontsize: int = 28,
    xlim: Tuple[float, float] = (0.59, 1),
    ylim: Tuple[float, float] = (0.3, 1),
    point_size: int = 500,
    fit_lines: bool = True,
    grid_alpha: float = 0.6,
    point_alpha: float = 0.6,
    line_alpha: float = 0.8,
    line_width: int = 2,
    legend1_position: tuple[float, float] = (0, 1),
    legend2_position: tuple[float, float] = (0, 0.7),
    save_fig_path: str = "",
    colors: List[str] = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ],
):
    # Set the matplotlib style
    plt.style.use(style)

    # Define markers
    markers = ["o", "s", "x", "D", "^", "v", "p", "*"]  # Markers for each column of A

    # Invert the metrics if specified
    if invert_consis_metric:
        consis_scores = 1 - consis_scores
    if invert_perf_metric:
        perf_scores = 1 - perf_scores

    # Initialize the figure
    _, ax = plt.subplots(figsize=figsize)

    # Loop over each column in A
    for col_idx in range(consis_scores.shape[1]):
        for row_idx in range(consis_scores.shape[0]):
            _ = ax.scatter(
                consis_scores[row_idx, col_idx],
                perf_scores[row_idx],
                color=colors[row_idx],
                marker=markers[col_idx],
                s=point_size,
                label=f"",  # Suppress auto-labeling
                alpha=point_alpha,
            )
        if fit_lines:
            # Fit a linear regression line
            slope, intercept = np.polyfit(consis_scores[:, col_idx], perf_scores, 1)
            x = np.linspace(xlim[0], xlim[1], 100)
            y = slope * x + intercept
            _ = ax.plot(
                x,
                y,
                color="gray",
                linestyle="--",
                alpha=line_alpha,
                linewidth=line_width,
            )

    # Custom legends
    color_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=10,
            label=f"{transfer_labels[i]}",
        )
        for i, c in enumerate(colors[: len(transfer_labels)])
    ]
    marker_legend = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="k",
            markersize=10,
            linestyle="None",
            label=f"{perturbation_labels[i]}",
        )
        for i, m in enumerate(markers[: len(perturbation_labels)])
    ]

    # Add the legends to the axis
    first_legend = ax.legend(
        handles=color_legend,
        title="Transfer",
        title_fontsize=legend_size,
        loc="upper left",
        bbox_to_anchor=legend1_position,
        prop={"size": legend_size},
    )
    _ = ax.add_artist(first_legend)
    _ = ax.legend(
        handles=marker_legend,
        title="Perturbation strength",
        title_fontsize=legend_size,
        loc="upper left",
        bbox_to_anchor=legend2_position,
        prop={"size": legend_size},
    )

    # Add labels and title
    _ = ax.set_xlabel(x_label, fontsize=fontsize)
    _ = ax.set_ylabel(y_label, fontsize=fontsize)
    _ = ax.set_title(title, fontsize=fontsize + 5)

    # Show the plot

    plt.grid(alpha=grid_alpha)
    _ = plt.xlim(xlim)
    _ = plt.ylim(ylim)
    _ = plt.xticks(fontsize=fontsize - 5)  # pyright: ignore[reportUnknownVariableType]
    _ = plt.yticks(fontsize=fontsize - 5)  # pyright: ignore[reportUnknownVariableType]
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()


def plot_multitarget_correlations(
    consis_scores: Mapping[str, NDArray[Any]],
    NA_perf_scores: Mapping[str, NDArray[Any]],
    save_path: str = "",
    title: str = "",
    perf_metric_name: str = "MAP score",
    consis_metric_name: str = "CTE",
    invert_consis_metric: bool = True,
    invert_perf_metric: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    fit_line: bool = True,
    xlim: Tuple[float, float] = (0.19, 1.05),
    ylim: Tuple[float, float] = (0, 1),
    point_size: int = 500,
    legend_size: int = 28,
    fontsize: int = 30,
    colors: List[str] = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ],
):

    _ = plt.figure(figsize=figsize)
    for i, (target, consis) in enumerate(consis_scores.items()):
        NA_perf = NA_perf_scores[target]
        assert (
            consis.shape == NA_perf.shape
        ), "Consistency and performance scores must have the same shape"
        if invert_consis_metric:
            consis = 1 - consis

        if invert_perf_metric:
            NA_perf = 1 - NA_perf

        _ = plt.scatter(
            consis,
            NA_perf,
            c=colors[i],
            label=f"To {target}",
            s=point_size,
            alpha=0.6,
        )
        if fit_line:
            # plot dotted line that is fitted to data
            x = np.linspace(
                np.min(consis) - 0.01,
                np.max(consis) + 0.01,
                100,
            )
            y = np.poly1d(np.polyfit(consis, NA_perf, 1))(x)
            _ = plt.plot(x, y, "--", c=colors[i], alpha=1, linewidth=3)
    plt.grid(alpha=0.6)
    _ = plt.legend(
        prop={"size": legend_size}, title="Target dataset", title_fontsize=legend_size
    )
    _ = plt.xlim(xlim)
    _ = plt.ylim(ylim)
    _ = plt.xlabel(consis_metric_name, fontsize=fontsize)
    _ = plt.ylabel(perf_metric_name, fontsize=fontsize)
    _ = plt.xticks(fontsize=fontsize - 5)  # pyright: ignore[reportUnknownVariableType]
    _ = plt.yticks(fontsize=fontsize - 5)  # pyright: ignore[reportUnknownVariableType]
    _ = plt.title(title, fontsize=fontsize + 5)
    if len(save_path) > 0:
        plt.savefig(save_path)

    plt.show()
