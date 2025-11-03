import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import random
from numpy.typing import NDArray
from scipy.special import logit  # pyright: ignore[reportMissingTypeStubs]
import seaborn as sns
from typing import Any, Dict, List, Sequence, Tuple, Mapping, Optional, Union

from model_ranking.results import load_transfer_metric_results
from model_ranking.utils import aug_name_to_sigma_tuple, add_decimal
from model_ranking.correlation import calculate_correlation_statistics
from model_ranking.dataclass import transferability_metric_names

MODEL_TO_DATASET = {
    "BC": "BBBC039",
    "DSB": "DSB2018",
    "GN": "Go-Nuclear",
    "HN": "HeLaNuc",
    "Hst": "Hoechst",
    "634": "S_BIAD634",
    "895": "S_BIAD895",
    "1196": "S_BIAD1196",
    "1410": "S_BIAD1410",
    "fw": "FlyWing",
    "ov": "Ovules",
    "p": "PNAS",
    "E": "EPFL",
    "Hm": "Hmito",
    "Rm": "Rmito",
    "V": "VNC",
    "H": "Hmito",
    "R": "Rmito",
}


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


def get_random_color_cmap(
    num_colors: int, cmap_name: str = "tab20", random_seed: int = 42
):
    # Generate colors using the specified colormap, and shuffle them
    cmap = plt.get_cmap(cmap_name)
    color_indices = np.linspace(0, 1, len(cmap.colors))  # pyright: ignore
    colors = [  # pyright: ignore[reportUnknownVariableType]
        cmap(ci) for ci in color_indices
    ]
    # use random color seed to get the same colors for each norm
    # use a random generator
    rng = random.Random(random_seed)
    rng.shuffle(colors)  # pyright: ignore[reportUnknownArgumentType]
    # select colors for each norm
    colors = colors[:num_colors]  # pyright: ignore[reportUnknownVariableType]

    # Use ListedColormap with the randomized colors
    cmap_custom = mcolors.ListedColormap(
        colors  # pyright: ignore[reportUnknownArgumentType]
    )
    cmap_norm = mcolors.BoundaryNorm(np.arange(num_colors + 1), cmap_custom.N)
    return cmap_custom, cmap_norm


def plot_alpha_sweep_specific_norm(
    consis_PA_str: Dict[str, Dict[str, Dict[str, NDArray[Any]]]],
    perf_scores: Union[
        Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, NDArray[Any]]]]
    ],
    augs: List[str],
    select_alphas: Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]],
    select_norms: Mapping[str, Sequence[Optional[str]]],
    per_transfer_aug: bool = False,
    per_transfer_norms: bool = True,
    single_target: Optional[str] = None,
    consistency_metric_name: str = "EI",
    transfer_metric_name: str = "F1",
    invert_consis_metric: bool = False,
    invert_perf_metric: bool = False,
    pos_correlation_line: bool = True,
    neg_correlation_line: bool = False,
    perf_logit_transformation: bool = False,
    fontsize: int = 18,
    n_rows: int = 2,
    n_cols: int = 2,
    figsize: Tuple[int, int] = (10, 10),
    per_aug_perf: bool = False,
    point_size: int = 100,
    legend_size: int = 12,
    transfers_selected: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (0.5, 1.0),
    ylim: Tuple[float, float] = (0, 1.0),
    cmap_name: str = "tab20",
    random_color_seed: int = 2,
    markers: List[str] = [
        "o",
        "s",
        "x",
        "d",
        "^",
        "v",
        ">",
        "<",
        "p",
        "P",
        "*",
        "h",
        "H",
        "+",
        "X",
        "D",
        "|",
    ],
):
    _, axs = plt.subplots(  # pyright: ignore[reportUnknownVariableType]
        n_rows, n_cols, figsize=figsize
    )
    if n_rows == 1 and n_cols == 1:
        axs = [axs]  # pyright: ignore[reportUnknownVariableType]
    else:
        axs = (  # pyright: ignore[reportUnknownVariableType]
            axs.flatten()  # pyright: ignore[reportAttributeAccessIssue]
        )

    # Get the list of transfers and create a color map
    if transfers_selected is not None:
        transfers = transfers_selected
    else:
        transfers = list(perf_scores.keys())

    cmap_custom, cmap_norm = get_random_color_cmap(
        len(transfers), cmap_name=cmap_name, random_seed=random_color_seed
    )
    if perf_logit_transformation == True:
        ylim = logit(ylim)

    for i, aug in enumerate(augs):
        for i2, transfer in enumerate(transfers):
            color = cmap_custom(i2)
            if per_transfer_aug:
                per_transfer_alphas = select_alphas[aug]
                assert isinstance(
                    per_transfer_alphas, dict
                ), "select_alphas[aug] must be a dict when per_transfer_aug is True"
                num_alphas = len(per_transfer_alphas[transfer])
            else:
                # Defensive: if select_alphas[aug] is a list, use it directly
                assert isinstance(
                    select_alphas[aug], list
                ), "select_alphas[aug] must be a list when per_transfer_aug is False"
                num_alphas = len(select_alphas[aug])
            for j in range(num_alphas):
                if per_transfer_norms:
                    norm = select_norms[transfer][0]
                elif single_target is not None:
                    norm = select_norms[single_target][0]
                else:
                    target = MODEL_TO_DATASET[transfer.split("_to_")[-1]]
                    norm = select_norms[target][0]
                if norm is None:
                    norm_name = "norm_Normalize"
                elif norm == "Normalize":
                    norm_name = "norm_Normalize"
                else:
                    norm_name = f"norm_{str(norm[0]).replace('.', '')}_{str(norm[1]).replace('.', '')}"
                if per_aug_perf:
                    per_aug_scores = perf_scores[transfer][norm_name]
                    assert isinstance(
                        per_aug_scores, dict
                    ), f"perf_scores[transfer][norm_name] must be a dict, got {type(perf_scores[transfer][norm_name])}"
                    perf = per_aug_scores[aug][j]
                else:
                    perf = perf_scores[transfer][norm_name]
                    assert isinstance(
                        perf, float
                    ), f"perf_scores[transfer][norm_name] must be a float, got {type(perf)}"
                if perf_logit_transformation:
                    perf = logit(perf)
                consis_score = consis_PA_str[transfer][norm_name][aug][j]
                if invert_consis_metric:
                    consis_score = 1 - consis_score
                if invert_perf_metric:
                    perf = 1 - perf
                _ = axs[i].scatter(  # pyright: ignore
                    consis_score,
                    perf,
                    marker=markers[j],
                    color=color,
                    s=point_size,
                )
        if per_transfer_aug == False:
            for k, alpha in enumerate(select_alphas[aug]):
                _ = axs[  # pyright: ignore[reportUnknownVariableType]
                    i
                ].scatter(  # pyright: ignore[reportAttributeAccessIssue]
                    [], [], marker=markers[k], label=alpha, color="black"
                )

        # Create the colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=cmap_norm, cmap=cmap_custom),
            ax=axs[i],  # pyright: ignore[reportUnknownArgumentType]
            boundaries=np.arange(len(transfers) + 1),
            ticks=np.arange(len(transfers)) + 0.5,
        )
        cbar.set_ticklabels(transfers, fontsize=fontsize - 5)
        cbar.set_label("Transfer", fontsize=fontsize)

        # Set labels, title, grid, and legend
        _ = axs[i].set_xlabel(  # pyright: ignore
            consistency_metric_name, fontsize=fontsize
        )

        if perf_logit_transformation == True:
            # Customize y-axis ticks
            yticks = np.linspace(0.01, 0.99, 10)  # F1 values to display on the y-axis
            _ = axs[i].set_yticks(logit(yticks))  # pyright: ignore
            _ = axs[i].set_yticklabels([f"{y:.2f}" for y in yticks])  # pyright: ignore

        if per_aug_perf:
            _ = axs[i].set_ylabel(  # pyright: ignore
                f"{transfer_metric_name} (per Aug)", fontsize=fontsize
            )
        else:
            _ = axs[i].set_ylabel(  # pyright: ignore
                f"{transfer_metric_name} (No Aug)", fontsize=fontsize
            )
        if single_target is not None:
            _ = axs[i].set_title(  # pyright: ignore
                f"Target: {single_target}, Aug sweep: {aug}", fontsize=fontsize
            )
        else:
            _ = axs[i].set_title(  # pyright: ignore
                f"Aug Sweep: {aug}", fontsize=fontsize
            )
        # check that both lines are not plotted

        assert not (
            pos_correlation_line and neg_correlation_line
        ), "Both pos and neg correlation lines cannot be plotted"
        if pos_correlation_line:
            _ = axs[i].plot(  # pyright: ignore
                xlim, ylim, "--", color="black", alpha=0.5
            )
        if neg_correlation_line:
            _ = axs[i].plot(  # pyright: ignore
                xlim, ylim[::-1], "--", color="black", alpha=0.5
            )
        _ = axs[i].set_xlim(xlim)  # pyright: ignore
        _ = axs[i].set_ylim(ylim)  # pyright: ignore
        _ = axs[i].grid()  # pyright: ignore

        lgnd = axs[i].legend(prop={"size": legend_size})  # pyright: ignore
        # for a in range(len(lgnd.legendHandles)):
        #    lgnd.legendHandles[a]._sizes = [point_size]
        _ = axs[i].tick_params(  # pyright: ignore
            axis="both", which="major", labelsize=fontsize - 2
        )
    plt.tight_layout()
    plt.show()


def plot_transfer_performance_heatmap(
    data: NDArray[Any],
    source_labels: List[str],
    target_labels: List[str],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    fontsize: int = 20,
    title_prefix: str = "",
):
    """
    Plots a heatmap for transfer performance.

    Parameters:
        data (np.ndarray): 2D array of performance scores.
        source_labels (list): Labels for the y-axis (source models).
        target_labels (list): Labels for the x-axis (target datasets).
        figsize (tuple): Figure size.
        cmap (str): Colormap for the heatmap.
    """
    _ = plt.figure(figsize=figsize)
    _ = sns.heatmap(
        data,
        xticklabels=target_labels,
        yticklabels=source_labels,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": "Performance Score"},
        annot_kws={"size": fontsize - 4},
    )
    _ = plt.xlabel("Target Dataset", fontsize=fontsize - 2)
    _ = plt.ylabel("Source Model", fontsize=fontsize - 2)
    _ = plt.title(f"{title_prefix} Transfer Performance Heatmap", fontsize=fontsize)
    _ = plt.tick_params(axis="both", which="major", labelsize=fontsize - 6)
    _ = plt.xticks(rotation=45)  # pyright: ignore[reportUnknownVariableType]
    _ = plt.yticks(rotation=0)  # pyright: ignore[reportUnknownVariableType]
    plt.tight_layout()
    plt.show()


def plot_performance_vs_transfer_metric(
    performance_scores: Dict[str, float],
    transfer_metrics: Dict[str, float],
    target: str,
    metric_name: str = "GBC",
    performance_metric_name: str = "F1",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    invert_perf_metric: bool = False,
    invert_transfer_metric: bool = False,
):
    """
    Plots a scatter plot of performance (F1 score) vs transfer metric for each model.

    Parameters:
    - performance_scores: dict, mapping model_name to F1_score
    - transfer_metrics: dict, mapping model_name to transfer_metric
    """
    model_names = list(performance_scores.keys())
    colors = plt.get_cmap("tab20", len(model_names))

    x: List[float] = []
    y: List[float] = []
    labels: List[str] = []
    for model in model_names:
        if model in transfer_metrics:
            trans_score = transfer_metrics[model]
            perf_score = performance_scores[model]
            if invert_transfer_metric:
                trans_score = 1 - trans_score
            if invert_perf_metric:
                perf_score = 1 - perf_score
            x.append(trans_score)
            y.append(perf_score)
            labels.append(model)

    f = plt.figure(figsize=(8, 6))
    for i, model in enumerate(labels):
        _ = plt.scatter(x[i], y[i], color=colors(i), label=model, s=80)

    _ = plt.xlabel(f"{metric_name} Score")
    _ = plt.ylabel(performance_metric_name)
    _ = plt.title(
        f"{target}: Performance ({performance_metric_name}) vs {metric_name} Score per Model"
    )
    _ = plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid()
    if show_plot:
        plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(f)
    return f


def plot_performance_vs_transfer_metric_multi_target(
    transfer_scores: Dict[str, Dict[str, float]],
    performance_scores: Dict[str, Dict[str, float]],
    transfer_metric: str,
    figsize: Tuple[int, int] = (16, 12),
    finetuned: bool = False,
    source_model_only: bool = False,
    invert_transfer_metric: bool = False,
    invert_perf_metric: bool = False,
    legend_bbox_anchor: Tuple[float, float] = (1.05, 1),
    performance_metric_key: str = "F1",
):
    # Create a 2x2 subplot figure for this augmentation
    _, axes = plt.subplots(  # pyright: ignore[reportUnknownVariableType]
        2, 2, figsize=figsize
    )
    axes = axes.flatten()  # pyright: ignore

    targets = list(transfer_scores.keys())
    print(f"Processing: {transfer_metric}")
    print(f"Number of targets: {len(targets)}")

    for i, (target, scores_per_model) in enumerate(transfer_scores.items()):
        if i >= 4:  # Only plot the first 4 targets
            break

        print(f"  Target {i+1}: {target}")

        # Get the data for plotting
        model_names = list(performance_scores[target].keys())

        if source_model_only:
            model_names = [
                m for m in model_names if MODEL_TO_DATASET[m.split("_")[0]] == target
            ]

        x: List[float] = []
        y: List[float] = []
        labels: List[str] = []

        for model in model_names:
            if model in scores_per_model:
                x.append(scores_per_model[model])
                y.append(performance_scores[target][model])
                labels.append(model)

        if invert_transfer_metric:
            x = [1 - val for val in x]
        if invert_perf_metric:
            y = [1 - val for val in y]

        # Plot on the specific subplot
        colors = plt.get_cmap("tab20", len(labels))
        for j, model in enumerate(labels):
            axes[i].scatter(x[j], y[j], color=colors(j), label=model, s=80)

        axes[i].set_xlabel(f"{transfer_metric}")
        axes[i].set_ylabel(f"{performance_metric_key} Score")
        axes[i].set_title(f"{target}: Performance vs {transfer_metric}")
        if (source_model_only == True) or (finetuned == True):
            axes[i].legend(
                title="Model", bbox_to_anchor=legend_bbox_anchor, loc="upper left"
            )
        axes[i].grid()

    # Hide any unused subplots
    for i in range(len(targets), 4):
        axes[i].set_visible(False)

    # Adjust layout and show
    performance_title = "Performance"
    if finetuned:
        performance_title += " (Finetuned)"
    _ = plt.suptitle(f"{performance_title} vs {transfer_metric}", fontsize=16)
    if (source_model_only == False) and (finetuned == False):
        _ = plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# Alternative function that creates separate figures for each target
def plot_model_performance_separate_figures(
    per_target_perf_results: Dict[str, Dict[str, Dict[str, float]]],
    figsize: Tuple[int, int] = (14, 8),
    save_plots: bool = False,
    save_dir: str = "./plots",
) -> None:
    """
    Plot separate multi-bar plots for each target dataset.

    Parameters:
    -----------
    per_target_perf_results : Dict[str, Dict[str, Dict[str, float]]]
        Performance results structured as {target: {model: {approach: score}}}
    figsize : tuple
        Figure size for each plot (width, height)
    save_plots : bool
        Whether to save plots to disk
    save_dir : str
        Directory to save plots if save_plots is True
    """
    # Create save directory if needed
    if save_plots:
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Get all unique approaches across all models and targets
    all_approaches: set[str] = set()
    for target_data in per_target_perf_results.values():
        for model_data in target_data.values():
            all_approaches.update(model_data.keys())
    approaches = sorted(list(all_approaches))

    # Color palette for approaches
    colors = plt.cm.Set3(range(len(approaches)))  # pyright: ignore
    approach_colors = dict(zip(approaches, colors))  # pyright: ignore

    for target, target_data in per_target_perf_results.items():
        _, ax = plt.subplots(figsize=figsize)

        # Prepare data for plotting
        models = list(target_data.keys())
        n_models = len(models)
        n_approaches = len(approaches)

        # Set up bar positions
        bar_width = 0.8 / n_approaches
        x_positions = range(n_models)

        # Plot bars for each approach
        for approach_idx, approach in enumerate(approaches):
            approach_scores = []
            for model in models:
                score = target_data[model].get(
                    approach, 0.0
                )  # Default to 0 if approach not available
                approach_scores.append(score)

            # Calculate bar positions for this approach
            bar_positions = [
                x + (approach_idx - n_approaches / 2 + 0.5) * bar_width
                for x in x_positions
            ]

            # Plot bars
            bars = ax.bar(
                bar_positions,
                approach_scores,  # pyright: ignore[reportUnknownArgumentType]
                bar_width,
                label=approach,
                color=approach_colors[
                    approach
                ],  # pyright: ignore[reportUnknownArgumentType]
                alpha=0.8,
            )

            # Add value labels on bars
            for bar, score in zip(bars, approach_scores):  # pyright: ignore
                if score > 0:  # Only label non-zero values
                    height = bar.get_height()  # pyright: ignore
                    _ = ax.text(
                        bar.get_x() + bar.get_width() / 2.0,  # pyright: ignore
                        height + 0.01,  # pyright: ignore
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        rotation=90,
                    )

        # Customize the plot
        _ = ax.set_xlabel("Models", fontsize=12)
        _ = ax.set_ylabel("Performance Score (F1)", fontsize=12)
        _ = ax.set_title(
            f"Model Performance on {target} Dataset", fontsize=14, fontweight="bold"
        )
        _ = ax.set_xticks(x_positions)
        _ = ax.set_xticklabels(models, rotation=45, ha="right")
        _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        _ = ax.grid(True, alpha=0.3)
        _ = ax.set_ylim(
            0,
            max([max(model_data.values()) for model_data in target_data.values()])
            * 1.15,
        )

        plt.tight_layout()

        if save_plots:
            plt.savefig(
                f"{save_dir}/model_performance_{target}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Plot saved to {save_dir}/model_performance_{target}.png")

        plt.show()


import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_single_consistency_vs_performance(
    augmentation_strength: str,
    target_dataset: str,
    performance_scores: Dict[str, Dict[str, float]],
    base_consistency_path: str = "/g/kreshuk/talks/consistency_results/patch_segmentation/mitochondria/transfer_results/consistency",
    custom_legend_labels: Optional[Dict[str, str]] = None,
    correlation_scores: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    marker_size: int = 200,
    fontsize: int = 16,
    cmap: str = "tab20",
):
    """
    Plot consistency scores vs performance scores for a single augmentation strength and target dataset.

    Parameters:
    -----------
    augmentation_strength : str
        The augmentation strength identifier (e.g., "a001-a003")
    target_dataset : str
        The target dataset name (e.g., "EPFL", "Hmito", "Rmito", "VNC")
    performance_scores : Dict
        Dictionary containing performance scores organized as {target: {model: score}}
    base_consistency_path : str
        Base path to the consistency scores JSON files
    custom_legend_labels : Optional[Dict[str, str]]
        Optional mapping from model names to custom legend labels
    correlation_scores : Optional[Dict[str, float]]
        Optional dictionary with correlation scores, e.g., {"kt": 0.95, "sp": 0.12, "p": 0.67}
        where "kt" is Kendall tau, "sp" is Spearman's ρ, and "p" is Pearson r
    figsize : tuple
        Figure size (width, height)
    alpha : float
        Transparency of the markers
    marker_size : int
        Size of the scatter plot markers

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Load consistency scores for the specified augmentation
    file_name = f"transfer_Gauss_{augmentation_strength}_CMB_05f_05b_EI_scores.json"
    consistency_path = Path(base_consistency_path) / file_name
    results = load_transfer_metric_results(consistency_path)
    consistency_scores = results["transfer_scores"]

    # Extract scores for the target dataset
    if target_dataset not in performance_scores:
        raise ValueError(
            f"Target dataset '{target_dataset}' not found in performance scores"
        )
    if target_dataset not in consistency_scores:
        raise ValueError(
            f"Target dataset '{target_dataset}' not found in consistency scores"
        )

    target_performance = performance_scores[target_dataset]
    target_consistency = consistency_scores[target_dataset]

    # Find common models between performance and consistency scores
    common_models: set[str] = set(target_performance.keys()) & set(
        target_consistency.keys()
    )
    if not common_models:
        raise ValueError(
            f"No common models found between performance and consistency scores for target '{target_dataset}'"
        )

    # Prepare data for plotting
    x_values: List[float] = []  # consistency scores
    y_values: List[float] = []  # performance scores
    labels: List[str] = []

    for model in sorted(common_models):
        x_values.append(target_consistency[model])
        y_values.append(target_performance[model])

        # Use custom label if provided, otherwise use model name
        if custom_legend_labels and model in custom_legend_labels:
            labels.append(custom_legend_labels[model])
        else:
            labels.append(model)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get the colormap
    colormap = plt.cm.get_cmap(cmap)

    # Create scatter plot with different colors for each model
    _ = ax.scatter(
        x_values, y_values, s=marker_size, alpha=alpha, c=range(len(labels)), cmap=cmap
    )

    sigmas = aug_name_to_sigma_tuple(augmentation_strength)
    # Add labels and title
    _ = ax.set_xlabel(f"Consistency Score", fontsize=fontsize)
    _ = ax.set_ylabel("Performance Score (F1)", fontsize=fontsize)
    _ = ax.set_title(
        f"Consistency vs Performance for {target_dataset}\n(Augmentation:  Gauss [{sigmas[0]}, {sigmas[1]}])",
        fontsize=fontsize,
    )

    # Add legend with matching colors
    for i, label in enumerate(labels):
        # Use the same colormap normalization as the scatter plot
        color = colormap(i / (len(labels) - 1) if len(labels) > 1 else 0)
        _ = ax.scatter([], [], c=[color], s=marker_size, alpha=alpha, label=label)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add correlation scores text if provided
    if correlation_scores:
        # Format correlation scores text
        corr_text = "Correlation Scores:\n"
        if "kt" in correlation_scores:
            corr_text += f"Kendall τ: {correlation_scores['kt']:.2f}\n"
        if "sp" in correlation_scores:
            corr_text += f"Spearman ρ: {correlation_scores['sp']:.2f}\n"
        if "p" in correlation_scores:
            corr_text += f"Pearson r: {correlation_scores['p']:.2f}"

        # Position the text below the legend
        legend_bbox = legend.get_window_extent(
            fig.canvas.get_renderer()  # pyright: ignore
        )
        # Convert to figure coordinates
        legend_bottom = legend_bbox.y0 / fig.bbox.height

        # Add text box below the legend
        _ = ax.text(
            1.05,
            legend_bottom - 0.05,
            corr_text,
            transform=ax.transAxes,
            fontsize=fontsize - 2,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add grid for better readability
    _ = ax.grid(True, alpha=0.8)

    # Tight layout to prevent legend cutoff
    plt.tight_layout()

    return fig, ax


def plot_single_consistency_vs_performance_CVPR(
    augmentation_strength: str,
    target_dataset: str,
    title: Optional[str] = None,
    make_title_bold: bool = True,
    performance_path: str = "/g/kreshuk/talks/consistency_results/patch_segmentation/mitochondria/transfer_results/transfer_performance_scores.json",
    base_consistency_path: str = "/g/kreshuk/talks/consistency_results/patch_segmentation/mitochondria/transfer_results/consistency",
    custom_legend_labels: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    marker_size: int = 200,
    fontsize: int = 16,
    invert_performance_score: bool = False,
    invert_consistency_score: bool = False,
    perturbation_type: str = "Gauss",
    file_name_postfix: str = "CMB_05f_05b_EI_scores",
    performance_score_key: str = "F1",
    source_abbreviations: List[str] = ["E", "Hm", "Rm", "V"],
    save_path: Optional[Union[str, Path]] = None,
    plot_style: str = "default",
    legend_loc: Optional[str] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (1.05, 1),
    corr_box_loc: Tuple[float, float] = (0.05, 0.05),
    tick_fontsize: Optional[int] = None,
    corr_fontsize: int = 18,
    legend_fontsize: int = 18,
):
    """
    Plot consistency scores vs performance scores for a single augmentation strength and target dataset.

    Parameters:
    -----------
    augmentation_strength : str
        The augmentation strength identifier (e.g., "a001-a003")
    target_dataset : str
        The target dataset name (e.g., "EPFL", "Hmito", "Rmito", "VNC")
    performance_path : str
        Path to the JSON file containing performance scores
    base_consistency_path : str
        Base path to the consistency scores JSON files
    custom_legend_labels : Optional[Dict[str, str]]
        Optional mapping from model names to custom legend labels
    figsize : tuple
        Figure size (width, height)
    alpha : float
        Transparency of the markers
    marker_size : int
        Size of the scatter plot markers
    fontsize : int
        Font size for labels, title, legend, and correlation scores
    invert_performance_score : bool
        If True, inverts performance scores (1 - score)
    invert_consistency_score : bool
        If True, inverts consistency scores (1 - score)
    perturbation_type : str
        Type of perturbation (e.g., "Gauss", "DO")
    file_name_postfix : str
        Postfix for the consistency scores filename
    performance_score_key : str
        Key for the performance metric (e.g., "F1", "MAP", "MSA")
    source_abbreviations : List[str]
        List of source abbreviations for grouping models
    save_path : Optional[Union[str, Path]]
        Path to save the figure. Both PNG (300 DPI) and SVG versions will be saved.
        If path includes extension, it will be replaced with .png and .svg.
        If no extension, .png and .svg will be appended.
        Example: "output/figure" or "output/figure.png" both save to
        "output/figure.png" and "output/figure.svg"
    plot_style : str
        Matplotlib style to use (e.g., "default", "seaborn-v0_8-colorblind")
    legend_loc : Optional[str]
        Legend location. If None, places legend outside the plot at bbox_to_anchor.
        Use standard matplotlib location strings like "upper left", "upper right", etc.
    legend_bbox_to_anchor : Optional[Tuple[float, float]]
        Custom bbox_to_anchor for legend. If None and legend_loc is None,
        defaults to (1.05, 1) for outside placement.
    corr_box_loc : Tuple[float, float]
        Location of correlation scores box in axes coordinates (x, y).
        Default is (0.05, 0.05) for lower left corner.
    tick_fontsize : Optional[int]
        Font size for x and y axis tick labels. If None, uses fontsize - 4.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set the plot style
    plt.style.use(plot_style)

    # Set tick fontsize if not provided
    if tick_fontsize is None:
        tick_fontsize = fontsize - 4

    # Load consistency scores for the specified augmentation
    file_name = (
        f"transfer_{perturbation_type}_{augmentation_strength}_{file_name_postfix}.json"
    )
    consistency_path = Path(base_consistency_path) / file_name
    results = load_transfer_metric_results(consistency_path)
    consistency_scores = results["transfer_scores"]

    performance_results = load_transfer_metric_results(performance_path)
    performance_scores = performance_results["performance_scores"]

    # Extract scores for the target dataset
    if target_dataset not in performance_scores:
        raise ValueError(
            f"Target dataset '{target_dataset}' not found in performance scores"
        )
    if target_dataset not in consistency_scores:
        raise ValueError(
            f"Target dataset '{target_dataset}' not found in consistency scores"
        )

    target_performance = performance_scores[target_dataset]
    target_consistency = consistency_scores[target_dataset]

    # Find common models between performance and consistency scores
    common_models: set[str] = set(target_performance.keys()) & set(
        target_consistency.keys()
    )
    if not common_models:
        raise ValueError(
            f"No common models found between performance and consistency scores for target '{target_dataset}'"
        )

    # Prepare data for plotting
    x_values: List[float] = []  # consistency scores
    y_values: List[float] = []  # performance scores
    labels: List[str] = []

    for model in sorted(common_models):
        x_values.append(target_consistency[model])
        y_values.append(target_performance[model])

        # Use custom label if provided, otherwise use model name
        if custom_legend_labels and model in custom_legend_labels:
            labels.append(custom_legend_labels[model])
        else:
            labels.append(model)

    if invert_consistency_score:
        x_values = [1 - val for val in x_values]
    if invert_performance_score:
        y_values = [1 - val for val in y_values]

    consis_array = np.array(x_values).reshape(-1, 1)  # Shape (n_models, 1)
    kt_scores, sp_scores, pearson_scores = calculate_correlation_statistics(
        consis_array, np.array(y_values)
    )
    pr = pearson_scores[0, 0]  # Pearson r
    sp = sp_scores[0, 0]  # Spearman rho
    kt = kt_scores[0, 0]  # Kendall tau

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Group models by source abbreviation and assign color families
    source_color_maps: Dict[str, Any] = {}
    color_maps = [  # pyright: ignore
        plt.cm.Blues,  # pyright: ignore
        plt.cm.Greens,  # pyright: ignore
        plt.cm.Oranges,  # pyright: ignore
        plt.cm.Purples,  # pyright: ignore
    ]
    for i, source_abbr in enumerate(source_abbreviations):
        source_color_maps[source_abbr] = color_maps[i]

    # Group models by their source abbreviation
    model_groups: Dict[str, List[int]] = {abbr: [] for abbr in source_abbreviations}
    for i, model in enumerate(sorted(common_models)):
        # Find which source abbreviation this model belongs to
        for abbr in source_abbreviations:
            if model.startswith(abbr + "_"):
                model_groups[abbr].append(i)
                break

    # Assign colors to each model based on their source group
    colors = []
    for i, model in enumerate(sorted(common_models)):
        # Find the source abbreviation for this model
        model_source = None
        for abbr in source_abbreviations:
            if model.startswith(abbr + "_"):
                model_source = abbr
                break

        if model_source and model_source in model_groups:
            # Get the index within the source group
            group_indices = model_groups[model_source]
            idx_in_group = group_indices.index(i)
            n_in_group = len(group_indices)

            # Generate color from the appropriate colormap
            # Use a range from 0.4 to 0.9 to avoid too light or too dark colors
            color_val = 0.4 + (0.5 * idx_in_group / max(n_in_group - 1, 1))
            colors.append(source_color_maps[model_source](color_val))
        else:
            # Fallback color if no source abbreviation matches
            colors.append("gray")

    # Create scatter plot with source-grouped colors
    _ = ax.scatter(
        x_values, y_values, s=marker_size, alpha=alpha, c=colors  # pyright: ignore
    )

    if perturbation_type == "DO":
        if augmentation_strength.startswith("a"):
            val = augmentation_strength[1:]
            pert_str: Union[float, Tuple[float, float]] = add_decimal(val)
        else:
            pert_str = add_decimal(augmentation_strength)

    else:
        pert_str = aug_name_to_sigma_tuple(augmentation_strength)
    # Add labels and title
    _ = ax.set_xlabel(f"CTE", fontsize=fontsize)
    _ = ax.set_ylabel(f"Performance Score ({performance_score_key})", fontsize=fontsize)

    if title is None:
        title = f"Consistency vs Performance for {target_dataset}\n{perturbation_type} {pert_str}"

    if make_title_bold:
        _ = ax.set_title(title, fontsize=fontsize, fontweight="bold")
    else:
        _ = ax.set_title(title, fontsize=fontsize)

    # Set tick label font sizes
    _ = ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # Add legend with matching colors
    legend_handles = []
    for i, label in enumerate(labels):
        # Use the color assigned to this model
        handle = ax.scatter(
            [],
            [],
            c=[colors[i]],  # pyright: ignore
            s=marker_size,
            alpha=alpha,
            label=label,
        )
        legend_handles.append(handle)

    # Configure legend position
    if legend_loc is not None:
        # Place legend inside the plot
        _ = ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    else:
        # Place legend outside the plot (default behavior)
        if legend_bbox_to_anchor is None:
            legend_bbox_to_anchor = (1.05, 1)

        _ = ax.legend(
            bbox_to_anchor=legend_bbox_to_anchor,
            loc="upper left",
            fontsize=legend_fontsize,
        )

    # Format correlation scores text with Greek symbols and abbreviations
    corr_text = "Correlation Scores:\n"
    corr_text += f"    K$\\tau$: {kt:.2f}\n"
    corr_text += f"    S$\\rho$: {sp:.2f}\n"
    corr_text += f"    Pr: {pr:.2f}"

    # Add correlation scores box inside the plot at specified location
    _ = ax.text(
        corr_box_loc[0],
        corr_box_loc[1],
        corr_text,
        transform=ax.transAxes,
        fontsize=corr_fontsize,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
            linestyle="dotted",
            linewidth=1.5,
        ),
    )

    # Add grid for better readability
    _ = ax.grid(True, alpha=0.8)

    # Tight layout to prevent legend cutoff
    fig.tight_layout()

    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Determine base path without extension
        if save_path_obj.suffix:
            # If path has extension, remove it to use as base
            base_path = save_path_obj.with_suffix("")
        else:
            # If no extension, use as-is
            base_path = save_path_obj

        # Save PNG version
        png_path = base_path.with_suffix(".png")
        try:
            fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
            print(f"Saved PNG figure to: {png_path}")
        except Exception as e:
            print(f"Error saving PNG: {e}")

        # Save SVG version
        svg_path = base_path.with_suffix(".svg")
        try:
            fig.savefig(svg_path, bbox_inches="tight", format="svg")
            print(f"Saved SVG figure to: {svg_path}")
        except Exception as e:
            print(f"Error saving SVG: {e}")

    plt.show()


def plot_cmb_classification_seg_transfer_metric_correlations_CVPR(
    classification_results_path: Path,
    segmentation_results_path: Path,
    transfer_metrics: Sequence[transferability_metric_names],
    nrows: int = 2,
    ncols: int = 4,
    classification_color: str = "blue",
    segmentation_color: str = "red",
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[Path] = None,
    point_size: int = 100,
    consistency_seg_results_path: Optional[Path] = None,
    consistency_class_results_path: Optional[Path] = None,
    consis_class_aug_str: Optional[str] = None,
    consis_seg_aug_str: Optional[str] = None,
    CTE_metric_key: Optional[str] = "EI",
):
    """
    Create combined classification and segmentation correlation plots per target dataset.

    Parameters:
    -----------
    classification_results_path : Path
        Base path to classification transfer metric results
    segmentation_results_path : Path
        Base path to segmentation transfer metric results
    transfer_metrics : list, optional
        List of transfer metrics to plot. If None, defaults to all standard metrics.
    nrows : int
        Number of rows in subplot grid
    ncols : int
        Number of columns in subplot grid
    classification_color : str
        Color for classification points
    segmentation_color : str
        Color for segmentation points
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Path, optional
        Path to save figures. If None, figures are displayed but not saved.

    Returns:
    --------
    None (creates and saves/displays figures)
    """

    # Sort transfer metrics alphabetically
    transfer_metrics = sorted(transfer_metrics)

    # Load all results
    classification_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    segmentation_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for metric in transfer_metrics:
        if "CTE" in metric:
            assert (
                consistency_class_results_path is not None
            ), "path to consistency classification results must be provided"
            assert (
                consistency_seg_results_path is not None
            ), "path to consistency segmentation results must be provided"
            assert (
                consis_class_aug_str is not None
            ), "selected classification aug strength must be provided"
            assert (
                consis_seg_aug_str is not None
            ), "selected seg aug strength must be provided"
            # Load classification results
            class_perf_path = (
                consistency_class_results_path / "transfer_performance_F1_scores.json"
            )
            class_perf_scores = load_transfer_metric_results(class_perf_path)
            class_consis_path = (
                consistency_class_results_path
                / f"transfer_Gauss_{consis_class_aug_str}_{CTE_metric_key}_scores.json"
            )
            class_consis_scores = load_transfer_metric_results(class_consis_path)
            classification_data[metric] = {**class_perf_scores, **class_consis_scores}

            # Load segmentation results
            seg_perf_path = (
                consistency_seg_results_path / "transfer_performance_F1_scores.json"
            )
            seg_perf_scores = load_transfer_metric_results(seg_perf_path)
            seg_consis_path = (
                consistency_seg_results_path
                / f"transfer_Gauss_{consis_seg_aug_str}_CMB_05f_05b_{CTE_metric_key}_scores.json"
            )
            seg_consis_scores = load_transfer_metric_results(seg_consis_path)
            segmentation_data[metric] = {**seg_perf_scores, **seg_consis_scores}

        else:
            # Load classification results
            class_path = classification_results_path / f"{metric}.json"
            classification_data[metric] = load_transfer_metric_results(class_path)

            # Load segmentation results
            seg_path = segmentation_results_path / f"mitochondria_{metric}.json"
            segmentation_data[metric] = load_transfer_metric_results(seg_path)

    # Get list of targets (assuming same targets for both tasks)
    targets: Sequence[str] = classification_data[  # pyright: ignore
        transfer_metrics[1]
    ]["metadata"]["targets"]

    # Create one figure per target
    for target in targets:
        fig, axes = plt.subplots(  # pyright: ignore
            nrows=nrows, ncols=ncols, figsize=figsize
        )
        axes = axes.flatten() if nrows * ncols > 1 else [axes]  # pyright: ignore

        # Plot each transfer metric in a subplot
        for idx, metric in enumerate(transfer_metrics):
            if idx >= len(axes):  # pyright: ignore
                break

            ax = axes[idx]  # pyright: ignore

            # Get classification data for this target and metric
            class_transfer_scores = classification_data[metric]["transfer_scores"][
                target
            ]
            class_performance_scores = classification_data[metric][
                "performance_scores"
            ][target]

            # Get segmentation data for this target and metric
            seg_transfer_scores = segmentation_data[metric]["transfer_scores"][target]
            seg_performance_scores = segmentation_data[metric]["performance_scores"][
                target
            ]

            # Convert to arrays for plotting
            class_x = np.array(list(class_transfer_scores.values()))
            class_y = np.array(list(class_performance_scores.values()))

            seg_x = np.array(list(seg_transfer_scores.values()))
            seg_y = np.array(list(seg_performance_scores.values()))

            # Plot classification points
            _ = ax.scatter(  # pyright: ignore
                class_x,
                class_y,
                marker="x",
                color=classification_color,
                s=point_size,
                linewidths=2,
                alpha=0.7,
                label="Classification" if idx == 0 else "",
            )

            # Plot segmentation points
            _ = ax.scatter(  # pyright: ignore
                seg_x,
                seg_y,
                marker="o",
                color=segmentation_color,
                s=point_size,
                alpha=0.7,
                label="Segmentation" if idx == 0 else "",
            )

            # Calculate correlation statistics for classification using calculate_correlation_statistics
            class_transfer_array = class_x.reshape(-1, 1)  # Shape (n_models, 1)
            class_kt_scores, class_sp_scores, class_pearson_scores = (
                calculate_correlation_statistics(class_transfer_array, class_y)
            )
            class_pr = class_pearson_scores[0, 0]  # Pearson r
            class_sp = class_sp_scores[0, 0]  # Spearman rho
            class_kt = class_kt_scores[0, 0]  # Kendall tau

            # Calculate correlation statistics for segmentation using calculate_correlation_statistics
            seg_transfer_array = seg_x.reshape(-1, 1)  # Shape (n_models, 1)
            seg_kt_scores, seg_sp_scores, seg_pearson_scores = (
                calculate_correlation_statistics(seg_transfer_array, seg_y)
            )
            seg_pr = seg_pearson_scores[0, 0]  # Pearson r
            seg_sp = seg_sp_scores[0, 0]  # Spearman rho
            seg_kt = seg_kt_scores[0, 0]  # Kendall tau

            # Add correlation statistics box in bottom right
            textstr = f"Classification:\n  Kτ: {class_kt:.2f}\n  ρ: {class_sp:.2f}\n  r: {class_pr:.2f}\n\n"
            textstr += f"Segmentation:\n  Kτ: {seg_kt:.2f}\n  ρ: {seg_sp:.2f}\n  r: {seg_pr:.2f}"

            props = dict(boxstyle="round", facecolor="white", alpha=0.5, linestyle="--")
            _ = ax.text(  # pyright: ignore
                0.95,
                0.05,
                textstr,
                transform=ax.transAxes,  # pyright: ignore
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=props,
            )

            # Set labels and title
            _ = ax.set_xlabel(f"{metric} Score", fontsize=12)  # pyright: ignore
            _ = ax.set_ylabel("Performance Score", fontsize=12)  # pyright: ignore
            _ = ax.set_title(metric, fontsize=14, fontweight="bold")  # pyright: ignore
            _ = ax.grid(True, alpha=0.5)  # pyright: ignore

        # Hide unused subplots
        for idx in range(len(transfer_metrics), len(axes)):  # pyright: ignore
            axes[idx].set_visible(False)  # pyright: ignore

        # Add a single shared legend
        handles, labels = axes[0].get_legend_handles_labels()  # pyright: ignore
        _ = fig.legend(
            handles,  # pyright: ignore
            labels,  # pyright: ignore
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            fontsize=14,
            frameon=True,
        )

        # Add overall title
        _ = fig.suptitle(
            f"Transfer Metrics vs Performance - Target: {target}",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # pyright: ignore

        # Save or show figure
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            fig_path = save_path / f"combined_metrics_{target}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {fig_path}")

        plt.show()

    return None
