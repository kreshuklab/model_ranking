import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import random
from numpy.typing import NDArray
from scipy.special import logit  # pyright: ignore[reportMissingTypeStubs]
import seaborn as sns
from typing import Any, Dict, List, Sequence, Tuple, Mapping, Optional, Union

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
                else:
                    norm_name = f"norm_{norm[0]}_{str(norm[1]).replace('.', '')}"
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

        # Plot on the specific subplot
        colors = plt.get_cmap("tab20", len(labels))
        for j, model in enumerate(labels):
            axes[i].scatter(x[j], y[j], color=colors(j), label=model, s=80)

        axes[i].set_xlabel(f"{transfer_metric}")
        axes[i].set_ylabel("F1 Score")
        axes[i].set_title(f"{target}: Performance vs {transfer_metric}")
        if (source_model_only == True) or (finetuned == True):
            axes[i].legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
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
