import pandas as pd
from typing import Dict, Optional, Union, List


def dataframe_to_latex_table(
    correlation_df: Union[pd.DataFrame, List[pd.DataFrame]],
    metric_name: Union[str, List[str]],
    metric_name_line2: Optional[Union[str, List[str]]] = None,
) -> str:
    """
    Convert correlation dataframe(s) to a LaTeX table with multi-level columns.

    Parameters:
    -----------
    correlation_df : pd.DataFrame or List[pd.DataFrame]
        Single DataFrame or list of DataFrames with columns: 'kt', 'kt pval', 's rho', 's rho pval', 'pr', 'pr pval'
        and index containing target dataset names
    metric_name : str or List[str]
        Name(s) of the metric(s) to be used in the first column (first line)
        If list, must match length of correlation_df list
    metric_name_line2 : str, List[str], or None, optional
        Second line(s) of the metric name(s). If provided, metric name will span two lines.
        If list, must match length of correlation_df list

    Returns:
    --------
    str
        LaTeX table string

    # Example usage:
    # Single correlation table with single-line metric name:
    # latex_table = dataframe_to_latex_table(correlation_df, "Consistency (a005-a007)")
    #
    # Single correlation table with two-line metric name:
    # latex_table = dataframe_to_latex_table(correlation_df, "CTE-EI", "(a005-a007)")
    #
    # Multiple correlation tables:
    # latex_table = dataframe_to_latex_table(
    #     correlation_df=[corr_df1, corr_df2, corr_df3],
    #     metric_name=["CTE-EI", "CTE-EI", "CTE-EI"],
    #     metric_name_line2=["(a001-a003)", "(a003-a005)", "(a005-a007)"]
    # )
    # print(latex_table)
    """
    # Convert single inputs to lists for uniform processing
    if isinstance(correlation_df, pd.DataFrame):
        correlation_dfs = [correlation_df]
    else:
        correlation_dfs = correlation_df

    if isinstance(metric_name, str):
        metric_names = [metric_name]
    else:
        metric_names = metric_name

    if metric_name_line2 is None:
        metric_names_line2 = [None] * len(correlation_dfs)
    elif isinstance(metric_name_line2, str):
        metric_names_line2 = [metric_name_line2]
    else:
        metric_names_line2 = metric_name_line2

    # Validate input lengths
    if not (len(correlation_dfs) == len(metric_names) == len(metric_names_line2)):
        raise ValueError(
            "correlation_df, metric_name, and metric_name_line2 must have the same length"
        )

    # Get target names from the first dataframe
    first_df = correlation_dfs[0]
    if isinstance(first_df.index, pd.MultiIndex):
        targets = first_df.index.get_level_values(-1).tolist()
    else:
        targets = first_df.index.tolist()

    # Number of targets
    n_targets = len(targets)

    # Build the LaTeX table
    latex_lines: List[str] = []
    latex_lines.append(r"\begin{table*}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\setlength{\tabcolsep}{3pt}"
    )  # Reduce column separation (default is 6pt)

    # Create tabular column specification with vertical lines: cc|ccc|ccc|ccc|ccc
    col_spec = "cc|" + "|".join(["ccc"] * n_targets)
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\hline")

    # First header row: Target names spanning 3 columns each
    header1 = r"\multirow{2}{*}{Metric} & {}"
    for i, target in enumerate(targets):
        if i < n_targets - 1:
            header1 += f" & \\multicolumn{{3}}{{c|}}{{{target}}}"
        else:
            # Last column doesn't need trailing pipe
            header1 += f" & \\multicolumn{{3}}{{c}}{{{target}}}"
    header1 += r" \\"
    latex_lines.append(header1)

    # Second header row: Kτ, Sρ, Pr for each target
    header2 = " & "
    for _ in targets:
        header2 += r" & K$\tau$ & S$\rho$ & P$r$"
    header2 += r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"\hline")

    # Process each correlation dataframe
    for _, (corr_df, name1, name2) in enumerate(
        zip(correlation_dfs, metric_names, metric_names_line2)
    ):
        # Format metric name with optional second line
        if name2:
            # Use a tabular environment for centered multi-line text
            formatted_metric = (
                f"\\begin{{tabular}}{{@{{}}c@{{}}}}{name1} \\\\ {name2}\\end{{tabular}}"
            )
        else:
            formatted_metric = name1

        # Data rows: metric name with correlation values and p-values
        # First row: correlation values with "Avg." label
        corr_row = f"\\multirow{{2}}{{*}}{{{formatted_metric}}} & "
        for i in range(n_targets):
            kt_val = corr_df.iloc[i]["kt"]
            sp_val = corr_df.iloc[i]["s rho"]
            pr_val = corr_df.iloc[i]["pr"]
            corr_row += f" & {kt_val:.2f} & {sp_val:.2f} & {pr_val:.2f}"
        corr_row += r" \\"
        latex_lines.append(corr_row)

        # Second row: p-values with "pval." label
        pval_row = " & \\textbf{pval.}"
        for i in range(n_targets):
            kt_pval = corr_df.iloc[i]["kt pval"]
            sp_pval = corr_df.iloc[i]["s rho pval"]
            pr_pval = corr_df.iloc[i]["pr pval"]
            pval_row += f" & ({kt_pval:.2f}) & ({sp_pval:.2f}) & ({pr_pval:.2f})"
        pval_row += r" \\"
        latex_lines.append(pval_row)

        # Add horizontal line after each metric (except potentially the last one)
        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular}")

    # Create caption using all metric names
    if len(metric_names) == 1:
        caption_text = (
            f"{metric_names[0]} {metric_names_line2[0]}"
            if metric_names_line2[0]
            else metric_names[0]
        )
    else:
        caption_text = "Transfer metric correlations"

    latex_lines.append(
        r"\caption{Correlation scores and p-values for " + caption_text + r"}"
    )
    label_text = (
        caption_text.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )
    latex_lines.append(r"\label{tab:" + label_text + r"}")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)


def avg_correlation_to_latex(
    df_avg_list: Union[pd.DataFrame, List[pd.DataFrame]],
    metric_name_list: Union[str, List[str]],
    metric_spec_list: Optional[Union[str, List[str], List[None]]] = None,
):
    """
    Generate a LaTeX table from a list of average correlation dataframes.

    Parameters
    ----------
    df_avg_list : pd.DataFrame or list of pd.DataFrame
        Single dataframe or list of average correlation dataframes with columns:
        KT_avg, KT_std, SR_avg, SR_std, PR_avg, PR_std and index with task names.
    metric_name_list : str or list of str
        Single metric name or list of metric names (e.g., "CTE-EI")
    metric_spec_list : str, list of str, or None, optional
        Single spec, list of specifications, or None (e.g., "a01-a012")

    Returns
    -------
    str
        LaTeX table string
    """
    # Normalize inputs to lists
    if not isinstance(df_avg_list, list):
        df_avg_list = [df_avg_list]
    if not isinstance(metric_name_list, list):
        metric_name_list = [metric_name_list]
    if metric_spec_list is None:
        metric_spec_list = [None] * len(df_avg_list)
    elif not isinstance(metric_spec_list, list):
        metric_spec_list = [metric_spec_list]

    # Build data structure: metric_key -> task -> values
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    tasks: List[str] = []

    for df_avg, metric_name, metric_spec in zip(
        df_avg_list, metric_name_list, metric_spec_list
    ):
        # Get task name
        task = df_avg.index[0]
        if task not in tasks:
            tasks.append(task)

        # Create metric key
        if metric_spec:
            metric_key = f"{metric_name}|||{metric_spec}"
        else:
            metric_key = metric_name

        # Store values
        if metric_key not in data:
            data[metric_key] = {}

        data[metric_key][task] = {
            "kt_avg": df_avg["KT_avg"].iloc[0],
            "kt_std": df_avg["KT_std"].iloc[0],
            "sr_avg": df_avg["SR_avg"].iloc[0],
            "sr_std": df_avg["SR_std"].iloc[0],
            "pr_avg": df_avg["PR_avg"].iloc[0],
            "pr_std": df_avg["PR_std"].iloc[0],
        }

    # Calculate table dimensions
    num_tasks = len(tasks)
    num_metrics = len(data)

    # Build column specification with dividers between tasks
    col_spec = "cc|"
    for i in range(num_tasks):
        if i < num_tasks - 1:
            col_spec += "ccc|"  # Add divider after each task except the last
        else:
            col_spec += "ccc"  # No divider after the last task

    # Build the LaTeX table
    latex = (
        r"""\begin{table}[htbp]
    \centering
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{"""
        + col_spec
        + r"""}
    \hline
    """
    )

    # Header row 1: Task names
    header1 = "\\multirow{2}{*}{Metric} & {}"
    for i, task in enumerate(tasks):
        if i < num_tasks - 1:
            header1 += f" & \\multicolumn{{3}}{{c|}}{{{task}}}"  # Add divider after multicolumn
        else:
            header1 += (
                f" & \\multicolumn{{3}}{{c}}{{{task}}}"  # No divider for last task
            )
    header1 += " \\\\\n"
    latex += header1

    # Header row 2: Correlation types
    header2 = " & "
    for i in range(num_tasks):
        if i < num_tasks - 1:
            header2 += " & K$\\tau$ & S$\\rho$ & P$r$"
        else:
            header2 += " & K$\\tau$ & S$\\rho$ & P$r$"
    header2 += " \\\\\n"
    latex += header2
    latex += r"\hline" + "\n"

    # Data rows
    for metric_key in data.keys():
        # Parse metric key
        if "|||" in metric_key:
            metric_name, metric_spec = metric_key.split("|||")
            metric_display = f"\\multirow{{2}}{{*}}{{\\begin{{tabular}}{{@{{}}c@{{}}}} {metric_name} \\\\ ({metric_spec})\\end{{tabular}}}}"
        else:
            metric_name = metric_key
            metric_spec = None
            metric_display = f"\\multirow{{2}}{{*}}{{{metric_name}}}"

        # Average row
        avg_row = f"{metric_display} & \\textbf{{Avg.}}"
        for task in tasks:
            if task in data[metric_key]:
                vals = data[metric_key][task]
                avg_row += f" & {vals['kt_avg']:.2f} & {vals['sr_avg']:.2f} & {vals['pr_avg']:.2f}"
            else:
                avg_row += " & - & - & -"
        avg_row += " \\\\\n"
        latex += avg_row

        # Std row
        std_row = " & \\textbf{std.}"
        for task in tasks:
            if task in data[metric_key]:
                vals = data[metric_key][task]
                std_row += f" & ±{vals['kt_std']:.2f} & ±{vals['sr_std']:.2f} & ±{vals['pr_std']:.2f}"
            else:
                std_row += " & - & - & -"
        std_row += " \\\\\n"
        latex += std_row

        latex += r"\hline" + "\n"

    # Caption and label
    if num_metrics == 1 and len(metric_name_list) == 1:
        # Single metric
        if metric_spec_list[0]:
            caption = f"Correlation scores and p-values for {metric_name_list[0]} ({metric_spec_list[0]})"
            label = f"{metric_name_list[0].lower().replace('-', '_')}_{metric_spec_list[0].replace('-', '_')}"
        else:
            caption = f"Correlation scores and p-values for {metric_name_list[0]}"
            label = f"{metric_name_list[0].lower().replace('-', '_')}"
    else:
        # Multiple metrics
        caption = "Correlation scores for multiple metrics"
        label = "multiple_metrics"

    latex += r"\end{tabular}" + "\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{tab:{label}}}\n"
    latex += r"\end{table}"

    return latex
