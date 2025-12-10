import pandas as pd
from typing import Dict, Optional, Union, List


def dataframe_to_latex_table_styled(
    correlation_df: Union[pd.DataFrame, List[pd.DataFrame]],
    metric_name: Union[str, List[str]],
    metric_name_line2: Optional[Union[str, List[str]]] = None,
    metric_name_line3: Optional[Union[str, List[str]]] = None,
    caption: str = "Correlation table",
    label: str = "correlation_table",
) -> str:
    """
    Convert correlation dataframe(s) to a styled LaTeX table with alternating column shading.

    This function creates a publication-ready table with:
    - Alternating column shading (requires GreyTable color defined in LaTeX document)
    - Special p-value formatting: (**) for p<0.01, (*) for p<0.05, otherwise value to 2dp
    - Bold correlation values when p-value is significant (p<0.05)
    - Each metric as a multirow with 3 sub-rows for K𝜏, S𝜌, P𝑟
    - Each target as a multicolumn with 2 sub-columns for correlation and p-value

    Parameters:
    -----------
    correlation_df : pd.DataFrame or List[pd.DataFrame]
        Single DataFrame or list of DataFrames with columns: 'kt', 'kt pval', 's rho', 's rho pval', 'pr', 'pr pval'
        and index containing target dataset names
    metric_name : str or List[str]
        Name(s) of the metric(s) to be used in the first column (first line)
        If list, must match length of correlation_df list
    metric_name_line2 : str, List[str], or None, optional
        Second line(s) of the metric name(s) in italics. If provided, metric name will span two lines.
        If list, must match length of correlation_df list
    metric_name_line3 : str, List[str], or None, optional
        Third line(s) of the metric name(s) in italics. If provided, metric name will span three lines.
        Requires metric_name_line2 to also be provided.
        If list, must match length of correlation_df list
    caption : str, optional
        Table caption (default: "Correlation table")
    label : str, optional
        Table label for referencing (default: "correlation_table")

    Returns:
    --------
    str
        LaTeX table string with styled format

    Example usage:
    --------------
    # Two-line metric names:
    metric_names = ['CTE-EI', 'CTE-NHD', 'CTE-EI', 'CTE-NHD']
    names2 = ['Gauss', 'Gauss', 'DO', 'DO']
    correlation_dfs = [df_gauss_EI, df_gauss_NHD, df_DO_EI, df_DO_NHD]
    latex_table = dataframe_to_latex_table_styled(
        correlation_dfs, metric_names, names2,
        caption="Transfer metric correlations",
        label="tab:styled_correlations"
    )

    # Three-line metric names:
    metric_names = ['CTE-EI', 'CTE-NHD']
    names2 = ['Gauss', 'Gauss']
    names3 = ['(a01-a012)', '(a012-a015)']
    correlation_dfs = [df_gauss_EI, df_gauss_NHD]
    latex_table = dataframe_to_latex_table_styled(
        correlation_dfs, metric_names, names2, names3,
        caption="Transfer metric correlations with 3-line names",
        label="tab:styled_correlations_3line"
    )
    print(latex_table)
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

    if metric_name_line3 is None:
        metric_names_line3 = [None] * len(correlation_dfs)
    elif isinstance(metric_name_line3, str):
        metric_names_line3 = [metric_name_line3]
    else:
        metric_names_line3 = metric_name_line3

    # Validate input lengths
    if not (
        len(correlation_dfs)
        == len(metric_names)
        == len(metric_names_line2)
        == len(metric_names_line3)
    ):
        raise ValueError(
            "correlation_df, metric_name, metric_name_line2, and metric_name_line3 must have the same length"
        )

    # Get target names from the first dataframe
    first_df = correlation_dfs[0]
    if isinstance(first_df.index, pd.MultiIndex):
        targets = first_df.index.get_level_values(-1).tolist()
    else:
        targets = first_df.index.tolist()

    # Number of targets
    n_targets = len(targets)

    def format_pval(pval: float) -> str:
        """Format p-value according to significance level."""
        if pval < 0.01:
            return "(**)"
        elif pval < 0.05:
            return "(*)"
        else:
            return f"({pval:.2f})"

    def format_corr_value(corr: float, pval: float) -> str:
        """Format correlation value, bold if significant."""
        return f"{corr:.2f}"

    # Build the LaTeX table
    latex_lines: List[str] = []
    latex_lines.append(r"\begin{table}[tb]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\scriptsize")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(r"\setlength{\tabcolsep}{3pt}")

    # Create tabular column specification with alternating shading
    # Pattern: c c >{\columncolor{GreyTable}}c >{\columncolor{GreyTable}}c c c ...
    col_spec = "@{}c c"
    for i in range(n_targets):
        if i % 2 == 0:
            # Even targets get grey shading
            col_spec += (
                "\n   >{\\columncolor{GreyTable}}c\n   >{\\columncolor{GreyTable}}c"
            )
        else:
            # Odd targets no shading
            col_spec += "\n   c c"
    col_spec += "@{}"

    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\toprule")

    # First header row: Target names spanning 2 columns each
    header1 = r"\multirow{2}{*}{Metric} & {}"
    for target in targets:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{target}}}"
    header1 += r" \\"
    latex_lines.append(header1)

    # Second header row: blank spaces and italic pval. for each target
    header2 = "& "
    for i in range(n_targets):
        if i % 2 == 0:
            # Grey columns get SecondaryColumnColor for the pval cell
            header2 += r" &  \cellcolor{SecondaryColumnColor} & \cellcolor{SecondaryColumnColor}\textit{pval.}"
        else:
            # Non-grey columns
            header2 += r" &  & \textit{pval.}"
    header2 += r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"\cmidrule{1-" + str(2 + 2 * n_targets) + "}")

    # Process each correlation dataframe
    for _, (corr_df, name1, name2, name3) in enumerate(
        zip(correlation_dfs, metric_names, metric_names_line2, metric_names_line3)
    ):
        # Format metric name with optional second and third lines
        if name3:
            # Three-line metric name with second and third lines in italics
            formatted_metric = f"\\multirow{{3}}{{*}}{{\\begin{{tabular}}{{@{{}}c@{{}}}}{name1} \\\\ \\textit{{{name2}}} \\\\ \\textit{{{name3}}}\\end{{tabular}}}}"
        elif name2:
            # Two-line metric name with second line in italics
            formatted_metric = f"\\multirow{{3}}{{*}}{{\\begin{{tabular}}{{@{{}}c@{{}}}}{name1} \\\\ \\textit{{{name2}}}\\end{{tabular}}}}"
        else:
            # Single-line metric name
            formatted_metric = f"\\multirow{{3}}{{*}}{{{name1}}}"

        # Row 1: K𝜏 values
        kt_row = f"{formatted_metric} & K$\\tau$"
        for i in range(n_targets):
            kt_val = corr_df.iloc[i]["kt"]
            kt_pval = corr_df.iloc[i]["kt pval"]
            kt_row += (
                f" & {format_corr_value(kt_val, kt_pval)} & {format_pval(kt_pval)}"
            )
        kt_row += r" \\"
        latex_lines.append(kt_row)

        # Row 2: S𝜌 values
        sp_row = "& S$\\rho$"
        for i in range(n_targets):
            sp_val = corr_df.iloc[i]["s rho"]
            sp_pval = corr_df.iloc[i]["s rho pval"]
            sp_row += (
                f" & {format_corr_value(sp_val, sp_pval)} & {format_pval(sp_pval)}"
            )
        sp_row += r" \\"
        latex_lines.append(sp_row)

        # Row 3: P𝑟 values
        pr_row = "& P$r$"
        for i in range(n_targets):
            pr_val = corr_df.iloc[i]["pr"]
            pr_pval = corr_df.iloc[i]["pr pval"]
            pr_row += (
                f" & {format_corr_value(pr_val, pr_pval)} & {format_pval(pr_pval)}"
            )
        pr_row += r" \\"
        latex_lines.append(pr_row)

        # Add horizontal line after each metric
        latex_lines.append(r"\cmidrule{1-" + str(2 + 2 * n_targets) + "}")

    # Replace last cmidrule with bottomrule
    latex_lines[-1] = r"\bottomrule"

    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\vspace{-12pt}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def dataframe_to_latex_table_swapped(
    correlation_df: Union[pd.DataFrame, List[pd.DataFrame]],
    metric_name: Union[str, List[str]],
    metric_name_line2: Optional[Union[str, List[str]]] = None,
) -> str:
    """
    Convert correlation dataframe(s) to a LaTeX table with swapped format.

    In this format:
    - Each metric becomes a multirow with 3 sub-rows for K𝜏, S𝜌, P𝑟
    - Each target becomes a multicolumn with 2 sub-columns for Avg. (correlation) and pval.

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
        LaTeX table string with swapped format

    Example usage:
    --------------
    metric_names = ['CTE-EI', 'CTE-NHD', 'CTE-EI', 'CTE-NHD']
    names2 = ['Gauss', 'Gauss', 'DO', 'DO']
    correlation_dfs = [df_gauss_EI, df_gauss_NHD, df_DO_EI, df_DO_NHD]
    latex_table = dataframe_to_latex_table_swapped(correlation_dfs, metric_names, names2)
    print(latex_table)
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

    # Create tabular column specification: cc|cc|cc|cc|cc
    # First column for Metric, second for correlation type (K𝜏, S𝜌, P𝑟), then 2 columns per target
    col_spec = "cc|" + "|".join(["cc"] * n_targets)
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\hline")

    # First header row: Target names spanning 2 columns each
    header1 = r"\multirow{2}{*}{Metric} & {}"
    for i, target in enumerate(targets):
        if i < n_targets - 1:
            header1 += f" & \\multicolumn{{2}}{{c|}}{{{target}}}"
        else:
            # Last column doesn't need trailing pipe
            header1 += f" & \\multicolumn{{2}}{{c}}{{{target}}}"
    header1 += r" \\"
    latex_lines.append(header1)

    # Second header row: blank space and pval. for each target
    header2 = " & "
    for _ in targets:
        header2 += r" &  & \textbf{pval.}"
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
            # Metric name spans 3 rows (for K𝜏, S𝜌, P𝑟)
            formatted_metric = f"\\multirow{{3}}{{*}}{{\\begin{{tabular}}{{@{{}}c@{{}}}}{name1} \\\\ {name2}\\end{{tabular}}}}"
        else:
            formatted_metric = f"\\multirow{{3}}{{*}}{{{name1}}}"

        # Row 1: K𝜏 values
        kt_row = f"{formatted_metric} & K$\\tau$"
        for i in range(n_targets):
            kt_val = corr_df.iloc[i]["kt"]
            kt_pval = corr_df.iloc[i]["kt pval"]
            kt_row += f" & {kt_val:.2f} & ({kt_pval:.1f})"
        kt_row += r" \\"
        latex_lines.append(kt_row)

        # Row 2: S𝜌 values
        sp_row = " & S$\\rho$"
        for i in range(n_targets):
            sp_val = corr_df.iloc[i]["s rho"]
            sp_pval = corr_df.iloc[i]["s rho pval"]
            sp_row += f" & {sp_val:.2f} & ({sp_pval:.1f})"
        sp_row += r" \\"
        latex_lines.append(sp_row)

        # Row 3: P𝑟 values
        pr_row = " & P$r$"
        for i in range(n_targets):
            pr_val = corr_df.iloc[i]["pr"]
            pr_pval = corr_df.iloc[i]["pr pval"]
            pr_row += f" & {pr_val:.2f} & ({pr_pval:.1f})"
        pr_row += r" \\"
        latex_lines.append(pr_row)

        # Add horizontal line after each metric
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
    latex_lines.append(r"\label{tab:" + label_text + r"_swapped}")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)


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
    \small
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
        avg_row = f"{metric_display} & \\textit{{Avg.}}"
        for task in tasks:
            if task in data[metric_key]:
                vals = data[metric_key][task]
                avg_row += f" & {vals['kt_avg']:.2f} & {vals['sr_avg']:.2f} & {vals['pr_avg']:.2f}"
            else:
                avg_row += " & - & - & -"
        avg_row += " \\\\\n"
        latex += avg_row

        # Std row
        std_row = " & \\textit{std.}"
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


def avg_correlation_to_latex_transposed(
    df_avg_list: Union[pd.DataFrame, List[pd.DataFrame]],
    metric_name_list: Union[str, List[str]],
    metric_spec_list: Optional[Union[str, List[str], List[None]]] = None,
):
    """
    Generate a transposed LaTeX table from a list of average correlation dataframes.

    In this transposed format:
    - Rows represent tasks (with K𝜏, S𝜌, P𝑟 as multirow subcategories)
    - Columns represent metrics (with Avg. and std. as subcolumns)

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
        LaTeX table string with transposed format
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

    # Build data structure: task -> metric_key -> values
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    metrics: List[str] = []

    for df_avg, metric_name, metric_spec in zip(
        df_avg_list, metric_name_list, metric_spec_list
    ):
        # Get task name
        task = df_avg.index[0]
        if task not in data:
            data[task] = {}

        # Create metric key
        if metric_spec:
            metric_key = f"{metric_name}|||{metric_spec}"
        else:
            metric_key = metric_name

        if metric_key not in metrics:
            metrics.append(metric_key)

        # Store values
        data[task][metric_key] = {
            "kt_avg": df_avg["KT_avg"].iloc[0],
            "kt_std": df_avg["KT_std"].iloc[0],
            "sr_avg": df_avg["SR_avg"].iloc[0],
            "sr_std": df_avg["SR_std"].iloc[0],
            "pr_avg": df_avg["PR_avg"].iloc[0],
            "pr_std": df_avg["PR_std"].iloc[0],
        }

    # Calculate table dimensions
    num_metrics = len(metrics)
    tasks = list(data.keys())

    # Build column specification: first column for task label, second for correlation type, then 2 columns per metric
    col_spec = "cc|" + "|".join(["cc"] * num_metrics)

    # Build the LaTeX table
    latex = (
        r"""\begin{table*}[htbp]
    \centering
    \setlength{\tabcolsep}{1.5pt}
    \begin{tabular}{"""
        + col_spec
        + r"""}
    \hline
    """
    )

    # Header row 1: "Task" label and Metric names spanning 2 columns each
    header1 = r"\multirow{2}{*}{Task} & "
    for i, metric_key in enumerate(metrics):
        # Parse metric key
        if "|||" in metric_key:
            metric_name, metric_spec = metric_key.split("|||")
            if metric_spec:
                metric_display = f"\\begin{{tabular}}{{@{{}}c@{{}}}} {metric_name} \\\\ ({metric_spec})\\end{{tabular}}"
            else:
                metric_display = metric_name
        else:
            metric_display = metric_key

        if i < num_metrics - 1:
            header1 += f" & \\multicolumn{{2}}{{c|}}{{{metric_display}}}"
        else:
            header1 += f" & \\multicolumn{{2}}{{c}}{{{metric_display}}}"
    header1 += " \\\\\n"
    latex += header1

    # Header row 2: Empty cell for Task row label, then Avg. and std. for each metric
    header2 = " & "
    for i in range(num_metrics):
        header2 += " & \\textit{Avg.} & \\textit{std.}"
    header2 += " \\\\\n"
    latex += header2
    latex += r"\hline" + "\n"

    # Data rows: one set of 3 rows per task
    for task in tasks:
        # Row 1: K𝜏 values with rotated task name
        kt_row = (
            f"\\multirow{{3}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{task}}}}} & K$\\tau$"
        )
        for metric_key in metrics:
            if metric_key in data[task]:
                vals = data[task][metric_key]
                kt_row += f" & {vals['kt_avg']:.2f} & ±{vals['kt_std']:.1f}"
            else:
                kt_row += " & - & -"
        kt_row += " \\\\\n"
        latex += kt_row

        # Row 2: S𝜌 values
        sr_row = " & S$\\rho$"
        for metric_key in metrics:
            if metric_key in data[task]:
                vals = data[task][metric_key]
                sr_row += f" & {vals['sr_avg']:.2f} & ±{vals['sr_std']:.1f}"
            else:
                sr_row += " & - & -"
        sr_row += " \\\\\n"
        latex += sr_row

        # Row 3: P𝑟 values
        pr_row = " & P$r$"
        for metric_key in metrics:
            if metric_key in data[task]:
                vals = data[task][metric_key]
                pr_row += f" & {vals['pr_avg']:.2f} & ±{vals['pr_std']:.1f}"
            else:
                pr_row += " & - & -"
        pr_row += " \\\\\n"
        latex += pr_row

        # Add horizontal line after each task
        latex += r"\hline" + "\n"

    # Caption and label
    if num_metrics == 1 and len(metric_name_list) == 1:
        # Single metric
        if metric_spec_list[0]:
            caption = f"Correlation scores and p-values for {metric_name_list[0]} ({metric_spec_list[0]})"
            label = f"{metric_name_list[0].lower().replace('-', '_')}_{metric_spec_list[0].replace('-', '_')}_transposed"
        else:
            caption = f"Correlation scores and p-values for {metric_name_list[0]}"
            label = f"{metric_name_list[0].lower().replace('-', '_')}_transposed"
    else:
        # Multiple metrics
        caption = "Correlation scores for multiple metrics (transposed)"
        label = "multiple_metrics_transposed"

    latex += r"\end{tabular}" + "\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{tab:{label}}}\n"
    latex += r"\end{table*}"

    return latex
