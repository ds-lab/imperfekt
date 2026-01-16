import polars as pl

from imperfekt.analysis.utils import pretty_printing, visualization_utils
from imperfekt.analysis.utils.kruskal_wallis import perform_statistical_analysis

############################################################
# Analyze Gap and Observation Lengths in Time Series Data  #
############################################################


def analyze_gap_lengths(
    mask_df: pl.DataFrame,
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
) -> pl.DataFrame:
    """
    Analyze gap lengths in a DataFrame with time series data.
    The function calculates the lengths of gaps by measuring the time between the previous and next observed values.
    If count_clock_no is 0 the distance between two adjacent observations has been considered as a gap.

    Parameters:
        mask_df (pl.DataFrame): Input DataFrame with time series data.
        cols (list): List of columns to analyze. If None, all columns except id_col, clock_col, and clock_no_col are used.
        id_col (str): Column name for the identifier (default: "id").
        clock_col (str): Column name for the clock time (default: "clock").
        clock_no_col (str): Column name for the clock number (default: "clock_no").

    Returns:
        pl.DataFrame: DataFrame with gap and observation lengths in seconds.
                      Contains columns: id, variable, count_clock_no, time_length, run_start_clock, run_end_clock.
    """
    # Identify value columns
    if cols is None:
        cols = mask_df.columns
    value_cols = [c for c in cols if c not in {id_col, clock_col, clock_no_col}]

    # Melt to long format: id, clock, clock_no, variable, value
    long_df = mask_df.unpivot(
        index=[id_col, clock_col, clock_no_col],
        on=value_cols,
        variable_name="variable",
        value_name="value",
    )

    # Sort by id, variable, clock_no
    long_df = long_df.sort([id_col, "variable", clock_no_col])

    # Filter out imperfect values
    observations_df = long_df.filter(pl.col("value") == 0)

    result = observations_df.with_columns(
        time_length=pl.col(clock_col).diff().over(id_col, "variable"),
        count_clock_no=(pl.col(clock_no_col).diff().over(id_col, "variable") - 1),
        run_start_clock=pl.col(clock_col).shift(1).over(id_col, "variable"),
        run_end_clock=pl.col(clock_col),
    )

    # Remove null fields (first observation)
    result = result.filter(pl.col("time_length").is_not_null())

    # Convert time lengths to seconds
    result = result.with_columns(pl.col("time_length").dt.total_seconds())

    # Replace time_length 0 with None
    result = result.with_columns(
        pl.when(pl.col("time_length") == 0)
        .then(None)
        .otherwise(pl.col("time_length"))
        .alias("time_length")
    )

    result = result.select(
        id_col,
        "variable",
        "count_clock_no",
        "time_length",
        "run_start_clock",
        "run_end_clock",
    )

    return result


def gap_lengths(
    lengths_df: pl.DataFrame,
    col: str,
    save_path: str = None,
    save_results: bool = True,
    plot_library: str = "matplotlib",
    renderer: str = "browser",
) -> tuple:
    """
    Visualize the gap and observation lengths in a DataFrame for a specific variable.

    Parameters:
        lengths_df (pl.DataFrame): DataFrame with gap and observation lengths.
        col (str): The variable to visualize.
        save_path (str): Path to save the visualizations. If None, visualizations will not be saved.
        save_results (bool): Whether to save the visualizations.
        plot_library (str): The plotting library to use for visualizations. Defaults to "matplotlib".
        renderer (str): Renderer for plotly visualizations, default is "browser".

    Returns:
        Visualizations of gap and observation lengths for the specified variable.
    """
    gaps = lengths_df.filter(pl.col("variable") == col)

    if renderer:
        pretty_printing.rich_info(f"Gap Lengths for {col}: {gaps.describe(interpolation='linear')}")

    gap_fig = visualization_utils.plot_violin(
        gaps,
        y="time_length",
        title=f"Gap Length Boxplot for {col}",
        yaxis_title="Gap Length (seconds)",
        library=plot_library,
        renderer=renderer,
        save_path=save_path / f"{col}_gap_length_boxplot.png" if save_results else None,
        save_results=save_results,
    )

    if save_path and save_results:
        gaps.describe(interpolation="linear").write_csv(
            save_path / f"{col}_gap_lengths_summary.csv"
        )

    return gaps, gap_fig


############################################################
#      Extract Gap Return Values from Time Series Data     #
############################################################


def extract_gap_return_values(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
) -> pl.DataFrame:
    """
    Identify gaps in a time series DataFrame and find the first observed value after each gap.
    Exploratory technique to understand how gaps in time series data relate to subsequent observations,
    useful for investigation MNAR (Missing Not At Random) patterns.

    Parameters:
        df (pl.DataFrame): Input DataFrame with time series data.
        mask_df (pl.DataFrame): Mask DataFrame indicating which values are imperfect (1=missing/noisy/indicated, 0=observed/normal).
        id_col (str): Column name for the identifier (default: "id").
        clock_col (str): Column name for the clock time (default: "clock").
        clock_no_col (str): Column name for the clock number (default: "clock_no").

    Returns:
        pl.DataFrame: DataFrame with gaps and their first observed values.
                      Contains columns: id, variable, run_id, return_time, return_value.
    """
    if cols is None:
        cols = [c for c in df.columns if c not in {id_col, clock_col, clock_no_col}]
    # Get Gap Runs
    gaps = analyze_gap_lengths(
        mask_df, cols, id_col=id_col, clock_col=clock_col, clock_no_col=clock_no_col
    )

    # For each gap, find the next observed row in the original df, unpivot so we have variable and value again.
    long = df.unpivot(
        index=[id_col, clock_col, clock_no_col],
        on=cols,
        variable_name="variable",
        value_name="value",
    ).sort([id_col, "variable", clock_col])

    # Join to find the return value after each gap
    joined = (
        gaps.join(long, on=[id_col, "variable"], how="left")
        .filter(pl.col(clock_col) == pl.col("run_end_clock"))  # Get match for return value
        .rename({clock_col: "return_time", "value": "return_value"})
    )

    return joined


def gap_returns(
    spans: pl.DataFrame,
    col: str = None,
    bins: list = None,
    plot_library: str = "matplotlib",
    renderer: str = "browser",
    save_path: str = None,
    save_results: bool = True,
) -> tuple:
    """
    Analyze the pattern mixture of gaps in a time series DataFrame.

    Parameters:
        spans (pl.DataFrame): DataFrame with gaps and their first observed values.
                              Should contain columns: id, variable, run_id, return_time, return_value.
        col (str): Column name for the gap and return value to analyze. If None, an error will be raised.
        bins (list): List of bin edges for categorizing gap lengths. If None, 0.125 quantiles will be used as bins.
        renderer (str): Renderer for visualizations, default is "browser".

    Returns:
        pl.DataFrame: Summary DataFrame with mean and standard deviation of return values,
                      and count of spans for each gap bin.
    """
    if col is None:
        raise ValueError("Column name for return value must be specified.")
    else:
        spans = spans.filter(pl.col("variable") == col)

    if bins is None:
        quantiles = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        bins = [
            spans.select(pl.col("time_length").quantile(q, interpolation="nearest")).to_series()[0]
            for q in quantiles
        ]
        bins = sorted(set(bins))  # Remove duplicates and sort

    labels = (
        [f"-inf - {bins[0]}"]
        + [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
        + [f"{bins[-1]} - inf"]
    )
    spans = spans.with_columns(
        pl.col("time_length").cut(breaks=bins, labels=labels).alias("gap_bin")
    )

    spans = spans.filter(pl.col("gap_bin").is_not_null(), pl.col("return_value").is_not_null())

    if spans.height > 0:
        gap_return_boxplot = visualization_utils.plot_boxplot(
            spans,
            y="return_value",
            x="gap_bin",
            title=f"Return Value by Gap Length Bin for {col}",
            yaxis_title=f"Return Value {col}",
            xaxis_title="Gap Length Bin (in seconds)",
            library=plot_library,
            renderer=renderer,
            category_order=labels,
            save_path=save_path / f"{col}_gap_return_boxplot.png"
            if save_path and save_results
            else None,
            save_results=save_results,
        )
    else:
        gap_return_boxplot = None

    kw_result, pval_heatmap_fig, es_heatmap_fig = perform_statistical_analysis(
        spans.to_pandas(),
        c="return_value",
        group_col="gap_bin",
        posthoc_method="dscf",
        renderer=renderer,
        save_path=save_path,
        save_results=save_results,
        analyzed_col=col,
    )

    summary_df = (
        spans.group_by("gap_bin")
        .agg(
            [
                pl.col("return_value").mean().alias("mean_return"),
                pl.col("return_value").std().alias("sd_return"),
                pl.col("return_value").median().alias("median_return"),
                pl.col("return_value").min().alias("min_return"),
                pl.col("return_value").max().alias("max_return"),
                pl.len().alias("n_spans"),
            ]
        )
        .sort("gap_bin")
    )

    if save_path and save_results:
        summary_df.write_csv(save_path / f"{col}_gap_return_summary.csv")
        print(f"Gap return summary saved to {save_path / f'{col}_gap_return_summary.csv'}")

    return kw_result, gap_return_boxplot, pval_heatmap_fig, es_heatmap_fig


if __name__ == "__main__":
    pl.Config.set_tbl_cols(25)
    pl.Config.set_tbl_rows(20)
    df = pl.DataFrame(
        [
            ("1", "2023-01-01 13:15:00", 120.0, 15.0, None, 90.0, 0),
            ("1", "2023-01-02 13:15:40", 130.0, 16.0, None, 90.0, 1),
            ("2023_225617845", "2023-01-02 13:15:41", None, 16.0, 180.0, 90.0, 0),
            ("2023_225617845", "2023-01-02 13:15:48", 129.0, None, None, 86.0, 1),
            ("2023_225617845", "2023-01-02 13:17:50", 38.0, None, None, 96.0, 2),
            ("2023_225617845", "2023-01-02 13:19:57", None, None, None, None, 3),
            ("2023_225617845", "2023-01-02 13:20:47", 121.0, None, 193.0, 90.0, 4),
            ("2023_225617845", "2023-01-02 13:22:19", None, None, None, None, 5),
            ("2023_225617845", "2023-01-02 13:22:50", 120.0, None, None, 96.0, 6),
        ],
        schema=["id", "clock", "hr", "dbp", "sbp", "o2sat", "clock_no"],
        orient="row",
    )
    df = df.with_columns(
        [
            pl.col("clock").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )
    mask_df = df.with_columns(
        [
            pl.col("sbp").is_null().cast(pl.Int8).alias("sbp"),
            pl.col("dbp").is_null().cast(pl.Int8).alias("dbp"),
            pl.col("hr").is_null().cast(pl.Int8).alias("hr"),
            pl.col("o2sat").is_null().cast(pl.Int8).alias("o2sat"),
        ]
    )

    result = analyze_gap_lengths(mask_df)
    print(result.filter(pl.col("variable") == "hr"))  #
    summary = gap_lengths(result, col="hr", save_path=None, save_results=False)
    print(summary)

    result = extract_gap_return_values(df, mask_df)
    print(result)
    kw_result, gap_return_boxplot, pval_heatmap_fig, es_heatmap_fig = gap_returns(result, col="dbp")
    print(kw_result)
