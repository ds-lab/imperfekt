from pathlib import Path

import polars as pl

from imperfekt.analysis.irregularity import burstiness as burstiness_module
from imperfekt.analysis.irregularity import interval_statistics as interval_statistics_module
from imperfekt.analysis.utils import pretty_printing, visualization_utils
from imperfekt.analysis.utils.kruskal_wallis import perform_statistical_analysis

############################################################
# Analyze Gap and Observation Lengths in Time Series Data  #
############################################################


def analyze_gap_lengths(
    mask_df: pl.DataFrame,
    cols: list | None = None,
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
    save_path: str | Path | None = None,
    save_results: bool = True,
    plot_library: str = "matplotlib",
    renderer: str | None = "browser",
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
    gaps = lengths_df.filter((pl.col("variable") == col) & (pl.col("count_clock_no") > 0))

    if gaps.is_empty():
        if renderer:
            pretty_printing.rich_info(
                f"Gap Lengths for {col}: no true gaps found (all observations are adjacent)."
            )
        return gaps, None

    if renderer:
        pretty_printing.rich_info(f"Gap Lengths for {col}: {gaps.describe(interpolation='linear')}")

    if type(save_path) is str:
        save_path: Path = Path(save_path)

    gap_fig = visualization_utils.plot_violin(
        gaps,
        y="time_length",
        title=f"Gap Length Boxplot for {col}",
        yaxis_title="Gap Length (seconds)",
        library=plot_library,
        renderer=renderer,
        save_path=save_path / f"{col}_gap_length_boxplot.png" if type(save_path) is Path else None,
        save_results=save_results,
    )

    if save_path and save_results:
        gaps.describe(interpolation="linear").write_csv(
            save_path / f"{col}_gap_lengths_summary.csv" if type(save_path) is Path else None
        )

    return gaps, gap_fig


############################################################
#      Extract Gap Return Values from Time Series Data     #
############################################################


def extract_gap_return_values(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    cols: list | None = None,
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
    col: str | None = None,
    bins: list | None = None,
    plot_library: str = "matplotlib",
    renderer: str | None = "browser",
    save_path: str | Path | None = None,
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
            if type(save_path) is Path
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
        save_path = save_path if type(save_path) is Path else Path(save_path)
        summary_df.write_csv(save_path / f"{col}_gap_return_summary.csv")
        print(f"Gap return summary saved to {save_path / f'{col}_gap_return_summary.csv'}")

    return kw_result, gap_return_boxplot, pval_heatmap_fig, es_heatmap_fig, bins


############################################################
#        Dominant Gap Length and Gap Burstiness            #
############################################################


def compute_gap_dominant_length(
    gaps_df: pl.DataFrame,
    bin_resolution_seconds: float = 60.0,
    adherence_tolerance: float = 0.5,
) -> pl.DataFrame:
    """
    Identify the modal gap length and quantify adherence and entropy.

    Delegates to interval_statistics_module.compute_dominant_frequency() after
    renaming 'time_length' → 'interval_seconds'. Null gap lengths are dropped.

    Parameters:
        gaps_df (pl.DataFrame): Rows for a single variable from gs_gaps_observation_runs,
                                must contain a 'time_length' column (seconds, non-null > 0).
        bin_resolution_seconds (float): Bin width in seconds for discretising gap lengths.
        adherence_tolerance (float): Fractional tolerance around the modal bin.

    Returns:
        pl.DataFrame: Single-row summary with columns:
            dominant_gap_seconds, dominant_gap_bin, gap_adherence_rate,
            n_total_gaps, n_adhering_gaps, gap_entropy_bits,
            gap_normalized_entropy, gap_n_unique_bins,
            bin_resolution_seconds, adherence_tolerance.
        Returns None if the input is empty after filtering nulls.
    """
    valid = gaps_df.filter(pl.col("time_length").is_not_null()).rename(
        {"time_length": "interval_seconds"}
    )
    if valid.is_empty():
        return pl.DataFrame(
            {
                "dominant_gap_seconds": [None],
                "dominant_gap_bin": [None],
                "gap_adherence_rate": [None],
                "n_total_gaps": [0],
                "n_adhering_gaps": [0],
                "gap_entropy_bits": [None],
                "gap_normalized_entropy": [None],
                "gap_n_unique_bins": [0],
                "bin_resolution_seconds": [bin_resolution_seconds],
                "adherence_tolerance": [adherence_tolerance],
            }
        )

    summary, _ = interval_statistics_module.compute_dominant_frequency(
        delta_t_df=valid,
        bin_resolution_seconds=bin_resolution_seconds,
        adherence_tolerance=adherence_tolerance,
    )

    return summary.rename(
        {
            "dominant_interval_seconds": "dominant_gap_seconds",
            "dominant_interval_bin": "dominant_gap_bin",
            "adherence_rate": "gap_adherence_rate",
            "n_total_intervals": "n_total_gaps",
            "n_adhering_intervals": "n_adhering_gaps",
            "interval_entropy_bits": "gap_entropy_bits",
            "normalized_entropy": "gap_normalized_entropy",
            "n_unique_bins": "gap_n_unique_bins",
        }
    )


def compute_gap_burstiness(
    gaps_df: pl.DataFrame,
    id_col: str = "id",
) -> pl.DataFrame:
    """
    Compute the global burstiness coefficient over all gap lengths for one variable.

    Delegates to burstiness_module.compute_global_burstiness() after renaming
    'time_length' → 'interval_seconds'. Null gap lengths are dropped.

    B = (std - mean) / (std + mean)  [Goh & Barabasi, 2008]
    Range [-1, 1]: -1 = perfectly regular, 0 = Poisson, >0 = bursty.

    Parameters:
        gaps_df (pl.DataFrame): Rows for a single variable from gs_gaps_observation_runs,
                                must contain 'time_length' and id_col columns.
        id_col (str): Entity identifier column name.

    Returns:
        pl.DataFrame: Single-row summary with columns:
            n_gaps, mean_gap_seconds, std_gap_seconds, gap_burstiness_coeff.
        Returns None if the input is empty after filtering nulls.
    """
    valid = gaps_df.filter(pl.col("time_length").is_not_null()).rename(
        {"time_length": "interval_seconds"}
    )
    if valid.is_empty():
        return pl.DataFrame(
            {
                "n_gaps": [0],
                "mean_gap_seconds": [None],
                "std_gap_seconds": [None],
                "gap_burstiness_coeff": [None],
            }
        )

    result = burstiness_module.compute_global_burstiness(valid, id_col=id_col)

    return result.rename(
        {
            "n_intervals": "n_gaps",
            "mean_interval": "mean_gap_seconds",
            "std_interval": "std_gap_seconds",
            "burstiness_coeff": "gap_burstiness_coeff",
        }
    )


############################################################
#        Per-Case Gap Metrics for Stratification           #
############################################################


def compute_case_gap_metrics(
    gaps_df: pl.DataFrame,
    mask_df: pl.DataFrame,
    id_col: str = "id",
    clock_col: str = "clock",
    bin_resolution_seconds: float = 60.0,
    adherence_tolerance: float = 0.5,
    min_gaps: int = 2,
    min_gaps_onset: int = 3,
    min_gaps_qcod: int = 4,
) -> pl.DataFrame:
    """
    Compute per-case, per-variable imperfection metrics for stratification.

    Mirrors the irregularity module's per-case interval metrics, applied to
    gap lengths rather than inter-observation intervals.

    Metrics computed per (id, variable):
        gap_cv              : CV of gap lengths (std / mean); requires min_gaps
        gap_qcod            : Quartile CoD (Q75-Q25)/(Q75+Q25); requires min_gaps_qcod
        gap_burstiness_coeff: Goh & Barabási B = (std-mean)/(std+mean); requires min_gaps >= 3
        gap_adherence_rate  : Fraction of gaps near the case's own dominant gap length
        gap_normalized_entropy: Normalised Shannon entropy of gap length distribution
        max_gap_fraction    : max_gap / total_observation_window; requires >= 1 gap
        gap_onset_cv        : CV of inter-onset intervals (spacing between gap start times);
                              requires min_gaps_onset gaps

    Parameters:
        gaps_df (pl.DataFrame): Output of analyze_gap_lengths(), with columns
                                [id_col, "variable", "count_clock_no", "time_length",
                                 "run_start_clock", "run_end_clock"].
        mask_df (pl.DataFrame): Original mask DataFrame with [id_col, clock_col, ...].
                                Used to compute the total observation window per case.
        id_col (str): Case identifier column.
        clock_col (str): Timestamp column in mask_df (used for window computation).
        bin_resolution_seconds (float): Bin width for entropy/adherence computation.
        adherence_tolerance (float): Fractional tolerance around the dominant gap length.
        min_gaps (int): Minimum number of gaps required for gap_cv and gap_burstiness_coeff.
        min_gaps_onset (int): Minimum gaps required for gap_onset_cv.
        min_gaps_qcod (int): Minimum gaps required for gap_qcod.

    Returns:
        pl.DataFrame: One row per (id_col, variable) with all computed metrics.
                      Missing metrics receive null.
    """
    # Only true gaps (count_clock_no > 0) and non-null time_length
    true_gaps = gaps_df.filter((pl.col("count_clock_no") > 0) & pl.col("time_length").is_not_null())

    all_pairs = true_gaps.select([id_col, "variable"]).unique().sort([id_col, "variable"])

    # --- Base stats: n_gaps, mean, std, q25, q75, max ---
    base = true_gaps.group_by([id_col, "variable"]).agg(
        pl.len().alias("n_gaps"),
        pl.col("time_length").mean().alias("_mean"),
        pl.col("time_length").std().alias("_std"),
        pl.col("time_length").quantile(0.25, interpolation="linear").alias("_q25"),
        pl.col("time_length").quantile(0.75, interpolation="linear").alias("_q75"),
        pl.col("time_length").max().alias("_max"),
    )

    base = base.with_columns(
        pl.when((pl.col("n_gaps") >= min_gaps) & (pl.col("_mean") > 0))
        .then(pl.col("_std") / pl.col("_mean"))
        .otherwise(None)
        .alias("gap_cv"),
        pl.when((pl.col("n_gaps") >= min_gaps_qcod) & ((pl.col("_q75") + pl.col("_q25")) > 0))
        .then((pl.col("_q75") - pl.col("_q25")) / (pl.col("_q75") + pl.col("_q25")))
        .otherwise(None)
        .alias("gap_qcod"),
        pl.when((pl.col("n_gaps") >= 3) & ((pl.col("_std") + pl.col("_mean")) > 0))
        .then((pl.col("_std") - pl.col("_mean")) / (pl.col("_std") + pl.col("_mean")))
        .otherwise(None)
        .alias("gap_burstiness_coeff"),
    )

    # --- max_gap_fraction: max_gap / total observation window per case ---
    window_df = mask_df.group_by(id_col).agg(
        (pl.col(clock_col).max() - pl.col(clock_col).min())
        .dt.total_seconds()
        .alias("_window_seconds")
    )

    base = (
        base.join(window_df, on=id_col, how="left")
        .with_columns(
            pl.when(pl.col("n_gaps").ge(1) & pl.col("_window_seconds").gt(0))
            .then(pl.col("_max") / pl.col("_window_seconds"))
            .otherwise(None)
            .alias("max_gap_fraction")
        )
        .drop("_mean", "_std", "_q25", "_q75", "_max", "_window_seconds")
    )

    # --- Entropy and adherence per (case, variable) ---
    binned = true_gaps.with_columns(
        (pl.col("time_length") / bin_resolution_seconds).round(0).cast(pl.Int64).alias("_gap_bin")
    )

    case_var_totals = binned.group_by([id_col, "variable"]).agg(pl.len().alias("_n_total"))

    case_bin_counts = (
        binned.group_by([id_col, "variable", "_gap_bin"])
        .agg(pl.len().alias("_bin_count"))
        .join(case_var_totals, on=[id_col, "variable"], how="left")
        .with_columns(
            (pl.col("_bin_count").cast(pl.Float64) / pl.col("_n_total")).alias("_fraction")
        )
        .with_columns(
            (-pl.col("_fraction") * pl.col("_fraction").log(base=2.0)).alias("_entropy_contrib")
        )
    )

    case_entropy = (
        case_bin_counts.group_by([id_col, "variable"])
        .agg(
            pl.col("_entropy_contrib").sum().alias("_entropy_bits"),
            pl.col("_gap_bin").count().cast(pl.Int64).alias("_n_unique_bins"),
            pl.col("_gap_bin")
            .sort_by(["_bin_count", "_gap_bin"], descending=True)
            .first()
            .alias("_dominant_bin"),
        )
        .with_columns(
            pl.when(pl.col("_n_unique_bins") > 1)
            .then(pl.col("_entropy_bits") / pl.col("_n_unique_bins").cast(pl.Float64).log(base=2.0))
            .otherwise(0.0)
            .alias("gap_normalized_entropy"),
            (pl.col("_dominant_bin") * bin_resolution_seconds).alias("_dominant_gap_seconds"),
        )
    )

    binned_with_dom = binned.join(
        case_entropy.select([id_col, "variable", "_dominant_gap_seconds"]),
        on=[id_col, "variable"],
        how="left",
    )

    adherence = (
        binned_with_dom.with_columns(
            pl.col("time_length")
            .is_between(
                pl.col("_dominant_gap_seconds") * (1.0 - adherence_tolerance),
                pl.col("_dominant_gap_seconds") * (1.0 + adherence_tolerance),
            )
            .cast(pl.Int32)
            .alias("_adheres")
        )
        .group_by([id_col, "variable"])
        .agg(
            pl.col("_adheres").sum().alias("_n_adhering"),
            pl.len().alias("_n_total_adh"),
        )
        .with_columns(
            (pl.col("_n_adhering").cast(pl.Float64) / pl.col("_n_total_adh")).alias(
                "gap_adherence_rate"
            )
        )
        .select([id_col, "variable", "gap_adherence_rate"])
    )

    entropy_result = case_entropy.select([id_col, "variable", "gap_normalized_entropy"])

    # --- gap_onset_cv: CV of inter-onset intervals (gap start times) ---
    onset_cv = (
        true_gaps.sort([id_col, "variable", "run_start_clock"])
        .with_columns(
            pl.col("run_start_clock")
            .diff()
            .over([id_col, "variable"])
            .dt.total_seconds()
            .alias("_onset_interval")
        )
        .filter(pl.col("_onset_interval").is_not_null() & pl.col("_onset_interval").gt(0))
        .group_by([id_col, "variable"])
        .agg(
            pl.len().alias("_n_onsets"),
            pl.col("_onset_interval").mean().alias("_onset_mean"),
            pl.col("_onset_interval").std().alias("_onset_std"),
        )
        .with_columns(
            pl.when((pl.col("_n_onsets") >= (min_gaps_onset - 1)) & (pl.col("_onset_mean") > 0))
            .then(pl.col("_onset_std") / pl.col("_onset_mean"))
            .otherwise(None)
            .alias("gap_onset_cv")
        )
        .select([id_col, "variable", "gap_onset_cv"])
    )

    # --- Assemble ---
    result = (
        all_pairs.join(base, on=[id_col, "variable"], how="left")
        .join(entropy_result, on=[id_col, "variable"], how="left")
        .join(adherence, on=[id_col, "variable"], how="left")
        .join(onset_cv, on=[id_col, "variable"], how="left")
        .select(
            [
                id_col,
                "variable",
                "n_gaps",
                "gap_cv",
                "gap_qcod",
                "gap_burstiness_coeff",
                "gap_normalized_entropy",
                "gap_adherence_rate",
                "max_gap_fraction",
                "gap_onset_cv",
            ]
        )
        .sort([id_col, "variable"])
    )

    return result


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

    burstiness = compute_gap_burstiness(result.filter(pl.col("variable") == "dbp"))
    print(burstiness)
