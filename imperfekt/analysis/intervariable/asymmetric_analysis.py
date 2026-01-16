from datetime import timedelta

import numpy as np
import polars as pl

from imperfekt.analysis.utils import visualization_utils
from imperfekt.analysis.utils.statistics_utils import mwu_two_subgroups


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks with tie correction."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    unique_vals, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            avg_rank = ranks[mask].mean()
            ranks[mask] = avg_rank
    return ranks


def _rank_biserial_correlation(indicator: np.ndarray, values: np.ndarray) -> float:
    """
    Compute rank biserial correlation between a binary indicator and numeric values.
    Resource: https://en.wikipedia.org/wiki/Mann–Whitney_U_test#Rank-biserial_correlation
    """
    indicator = np.asarray(indicator, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = ~np.isnan(indicator) & ~np.isnan(values)
    if not mask.any():
        return np.nan

    indicator = indicator[mask]
    values = values[mask]

    unique_vals = np.unique(indicator)
    if unique_vals.size < 2:
        return np.nan

    # Determine and count positive and negative groups
    positive_label = 1.0 if 1.0 in unique_vals else unique_vals.max()
    positive_mask = indicator == positive_label
    n_pos = int(positive_mask.sum())
    n_neg = indicator.size - n_pos

    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Compute ranks of observed values
    ranks = _average_ranks(values)
    # Sum ranks for positive group
    rank_sum_pos = ranks[positive_mask].sum()
    # Convert sum of ranks to U statistic
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0

    return (2.0 * u_stat) / (n_pos * n_neg) - 1.0  # Rank biserial correlation formula


def extract_observations_on_conditional(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    indicated_col: str,
    id_col: str = "id",
    clock_col: str = "clock",
) -> pl.DataFrame:
    """
    Calculate the conditional distribution of a column based on the indicated imperfection (missingess etc.) of another column.

    Parameters:
        df (pl.DataFrame): The DataFrame containing the data.
        mask_df (pl.DataFrame): The DataFrame containing the mask indicating imperfect values. (1=missing/noisy/indicated, 0=observed/normal)
        indicated_col (str): The column to check.

    Returns:
        pl.DataFrame: A DataFrame containing the conditional distribution.
    """
    if indicated_col not in df.columns:
        raise ValueError(f"Columns '{indicated_col}' not found in DataFrame.")

    result_df = (
        mask_df.lazy()
        .filter(pl.col(indicated_col) == 1)
        .join(
            df.lazy(),
            on=[id_col, clock_col],
            how="inner",
            suffix="_real",
        )
        .select(
            pl.col(id_col),
            pl.col(clock_col),
            *[
                pl.col(c + "_real")
                for c in df.columns
                if c not in {id_col, clock_col, indicated_col}
            ],
        )
        .rename({c + "_real": c for c in df.columns if c not in {id_col, clock_col, indicated_col}})
        .collect()
    )

    return result_df


def extract_observations_on_conditional_window(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    indicated_col: str,
    id_col: str = "id",
    clock_col: str = "clock",
    window: timedelta = timedelta(minutes=5),
    window_location: str = "both",
) -> pl.DataFrame:
    """
    Extract all observations in a time window around timestamps where `indicated_col` is 1.

    Parameters:
        df (pl.DataFrame): Input time-series data.
        mask_df (pl.DataFrame): DataFrame with a mask indicating "imperfect" values (1=missing/noisy/indicated, 0=observed).
        indicated_col (str): Column whose imperfection defines the anchor timestamps.
        id_col (str): Column that identifies each subject/unit.
        clock_col (str): Name of the timestamp column.
        window (timedelta): Time window to extract data around imperfect timestamps.
        window_location (str): 'before', 'after', or 'both'.

    Returns:
        pl.DataFrame: All rows within the time window around each imperfect value,
                      annotated with which imperfect timestamp they relate to.
    """
    if indicated_col not in df.columns:
        raise ValueError(f"Column '{indicated_col}' not found in DataFrame.")
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame.")
    if clock_col not in df.columns:
        raise ValueError(f"Column '{clock_col}' not found in DataFrame.")

    # 1. Identify timestamps where the value is imperfect
    imperfect_df = (
        mask_df.lazy()
        .filter(pl.col(indicated_col) == 1)
        .join(
            df.lazy(),
            on=[id_col, clock_col],
            how="inner",
            suffix="_mask",
        )
        .select(
            pl.col(id_col),
            pl.col(clock_col).alias("imperfect_time"),
        )
        .collect()
    )

    if imperfect_df.is_empty():
        return df.clear().with_columns(pl.Series("imperfect_time", [], dtype=df[clock_col].dtype))

    # 2. Get the time window around each imperfect timestamp
    if window_location == "before":
        imperfect_df = imperfect_df.with_columns(
            (pl.col("imperfect_time") - window).alias("start_time"),
            pl.col("imperfect_time").alias("end_time"),
        )
    elif window_location == "after":
        imperfect_df = imperfect_df.with_columns(
            pl.col("imperfect_time").alias("start_time"),
            (pl.col("imperfect_time") + window).alias("end_time"),
        )
    else:  # 'both'
        imperfect_df = imperfect_df.with_columns(
            (pl.col("imperfect_time") - window).alias("start_time"),
            (pl.col("imperfect_time") + window).alias("end_time"),
        )

    # 3. Join the original DataFrame with the imperfect timestamps on the ID column and filter based on the time window
    result_df = (
        imperfect_df.join(df, on=id_col, how="inner")
        .filter(pl.col(clock_col).is_between(pl.col("start_time"), pl.col("end_time")))
        .select(
            pl.col(id_col),
            pl.col(clock_col),
            pl.col("imperfect_time"),
            *[pl.col(c) for c in df.columns if c not in {id_col, clock_col, indicated_col}],
        )
        .group_by([pl.col(id_col), pl.col(clock_col)])
        .agg(
            pl.all().max(), pl.col("imperfect_time").implode().alias("imperfect_times")
        )  # imperfect_time values are collected into a list
        .select(
            pl.col(id_col),
            pl.col(clock_col),
            pl.col("imperfect_times"),
            *[pl.col(c) for c in df.columns if c not in {id_col, clock_col, indicated_col}],
        )
    )

    return result_df


def asymmetric_missing_observation_lagged_correlation(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    indicated_col: str,
    observation_col: str,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    max_lag: int = 10,
):
    """
    Compute lagged rank biserial correlation between missing indicators of one variable and
    observations of another. This is an asymmetric analysis where we compare the missing
    pattern of one variable with the actual observed values of another variable.

    Negative lags indicate that missingness in indicated_col leads changes in observation_col values.
    Positive lags indicate that observation_col values precede missingness in indicated_col.

    Parameters:
        df (pl.DataFrame): DataFrame containing the actual data values.
        mask_df (pl.DataFrame): DataFrame with binary values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected).
        indicated_col (str): Column whose missingness pattern we want to analyze.
        observation_col (str): Column whose observed values we want to correlate with missingness.
        id_col (str): ID column.
        clock_no_col (str): Time ordering column.
        max_lag (int): Maximum lag (both positive and negative) to compute.

    Returns:
        tuple: (lags, crosscorrs)
            lags (np.ndarray): Array of lag values (negative to positive).
            crosscorrs (np.ndarray): Cross-correlation at each lag.
    """
    if indicated_col not in mask_df.columns:
        raise ValueError(f"Column '{indicated_col}' not found in mask DataFrame.")
    if observation_col not in df.columns:
        raise ValueError(f"Column '{observation_col}' not found in data DataFrame.")

    if max_lag > mask_df.height:
        max_lag = mask_df.height - 1

    lags = np.arange(-max_lag, max_lag + 1)

    # Join mask and data on id and time columns
    combined_df = (
        mask_df.select([id_col, clock_no_col, indicated_col])
        .join(
            df.select([id_col, clock_no_col, observation_col]),
            on=[id_col, clock_no_col],
            how="inner",
        )
        .sort([id_col, clock_no_col])
        # Filter out rows where observation_col is null
        .filter(pl.col(observation_col).is_not_null())
    )

    # Check for constant series (zero variance)
    stats = combined_df.select(
        pl.var(indicated_col).alias("var_missing"),
        pl.var(observation_col).alias("var_obs"),
    )

    if stats["var_missing"][0] == 0 or stats["var_obs"][0] == 0:
        raise ValueError("One or both series are constant. Cross-correlation is undefined.")

    # Calculate rank biserial correlation for each lag
    crosscorrs = []
    for lag in lags:
        lagged_df = combined_df.select(
            pl.col(indicated_col).alias("indicator"),
            pl.col(observation_col).shift(lag).over(id_col).alias("lagged_observation"),
        ).drop_nulls()

        if lagged_df.is_empty():
            crosscorrs.append(np.nan)
            continue

        indicator = lagged_df["indicator"].to_numpy()
        values = lagged_df["lagged_observation"].to_numpy()
        crosscorrs.append(_rank_biserial_correlation(indicator, values))

    crosscorrs = np.asarray(crosscorrs, dtype=float)

    ccfs = pl.DataFrame({"lag": lags, "crosscorr": crosscorrs})

    return ccfs


def get_singular_correlation_from_lagged(ccfs: pl.DataFrame):
    """
    Extract the lag=0 correlation from lagged correlation results.

    Parameters:
        lags (np.ndarray): Array of lag values.
        crosscorrs (np.ndarray): Cross-correlation at each lag.

    Returns:
        float: Correlation at lag=0, or None if not found.
    """
    return (
        ccfs.filter(pl.col("lag") == 0).select(pl.col("crosscorr")).to_numpy().item(0)
        if ccfs.height > 0
        else None
    )


def asymmetric_missing_observation_matrix(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    missing_cols: list[str] = None,
    observation_cols: list[str] = None,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    max_lag: int = 10,
    include_singular: bool = True,
) -> pl.DataFrame:
    """
    Create a matrix of asymmetric correlations between missing patterns and observations.

    For each pair (indicated_col, observation_col), computes the correlation between
    the missingness indicator of indicated_col and the observed values of observation_col
    at various time lags.

    Parameters:
        df (pl.DataFrame): DataFrame containing the actual data values.
        mask_df (pl.DataFrame): DataFrame with binary missingness indicators.
        missing_cols (list[str]): Columns whose missingness patterns to analyze. If None, uses all mask columns.
        observation_cols (list[str]): Columns whose observations to correlate with missingness. If None, uses all data columns.
        id_col (str): ID column.
        clock_no_col (str): Time ordering column.
        max_lag (int): Maximum lag to compute.
        include_singular (bool): Whether to include lag=0 correlation as a separate column.

    Returns:
        pl.DataFrame: Matrix with correlation results for each indicated_col vs observation_col pair.
    """
    if missing_cols is None:
        missing_cols = [c for c in mask_df.columns if c not in {id_col, clock_no_col}]
    if observation_cols is None:
        observation_cols = [c for c in df.columns if c not in {id_col, clock_no_col}]

    results = []
    lags = np.arange(-max_lag, max_lag + 1)

    for indicated_col in missing_cols:
        for obs_col in observation_cols:
            if indicated_col == obs_col:
                continue  # Skip self-correlation

            try:
                ccfs = asymmetric_missing_observation_lagged_correlation(
                    df=df,
                    mask_df=mask_df,
                    indicated_col=indicated_col,
                    observation_col=obs_col,
                    id_col=id_col,
                    clock_no_col=clock_no_col,
                    max_lag=max_lag,
                )

                # Filter out NaN values when finding max
                valid_corrs = ccfs.filter(pl.col("crosscorr").is_not_null())
                if len(valid_corrs) > 0:
                    # Find the lag with maximum absolute correlation among valid values
                    valid_indices = np.where(~np.isnan(ccfs["crosscorr"].to_numpy()))[0]
                    valid_abs_corrs = np.abs(valid_corrs["crosscorr"].to_numpy())
                    max_valid_idx = np.argmax(valid_abs_corrs)
                    max_corr_idx = valid_indices[max_valid_idx]
                    max_lag_val = lags[max_corr_idx]
                    max_corr_val = ccfs["crosscorr"].to_numpy()[max_corr_idx]
                else:
                    max_corr_val = None
                    max_lag_val = None

                # Extract singular correlation (lag=0) if requested
                singular_corr = (
                    get_singular_correlation_from_lagged(ccfs) if include_singular else None
                )

                result_dict = {
                    "indicated_col": indicated_col,
                    "observation_col": obs_col,
                    "max_correlation": max_corr_val,
                    "max_correlation_lag": max_lag_val,
                    "all_correlations": ccfs.select(pl.col("crosscorr")).to_numpy(),
                    "lags": ccfs.select(pl.col("lag")).to_numpy(),
                    "n_valid_correlations": len(valid_corrs),
                }

                if include_singular:
                    result_dict["singular_correlation"] = singular_corr

                results.append(result_dict)

            except ValueError as e:
                # Handle cases where correlation cannot be computed
                result_dict = {
                    "indicated_col": indicated_col,
                    "observation_col": obs_col,
                    "max_correlation": None,
                    "max_correlation_lag": None,
                    "all_correlations": None,
                    "lags": None,
                    "n_valid_correlations": 0,
                    "error": str(e),
                }

                if include_singular:
                    result_dict["singular_correlation"] = None

                results.append(result_dict)

    return pl.DataFrame(results)


def asymmetric_statistical_comparison(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    indicated_col: str,
    observation_col: str,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    statistical_tests: bool = True,
    save_path: str = None,
    save_results: bool = True,
    plot_library: str = "matplotlib",
    renderer: str = None,
) -> dict:
    """
    Comprehensive statistical comparison between observed values when a variable is missing
    vs. when it's not missing. Combines correlation analysis with conditional distribution analysis.

    Parameters:
        df (pl.DataFrame): DataFrame containing the actual data values.
        mask_df (pl.DataFrame): DataFrame with binary missingness indicators.
        indicated_col (str): Column whose missingness defines the condition.
        observation_col (str): Column whose values we want to analyze.
        id_col (str): ID column.
        clock_col (str): Time column.
        clock_no_col (str): Time ordering column.
        statistical_tests (bool): Whether to perform statistical tests (t-test, etc.).
        save_path (str): Path to save results (if any).
        save_results (bool): Whether results should be stored at save_path.
        plot_library (str): Library to use for plotting (e.g., "matplotlib", "plotly").
        renderer (str): Renderer to use for plots (e.g., "notebook", "browser").

    Returns:
        dict: Comprehensive analysis results including:
            - correlation: Direct correlation between missing indicator and observations (lag=0)
            - conditional_stats: Statistics for observations when missing vs. not missing
            - statistical_tests: Results of statistical tests if requested
    """
    # 1. Direct correlation analysis using lag=0 from lagged correlation
    try:
        ccfs = asymmetric_missing_observation_lagged_correlation(
            df=df,
            mask_df=mask_df,
            indicated_col=indicated_col,
            observation_col=observation_col,
            id_col=id_col,
            clock_no_col=clock_no_col,
            max_lag=0,  # Only compute lag=0
        )
        correlation = get_singular_correlation_from_lagged(ccfs)

    except ValueError as e:
        correlation = None
        correlation_error = str(e)

    # 2. Extract observations when indicated_col is missing
    obs_when_missing = extract_observations_on_conditional(
        df=df,
        mask_df=mask_df,
        indicated_col=indicated_col,
        id_col=id_col,
        clock_col=clock_col,
    )

    # 3. Extract observations when indicated_col is NOT missing (remaining observations)
    obs_when_present = (
        df.join(obs_when_missing, on=[id_col, clock_col], how="anti")
        .filter(pl.col(observation_col).is_not_null())
        .select([id_col, clock_col, observation_col])
    )

    # 4. Calculate descriptive statistics
    results = {
        "indicated_col": indicated_col,
        "observation_col": observation_col,
        "correlation": correlation,
    }

    if correlation is None:
        results["correlation_error"] = correlation_error

    # Statistics when indicated_col is missing
    if obs_when_missing.height > 0 and observation_col in obs_when_missing.columns:
        results["stats_when_missing"] = {
            "count": obs_when_missing.height,
            "mean": obs_when_missing.select(pl.col(observation_col).mean()).item(),
            "std": obs_when_missing.select(pl.col(observation_col).std()).item(),
            "min": obs_when_missing.select(pl.col(observation_col).min()).item(),
            "max": obs_when_missing.select(pl.col(observation_col).max()).item(),
            "median": obs_when_missing.select(pl.col(observation_col).median()).item(),
        }
    else:
        results["stats_when_missing"] = {"count": 0}

    # Statistics when indicated_col is present
    if obs_when_present.height > 0:
        results["stats_when_present"] = {
            "count": obs_when_present.height,
            "mean": obs_when_present.select(pl.col(observation_col).mean()).item(),
            "std": obs_when_present.select(pl.col(observation_col).std()).item(),
            "min": obs_when_present.select(pl.col(observation_col).min()).item(),
            "max": obs_when_present.select(pl.col(observation_col).max()).item(),
            "median": obs_when_present.select(pl.col(observation_col).median()).item(),
        }
    else:
        results["stats_when_present"] = {"count": 0}

    # 5. Statistical tests if requested
    if (
        statistical_tests
        and results["stats_when_missing"]["count"] > 0
        and results["stats_when_present"]["count"] > 0
    ):
        try:
            # Mann-Whitney U test (non-parametric)
            results["mann_whitney_u"] = mwu_two_subgroups(
                df1=obs_when_missing,
                df2=obs_when_present,
                col1=observation_col,
                col2=observation_col,
                alpha=0.05,
                print_info=False,
                save_results=False,
            )

        except Exception as e:
            results["mann_whitney_u"] = {"error": str(e)}

    if save_path and save_results:
        # Write results to CSV
        results_df = pl.DataFrame([results])
        results_df.write_csv(save_path / "statistical_comparison_results.csv")
        print(f"Statistical comparison results saved to {save_path}")

    hist_overlay_fig = visualization_utils.plot_overlay_histograms(
        dfs=[obs_when_missing, obs_when_present],
        x=observation_col,
        group_names=[
            f"{observation_col} Observed When {indicated_col} Imperfect",
            f"{observation_col} Observed When {indicated_col} Observed",
        ],
        title=f"Overlay Histogram of {observation_col}",
        xaxis_title=f"{observation_col}",
        yaxis_title="Frequency",
        histnorm="probability",
        library=plot_library,
        renderer=renderer,
        save_path=save_path / f"{observation_col}_o_{indicated_col}_i_overlay_histogram.png",
        save_results=save_results,
    )

    multi_box_fig = visualization_utils.plot_multi_boxplot(
        dfs=[obs_when_missing, obs_when_present],
        y=observation_col,
        group_names=[
            f"{observation_col} Observed When {indicated_col} Imperfect",
            f"{observation_col} Observed When {indicated_col} Observed",
        ],
        title=f"Boxplot of {observation_col} by {indicated_col} Imperfection",
        yaxis_title=f"{observation_col}",
        library=plot_library,
        renderer=renderer,
        save_path=save_path / f"{observation_col}_o_{indicated_col}_i_boxplot.png",
        save_results=save_results,
    )

    return results, hist_overlay_fig, multi_box_fig


if __name__ == "__main__":
    # Example usage
    df = pl.DataFrame(
        {
            "id": ["a"] * 10 + ["b"] * 10,
            "clock": [f"2023-10-01 12:{i:02d}:00" for i in range(10)] * 2,
            "clock_no": list(range(1, 11)) * 2,
            "A": [1.0, 2.0, None, 4.0, 5.0, 6.0, None, 8.0, 9.0, 10.0] * 2,
            "B": [10.0, None, 12.0, 113.0, 14.0, None, 116.0, 17.0, 18.0, 19.0] * 2,
            "C": [20.0, 221.0, 22.0, None, 24.0, 252.0, 26.0, None, 28.0, 29.0] * 2,
        }
    ).with_columns(
        [
            pl.col("clock").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )
    mask_df = df.with_columns(
        [
            pl.col("A").is_null().cast(pl.Int8).alias("A"),
            pl.col("B").is_null().cast(pl.Int8).alias("B"),
            pl.col("C").is_null().cast(pl.Int8).alias("C"),
        ]
    )

    print("\n=== Observations on Conditional ===")
    conditional_obs = extract_observations_on_conditional(df, mask_df, "A")
    print(f"Observations when A is missing: {conditional_obs.height} rows")
    if conditional_obs.height > 0:
        print(conditional_obs.head())

    print("\n=== Asymmetric Missing-Observation Correlation ===")
    # Example: How does missingness in A correlate with observed values of B?
    try:
        ccfs = asymmetric_missing_observation_lagged_correlation(
            df=df,
            mask_df=mask_df,
            indicated_col="A",
            observation_col="B",
            id_col="id",
            clock_no_col="clock_no",
            max_lag=2,
        )
        print(ccfs)

        # Extract lag=0 correlation
        singular_corr = get_singular_correlation_from_lagged(ccfs)
        print(f"Lag=0 correlation: {singular_corr:.4f}")
    except ValueError as e:
        print(f"Could not compute correlation: {e}")

    print("\n=== Asymmetric Matrix with Singular Correlations ===")
    try:
        matrix_with_singular = asymmetric_missing_observation_matrix(
            df=df,
            mask_df=mask_df,
            missing_cols=["A", "B", "C"],
            observation_cols=["A", "B", "C"],
            id_col="id",
            clock_no_col="clock_no",
            max_lag=2,
            include_singular=True,
        )
        print("Matrix with both lagged and singular correlations:")
        print(
            matrix_with_singular.select(
                pl.col("indicated_col"),
                pl.col("observation_col"),
                pl.col("singular_correlation"),
            )
        )
    except Exception as e:
        print(f"Error in matrix computation: {e}")

    print("\n=== Comprehensive Statistical Comparison ===")
    # Detailed statistical analysis comparing observations when A is missing vs present
    try:
        comparison = asymmetric_statistical_comparison(
            df=df,
            mask_df=mask_df,
            indicated_col="A",
            observation_col="B",
            id_col="id",
            clock_col="clock",
            clock_no_col="clock_no",
            statistical_tests=True,
        )

        print(f"Missing column: {comparison['indicated_col']}")
        print(f"Observation column: {comparison['observation_col']}")
        print(f"Direct correlation (lag=0): {comparison.get('correlation', 'N/A')}")

        print("\nWhen A is missing:")
        print(f"  Count: {comparison['stats_when_missing']['count']}")
        if comparison["stats_when_missing"]["count"] > 0:
            print(f"  Mean B: {comparison['stats_when_missing']['mean']:.2f}")
            print(f"  Std B: {comparison['stats_when_missing']['std']:.2f}")

        print("\nWhen A is present:")
        print(f"  Count: {comparison['stats_when_present']['count']}")
        if comparison["stats_when_present"]["count"] > 0:
            print(f"  Mean B: {comparison['stats_when_present']['mean']:.2f}")
            print(f"  Std B: {comparison['stats_when_present']['std']:.2f}")

        if "statistical_tests" in comparison:
            print("\nStatistical Tests:")
            if "t_test" in comparison["statistical_tests"]:
                t_test = comparison["statistical_tests"]["t_test"]
                print(
                    f"  T-test p-value: {t_test['p_value']:.4f} (significant: {t_test['significant_05']})"
                )
            if "mann_whitney_u" in comparison["statistical_tests"]:
                u_test = comparison["statistical_tests"]["mann_whitney_u"]
                print(
                    f"  Mann-Whitney U p-value: {u_test['p_value']:.4f} (significant: {u_test['significant_05']})"
                )

    except Exception as e:
        print(f"Error in statistical comparison: {e}")

    print("\n=== Asymmetric Missing-Observation Matrix ===")
    # Analyze all combinations
    try:
        result_matrix = asymmetric_missing_observation_matrix(
            df=df,
            mask_df=mask_df,
            missing_cols=["A", "B", "C"],
            observation_cols=["A", "B", "C"],
            id_col="id",
            clock_no_col="clock_no",
            max_lag=2,
            include_singular=False,  # Just show lagged results
        )
        print(f"Matrix shape: {result_matrix.shape}")
        # Show results with valid correlations
        valid_results = result_matrix.filter(pl.col("max_correlation").is_not_null())
        if valid_results.height > 0:
            print("Valid correlations found:")
            print(
                valid_results.select(
                    [
                        "indicated_col",
                        "observation_col",
                        "max_correlation",
                        "max_correlation_lag",
                    ]
                )
            )
        else:
            print("No valid correlations computed")
    except Exception as e:
        print(f"Error in matrix computation: {e}")
