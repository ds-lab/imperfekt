import numpy as np
import plotly.graph_objects as go
import polars as pl
import upsetty
from scipy.stats import chi2

from imperfekt.analysis.utils import pretty_printing


def get_patterns(
    mask_df: pl.DataFrame,
    cols: list[str] = None,
) -> pl.DataFrame:
    """
    Extract patterns from the DataFrame based on the mask DataFrame.

    Parameters:
        df (pl.DataFrame): The DataFrame containing the data.
        mask_df (pl.DataFrame): The DataFrame containing the mask indicating imperfect values. (1=missing/noisy/indicated, 0=observed/normal)
        indicated_col (str): The column to check.
        id_col (str): The column that identifies each subject/unit.
        clock_col (str): The name of the timestamp column.

    Returns:
        pl.DataFrame: A DataFrame containing the patterns.
    """
    if cols is None:
        cols = mask_df.columns
    patterns = mask_df.group_by(cols).agg(pl.len().alias("count")).sort(cols, descending=True)

    return patterns


def little_mcar(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    id_col: str = "id",
    clock_col: str = "clock",
    cols: list[str] = None,
    alpha: float = 0.05,
) -> dict:
    """
    Perform Little's MCAR test on the specified columns of a DataFrame.

    Parameters:
        df (pl.DataFrame): The DataFrame containing the data.
        mask_df (pl.DataFrame): The DataFrame containing the mask indicating imperfect values. (1=missing/noisy/indicated, 0=observed/normal)
        id_col (str): The column that identifies each subject/unit.
        clock_col (str): The column that represents the time dimension.
        cols (list[str]): The columns to include in the test.
        alpha (float): The significance level for the test.

    Returns:
        dict: Results of the MCAR test.
    """
    if cols is None:
        cols = df.columns
    m = len(cols)
    if m < 2:
        raise ValueError("Little's MCAR test requires at least two columns to compare patterns.")

    cols_indices = {c: j for j, c in enumerate(cols)}

    # Pattern as bitmask
    df = df.join(mask_df, on=[id_col, clock_col], how="inner", suffix="_mask")
    for j, c in enumerate(cols):
        df = df.with_columns(
            (pl.when(pl.col(c + "_mask") == 1).then(1 << j).otherwise(0)).alias(f"flag_{j}")
        )
    # Sum flags across columns to get one integer pattern per row
    flag_cols = [f"flag_{j}" for j in range(m)]
    df = df.with_columns(sum(pl.col(f) for f in flag_cols).alias("mdp")).drop(flag_cols)

    # Remap patterns to consecutive indices 0..num_patterns−1
    uniq = df.select("mdp").unique().sort("mdp").to_series().to_list()
    df = df.with_columns(
        pl.col("mdp").map_elements(lambda x: uniq.index(x), return_dtype=pl.Int32).alias("mdp")
    )

    complete_cases = df.filter(pl.col("mdp") == 0)
    if complete_cases.height < m:
        raise ValueError("Not enough complete cases to estimate global mean and covariance.")

    global_mean = complete_cases.select(cols).mean().to_numpy().flatten()
    # Use numpy for covariance as polars doesn't have a direct cov matrix function
    global_cov = np.atleast_2d(np.cov(complete_cases.select(cols).to_numpy(), rowvar=False))

    # Loop over each pattern
    test_statistic = 0
    df_sum = 0  # for degrees of freedom
    num_patterns = len(uniq)

    for pat in range(num_patterns):
        group = df.filter(pl.col("mdp") == pat)
        # Determine observed columns (mask == 0 for this pattern)
        observed_cols = [
            c for c in cols if group.select(pl.col(c + "_mask").sum()).to_series()[0] == 0
        ]
        if len(observed_cols) < 1 or group.height < 2:
            continue

        observed_group_data = group.select(observed_cols).to_numpy()
        mean_vec = np.mean(observed_group_data, axis=0)
        observed_cols_indices = [cols_indices[c] for c in observed_cols]
        inv_cov = np.linalg.pinv(global_cov[np.ix_(observed_cols_indices, observed_cols_indices)])

        diff = mean_vec - global_mean[observed_cols_indices]
        d = diff.T @ inv_cov @ diff
        test_statistic += d * group.shape[0]
        df_sum += len(observed_cols)

    degrees_of_freedom = df_sum - m
    p_value = 1 - chi2.cdf(test_statistic, degrees_of_freedom)

    # phi coefficient measuring the strength of association
    phi = (test_statistic / df.height) ** 0.5
    # Cramer's V statistic measuring the strength of association
    cramers_v = (test_statistic / (df.height * min(num_patterns - 1, m - 1))) ** 0.5

    return {
        "test_statistic": test_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "significant": p_value < alpha,
        "phi": phi,
        "cramers_v": cramers_v,
        "complete_cases": complete_cases.shape[0],
    }


def mcar_test(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    id_col: str = "id",
    clock_col: str = "clock",
    cols: list[str] = None,
    alpha: float = 0.05,
) -> tuple[dict, tuple]:
    """
    Perform Little's MCAR test on the specified columns of a DataFrame.
    Answers the questions:
    1. What row-wise patterns of imperfection (missingness, noise,...) are present in the data? (patterns)
    2. Across all variables together, is there any systematic structure to who is imperfect? (little_mcar_test)

    Parameters:
        df (pl.DataFrame): The DataFrame containing the data.
        mask_df (pl.DataFrame): The DataFrame containing the mask indicating imperfect values. (1=missing/noisy/indicated, 0=observed/normal)
        id_col (str): The column that identifies each subject/unit.
        clock_col (str): The column that represents the time dimension.
        cols (list[str]): The columns to include in the test.
        alpha (float): The significance level for the test.

    Returns:
        tuple:
            dict: Results of the MCAR test.
                little_mcar_test: Result of Little's MCAR test answers the question “Across all variables together, is there any systematic structure to who is imperfect?”
                    - The p-values of t-tests for each pair of features. Null hypothesis for cell `pvalues[h,j]`: data in feature `h` is Missing Completely At Random (MCAR) with respect to feature `j` for all `h,j` in `{1,2,...m}`.
                patterns: DataFrame containing the patterns.
            tuple: Shape (rows, columns) of the DataFrame after excluding non-numeric columns.
    """
    if cols is None:
        cols = df.columns
    # Check if the DataFrame contains any non-numeric columns
    non_numeric_cols = df.select(pl.col(cols).exclude(pl.Float64, pl.Int64, pl.Boolean))
    if non_numeric_cols.is_empty():
        df_numeric = df.select(pl.col(cols), pl.col(id_col), pl.col(clock_col))
        mask_df = mask_df.select(pl.col(cols), pl.col(id_col), pl.col(clock_col))
    else:
        df_numeric = df.select(pl.all().exclude(non_numeric_cols.columns))
        mask_df = mask_df.select(pl.all().exclude(non_numeric_cols.columns))

    if non_numeric_cols.shape[1] > 0:
        pretty_printing.rich_warning(
            f"⚠️ Non-numeric columns detected in the DataFrame: {non_numeric_cols.columns}. "
            "Little's MCAR test requires numeric data. "
            "Please convert non-numeric columns to numeric types or exclude them from the analysis."
        )
        if len(cols) - len(non_numeric_cols.columns) < 2:
            raise ValueError(
                "Little's MCAR test requires at least two numeric columns to compare patterns."
            )

    mdp = get_patterns(mask_df=mask_df, cols=cols)

    little_result = little_mcar(
        df_numeric,
        mask_df=mask_df,
        cols=cols,
        id_col=id_col,
        clock_col=clock_col,
        alpha=alpha,
    )

    results = {
        "little_mcar_test": little_result,
        "patterns": mdp,
    }

    return results


def upset(
    mask_df: pl.DataFrame,
    cols: list[str] = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    renderer: str = "browser",
    save_path: str = None,
    save_results: bool = True,
) -> go.Figure:
    """
    Visualizes co-imperfection in the DataFrame using an upset plot using upsetty.

    Parameters:
        mask_df (pl.DataFrame): DataFrame with binary values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected).
        cols (list[str]): List of column names to include in the upset plot. If None, all columns except id_col, clock_col, and clock_no_col will be used.
        id_col (str): Column name for the ID column. Default is "id".
        clock_col (str): Column name for the clock column. Default is "clock".
        clock_no_col (str): Column name for the clock number column. Default is "clock_no".
        renderer (str): Renderer to use for displaying the plot. Default is "browser".
        save_path (str): Path to save the figure. If None, the figure will not be saved.
        save_results (bool): Whether to save the figure to the specified path. Default is False.

    Returns:
        None: Displays the upset plot visualization.
    """
    if cols is None:
        cols = [c for c in mask_df.columns if c not in [id_col, clock_col, clock_no_col]]
    mask_df = mask_df.select(cols)
    pd_df = mask_df.to_pandas().astype(bool)
    upset = upsetty.Upset.generate_plot(pd_df)
    if renderer:
        upset.show(renderer=renderer, width=1600, height=600)
    if save_results and save_path:
        upset.write_image(save_path, width=1600, height=600)
        print(f"Upset plot saved to {save_path}")

    return upset


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

    print("=== MCAR Test ===")
    try:
        result = mcar_test(df, mask_df, cols=["A", "B", "C"])
        print(f"Little's MCAR test p-value: {result[0]['little_mcar_test']['p_value']:.4f}")
        print(f"Phi coefficient: {result[0]['little_mcar_test']['phi']:.4f}")
        print(f"Number of patterns: {result[0]['patterns'].height}")
    except Exception as e:
        print(f"MCAR test failed: {e}")
