from pathlib import Path

import polars as pl


def analyze_all_null_rows(
    mask_df: pl.DataFrame,
    cols: list | None = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str | None = None,
    save_results: bool = True,
) -> tuple[int, float]:
    """
    Get all-null rows in the DataFrame and calculate the percentage of such rows.

    Parameters:
        mask_df (pl.DataFrame): imperfection mask DataFrame where 1 indicates imperfect values and 0 indicates present values.
        cols (list, optional): List of columns to check for all-null rows. If None, all columns are considered.
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_col (str): The name of the column representing the clock time. Defaults to "clock".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".
        save_path (str, optional): Path to save the results. If None, results are not saved.
        save_results (bool): Whether to save the results to a CSV file. Defaults to True.

    Returns:
        Tuple[int, float]: A tuple containing the count of all-null rows and the percentage of such rows in the DataFrame.
                           The percentage is calculated as (all-null rows / total rows) * 100.
    """
    if cols is None:
        cols = mask_df.columns
    cols = [c for c in cols if c not in {id_col, clock_col, clock_no_col}]
    # In mask_df, 1 indicates imperfect values, so we filter where all specified columns equal 1
    all_null_rows = mask_df.filter(pl.all_horizontal(pl.col(col) == 1 for col in cols)).height

    n_rows = mask_df.height
    percentage_all_null = (all_null_rows / n_rows) * 100 if n_rows > 0 else 0

    if save_path and save_results:
        # Save the results to a CSV file if a path is provided
        result_df = pl.DataFrame(
            {
                "all_null_rows": [all_null_rows],
                "percentage_all_null": [percentage_all_null],
            }
        )
        result_df.write_csv(save_path)

    return all_null_rows, percentage_all_null


def analyze_all_null_rows_per_id(
    mask_df: pl.DataFrame,
    cols: list | None = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str | Path | None = None,
    save_results: bool = True,
) -> pl.DataFrame:
    """
    Analyze all-null rows per ID in the DataFrame and calculate the percentage of such rows per ID,
    useful to identify if some subjects have substantially more imperfect data than others.

    Parameters:
        mask_df (pl.DataFrame): imperfection mask DataFrame where 1 indicates imperfect values and 0 indicates present values.
        cols (list, optional): List of columns to check for all-null rows. If None, all columns are considered.
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_col (str): The name of the column representing the clock time. Defaults to "clock".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".
        save_path (str, optional): Path to save the results. If None, results are not saved.
        save_results (bool): Whether to save the results to a CSV file. Defaults to True.

    Returns:
        pl.DataFrame: A DataFrame containing the count of all-null rows, total rows per ID, and the percentage of all-null rows per ID.
                      The DataFrame will have columns 'id', 'null_vitals_count', 'total_rows_per_id', and 'null_vitals_pct'.
    """
    if cols is None:
        cols = mask_df.columns
    cols = [c for c in cols if c not in {id_col, clock_col, clock_no_col}]

    # Create a boolean expression for rows where all specified columns are null
    all_null_expr = pl.all_horizontal(pl.col(col) == 1 for col in cols)

    mask_df = mask_df.with_columns(
        all_null_expr.alias("is_all_null"),
    )

    # Group by id and aggregate to get counts and calculate percentage
    per_id_df = (
        mask_df.group_by(id_col)
        .agg(
            pl.col("is_all_null").sum().alias("null_vitals_count"),
            pl.len().alias("total_rows_per_id"),
        )
        .with_columns(
            (pl.col("null_vitals_count") / pl.col("total_rows_per_id") * 100).alias(
                "null_vitals_pct"
            )
        )
        .sort("null_vitals_pct", descending=True)
    )

    if save_path and save_results:
        # Save the results to a CSV file if a path is provided
        per_id_df.describe().write_csv(save_path)

    return per_id_df


def analyze_row_imperfection(
    mask_df: pl.DataFrame,
    cols: list | None = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str | None = None,
    save_results: bool = True,
) -> pl.DataFrame | None:
    """
    Analyze the completeness of rows in the DataFrame and calculate the percentage of imperfect variables.

    Parameters:
        mask_df (pl.DataFrame): imperfection mask DataFrame where 1 indicates imperfect values and 0 indicates present values.
        cols (list, optional): List of columns to analyze for imperfection. If None, all columns are considered.
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_col (str): The name of the column representing the clock time. Defaults to "clock".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".
        save_path (str, optional): Path to save the results. If None, results are not saved.
        save_results (bool): Whether to save the results to a CSV file. Defaults to True.

    Returns:
        pl.DataFrame: A DataFrame containing the count of imperfect variables per row and their percentage.
                      The DataFrame will have columns 'indicated_vars' and 'indicated_vars_pct'.
    """
    if cols is None:
        cols = mask_df.columns
    cols = [c for c in cols if c not in {id_col, clock_col, clock_no_col}]

    total_rows = mask_df.height

    if total_rows == 0:
        return None

    expr = pl.fold(
        acc=pl.lit(0),
        function=lambda acc, x: acc + x,
        exprs=[pl.col(c) for c in cols],
    ).alias("indicated_vars")
    indicated_vars_per_row = mask_df.with_columns(expr)

    indicated_vars_per_row = indicated_vars_per_row.with_columns(
        (pl.col("indicated_vars") / len(cols) * 100).alias("indicated_vars_pct")
    )

    if save_path and save_results:
        # Save the results to a CSV file if a path is provided
        indicated_vars_per_row.describe().write_csv(save_path)
        print(f"Overall row imperfection stats saved to {save_path}.")

    return indicated_vars_per_row


def analyze_row_imperfection_per_id(
    mask_df: pl.DataFrame,
    cols: list | None = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str | None = None,
    save_results: bool = True,
) -> pl.DataFrame:
    """
    Analyze the completeness of rows per ID in the DataFrame and calculate the average percentage of imperfect/imperfect variables.

    Parameters:
        mask_df (pl.DataFrame): imperfection mask DataFrame where 1 indicates imperfect/imperfect values and 0 indicates present values.
        cols (list, optional): List of columns to analyze for imperfection. If None, all columns are considered.
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_col (str): The name of the column representing the clock time. Defaults to "clock".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".
        save_path (str, optional): Path to save the results. If None, results are not saved.
        save_results (bool): Whether to save the results to a CSV file. Defaults to True.

    Returns:
        pl.DataFrame: A DataFrame containing the average percentage of imperfect/imperfect variables per ID.
                      The DataFrame will have columns 'id' and 'avg_indicated_vars_pct'.
    """
    if cols is None:
        cols = mask_df.columns
    cols = [c for c in cols if c not in {id_col, clock_col, clock_no_col}]

    df = analyze_row_imperfection(mask_df, cols)

    if df is None:
        return pl.DataFrame(
            {
                id_col: [],
                "avg_indicated_vars_pct": [],
            }
        )

    # Group by ID and calculate the average percentage of imperfect variables
    per_id_df = (
        df.group_by(id_col)
        .agg(pl.mean("indicated_vars_pct").alias("avg_indicated_vars_pct"))
        .sort("avg_indicated_vars_pct", descending=True)
    )

    if save_path and save_results:
        # Save the results to a CSV file if a path is provided
        per_id_df.describe().write_csv(save_path)
        print(f"Row completeness per ID stats saved to {save_path}.")

    return per_id_df


def compute_case_intervariable_metrics(
    mask_df: pl.DataFrame,
    cols: list | None = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
) -> pl.DataFrame | None:
    """
    Compute per-case cross-variable imperfection metrics for stratification.

    Metrics computed per case (across all cols):
        avg_indicated_vars_pct    : Mean row-level imperfection percentage across all rows.
        co_missingness_concentration: Mean indicated_vars_pct conditioned on rows with any
                                      imperfection (indicated_vars > 0). Null if no imperfect rows.
        missing_variable_breadth  : Fraction of variables that have any missingness for this case.
        pattern_entropy           : Normalised Shannon entropy over per-row missingness bitmasks
                                    (computed on imperfect rows only). 0 = always the same co-dropout
                                    pattern; 1 = maximally varied. Null if fewer than 2 imperfect rows.
        max_pairwise_co_missingness: Max over all variable pairs of
                                     count(A AND B missing) / min(count(A missing), count(B missing)).
                                     Null if no pair has any co-missingness.

    Parameters:
        mask_df (pl.DataFrame): Binary mask (1=imperfect, 0=observed) with columns
                                [id_col, clock_col, clock_no_col, ...variable cols...].
        cols (list): Variable columns to analyse. If None, all non-index columns are used.
        id_col (str): Case identifier column.
        clock_col (str): Timestamp column.
        clock_no_col (str): Integer time-ordering column.

    Returns:
        pl.DataFrame: One row per case with all five metrics.
    """
    if cols is None:
        cols = [c for c in mask_df.columns if c not in {id_col, clock_col, clock_no_col}]

    n_cols = len(cols)

    # --- Row-level indicated_vars and indicated_vars_pct ---
    row_df = mask_df.with_columns(
        pl.fold(
            acc=pl.lit(0),
            function=lambda acc, x: acc + x,
            exprs=[pl.col(c) for c in cols],
        ).alias("_indicated_vars")
    ).with_columns((pl.col("_indicated_vars") / n_cols * 100).alias("_indicated_vars_pct"))

    # --- avg_indicated_vars_pct and co_missingness_concentration ---
    per_id_base = row_df.group_by(id_col).agg(
        pl.col("_indicated_vars_pct").mean().alias("avg_indicated_vars_pct"),
        pl.col("_indicated_vars_pct")
        .filter(pl.col("_indicated_vars") > 0)
        .mean()
        .alias(
            "co_missingness_concentration"
        ),  # the mean if conditioned on imperfect rows only, else null
    )

    # --- missing_variable_breadth, how many variables have any missingness ---
    breadth = (
        mask_df.group_by(id_col)
        .agg(
            pl.sum_horizontal([pl.col(c).max().cast(pl.Int32) for c in cols]).alias(
                "_n_vars_with_any"
            )
        )
        .with_columns(
            (pl.col("_n_vars_with_any").cast(pl.Float64) / n_cols).alias("missing_variable_breadth")
        )
        .select([id_col, "missing_variable_breadth"])
    )

    # --- pattern_entropy: normalised Shannon entropy over bitmask patterns (imperfect rows only) ---
    # Build a string bitmask per row from imperfect rows only
    imperfect_rows = mask_df.with_columns(
        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cols], separator="").alias("_bitmask")
    ).filter(
        pl.fold(
            acc=pl.lit(0),
            function=lambda acc, x: acc + x,
            exprs=[pl.col(c) for c in cols],
        )
        > 0  # 000...0 is not a pattern we want to include in the entropy calculation, since it represents no imperfection
    )

    pattern_counts = imperfect_rows.group_by([id_col, "_bitmask"]).agg(pl.len().alias("_pat_count"))

    pattern_totals = imperfect_rows.group_by(id_col).agg(pl.len().alias("_n_imperfect_rows"))

    entropy_df = (
        pattern_counts.join(pattern_totals, on=id_col, how="left")
        .with_columns(
            (pl.col("_pat_count").cast(pl.Float64) / pl.col("_n_imperfect_rows")).alias("_frac")
        )
        .with_columns((-pl.col("_frac") * pl.col("_frac").log(base=2.0)).alias("_entropy_contrib"))
        .group_by(id_col)
        .agg(
            pl.col("_entropy_contrib").sum().alias("_entropy_bits"),
            pl.col("_bitmask").count().cast(pl.Int64).alias("_n_unique_patterns"),
            pl.col("_n_imperfect_rows").first().alias("_n_imperfect_rows"),
        )
        .with_columns(
            pl.when((pl.col("_n_imperfect_rows") >= 2) & (pl.col("_n_unique_patterns") > 1))
            .then(
                pl.col("_entropy_bits")
                / pl.col("_n_unique_patterns").cast(pl.Float64).log(base=2.0)
            )
            .when(pl.col("_n_imperfect_rows") >= 2)
            .then(pl.lit(0.0))
            .otherwise(None)
            .alias("pattern_entropy")
        )
        .select([id_col, "pattern_entropy"])
    )

    # --- max_pairwise_co_missingness ---
    # For each pair (A, B): overlap = count(A AND B missing) / min(count(A missing), count(B missing))
    per_col_miss = mask_df.group_by(id_col).agg([pl.col(c).sum().alias(f"_miss_{c}") for c in cols])

    import itertools

    pair_overlaps = mask_df.group_by(id_col).agg(
        [
            (pl.col(a) * pl.col(b)).sum().alias(f"_co_{a}__{b}")
            for a, b in itertools.combinations(cols, 2)
        ]
    )

    co_df = per_col_miss.join(pair_overlaps, on=id_col, how="left")

    overlap_exprs = []
    for a, b in itertools.combinations(cols, 2):
        co_col = f"_co_{a}__{b}"
        min_col = f"_min_{a}_{b}"
        overlap_col = f"_ov_{a}_{b}"
        co_df = co_df.with_columns(
            pl.min_horizontal(pl.col(f"_miss_{a}"), pl.col(f"_miss_{b}")).alias(min_col)
        ).with_columns(
            pl.when(pl.col(min_col) > 0)
            .then(pl.col(co_col).cast(pl.Float64) / pl.col(min_col).cast(pl.Float64))
            .otherwise(None)
            .alias(overlap_col)
        )
        overlap_exprs.append(overlap_col)

    if overlap_exprs:
        co_df = co_df.with_columns(
            pl.max_horizontal([pl.col(c) for c in overlap_exprs]).alias(
                "max_pairwise_co_missingness"
            )
        ).select([id_col, "max_pairwise_co_missingness"])
    else:
        co_df = co_df.select([id_col]).with_columns(
            pl.lit(None).cast(pl.Float64).alias("max_pairwise_co_missingness")
        )

    # --- Assemble ---
    all_cases = mask_df.select(pl.col(id_col).unique()).sort(id_col)
    result = (
        all_cases.join(per_id_base, on=id_col, how="left")
        .join(breadth, on=id_col, how="left")
        .join(entropy_df, on=id_col, how="left")
        .join(co_df, on=id_col, how="left")
        .select(
            [
                id_col,
                "avg_indicated_vars_pct",
                "co_missingness_concentration",
                "missing_variable_breadth",
                "pattern_entropy",
                "max_pairwise_co_missingness",
            ]
        )
        .sort(id_col)
    )

    return result


if __name__ == "__main__":
    # Example usage
    vitals_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 6],
            "heartrate": [None, None, 80, None, None, None, 80],
            "resprate": [None, 3, 20, None, None, None, None],
            "o2sat": [None, None, 98, None, None, None, None],
            "sbp": [None, None, 120, None, None, None, None],
        }
    )
    mask_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 6],
            "clock_no": [1, 2, 3, 4, 5, 6, 7],
            "clock": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
            ],
            "heartrate": [1, 1, 0, 1, 1, 1, 0],
            "resprate": [1, 0, 0, 1, 1, 1, 1],
            "o2sat": [1, 1, 0, 1, 1, 1, 1],
            "sbp": [1, 1, 0, 1, 1, 1, 1],
        }
    )
    all_null_rows, percentage_null_rows = analyze_all_null_rows(mask_df)
    print(
        f"All null rows: {all_null_rows}, Percentage of all null rows: {percentage_null_rows:.2f}%"
    )
    per_id_df = analyze_all_null_rows_per_id(mask_df)
    print(per_id_df)
    indicated_vars_count = analyze_row_imperfection(mask_df)
    print(f"indicated parameter observations at a certain point in time: {indicated_vars_count}")
    per_id_indicated_vars = analyze_row_imperfection_per_id(mask_df)
    print(per_id_indicated_vars)
