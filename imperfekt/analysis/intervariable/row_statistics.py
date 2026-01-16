import polars as pl


def analyze_all_null_rows(
    mask_df: pl.DataFrame,
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str = None,
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
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str = None,
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
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str = None,
    save_results: bool = True,
) -> tuple[int, float]:
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
        return 0, 0.0

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
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str = None,
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
