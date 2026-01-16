import polars as pl

############################################################
#               COLUMN COMPLETENESS ANALYSIS               #
############################################################


def analyze_column_imperfection(
    mask_df: pl.DataFrame,
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    save_path: str = None,
    save_results: bool = True,
) -> pl.DataFrame:
    """
    Analyze the completeness of each column in the DataFrame and calculate the percentage of non-null values per column.

    This function calculates the number of null and non-null values,
    the percentage of null values, data types, and the number of unique values for each column
    in the provided DataFrame. The results are returned as a summary DataFrame
    sorted by the percentage of imperfect values in descending order.

    Parameters:
        mask_df (pl.DataFrame): The input DataFrame containing the data to analyze.
        cols (list, optional): A list of column names to analyze. If not provided, all columns in the DataFrame will be analyzed.
        id_col (str, optional): The name of the column that serves as the unique identifier. Defaults to "id".
        clock_col (str, optional): The name of the column that serves as the time ordering. Defaults to "clock".
        clock_no_col (str, optional): The name of the column that serves as the time ordering number. Defaults to "clock_no".
        save_path (str, optional): The path to save the summary DataFrame as a CSV file. If None, the DataFrame will not be saved.
        save_results (bool, optional): Whether to save the summary DataFrame as a CSV file. Defaults to True.
    Returns:
        DataFrame: A summary DataFrame containing the following columns:
            - column: The name of the column.
            - indicated_count: The number of imperfect values in the column.
            - indicated_pct: The percentage of imperfect values in the column.
            - non_indicated_count: The number of non-imperfect values in the column.
            - dtype: The data type of the column.
            - unique_values: The number of unique values in the column.
    """
    if cols is None:
        cols = [c for c in mask_df.columns if c not in [id_col, clock_col, clock_no_col]]
    if not all(c in mask_df.columns for c in cols) or id_col not in mask_df.columns:
        raise ValueError(
            f"Columns {cols} not found in DataFrame. Available columns: {mask_df.columns}"
        )

    # Filter df columns to cols
    if cols:
        mask_df = mask_df.select(cols)

    # Null count by building the sum over each column of the mask
    null_count = mask_df.select(
        [pl.col(col).sum().alias(f"{col}_indicated_count") for col in cols]
    ).row(0)

    summary_df = pl.DataFrame(
        {
            "column": mask_df.columns,
            "indicated_count": null_count,
            "indicated_pct": [count / mask_df.height * 100 for count in null_count],
            "non_indicated_count": [mask_df.height - count for count in null_count],
        }
    )

    summary_df = summary_df.sort("indicated_pct", descending=True)

    if save_path and save_results:
        summary_df.write_csv(save_path)
        print(f"Column completeness summary saved to {save_path}")

    return summary_df


def analyze_column_imperfection_per_id(
    mask_df: pl.DataFrame,
    cols: list = None,
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    threshold: float = 5,
    save_path: str = None,
    save_results: bool = True,
) -> pl.DataFrame:
    """
    Analyze the completeness of specified columns for each unique ID in the given DataFrame.
    This function calculates the count and percentage of imperfect values for each column
    specified in `cols` grouped by the unique identifier `id`. It also computes the total
    number of rows per `id` and joins this information to provide a comprehensive view
    of imperfection per column.

    Parameters:
        mask_df (pl.DataFrame): The input DataFrame containing the data to analyze.
        cols (list, optional): A list of column names to analyze for imperfection. If not provided, all columns in the DataFrame will be analyzed.
        id_col (str, optional): The name of the column that serves as the unique identifier.
                             Defaults to "id".
        clock_col (str, optional): The name of the column that serves as the time ordering. Defaults to "clock".
        clock_no_col (str, optional): The name of the column that serves as the time ordering number. Defaults to "clock_no".
        threshold (float, optional): The percentage threshold to determine if a column's
                                     imperfection is significant. Defaults to 5.
        save_path (str, optional): The path to save the summary DataFrame as a CSV file.
        save_results (bool, optional): Whether to save the summary DataFrame as a CSV file. Defaults to True.

    Returns:
        pl.DataFrame: A DataFrame containing the following columns for each `id`:
            - `<column>_indicated_count`: The count of imperfect values for each column.
            - `total_count`: The total number of rows for each `id`.
            - `<column>_indicated_pct`: The percentage of imperfect values for each column.
            - `<column>_above_<threshold>pct_threshold`: A boolean indicating if the
              percentage of imperfect values for that column exceeds the specified threshold.
        imperfect percentages are calculated as `(indicated_count / total_count) * 100`.
        Any NaN or None values are filled with 0 for IDs with no imperfect values.
    """
    if cols is None:
        cols = [c for c in mask_df.columns if c not in [id_col, clock_col, clock_no_col]]
    if not all(c in mask_df.columns for c in cols) or id_col not in mask_df.columns:
        raise ValueError(
            f"Columns {cols} not found in DataFrame. Available columns: {mask_df.columns}"
        )

    # 1. Group by id and calculate imperfect counts for each column
    grouped_df = mask_df.group_by(id_col).agg(
        [pl.col(col).sum().alias(f"{col}_indicated_count") for col in cols]
    )

    # 2. Get total rows per id
    total_rows_per_id = mask_df.group_by(id_col).agg(pl.len().alias("total_count"))

    # 3. Join the total rows with the grouped imperfect counts
    #    and calculate the percentage of imperfect values
    grouped_df = (
        grouped_df.join(total_rows_per_id, on=id_col, how="inner")
        .with_columns(
            [
                (pl.col(f"{col}_indicated_count") / pl.col("total_count") * 100).alias(
                    f"{col}_indicated_pct"
                )
                for col in cols
            ]
        )
        .fill_null(0)
    )  # Fill NaN/None with 0 for ids with no imperfect values

    # 4. Show if any column has imperfect values above the threshold
    grouped_df = grouped_df.with_columns(
        [
            (pl.col(f"{col}_indicated_pct") > threshold).alias(
                f"{col}_above_{threshold}pct_threshold"
            )
            for col in cols
        ]
    )

    if save_path and save_results:
        desc = grouped_df.describe()
        desc.write_csv(save_path)
        print(f"Column completeness per ID summary saved to {save_path}")

    return grouped_df


if __name__ == "__main__":
    # Example usage
    pl.Config.set_tbl_cols(20)  # Set Polars to show more rows in output
    df = pl.DataFrame(
        {
            "id": ["1", "2", "2", "4", "5"],
            "clock_no": [1, 1, 2, 1, 1],
            "heartrate": [60, None, 70, None, None],
            "resprate": [12, 20, None, 25, None],
            "o2sat": [95, None, None, None, None],
            "sbp": [120, None, 130, None, None],
        }
    )

    mask_df = df.select(
        pl.col("id"),
        pl.col("clock_no"),
        (pl.col("heartrate").is_null().cast(pl.Int8)).alias("heartrate"),
        (pl.col("resprate").is_null().cast(pl.Int8)).alias("resprate"),
        (pl.col("o2sat").is_null().cast(pl.Int8)).alias("o2sat"),
        (pl.col("sbp").is_null().cast(pl.Int8)).alias("sbp"),
    )

    column_stats = analyze_column_imperfection(mask_df)
    print("Column Completeness:")
    print(column_stats)

    column_per_id_stats = analyze_column_imperfection_per_id(mask_df)
    print("Column Completeness per ID:")
    print(column_per_id_stats)
