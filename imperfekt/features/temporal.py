import polars as pl


def add_lag_mask(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
    lag: int = 1,
    replace_nulls_with_zero: bool = True,
) -> pl.DataFrame:
    """
    Adds lagged versions of mask columns to the DataFrame.

    This function calculates the lags on the separate mask DataFrame and then
    joins them back, ensuring alignment and avoiding redundant joins in a pipeline.

    Args:
        df: The main DataFrame.
        mask: DataFrame containing the id, clock, and mask columns.
        variable_cols: The list of original variable names to find corresponding masks.
        id_col: The column identifying individual time series.
        clock_col: The time column, used for joining.
        lag: The lag to apply. Default is 1.
        replace_nulls_with_zero: Whether to replace nulls with zero in the lagged columns.

    Returns:
        The DataFrame with added lagged mask columns.
    """
    mask_cols = [f"{col}_mask" for col in variable_cols]
    lagged_cols_names = [f"{col}_mask_lag_{lag}" for col in variable_cols]

    # The mask DataFrame already contains id_col and clock_col.
    # We can directly calculate the lags on it.
    # The mask is assumed to be sorted by id_col, clock_col from FeatureGenerator
    lagged_masks = mask.with_columns(
        [pl.col(col).shift(lag).over(id_col).alias(f"{col}_lag_{lag}") for col in mask_cols]
    ).select(id_col, clock_col, *lagged_cols_names)

    if replace_nulls_with_zero:
        lagged_masks = lagged_masks.fill_null(0)

    # Join the results back to the main DataFrame
    # Drop columns first to make the operation idempotent
    df = df.drop(lagged_cols_names, strict=False)
    df = df.join(lagged_masks, on=[id_col, clock_col], how="inner")

    return df


def add_consecutive_counts(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
) -> pl.DataFrame:
    """
    Adds columns for the count of consecutive imperfections for each variable.

    Example: For a mask [0, 1, 1, 1, 0, 1], the feature is [0, 1, 2, 3, 0, 1].

    Args:
        df: The main DataFrame.
        mask: DataFrame with id, clock, and mask columns.
        variable_cols: List of original variable names.
        id_col: The column identifying individual time series.
        clock_col: The time column for joining.

    Returns:
        The DataFrame with added consecutive count columns.
    """
    mask_cols = [f"{col}_mask" for col in variable_cols]
    new_cols_names = [f"{col}_mask_consecutive" for col in variable_cols]

    # Create expressions for each mask column
    consecutive_count_exprs = []
    for col in mask_cols:
        # ID for each block of consecutive identical values
        block_id = pl.col(col).rle_id()

        # Calculate the cumulative sum of the mask within each block
        # and multiply by the mask itself to ensure counts are zero when mask is zero.
        consecutive_count = pl.col(col).cum_sum().over([id_col, block_id]) * pl.col(col)
        consecutive_count_exprs.append(
            consecutive_count.alias(col.replace("_mask", "_mask_consecutive"))
        )

    # Calculate the new features on the mask DataFrame
    consecutive_counts_df = mask.with_columns(consecutive_count_exprs).select(
        id_col, clock_col, *new_cols_names
    )

    # Join the results back to the main DataFrame
    df = df.drop(new_cols_names, strict=False)
    df = df.join(consecutive_counts_df, on=[id_col, clock_col], how="left")

    return df


def add_time_since(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
    cap_seconds: int = None,
) -> pl.DataFrame:
    """
    Adds columns for time since last imperfect and non-imperfect measurement.

    Args:
        df: The main DataFrame.
        mask: DataFrame with id, clock, and mask columns.
        variable_cols: List of original variable names.
        id_col: The column identifying individual time series.
        clock_col: The time column for calculations.
        cap_seconds: Optional value to cap the time difference in seconds.

    Returns:
        The DataFrame with added time-since columns.
    """
    mask_cols = [f"{col}_mask" for col in variable_cols]
    new_cols_names = []

    time_since_exprs = []

    for col in mask_cols:
        var_name = col.replace("_mask", "")

        # Time since last imperfect measurement (mask == 1)
        last_imperfect_time = (
            pl.when(pl.col(col) == 1).then(pl.col(clock_col)).forward_fill().over(id_col)
        )
        time_since_imperfect = (
            (pl.col(clock_col) - last_imperfect_time)
            .dt.total_seconds()
            .fill_null(0)  # Fill initial nulls with 0
        )

        # Time since last non-imperfect measurement (mask == 0)
        last_non_imperfect_time = (
            pl.when(pl.col(col) == 0).then(pl.col(clock_col)).forward_fill().over(id_col)
        )
        time_since_non_imperfect = (
            (pl.col(clock_col) - last_non_imperfect_time)
            .dt.total_seconds()
            .fill_null(0)  # Fill initial nulls with 0
        )

        if cap_seconds:
            time_since_imperfect = time_since_imperfect.clip(upper_bound=cap_seconds)
            time_since_non_imperfect = time_since_non_imperfect.clip(upper_bound=cap_seconds)

        new_imperfect_col_name = f"{var_name}_time_since_imperfect"
        new_non_imperfect_col_name = f"{var_name}_time_since_non_imperfect"

        time_since_exprs.append(time_since_imperfect.alias(new_imperfect_col_name))
        time_since_exprs.append(time_since_non_imperfect.alias(new_non_imperfect_col_name))

        new_cols_names.extend([new_imperfect_col_name, new_non_imperfect_col_name])

    # Calculate on the mask DataFrame which contains the necessary columns
    time_since_df = mask.with_columns(time_since_exprs).select(id_col, clock_col, *new_cols_names)

    # Join back to the main df
    df = df.drop(new_cols_names, strict=False)
    df = df.join(time_since_df, on=[id_col, clock_col], how="left")

    return df
