import polars as pl

from imperfekt.analysis.utils import pretty_printing


def add_rolling_window_features(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
    window_size: int,
    replace_nulls_with_zero: bool = True,
) -> pl.DataFrame:
    """
    Adds rolling window features (sum, variance) for the mask of each variable.
    Sum: Number of imperfections in the past ´window_size´ observations.
    Variance: Volatility of imperfections in the past ´window_size´ observations.

    Args:
        df: The main DataFrame.
        mask: DataFrame with id, clock, and mask columns.
        variable_cols: List of original variable names.
        id_col: The column identifying individual time series.
        clock_col: The time column for joining.
        window_size: The size of the rolling window in number of observations.

    Returns:
        The DataFrame with added rolling window features.
    """
    mask_cols = [f"{col}_mask" for col in variable_cols]
    new_cols_names = []
    rolling_exprs = []

    for col in mask_cols:
        var_name = col.replace("_mask", "")

        # Rolling sum of imperfections (count of missing values in window)
        rolling_sum_col = f"{var_name}_mask_rolling_sum_{window_size}"
        rolling_exprs.append(
            pl.col(col).rolling_sum(window_size=window_size).over(id_col).alias(rolling_sum_col)
        )
        new_cols_names.append(rolling_sum_col)

        # Rolling variance of imperfections (volatility of missingness in window)
        rolling_var_col = f"{var_name}_mask_rolling_var_{window_size}"
        rolling_exprs.append(
            pl.col(col).rolling_var(window_size=window_size).over(id_col).alias(rolling_var_col)
        )
        new_cols_names.append(rolling_var_col)

    # Calculate on the mask DataFrame
    rolling_df = mask.with_columns(rolling_exprs).select(id_col, clock_col, *new_cols_names)

    if replace_nulls_with_zero:
        rolling_df = rolling_df.with_columns(pl.col(new_cols_names).fill_null(0))

    # Join back to the main df
    df = df.drop(new_cols_names, strict=False)
    df = df.join(rolling_df, on=[id_col, clock_col], how="left")

    return df


def add_exponential_moving_average(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
    span: int = None,
    alpha: float = None,
    by_half_life: str = None,
) -> pl.DataFrame:
    """
    Adds exponential moving average (EWMA) features for the mask of each variable.

    Args:
        df: The main DataFrame.
        mask: DataFrame with id, clock, and mask columns.
        variable_cols: List of original variable names.
        id_col: The column identifying individual time series.
        clock_col: The time column for joining.
        span: The span for the EWMA calculation.
        alpha: The smoothing factor for the EWMA.
        by_half_life: The half-life for the EWMA calculation, e.g.: '1s': 1 second, '2m': 2 minutes, '24h': 24 hours. Takes the clock_col into account.

    Returns:
        The DataFrame with added EWMA features.
    """
    mask_cols = [f"{col}_mask" for col in variable_cols]
    new_cols_names = [f"{col}_mask_ewma_{alpha}" for col in variable_cols]

    if span or alpha:
        try:
            mask_cols = [f"{col}_mask" for col in variable_cols]
            new_cols_names = [
                f"{col}_mask_ewma_{alpha if alpha else span}" for col in variable_cols
            ]
            ewma_exprs = [
                pl.col(col)
                .ewm_mean(span=span, alpha=alpha, adjust=False)
                .over(id_col)
                .alias(col.replace("_mask", f"_mask_ewma_{alpha if alpha else span}"))
                for col in mask_cols
            ]
        except ValueError as e:
            pretty_printing.rich_error(e)
            return df
    elif by_half_life:
        try:
            mask_cols = [f"{col}_mask" for col in variable_cols]
            new_cols_names = [f"{col}_mask_ewma_{by_half_life}" for col in variable_cols]
            ewma_exprs = [
                pl.col(col)
                .ewm_mean_by(
                    by=clock_col, half_life=by_half_life
                )  # other possibility, but requires careful consideration of time, suggested for non-equi-distant timestamps
                .over(id_col)
                .alias(col.replace("_mask", f"_mask_ewma_{by_half_life}"))
                for col in mask_cols
            ]
        except ValueError as e:
            pretty_printing.rich_error(e)
            return df
    else:
        pretty_printing.rich_error(
            "Either span, alpha or halflife must be provided for EWMA calculation."
        )

    # Calculate on the mask DataFrame
    ewma_df = mask.with_columns(ewma_exprs).select(id_col, clock_col, *new_cols_names)

    # Join back to the main df
    df = df.drop(new_cols_names, strict=False)
    df = df.join(ewma_df, on=[id_col, clock_col], how="left")

    return df
