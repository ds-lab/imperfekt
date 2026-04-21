import polars as pl


def _compute_intervals(df: pl.DataFrame, id_col: str, clock_col: str) -> pl.DataFrame:
    """
    Compute per-entity inter-observation intervals in seconds.

    Handles both Datetime and numeric clock columns. Returns the sorted DataFrame
    with an added ``interval_seconds`` column. The first observation per entity has
    a null interval.
    """
    sorted_df = df.sort([id_col, clock_col])
    clock_dtype = sorted_df[clock_col].dtype
    is_temporal = (
        clock_dtype == pl.Datetime
        or clock_dtype == pl.Date
        or clock_dtype == pl.Duration
        or str(clock_dtype).startswith("Datetime")
    )

    if is_temporal:
        interval_expr = (
            pl.col(clock_col)
            .diff()
            .over(id_col)
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("interval_seconds")
        )
    else:
        interval_expr = (
            pl.col(clock_col)
            .diff()
            .over(id_col)
            .cast(pl.Float64)
            .alias("interval_seconds")
        )

    return sorted_df.with_columns(interval_expr)


def add_interval_features(
    df: pl.DataFrame,
    id_col: str,
    clock_col: str,
) -> pl.DataFrame:
    """
    Adds row-level irregularity features derived from inter-observation intervals.

    Features added:
        - ``interval_seconds``: gap to the previous observation for this entity (null
          for each entity's first row).
        - ``interval_z_score``: how many entity-level standard deviations this gap is
          from the entity's mean gap  ``(Δt - μ) / σ``.  Null when σ = 0 or when
          interval_seconds is null.
        - ``interval_cv_local``: rolling coefficient of variation (std / mean) of the
          last 5 intervals per entity, capturing *local* rhythm irregularity.  Null
          when the rolling mean is zero.

    Args:
        df: The main DataFrame, sorted by id_col and clock_col.
        id_col: Column identifying individual time series.
        clock_col: Time column (Datetime or numeric seconds).

    Returns:
        The DataFrame with three new columns added.
    """
    new_cols = ["interval_seconds", "interval_z_score", "interval_cv_local"]

    interval_df = _compute_intervals(df, id_col, clock_col)

    # Entity-level mean and std of intervals
    entity_stats = (
        interval_df
        .group_by(id_col)
        .agg(
            pl.col("interval_seconds").mean().alias("_entity_mean"),
            pl.col("interval_seconds").std().alias("_entity_std"),
        )
    )

    interval_df = interval_df.join(entity_stats, on=id_col, how="left")

    interval_df = interval_df.with_columns(
        pl.when(
            pl.col("_entity_std").is_not_null() & (pl.col("_entity_std") > 0)
        )
        .then((pl.col("interval_seconds") - pl.col("_entity_mean")) / pl.col("_entity_std"))
        .otherwise(None)
        .alias("interval_z_score"),
    )

    # Local CV: rolling std / rolling mean over the last 5 intervals
    local_cv_expr = (
        pl.when(
            pl.col("interval_seconds").rolling_mean(window_size=5).over(id_col) > 0
        )
        .then(
            pl.col("interval_seconds").rolling_std(window_size=5).over(id_col)
            / pl.col("interval_seconds").rolling_mean(window_size=5).over(id_col)
        )
        .otherwise(None)
        .alias("interval_cv_local")
    )
    interval_df = interval_df.with_columns(local_cv_expr)

    result = interval_df.select([id_col, clock_col] + new_cols)

    df = df.drop(new_cols, strict=False)
    df = df.join(result, on=[id_col, clock_col], how="left")

    return df


def add_windowed_acceleration(
    df: pl.DataFrame,
    id_col: str,
    clock_col: str,
    window_size: int = 5,
) -> pl.DataFrame:
    """
    Adds windowed interval-acceleration features.

    "Acceleration" here is the first-order difference of the interval sequence —
    how fast the rhythm is *changing*:

        acceleration_i = interval_i - interval_{i-1}

    Positive acceleration means the gap just grew (slowing down / spacing out).
    Negative acceleration means the gap just shrank (speeding up / bunching together).

    Features added (all computed per entity using a rolling window of ``window_size``):
        - ``interval_acceleration``: raw Δ(interval) = interval_i − interval_{i-1}.
          Null for the first two observations per entity.
        - ``rolling_mean_acceleration_{window_size}``: smoothed trend of acceleration
          over the window — are gaps steadily growing or shrinking?
        - ``rolling_abs_acceleration_{window_size}``: rolling mean of |acceleration|
          — magnitude of rhythm change regardless of direction.
        - ``rolling_std_acceleration_{window_size}``: rolling std of acceleration —
          how *volatile* the rhythm change is within the window.

    Args:
        df: The main DataFrame.
        id_col: Column identifying individual time series.
        clock_col: Time column (Datetime or numeric seconds).
        window_size: Number of observations for the rolling statistics. Default 5.

    Returns:
        The DataFrame with four new columns added.
    """
    accel_col = "interval_acceleration"
    mean_col = f"rolling_mean_acceleration"
    abs_col = f"rolling_abs_acceleration"
    std_col = f"rolling_std_acceleration"
    new_cols = [accel_col, mean_col, abs_col, std_col]

    interval_df = _compute_intervals(df, id_col, clock_col)

    interval_df = interval_df.with_columns(
        pl.col("interval_seconds")
        .diff()
        .over(id_col)
        .alias(accel_col)
    )

    interval_df = interval_df.with_columns(
        pl.col(accel_col)
        .rolling_mean(window_size=window_size)
        .over(id_col)
        .alias(mean_col),
        pl.col(accel_col)
        .abs()
        .rolling_mean(window_size=window_size)
        .over(id_col)
        .alias(abs_col),
        pl.col(accel_col)
        .rolling_std(window_size=window_size)
        .over(id_col)
        .alias(std_col),
    )

    result = interval_df.select([id_col, clock_col] + new_cols)

    df = df.drop(new_cols, strict=False)
    df = df.join(result, on=[id_col, clock_col], how="left")

    return df
