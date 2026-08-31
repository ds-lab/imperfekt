import polars as pl

from imperfekt.analysis.utils import pretty_printing

NUMERIC_DTYPES = [pl.Float64, pl.Float32, pl.Int64, pl.Int32]

# Seconds per unit, for expressing value_slope as a rate of change.
SLOPE_TIME_UNITS = {"second": 1.0, "minute": 60.0, "half-hour": 1800.0, "hour": 3600.0, "day": 86400.0}

CASE_VALUE_METRIC_COLS = [
    "value_mean",
    "value_min",
    "value_max",
    "value_iqr",
    "value_slope",
    "value_first",
]


def compute_case_value_metrics(
    df: pl.DataFrame,
    cols: list[str],
    id_col: str = "id",
    clock_col: str = "clock",
    clock_no_col: str = "clock_no",
    min_obs_spread: int = 2,
    min_obs_slope: int = 2,
    slope_time_unit: str = "hour",
) -> pl.DataFrame:
    """
    Compute per-case, per-variable summary measures of the *observed values*.

    Metrics computed per (id, variable), over observed (non-null) values only:
        value_mean  : arithmetic mean
        value_min   : smallest observed value
        value_max   : largest observed value
        value_iqr   : Q75 - Q25 (linear interpolation); within-case spread.
                      Requires min_obs_spread observations.
        value_slope : OLS slope of value on time — a rate of change, in the variable's
                      own units per slope_time_unit (default: per hour).
        value_first : first observed value in clock order (baseline / initial reading)

    A case with no observed values of a variable yields an all-null row rather than no
    row, so that differential availability stays visible to the definedness test in
    the group comparison rather than silently shrinking the sample.

    Parameters:
        df (pl.DataFrame): Frame of observed values.
        cols (list[str]): Variables to summarize. Non-numeric ones are skipped.
        id_col (str): Case identifier column.
        clock_col (str): Timestamp column, used as the time axis for value_slope
            when it is temporal, and to order value_first.
        clock_no_col (str): Fallback time axis for value_slope when clock_col is not
            temporal; value_slope is then per observation and slope_time_unit is ignored.
        min_obs_spread (int): Minimum observations for value_iqr.
        min_obs_slope (int): Minimum observations for value_slope.
        slope_time_unit (str): Time unit of value_slope — "second", "minute", "half-hour", "hour"
            (default) or "day".

    Returns:
        pl.DataFrame: One row per (id, variable) with the six metric columns.
    """
    if slope_time_unit not in SLOPE_TIME_UNITS:
        raise ValueError(
            f"Unsupported slope_time_unit: {slope_time_unit!r}. "
            f"Supported units: {', '.join(SLOPE_TIME_UNITS)}."
        )

    numeric_cols = []
    for c in cols:
        if c not in df.columns:
            pretty_printing.rich_warning(f"Column '{c}' not found in the dataframe — skipped.")
        elif df[c].dtype not in NUMERIC_DTYPES:
            pretty_printing.rich_warning(f"Skipping non-numeric column for case metrics: {c}")
        else:
            numeric_cols.append(c)

    schema = {id_col: df[id_col].dtype if id_col in df.columns else pl.Utf8, "variable": pl.Utf8}
    schema.update({m: pl.Float64 for m in CASE_VALUE_METRIC_COLS})
    if not numeric_cols:
        pretty_printing.rich_warning(
            "No numeric columns available — no case-level value metrics computed."
        )
        return pl.DataFrame(schema=schema)

    df = df.sort([id_col, clock_col])

    # Time axis for the slope: seconds when the clock is temporal, observation index
    # otherwise. time_scale converts the per-second slope into the requested unit; with
    # an index axis there is no duration to convert, so the rate is per observation.
    if df[clock_col].dtype in (pl.Datetime, pl.Date):
        time_expr = pl.col(clock_col).cast(pl.Datetime).dt.timestamp("ms").cast(pl.Float64) / 1000.0
        time_scale = SLOPE_TIME_UNITS[slope_time_unit]
    else:
        time_expr = pl.col(clock_no_col).cast(pl.Float64)
        time_scale = 1.0

    frames = []
    for c in numeric_cols:
        observed = pl.col(c).is_not_null()
        value = pl.col(c).cast(pl.Float64).filter(observed)
        time = time_expr.filter(observed)

        n_obs = value.len()
        # Rate of change per time_scale (e.g. bpm/hour)
        slope = pl.cov(time, value) / time.var() * time_scale

        frames.append(
            df.group_by(id_col)
            .agg(
                value.mean().alias("value_mean"),
                value.min().alias("value_min"),
                value.max().alias("value_max"),
                pl.when(n_obs >= min_obs_spread)
                .then(
                    value.quantile(0.75, interpolation="linear")
                    - value.quantile(0.25, interpolation="linear")
                )
                .alias("value_iqr"),
                pl.when((n_obs >= min_obs_slope) & (time.var() > 0))
                .then(slope)
                .alias("value_slope"),
                value.first().alias("value_first"),
            )
            .with_columns(pl.lit(c, dtype=pl.Utf8).alias("variable"))
        )

    return (
        pl.concat(frames, how="vertical_relaxed")
        .select(list(schema))
        .cast({m: pl.Float64 for m in CASE_VALUE_METRIC_COLS})
        .sort([id_col, "variable"])
    )
