from datetime import timedelta

import polars as pl

############################################################
#      WINDOWED TEMPORAL CONTEXT AROUND IMPERFECT VALUES       #
############################################################


def extract_values_near_indicated(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    col: str = None,
    id_col: str = "id",
    clock_col: str = "clock",
    window: timedelta = timedelta(minutes=5),
    window_location: str = "both",
) -> pl.DataFrame:
    """
    Analyze the distribution of values around imperfect instances in a specified column over a specified time window,
    useful for analyzing MNAR (Missing Not At Random) scenarios.

    Parameters:
        df (pl.DataFrame): The DataFrame containing the data.
        mask_df (pl.DataFrame): A DataFrame with a mask indicating imperfect values (1=missing/noisy/indicated, 0=observed/normal).
        col (str): The name of the column to analyze for imperfect values.
        id_col (str): The name of the column containing unique identifiers for grouping. Default is 'id'.
        clock_col (str): The name of the column containing time information.
        window (timedelta): The time window around each imperfect instance to consider for aggregation.
        window_location (str): The location of the window relative to the imperfect instance.
            Options are 'before', 'after', or 'both'. Default is 'both'.

    Returns:
        pl.DataFrame: A DataFrame containing the imperfect time, count of values around the imperfect instance, and the aggregated value.
    """
    if col is None:
        raise ValueError("col must be specified")
    if clock_col not in df.columns:
        raise ValueError(f"clock_col '{clock_col}' not found in DataFrame")

    # 1. Get imperfect instances
    indicated_df = mask_df.filter(pl.col(col) == 1).select(
        [
            pl.col(id_col),
            pl.col(clock_col).alias("indicated_time"),
        ]
    )

    if indicated_df.is_empty():
        return pl.DataFrame(
            {id_col: [], "indicated_time": [], clock_col: [], col: []},
            schema=[id_col, "indicated_time", clock_col, col],
        )

    # Create start and end times for the window based on window_location
    if window_location == "before":
        indicated_df = indicated_df.with_columns(
            (pl.col("indicated_time") - window).alias("start_time"),
            pl.col("indicated_time").alias("end_time"),
        )
    elif window_location == "after":
        indicated_df = indicated_df.with_columns(
            pl.col("indicated_time").alias("start_time"),
            (pl.col("indicated_time") + window).alias("end_time"),
        )
    else:  # 'both'
        indicated_df = indicated_df.with_columns(
            (pl.col("indicated_time") - window).alias("start_time"),
            (pl.col("indicated_time") + window).alias("end_time"),
        )

    # Make sure the time columns are in the correct format
    indicated_df = indicated_df.with_columns(
        pl.col("indicated_time").cast(pl.Datetime("ms", "UTC")),
        pl.col("start_time").cast(pl.Datetime("ms", "UTC")),
        pl.col("end_time").cast(pl.Datetime("ms", "UTC")),
    )

    # Join the original DataFrame with the indicated_df to collect values around imperfect instances
    result_df = (
        indicated_df.join(df, on=id_col, how="inner")  # step 1
        .filter(pl.col(clock_col).is_between(pl.col("start_time"), pl.col("end_time")))  # step 2
        .select(
            pl.col(id_col),
            pl.col("indicated_time"),
            pl.col(clock_col),
            pl.col(col),
        )
        .filter(pl.col("indicated_time") != pl.col(clock_col))  # Filter out imperfect times
        .group_by([pl.col(id_col), pl.col(clock_col)])
        .agg(
            pl.col(col).max(),
            pl.col("indicated_time").implode().alias("indicated_times"),
        )  # indicated_time values are collected into a list
    )
    return result_df


if __name__ == "__main__":
    # Example usage
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b"],
            "clock": [
                "2023-10-01 12:01:00",
                "2023-10-01 12:02:00",
                "2023-10-01 12:03:00",
                "2023-10-01 12:04:00",
                "2023-10-01 12:05:00",
                "2023-10-01 12:06:00",
            ],
            "variable": [1, None, 2, None, 3, None],
        }
    )
    df = df.with_columns(
        [
            pl.col("clock").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )
    mask_df = df.with_columns(pl.col("variable").is_null().cast(pl.Int8).alias("variable"))

    result = extract_values_near_indicated(
        df, mask_df, clock_col="clock", col="variable", window=timedelta(minutes=2)
    )
    print(result)
