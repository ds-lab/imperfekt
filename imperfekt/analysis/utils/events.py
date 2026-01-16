import polars as pl


def filter_events_df(
    events_df: pl.DataFrame,
    event_name_col: str = None,
    included_event_names: list = None,
) -> pl.DataFrame:
    """
    Filter events DataFrame based on included event names.

    Parameters:
        events_df (pl.DataFrame): The DataFrame containing event data.
        event_name_col (str): The column name in events_df that contains event names.
        included_event_names (list): A list of event names to include in the filtered DataFrame.
    Returns:
        pl.DataFrame: The filtered DataFrame containing only the specified event names.
    """
    if included_event_names is not None and event_name_col is not None:
        events_df = events_df.filter(pl.col(event_name_col).is_in(included_event_names))

    return events_df


def create_event_mask(
    df: pl.DataFrame,
    events_df: pl.DataFrame,
    id_col: str = "id",
    clock_col: str = "clock",
    window_size: int = 0,
    window_location: str = "both",
    remove_ids_without_events: bool = True,
) -> tuple:
    """
    Creates a mask DataFrame based on events and a specified window size.

    Parameters:
        df (pl.DataFrame): The main DataFrame containing time series data.
        events_df (pl.DataFrame): DataFrame containing event timestamps.
        id_col (str): Column name for unique identifiers in both DataFrames. Default: "id"
        clock_col (str): Column name for timestamps in both DataFrames. Default: "clock"
        window_size (int): Size of the window around each event timestamp. Default: 0 -> Finds closest time match.
        window_location (str): Location of the window relative to the event ('both', 'before', 'after'). Only considered if window_size > 0. Default: "both"
        remove_ids_without_events (bool): If True, removes IDs without events from the mask. Default: True

    Returns:
        pl.DataFrame: A mask DataFrame with the same shape as df, where True indicates
                      rows within the specified window around any event timestamp.
    """
    df = df.sort(by=[id_col, clock_col]).with_row_index("unique_id")
    events_df = events_df.sort(by=[id_col, clock_col]).with_row_index("event_id")

    # Find closest timestamp in df to each events_df
    merged_df = df.join(events_df, on=id_col, how="left", suffix="_event").with_columns(
        (pl.col(clock_col) - pl.col(f"{clock_col}_event")).abs().alias("time_diff")
    )

    if window_size > 0:
        if window_location == "both":
            event_matches = merged_df.filter(pl.col("time_diff") <= window_size)
        elif window_location == "before":
            event_matches = merged_df.filter(
                (pl.col("time_diff") <= window_size)
                & (pl.col(clock_col) < pl.col(f"{clock_col}_event"))
            )
        elif window_location == "after":
            event_matches = merged_df.filter(
                (pl.col("time_diff") <= window_size)
                & (pl.col(clock_col) > pl.col(f"{clock_col}_event"))
            )
    else:
        # minimize time_diff per event, the timestamps don't match exactly
        min_time_diff_df = merged_df.group_by([id_col, "event_id"]).agg(
            pl.col("time_diff").min().alias("min_time_diff")
        )
        event_matches = merged_df.join(
            min_time_diff_df, on=["event_id", id_col], how="left"
        ).filter(pl.col("time_diff") == pl.col("min_time_diff"))

    mask = df.with_columns(
        pl.col("unique_id").is_in(event_matches["unique_id"].implode()).alias("event_mask")
    )

    if remove_ids_without_events:
        # Remove IDs without events (all timestamps are annotated with event_mask 'false')
        temp = mask.select(id_col, "event_mask").group_by(id_col).n_unique()
        temp = temp.filter(pl.col("event_mask") == 2)
        mask = mask.with_columns(pl.col(id_col).is_in(temp[id_col].implode()).alias("id_in_events"))
        mask = mask.filter(pl.col("id_in_events")).drop("id_in_events")

    return event_matches, mask


def calculate_event_percentage(event_mask: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the percentage of events occurring within a specified window.

    Parameters:
        event_mask (pl.DataFrame): The mask DataFrame indicating event occurrences.

    Returns:
        pl.DataFrame: A DataFrame with the event percentage for each unique identifier.
    """
    # Calculate the event percentage
    event_count = event_mask.filter(pl.col("event_mask")).height
    total_count = event_mask.height
    event_percentage = (event_count / total_count) * 100

    return event_percentage
