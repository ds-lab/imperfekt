import plotly.graph_objects as go
import polars as pl

############################################################
#   DATETIME DISTRIBUTION ANALYSIS FOR A COLUMN OVER TIME  #
############################################################


def extract_datetime_distribution(
    mask_df: pl.DataFrame,
    col: str,
    clock_col: str = "clock",
    save_path: str = None,
    save_results: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Analyze the distribution of a specified column over time, grouped by month, weekday, and hour,
    useful for MAR and MNAR scenarios.
    Uncovers provider-level/system-level patterns in missingness or noise.

    Parameters:
        mask_df (pl.DataFrame): A DataFrame with a mask indicating imperfect values (1=missing/noisy/indicated, 0=observed/normal).
        col (str): The name of the column to analyze.
        clock_col (str): The name of the column containing time information.
        save_path (str): The path to save the results. If None, results will not be saved.
        save_results (bool): Whether to save the results. Default is False.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Three DataFrames containing the mean and count of the specified column"""
    if clock_col not in mask_df.columns:
        raise ValueError(f"clock_col '{clock_col}' not found in DataFrame")
    df = mask_df.with_columns(
        pl.col(clock_col).dt.month().alias("month"),
        pl.col(clock_col).dt.weekday().alias("weekday"),
        pl.col(clock_col).dt.hour().alias("hour"),
        month_name=(pl.col(clock_col).dt.strftime("%B")),
        weekday_name=(pl.col(clock_col).dt.strftime("%A")),
    )

    months_df = (
        df.group_by(["month", "month_name"])
        .agg(
            pl.col(col).mean().alias("mean"),
            pl.col(col).sum().alias("count"),
        )
        .sort("month")
    )

    weekdays_df = (
        df.group_by(["weekday", "weekday_name"])
        .agg(
            pl.col(col).mean().alias("mean"),
            pl.col(col).sum().alias("count"),
        )
        .sort("weekday")
    )

    daytime_df = (
        df.group_by("hour")
        .agg(
            pl.col(col).mean().alias("mean"),
            pl.col(col).sum().alias("count"),
        )
        .sort("hour")
    )

    if save_results and save_path:
        months_df.write_csv(f"{save_path}/{col}_monthly_distribution.csv")
        weekdays_df.write_csv(f"{save_path}/{col}_weekly_distribution.csv")
        daytime_df.write_csv(f"{save_path}/{col}_daytime_distribution.csv")

    return months_df, weekdays_df, daytime_df


def visualize_month_daytime_heatmap(
    mask_df: pl.DataFrame,
    col: str,
    clock_col: str = "clock",
    renderer: str = "browser",
    save_path: str = None,
    save_results: bool = True,
) -> go.Figure:
    """
    Visualize the provider/system-level distribution of a specified column over time, grouped by month, weekday, and hour.

    Parameters:
        df (pl.DataFrame): The DataFrame containing the data.
        col (str): The name of the column to analyze.
        clock_col (str): The name of the column containing time information.
        renderer (str): The renderer to use for Plotly. Default is 'browser'.
        save_path (str): The path to save the figure. If None, the figure will not be saved.
        save_results (bool): Whether to save the figure. Default is False.

    Returns:
        None: Displays a heatmap of the imperfect percentage of the specified column by month and hour.
    """
    if clock_col not in mask_df.columns:
        raise ValueError(f"clock_col '{clock_col}' not found in DataFrame")
    df = mask_df.with_columns(
        pl.col(clock_col).dt.month().alias("month"),
        pl.col(clock_col).dt.weekday().alias("weekday"),
        pl.col(clock_col).dt.hour().alias("hour"),
        month_name=(pl.col(clock_col).dt.strftime("%B")),
        weekday_name=(pl.col(clock_col).dt.strftime("%A")),
    )

    # Group by month and daytime, if no data in a group then fill with 0
    month_hour_df = (
        df.group_by(["month", "month_name", "hour"])
        .agg(
            pl.col(col).mean().alias("indicated_pct"),
        )
        .sort(["month", "hour"])
    )

    # Always use months 1-12 and hours 0-23 for heatmap axes
    all_months = list(range(1, 13))
    all_hours = list(range(24))
    all_combinations = [(m, h) for m in all_months for h in all_hours]

    full_df = pl.DataFrame(all_combinations, schema=["month", "hour"], orient="row")
    full_df = full_df.join(month_hour_df, on=["month", "hour"], how="left").fill_null(1.0)

    month_hour_df = full_df.with_columns(pl.col("indicated_pct").cast(pl.Float64))

    # Plot heatmap for month and hour
    fig = go.Figure(
        data=go.Heatmap(
            z=month_hour_df["indicated_pct"]
            .to_numpy()
            .reshape(len(month_hour_df["month"].unique()), 24),
            x=month_hour_df["hour"].unique().to_list(),
            y=month_hour_df["month"].unique().to_list(),
            colorscale="Viridis",
            colorbar=dict(title="Imperfect Percentage"),
        )
    )
    fig.update_layout(
        title=f"Imperfect Percentage of {col} by Month and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Month",
    )

    if renderer:
        fig.show(renderer=renderer)

    if save_results and save_path:
        fig.write_image(save_path)
        print(f"Month x Daytime Heatmap saved to {save_path}")

    return fig


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

    date_df = extract_datetime_distribution(mask_df, col="variable", clock_col="clock")
    print(date_df)
