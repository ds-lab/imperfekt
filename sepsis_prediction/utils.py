import polars as pl

from imperfekt.features.core import FeatureGenerator


def generate_cohort(
    current_model_input_df: pl.DataFrame,
    first_minute: int = 30,
    minimum_dur: int = 15,
    minimum_count: int = 10,
) -> pl.DataFrame:
    """Generate cohort by filtering to first N minutes and minimum duration and count.

    Parameters:
        current_model_input_df (pl.DataFrame): Input DataFrame containing model input data.
        first_minute (int): Keep data only within the first N minutes per id.
        minimum_dur (int): Minimum duration (in minutes) per id to be included.
        minimum_count (int): Minimum number of records per id to be included.

    Returns:
        pl.DataFrame: Filtered DataFrame containing the cohort.
    """
    current_model_input_df = generate_clock_no_col(current_model_input_df)

    # Filter to keep only records to only include the first 30 minutes per id but minimum 15 minutes and min count = 10
    current_model_input_df = current_model_input_df.with_columns(
        [
            pl.col("clock").min().over("id").alias("min_clock_per_id"),
            pl.col("clock").max().over("id").alias("max_clock_per_id"),
        ]
    )
    current_model_input_df = current_model_input_df.filter(
        (
            (pl.col("clock") - pl.col("min_clock_per_id")) <= pl.duration(minutes=first_minute)
        )  # first 30 minutes
        & (
            (pl.col("max_clock_per_id") - pl.col("min_clock_per_id"))
            >= pl.duration(minutes=minimum_dur)
        )  # at least 15 minutes total
    ).select(pl.exclude("min_clock_per_id", "max_clock_per_id"))

    current_model_input_df = current_model_input_df.with_columns(
        [pl.count("clock").over("id").alias("count_per_id")]
    )

    current_model_input_df = current_model_input_df.filter(
        pl.col("count_per_id") >= minimum_count
    ).select(pl.exclude(["count_per_id"]))
    print(current_model_input_df.shape)

    # Count unique ids per sepsis outcome
    cohort_stats = current_model_input_df.group_by("sepsis_outcome").agg(
        pl.n_unique("id").alias("unique_ids"),
        (pl.n_unique("id") / current_model_input_df.n_unique("id")).alias("rate"),
    )
    print(cohort_stats)

    return current_model_input_df


def generate_imperfekt_non_seq_features(
    df: pl.DataFrame,
    target_col: str = "ROSC_at_ED_arrival",
    id_col: str = "id",
    clock_col: str = "clock",
    cols: list = None,
    fill_nulls: bool = False,
    feature_name_filepath: str = None,
) -> pl.DataFrame:
    """
    Generates Imperfekt features and aggregates them per encounter.
    Parameters:
        df (pl.DataFrame): Input DataFrame containing vitals and target variable.
        target_col (str): Name of the target variable column.
        id_col (str): Name of the encounter ID column.
        clock_col (str): Name of the time column.
        cols (list): List of vital sign columns to generate features for.
        fill_nulls (bool): Whether to fill null values in the resulting features.
        feature_name_filepath (str): File path to save the generated feature names.

    Returns:
        pl.DataFrame: DataFrame containing aggregated Imperfekt features per encounter.
    """
    if cols is None:
        raise ValueError("Please provide a list of vital sign columns to generate features for.")
    print("Generating Imperfekt features...")
    fg = FeatureGenerator(
        df=df,
        id_col=id_col,
        clock_col=clock_col,
        variable_cols=cols,
    )
    fg = (
        fg.add_binary_masks()
        .add_circular_features()
        .add_temporal_features()
        .add_row_imperfection_pct()
    )

    imperfekt_df = fg.df

    feature_cols = [
        col
        for col in imperfekt_df.columns
        if col not in df.columns and col != id_col and col != target_col
    ]

    imperfekt_features, _ = _non_seq_feature_generation(
        df=imperfekt_df, cols=feature_cols, id_col=id_col, target_col=target_col
    )

    if fill_nulls:
        imperfekt_features = imperfekt_features.fill_null(value=-1000000).fill_nan(value=-1000000)

    # Write feature names to file for reference from imperfekt_feature cols
    if feature_name_filepath is not None:
        with open(feature_name_filepath, "w") as f:
            for feature_name in imperfekt_features.columns:
                if feature_name not in [id_col, target_col]:
                    f.write(f"{feature_name}\n")

    return imperfekt_features


def generate_baseline_non_seq_features(
    input_df: pl.DataFrame,
    target_col: str = "ROSC_at_ED_arrival",
    cols: list = None,
    id_col: str = "id",
    fill_nulls: bool = False,
    feature_name_filepath: str = None,
):
    """Generates baseline features by aggregating vital signs per encounter.

    Parameters:
        input_df (pl.DataFrame): Input DataFrame containing vitals and target variable.
        target_col (str): Name of the target variable column.
        cols (list): List of vital sign columns to aggregate.

    Returns:
        pl.DataFrame: DataFrame containing aggregated baseline features per encounter.
    """
    if cols is None:
        raise ValueError("Please provide a list of vital sign columns to generate features for.")
    print("\nGenerating baseline vital sign features...")

    _, baseline_features = _non_seq_feature_generation(
        df=input_df, cols=cols, id_col=id_col, target_col=target_col
    )

    if fill_nulls:
        baseline_features = baseline_features.fill_null(value=-1000000).fill_nan(value=-1000000)

    if feature_name_filepath is not None:
        with open(feature_name_filepath, "w") as f:
            for feature_name in baseline_features.columns:
                if feature_name not in [id_col, target_col]:
                    f.write(f"{feature_name}\n")

    return baseline_features


def _non_seq_feature_generation(
    df: pl.DataFrame, cols: list, id_col: str, target_col: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Helper function to generate non-sequential features by aggregating vital signs per encounter.

    Parameters:
        df (pl.DataFrame): Input DataFrame containing vitals.
        cols (list): List of vital sign columns to aggregate.
        id_col (str): Name of the encounter ID column.
        clock_col (str): Name of the time column.

    Returns:
        pl.DataFrame: DataFrame containing aggregated features per encounter.
    """
    small_feature_df = df.group_by(id_col, target_col).agg(
        [pl.mean(c).alias(f"{c}_mean") for c in cols]
        + [pl.std(c).alias(f"{c}_std") for c in cols]
        + [pl.min(c).alias(f"{c}_min") for c in cols]
        + [pl.max(c).alias(f"{c}_max") for c in cols]
    )

    large_feature_df = df.group_by(id_col, target_col).agg(
        [pl.mean(c).alias(f"{c}_mean") for c in cols]
        + [pl.std(c).alias(f"{c}_std") for c in cols]
        + [pl.min(c).alias(f"{c}_min") for c in cols]
        + [pl.max(c).alias(f"{c}_max") for c in cols]
        + [pl.median(c).alias(f"{c}_median") for c in cols]
        + [pl.first(c).alias(f"{c}_first") for c in cols]
        + [pl.last(c).alias(f"{c}_last") for c in cols]
        + [(pl.max(c) - pl.min(c)).alias(f"{c}_range") for c in cols]
        + [(pl.std(c) / (pl.mean(c) + 1e-6)).alias(f"{c}_coefvar") for c in cols]
        + [pl.col(c).skew().alias(f"{c}_skew") for c in cols]
        + [pl.col(c).kurtosis().alias(f"{c}_kurtosis") for c in cols]
        + [pl.quantile(c, 0.25).alias(f"{c}_q1") for c in cols]
        + [pl.quantile(c, 0.75).alias(f"{c}_q3") for c in cols]
        + [(pl.quantile(c, 0.75) - pl.quantile(c, 0.25)).alias(f"{c}_iqr") for c in cols]
        + [pl.var(c).alias(f"{c}_var") for c in cols]
    )

    return small_feature_df, large_feature_df


def generate_clock_no_col(
    df: pl.DataFrame, id_col: str = "id", clock_col="clock", clock_no_col="clock_no"
) -> pl.DataFrame:
    df = df.sort([id_col, clock_col])
    if clock_no_col not in df.columns:
        df = df.with_columns(pl.cum_count(id_col).over(id_col).alias(clock_no_col))
    return df


def compute_encounter_missingness(
    df: pl.DataFrame,
    cols: list[str],
    id_col: str = "id",
) -> pl.DataFrame:
    """
    Compute the missingness rate per encounter (id) across specified columns.

    For each encounter, calculates the proportion of missing values across all
    specified columns and all time points.

    Parameters:
        df (pl.DataFrame): Input DataFrame with time series data.
        cols (list[str]): List of columns to check for missingness.
        id_col (str): Name of the encounter ID column.

    Returns:
        pl.DataFrame: DataFrame with columns [id_col, "missingness_rate"] where
                      missingness_rate is between 0.0 (no missing) and 1.0 (all missing).
    """
    return (
        df.group_by(id_col)
        .agg(
            pl.len().alias("row_count"),
            # sum of nulls across all requested columns
            pl.sum_horizontal(*[pl.col(c).is_null().sum() for c in cols]).alias("total_nulls"),
        )
        .with_columns(
            (pl.col("total_nulls") / (pl.col("row_count") * len(cols))).alias("missingness_rate")
        )
        .select(id_col, "missingness_rate")
    )


def stratify_by_missingness(
    feature_df: pl.DataFrame,
    missingness_df: pl.DataFrame,
    id_col: str = "id",
) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """
    Split a feature DataFrame into high and low missingness groups based on the
    global median missingness rate.

    Parameters:
        feature_df (pl.DataFrame): DataFrame with features, must contain id_col.
        missingness_df (pl.DataFrame): DataFrame with [id_col, "missingness_rate"].
        id_col (str): Name of the encounter ID column.

    Returns:
        tuple: (low_missingness_df, high_missingness_df, median_threshold)
               - low_missingness_df: Encounters with missingness <= median
               - high_missingness_df: Encounters with missingness > median
               - median_threshold: The median missingness rate used as threshold
    """
    median_threshold = missingness_df["missingness_rate"].median()

    low_miss_ids = missingness_df.filter(pl.col("missingness_rate") <= median_threshold).select(
        id_col
    )

    high_miss_ids = missingness_df.filter(pl.col("missingness_rate") > median_threshold).select(
        id_col
    )

    low_missingness_df = feature_df.join(low_miss_ids, on=id_col, how="inner")
    high_missingness_df = feature_df.join(high_miss_ids, on=id_col, how="inner")

    return low_missingness_df, high_missingness_df, median_threshold
