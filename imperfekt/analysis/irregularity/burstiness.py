import polars as pl


############################################################
#          Burstiness Coefficient Analysis                 #
############################################################


def compute_burstiness_coefficient(
    delta_t_df: pl.DataFrame,
    id_col: str = "id",
) -> pl.DataFrame:
    """
    Compute the burstiness coefficient B per entity.

    B = (std - mean) / (std + mean)  (Goh & Barabasi, 2006)
    Range: [-1, 1]
        B = -1  perfectly regular (all intervals equal)
        B =  0  Poisson process (random)
        B >  0  bursty (clusters of events separated by long gaps)

    Entities with fewer than 3 intervals receive NaN for std and burstiness_coeff.

    Parameters:
        delta_t_df (pl.DataFrame): DataFrame with [id_col, ..., "interval_seconds"],
                                   one row per interval (not-null, > 0).
        id_col (str): Entity identifier column.

    Returns:
        pl.DataFrame: One row per entity with columns:
            id, n_intervals, mean_interval, std_interval, burstiness_coeff.
    """
    entity_stats = (
        delta_t_df
        .group_by(id_col)
        .agg(
            pl.len().alias("n_intervals"),
            pl.col("interval_seconds").mean().alias("mean_interval"),
            pl.col("interval_seconds").std().alias("std_interval"),
        )
        .with_columns(
            pl.when(
                (pl.col("n_intervals") >= 3)
                & (pl.col("mean_interval") + pl.col("std_interval") != 0)
            )
            .then(
                (pl.col("std_interval") - pl.col("mean_interval"))
                / (pl.col("std_interval") + pl.col("mean_interval"))
            )
            .otherwise(None)
            .alias("burstiness_coeff")
        )
        .select([id_col, "n_intervals", "mean_interval", "std_interval", "burstiness_coeff"])
        .sort(id_col)
    )

    # Preserve entities that have 0 intervals (single-observation entities)
    all_entities = delta_t_df.select(pl.col(id_col).unique()).sort(id_col)
    entity_stats = all_entities.join(entity_stats, on=id_col, how="left")

    return entity_stats


def compute_global_burstiness(
    delta_t_df: pl.DataFrame,
    id_col: str = "id",
) -> pl.DataFrame:
    """
    Compute the burstiness coefficient B over all pooled inter-observation intervals,
    restricted to entities that have at least 3 intervals (the same threshold used at
    the entity level). Entities with fewer than 3 intervals are excluded because their
    rhythm cannot be reliably characterised individually, and including their sparse
    intervals would distort the global estimate.

    B = (std - mean) / (std + mean)  (Goh & Barabasi, 2006, https://doi.org/10.48550/arXiv.physics/0610233)
    Range: [-1, 1]
        B = -1  perfectly regular (all intervals equal)
        B =  0  Poisson process (random)
        B >  0  bursty (clusters of events separated by long gaps)

    Parameters:
        delta_t_df (pl.DataFrame): DataFrame with columns [id_col, ..., "interval_seconds"].
        id_col (str): Entity identifier column.

    Returns:
        pl.DataFrame: Single-row DataFrame with columns:
            n_intervals, mean_interval, std_interval, burstiness_coeff.
    """
    eligible_ids = (
        delta_t_df
        .group_by(id_col)
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 3)
        .select(id_col)
    )
    pooled = delta_t_df.join(eligible_ids, on=id_col, how="inner")

    stats = pooled.select(
        pl.len().alias("n_intervals"),
        pl.col("interval_seconds").mean().alias("mean_interval"),
        pl.col("interval_seconds").std().alias("std_interval"),
    )
    row = stats.row(0, named=True)
    n = row["n_intervals"]
    mean_val = row["mean_interval"]
    std_val = row["std_interval"]

    if n >= 3 and (mean_val + std_val) != 0:
        b = (std_val - mean_val) / (std_val + mean_val)
    else:
        b = None

    return pl.DataFrame({
        "n_intervals": [n],
        "mean_interval": [mean_val],
        "std_interval": [std_val],
        "burstiness_coeff": [b],
    })
