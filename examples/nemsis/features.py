from __future__ import annotations

from examples.nemsis.config import STRUCTURAL_FEATURE_COLS, VITAL_COLS
import polars as pl


def _vital_agg_exprs() -> list:
    return (
        [pl.col(c).mean().alias(f"{c}_mean") for c in VITAL_COLS]
        + [pl.col(c).median().alias(f"{c}_median") for c in VITAL_COLS]
        + [pl.col(c).min().alias(f"{c}_min") for c in VITAL_COLS]
        + [pl.col(c).max().alias(f"{c}_max") for c in VITAL_COLS]
        + [pl.col(c).std().alias(f"{c}_std") for c in VITAL_COLS]
        + [pl.col(c).first().alias(f"{c}_first") for c in VITAL_COLS]
        + [pl.col(c).last().alias(f"{c}_last") for c in VITAL_COLS]
    )


def _imperfekt_agg_exprs() -> list:
    return (
        [pl.col(ic).mean().alias(f"{ic}_mean") for ic in STRUCTURAL_FEATURE_COLS]
        + [pl.col(ic).median().alias(f"{ic}_median") for ic in STRUCTURAL_FEATURE_COLS]
        + [pl.col(ic).min().alias(f"{ic}_min") for ic in STRUCTURAL_FEATURE_COLS]
        + [pl.col(ic).max().alias(f"{ic}_max") for ic in STRUCTURAL_FEATURE_COLS]
        + [pl.col(ic).std().alias(f"{ic}_std") for ic in STRUCTURAL_FEATURE_COLS]
    )


def pipeline_0_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Raw irregular timestamps, statistical aggregates of vital signs only.
    No resampling, no imputation, no imperfekt features.
    Baseline that isolates the contribution of temporal structure.
    """
    return df.sort(["id", "clock"]).group_by("id").agg(_vital_agg_exprs())


def build_feature_set(ts_df: pl.DataFrame, feature_fn) -> pl.DataFrame:
    """Build stay-level feature frame with outcome label attached."""
    features = feature_fn(ts_df)
    stay_meta = (
        ts_df.select(["id", "label"])
        .unique("id", keep="first")
       # .with_columns(sex_female=pl.col("sex").eq("F").cast(pl.Int8))
       # .drop("sex")
    )
    return features.join(stay_meta, on="id", how="left").drop_nulls("label")
