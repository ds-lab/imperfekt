from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from examples.mimic_iv_ed.config import IREG_FEATURE_COLS, OUTCOME_COL, VITAL_COLS  # noqa: E402
from imperfekt.features.irregularity import (  # noqa: E402
    add_interval_features,
    add_windowed_acceleration,
)


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


def _ireg_agg_exprs() -> list:
    return (
        [pl.col(ic).mean().alias(f"{ic}_mean") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).median().alias(f"{ic}_median") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).min().alias(f"{ic}_min") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).max().alias(f"{ic}_max") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).std().alias(f"{ic}_std") for ic in IREG_FEATURE_COLS]
    )


def pipeline_0_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Raw irregular timestamps, statistical aggregates of vital signs only.
    No resampling, no imputation, no imperfekt features.
    Baseline that isolates the contribution of temporal structure.
    """
    return df.sort(["stay_id", "charttime"]).group_by("stay_id").agg(_vital_agg_exprs())


def pipeline_d_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pipeline 0 plus observation-count feature: raw irregular timestamps,
    vital-sign statistical aggregates, and number of timestamped
    observations (rows) per stay.
    """
    return (
        df.sort(["stay_id", "charttime"])
        .group_by("stay_id")
        .agg(_vital_agg_exprs() + [pl.col("charttime").count().alias("n_observations")])
    )


def pipeline_a_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Resample to a regular 30-min grid per stay (forward-fill then backward-fill
    within stay only), then compute per-stay statistical aggregates.
    """
    filled = (
        df.sort(["stay_id", "charttime"])
        .upsample(time_column="charttime", every="30m", group_by="stay_id", maintain_order=True)
        .with_columns([pl.col(c).forward_fill().over("stay_id") for c in VITAL_COLS])
        .with_columns([pl.col(c).backward_fill().over("stay_id") for c in VITAL_COLS])
    )
    return filled.group_by("stay_id").agg(_vital_agg_exprs())


def pipeline_b_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep raw irregular timestamps.  Add imperfekt interval and acceleration
    features (global per stay — all charttimes treated as a single event
    sequence), then compute per-stay aggregates for both vital signs and
    irregularity features.
    """
    df_ireg = (
        df.sort(["stay_id", "charttime"])
        .pipe(add_interval_features, id_col="stay_id", clock_col="charttime")
        .pipe(add_windowed_acceleration, window_size=5, id_col="stay_id", clock_col="charttime")
    )
    return df_ireg.group_by("stay_id").agg(_vital_agg_exprs() + _ireg_agg_exprs())


def pipeline_c_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute imperfekt features on raw irregular timestamps first, then
    resample to a regular 30-min grid and fill missing values within stay.
    Finally, compute per-stay aggregates for vital and irregularity features.
    """
    df_ireg = (
        df.sort(["stay_id", "charttime"])
        .pipe(add_interval_features, id_col="stay_id", clock_col="charttime")
        .pipe(add_windowed_acceleration, window_size=5, id_col="stay_id", clock_col="charttime")
    )
    fill_cols = VITAL_COLS + IREG_FEATURE_COLS
    filled = (
        df_ireg.upsample(
            time_column="charttime", every="30m", group_by="stay_id", maintain_order=True
        )
        .with_columns([pl.col(c).forward_fill().over("stay_id") for c in fill_cols])
        .with_columns([pl.col(c).backward_fill().over("stay_id") for c in fill_cols])
    )
    return filled.group_by("stay_id").agg(_vital_agg_exprs() + _ireg_agg_exprs())


def build_stay_level(ts_df: pl.DataFrame, feature_fn) -> pl.DataFrame:
    """Build stay-level feature frame with outcome label attached."""
    features = feature_fn(ts_df)
    stay_meta = (
        ts_df.select(["stay_id", "subject_id", OUTCOME_COL, "age_at_visit", "sex"])
        .unique("stay_id", keep="first")
        .with_columns(sex_female=pl.col("sex").eq("F").cast(pl.Int8))
        .drop("sex")
    )
    return features.join(stay_meta, on="stay_id", how="left").drop_nulls(OUTCOME_COL)
