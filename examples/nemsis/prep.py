import sys
from pathlib import Path

import polars as pl
from config import COHORT_MIN_READINGS, COHORT_WINDOW_MINUTES

from imperfekt.analysis.utils.masking import create_plausibility_mask

_SEP_DIR = Path(__file__).resolve().parent.parent / "sepsis_prediction"
if str(_SEP_DIR) not in sys.path:
    sys.path.insert(0, str(_SEP_DIR))
from imputation import impute  # noqa: E402

VITAL_COLS = ["sbp", "hr", "o2sat", "rr"]

CONFIGS: dict[str, dict[str, str]] = {
    "iq_pk_in": {"method": "iqr", "plaus": "keep", "imp": "none"},
    "iq_pk_il": {"method": "iqr", "plaus": "keep", "imp": "locf"},
    "iq_pr_in": {"method": "iqr", "plaus": "remove", "imp": "none"},
    "iq_pr_il": {"method": "iqr", "plaus": "remove", "imp": "locf"},
    "ma_pk_in": {"method": "mad", "plaus": "keep", "imp": "none"},
    "ma_pk_il": {"method": "mad", "plaus": "keep", "imp": "locf"},
    "ma_pr_in": {"method": "mad", "plaus": "remove", "imp": "none"},
    "ma_pr_il": {"method": "mad", "plaus": "remove", "imp": "locf"},
}


def filter_cohort(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows within the configured window and minimum reading count."""
    df_win = (
        df.with_columns(pl.col("clock").min().over("PcrKey").alias("_start_clock"))
        .with_columns(
            ((pl.col("clock") - pl.col("_start_clock")).dt.total_minutes()).alias(
                "_minutes_from_start"
            )
        )
        .filter(pl.col("_minutes_from_start") <= COHORT_WINDOW_MINUTES)
        .drop(["_start_clock", "_minutes_from_start"])
    )
    valid_keys = (
        df_win.group_by("PcrKey")
        .agg(pl.col("clock").len().alias("_n"))
        .filter(pl.col("_n") >= COHORT_MIN_READINGS)
        .select("PcrKey")
    )
    return df_win.join(valid_keys, on="PcrKey", how="inner")


def make_plausibility_mask(df: pl.DataFrame, method: str) -> pl.DataFrame:
    """Compute a frozen plausibility mask (IQR or MAD, pure statistical, no reference ranges).

    Returns a DataFrame with columns PcrKey, clock, sbp, hr, o2sat, rr (Int8: 1=implausible).
    """
    if method not in ("iqr", "mad"):
        raise ValueError(f"method must be 'iqr' or 'mad', got {method!r}")
    threshold = 1.5 if method == "iqr" else 3.5
    return create_plausibility_mask(
        df,
        id_col="PcrKey",
        clock_col="clock",
        cols=VITAL_COLS,
        method=method,
        threshold=threshold,
        scope="global",
        reference_ranges=None,
    )


def make_configs(
    df: pl.DataFrame,
    mask_iqr: pl.DataFrame,
    mask_mad: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Produce all 8 config DataFrames from a filtered cohort and frozen masks.

    Returns a dict mapping config_id → DataFrame (same schema and row count as df).
    """
    masks = {"iqr": mask_iqr, "mad": mask_mad}
    results: dict[str, pl.DataFrame] = {}

    for config_id, cfg in CONFIGS.items():
        mask = masks[cfg["method"]]
        df_out = _apply_plausibility(df, mask, cfg["plaus"])
        df_out = _apply_imputation(df_out, cfg["imp"])
        results[config_id] = df_out

    return results


def _apply_plausibility(df: pl.DataFrame, mask: pl.DataFrame, plaus: str) -> pl.DataFrame:
    if plaus == "keep":
        return df
    # plaus == "remove": null out vitals where mask flag == 1
    mask_renamed = mask.rename({v: f"_m_{v}" for v in VITAL_COLS})
    return (
        df.join(mask_renamed, on=["PcrKey", "clock"], how="left")
        .with_columns(
            [
                pl.when(pl.col(f"_m_{v}") == 1).then(None).otherwise(pl.col(v)).alias(v)
                for v in VITAL_COLS
            ]
        )
        .drop([f"_m_{v}" for v in VITAL_COLS])
    )


def _apply_imputation(df: pl.DataFrame, imp: str) -> pl.DataFrame:
    if imp == "none":
        return df
    # imp == "locf": forward-fill within patient, fall back to per-patient mean for leading nulls
    df_out, _ = impute(df, cols=VITAL_COLS, strategy="locf", id_col="PcrKey", time_col="clock")
    return df_out
