import logging

import polars as pl

from imperfekt.analysis.utils.masking import create_plausibility_mask
from config import (
    COHORT_MAX_READINGS,
    COHORT_MIN_READINGS,
    COHORT_WINDOW_MINUTES,
    STAGE_3_CONFIGS,
    VITAL_COLS,
    saits_model_path,
)
from imputation import impute

logger = logging.getLogger(__name__)


def filter_cohort(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows within the configured window and minimum reading count."""
    df_win = (
        df.with_columns(pl.col("clock").min().over("id").alias("_start_clock"))
        .with_columns(
            ((pl.col("clock") - pl.col("_start_clock")).dt.total_minutes())
            .alias("_minutes_from_start")
        )
        .filter(pl.col("_minutes_from_start") <= COHORT_WINDOW_MINUTES)
        .drop(["_start_clock", "_minutes_from_start"])
    )
    count_filter = pl.col("_n") >= COHORT_MIN_READINGS
    if COHORT_MAX_READINGS is not None:
        count_filter = count_filter & (pl.col("_n") <= COHORT_MAX_READINGS)
    valid_keys = (
        df_win.group_by("id")
        .agg(pl.col("clock").len().alias("_n"))
        .filter(count_filter)
        .select("id")
    )
    return df_win.join(valid_keys, on="id", how="inner")


def make_plausibility_mask(df: pl.DataFrame, method: str) -> pl.DataFrame:
    """Compute a frozen plausibility mask (IQR or MAD, pure statistical, no reference ranges).

    Returns a DataFrame with columns id, clock, sbp, hr, o2sat, rr (Int8: 1=implausible).
    """
    if method not in ("iqr", "mad"):
        raise ValueError(f"method must be 'iqr' or 'mad', got {method!r}")
    threshold = 1.5 if method == "iqr" else 3.5
    return create_plausibility_mask(
        df,
        id_col="id",
        clock_col="clock",
        cols=VITAL_COLS,
        method=method,
        threshold=threshold,
        scope="global",
        reference_ranges=None,
    )


class ConfigBuilder:
    """Builds plausibility masks and config DataFrames lazily, on demand.

    Masks (iqr/mad) and built configs are memoized, so building several configs
    that share a mask only computes that mask once, and a single config feeding
    many feature sets is built only once.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df
        self._masks: dict[str, pl.DataFrame] = {}
        self._configs: dict[str, pl.DataFrame] = {}

    def mask(self, method: str) -> pl.DataFrame:
        if method not in self._masks:
            self._masks[method] = make_plausibility_mask(self.df, method=method)
        return self._masks[method]

    def config(self, config_id: str) -> pl.DataFrame | None:
        """Build (and memoize) a config DataFrame, or None if it should be skipped.
        """
        if config_id not in self._configs:
            cfg = STAGE_3_CONFIGS[config_id]
            df_out = _apply_plausibility(self.df, self.mask(cfg["method"]), cfg["plaus"])
            df_out = _apply_imputation(df_out, cfg["imp"], cfg["plaus"])
            self._configs[config_id] = df_out
        return self._configs[config_id]


def _apply_plausibility(df: pl.DataFrame, mask: pl.DataFrame, plaus: str) -> pl.DataFrame:
    if plaus == "keep":
        return df
    # plaus == "remove": null out vitals where mask flag == 1
    mask_renamed = mask.rename({v: f"_m_{v}" for v in VITAL_COLS})
    return (
        df.join(mask_renamed, on=["id", "clock"], how="left")
        .with_columns([
            pl.when(pl.col(f"_m_{v}") == 1).then(None).otherwise(pl.col(v)).alias(v)
            for v in VITAL_COLS
        ])
        .drop([f"_m_{v}" for v in VITAL_COLS])
    )


def _apply_imputation(df: pl.DataFrame, imp: str, plaus: str) -> pl.DataFrame | None:
    """Apply the imputation strategy for a config.

    Returns the imputed DataFrame, or ``None`` to signal the config should be
    *skipped* (currently only when ``imp == "saits"`` but the pre-trained model
    for this plausibility branch is missing). ``plaus`` ("keep"/"remove") selects
    the per-branch SAITS model.
    """
    if imp == "none":
        return df
    if imp == "locf":
        # forward-fill within patient, fall back to per-patient mean for leading nulls
        df_out, _ = impute(df, cols=VITAL_COLS, strategy="locf", id_col="id", time_col="clock")
        return df_out
    if imp == "saits":
        model_path = saits_model_path(plaus)
        if not model_path.exists():
            logger.warning(
                "Skipping SAITS config (plaus=%s): no trained model at %s. "
                "Run train_saits.py to enable it.",
                plaus,
                model_path,
            )
            return None
        df_out, _ = impute(
            df,
            cols=VITAL_COLS,
            strategy="saits",
            id_col="id",
            time_col="clock",
            saits_model_path=str(model_path),
        )
        return df_out
    raise ValueError(f"Unknown imputation strategy '{imp}' (expected none|locf|saits).")
