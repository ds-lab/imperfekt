from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import polars as pl

from config import (
    STAGE_3_CONFIGS,
    STAGE_4_CONFIGS,
    VITAL_COLS,
    RESULTS_DIR,
    data_fingerprint,
    data_fingerprint_tag,
)
from imperfekt.features.core import FeatureGenerator


STRUCTURAL_PREFIX_MISS = "iv_miss_"
STRUCTURAL_PREFIX_PLAUS = "iv_plaus_"

_FEATURE_CACHE_ROOT = RESULTS_DIR / "feature_sets_cache"
_FEATURE_CACHE_VERSION = 1


def is_structural_feature(col: str) -> bool:
    return col.startswith(STRUCTURAL_PREFIX_MISS) or col.startswith(STRUCTURAL_PREFIX_PLAUS)


def feature_group(col: str) -> str:
    if col.startswith(STRUCTURAL_PREFIX_MISS):
        return "structural_miss"
    if col.startswith(STRUCTURAL_PREFIX_PLAUS):
        return "structural_plaus"
    if any(col == v or col.startswith(f"{v}_") for v in VITAL_COLS):
        return "physiology"
    return "metadata"


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


def _structural_agg_exprs(structural_cols: list[str]) -> list:
    return (
        [pl.col(c).mean().alias(f"{c}_mean") for c in structural_cols]
        + [pl.col(c).median().alias(f"{c}_median") for c in structural_cols]
        + [pl.col(c).min().alias(f"{c}_min") for c in structural_cols]
        + [pl.col(c).max().alias(f"{c}_max") for c in structural_cols]
        + [pl.col(c).std().alias(f"{c}_std") for c in structural_cols]
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
        ts_df.select(["id", "label", "age"])
        .unique("id", keep="first")
       # .with_columns(sex_female=pl.col("sex").eq("F").cast(pl.Int8))
       # .drop("sex")
    )
    return features.join(stay_meta, on="id", how="left").drop_nulls("label")


def _run_fg(ts_df: pl.DataFrame, imperfection: str, prefix: str) -> pl.DataFrame:
    """
    Run FeatureGenerator on ts_df with VITAL_COLS as variable_cols, applying
    only the three feature families we want (binary masks, temporal features,
    row imperfection pct). Returns ts_df enriched with new columns, each
    renamed with the given prefix so downstream code can identify structural
    features purely by name.
    """
    original_cols = set(ts_df.columns)
    fg = FeatureGenerator(
        ts_df,
        id_col="id",
        clock_col="clock",
        variable_cols=VITAL_COLS,
        imperfection=imperfection,
    )
    fg.add_binary_masks().add_temporal_features().add_row_imperfection_pct()
    enriched = fg.df

    rename_map = {
        c: f"{prefix}{c}"
        for c in enriched.columns
        if c not in original_cols and c != "clock_no"
    }
    return enriched.rename(rename_map)


def _stay_level_aggregate(
    enriched_ts_df: pl.DataFrame,
    case_metrics: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Group enriched per-observation frame by id and produce one row per stay.
    Vital cols get mean/median/min/max/std/first/last; structural cols (prefix
    match) get mean/median/min/max/std. The outcome label is taken as the
    per-id max (any positive observation -> positive stay), and case_metrics
    (id-level) is joined if provided, with its numeric columns prefixed to
    mark them structural-miss.
    """
    structural_cols = [c for c in enriched_ts_df.columns if is_structural_feature(c)]
    agg_exprs = _vital_agg_exprs() + _structural_agg_exprs(structural_cols)

    stay_df = (
        enriched_ts_df.sort(["id", "clock"])
        .group_by("id")
        .agg(agg_exprs)
    )

    label = (
        enriched_ts_df.select(["id", "label", "age"])
        .group_by("id")
        .agg(
            pl.col("label").max().alias("label"),
            pl.col("age").first().alias("age"),
        )
    )
    stay_df = stay_df.join(label, on="id", how="left")

    if case_metrics is not None:
        numeric_cols = [
            c for c in case_metrics.columns
            if c != "id" and case_metrics.schema[c].is_numeric()
        ]
        if numeric_cols:
            case_subset = case_metrics.select(["id"] + numeric_cols).rename(
                {c: f"{STRUCTURAL_PREFIX_MISS}{c}" for c in numeric_cols}
            )
            stay_df = stay_df.join(case_subset, on="id", how="left")

    return stay_df.drop_nulls("label")


def _feature_set_cache_paths(cohort_path: Path, config_name: str, setup_name: str) -> tuple[Path, Path]:
    tag = data_fingerprint_tag(cohort_path)
    base_dir = _FEATURE_CACHE_ROOT / f"{cohort_path.stem}_{tag}" / config_name
    return base_dir / f"{setup_name}.parquet", base_dir / f"{setup_name}.meta.json"


def _feature_set_cache_key(
    cohort_path: Path,
    config_name: str,
    setup_name: str,
    flags: dict,
) -> dict:
    return {
        "version": _FEATURE_CACHE_VERSION,
        "data": data_fingerprint(cohort_path),
        "config_name": config_name,
        "config_recipe": STAGE_3_CONFIGS.get(config_name),
        "setup_name": setup_name,
        "flags": flags,
        "vital_cols": list(VITAL_COLS),
        "structural_prefix_miss": STRUCTURAL_PREFIX_MISS,
        "structural_prefix_plaus": STRUCTURAL_PREFIX_PLAUS,
    }


def make_feature_sets(
    config_provider: Callable[[], pl.DataFrame],
    *,
    config_name: str,
    cohort_path: Path | None = None,
    case_metrics: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Build one stay-level feature frame per STAGE_4_CONFIGS entry.

    ``config_provider`` is a no-arg callable returning the per-observation config
    DataFrame. It is invoked lazily — only the first time a setup is *not* served
    from cache — so a fully cached run never builds the config (or its masks) at
    all. Its result is memoized here, so multiple uncached setups sharing the
    config build it once.

    For each setup, optional FeatureGenerator passes enrich the per-observation
    frame with missingness- and/or plausibility-derived columns (prefixed
    iv_miss_ / iv_plaus_), then the result is aggregated to stay level. When
    miss is enabled, the id-level case_metrics numeric columns are also joined
    in (also prefixed iv_miss_).

    If cohort_path is given, each setup is cached per
    (cohort, data fingerprint, config_name, setup_name) at
    RESULTS_DIR/feature_sets_cache/<cohort_stem>_<data_tag>/<config_name>/
    <setup_name>.parquet. Each distinct data setting (file or DEBUG sampling)
    gets its own slot, so switching back and forth reuses prior slices instead
    of overwriting them. Adding a new setup later does not invalidate existing
    ones.
    """
    setups: dict[str, pl.DataFrame] = {}
    _config_df: list[pl.DataFrame] = []  # one-slot memo for the lazy config build

    def get_config() -> pl.DataFrame:
        if not _config_df:
            _config_df.append(config_provider())
        return _config_df[0]

    for setup_name, flags in STAGE_4_CONFIGS.items():
        cache_dir_paths = (
            _feature_set_cache_paths(cohort_path, config_name, setup_name)
            if cohort_path is not None
            else (None, None)
        )
        parquet_path, meta_path = cache_dir_paths

        if (
            cohort_path is not None
            and parquet_path.exists()
            and meta_path.exists()
        ):
            expected_key = _feature_set_cache_key(cohort_path, config_name, setup_name, flags)
            cached_meta = json.loads(meta_path.read_text())
            if cached_meta.get("key") == expected_key:
                setups[setup_name] = pl.read_parquet(parquet_path)
                print(f"Loaded feature set [{config_name}/{setup_name}] from cache")
                continue

        enriched = get_config()
        if flags.get("miss"):
            enriched = _run_fg(enriched, imperfection="missingness", prefix=STRUCTURAL_PREFIX_MISS)
        if flags.get("plaus"):
            enriched = _run_fg(enriched, imperfection="plausibility", prefix=STRUCTURAL_PREFIX_PLAUS)

        stay_df = _stay_level_aggregate(
            enriched,
            case_metrics=case_metrics if flags.get("miss") else None,
        )
        setups[setup_name] = stay_df

        if cohort_path is not None:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            stay_df.write_parquet(parquet_path)
            meta_path.write_text(
                json.dumps(
                    {"key": _feature_set_cache_key(cohort_path, config_name, setup_name, flags)},
                    indent=2,
                )
            )
            print(f"Cached feature set [{config_name}/{setup_name}] to {parquet_path}")

    return setups
