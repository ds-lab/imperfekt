# %%
from pathlib import Path
from typing import TypedDict

import polars as pl

from imperfekt import Imperfekt
from config import COHORT_PATH

pl.Config.set_tbl_cols(8)

class Modus(TypedDict):
    method: str | None
    threshold: float
    missing_as: str
    ranges: bool

df = pl.read_parquet(Path(COHORT_PATH))

# %%
df_filtered = df.with_columns(
    pl.col("clock").min().over("PcrKey").alias("_start_clock")
).with_columns(
    ((pl.col("clock") - pl.col("_start_clock")).dt.total_minutes()).alias("_minutes_from_start")
).filter(pl.col("_minutes_from_start") <= 30).drop(["_start_clock", "_minutes_from_start"])

valid_keys = (
    df_filtered.group_by("PcrKey")
    .agg(pl.col("clock").count().alias("_num_vitals"))
    .filter(pl.col("_num_vitals") >= 5)
    .select("PcrKey")
)

df_filtered = df_filtered.join(valid_keys, on="PcrKey", how="inner")

# %%
# Sanity check: actual value ranges in the filtered cohort
print(df_filtered.select(["sbp", "hr", "o2sat", "rr"]).describe())

# %%
REFERENCE_RANGES = {
    "hr":    (15.0, 300.0),
    "sbp":   (10.0, 350.0),
    "o2sat": (10.0, 100.0),
    "rr":    (2.0, 80.0),
}

# method=None means ranges-only detection (no statistical method)
MODI: list[Modus] = [
    {"method": "iqr",  "threshold": 1.5, "missing_as": "ignore", "ranges": False},
    {"method": "iqr",  "threshold": 1.5, "missing_as": "ignore", "ranges": True},
    {"method": "mad",  "threshold": 3.5, "missing_as": "ignore", "ranges": False},
    {"method": "mad",  "threshold": 3.5, "missing_as": "ignore", "ranges": True},
    {"method": None,   "threshold": 1.5, "missing_as": "ignore", "ranges": True},
]

# %%
results = []

for modus in MODI:
    method = modus["method"]
    threshold = modus["threshold"]
    missing_as = modus["missing_as"]
    use_ranges = modus["ranges"]

    method_str = method or "ranges_only"
    name = f"{method_str}_missing-{missing_as}_ranges-{'yes' if use_ranges else 'no'}"

    imp = Imperfekt(
        imperfection="plausibility",
        df=df_filtered,
        id_col="PcrKey",
        clock_col="clock",
        cols=["sbp", "hr", "o2sat", "rr"],
        save_path=None,
        renderer=None,
        plot_library="matplotlib",
        plausibility_method=method,
        plausibility_threshold=threshold,
        plausibility_missing_as=missing_as,
        plausibility_reference_ranges=REFERENCE_RANGES if use_ranges else None,
        plausibility_scope="global",
    )
    imp.intravariable.column_statistics(save_results=False)

    stats = imp.intravariable.results.cs_overall_statistics
    if stats is not None:
        results.append(
            stats.select(["column", "indicated_pct"])
            .rename({"indicated_pct": name})
        )

# %%
comparison = results[0]
for tbl in results[1:]:
    comparison = comparison.join(tbl, on="column", how="left")

print(comparison)

