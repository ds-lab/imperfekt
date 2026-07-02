# %%
from typing import TypedDict

import polars as pl

from imperfekt import Imperfekt
from config import VITAL_COLS, load_cohort, RESULTS_DIR
pl.Config.set_tbl_cols(8)

class Modus(TypedDict):
    method: str | None
    threshold: float
    missing_as: str
    ranges: bool

df = load_cohort()
print(f"Cohort: {df['id'].n_unique()} stays, {len(df)} observations")

# %%
# Sanity check: actual value ranges in the cohort
print(df.select(VITAL_COLS).describe())

# %%
REFERENCE_RANGES = {
    "hr":    (15.0, 300.0),
    "sbp":   (10.0, 350.0),
    "o2sat": (10.0, 100.0),
    "rr":    (2.0, 80.0),
}

# method=None means ranges-only detection (no statistical method)
MODI: list[Modus] = [
  #  {"method": "iqr",  "threshold": 1.5, "missing_as": "ignore", "ranges": False},
   # {"method": "iqr",  "threshold": 1.5, "missing_as": "ignore", "ranges": True},
   {"method": "mad",  "threshold": 3.5, "missing_as": "ignore", "ranges": False},
 #   {"method": "mad",  "threshold": 3.5, "missing_as": "ignore", "ranges": True},
  #  {"method": None,   "threshold": 1.5, "missing_as": "ignore", "ranges": True},
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
        df=df,
        id_col="id",
        clock_col="clock",
        cols=VITAL_COLS,
        save_path=RESULTS_DIR / "plausibility_experiments",
        renderer=None,
        plot_library="matplotlib",
        plausibility_method=method,
        plausibility_threshold=threshold,
        plausibility_missing_as=missing_as,
        plausibility_reference_ranges=REFERENCE_RANGES if use_ranges else None,
        plausibility_scope="global",
    )
    imp.intravariable.column_statistics(save_results=True)

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

