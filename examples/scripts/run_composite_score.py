# %%
"""
Composite Score: Irregularity, Intravariable & Intervariable Imperfection Stratification
=========================================================================================
Assigns each case to one of five imperfection strata (Q_complete / Q_alpha / Q_beta /
Q_gamma / Q_delta) using the least-correlated pair of metrics as axes,
median-bisected per cohort.

Three analyses are demonstrated:
  1. Irregularity composite score   — temporal observation spacing
  2. Intravariable composite score  — cross-variable missingness burden (one stratum per case)
  3. Intervariable composite score  — cross-variable co-missingness structure (one stratum per case)

Replace the synthetic dataset block with your own data load and adjust
SAVE_RESULTS_PATH / RENDERER as needed.
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
from imperfekt.analysis.intravariable.intravariable import IntravariableImperfection
from imperfekt.analysis.irregularity.irregularity import Irregularity

pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(30)

SAVE_RESULTS_PATH = Path("results/composite_score")
RENDERER = None  # set to "notebook_connected" or "browser" for plots
SAVE_RESULTS = True

# %%
##############################
#       SYNTHETIC DATA       #
##############################
# 20 cases, 40 observations each. Missingness and observation spacing are
# deliberately varied so that all four quadrants are populated.
#
# Replace this block with:
#   vitals_df = pl.read_parquet(data_path / "vitals.parquet")
#   cols = VITALS.PARAMS

rng_state = 42
import random

random.seed(rng_state)

N_CASES = 20
N_OBS = 40
base_time = datetime(2023, 1, 1, 0, 0, 0)

rows = []
for case_id in range(N_CASES):
    # Vary inter-observation spacing: regular, bursty, or random
    pattern = case_id % 4
    t = base_time + timedelta(hours=case_id * 2)
    for i in range(N_OBS):
        if pattern == 0:  # regular ~60s spacing
            gap = 60
        elif pattern == 1:  # bursty: alternating short/long
            gap = 10 if i % 5 != 0 else 300
        elif pattern == 2:  # random uniform
            gap = random.randint(30, 180)
        else:  # sparse, wide gaps
            gap = random.randint(120, 600)
        t += timedelta(seconds=gap)

        # missingness: cases 0-9 low, 10-19 high; some bursty
        missing_prob = 0.05 if case_id < 10 else 0.30
        if case_id >= 15 and 10 <= i <= 15:
            missing_prob = 0.90  # concentrated gap burst

        hr = None if random.random() < missing_prob else float(70 + random.randint(-15, 15))
        bp = None if random.random() < missing_prob * 0.7 else float(120 + random.randint(-20, 20))
        rows.append((str(case_id), t, hr, bp, i))

vitals_df = pl.DataFrame(
    rows,
    schema=["id", "clock", "heartrate", "blood_pressure", "clock_no"],
    orient="row",
)
cols = ["heartrate", "blood_pressure"]

print(f"Dataset: {vitals_df['id'].n_unique()} cases, {vitals_df.height} rows")
print(vitals_df.head(5))

# %%
##############################
# 1. IRREGULARITY COMPOSITE  #
##############################
irr = Irregularity(
    df=vitals_df,
    id_col="id",
    clock_col="clock",
    save_path=SAVE_RESULTS_PATH / "irregularity" if SAVE_RESULTS else None,
    renderer=RENDERER,
)

irr = irr.composite_score(save_results=SAVE_RESULTS)

# %%
print("\n=== Irregularity: case scores (selected columns) ===")
print(
    irr.results.cs_case_scores.select(
        [
            "id",
            "cv",
            "burstiness_coeff",
            "adherence_rate",
            "axis_x",
            "axis_y",
            "axis_pair_corr",
            "irregularity_stratum",
        ]
    )
)

# %%
print("\n=== Irregularity: pairwise axis correlations ===")
print(irr.results.cs_pairwise_correlations)

# %%
print("\n=== Irregularity: stratum prevalence ===")
scores = irr.results.cs_case_scores
total = scores.height
print(
    scores.filter(pl.col("irregularity_stratum").is_not_null())
    .group_by("irregularity_stratum")
    .agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / total * 100).round(1).alias("pct"))
    .sort("irregularity_stratum")
)

# %%
# Cross-validation usage: fit medians on train, apply to held-out test
# -------------------------------------------------------------------
# split = int(0.8 * scores.height)
# train = scores[:split]
# test  = scores[split:]
#
# x_median_train = float(train[irr.results.cs_case_scores["axis_x"][0]].median())
# y_median_train = float(train[irr.results.cs_case_scores["axis_y"][0]].median())
#
# test_with_strata = Irregularity.assign_strata(
#     test, axis_x, axis_y, x_median_train, y_median_train
# )

# %%
##################################
# 2. INTRAVARIABLE COMPOSITE     #
##################################
iv = IntravariableImperfection(
    df=vitals_df,
    id_col="id",
    clock_col="clock",
    clock_no_col="clock_no",
    cols=cols,
    save_path=SAVE_RESULTS_PATH / "intravariable" if SAVE_RESULTS else None,
    renderer=RENDERER,
)

# column_statistics and gap_statistics are required prerequisites;
# composite_score() calls them automatically if not yet run.
iv = iv.composite_score(save_results=SAVE_RESULTS)

# %%
print("\n=== Intravariable: case scores ===")
composite = iv.results.iv_composite_scores
indicated_cols = [c for c in composite.columns if c.endswith("_indicated_pct")]
print(
    composite.select(
        ["id"] + indicated_cols + ["axis_x", "axis_y", "axis_pair_corr", "imperfection_stratum"]
    )
)

# %%
print("\n=== Intravariable: stratum prevalence ===")
total = composite.height
axis_x = composite["axis_x"][0]
axis_y = composite["axis_y"][0]
corr = composite["axis_pair_corr"][0]
print(f"  axis: {axis_x} × {axis_y}  (corr={corr:.3f})")
print(
    composite.filter(pl.col("imperfection_stratum").is_not_null())
    .group_by("imperfection_stratum")
    .agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / total * 100).round(1).alias("pct"))
    .sort("imperfection_stratum")
)

# %%
print("\n=== Intravariable: pooled axis-pair correlations ===")
print(iv.results.iv_pooled_corr_table)

# %%
# Cross-validation usage: fit medians on train, apply to held-out test
# -------------------------------------------------------------------
# For intravariable (axis pair is shared across all variables):
#
# axis_x = train_scores["axis_x"][0]
# axis_y = train_scores["axis_y"][0]
# x_med = float(train_scores[axis_x].median())
# y_med = float(train_scores[axis_y].median())
# test_stratified = IntravariableImperfection.assign_strata(
#     test_scores, axis_x, axis_y, x_med, y_med, cols,
# )

# %%
####################################
# 3. INTERVARIABLE COMPOSITE       #
####################################
ivv = IntervariableImperfection(
    df=vitals_df,
    id_col="id",
    clock_col="clock",
    clock_no_col="clock_no",
    cols=cols,
    save_path=SAVE_RESULTS_PATH / "intervariable" if SAVE_RESULTS else None,
    renderer=RENDERER,
)

# row_statistics() is called automatically if not yet run.
ivv = ivv.composite_score(save_results=SAVE_RESULTS)

# %%
print("\n=== Intervariable: case scores ===")
print(
    ivv.results.iv_composite_scores.select(
        [
            "id",
            "avg_indicated_vars_pct",
            "co_missingness_concentration",
            "missing_variable_breadth",
            "pattern_entropy",
            "max_pairwise_co_missingness",
            "axis_x",
            "axis_y",
            "axis_pair_corr",
            "intervariable_stratum",
        ]
    )
)

# %%
print("\n=== Intervariable: stratum prevalence ===")
iv_scores = ivv.results.iv_composite_scores
total_iv = iv_scores.height
print(
    iv_scores.filter(pl.col("intervariable_stratum").is_not_null())
    .group_by("intervariable_stratum")
    .agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / total_iv * 100).round(1).alias("pct"))
    .sort("intervariable_stratum")
)

# %%
print("\n=== Intervariable: pairwise axis correlations ===")
print(ivv.results.iv_pairwise_correlations)

# %%
# Cross-validation usage: fit medians on train, apply to held-out test
# -------------------------------------------------------------------
# axis_x = iv_scores["axis_x"][0]
# axis_y = iv_scores["axis_y"][0]
# split  = int(0.8 * iv_scores.height)
# train  = iv_scores[:split]
# test   = iv_scores[split:]
#
# x_med = float(train[axis_x].median())
# y_med = float(train[axis_y].median())
# test_with_strata = IntervariableImperfection.assign_strata(
#     test, axis_x, axis_y, x_med, y_med,
# )
