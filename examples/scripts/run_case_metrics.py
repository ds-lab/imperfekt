# %%
"""
Composite Score: Irregularity, Intravariable & Intervariable Imperfection Stratification
=========================================================================================
Assigns each case to one of five imperfection strata (Q_complete / Q_alpha / Q_beta /
Q_gamma / Q_delta) using the least-correlated pair of metrics as axes,
median-bisected per variable / cohort.

Three analyses are demonstrated:
  1. Irregularity case metrics   — temporal observation spacing
  2. Intravariable case metrics  — per-variable missingness pattern (one stratum per case × variable)
  3. Intervariable case metrics  — cross-variable co-missingness structure (one stratum per case)

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

SAVE_RESULTS_PATH = Path("results/case_metrics")
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

irr = irr.case_metrics(stratify=True, save_results=SAVE_RESULTS)

# %%
print("\n=== Irregularity: case scores (selected columns) ===")
print(
    irr.results.cm_case_metrics.select(
        [
            "id",
            "interval_cv",
            "interval_qcod",
            "interval_adh_rate",
            "axis_x",
            "axis_y",
            "axis_pair_corr",
            "irregularity_stratum",
        ]
    )
)

# %%
print("\n=== Irregularity: pairwise axis correlations ===")
print(irr.results.cm_pairwise_correlations)

# %%
print("\n=== Irregularity: stratum prevalence ===")
scores = irr.results.cm_case_metrics
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
# x_median_train = float(train[irr.results.cm_case_metrics["axis_x"][0]].median())
# y_median_train = float(train[irr.results.cm_case_metrics["axis_y"][0]].median())
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
# case_metrics() calls them automatically if not yet run.
iv = iv.case_metrics(stratify=True, save_results=SAVE_RESULTS)

# %%
print("\n=== Intravariable: case scores per variable ===")
print(
    iv.results.cm_case_metrics.select(
        [
            "id",
            "variable",
            "indicated_pct",
            "indicated_centroid",
            "gap_adh_rate",
            "gap_entropy",
            "axis_x",
            "axis_y",
            "axis_pair_corr",
            "imperfection_stratum",
        ]
    )
)

# %%
print("\n=== Intravariable: stratum prevalence per variable ===")
intra_metrics = iv.results.cm_case_metrics
for var in cols:
    var_scores = intra_metrics.filter(pl.col("variable") == var)
    total_var = var_scores.height
    print(
        f"\n  {var}  (axis: "
        f"{var_scores['axis_x'][0]} × {var_scores['axis_y'][0]}, "
        f"corr={var_scores['axis_pair_corr'][0]:.3f})"
    )
    print(
        var_scores.filter(pl.col("imperfection_stratum").is_not_null())
        .group_by("imperfection_stratum")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / total_var * 100).round(1).alias("pct"))
        .sort("imperfection_stratum")
    )

# %%
print("\n=== Intravariable: pairwise axis correlations (heartrate) ===")
print(iv.results.cm_pairwise_correlations[cols[0]])

# %%
# Cross-validation usage: fit medians on train, apply to held-out test
# -------------------------------------------------------------------
# For intravariable, do this per variable:
#
# for var in cols:
#     var_train = train_scores.filter(pl.col("variable") == var)
#     x_med = float(var_train[axis_x].median())
#     y_med = float(var_train[axis_y].median())
#     test_var = IntravariableImperfection.assign_strata(
#         test_scores.filter(pl.col("variable") == var),
#         axis_x, axis_y, x_med, y_med,
#     )

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
ivv = ivv.case_metrics(stratify=True, save_results=SAVE_RESULTS)

# %%
print("\n=== Intervariable: case scores ===")
print(
    ivv.results.cm_case_metrics.select(
        [
            "id",
            "avg_indicated_pct",
            "co_concentration",
            "breadth",
            "pattern_entropy",
            "max_pair_overlap",
            "axis_x",
            "axis_y",
            "axis_pair_corr",
            "intervariable_stratum",
        ]
    )
)

# %%
print("\n=== Intervariable: stratum prevalence ===")
iv_scores = ivv.results.cm_case_metrics
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
print(ivv.results.cm_pairwise_correlations)

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
