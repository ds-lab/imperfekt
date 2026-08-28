from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "mimic_iv_ed_results"
RANDOM_STATE = 42
WINDOW_HOURS = 6
MIN_OBS = 6
MAX_MISSINGNESS = 0.5
OUTCOME_COL = "critical_outcome"

CV_N_SPLITS = 5
CV_N_REPEATS = 10

VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

IREG_FEATURE_COLS = [
    "interval_seconds",
    "interval_z_score",
    "interval_cv_local",
    "interval_acceleration",
    "rolling_mean_acceleration",
    "rolling_abs_acceleration",
    "rolling_std_acceleration",
]

SPEARMAN_TOP_K_PHYS = 10
SPEARMAN_TOP_K_STRUCT = 10
