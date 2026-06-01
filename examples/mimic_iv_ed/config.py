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

# Plotting
SHOW_LEGEND = True
# Extended, colorblind-friendly palette. The first 5 are the Okabe-Ito colors
# used historically; the rest extend the cycle so many pipelines stay distinct.
# 24 distinct entries cover the full STAGE_3 × STAGE_4 grid without wrapping.
PLOT_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#999999",  # grey
    "#882255",  # wine
    "#44AA99",  # teal
    "#AA4499",  # purple
    "#117733",  # dark green
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#661100",  # dark red
    "#88CCEE",  # light blue
    "#999933",  # olive
    "#CC6677",  # rose
    "#6699CC",  # steel blue
    "#AA3377",  # magenta
    "#228833",  # forest green
    "#EE7733",  # burnt orange
    "#BBBBBB",  # light grey
]
