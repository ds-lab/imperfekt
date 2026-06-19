import hashlib
import json
from pathlib import Path

import polars as pl

PATH = Path(__file__).parent.parent

DATASET_NAME = "mcmed" # "nemsis" or "mcmed"
NEMSIS_YEAR = "2024+2025" # "2024", "2025" or combo ("2024+2025")

def _debug_stay_sample(lf: pl.LazyFrame) -> pl.DataFrame:
    """Stratified-by-outcome sample of whole stays for the debug slice.

    Draws DEBUG_N_STAYS stay ids stratified by their (id-level) label, keeping
    every row of each sampled stay. At least DEBUG_MIN_POS_FRAC of the sampled
    stays are positive (capped at the positives available), so CV always sees
    both classes. Deterministic given RANDOM_STATE.
    """
    # Sort by id so the row order fed to sample() is deterministic: group_by
    # output order is not stable, which would otherwise make the seeded sample
    # draw different stays on each call.
    stay_labels = (
        lf.group_by("id")
        .agg(pl.col("label").max().alias("label"))
        .sort("id")
        .collect()
    )

    pos = stay_labels.filter(pl.col("label") == 1)
    neg = stay_labels.filter(pl.col("label") != 1)

    n_pos = min(
        pos.height,
        max(int(round(DEBUG_N_STAYS * DEBUG_MIN_POS_FRAC)), 0),
    )
    n_neg = min(neg.height, DEBUG_N_STAYS - n_pos)

    sampled_pos = pos.sample(n=n_pos, shuffle=True, seed=RANDOM_STATE)
    sampled_neg = neg.sample(n=n_neg, shuffle=True, seed=RANDOM_STATE)
    sampled_ids = pl.concat([sampled_pos, sampled_neg])["id"]

    return lf.filter(pl.col("id").is_in(sampled_ids)).collect()


def load_cohort(cohort_path: Path | str | None = None) -> pl.DataFrame:
    """Read the cohort parquet, applying the debug stay sample when DEBUG.

    All entry points should load the cohort through this helper so the debug
    slice is applied consistently and matches what data_fingerprint records.
    """
    path = Path(cohort_path) if cohort_path is not None else Path(COHORT_PATH)
    if not DEBUG:
        return pl.read_parquet(path)
    return _debug_stay_sample(pl.scan_parquet(path))


def data_fingerprint(cohort_path: Path | str | None = None) -> dict:
    """Identity of the data actually fed to the pipeline, for cache keys.

    Combines the cohort file's stat (mtime + size) with the active debug
    sampling settings so that changing the file *or* the DEBUG / sampling
    parameters invalidates any cache keyed on it. Without the sampling params,
    changing them would leave the file unchanged and stale results would be
    served.
    """
    path = Path(cohort_path) if cohort_path is not None else Path(COHORT_PATH)
    st = path.stat()
    return {
        "cohort_path": str(path),
        "cohort_mtime_ns": st.st_mtime_ns,
        "cohort_size": st.st_size,
        "debug": DEBUG,
        "n_stays": DEBUG_N_STAYS if DEBUG else None,
        "min_pos_frac": DEBUG_MIN_POS_FRAC if DEBUG else None,
    }


def data_fingerprint_tag(cohort_path: Path | str | None = None) -> str:
    """Short stable hash of data_fingerprint, for use in cache directory names.

    Embedding this in the cache path (rather than only in the meta key) gives
    each distinct data setting its own cache slot, so switching DEBUG settings
    back and forth reuses the prior slice instead of overwriting it. Stale slots
    for abandoned settings accumulate on disk and must be cleaned up manually.
    """
    payload = json.dumps(data_fingerprint(cohort_path), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]

# Run full cohort or just a slice (with sure positives)
DEBUG = True
DEBUG_N_STAYS = 1200 * 10
DEBUG_MIN_POS_FRAC = 0.1

if DATASET_NAME == "nemsis":
    COHORT_WINDOW_MINUTES = 20
    COHORT_MIN_READINGS = 5
    COHORT_MAX_READINGS = 40
elif DATASET_NAME == "mcmed":
    COHORT_WINDOW_MINUTES = 60
    COHORT_MIN_READINGS = 15
    COHORT_MAX_READINGS = None

FILTER_ALWAYS_NULL_VITALS = False

CLINICAL_ENDPOINT = "sepsis" # "destination" or "sepsis"
if DATASET_NAME == "nemsis":
    S3_BASE = f"ewai/data/nemsis/{NEMSIS_YEAR}/raw_parquet"
    COHORT_PATH = (
        f"{PATH}/data/{DATASET_NAME}/"
        f"{CLINICAL_ENDPOINT}_{NEMSIS_YEAR}_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}_{COHORT_MAX_READINGS}_{FILTER_ALWAYS_NULL_VITALS}.parquet"
    )
    RESULTS_DIR = PATH / f"data/{DATASET_NAME}/post_publication_results"
    VITAL_COLS = ["sbp", "hr", "o2sat", "rr"]
    REQUIRED_VITAL_COLS = VITAL_COLS # for cohort inclusion
elif DATASET_NAME == "mcmed":
    S3_BASE = f"ewai/data/mc-med/data/parquet"
    COHORT_PATH = (
        f"{PATH}/data/{DATASET_NAME}/"
        f"{CLINICAL_ENDPOINT}_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}_{FILTER_ALWAYS_NULL_VITALS}.parquet"
    )
    RESULTS_DIR = PATH / f"data/{DATASET_NAME}/post_publication_results"
    VITAL_COLS = ["sbp", "hr", "o2sat", "rr"] #"1min_HRV", "5min_HRV"
    REQUIRED_VITAL_COLS = ["sbp", "hr", "o2sat", "rr"] # for cohort inclusion; HRV columns may be missing for some cases
else:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")


# Pre-trained SAITS models live here, one per plausibility branch, written by
# train_saits.py as ``<DATASET_NAME>_plaus_<keep|remove>.pypots``. Single source
# of truth so the trainer and the experiment pipeline agree on the location.
SAITS_MODEL_DIR = RESULTS_DIR / data_fingerprint_tag(COHORT_PATH) / "saits"


def saits_model_path(plaus: str) -> Path:
    """Path to the pre-trained SAITS model for a plausibility branch.

    ``plaus`` is "keep" or "remove" (as stored in STAGE_3_CONFIGS). The returned
    path may not exist yet — callers that want to skip-when-missing should check
    ``.exists()`` (see prep._apply_imputation).
    """
    return SAITS_MODEL_DIR / f"{DATASET_NAME}_plaus_{plaus}.pypots"


STAGE_3_CONFIGS: dict[str, dict[str, str]] = {
    # "iq_pk_in": {"method": "iqr", "plaus": "keep",   "imp": "none"},
    # "iq_pk_il": {"method": "iqr", "plaus": "keep",   "imp": "locf"},
    # "iq_pr_in": {"method": "iqr", "plaus": "remove", "imp": "none"},
    # "iq_pr_il": {"method": "iqr", "plaus": "remove", "imp": "locf"},
    "ma_pk_in": {"method": "mad", "plaus": "keep",   "imp": "none"},
    "ma_pk_il": {"method": "mad", "plaus": "keep",   "imp": "locf"},
    "ma_pk_is": {"method": "mad", "plaus": "keep",   "imp": "saits"},
    "ma_pr_in": {"method": "mad", "plaus": "remove", "imp": "none"},
    "ma_pr_il": {"method": "mad", "plaus": "remove", "imp": "locf"},
    "ma_pr_is": {"method": "mad", "plaus": "remove", "imp": "saits"},
}

STAGE_4_CONFIGS: dict[str, dict[str, bool]] = {
    "base": {"base": True, "miss": False, "plaus": False},
    "base+miss": {"base": True, "miss": True,  "plaus": False},
    "base+plaus": {"base": True, "miss": False, "plaus": True},
    "base+miss+plaus": {"base": True, "miss": True,  "plaus": True},
}

STRUCTURAL_FEATURE_COLS = []

RANDOM_STATE = 42
CV_N_SPLITS = 5
CV_N_REPEATS = 10

# "intervariable" uses avg_indicated_vars_pct × pattern_entropy (co-missingness structure)
# "intravariable"  uses the least-correlated {col}_indicated_pct pair (per-variable burden)
STRATIFICATION_MODE: str = "intervariable"  # or "intravariable" / "intervariable"

AXES_INTERVARIABLE = ("avg_indicated_vars_pct", "pattern_entropy")
AXES_INTRAVARIABLE = ("sbp_indicated_pct", "rr_indicated_pct")

AXES = AXES_INTRAVARIABLE if STRATIFICATION_MODE == "intravariable" else AXES_INTERVARIABLE

# Train-fold undersampling + Bayesian prior correction.
# Within each CV fold we keep all positives and randomly subsample negatives
# (without replacement) to a fixed pos:neg ratio before fitting XGBoost only.
# The held-out validation fold is never undersampled, so it retains the natural
# (~0.19%) prevalence; raw predict_proba is then rescaled in odds space back to
# the true eligible-cohort prevalence (prior correction).
APPLY_UNDERSAMPLING = True  # set False to train on the full (imbalanced) train fold
TRAIN_NEG_POS_RATIO = 10  # target 1:10 pos:neg in the undersampled training set
UNDERSAMPLE_RANDOM_STATE = 1234  # combined with the fold index for per-fold reproducibility
APPLY_PRIOR_CORRECTION = True

# SHAP subsampling: explaining every held-out test row with TreeExplainer is
# expensive for large cohorts and the mean |SHAP| per feature is just a row
# average, so a random subsample gives an unbiased estimate at a fraction of the
# cost. Subsampling is *stratified* (capped per intervariable stratum) so the
# rare strata keep enough rows for a usable per-stratum estimate. Set to None to
# explain all test rows (exact, slow).
SHAP_MAX_ROWS_PER_STRATUM = 2000
SHAP_SUBSAMPLE_RANDOM_STATE = 7

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


