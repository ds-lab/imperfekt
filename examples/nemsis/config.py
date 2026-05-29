from pathlib import Path

PATH = Path(__file__).parent.parent

DATASET_NAME = "nemsis" # "nemsis" or "mcmed"
NEMSIS_YEAR = "2024+2025" # "2024", "2025" or combo ("2024+2025")

if DATASET_NAME == "nemsis":
    COHORT_WINDOW_MINUTES = 20
    COHORT_MIN_READINGS = 5
elif DATASET_NAME == "mcmed":
    COHORT_WINDOW_MINUTES = 60
    COHORT_MIN_READINGS = 15
    
FILTER_ALWAYS_NULL_VITALS = False

CLINICAL_ENDPOINT = "sepsis" # "destination" or "sepsis"
if DATASET_NAME == "nemsis":
    S3_BASE = f"ewai/data/nemsis/{NEMSIS_YEAR}/raw_parquet"
    COHORT_PATH = (
        f"{PATH}/data/{DATASET_NAME}/"
        f"{CLINICAL_ENDPOINT}_{NEMSIS_YEAR}_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}_{FILTER_ALWAYS_NULL_VITALS}.parquet"
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
    VITAL_COLS = ["sbp", "hr", "o2sat", "rr", "1min_HRV", "5min_HRV"]
    REQUIRED_VITAL_COLS = ["sbp", "hr", "o2sat", "rr"] # for cohort inclusion; HRV columns may be missing for some cases
else:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")


STAGE_3_CONFIGS: dict[str, dict[str, str]] = {
    "iq_pk_in": {"method": "iqr", "plaus": "keep",   "imp": "none"},
    "iq_pk_il": {"method": "iqr", "plaus": "keep",   "imp": "locf"},
    "iq_pr_in": {"method": "iqr", "plaus": "remove", "imp": "none"},
    "iq_pr_il": {"method": "iqr", "plaus": "remove", "imp": "locf"},
    "ma_pk_in": {"method": "mad", "plaus": "keep",   "imp": "none"},
    "ma_pk_il": {"method": "mad", "plaus": "keep",   "imp": "locf"},
    "ma_pr_in": {"method": "mad", "plaus": "remove", "imp": "none"},
    "ma_pr_il": {"method": "mad", "plaus": "remove", "imp": "locf"},
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