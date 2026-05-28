from pathlib import Path

PATH = Path(__file__).parent.parent


DATASET_NAME = "mcmed" # "nemsis" or "mcmed"
NEMSIS_YEAR = "2024+2025" # "2024", "2025" or combo ("2024+2025")

if DATASET_NAME == "nemsis":
    COHORT_WINDOW_MINUTES = 20
    COHORT_MIN_READINGS = 5
elif DATASET_NAME == "mcmed":
    COHORT_WINDOW_MINUTES = 60
    COHORT_MIN_READINGS = 15

CLINICAL_ENDPOINT = "sepsis" # "destination" or "sepsis"
if DATASET_NAME == "nemsis":
    COHORT_PATH = (
        f"{PATH}/data/{DATASET_NAME}/"
        f"{CLINICAL_ENDPOINT}_{NEMSIS_YEAR}_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}.parquet"
    )
elif DATASET_NAME == "mcmed":
    COHORT_PATH = (
        f"{PATH}/data/{DATASET_NAME}/"
        f"{CLINICAL_ENDPOINT}_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}.parquet"
    )
else:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

if DATASET_NAME == "nemsis":
    S3_BASE = f"ewai/data/nemsis/{NEMSIS_YEAR}/raw_parquet"
elif DATASET_NAME == "mcmed":
    S3_BASE = f"ewai/data/mc-med/data/parquet"