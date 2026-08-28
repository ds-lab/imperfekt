DATASET_NAME = "nemsis"  # "nemsis" or "mcmed"
NEMSIS_YEAR = "2025"  # "2024", "2025" or combo ("2024+2025")

COHORT_WINDOW_MINUTES = 20
COHORT_MIN_READINGS = 5

CLINICAL_ENDPOINT = "sepsis"  # "destination" or "sepsis"
COHORT_PATH = f"/workspaces/imperfekt/data/{DATASET_NAME}/{CLINICAL_ENDPOINT}_{NEMSIS_YEAR}_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}.parquet"

if DATASET_NAME == "nemsis":
    S3_BASE = f"ewai/data/nemsis/{NEMSIS_YEAR}/raw_parquet"
elif DATASET_NAME == "mcmed":
    S3_BASE = "ewai/data/mc-med/data/parquet"
