# %%
from pathlib import Path

import polars as pl
import s3fs

from config import (
    CLINICAL_ENDPOINT,
    COHORT_MAX_READINGS,
    COHORT_MIN_READINGS,
    COHORT_PATH,
    COHORT_WINDOW_MINUTES,
    DATASET_NAME,
    FILTER_ALWAYS_NULL_VITALS,
    NEMSIS_YEAR,
    PATH,
    REQUIRED_VITAL_COLS,
    S3_BASE,
    VITAL_COLS,
)
from prep import filter_cohort

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

LOCAL = False
PERTINENT_NEGATIVE_CODES = [8801019, 8801023, 7701001, 7701003, 8801005]

NEMSIS_FILES = {
    "2024": {
        "events": "pcr_events.parquet",
        "vitals": "vitals.parquet",
        "diagnosis": "hospital_diagnoses.parquet",
    },
    "2025": {
        "events": "Pub_PCRevents.parquet",
        "vitals": "FACTPCRVITAL.parquet",
        "diagnosis": "FactPcreOutcomeHospDiag.parquet",
    },
}

MCMED_FILES = {
    "visits": "visits.parquet",
    "vitals": "numerics.parquet",
}


fs = s3fs.S3FileSystem(
    endpoint_url="https://s3.storage.ds-lab.org",
    profile="seaweedfs",
)

COHORT_PATH = Path(COHORT_PATH)
COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log(label: str, df: pl.DataFrame, id_key: str) -> None:
    print(f"  {df[id_key].n_unique():>9,}  {id_key}  │  {label}")


def _read_s3_parquet(file_name: str, columns: list[str] | None = None) -> pl.DataFrame:
    with fs.open(f"{S3_BASE}/{file_name}") as f:
        df = pl.read_parquet(f)
        return df.select(columns) if columns else df


def _scan_s3_parquet(file_name: str) -> pl.LazyFrame:
    with fs.open(f"{S3_BASE}/{file_name}") as f:
        return pl.scan_parquet(f)


def _nemsis_read_parquet(file_name: str, columns: list[str] | None = None) -> pl.DataFrame:
    if LOCAL:
        return pl.read_parquet(f"{PATH}/data/{DATASET_NAME}/{NEMSIS_YEAR}/{file_name}", columns=columns)
    return _read_s3_parquet(file_name, columns=columns)


def _nemsis_scan_vitals(file_name: str) -> pl.LazyFrame:
    if LOCAL:
        return pl.scan_parquet(f"{PATH}/data/{DATASET_NAME}/{NEMSIS_YEAR}/{file_name}")
    return _scan_s3_parquet(file_name)


def _apply_binary_label(df: pl.DataFrame, id_col: str, positive_ids: pl.DataFrame) -> pl.DataFrame:
    """Left-join positive_ids (with a `label=1` column) onto df and fill missing labels with 0."""
    return df.join(positive_ids, on=id_col, how="left").with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )


def _nemsis_build_combined_years() -> pl.DataFrame:
    """Concatenate prebuilt 2024 and 2025 cohorts with year-suffixed IDs."""
    path_2024 = COHORT_PATH.parent / f"{CLINICAL_ENDPOINT}_2024_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}_{COHORT_MAX_READINGS}_{FILTER_ALWAYS_NULL_VITALS}.parquet"
    path_2025 = COHORT_PATH.parent / f"{CLINICAL_ENDPOINT}_2025_{COHORT_WINDOW_MINUTES}_{COHORT_MIN_READINGS}_{COHORT_MAX_READINGS}_{FILTER_ALWAYS_NULL_VITALS}.parquet"
    for p in (path_2024, path_2025):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Please build the {p.stem.split('_')[1]} cohort first.")

    df_2024 = pl.read_parquet(path_2024).with_columns(
        (pl.col("id").cast(pl.Utf8) + "_2024").alias("id")
    )
    df_2025 = pl.read_parquet(path_2025).with_columns(
        (pl.col("id").cast(pl.Utf8) + "_2025").alias("id")
    )
    return pl.concat([df_2024, df_2025], how="vertical")


def _nemsis_label_sepsis(call_df: pl.DataFrame, diagnosis_file_name: str) -> pl.DataFrame:
    diagnosis_df = _nemsis_read_parquet(diagnosis_file_name, columns=["PcrKey", "eOutcome_13"])
    sepsis_ids = (
        diagnosis_df.filter(pl.col("eOutcome_13").str.contains(r"^A41|^R65"))
        .select("PcrKey")
        .unique()
        .with_columns(pl.lit(1).alias("label"))
    )
    return _apply_binary_label(call_df, "PcrKey", sepsis_ids)


def _nemsis_label_destination(call_df: pl.DataFrame) -> pl.DataFrame:
    # Class 1: critically ill / ICU-level destinations.
    icu_codes = [
        "4222013",  # ICU
        "4222005",  # CCU
        "4222021",  # MICU
        "4222049",  # SICU
        #  "4222031", # OR (Optional: Add if you want to capture emergency surgery)
        #  "4222003", # Cath Lab (Optional: Add if you want to capture heart attacks)
    ]
    # Class 0: stable admitted / general ward destinations.
    ward_codes = [
        "4222017",  # Med/Surg
        "4222033",  # Orthopedic
        # "4222051",  # Oncology
    ]
    return call_df.filter(pl.col("eDisposition_22").is_in(icu_codes + ward_codes)).with_columns(
        pl.when(pl.col("eDisposition_22").is_in(icu_codes))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("label"),
    )


def _filter_always_null_vitals(df: pl.DataFrame, id_key: str) -> pl.DataFrame:
    """Drop IDs where any one of REQUIRED_VITAL_COLS is null for every row of that ID."""
    keep_ids = (
        df.group_by(id_key)
        .agg([pl.col(c).is_not_null().any().alias(c) for c in REQUIRED_VITAL_COLS])
        .filter(pl.all_horizontal([pl.col(c) for c in REQUIRED_VITAL_COLS]))
        .select(id_key)
    )
    return df.join(keep_ids, on=id_key, how="semi")


def _nemsis_load_vitals(vitals_file_name: str, keys: pl.DataFrame) -> pl.DataFrame:
    return (
        _nemsis_scan_vitals(vitals_file_name)
        .select(["PcrKey", "eVitals_01", "eVitals_06", "eVitals_10", "eVitals_12", "eVitals_14"])
        .join(keys.lazy().select("PcrKey"), on="PcrKey", how="semi")
        .rename({
            "eVitals_01": "clock",
            "eVitals_06": "sbp",
            "eVitals_10": "hr",
            "eVitals_12": "o2sat",
            "eVitals_14": "rr",
        })
        .with_columns(pl.col("clock").str.to_datetime("  %d%b%Y:%H:%M:%S", strict=False))
        .with_columns([
            pl.when(pl.col(c).is_in(PERTINENT_NEGATIVE_CODES))
            .then(pl.lit(None))
            .otherwise(pl.col(c))
            .alias(c)
            for c in VITAL_COLS
        ])
        .filter(pl.col("clock").is_not_null())
        .sort(["PcrKey", "clock"])
        .unique(subset=["PcrKey", "clock"], keep="first", maintain_order=True)
        .filter(~pl.all_horizontal(pl.col(c).is_null() for c in VITAL_COLS))
        .collect()
    )


def _build_nemsis_cohort() -> pl.DataFrame:
    if NEMSIS_YEAR == "2024+2025":
        return _nemsis_build_combined_years()

    if NEMSIS_YEAR not in NEMSIS_FILES:
        raise ValueError(f"Unsupported NEMSIS_YEAR: {NEMSIS_YEAR}")
    files = NEMSIS_FILES[NEMSIS_YEAR]

    # Note: NEMSIS sex/gender (ePatient_13 / ePatient_25) is not present in the 2024 or 2025 parquet
    # exports, so it is omitted from the cohort.
    events_df = _nemsis_read_parquet(
        files["events"],
        columns=["PcrKey", "eResponse_05", "eDisposition_22", "ePatient_15", "ePatient_16"],
    )
    _log("all events loaded", events_df, "PcrKey")

    # Only 911 calls, emergency responses
    call_df = events_df.filter(pl.col("eResponse_05").is_in(["2205001", "2205003", "2205009"]))
    _log("after 911/emergency response filter (eResponse_05)", call_df, "PcrKey")
    
    call_df = call_df.filter((pl.col("ePatient_15") >= 18.0) & (pl.col("ePatient_16") == "2516009"))
    _log("after age >= 18 filter", call_df, "PcrKey")

    if CLINICAL_ENDPOINT == "sepsis":
        binary_df = _nemsis_label_sepsis(call_df, files["diagnosis"])
        _log("after sepsis label join", binary_df, "PcrKey")
    elif CLINICAL_ENDPOINT == "destination":
        binary_df = _nemsis_label_destination(call_df)
        _log("after destination filter (ICU or ward codes)", binary_df, "PcrKey")
    else:
        raise ValueError(f"Unsupported CLINICAL_ENDPOINT: {CLINICAL_ENDPOINT}")

    binary_df = binary_df.select(["PcrKey", "label", "ePatient_15"])

    vitals_df = _nemsis_load_vitals(files["vitals"], binary_df)
    _log("after clock parse + pertinent-negative recode + dedup + null row filtering", vitals_df, "PcrKey")
    vitals_df = vitals_df.rename({"PcrKey": "id", })
    binary_df = binary_df.rename({"PcrKey": "id", "ePatient_15": "age"})

    if FILTER_ALWAYS_NULL_VITALS:
        vitals_df = _filter_always_null_vitals(vitals_df, "id")
        _log("after dropping IDs with any always-null required vital column", vitals_df, "id")

    vitals_df = filter_cohort(vitals_df)
    _log("after filtering to cohort window and min readings", vitals_df, "id")

    df = (
        binary_df.join(vitals_df, on="id", how="inner")
        .select(["id", "clock", "sbp", "hr", "o2sat", "rr", "label", "age"])
    )
    _log("final cohort (after inner join with vitals)", df, "id")
    return df


def _build_mcmed_cohort() -> pl.DataFrame:
    if LOCAL:
        raise NotImplementedError("Local loading not implemented for MCMED yet.")
    files = MCMED_FILES

    visits_df = _read_s3_parquet(files["visits"])
    
    _log("all visits loaded", visits_df, "CSN")

    measures = ["SpO2", "Perf", "SBP", "DBP", "MAP", "HR", "RR"] # "1min_HRV", "5min_HRV"
    vitals_df = (
        _scan_s3_parquet(files["vitals"])
        .filter((pl.col("Source") == "Monitor") & pl.col("Measure").is_in(measures))
        .collect()
        .pivot(
            on="Measure",
            index=["CSN", "Time"],
            values="Value",
            aggregate_function="max",
        )
        .sort(["CSN", "Time"])
    )
    _log("all vitals loaded", vitals_df, "CSN")

    visits_df = visits_df.filter(pl.col("Age") >= 18)
    _log("after age >= 18 filter", visits_df, "CSN")

    if CLINICAL_ENDPOINT == "sepsis":
        def _parse_icd_cell(value: str | None) -> list[str]:
            if not value or not value.strip():
                return []
            return [p.strip() for p in value.split(",") if p.strip()]

        def matches(value: str | None) -> bool:
            return any(
                code.upper().replace(".", "").startswith(p.replace(".", ""))
                for code in _parse_icd_cell(value)
                for p in ["A41", "R65"]
            )

        positive_ids = (
            visits_df.filter(pl.col("Dx_ICD10").map_elements(matches, return_dtype=pl.Boolean))
            .select("CSN")
            .unique()
            .with_columns(pl.lit(1).alias("label"))
        )
        visits_df = _apply_binary_label(visits_df, "CSN", positive_ids)
        _log("after sepsis label join", visits_df, "CSN")

    elif CLINICAL_ENDPOINT == "destination":
        positive_ids = (
            visits_df.filter(pl.col("ED_dispo") == "ICU")
            .select("CSN")
            .unique()
            .with_columns(pl.lit(1).alias("label"))
        )
        visits_df = _apply_binary_label(visits_df, "CSN", positive_ids)
        _log("after destination label join", visits_df, "CSN")
    else:
        raise ValueError(f"Unsupported CLINICAL_ENDPOINT: {CLINICAL_ENDPOINT}")

    vitals_df = (
        vitals_df.rename({"CSN": "id", "Time": "clock", "SBP": "sbp", "HR": "hr", "SpO2": "o2sat", "RR": "rr"})
        .with_columns(pl.col("clock").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False))
        .filter(pl.col("clock").is_not_null())
        .filter(~pl.all_horizontal(pl.col(c).is_null() for c in VITAL_COLS))
    )
    _log("after clock parse + null row filtering", vitals_df, "id")

    if FILTER_ALWAYS_NULL_VITALS:
        vitals_df = _filter_always_null_vitals(vitals_df, "id")
        _log("after dropping IDs with any always-null required vital column", vitals_df, "id")

    vitals_df = filter_cohort(vitals_df)
    _log("after filtering to cohort window and min readings", vitals_df, "id")

    df = (
        visits_df.select(["CSN", "label", "Age"])
        .rename({"CSN": "id", "Age": "age"})
        .join(vitals_df, on="id", how="inner")
        .select(["id", "clock", "label", "age", *VITAL_COLS])
    )
    _log("after inner join with vitals", df, "id")
    return df


if COHORT_PATH.exists():
    df = pl.read_parquet(COHORT_PATH)
elif DATASET_NAME == "mcmed":
    df = _build_mcmed_cohort()
    df.write_parquet(COHORT_PATH)
    print(f"Built cohort and saved to {COHORT_PATH}")
elif DATASET_NAME == "nemsis":
    df = _build_nemsis_cohort()
    df.write_parquet(COHORT_PATH)
    print(f"Built cohort and saved to {COHORT_PATH}")
else:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

# %%

# %% cohort size summary (always runs; per-step counts print only on fresh build)
print(f"  {df['id'].n_unique():>9,}  IDs  │  final cohort (loaded from cache or built fresh)")
print(df.group_by("label").agg(pl.col("id").n_unique().alias("n_ids")).sort("label"))
# %% describe number of unique clock values per ID to get a sense of how many time points we have per patient
df.group_by("id").agg(pl.col("clock").n_unique().alias("num_time_points")).describe()

# %% Describe length first to last vital sign per patient to get a sense of how long the time series are
df.group_by("id").agg((pl.col("clock").max() - pl.col("clock").min()).alias("duration")).select(
    "duration"
).describe()

# %% get class distribution for different lengths (10,20,30 minutes) and minimum count of vitals per patient
thresholds = [10, 20, 30, 45, 60, 120, 99999]
min_vitals = [3, 5, 8, 10]

# Attach each patient's first clock so we can compute per-row offset
df_with_start = df.with_columns(
    pl.col("clock").min().over("id").alias("start_clock")
).with_columns(
    ((pl.col("clock") - pl.col("start_clock")).dt.total_minutes()).alias("minutes_from_start")
)

for t in thresholds:
    patient_stats = (
        df_with_start.filter(pl.col("minutes_from_start") <= t)
        .group_by("id")
        .agg(
            pl.len().alias("num_vitals"),
            pl.col("label").first().alias("label"),
        )
    )
    for min_v in min_vitals:
        cohort = patient_stats.filter(pl.col("num_vitals") >= min_v)
        dist = cohort.group_by("label").agg(pl.len().alias("n")).sort("label")
        print(f"window <= {t}min, vitals >= {min_v}: {dist.to_dicts()}")
# %%