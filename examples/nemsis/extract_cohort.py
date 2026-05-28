# %%
from pathlib import Path

import polars as pl
import s3fs

from config import S3_BASE, DATASET_NAME, NEMSIS_YEAR, COHORT_PATH, CLINICAL_ENDPOINT
from prep import filter_cohort

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)


fs = s3fs.S3FileSystem(
    endpoint_url="https://s3.storage.ds-lab.org",
    profile="seaweedfs",
)

local = True

while(True):
    if Path(COHORT_PATH).exists():
        df = pl.read_parquet(Path(COHORT_PATH))
        break
    else:
        if DATASET_NAME == "mcmed":
            print("TBD")
            pass
        elif DATASET_NAME == "nemsis":
            if NEMSIS_YEAR == "2024":
                events_file_name = "pcr_events.parquet"
                vitals_file_name = "vitals.parquet"
                diagnosis_file_name = "hospital_diagnoses.parquet"
            elif NEMSIS_YEAR == "2025":
                events_file_name = "Pub_PCRevents.parquet"
                vitals_file_name = "FACTPCRVITAL.parquet"
                diagnosis_file_name = "FactPcreOutcomeHospDiag.parquet"
            elif NEMSIS_YEAR == "2024+2025":
                # Check if COHORT_PATH exists for 2024 abd 2025, if not tell user to build those first since we need them both to build the combo.
                path_2024 = f"/workspaces/imperfekt/data/nemsis/{CLINICAL_ENDPOINT}_2024.parquet"
                path_2025 = f"/workspaces/imperfekt/data/nemsis/{CLINICAL_ENDPOINT}_2025.parquet"
                if not Path(path_2024).exists():
                    raise FileNotFoundError(f"{path_2024} not found. Please build the 2024 cohort first.")
                if not Path(path_2025).exists():
                    raise FileNotFoundError(f"{path_2025} not found. Please build the 2025 cohort first.")
                # If both exist, read them and concatenate
                df_2024 = pl.read_parquet(path_2024)
                df_2025 = pl.read_parquet(path_2025)
                # append to the id the year
                df_2024 = df_2024.with_columns((pl.col("id").cast(pl.Utf8) + "_2024").alias("id"))
                df_2025 = df_2025.with_columns((pl.col("id").cast(pl.Utf8) + "_2025").alias("id"))
                df = pl.concat([df_2024, df_2025], how="vertical")
                # parquet
                df.write_parquet(Path(COHORT_PATH))
                print(f"Combined 2024 and 2025 cohorts and saved to {COHORT_PATH}")
                break
            if local:
                events_df = pl.read_parquet(f"/workspaces/imperfekt/data/nemsis/{NEMSIS_YEAR}/{events_file_name}", columns=["PcrKey", "eResponse_05", "eDisposition_22", "ePatient_15"])
            else:
                with fs.open(f"{S3_BASE}/{events_file_name}") as f:
                    events_df = pl.read_parquet(f).select(["PcrKey", "eResponse_05", "eDisposition_22", "ePatient_15"])

            def _log(label: str, df: pl.DataFrame, id_key: str) -> None:
                print(f"  {df[id_key].n_unique():>9,}  {id_key}  │  {label}")

            _log("all events loaded", events_df, "PcrKey")

            # Only 911 calls, emergency resonses
            call_df = events_df.filter(pl.col("eResponse_05").is_in(["2205001", "2205003", "2205009"]))
            del events_df
            _log("after 911/emergency response filter (eResponse_05)", call_df, "PcrKey")
            if CLINICAL_ENDPOINT == "sepsis":
                if local: 
                    diagnosis_df = pl.read_parquet(f"/workspaces/imperfekt/data/nemsis/{NEMSIS_YEAR}/{diagnosis_file_name}")
                else: 
                    with fs.open(f"{S3_BASE}/{diagnosis_file_name}") as f:
                        diagnosis_df = pl.read_parquet(f).select(["PcrKey", "eOutcome_13"]) 
                sepsis_ids = diagnosis_df.filter(pl.col("eOutcome_13").str.contains(r"^A41|^R65")).select("PcrKey").unique()
                del diagnosis_df

                # Attach label at id level so every timestamp row carries it.
                binary_df = call_df.join(
                    sepsis_ids.with_columns(pl.lit(1).alias("label")), on="PcrKey", how="left"
                ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
                del call_df, sepsis_ids

                _log("after sepsis label join", binary_df, "PcrKey")
            elif CLINICAL_ENDPOINT == "destination":
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
                    #"4222051",  # Oncology
                ]

                # Use this list to filter to only the ICU and ward dispositions above.
                filter_for = icu_codes + ward_codes

                binary_df = call_df.filter(pl.col("eDisposition_22").is_in(filter_for)).with_columns(
                    pl.when(pl.col("eDisposition_22").is_in(icu_codes))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .cast(pl.Int8)
                    .alias("label"),
                )
                del call_df
                _log("after destination filter (ICU or ward codes)", binary_df, "PcrKey")

            # Filter out underage patients
            binary_df = binary_df.filter(pl.col("ePatient_15") >= 18.0)
            _log("after age >= 18 filter", binary_df, "PcrKey")

            # Select necessary columns for joining and modeling
            binary_df = binary_df.select(["PcrKey", "label"])

            if local:
                vitals_scan = pl.scan_parquet(f"/workspaces/imperfekt/data/nemsis/{NEMSIS_YEAR}/{vitals_file_name}")
            else:
                with fs.open(f"{S3_BASE}/{vitals_file_name}") as f:
                    vitals_scan = pl.scan_parquet(f)
            vitals_df = (
                vitals_scan
                .select(["PcrKey", "eVitals_01", "eVitals_06", "eVitals_10", "eVitals_12", "eVitals_14"])
                .join(binary_df.lazy().select("PcrKey"), on="PcrKey", how="semi")
                .rename(
                    {
                        "eVitals_01": "clock",
                        "eVitals_06": "sbp",
                        "eVitals_10": "hr",
                        "eVitals_12": "o2sat",
                        "eVitals_14": "rr",
                    }
                )
                .with_columns(pl.col("clock").str.to_datetime("  %d%b%Y:%H:%M:%S", strict=False).alias("clock"))
                .filter(pl.col("clock").is_not_null())
                .sort(["PcrKey", "clock"])
                .unique(subset=["PcrKey", "clock"], keep="first", maintain_order=True)
                .filter(
                        ~(
                            pl.col("sbp").is_null()
                            & pl.col("hr").is_null()
                            & pl.col("o2sat").is_null()
                            & pl.col("rr").is_null()
                        ))
                .collect()
            )
            _log("after clock parse + dedup + null row filtering", vitals_df, "PcrKey")
            vitals_df = filter_cohort(vitals_df)
            _log("after filtering to cohort window and min readings", vitals_df, "PcrKey")

            df = binary_df.join(vitals_df, on="PcrKey", how="inner")
            _log("after inner join with vitals", df, "PcrKey")

            df = df.rename({"PcrKey": "id"})
            df = df.select(["id", "clock", "sbp", "hr", "o2sat", "rr", "label"])
            date_df = df
            _log("after clock parse (drop unparseable timestamps)", date_df, "id")

            # per vital turn the codes into None
            PERTINENT_NEGATIVE_CODES = [8801019, 8801023, 7701001, 7701003, 8801005]
            date_df = date_df.with_columns(
                pl.when(pl.col("sbp").is_in(PERTINENT_NEGATIVE_CODES))
                .then(pl.lit(None))
                .otherwise(pl.col("sbp"))
                .alias("sbp"),
                pl.when(pl.col("hr").is_in(PERTINENT_NEGATIVE_CODES))
                .then(pl.lit(None))
                .otherwise(pl.col("hr"))
                .alias("hr"),
                pl.when(pl.col("o2sat").is_in(PERTINENT_NEGATIVE_CODES))
                .then(pl.lit(None))
                .otherwise(pl.col("o2sat"))
                .alias("o2sat"),
                pl.when(pl.col("rr").is_in(PERTINENT_NEGATIVE_CODES))
                .then(pl.lit(None))
                .otherwise(pl.col("rr"))
                .alias("rr"),
            )
            _log("final cohort (pertinent negatives recoded to null)", date_df, "id")
            df = date_df

        df.write_parquet(Path(COHORT_PATH))
        print(f"Built cohort and saved to {COHORT_PATH}")
        break
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
