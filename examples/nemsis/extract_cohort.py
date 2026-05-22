# %%
from pathlib import Path

import polars as pl
import s3fs

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

S3_BASE = "ewai/data/nemsis/2024/raw_parquet"
fs = s3fs.S3FileSystem(
    endpoint_url="https://s3.storage.ds-lab.org",
    profile="seaweedfs",
)

if Path("/workspaces/imperfekt/data/nemsis/destinations.parquet").exists():
    df = pl.read_parquet(Path("/workspaces/imperfekt/data/nemsis/destinations.parquet"))
else:
    with fs.open(f"{S3_BASE}/pcr_events.parquet") as f:
        events_df = pl.read_parquet(f)
    with fs.open(f"{S3_BASE}/vitals.parquet") as f:
        vitals_df = pl.read_parquet(f)
    def _log(label: str, df: pl.DataFrame) -> None:
        print(f"  {df['PcrKey'].n_unique():>9,}  PcrKeys  │  {label}")

    _log("all events loaded", events_df)

    # Only 911 calls, emergency resonses
    call_df = events_df.filter(pl.col("eResponse_05").is_in(["2205001", "2205003", "2205009"]))
    _log("after 911/emergency response filter (eResponse_05)", call_df)

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
        "4222051",  # Oncology
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
    _log("after destination filter (ICU or ward codes)", binary_df)

    # Filter out underage patients
    binary_df = binary_df.filter(pl.col("ePatient_15") >= 18.0)
    _log("after age >= 18 filter", binary_df)

    # Select necessary columns for joining and modeling
    binary_df = binary_df.select(["PcrKey", "label"])
    # join on PcrKey
    df = binary_df.join(vitals_df, on="PcrKey", how="inner")
    _log("after inner join with vitals", df)

    df = df.rename(
        {
            "eVitals_01": "clock",
            "eVitals_06": "sbp",
            "eVitals_10": "hr",
            "eVitals_12": "o2sat",
            "eVitals_14": "rr",
        }
    )
    df = df.select(["PcrKey", "clock", "sbp", "hr", "o2sat", "rr", "label"])
    date_df = df.with_columns(
        pl.col("clock").str.to_datetime("  %d%b%Y:%H:%M:%S", strict=False).alias("clock")
    ).filter(pl.col("clock").is_not_null())
    _log("after clock parse (drop unparseable timestamps)", date_df)

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
    _log("final cohort (pertinent negatives recoded to null)", date_df)

    date_df.write_parquet(Path("/workspaces/imperfekt/data/nemsis/destinations.parquet"))
    df = date_df
# %% deduplicate: keep first row per (PcrKey, clock) — must happen before any filtering
df = df.sort(["PcrKey", "clock"]).unique(subset=["PcrKey", "clock"], keep="first", maintain_order=True)

# %% cohort size summary (always runs; per-step counts print only on fresh build)
print(f"  {df['PcrKey'].n_unique():>9,}  PcrKeys  │  final cohort (loaded from cache or built fresh)")
print(df.group_by("label").agg(pl.col("PcrKey").n_unique().alias("n_PcrKeys")).sort("label"))
# %% describe number of unique clock values per PCR key to get a sense of how many time points we have per patient
df.group_by("PcrKey").agg(pl.col("clock").n_unique().alias("num_time_points")).describe()

# %% Describe length first to last vital sign per patient to get a sense of how long the time series are
df.group_by("PcrKey").agg((pl.col("clock").max() - pl.col("clock").min()).alias("duration")).select(
    "duration"
).describe()

# %% get class distribution for different lengths (10,20,30 minutes) and minimum count of vitals per patient
thresholds = [10, 20, 30, 45, 60, 120, 99999]
min_vitals = [3, 5, 8, 10]

# Attach each patient's first clock so we can compute per-row offset
df_with_start = df.with_columns(
    pl.col("clock").min().over("PcrKey").alias("start_clock")
).with_columns(
    ((pl.col("clock") - pl.col("start_clock")).dt.total_minutes()).alias("minutes_from_start")
)

for t in thresholds:
    patient_stats = (
        df_with_start.filter(pl.col("minutes_from_start") <= t)
        .group_by("PcrKey")
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
