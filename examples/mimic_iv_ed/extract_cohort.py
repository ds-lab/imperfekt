import polars as pl
from functools import cache
from typing import Literal

# ── paths ────────────────────────────────────────────────────────────────────
ED_BASE = "/mnt/dataset/mimic-iv-ed/mimic-iv-ed-2.2/ed"
HOSP_BASE = "/workspaces/imperfekt/data/physionet.org/files/mimiciv/3.1/hosp"

VITALSIGN_PATH  = f"{ED_BASE}/vitalsign.csv"
DIAGNOSIS_PATH  = f"{ED_BASE}/diagnosis.csv"
EDSTAYS_PATH    = f"{ED_BASE}/edstays.csv"
ADMISSIONS_PATH = f"{HOSP_BASE}/admissions.csv.gz"

# ── ICD code definitions ──────────────────────────────────────────────────────
SEPSIS_ICD_PREFIXES = [
    # ICD-9
    "99591", "99592", "78552", "99590",
    # ICD-10
    "A40", "A41", "R65",
]

Outcome = Literal["sepsis", "mortality", "readmission_30d"]


# ── cached loaders (each file is read at most once per process) ───────────────

@cache
def _load_vitalsigns() -> pl.DataFrame:
    return (
        pl.read_csv(VITALSIGN_PATH)
        .with_columns(
            charttime=pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            temperature=pl.when(pl.col("temperature").is_not_null())
            .then((pl.col("temperature") - 32) * (5 / 9))
            .otherwise(pl.col("temperature")),
        )
    )


@cache
def _load_diagnosis() -> pl.DataFrame:
    return pl.read_csv(DIAGNOSIS_PATH)


@cache
def _load_edstays() -> pl.DataFrame:
    return pl.read_csv(EDSTAYS_PATH)


@cache
def _load_admissions() -> pl.DataFrame:
    return pl.read_csv(ADMISSIONS_PATH)


# ── cohort filters ────────────────────────────────────────────────────────────

def filter_by_icd_prefixes(
    diagnosis_df: pl.DataFrame,
    prefixes: list[str],
) -> pl.DataFrame:
    """Return rows whose icd_code starts with any of the given prefixes."""
    return diagnosis_df.filter(
        pl.any_horizontal(
            [pl.col("icd_code").str.starts_with(p) for p in prefixes]
        )
    )


# ── outcome extractors ────────────────────────────────────────────────────────

def extract_sepsis() -> pl.DataFrame:
    """One row per stay_id with sepsis=True."""
    sepsis_dx = filter_by_icd_prefixes(_load_diagnosis(), SEPSIS_ICD_PREFIXES)
    return (
        sepsis_dx.select("subject_id", "stay_id")
        .unique()
        .with_columns(sepsis=pl.lit(True))
    )


def extract_mortality() -> pl.DataFrame:
    """
    One row per stay_id with in_hospital_mortality flag.

    Join chain: stay_id → edstays (hadm_id) → admissions (hospital_expire_flag).
    """
    stay_hadm = _load_edstays().select("stay_id", "hadm_id").drop_nulls("hadm_id")
    adm_mortality = _load_admissions().select(
        "hadm_id",
        pl.col("hospital_expire_flag").cast(pl.Boolean).alias("in_hospital_mortality"),
    )
    # 26 hadm_ids in edstays have no matching row in admissions (MIMIC data gap).
    # Drop them so every returned row has a known outcome value.
    return (
        stay_hadm
        .join(adm_mortality, on="hadm_id", how="left")
        .drop_nulls("in_hospital_mortality")
    )


def extract_readmission_30d() -> pl.DataFrame:
    """
    One row per stay_id with readmission_30d flag.

    Definition: any subsequent hospital admission for the same subject_id
    within 30 days after the dischtime of the index admission.
    Patients who died in-hospital are excluded (no readmission possible).

    Join chain: stay_id → edstays (hadm_id) → admissions (dischtime)
                subject_id → admissions (admittime of next admission).
    """
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    stay_hadm = _load_edstays().select("stay_id", "hadm_id").drop_nulls("hadm_id")
    adm_parsed = _load_admissions().with_columns(
        admittime=pl.col("admittime").str.strptime(pl.Datetime, dt_fmt),
        dischtime=pl.col("dischtime").str.strptime(pl.Datetime, dt_fmt),
    )
    index_discharges = (
        stay_hadm
        .join(
            adm_parsed.select("subject_id", "hadm_id", "dischtime", "hospital_expire_flag"),
            on="hadm_id",
            how="left",
        )
        # 26 hadm_ids in edstays have no matching row in admissions (MIMIC data gap) — drop them.
        .drop_nulls("dischtime")
        .filter(pl.col("hospital_expire_flag") == 0)
        .drop("hospital_expire_flag")
    )
    readmitted = (
        index_discharges
        .join(adm_parsed.select("subject_id", "admittime"), on="subject_id", how="left")
        .filter(
            (pl.col("admittime") > pl.col("dischtime"))
            & (pl.col("admittime") <= pl.col("dischtime") + pl.duration(days=30))
        )
        .select("stay_id")
        .unique()
        .with_columns(readmission_30d=pl.lit(True))
    )
    return (
        index_discharges.select("stay_id")
        .join(readmitted, on="stay_id", how="left")
        .with_columns(pl.col("readmission_30d").fill_null(False))
    )


# ── registry ──────────────────────────────────────────────────────────────────

_OUTCOME_REGISTRY: dict[Outcome, tuple[str, callable]] = {
    "sepsis":          ("sepsis",               extract_sepsis),
    "mortality":       ("in_hospital_mortality", extract_mortality),
    "readmission_30d": ("readmission_30d",       extract_readmission_30d),
}


# ── public API ────────────────────────────────────────────────────────────────

def build_cohort(outcomes: list[Outcome], min_observations: int = 10) -> pl.DataFrame:
    """
    Build a vitalsign cohort restricted to stays with known outcomes.

    Only stay_ids that appear in *every* requested outcome extractor are
    included — i.e. stays for which we can actually evaluate the outcome
    (e.g. a stay must have a linked hadm_id to be eligible for mortality
    or readmission).  Stays with no coverage in any extractor are dropped,
    so a False value means the outcome did not occur, not that it is unknown.

    Stays with fewer than min_observations vitalsign rows are also excluded,
    as entropy and irregularity metrics are not meaningful for short series.

    Parameters
    ----------
    outcomes:
        Any subset of "sepsis", "mortality", "readmission_30d".
    min_observations:
        Minimum number of vitalsign rows a stay must have to be included.
        Defaults to 10.

    Returns
    -------
    DataFrame with one row per (stay_id, charttime) restricted to eligible
    stays, with one boolean column per requested outcome.

    Examples
    --------
    >>> build_cohort(["mortality"])
    >>> build_cohort(["sepsis", "mortality", "readmission_30d"], min_observations=5)
    """
    if not outcomes:
        raise ValueError("Provide at least one outcome.")

    unknown = set(outcomes) - _OUTCOME_REGISTRY.keys()
    if unknown:
        raise ValueError(f"Unknown outcomes: {unknown}. Choose from {set(_OUTCOME_REGISTRY)}")

    # Collect outcome DataFrames; each covers only the stays it can evaluate
    outcome_dfs: dict[str, pl.DataFrame] = {}
    for outcome in outcomes:
        col_name, extractor = _OUTCOME_REGISTRY[outcome]
        outcome_dfs[col_name] = extractor()

    # Restrict to stays present in every outcome (intersection)
    eligible_stay_ids: set[int] = set(outcome_dfs[next(iter(outcome_dfs))]["stay_id"].to_list())
    for df in outcome_dfs.values():
        eligible_stay_ids &= set(df["stay_id"].to_list())

    # Filter vitalsigns to eligible stays with sufficient observations
    vitalsigns = _load_vitalsigns()
    sufficient = (
        vitalsigns
        .filter(pl.col("stay_id").is_in(list(eligible_stay_ids)))
        .filter(
            pl.col("stay_id").count().over("stay_id") >= min_observations
        )
    )
    cohort = sufficient
    for col_name, outcome_df in outcome_dfs.items():
        cohort = cohort.join(
            outcome_df.select("stay_id", col_name),
            on="stay_id",
            how="left",
        )

    return cohort


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cohort = build_cohort(["mortality"], min_observations=10)
    print(cohort.columns)
    print(cohort.shape)
    print(cohort.head())
    
    # group by mortality to get the prevalence
    print(cohort.group_by("in_hospital_mortality").len())
