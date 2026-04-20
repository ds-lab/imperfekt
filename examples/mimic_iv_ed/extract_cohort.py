import polars as pl
from functools import cache
from typing import Literal

# ── paths ────────────────────────────────────────────────────────────────────
ED_BASE = "/mnt/dataset/mimic-iv-ed/mimic-iv-ed-2.2/ed"
HOSP_BASE = "/workspaces/imperfekt/data/physionet.org/files/mimiciv/3.1/hosp"
ICU_BASE  = "/workspaces/imperfekt/data/physionet.org/files/mimiciv/3.1/icu"

VITALSIGN_PATH  = f"{ED_BASE}/vitalsign.csv"
DIAGNOSIS_PATH  = f"{ED_BASE}/diagnosis.csv"
EDSTAYS_PATH    = f"{ED_BASE}/edstays.csv"
ADMISSIONS_PATH = f"{HOSP_BASE}/admissions.csv.gz"
PATIENTS_PATH   = f"{HOSP_BASE}/patients.csv.gz"
ICUSTAYS_PATH   = f"{ICU_BASE}/icustays.csv.gz"

# ── ICD code definitions ──────────────────────────────────────────────────────
SEPSIS_ICD_PREFIXES = [
    # ICD-9
    "99591", "99592", "78552", "99590",
    # ICD-10
    "A40", "A41", "R65",
]

Outcome = Literal["sepsis", "mortality", "readmission_30d", "ed_stay_length", "icu_admission", "critical_outcome"]


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
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    return (
        pl.read_csv(EDSTAYS_PATH)
        .with_columns(
            intime=pl.col("intime").str.strptime(pl.Datetime, dt_fmt),
            outtime=pl.col("outtime").str.strptime(pl.Datetime, dt_fmt),
        )
    )


@cache
def _load_admissions() -> pl.DataFrame:
    return pl.read_csv(ADMISSIONS_PATH)


@cache
def _load_patients() -> pl.DataFrame:
    return pl.read_csv(PATIENTS_PATH).select("subject_id", "gender", "anchor_age", "anchor_year")


@cache
def _load_icustays() -> pl.DataFrame:
    return pl.read_csv(ICUSTAYS_PATH).select("hadm_id").unique()


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


def extract_mortality(within_hours: int | None = None) -> pl.DataFrame:
    """
    One row per stay_id with in_hospital_mortality flag.

    Join chain: stay_id → edstays (hadm_id, intime) → admissions (deathtime / hospital_expire_flag).

    Parameters
    ----------
    within_hours:
        If set, a death only counts as positive if deathtime <= ED intime +
        within_hours.  Uses admissions.deathtime (null for survivors).
        5 stays in MIMIC have deathtime < intime (data artefacts) — these are
        treated as negative.  If None, uses hospital_expire_flag (death at any
        point during the admission), which is the standard definition.
    """
    stay_hadm = _load_edstays().select("stay_id", "hadm_id", "intime").drop_nulls("hadm_id")
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    adm = _load_admissions().with_columns(
        deathtime=pl.col("deathtime").str.strptime(pl.Datetime, dt_fmt, strict=False),
    ).select("hadm_id", "hospital_expire_flag", "deathtime")

    joined = stay_hadm.join(adm, on="hadm_id", how="left").drop_nulls("hospital_expire_flag")

    if within_hours is None:
        return (
            joined
            .with_columns(
                pl.col("hospital_expire_flag").cast(pl.Boolean).alias("in_hospital_mortality")
            )
            .select("stay_id", "in_hospital_mortality")
        )

    return (
        joined
        .with_columns(
            in_hospital_mortality=(
                pl.col("deathtime").is_not_null()
                & (pl.col("deathtime") >= pl.col("intime"))
                & (pl.col("deathtime") <= pl.col("intime") + pl.duration(hours=within_hours))
            )
        )
        .select("stay_id", "in_hospital_mortality")
    )


def extract_ed_stay_length() -> pl.DataFrame:
    """One row per stay_id with ed_stay_length_hours (float)."""
    return (
        _load_edstays()
        .select("stay_id", "intime", "outtime")
        .drop_nulls()
        .with_columns(
            ed_stay_length_hours=(
                (pl.col("outtime") - pl.col("intime")).dt.total_minutes() / 60
            ).cast(pl.Float64)
        )
        .select("stay_id", "ed_stay_length_hours")
        .filter(pl.col("ed_stay_length_hours") >= 4.0)
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


def extract_icu_admission() -> pl.DataFrame:
    """
    One row per ED stay_id with icu_admission flag.

    Join chain: stay_id → edstays (hadm_id) → icustays (hadm_id).
    A stay is positive if at least one ICU stay shares the same hadm_id.
    Stays without a hadm_id (ED discharge without admission) are kept as False.
    """
    stay_hadm = _load_edstays().select("stay_id", "hadm_id")
    icu_hadm_ids = _load_icustays().with_columns(icu_admission=pl.lit(True))
    return (
        stay_hadm
        .join(icu_hadm_ids, on="hadm_id", how="left")
        .select("stay_id", "icu_admission")
        .unique("stay_id", keep="first")
        .with_columns(pl.col("icu_admission").fill_null(False))
    )


def extract_critical_outcome() -> pl.DataFrame:
    """
    One row per ED stay_id with critical_outcome flag.

    Positive = icu_admission OR in_hospital_mortality.
    Stays must have a known mortality flag (hadm_id required); stays without
    a hadm_id are dropped because their mortality status is unknowable.
    ICU admission is derived from icustays via hadm_id.
    """
    mortality_df = extract_mortality(within_hours=MORTALITY_WINDOW_HOURS).select("stay_id", "in_hospital_mortality")
    icu_df = extract_icu_admission().select("stay_id", "icu_admission")
    return (
        mortality_df
        .join(icu_df, on="stay_id", how="left")
        .with_columns(pl.col("icu_admission").fill_null(False))
        .with_columns(
            critical_outcome=(
                pl.col("in_hospital_mortality") | pl.col("icu_admission")
            )
        )
        .select("stay_id", "critical_outcome")
    )


# ── demographics ─────────────────────────────────────────────────────────────

def extract_demographics() -> pl.DataFrame:
    """
    One row per stay_id with age_at_visit (int) and sex (str, 'M'/'F').

    Age is computed as anchor_age + (ED intime year - anchor_year), which
    corrects for the MIMIC time-shift applied per patient.
    """
    stays = _load_edstays().select("stay_id", "subject_id", "intime")
    patients = _load_patients()
    return (
        stays
        .join(patients, on="subject_id", how="left")
        .with_columns(
            age_at_visit=(pl.col("anchor_age") + pl.col("intime").dt.year() - pl.col("anchor_year")).cast(pl.Int32),
            sex=pl.col("gender"),
        )
        .select("stay_id", "age_at_visit", "sex")
    )


# ── registry ──────────────────────────────────────────────────────────────────

MORTALITY_WINDOW_HOURS = 48

_OUTCOME_REGISTRY: dict[Outcome, tuple[str, callable]] = {
    "sepsis":               ("sepsis",               extract_sepsis),
    "in_hospital_mortality":("in_hospital_mortality", lambda: extract_mortality(within_hours=MORTALITY_WINDOW_HOURS)),
    "readmission_30d":      ("readmission_30d",       extract_readmission_30d),
    "ed_stay_length":       ("ed_stay_length_hours",  extract_ed_stay_length),
    "icu_admission":        ("icu_admission",         extract_icu_admission),
    "critical_outcome":     ("critical_outcome",      extract_critical_outcome),
}


# ── public API ────────────────────────────────────────────────────────────────

VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]


def build_cohort(
    outcomes: list[Outcome],
    min_observations: int | None = None,
    window_hours: int | None = None,
    max_missingness: float = 0.5,
) -> pl.DataFrame:
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
        Any subset of "sepsis", "in_hospital_mortality", "readmission_30d", "ed_stay_length".
    min_observations:
        Minimum number of vitalsign rows a stay must have to be included.
        Defaults to None.
    window_hours:
        If set, truncate each stay to the first window_hours hours after ED
        arrival (intime). Observations with charttime > intime + window_hours
        are dropped. min_observations is re-applied after truncation so stays
        that become too short are excluded.
    max_missingness:
        Maximum fraction of rows per stay that may have at least one null vital
        sign (row-level missingness). Stays exceeding this threshold are dropped.
        Applied after window truncation and min_observations filter.
        Defaults to 0.5.

    Returns
    -------
    DataFrame with one row per (stay_id, charttime) restricted to eligible
    stays, with one boolean column per requested outcome.

    Examples
    --------
    >>> build_cohort(["in_hospital_mortality"])
    >>> build_cohort(["in_hospital_mortality"], window_hours=4, max_missingness=0.5)
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

    # Filter vitalsigns to eligible stays, optionally truncating to a time window
    vitalsigns = (
        _load_vitalsigns()
        .filter(pl.col("stay_id").is_in(list(eligible_stay_ids)))
    )
    if window_hours is not None:
        stay_intimes = _load_edstays().select("stay_id", "intime")
        vitalsigns = (
            vitalsigns
            .join(stay_intimes, on="stay_id", how="left")
            .filter(pl.col("charttime") <= pl.col("intime") + pl.duration(hours=window_hours))
            .drop("intime")
        )
        
    if min_observations is not None:
        sufficient = vitalsigns.filter(
            pl.col("stay_id").count().over("stay_id") >= min_observations
        )
    else:
        sufficient = vitalsigns

    # Row is missing if any of the core vital sign columns is null.
    # Stays where more than max_missingness fraction of rows are missing are dropped.
    any_null = pl.any_horizontal([pl.col(c).is_null() for c in VITAL_COLS])
    sufficient = sufficient.filter(
        (any_null.cast(pl.Float64).sum().over("stay_id") / pl.col("stay_id").count().over("stay_id"))
        <= max_missingness
    )

    cohort = sufficient
    for col_name, outcome_df in outcome_dfs.items():
        cohort = cohort.join(
            outcome_df.select("stay_id", col_name),
            on="stay_id",
            how="left",
        )

    demographics = extract_demographics().select("stay_id", "age_at_visit", "sex")
    cohort = cohort.join(demographics, on="stay_id", how="left")

    return cohort


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cohort = build_cohort(["in_hospital_mortality", "ed_stay_length"], min_observations=10)
    print(cohort.columns)
    print(cohort.shape)
    print(cohort.head())
    
    # group by mortality to get the prevalence
    print(cohort.group_by("in_hospital_mortality").len())
    print(cohort.describe())
