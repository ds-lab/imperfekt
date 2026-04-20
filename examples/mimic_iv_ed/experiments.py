"""
MIMIC-IV-ED irregularity experiment
====================================
Compares XGBoost pipelines for different prediction tasks (e.g. 30-day readmission, in-hospital mortality):

  Pipeline A – regular 30-min resampled grid, statistical aggregates only
  Pipeline B – raw irregular intervals, statistical + imperfekt irregularity aggregates
    Pipeline C – raw imperfekt irregularity features, then 30-min resample + fill,
                             then statistical + imperfekt irregularity aggregates
    Pipeline D – Pipeline 0 + observation-count feature (timestamps per stay)

Performance is estimated with repeated stratified k-fold cross-validation
(5 folds × 10 repeats = 50 fits per pipeline).  The test set is stratified
by per-stay composite irregularity score computed once on the full dataset
(strata are a descriptor of the data, not a target — no leakage).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from examples.utils.models import XGBoostModel  # noqa: E402
from examples.mimic_iv_ed.extract_cohort import build_cohort  # noqa: E402
from imperfekt.features.irregularity import (  # noqa: E402
    add_interval_features,
    add_windowed_acceleration,
)
from imperfekt.analysis.irregularity.irregularity import Irregularity  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "mimic_iv_ed_results"
RANDOM_STATE = 42
WINDOW_HOURS = 5
MIN_OBS = 5
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
    "rolling_mean_acceleration_5",
    "rolling_abs_acceleration_5",
    "rolling_std_acceleration_5",
]

SPEARMAN_TOP_K_PHYS = 5
SPEARMAN_TOP_K_STRUCT = 5



# ── data loading ──────────────────────────────────────────────────────────────

def load_cohort() -> pl.DataFrame:
    return build_cohort(
        ["critical_outcome", "ed_stay_length"],
        min_observations=MIN_OBS,
        window_hours=WINDOW_HOURS,
        max_missingness=MAX_MISSINGNESS,
    )


# ── feature engineering ───────────────────────────────────────────────────────

def _vital_agg_exprs() -> list:
    return (
        [pl.col(c).mean().alias(f"{c}_mean") for c in VITAL_COLS]
        + [pl.col(c).median().alias(f"{c}_median") for c in VITAL_COLS]
        + [pl.col(c).min().alias(f"{c}_min") for c in VITAL_COLS]
        + [pl.col(c).max().alias(f"{c}_max") for c in VITAL_COLS]
        + [pl.col(c).std().alias(f"{c}_std") for c in VITAL_COLS]
        + [pl.col(c).first().alias(f"{c}_first") for c in VITAL_COLS]
        + [pl.col(c).last().alias(f"{c}_last") for c in VITAL_COLS]
    )


def _ireg_agg_exprs() -> list:
    return (
        [pl.col(ic).mean().alias(f"{ic}_mean") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).median().alias(f"{ic}_median") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).min().alias(f"{ic}_min") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).max().alias(f"{ic}_max") for ic in IREG_FEATURE_COLS]
        + [pl.col(ic).std().alias(f"{ic}_std") for ic in IREG_FEATURE_COLS]
    )


def pipeline_0_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Raw irregular timestamps, statistical aggregates of vital signs only.
    No resampling, no imputation, no imperfekt features.
    Baseline that isolates the contribution of temporal structure.
    """
    return df.sort(["stay_id", "charttime"]).group_by("stay_id").agg(_vital_agg_exprs())


def pipeline_d_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pipeline 0 plus observation-count feature: raw irregular timestamps,
    vital-sign statistical aggregates, and number of timestamped
    observations (rows) per stay.
    """
    return (
        df.sort(["stay_id", "charttime"])
        .group_by("stay_id")
        .agg(_vital_agg_exprs() + [pl.col("charttime").count().alias("n_observations")])
    )


def pipeline_a_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Resample to a regular 30-min grid per stay (forward-fill then backward-fill
    within stay only), then compute per-stay statistical aggregates.
    """
    filled = (
        df.sort(["stay_id", "charttime"])
        .upsample(time_column="charttime", every="30m", group_by="stay_id", maintain_order=True)
        .with_columns([pl.col(c).forward_fill().over("stay_id") for c in VITAL_COLS])
        .with_columns([pl.col(c).backward_fill().over("stay_id") for c in VITAL_COLS])
    )
    return filled.group_by("stay_id").agg(_vital_agg_exprs())


def pipeline_b_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep raw irregular timestamps.  Add imperfekt interval and acceleration
    features (global per stay — all charttimes treated as a single event
    sequence), then compute per-stay aggregates for both vital signs and
    irregularity features.
    """
    df_ireg = (
        df.sort(["stay_id", "charttime"])
        .pipe(add_interval_features, id_col="stay_id", clock_col="charttime")
        .pipe(add_windowed_acceleration, id_col="stay_id", clock_col="charttime")
    )
    return df_ireg.group_by("stay_id").agg(_vital_agg_exprs() + _ireg_agg_exprs())


def pipeline_c_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute imperfekt features on raw irregular timestamps first, then
    resample to a regular 30-min grid and fill missing values within stay.
    Finally, compute per-stay aggregates for vital and irregularity features.
    """
    df_ireg = (
        df.sort(["stay_id", "charttime"])
        .pipe(add_interval_features, id_col="stay_id", clock_col="charttime")
        .pipe(add_windowed_acceleration, id_col="stay_id", clock_col="charttime")
    )
    fill_cols = VITAL_COLS + IREG_FEATURE_COLS
    filled = (
        df_ireg
        .upsample(time_column="charttime", every="30m", group_by="stay_id", maintain_order=True)
        .with_columns([pl.col(c).forward_fill().over("stay_id") for c in fill_cols])
        .with_columns([pl.col(c).backward_fill().over("stay_id") for c in fill_cols])
    )
    return filled.group_by("stay_id").agg(_vital_agg_exprs() + _ireg_agg_exprs())


def build_stay_level(ts_df: pl.DataFrame, feature_fn) -> pl.DataFrame:
    """Build stay-level feature frame with outcome label attached."""
    features = feature_fn(ts_df)
    stay_meta = (
        ts_df.select(["stay_id", OUTCOME_COL, "age_at_visit", "sex"])
        .unique("stay_id", keep="first")
        .with_columns(sex_female=pl.col("sex").eq("F").cast(pl.Int8))
        .drop("sex")
    )
    return (
        features
        .join(stay_meta, on="stay_id", how="left")
        .drop_nulls(OUTCOME_COL)
    )


# ── irregularity stratification ───────────────────────────────────────────────

def compute_irregularity_strata(ts_df: pl.DataFrame) -> tuple[pl.DataFrame, "Irregularity"]:
    """
    Compute per-stay composite irregularity score and stratum on the full
    dataset. Strata are derived once so boundaries are consistent across
    all CV folds — they describe data heterogeneity, not the outcome.
    Returns (strata DataFrame with stay_id/irregularity_stratum,
             the fitted Irregularity object).
    """
    strata_dir = RESULTS_DIR / "irregularity_strata"
    ireg = Irregularity(
        ts_df.select(["stay_id", "charttime"]),
        id_col="stay_id",
        clock_col="charttime",
        save_path=strata_dir,
    )
    ireg.run(save_results=True, n_strata_quantiles=8)
    strata = ireg.results.cs_case_scores.select(
        ["stay_id", "irregularity_stratum"]
    )
    return strata, ireg


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict | None:
    if len(y_true) < 2 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    prevalence = y_true.mean()
    brier = brier_score_loss(y_true, y_proba)
    brier_ref = prevalence * (1 - prevalence)
    bss = 1.0 - brier / brier_ref if brier_ref > 0 else float("nan")
    auprc = average_precision_score(y_true, y_proba)
    return {
        "auprc": auprc,
        "auprc_lift": auprc / prevalence,
        "auroc": roc_auc_score(y_true, y_proba),
        "brier": brier,
        "brier_skill_score": bss,
        "n_pos": int(y_true.sum()),
        "n_pos_pct": prevalence,
        "total": len(y_true),
    }


# ── cross-validation ──────────────────────────────────────────────────────────

def run_cv(
    stay_df: pl.DataFrame,
    strata: pl.DataFrame,
    pipeline_name: str,
) -> tuple[
    dict[str, list],
    XGBoostModel | None,
    np.ndarray | None,
    pl.DataFrame | None,
    list[str],
]:
    """
    Repeated stratified k-fold CV (CV_N_SPLITS x CV_N_REPEATS).
    Splits are on subject_id to prevent patient-level leakage across folds.
    Strata assignment is fixed (computed on the full dataset before CV).

    Returns:
      fold_metrics  - dict mapping "overall" and each stratum label to a list
                      of metric dicts, one per fold
      last_model    - trained model from the final fold (for SHAP)
      last_X_test   - test features from the final fold
      last_test_df  - exact test DataFrame from the final fold
      feature_cols  - ordered list of feature column names
    """
    feature_cols = [
        c for c in stay_df.columns if c not in ("stay_id", OUTCOME_COL, "subject_id")
    ]

    subject_labels = (
        stay_df.select(["subject_id", OUTCOME_COL])
        .group_by("subject_id")
        .agg(pl.col(OUTCOME_COL).any().alias("any_outcome"))
        .sort("subject_id")
    )

    subjects = subject_labels["subject_id"].to_numpy()
    subject_outcomes = subject_labels["any_outcome"].cast(pl.Int8).to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    fold_metrics: dict[str, list] = defaultdict(list)
    last_model = None
    last_X_test = None
    last_test_df = None

    total_folds = CV_N_SPLITS * CV_N_REPEATS
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(subjects, subject_outcomes)):
        print(f"  [{pipeline_name}] fold {fold_idx + 1}/{total_folds}", end="\r")

        train_subjects = set(subjects[train_idx].tolist())
        test_subjects = set(subjects[test_idx].tolist())

        train = stay_df.filter(pl.col("subject_id").is_in(train_subjects))
        test = stay_df.filter(pl.col("subject_id").is_in(test_subjects))

        X_train = train.select(feature_cols).to_numpy().astype(np.float32)
        y_train = train[OUTCOME_COL].cast(pl.Int8).to_numpy()
        X_test = test.select(feature_cols).to_numpy().astype(np.float32)
        y_test = test[OUTCOME_COL].cast(pl.Int8).to_numpy()

        model = XGBoostModel(feature_mode=pipeline_name, random_state=RANDOM_STATE + fold_idx)
        model._train_model(X_train, y_train)
        _, y_proba = model._predict(X_test)

        m = _compute_metrics(y_test, y_proba)
        if m:
            fold_metrics["overall"].append(m)

        test_stay_ids = test["stay_id"]
        for (stratum_label,), grp in strata.group_by("irregularity_stratum"):
            mask = test_stay_ids.is_in(grp["stay_id"].to_list()).to_numpy()
            m_s = _compute_metrics(y_test[mask], y_proba[mask])
            if m_s:
                fold_metrics[stratum_label].append(m_s)

        last_model = model
        last_test_df = test
        last_X_test = X_test

    print()
    return dict(fold_metrics), last_model, last_X_test, last_test_df, feature_cols


def summarise_cv(fold_metrics: dict[str, list], pipeline_name: str) -> dict[str, dict]:
    """
    Aggregate per-fold metric lists into mean ± 95% CI (t-distribution).
    Returns a dict keyed by stratum label (plus "overall").
    """
    from scipy import stats

    summary = {}
    for key, folds in fold_metrics.items():
        for metric in ("auprc", "auprc_lift", "auroc", "brier_skill_score"):
            vals = np.array([f[metric] for f in folds if not np.isnan(f[metric])])
            if len(vals) == 0:
                continue
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ci = stats.t.ppf(0.975, df=len(vals) - 1) * se
            summary.setdefault(key, {})[metric] = {"mean": mean, "ci": ci}

    o = summary.get("overall", {})
    print(f"\n[{pipeline_name}] overall  " + "  ".join(
        f"{m.upper()}={v['mean']:.3f}±{v['ci']:.3f}"
        for m, v in o.items()
    ))
    for s in sorted(k for k in summary if k != "overall"):
        vals = summary[s]
        print(f"  {s}  " + "  ".join(
            f"{m.upper()}={v['mean']:.3f}±{v['ci']:.3f}"
            for m, v in vals.items()
        ))
    return summary


# ── plotting ──────────────────────────────────────────────────────────────────

def _stratum_prevalence_labels(strata: pl.DataFrame, stay_df: pl.DataFrame) -> dict[str, str]:
    """Return {stratum: "Q1 (46%)"} using outcome prevalence per stratum."""
    outcome = stay_df.select(["stay_id", OUTCOME_COL]).unique("stay_id", keep="first")
    joined = strata.join(outcome, on="stay_id", how="left").drop_nulls("irregularity_stratum")
    summary = (
        joined.group_by("irregularity_stratum")
        .agg(pl.col(OUTCOME_COL).mean().alias("prevalence"))
        .sort("irregularity_stratum")
    )
    return {
        row["irregularity_stratum"]: f"{row['irregularity_stratum']} ({row['prevalence']:.0%})"
        for row in summary.iter_rows(named=True)
    }


def plot_auprc_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    stratum_labels: dict[str, str] | None = None,
) -> None:
    """
    Line plot: mean AUPRC per irregularity stratum (Q1→Qn, sorted by
    increasing irregularity) for all provided pipelines.
    95% CI shown as dashed lines above and below the mean.
    Overall AUPRC shown as horizontal dashed reference lines.
    """
    strata_keys = sorted({
        k
        for _, summary in pipeline_summaries
        for k in summary
        if k != "overall"
    })

    def _vals(summary):
        means, lo, hi = [], [], []
        for k in strata_keys:
            v = summary.get(k, {}).get("auprc")
            if v:
                means.append(v["mean"])
                lo.append(v["mean"] - v["ci"])
                hi.append(v["mean"] + v["ci"])
            else:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
        return np.array(means), np.array(lo), np.array(hi)

    x = np.arange(len(strata_keys))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)

        overall = summary.get("overall", {}).get("auprc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([stratum_labels.get(k, k) if stratum_labels else k for k in strata_keys])
    ax.set_xlabel("Irregularity stratum — prevalence of outcome (Q1 = most regular)")
    ax.set_ylabel("AUPRC (mean ± 95% CI)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"AUPRC by irregularity stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_auprc_lift_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    stratum_labels: dict[str, str] | None = None,
) -> None:
    """
    Line plot: mean AUPRC lift (AUPRC / prevalence) per irregularity stratum.
    Lift > 1 means the model beats the no-skill baseline within that stratum.
    Reference line at lift = 1 (no-skill).
    """
    strata_keys = sorted({
        k
        for _, summary in pipeline_summaries
        for k in summary
        if k != "overall"
    })

    def _vals(summary):
        means, lo, hi = [], [], []
        for k in strata_keys:
            v = summary.get(k, {}).get("auprc_lift")
            if v:
                means.append(v["mean"])
                lo.append(v["mean"] - v["ci"])
                hi.append(v["mean"] + v["ci"])
            else:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
        return np.array(means), np.array(lo), np.array(hi)

    x = np.arange(len(strata_keys))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="No-skill baseline")

    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels([stratum_labels.get(k, k) if stratum_labels else k for k in strata_keys])
    ax.set_xlabel("Irregularity stratum — prevalence of outcome (Q1 = most regular)")
    ax.set_ylabel("AUPRC lift = AUPRC / prevalence (mean ± 95% CI)")
    ax.legend()
    ax.set_title(f"AUPRC lift by irregularity stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def _split_feature_groups(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Split aggregated feature names into physiological and structural groups.

    Structural features are the aggregated outputs derived from imperfekt
    interval/acceleration primitives. Physiological features are everything else.
    """
    struct_cols = [
        c
        for c in feature_cols
        if any(c == base or c.startswith(f"{base}_") for base in IREG_FEATURE_COLS)
    ]
    phys_cols = [c for c in feature_cols if c not in struct_cols]
    return phys_cols, struct_cols


def _compute_rfi_by_stratum(
    abs_shap: np.ndarray,
    stay_ids: np.ndarray,
    strata: pl.DataFrame,
    phys_idx: np.ndarray,
    struct_idx: np.ndarray,
) -> pl.DataFrame:
    """
    Relative Feature Importance (RFI) per stratum using summed absolute SHAP.

    For each stratum s:
      phys_sum  = sum over cases and physiology features of |SHAP|
      struct_sum = sum over cases and metadata features of |SHAP|
      phys_rfi  = phys_sum  / (phys_sum + struct_sum) * 100
      struct_rfi = struct_sum / (phys_sum + struct_sum) * 100

    Using the sum preserves the total attribution mass so that
    a metadata group with 5 features holding 30% of the sum is directly
    comparable to 50 physiology features — each metadata feature carries on
    average 10x the weight of a physiology feature.

    Also records per-feature-average importance for each group so callers can
    compute the per-feature multiplier (struct_per_feat / phys_per_feat).

    Returns DataFrame with columns:
      irregularity_stratum, phys_rfi, struct_rfi,
      phys_sum, struct_sum, n_phys, n_struct, per_feat_ratio
    """
    strata_map = {sid: s for sid, s in zip(
        strata["stay_id"].to_list(), strata["irregularity_stratum"].to_list()
    )}
    n_phys = len(phys_idx)
    n_struct = len(struct_idx)
    rows = []
    for stratum in sorted(set(strata["irregularity_stratum"].to_list())):
        mask = np.array([strata_map.get(sid) == stratum for sid in stay_ids])
        if mask.sum() == 0:
            continue
        # sum over all cases AND all features in each group
        phys_sum = float(abs_shap[mask][:, phys_idx].sum())
        struct_sum = float(abs_shap[mask][:, struct_idx].sum())
        total = phys_sum + struct_sum
        if total == 0:
            continue
        phys_rfi = phys_sum / total * 100.0
        struct_rfi = struct_sum / total * 100.0
        # average importance per individual feature — reveals per-feature multiplier
        phys_per_feat = phys_sum / n_phys if n_phys > 0 else 0.0
        struct_per_feat = struct_sum / n_struct if n_struct > 0 else 0.0
        per_feat_ratio = struct_per_feat / phys_per_feat if phys_per_feat > 0 else float("nan")
        rows.append({
            "irregularity_stratum": stratum,
            "phys_rfi": phys_rfi,
            "struct_rfi": struct_rfi,
            "phys_sum": phys_sum,
            "struct_sum": struct_sum,
            "n_phys": n_phys,
            "n_struct": n_struct,
            "per_feat_ratio": per_feat_ratio,
        })
    return pl.DataFrame(rows)


def _plot_group_importance_by_stratum(
    abs_shap: np.ndarray,
    stay_ids: np.ndarray,
    strata: pl.DataFrame,
    phys_idx: np.ndarray,
    struct_idx: np.ndarray,
    save_path: Path,
    pipeline_name: str,
    stratum_labels: dict[str, str] | None = None,
) -> None:
    rfi_df = _compute_rfi_by_stratum(abs_shap, stay_ids, strata, phys_idx, struct_idx)
    if rfi_df.height == 0:
        print(f"[{pipeline_name}] Skipping SHAP group plot: no overlapping stay_id rows.")
        return

    labels = rfi_df["irregularity_stratum"].to_list()
    tick_labels = [stratum_labels.get(l, l) if stratum_labels else l for l in labels]
    phys_pct = rfi_df["phys_rfi"].to_numpy()
    struct_pct = rfi_df["struct_rfi"].to_numpy()
    per_feat_ratio = rfi_df["per_feat_ratio"].to_numpy()
    n_phys = int(rfi_df["n_phys"][0])
    n_struct = int(rfi_df["n_struct"][0])

    x = np.arange(len(labels))
    fig, (ax_stack, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: stacked bar — summed SHAP mass share
    ax_stack.bar(x, phys_pct, color="#4C72B0", label=f"Physiological ({n_phys} features)")
    ax_stack.bar(x, struct_pct, bottom=phys_pct, color="#DD8452", label=f"imperfekt metadata ({n_struct} features)")
    for i, (p, s) in enumerate(zip(phys_pct, struct_pct)):
        ax_stack.text(i, p + s / 2, f"{s:.1f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax_stack.set_ylim(0, 100)
    ax_stack.set_yticks(np.arange(0, 101, 20))
    ax_stack.set_ylabel("Share of total |SHAP| mass (%)")
    ax_stack.set_xlabel("Irregularity stratum")
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(tick_labels)
    ax_stack.set_title("Attention shift (summed |SHAP| by group)")
    ax_stack.legend(loc="upper right", fontsize=8)

    # right: per-feature multiplier — how many times more influential is one metadata
    # feature vs one physiology feature within that stratum
    bar_colors = ["#c84b31" if r > 1 else "#4C72B0" for r in per_feat_ratio]
    ax_ratio.bar(x, per_feat_ratio, color=bar_colors)
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Equal influence (×1)")
    for i, r in enumerate(per_feat_ratio):
        if not np.isnan(r):
            ax_ratio.text(i, r + 0.05, f"×{r:.1f}", ha="center", va="bottom", fontsize=8)
    ax_ratio.set_ylabel("Per-feature influence multiplier\n(avg metadata SHAP / avg physiology SHAP)")
    ax_ratio.set_xlabel("Irregularity stratum")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(tick_labels)
    ax_ratio.set_title("Per-feature influence ratio (metadata vs physiology)")
    ax_ratio.legend(fontsize=8)

    fig.suptitle(f"{pipeline_name}: Structural floor analysis", fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"[{pipeline_name}] SHAP structural-floor plot saved to {save_path}")


def _spearman_structural_floor(
    abs_shap: np.ndarray,
    raw_features: np.ndarray,
    feature_cols: list[str],
    phys_cols: list[str],
    struct_cols: list[str],
    pipeline_name: str,
) -> pl.DataFrame | None:
    """
    Pairwise Spearman rho between top metadata and physiology raw features.

    Feature ranking is defined by mean |SHAP| over samples.
    We select the top-k features from each group by this ranking and compute
    pairwise Spearman correlations on aligned raw feature values only.

    Returns:
        A Polars DataFrame sorted by absolute rho (descending), or None if
        one of the groups is empty.
    """
    from scipy.stats import spearmanr

    if len(phys_cols) == 0 or len(struct_cols) == 0:
        print(
            f"[{pipeline_name}] Skipping Spearman structural-floor test: "
            f"phys={len(phys_cols)}, struct={len(struct_cols)}."
        )
        return None

    if raw_features.shape[0] != abs_shap.shape[0]:
        raise ValueError(
            f"[{pipeline_name}] Spearman alignment error: raw_features rows "
            f"({raw_features.shape[0]}) != abs_shap rows ({abs_shap.shape[0]})."
        )

    mean_abs = abs_shap.mean(axis=0)
    mean_raw = np.nanmean(np.where(np.isfinite(raw_features), raw_features, np.nan), axis=0)

    phys_idx_map = {c: feature_cols.index(c) for c in phys_cols}
    struct_idx_map = {c: feature_cols.index(c) for c in struct_cols}

    top_k_phys = min(SPEARMAN_TOP_K_PHYS, len(phys_idx_map))
    top_k_struct = min(SPEARMAN_TOP_K_STRUCT, len(struct_idx_map))

    top_phys = sorted(phys_idx_map, key=lambda c: mean_abs[phys_idx_map[c]], reverse=True)[:top_k_phys]
    top_struct = sorted(struct_idx_map, key=lambda c: mean_abs[struct_idx_map[c]], reverse=True)[:top_k_struct]

    rows: list[dict] = []
    for struct_feat in top_struct:
        struct_vals = raw_features[:, struct_idx_map[struct_feat]]
        for phys_feat in top_phys:
            phys_vals = raw_features[:, phys_idx_map[phys_feat]]

            valid_mask = np.isfinite(phys_vals) & np.isfinite(struct_vals)
            valid_n = int(valid_mask.sum())
            if valid_n < 3:
                rho, pval = float("nan"), float("nan")
            else:
                phys_valid = phys_vals[valid_mask]
                struct_valid = struct_vals[valid_mask]
                if np.nanstd(phys_valid) == 0 or np.nanstd(struct_valid) == 0:
                    rho, pval = float("nan"), float("nan")
                else:
                    rho, pval = spearmanr(phys_valid, struct_valid)

            if np.isnan(rho) or np.isnan(pval):
                significance = "undefined"
            elif pval < 0.001:
                significance = "*** (p<0.001)"
            elif pval < 0.01:
                significance = "** (p<0.01)"
            elif pval < 0.05:
                significance = "* (p<0.05)"
            else:
                significance = "ns"

            rows.append(
                {
                    "metadata_feature": struct_feat,
                    "physiology_feature": phys_feat,
                    "metadata_mean_abs_shap": float(mean_abs[struct_idx_map[struct_feat]]),
                    "physiology_mean_abs_shap": float(mean_abs[phys_idx_map[phys_feat]]),
                    "metadata_mean_raw": float(mean_raw[struct_idx_map[struct_feat]]),
                    "physiology_mean_raw": float(mean_raw[phys_idx_map[phys_feat]]),
                    "rho": float(rho),
                    "abs_rho": float(abs(rho)) if not np.isnan(rho) else float("nan"),
                    "p_value": float(pval),
                    "significance": significance,
                    "valid_n": valid_n,
                }
            )

    if len(rows) == 0:
        return None

    spearman_df = pl.DataFrame(rows).sort(["abs_rho", "p_value"], descending=[True, False])

    abs_rhos = np.array(spearman_df["abs_rho"].to_list(), dtype=float)
    pvals = np.array(spearman_df["p_value"].to_list(), dtype=float)
    mean_abs_rho = float(np.nanmean(abs_rhos)) if len(abs_rhos) > 0 else float("nan")
    sig_pairs = int(np.sum((~np.isnan(pvals)) & (pvals < 0.05)))

    top_phys_desc = ", ".join(
        f"{feat} ({mean_abs[phys_idx_map[feat]]:.4f})" for feat in top_phys
    )
    top_struct_desc = ", ".join(
        f"{feat} ({mean_abs[struct_idx_map[feat]]:.4f})" for feat in top_struct
    )

    print(
        f"[{pipeline_name}] Spearman pairwise structural-floor test: "
        f"{top_k_struct} metadata × {top_k_phys} physiology = {spearman_df.height} pairs\n"
        f"  top physiology features (mean |SHAP| for ordering): {top_phys_desc}\n"
        f"  top metadata features (mean |SHAP| for ordering): {top_struct_desc}\n"
        f"  mean |rho| across raw-feature pairs = {mean_abs_rho:.3f}; "
        f"significant pairs (p<0.05) = {sig_pairs}/{spearman_df.height}"
    )
    print(spearman_df)

    return spearman_df


def _plot_spearman_heatmap(
    spearman_df: pl.DataFrame,
    pipeline_name: str,
    save_path: Path,
) -> None:
    """Plot rho heatmap for metadata vs physiology feature pairs."""
    if spearman_df.height == 0:
        print(f"[{pipeline_name}] Skipping Spearman heatmap: empty pair table.")
        return

    meta_order = (
        spearman_df
        .select(["metadata_feature", "metadata_mean_abs_shap"])
        .unique(subset=["metadata_feature"], keep="first")
        .sort("metadata_mean_abs_shap", descending=True)["metadata_feature"]
        .to_list()
    )
    phys_order = (
        spearman_df
        .select(["physiology_feature", "physiology_mean_abs_shap"])
        .unique(subset=["physiology_feature"], keep="first")
        .sort("physiology_mean_abs_shap", descending=True)["physiology_feature"]
        .to_list()
    )

    rho_lookup = {
        (row["metadata_feature"], row["physiology_feature"]): row["rho"]
        for row in spearman_df.iter_rows(named=True)
    }
    sig_lookup = {
        (row["metadata_feature"], row["physiology_feature"]): row["significance"]
        for row in spearman_df.iter_rows(named=True)
    }

    rho_mat = np.full((len(meta_order), len(phys_order)), np.nan, dtype=float)
    for i, meta in enumerate(meta_order):
        for j, phys in enumerate(phys_order):
            rho = rho_lookup.get((meta, phys), float("nan"))
            rho_mat[i, j] = rho

    fig_w = max(8.0, 1.2 * len(phys_order) + 3.0)
    fig_h = max(5.0, 0.8 * len(meta_order) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(rho_mat, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")

    ax.set_xticks(np.arange(len(phys_order)))
    ax.set_xticklabels(phys_order, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(meta_order)))
    ax.set_yticklabels(meta_order)
    ax.set_xlabel("Physiology features (SHAP-ranked)")
    ax.set_ylabel("Metadata features (SHAP-ranked)")
    ax.set_title(f"{pipeline_name}: Spearman heatmap (raw-feature pairs)")

    for i, meta in enumerate(meta_order):
        for j, phys in enumerate(phys_order):
            rho = rho_mat[i, j]
            sig = sig_lookup.get((meta, phys), "")
            if np.isnan(rho):
                text = "NA"
            else:
                star = "" if sig in ("", "ns", "undefined") else "*"
                text = f"{rho:.2f}{star}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"[{pipeline_name}] Spearman heatmap saved to {save_path}")


def run_shap_group_analysis(
    model: XGBoostModel,
    X_test: np.ndarray,
    test_stay_df: pl.DataFrame,
    strata: pl.DataFrame,
    feature_cols: list[str],
    pipeline_name: str,
    stratum_labels: dict[str, str] | None = None,
) -> None:
    if X_test.shape[0] == 0 or test_stay_df.height == 0:
        print(f"[{pipeline_name}] Skipping SHAP: empty final-fold test data.")
        return

    n_rows = min(X_test.shape[0], test_stay_df.height)
    if X_test.shape[0] != test_stay_df.height:
        print(
            f"[{pipeline_name}] SHAP alignment warning: X_test has {X_test.shape[0]} rows, "
            f"test metadata has {test_stay_df.height} rows. Using first {n_rows} rows."
        )

    X_aligned = X_test[:n_rows]
    test_aligned = test_stay_df.head(n_rows).select(["stay_id"])

    explanation = model.compute_shap(
        X=X_aligned,
        feature_names=feature_cols,
        save_dir=None,
        prefix=f"{pipeline_name}_last_fold",
    )
    if explanation is None:
        print(f"[{pipeline_name}] Skipping SHAP: explanation backend unavailable.")
        return

    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    n_aligned = min(shap_values.shape[0], test_aligned.height)
    if shap_values.shape[0] != test_aligned.height:
        print(
            f"[{pipeline_name}] SHAP value alignment warning: explanation has {shap_values.shape[0]} rows, "
            f"metadata has {test_aligned.height} rows. Using first {n_aligned} rows."
        )

    abs_shap = np.abs(shap_values[:n_aligned])
    stay_ids = test_aligned["stay_id"].to_numpy()[:n_aligned]

    phys_cols, struct_cols = _split_feature_groups(feature_cols)
    if len(phys_cols) == 0 or len(struct_cols) == 0:
        print(
            f"[{pipeline_name}] Skipping SHAP group analysis: "
            f"phys={len(phys_cols)}, struct={len(struct_cols)} feature groups."
        )
        return

    phys_idx = np.array([feature_cols.index(c) for c in phys_cols], dtype=int)
    struct_idx = np.array([feature_cols.index(c) for c in struct_cols], dtype=int)

    _plot_group_importance_by_stratum(
        abs_shap=abs_shap,
        stay_ids=stay_ids,
        strata=strata,
        phys_idx=phys_idx,
        struct_idx=struct_idx,
        save_path=RESULTS_DIR / "figures" / f"shap_group_importance_{pipeline_name}.svg",
        pipeline_name=pipeline_name,
        stratum_labels=stratum_labels,
    )

    spearman_df = _spearman_structural_floor(
        abs_shap=abs_shap,
        raw_features=X_aligned[:n_aligned],
        feature_cols=feature_cols,
        phys_cols=phys_cols,
        struct_cols=struct_cols,
        pipeline_name=pipeline_name,
    )

    if spearman_df is not None and spearman_df.height > 0:
        _plot_spearman_heatmap(
            spearman_df=spearman_df,
            pipeline_name=pipeline_name,
            save_path=RESULTS_DIR / "figures" / f"spearman_heatmap_{pipeline_name}.svg",
        )

        spearman_path = RESULTS_DIR / "figures" / f"spearman_pairwise_{pipeline_name}.csv"
        spearman_path.parent.mkdir(parents=True, exist_ok=True)
        spearman_df.write_csv(spearman_path)
        print(f"[{pipeline_name}] Spearman pairwise table saved to {spearman_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading cohort (first {WINDOW_HOURS} h of ED stay, ≥{MIN_OBS} observations)…")
    ts_df = load_cohort()
    print(f"Cohort: {ts_df['stay_id'].n_unique()} stays, {len(ts_df)} observations")
    print(f"Outcome prevalence: {ts_df[OUTCOME_COL].mean():.3f} ({ts_df[OUTCOME_COL].sum()}/{len(ts_df)})")

    print("\nComputing irregularity strata on full dataset (fixed across all folds)…")
    strata, ireg = compute_irregularity_strata(ts_df)
    stay_outcomes = ts_df.select(["stay_id", OUTCOME_COL]).unique("stay_id", keep="first")
    ireg_scores = ireg.results.cs_case_scores.select(
        ["stay_id", "irregularity_stratum", "cv", "burstiness_coeff", "normalized_entropy"]
    )
    prevalence_by_stratum = (
        ireg_scores.join(stay_outcomes, on="stay_id", how="left")
        .drop_nulls("irregularity_stratum")
        .group_by("irregularity_stratum")
        .agg(
            pl.col(OUTCOME_COL).mean().alias("prevalence"),
            pl.len().alias("count"),
            pl.col("cv").mean().alias("mean_cv"),
            pl.col("burstiness_coeff").mean().alias("mean_burstiness"),
            pl.col("normalized_entropy").mean().alias("mean_entropy"),
        )
        .sort("irregularity_stratum")
    )
    print(prevalence_by_stratum)
    stratum_labels = _stratum_prevalence_labels(strata, ts_df)

    print("\nBuilding stay-level feature frames…")
    stay_0 = build_stay_level(ts_df, pipeline_0_features)
    stay_d = build_stay_level(ts_df, pipeline_d_features)
    stay_a = build_stay_level(ts_df, pipeline_a_features)
    stay_b = build_stay_level(ts_df, pipeline_b_features)
    stay_c = build_stay_level(ts_df, pipeline_c_features)

    # subject_id is needed inside run_cv for group-level splitting; carry it over
    subj_map = ts_df.select(["stay_id", "subject_id"]).unique("stay_id", keep="first")
    stay_0 = stay_0.join(subj_map, on="stay_id", how="left")
    stay_d = stay_d.join(subj_map, on="stay_id", how="left")
    stay_a = stay_a.join(subj_map, on="stay_id", how="left")
    stay_b = stay_b.join(subj_map, on="stay_id", how="left")
    stay_c = stay_c.join(subj_map, on="stay_id", how="left")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline 0…")
    folds_0, _, _, _, _ = run_cv(stay_0, strata, "Pipeline0")
    summary_0 = summarise_cv(folds_0, "Pipeline0")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline D…")
    folds_d, _, _, _, _ = run_cv(stay_d, strata, "PipelineD")
    summary_d = summarise_cv(folds_d, "PipelineD")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline A…")
    folds_a, _, _, _, _ = run_cv(stay_a, strata, "PipelineA")
    summary_a = summarise_cv(folds_a, "PipelineA")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline B…")
    folds_b, last_model_b, last_X_test_b, last_test_df_b, feat_cols_b = run_cv(stay_b, strata, "PipelineB")
    summary_b = summarise_cv(folds_b, "PipelineB")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline C…")
    folds_c, last_model_c, last_X_test_c, last_test_df_c, feat_cols_c = run_cv(stay_c, strata, "PipelineC")
    summary_c = summarise_cv(folds_c, "PipelineC")

    print("\nPlotting AUPRC by stratum…")
    pipeline_summaries = [
        ("Pipeline 0 (raw stats, no imperfekt)", summary_0),
        ("Pipeline D (raw stats + observation count)", summary_d),
        ("Pipeline A (resampled)", summary_a),
        ("Pipeline B (imperfekt)", summary_b),
        ("Pipeline C (raw->resampled imperfekt)", summary_c),
    ]
    plot_auprc_by_stratum(
        pipeline_summaries,
        RESULTS_DIR / "figures" / "auprc_by_stratum.svg",
        stratum_labels=stratum_labels,
    )
    plot_auprc_lift_by_stratum(
        pipeline_summaries,
        RESULTS_DIR / "figures" / "auprc_lift_by_stratum.svg",
        stratum_labels=stratum_labels,
    )

    print("\nComputing SHAP group-importance analysis for Pipeline B (last CV fold)…")
    if last_model_b is not None and last_X_test_b is not None and last_test_df_b is not None:
        run_shap_group_analysis(
            model=last_model_b,
            X_test=last_X_test_b,
            test_stay_df=last_test_df_b,
            strata=strata,
            feature_cols=feat_cols_b,
            pipeline_name="PipelineB",
            stratum_labels=stratum_labels,
        )
    else:
        print("Skipping SHAP for Pipeline B: no final-fold artifacts available.")

    print("\nDone.")


if __name__ == "__main__":
    main()
