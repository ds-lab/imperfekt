from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from imperfekt.analysis.irregularity.irregularity import Irregularity  # noqa: E402
from examples.utils.models import XGBoostModel  # noqa: E402
from examples.mimic_iv_ed.config import (  # noqa: E402
    RESULTS_DIR,
    OUTCOME_COL,
    CV_N_SPLITS,
    CV_N_REPEATS,
    RANDOM_STATE,
)


def compute_irregularity_strata(
    ts_df: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[str, str]]:
    """
    Run Irregularity on the full dataset to extract per-stay raw metrics and
    the dynamically selected orthogonal axis names.

    Returns:
      case_metrics  - full cs_case_scores DataFrame (stay_id, cv,
                      adherence_rate, burstiness_coeff, axis_x, axis_y, …)
                      The library-generated irregularity_stratum column is
                      present but must NOT be used for CV evaluation — it was
                      computed from global medians and would leak test data.
      axes          - (axis_x_name, axis_y_name) selected by the library
    """
    strata_dir = RESULTS_DIR / "irregularity_strata"
    ireg = Irregularity(
        ts_df.select(["stay_id", "charttime"]),
        id_col="stay_id",
        clock_col="charttime",
        save_path=strata_dir,
    )
    ireg.run(save_results=True)
    case_metrics = ireg.results.cs_case_scores
    axis_x = case_metrics["axis_x"][0]
    axis_y = case_metrics["axis_y"][0]
    return case_metrics, (axis_x, axis_y)


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


_INVERTED_AXES = {"adherence_rate"}


def _hi(col: str, med: float) -> pl.Expr:
    return (pl.col(col) <= med) if col in _INVERTED_AXES else (pl.col(col) > med)


def run_cv(
    stay_df: pl.DataFrame,
    case_metrics: pl.DataFrame,
    axes: tuple[str, str],
    pipeline_name: str,
) -> tuple[
    dict[str, list],
    XGBoostModel | None,
    np.ndarray | None,
    pl.DataFrame | None,
    list[str],
    pl.DataFrame | None,
]:
    """
    Repeated stratified k-fold CV (CV_N_SPLITS x CV_N_REPEATS).
    Splits are on subject_id to prevent patient-level leakage across folds.

    Irregularity quadrant thresholds (Q_alpha/Q_beta/Q_gamma/Q_delta) are derived strictly from
    the *train* fold on each iteration, then applied to test stays, so no
    test-set information leaks into the stratum evaluation boundaries.

    Returns:
      fold_metrics     - dict mapping "overall" and each stratum label to a
                         list of metric dicts, one per fold
      last_model       - trained model from the final fold (for SHAP)
      last_X_test      - test features from the final fold
      last_test_df     - exact test DataFrame from the final fold
      feature_cols     - ordered list of feature column names
      last_test_strata - stay_id/irregularity_stratum for the final fold's
                         test set (train-derived thresholds)
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

    axis_x, axis_y = axes

    fold_metrics: dict[str, list] = defaultdict(list)
    last_model = None
    last_X_test = None
    last_test_df = None
    last_test_strata = None

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

        # Medians from train only; thresholds applied to test — no leakage.
        train_metrics = case_metrics.filter(pl.col("stay_id").is_in(train["stay_id"].to_list()))
        med_x = train_metrics[axis_x].median()
        med_y = train_metrics[axis_y].median()
        x_high = _hi(axis_x, med_x)
        y_high = _hi(axis_y, med_y)

        test_stay_ids = test["stay_id"]
        test_strata = (
            case_metrics
            .filter(pl.col("stay_id").is_in(test_stay_ids.to_list()))
            .select(["stay_id", axis_x, axis_y])
            .drop_nulls([axis_x, axis_y])
            .with_columns(
                pl.when(~x_high & ~y_high).then(pl.lit("Q_alpha"))
                .when(x_high & ~y_high).then(pl.lit("Q_beta"))
                .when(~x_high & y_high).then(pl.lit("Q_gamma"))
                .when(x_high & y_high).then(pl.lit("Q_delta"))
                .otherwise(pl.lit(None))
                .alias("irregularity_stratum")
            )
            .select(["stay_id", "irregularity_stratum"])
        )

        # Join test strata onto test rows to get a per-row stratum label aligned
        # with y_test/y_proba, then group by stratum without repeated is_in scans.
        strata_arr = (
            test.select("stay_id")
            .join(test_strata, on="stay_id", how="left")
            ["irregularity_stratum"]
            .fill_null("")
            .to_numpy()
        )
        for stratum_label in np.unique(strata_arr):
            if stratum_label == "":
                continue
            mask = strata_arr == stratum_label
            m_s = _compute_metrics(y_test[mask], y_proba[mask])
            if m_s:
                fold_metrics[stratum_label].append(m_s)

        last_model = model
        last_test_df = test
        last_X_test = X_test
        last_test_strata = test_strata

    print()
    return dict(fold_metrics), last_model, last_X_test, last_test_df, feature_cols, last_test_strata


def summarise_cv(fold_metrics: dict[str, list], pipeline_name: str) -> dict[str, dict]:
    """
    Aggregate per-fold metric lists into mean ± 95% CI (t-distribution).
    Returns a dict keyed by stratum label (plus "overall").
    """
    from scipy import stats

    summary = {}
    for key, folds in fold_metrics.items():
        for metric in ("auprc", "auprc_lift", "auroc", "brier_skill_score", "n_pos_pct"):
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


def print_information_gain_ratio(
    baseline_summary: dict[str, dict],
    candidate_summary: dict[str, dict],
    baseline_name: str,
    candidate_name: str,
    metric: str = "auprc",
    test_set_name: str = "overall"
) -> None:
    """Print overall relative gain ratio: (candidate - baseline) / baseline."""
    baseline = baseline_summary.get(test_set_name, {}).get(metric, {}).get("mean", float("nan"))
    candidate = candidate_summary.get(test_set_name, {}).get(metric, {}).get("mean", float("nan"))

    if np.isnan(baseline) or np.isnan(candidate) or baseline == 0:
        print(f"Information gain ratio ({metric.upper()}) {baseline_name} -> {candidate_name}: nan")
        return

    igr = (candidate - baseline) / baseline
    print(
        f"Information gain ratio ({metric.upper()}) {baseline_name} -> {candidate_name}: "
        f"{igr:.3f} ({igr * 100:+.1f}%) [{candidate:.3f} vs {baseline:.3f}]"
    )
