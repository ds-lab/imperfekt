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
    VITAL_COLS,
    IREG_FEATURE_COLS,
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
_STRATUM_ORDER = ["overall", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]


def _hi(col: str, med: float) -> pl.Expr:
    return (pl.col(col) <= med) if col in _INVERTED_AXES else (pl.col(col) > med)


def _is_structural_feature(col: str) -> bool:
    return any(col == base or col.startswith(f"{base}_") for base in IREG_FEATURE_COLS)


def _feature_group(col: str) -> str:
    if _is_structural_feature(col):
        return "structural"
    if any(col.startswith(f"{vital}_") for vital in VITAL_COLS):
        return "physiology"
    return "metadata"


def _select_feature_columns(stay_df: pl.DataFrame) -> list[str]:
    return [
        c for c in stay_df.columns if c not in ("stay_id", OUTCOME_COL, "subject_id")
    ]


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
        # Also join cv/qcod/adherence_rate for irregularity characterisation per stratum.
        ireg_cols = ["stay_id", "cv", "qcod", "adherence_rate"]
        available_ireg = [c for c in ireg_cols if c in case_metrics.columns]
        test_strata_ireg = (
            test_strata
            .join(
                case_metrics.select(available_ireg),
                on="stay_id",
                how="left",
            )
        )

        strata_arr = (
            test.select("stay_id")
            .join(test_strata_ireg.select(["stay_id", "irregularity_stratum"]), on="stay_id", how="left")
            ["irregularity_stratum"]
            .fill_null("")
            .to_numpy()
        )

        ireg_lookup: dict[str, dict[str, float]] = {}
        for row in test_strata_ireg.iter_rows(named=True):
            sid = row["stay_id"]
            ireg_lookup[sid] = {
                c: row[c] for c in ("cv", "qcod", "adherence_rate") if c in row
            }

        stay_ids_arr = test["stay_id"].to_numpy()

        for stratum_label in np.unique(strata_arr):
            if stratum_label == "":
                continue
            mask = strata_arr == stratum_label
            m_s = _compute_metrics(y_test[mask], y_proba[mask])
            if m_s:
                for ireg_metric in ("cv", "qcod", "adherence_rate"):
                    vals = [
                        ireg_lookup[sid][ireg_metric]
                        for sid in stay_ids_arr[mask]
                        if sid in ireg_lookup and ireg_metric in ireg_lookup[sid]
                        and ireg_lookup[sid][ireg_metric] is not None
                    ]
                    m_s[ireg_metric] = float(np.mean(vals)) if vals else float("nan")
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
        for metric in ("auprc", "auprc_lift", "auroc", "brier_skill_score", "n_pos_pct", "cv", "qcod", "adherence_rate"):
            vals = np.array([f[metric] for f in folds if metric in f and not np.isnan(f[metric])])
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


def save_cv_results(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
) -> None:
    """
    Write a tidy CSV with one row per pipeline × stratum, columns:
      pipeline, stratum, auprc_mean, auprc_ci, auprc_lift_mean, auprc_lift_ci,
      auroc_mean, auroc_ci, brier_skill_score_mean, brier_skill_score_ci,
      n_pos_pct_mean, n_pos_pct_ci
    """
    metrics = ("auprc", "auprc_lift", "auroc", "brier_skill_score", "n_pos_pct", "cv", "qcod", "adherence_rate")

    rows = []
    for pipeline_name, summary in pipeline_summaries:
        strata = [s for s in _STRATUM_ORDER if s in summary] + sorted(
            s for s in summary if s not in _STRATUM_ORDER
        )
        for stratum in strata:
            row: dict = {"pipeline": pipeline_name, "stratum": stratum}
            for m in metrics:
                v = summary.get(stratum, {}).get(m)
                row[f"{m}_mean"] = v["mean"] if v else float("nan")
                row[f"{m}_ci"] = v["ci"] if v else float("nan")
            rows.append(row)

    df = pl.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(save_path)
    print(f"CV results saved to {save_path}")


def save_feature_distribution_by_outcome(
    stay_df: pl.DataFrame,
    save_path: Path,
) -> None:
    """
    Write feature-distribution table split by binary outcome.

    Includes all stay-level model features (physiology, metadata, and
    structural irregularity), with mean ± 95% CI and quantile summaries.
    """
    from scipy import stats

    feature_cols = _select_feature_columns(stay_df)
    if not feature_cols:
        print("Skipping feature-by-outcome table: no feature columns found.")
        return

    long_df = (
        stay_df
        .select([OUTCOME_COL] + feature_cols)
        .unpivot(
            index=[OUTCOME_COL],
            on=feature_cols,
            variable_name="feature",
            value_name="value",
        )
        .drop_nulls("value")
    )
    if long_df.height == 0:
        print("Skipping feature-by-outcome table: all selected feature values are null.")
        return

    grouped = long_df.group_by([OUTCOME_COL, "feature"]).agg(
        [
            pl.col("value").count().alias("n_non_null"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
            pl.col("value").median().alias("median"),
            pl.col("value").quantile(0.25).alias("q25"),
            pl.col("value").quantile(0.75).alias("q75"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
        ]
    )

    rows: list[dict] = []
    for row in grouped.iter_rows(named=True):
        n = int(row["n_non_null"])
        std = row["std"]
        if n > 1 and std is not None and not np.isnan(std):
            se = float(std) / np.sqrt(n)
            ci = float(stats.t.ppf(0.975, df=n - 1) * se)
        else:
            ci = float("nan")

        outcome_val = row[OUTCOME_COL]
        outcome_int = int(outcome_val) if outcome_val is not None else None

        rows.append(
            {
                "feature": row["feature"],
                "feature_group": _feature_group(row["feature"]),
                "outcome": outcome_int,
                "n_non_null": n,
                "mean": row["mean"],
                "ci": ci,
                "std": std,
                "median": row["median"],
                "q25": row["q25"],
                "q75": row["q75"],
                "min": row["min"],
                "max": row["max"],
            }
        )

    out_df = pl.DataFrame(rows).sort(["feature", "outcome"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(save_path)
    print(f"Feature-by-outcome table saved to {save_path}")


def save_feature_distribution_by_quadrant_cv(
    stay_df: pl.DataFrame,
    case_metrics: pl.DataFrame,
    axes: tuple[str, str],
    save_path: Path,
) -> None:
    """
    Write per-feature mean ± 95% CI split by irregularity quadrant.

    Quadrants are assigned per fold using train-only medians (same leakage-safe
    protocol as model CV). CIs are over fold-level feature means.
    Also appends fold-aggregated outcome prevalence (mean ± 95% CI) for each
    stratum and overall.
    """
    from scipy import stats

    feature_cols = _select_feature_columns(stay_df)
    if not feature_cols:
        print("Skipping feature-by-quadrant table: no feature columns found.")
        return

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
    fold_feature_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    fold_outcome_prevalence: dict[str, list[float]] = defaultdict(list)

    total_folds = CV_N_SPLITS * CV_N_REPEATS
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(subjects, subject_outcomes)):
        print(f"  [feature-distribution] fold {fold_idx + 1}/{total_folds}", end="\r")

        train_subjects = set(subjects[train_idx].tolist())
        test_subjects = set(subjects[test_idx].tolist())

        train = stay_df.filter(pl.col("subject_id").is_in(train_subjects))
        test = stay_df.filter(pl.col("subject_id").is_in(test_subjects))

        train_metrics = case_metrics.filter(pl.col("stay_id").is_in(train["stay_id"].to_list()))
        med_x = train_metrics[axis_x].median()
        med_y = train_metrics[axis_y].median()
        if med_x is None or med_y is None:
            continue

        x_high = _hi(axis_x, med_x)
        y_high = _hi(axis_y, med_y)

        test_strata = (
            case_metrics
            .filter(pl.col("stay_id").is_in(test["stay_id"].to_list()))
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

        overall_prevalence = test[OUTCOME_COL].mean()
        if overall_prevalence is not None and not np.isnan(overall_prevalence):
            fold_outcome_prevalence["overall"].append(float(overall_prevalence))

        overall_means = (
            test
            .select(feature_cols)
            .melt(
                value_vars=feature_cols,
                variable_name="feature",
                value_name="value",
            )
            .drop_nulls("value")
            .group_by("feature")
            .agg(pl.col("value").mean().alias("fold_mean"))
        )
        for row in overall_means.iter_rows(named=True):
            fold_feature_means[("overall", row["feature"])].append(float(row["fold_mean"]))

        test_with_strata = (
            test.select(["stay_id", OUTCOME_COL] + feature_cols)
            .join(test_strata, on="stay_id", how="left")
            .drop_nulls("irregularity_stratum")
        )
        if test_with_strata.height == 0:
            continue

        stratum_prevalence = (
            test_with_strata
            .group_by("irregularity_stratum")
            .agg(pl.col(OUTCOME_COL).mean().alias("fold_prevalence"))
        )
        for row in stratum_prevalence.iter_rows(named=True):
            prev = row["fold_prevalence"]
            if prev is None or np.isnan(prev):
                continue
            fold_outcome_prevalence[row["irregularity_stratum"]].append(float(prev))

        stratum_means = (
            test_with_strata
            .melt(
                id_vars=["irregularity_stratum"],
                value_vars=feature_cols,
                variable_name="feature",
                value_name="value",
            )
            .drop_nulls("value")
            .group_by(["irregularity_stratum", "feature"])
            .agg(pl.col("value").mean().alias("fold_mean"))
        )
        for row in stratum_means.iter_rows(named=True):
            fold_feature_means[(row["irregularity_stratum"], row["feature"])].append(
                float(row["fold_mean"])
            )

    print()

    prevalence_summary: dict[str, dict[str, float]] = {}
    for stratum, vals in fold_outcome_prevalence.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            continue
        prev_mean = float(arr.mean())
        if arr.size > 1:
            prev_se = float(arr.std(ddof=1) / np.sqrt(arr.size))
            prev_ci = float(stats.t.ppf(0.975, df=arr.size - 1) * prev_se)
        else:
            prev_ci = float("nan")
        prevalence_summary[stratum] = {
            "mean": prev_mean,
            "ci": prev_ci,
            "n_folds": int(arr.size),
        }

    rows: list[dict] = []
    for (stratum, feature), vals in fold_feature_means.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            continue

        mean = float(arr.mean())
        if arr.size > 1:
            se = float(arr.std(ddof=1) / np.sqrt(arr.size))
            ci = float(stats.t.ppf(0.975, df=arr.size - 1) * se)
        else:
            ci = float("nan")

        prev = prevalence_summary.get(stratum)

        rows.append(
            {
                "feature": feature,
                "feature_group": _feature_group(feature),
                "stratum": stratum,
                "n_folds": int(arr.size),
                "mean": mean,
                "ci": ci,
                "outcome_prevalence_mean": prev["mean"] if prev else float("nan"),
                "outcome_prevalence_ci": prev["ci"] if prev else float("nan"),
                "outcome_prevalence_n_folds": prev["n_folds"] if prev else 0,
            }
        )

    if not rows:
        print("Skipping feature-by-quadrant table: no fold summaries were produced.")
        return

    stratum_rank = {s: i for i, s in enumerate(_STRATUM_ORDER)}
    rows.sort(
        key=lambda r: (
            r["feature"],
            stratum_rank.get(r["stratum"], len(stratum_rank)),
            r["stratum"],
        )
    )

    out_df = pl.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(save_path)
    print(f"Feature-by-quadrant table saved to {save_path}")


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
