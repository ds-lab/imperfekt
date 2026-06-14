"""GRU cross-validation loop for the NEMSIS/MC-MED sequence experiments.

Mirrors the structure of cv.py's run_cv() but operates on the raw long-format
cohort DataFrame rather than stay-level aggregated features. Each fold converts
train/test subsets to (N, T, D) 3D arrays via df_to_3d_array, then trains a
GRUModel. SHAP is not computed for GRU runs.

Shared utilities (undersample_train, prior_correct_probs, _compute_metrics,
summarise_cv, save_cv_results) are imported directly from cv.py to avoid drift.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
from sklearn.model_selection import RepeatedStratifiedKFold

from config import (
    APPLY_PRIOR_CORRECTION,
    APPLY_UNDERSAMPLING,
    CV_N_REPEATS,
    CV_N_SPLITS,
    RANDOM_STATE,
    TRAIN_NEG_POS_RATIO,
    UNDERSAMPLE_RANDOM_STATE,
    VITAL_COLS,
)
from cv import (
    _compute_metrics,
    prior_correct_probs,
)
from examples.utils.gru_model import GRUModel
from seq_array import df_to_3d_array


def _undersample_indices(y: np.ndarray, ratio: int, random_state: int) -> np.ndarray:
    """Return indices that keep all positives and a random sample of negatives."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y != 1)
    n_neg_target = min(len(neg_idx), ratio * len(pos_idx))
    rng = np.random.default_rng(random_state)
    sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
    keep = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep)
    return keep


def run_cv_gru(
    cohort_df: pl.DataFrame,
    case_metrics: pl.DataFrame,
    axes: tuple[str, str],
    pipeline_name: str,
    use_mask: bool = True,
) -> tuple[
    dict[str, list],
    GRUModel | None,
    None,
    pl.DataFrame | None,
    list[str],
    pl.DataFrame | None,
]:
    """Repeated stratified k-fold CV using a GRU sequence model.

    Args:
        cohort_df:     Long-format DataFrame (id, clock, vitals..., label) after
                       plausibility filtering and imputation by ConfigBuilder.
        case_metrics:  Per-case intervariable missingness metrics from
                       compute_intervariable_missingness_strata().
        axes:          (axis_x_name, axis_y_name) for intervariable strata.
        pipeline_name: Label used in progress output and result keys.
        use_mask:      When True the GRU input includes the binary observed-mask
                       channel; when False it is values-only (mask ablation).
                       Comparing the two per stratum shows where the missingness
                       indicator adds predictive information.

    Returns:
        fold_metrics     - dict mapping "overall" and each stratum label to a
                           list of metric dicts, one per fold.
        last_model       - trained GRUModel from the final fold.
        None             - placeholder for SHAP compatibility with run_cv signature.
        last_test_df     - exact test DataFrame (long-format) from the final fold.
        feature_cols     - VITAL_COLS used as sequence features.
        last_test_strata - id/intervariable_stratum for the final fold's test set.
    """
    # Stay-level labels for stratified splitting (one row per unique id).
    subject_labels = (
        cohort_df.select(["id", "label"])
        .group_by("id")
        .agg(pl.col("label").max().alias("any_outcome"))
        .sort("id")
    )

    subjects = subject_labels["id"].to_numpy()
    subject_outcomes = subject_labels["any_outcome"].cast(pl.Int8).to_numpy()

    TRUE_POPULATION_PREVALENCE = float(subject_outcomes.mean())
    print(
        f"  [{pipeline_name}] true population prevalence (π₀) = "
        f"{TRUE_POPULATION_PREVALENCE:.5f}"
    )

    rskf = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    axis_x, axis_y = axes

    imperfekt_cols = [
        "avg_indicated_vars_pct",
        "co_missingness_concentration",
        "missing_variable_breadth",
        "pattern_entropy",
        "max_pairwise_co_missingness",
    ]

    fold_metrics: dict[str, list] = defaultdict(list)
    last_model: GRUModel | None = None
    last_test_df: pl.DataFrame | None = None
    last_test_strata: pl.DataFrame | None = None

    total_folds = CV_N_SPLITS * CV_N_REPEATS

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(subjects, subject_outcomes)):
        print(f"  [{pipeline_name}] fold {fold_idx + 1}/{total_folds}", end="\r")

        train_subjects = set(subjects[train_idx].tolist())
        test_subjects = set(subjects[test_idx].tolist())

        train_df = cohort_df.filter(pl.col("id").is_in(train_subjects))
        test_df = cohort_df.filter(pl.col("id").is_in(test_subjects))

        # Build (N, T, D) arrays. df_to_3d_array sorts by id then clock and
        # pads shorter sequences with NaN up to the longest in that split.
        train_ids, X_train_3d, _, _ = df_to_3d_array(train_df, "id", "clock", list(VITAL_COLS))
        test_ids, X_test_3d, _, _ = df_to_3d_array(test_df, "id", "clock", list(VITAL_COLS))

        # Static features: one row per subject, aligned to sorted id order.
        static_cols = ["age"] if "age" in cohort_df.columns else []
        if static_cols:
            id_to_static_train = (
                train_df.group_by("id").agg([pl.col(c).first() for c in static_cols])
            )
            id_to_static_test = (
                test_df.group_by("id").agg([pl.col(c).first() for c in static_cols])
            )
            X_train_static = np.array(
                [id_to_static_train.filter(pl.col("id") == i).select(static_cols).row(0) for i in train_ids],
                dtype=np.float32,
            )
            X_test_static = np.array(
                [id_to_static_test.filter(pl.col("id") == i).select(static_cols).row(0) for i in test_ids],
                dtype=np.float32,
            )
        else:
            X_train_static = None
            X_test_static = None

        # Labels aligned to sorted id order from df_to_3d_array.
        id_to_label_train = dict(
            zip(
                subject_labels.filter(pl.col("id").is_in(train_subjects))["id"].to_list(),
                subject_labels.filter(pl.col("id").is_in(train_subjects))["any_outcome"].to_list(),
            )
        )
        id_to_label_test = dict(
            zip(
                subject_labels.filter(pl.col("id").is_in(test_subjects))["id"].to_list(),
                subject_labels.filter(pl.col("id").is_in(test_subjects))["any_outcome"].to_list(),
            )
        )

        y_train = np.array([id_to_label_train[i] for i in train_ids], dtype=np.int8)
        y_test = np.array([id_to_label_test[i] for i in test_ids], dtype=np.int8)

        # Undersampling at stay level (N axis). X[keep_idx] works for 3D arrays.
        if APPLY_UNDERSAMPLING:
            keep_idx = _undersample_indices(
                y_train,
                ratio=TRAIN_NEG_POS_RATIO,
                random_state=UNDERSAMPLE_RANDOM_STATE + fold_idx,
            )
            X_train_us, y_train_us = X_train_3d[keep_idx], y_train[keep_idx]
            X_train_static_us = X_train_static[keep_idx] if X_train_static is not None else None
        else:
            X_train_us, y_train_us = X_train_3d, y_train
            X_train_static_us = X_train_static

        pi_train_art = float(y_train_us.mean())

        model = GRUModel(
            random_state=RANDOM_STATE + fold_idx,
            feature_mode=pipeline_name,
            use_mask=use_mask,
        )
        model._train_model(X_train_us.astype(np.float32), y_train_us, X_train_static_us)
        _, y_proba = model._predict(X_test_3d.astype(np.float32), X_test_static)

        if APPLY_PRIOR_CORRECTION:
            y_proba = prior_correct_probs(
                y_proba,
                pi_true=TRUE_POPULATION_PREVALENCE,
                pi_train=pi_train_art,
            )

        n_val = len(y_test)
        val_prevalence = float(y_test.mean()) if n_val else float("nan")
        n_train_pos = int((y_train_us == 1).sum())
        n_train_neg = int((y_train_us != 1).sum())
        print(
            f"  [{pipeline_name}] fold {fold_idx + 1}/{total_folds} "
            f"n_train_pos={n_train_pos} n_train_neg={n_train_neg} "
            f"pi_train_art={pi_train_art:.4f} n_val={n_val} "
            f"val_prevalence={val_prevalence:.5f}"
        )

        m = _compute_metrics(y_test, y_proba)
        if m:
            fold_metrics["overall"].append(m)

        # Stratum thresholds derived from train fold only — no leakage.
        train_metrics = case_metrics.filter(pl.col("id").is_in(list(train_subjects)))
        med_x = train_metrics[axis_x].median()
        med_y = train_metrics[axis_y].median()

        strata_input_cols = ["id", axis_x, axis_y]
        if "avg_indicated_vars_pct" not in strata_input_cols:
            strata_input_cols.append("avg_indicated_vars_pct")

        test_strata = (
            IntervariableImperfection.assign_strata(
                case_metrics.filter(pl.col("id").is_in(list(test_subjects))).select(
                    strata_input_cols
                ),
                axis_x,
                axis_y,
                med_x,
                med_y,
            )
            .select(["id", "intervariable_stratum"])
            .drop_nulls("intervariable_stratum")
        )

        available_imperfekt = [c for c in imperfekt_cols if c in case_metrics.columns]
        test_strata_imperfekt = test_strata.join(
            case_metrics.select(available_imperfekt + ["id"]),
            on="id",
            how="left",
        )

        # Align strata to the sorted test_ids order from df_to_3d_array.
        test_ids_df = pl.DataFrame({"id": test_ids})
        strata_arr = (
            test_ids_df.join(
                test_strata_imperfekt.select(["id", "intervariable_stratum"]),
                on="id",
                how="left",
            )["intervariable_stratum"]
            .fill_null("")
            .to_numpy()
        )

        imperfekt_lookup: dict[str, dict[str, float]] = {}
        for row in test_strata_imperfekt.iter_rows(named=True):
            sid = row["id"]
            imperfekt_lookup[sid] = {c: row[c] for c in imperfekt_cols if c in row}

        for stratum_label in np.unique(strata_arr):
            if stratum_label == "":
                continue
            mask = strata_arr == stratum_label
            m_s = _compute_metrics(y_test[mask], y_proba[mask])
            if m_s:
                for imperfekt_metric in imperfekt_cols:
                    vals = [
                        imperfekt_lookup[sid][imperfekt_metric]
                        for sid in np.asarray(test_ids)[mask]
                        if sid in imperfekt_lookup
                        and imperfekt_metric in imperfekt_lookup[sid]
                        and imperfekt_lookup[sid][imperfekt_metric] is not None
                    ]
                    m_s[imperfekt_metric] = float(np.mean(vals)) if vals else float("nan")
                fold_metrics[stratum_label].append(m_s)

        last_model = model
        last_test_df = test_df
        last_test_strata = test_strata

    print()
    return dict(fold_metrics), last_model, None, last_test_df, list(VITAL_COLS), last_test_strata
