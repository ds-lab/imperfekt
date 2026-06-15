import json
from pathlib import Path

from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
from imperfekt.analysis.intravariable.intravariable import IntravariableImperfection
import polars as pl

import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from config import (
    RESULTS_DIR,
    VITAL_COLS,
    RANDOM_STATE,
    CV_N_SPLITS,
    CV_N_REPEATS,
    APPLY_UNDERSAMPLING,
    TRAIN_NEG_POS_RATIO,
    UNDERSAMPLE_RANDOM_STATE,
    APPLY_PRIOR_CORRECTION,
    SHAP_MAX_ROWS_PER_STRATUM,
    SHAP_SUBSAMPLE_RANDOM_STATE,
    data_fingerprint,
    data_fingerprint_tag,
)
from examples.nemsis.features import feature_group, is_structural_feature
from examples.utils.models import XGBoostModel  # noqa: E402

_STRATA_CACHE_ROOT = RESULTS_DIR / "intervariable_strata_cache"
_INTRA_STRATA_CACHE_ROOT = RESULTS_DIR / "intravariable_strata_cache"
_STRATA_CACHE_VERSION = 1


def _strata_cache_dir(cohort_path: Path) -> Path:
    return _STRATA_CACHE_ROOT / f"{cohort_path.stem}_{data_fingerprint_tag(cohort_path)}"


def _strata_cache_key(cohort_path: Path) -> dict:
    return {
        "version": _STRATA_CACHE_VERSION,
        "data": data_fingerprint(cohort_path),
        "vital_cols": list(VITAL_COLS),
    }

def compute_intervariable_missingness_strata(
    df: pl.DataFrame,
    cohort_path: Path | None = None,
) -> tuple[pl.DataFrame, tuple[str, str]]:
    """
    Run Irregularity on the full dataset to extract per-stay raw metrics and
    the dynamically selected orthogonal axis names.

    If cohort_path is given, the result is cached at RESULTS_DIR/
    intervariable_strata_cache/<cohort_stem>/ keyed by (cohort mtime, cohort
    size, VITAL_COLS). The per-cohort subdirectory ensures different cohorts
    (different dataset/endpoint/window/min-readings) don't overwrite each
    other. Subsequent calls with a matching key load from disk instead of
    rerunning the (expensive) library computation.

    Returns:
      case_metrics  - full iv_composite_scores DataFrame (id, cv,
                      adherence_rate, burstiness_coeff, axis_x, axis_y, …)
                      The library-generated irregularity_stratum column is
                      present but must NOT be used for CV evaluation — it was
                      computed from global medians and would leak test data.
      axes          - (axis_x_name, axis_y_name) selected by the library
    """
    cache_dir = _strata_cache_dir(cohort_path) if cohort_path is not None else None
    cache_meta_path = cache_dir / "meta.json" if cache_dir is not None else None
    cache_metrics_path = cache_dir / "case_metrics.parquet" if cache_dir is not None else None

    if cohort_path is not None and cache_meta_path.exists() and cache_metrics_path.exists():
        expected_key = _strata_cache_key(cohort_path)
        cached_meta = json.loads(cache_meta_path.read_text())
        if cached_meta.get("key") == expected_key:
            case_metrics = pl.read_parquet(cache_metrics_path)
            axes = tuple(cached_meta["axes"])
            print(f"Loaded intervariable strata from cache: {cache_dir}")
            return case_metrics, axes

    strata_dir = RESULTS_DIR / "intervariable_strata"
    imp = IntervariableImperfection(
        df,
        imperfection="missingness",
        id_col="id",
        clock_col="clock",
        cols=VITAL_COLS,
        save_path=strata_dir,
    )
    imp.composite_score(save_results=True)
    case_metrics = imp.results.iv_composite_scores
    axis_x = case_metrics["axis_x"][0]
    axis_y = case_metrics["axis_y"][0]

    if cohort_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        case_metrics.write_parquet(cache_metrics_path)
        cache_meta_path.write_text(
            json.dumps(
                {"key": _strata_cache_key(cohort_path), "axes": [axis_x, axis_y]},
                indent=2,
            )
        )
        print(f"Cached intervariable strata to {cache_dir}")

    return case_metrics, (axis_x, axis_y)


def compute_intravariable_missingness_strata(
    df: pl.DataFrame,
    cohort_path: Path | None = None,
) -> tuple[pl.DataFrame, tuple[str, str]]:
    """
    Run IntravariableImperfection on the full dataset to extract per-stay
    cross-variable missingness burden and the dynamically selected axis pair.

    Cached identically to compute_intervariable_missingness_strata under
    RESULTS_DIR/intravariable_strata_cache/. Cache key is the same structure
    (version, data fingerprint, vital_cols).

    Returns:
      case_metrics  - iv_composite_scores (id, {col}_indicated_pct, axis_x, axis_y, …)
      axes          - (axis_x_name, axis_y_name) selected by the library
    """
    cache_dir = (
        _INTRA_STRATA_CACHE_ROOT / f"{cohort_path.stem}_{data_fingerprint_tag(cohort_path)}"
        if cohort_path is not None else None
    )
    cache_meta_path = cache_dir / "meta.json" if cache_dir is not None else None
    cache_metrics_path = cache_dir / "case_metrics.parquet" if cache_dir is not None else None

    if cohort_path is not None and cache_meta_path.exists() and cache_metrics_path.exists():
        expected_key = _strata_cache_key(cohort_path)
        cached_meta = json.loads(cache_meta_path.read_text())
        if cached_meta.get("key") == expected_key:
            case_metrics = pl.read_parquet(cache_metrics_path)
            axes = tuple(cached_meta["axes"])
            print(f"Loaded intravariable strata from cache: {cache_dir}")
            return case_metrics, axes

    strata_dir = RESULTS_DIR / "intravariable_strata"
    imp = IntravariableImperfection(
        df,
        imperfection="missingness",
        id_col="id",
        clock_col="clock",
        cols=VITAL_COLS,
        save_path=strata_dir,
    )
    imp.composite_score(save_results=True)
    case_metrics = imp.results.iv_composite_scores
    axis_x = case_metrics["axis_x"][0]
    axis_y = case_metrics["axis_y"][0]

    if cohort_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        case_metrics.write_parquet(cache_metrics_path)
        cache_meta_path.write_text(
            json.dumps(
                {"key": _strata_cache_key(cohort_path), "axes": [axis_x, axis_y]},
                indent=2,
            )
        )
        print(f"Cached intravariable strata to {cache_dir}")

    return case_metrics, (axis_x, axis_y)


def combine_case_metrics(
    inter_metrics: pl.DataFrame,
    intra_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """
    Left-join intravariable metrics onto intervariable metrics on id.

    Drops mode-specific axis metadata and stratum columns from intra before
    joining (they would collide with the generic axis_x/axis_y names from
    inter, and each mode's individual DataFrame remains the authoritative
    source for axis selection). The combined frame carries all numeric metrics
    from both sources for stratum characteristic reporting and feature
    engineering.
    """
    _drop = {
        "axis_x", "axis_y", "axis_pair_corr",
        "axis_x_median_threshold", "axis_y_median_threshold",
        "imperfection_stratum",
    }
    intra_slim = intra_metrics.drop([c for c in _drop if c in intra_metrics.columns])
    return inter_metrics.join(intra_slim, on="id", how="left")


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


_STRATUM_ORDER = ["overall", "Q_complete", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]


_is_structural_feature = is_structural_feature
_feature_group = feature_group


def _select_feature_columns(stay_df: pl.DataFrame) -> list[str]:
    return [c for c in stay_df.columns if c not in ("id", "label")]


def undersample_train(
    X: np.ndarray,
    y: np.ndarray,
    ratio: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-only negative undersampling to a fixed pos:neg ratio.

    Keeps **all** positives and randomly draws negatives **without replacement**
    until ``n_neg = ratio * n_pos`` (or all negatives, whichever is fewer). This
    rebalances the *training* fold so XGBoost is not overwhelmed by the extreme
    (~0.19%) class imbalance, while the held-out validation fold is left at its
    natural prevalence. Because undersampling shifts the training prior away from
    the population prior, predictions must be rescaled afterwards with
    :func:`prior_correct_probs` (train-only undersampling + Bayesian prior
    correction; cf. Dal Pozzolo et al. 2015, "Calibrating Probability with
    Undersampling for Unbalanced Classification").

    Must only ever be applied to a training subset, never to validation/test
    data, to avoid optimistically distorting the evaluated prevalence.

    Args:
        X: Training feature matrix, shape (n, d).
        y: Binary training labels, shape (n,).
        ratio: Target number of negatives per positive after undersampling.
        random_state: Seed for the negative draw (combine with the fold index
            upstream for per-fold reproducibility).

    Returns:
        (X_under, y_under) with all positives and the sampled negatives, in a
        shuffled order.
    """
    y = np.asarray(y)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y != 1)

    n_neg_target = min(len(neg_idx), ratio * len(pos_idx))

    rng = np.random.default_rng(random_state)
    sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)

    keep = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep)
    return X[keep], y[keep]


def prior_correct_probs(
    p: np.ndarray,
    pi_true: float,
    pi_train: float,
) -> np.ndarray:
    """Bayesian prior correction for probabilities from an undersampled model.

    Args:
        p: Raw ``predict_proba`` for the positive class, shape (n,).
        pi_true: True population (eligible-cohort) prevalence π₀.
        pi_train: Actual positive prevalence in the undersampled training set π_t.

    Returns:
        Prior-corrected probabilities, shape (n,).
    """
    # handle edge cases to avoid divide-by-zero or zero probabilities; clip to [1e-15, 1-1e-15]
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    odds = p / (1 - p)
    factor = (pi_true / (1 - pi_true)) / (pi_train / (1 - pi_train))
    corrected_odds = odds * factor
    return corrected_odds / (1 + corrected_odds)


def _stratified_shap_subsample(
    strata_arr: np.ndarray,
    max_per_stratum: int | None,
    random_state: int,
) -> np.ndarray:
    """Indices of a stratified row subsample for SHAP estimation.

    Caps the number of rows explained *per intervariable stratum* (and within
    the empty-label ``""`` group) at ``max_per_stratum``, drawing without
    replacement. Capping per stratum rather than globally keeps the rare strata
    (e.g. Q_delta) well-represented so their per-stratum mean |SHAP| stays a
    usable estimate. Returns all indices unchanged if ``max_per_stratum`` is
    None or no stratum exceeds the cap.

    The returned indices are sorted, so any positional alignment with
    ``X_test`` / ``strata_arr`` is preserved when the caller slices both by
    these indices.
    """
    n = len(strata_arr)
    if max_per_stratum is None or n <= max_per_stratum:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    keep: list[np.ndarray] = []
    for label in np.unique(strata_arr):
        idx = np.flatnonzero(strata_arr == label)
        if len(idx) > max_per_stratum:
            idx = rng.choice(idx, size=max_per_stratum, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def run_cv(
    stay_df: pl.DataFrame,
    case_metrics: pl.DataFrame,
    axes: tuple[str, str],
    pipeline_name: str,
    stratification_mode: str = "intervariable",
    shap_save_path: Path | None = None,
    features_save_path: Path | None = None,
    shap_full: bool = True,
    shap_interactions: bool = False,
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
    Splits are on id to prevent patient-level leakage across folds.

    Quadrant thresholds (Q_alpha/Q_beta/Q_gamma/Q_delta) are derived strictly from
    the *train* fold on each iteration, then applied to test stays, so no
    test-set information leaks into the stratum evaluation boundaries.
    ``stratification_mode`` selects which axis space is used ("intervariable" or
    "intravariable"); both sets of metrics are always reported as stratum
    characteristics via the combined case_metrics frame.

    If feature_distribution_save_path is given, per-fold feature means and
    outcome prevalence are accumulated per stratum (and "overall") and written
    as a tidy CSV at the end — using the same leakage-safe fold splits.

    Returns:
      fold_metrics     - dict mapping "overall" and each stratum label to a
                         list of metric dicts, one per fold
      last_model       - trained model from the final fold (for SHAP)
      last_X_test      - test features from the final fold
      last_test_df     - exact test DataFrame from the final fold
      feature_cols     - ordered list of feature column names
      last_test_strata - id/active_stratum for the final fold's
                         test set (train-derived thresholds)
    """
    feature_cols = [c for c in stay_df.columns if c not in ("id", "label")]
    fold_shap_abs: dict[str, list[np.ndarray]] = defaultdict(list)
    last_shap_interactions: dict[str, np.ndarray] = {}
    last_X_test_raw: np.ndarray | None = None
    last_y_test_raw: np.ndarray | None = None
    all_X_test_chunks: list[np.ndarray] = []
    all_y_test_chunks: list[np.ndarray] = []
    all_strata_chunks: list[np.ndarray] = []

    subject_labels = (
        stay_df.select(["id", "label"])
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

        train = stay_df.filter(pl.col("id").is_in(train_subjects))
        test = stay_df.filter(pl.col("id").is_in(test_subjects))

        X_train = train.select(feature_cols).to_numpy().astype(np.float32)
        y_train = train["label"].cast(pl.Int8).to_numpy()
        X_test = test.select(feature_cols).to_numpy().astype(np.float32)
        y_test = test["label"].cast(pl.Int8).to_numpy()

        # Train-only undersampling: keep all positives, subsample negatives to a
        # fixed pos:neg ratio.
        if APPLY_UNDERSAMPLING:
            X_train_us, y_train_us = undersample_train(
                X_train,
                y_train,
                ratio=TRAIN_NEG_POS_RATIO,
                random_state=UNDERSAMPLE_RANDOM_STATE + fold_idx,
            )
        else:
            X_train_us, y_train_us = X_train, y_train
        n_train_pos = int((y_train_us == 1).sum())
        n_train_neg = int((y_train_us != 1).sum())
        pi_train_art = float(y_train_us.mean())

        model = XGBoostModel(feature_mode=pipeline_name, random_state=RANDOM_STATE + fold_idx)
        if APPLY_UNDERSAMPLING:
            model.params["scale_pos_weight"] = 1.0
        model._train_model(X_train_us, y_train_us)
        _, y_proba = model._predict(X_test)

        # Prior correction
        if APPLY_PRIOR_CORRECTION:
            y_proba = prior_correct_probs(
                y_proba,
                pi_true=TRUE_POPULATION_PREVALENCE,
                pi_train=pi_train_art,
            )

        n_val = len(y_test)
        val_prevalence = float(y_test.mean()) if n_val else float("nan")
        print(
            f"  [{pipeline_name}] fold {fold_idx + 1}/{total_folds} "
            f"n_train_pos={n_train_pos} n_train_neg={n_train_neg} "
            f"pi_train_art={pi_train_art:.4f} n_val={n_val} "
            f"val_prevalence={val_prevalence:.5f}"
        )

        m = _compute_metrics(y_test, y_proba)
        if m:
            fold_metrics["overall"].append(m)

        # Medians from train only; thresholds applied to test — no leakage.
        train_metrics = case_metrics.filter(pl.col("id").is_in(train["id"].to_list()))
        med_x = train_metrics[axis_x].median()
        med_y = train_metrics[axis_y].median()

        test_ids = test["id"]
        test_case_rows = case_metrics.filter(pl.col("id").is_in(test_ids.to_list()))

        # Do NOT drop null-axis rows before assign_strata: Q_complete cases (no
        # missingness) have null axes by construction, and assign_strata labels them
        # Q_complete *before* its null-axis check. Drop unassignable rows (null
        # active_stratum) afterwards to keep Q_complete while excluding genuinely
        # undefined cases.
        if stratification_mode == "intravariable":
            _base_intra = {"id", axis_x, axis_y}
            intra_cols_needed = list(_base_intra) + [
                f"{c}_indicated_pct" for c in VITAL_COLS
                if f"{c}_indicated_pct" in case_metrics.columns
                and f"{c}_indicated_pct" not in _base_intra
            ]
            test_strata = (
                IntravariableImperfection.assign_strata(
                    test_case_rows.select(intra_cols_needed),
                    axis_x, axis_y, med_x, med_y, VITAL_COLS,
                )
                .select(["id", "imperfection_stratum"])
                .rename({"imperfection_stratum": "active_stratum"})
                .drop_nulls("active_stratum")
            )
        else:
            inter_cols_needed = ["id", axis_x, axis_y]
            if "avg_indicated_vars_pct" not in inter_cols_needed:
                inter_cols_needed.append("avg_indicated_vars_pct")
            test_strata = (
                IntervariableImperfection.assign_strata(
                    test_case_rows.select(inter_cols_needed),
                    axis_x, axis_y, med_x, med_y,
                )
                .select(["id", "intervariable_stratum"])
                .rename({"intervariable_stratum": "active_stratum"})
                .drop_nulls("active_stratum")
            )

        # All numeric case-level metrics from the (combined) case_metrics frame —
        # reported as stratum characteristics regardless of which mode drives strata.
        _stratum_meta_cols = {
            "id", "axis_x", "axis_y", "axis_pair_corr",
            "axis_x_median_threshold", "axis_y_median_threshold",
            "active_stratum", "intervariable_stratum", "imperfection_stratum",
        }
        imperfekt_cols = [
            c for c in case_metrics.columns
            if c not in _stratum_meta_cols and case_metrics.schema[c].is_numeric()
        ]
        available_imperfekt = [c for c in imperfekt_cols if c in case_metrics.columns]
        test_strata_imperfekt = test_strata.join(
            case_metrics.select(available_imperfekt + ["id"]),
            on="id",
            how="left",
        )

        strata_arr = (
            test.select("id")
            .join(
                test_strata_imperfekt.select(["id", "active_stratum"]),
                on="id",
                how="left",
            )["active_stratum"]
            .fill_null("")
            .to_numpy()
        )

        imperfekt_lookup: dict[str, dict[str, float]] = {}
        for row in test_strata_imperfekt.iter_rows(named=True):
            sid = row["id"]
            imperfekt_lookup[sid] = {c: row[c] for c in imperfekt_cols if c in row}

        ids_arr = test["id"].to_numpy()

        for stratum_label in np.unique(strata_arr):
            if stratum_label == "":
                continue
            mask = strata_arr == stratum_label
            m_s = _compute_metrics(y_test[mask], y_proba[mask])
            if m_s:
                for imperfekt_metric in imperfekt_cols:
                    vals = [
                        imperfekt_lookup[sid][imperfekt_metric]
                        for sid in ids_arr[mask]
                        if sid in imperfekt_lookup
                        and imperfekt_metric in imperfekt_lookup[sid]
                        and imperfekt_lookup[sid][imperfekt_metric] is not None
                    ]
                    m_s[imperfekt_metric] = float(np.mean(vals)) if vals else float("nan")
                fold_metrics[stratum_label].append(m_s)

        if shap_full:
            try:
                import shap as _shap
                import shap.explainers._tree as _shap_tree
                import shap.explainers.other._ubjson as _shap_ubj
                # XGBoost ≥2.0 stores base_score as '[5E-1]' in UBJ; SHAP 0.49
                # calls float() on it directly and crashes. Patch the UBJ decoder
                # in-place to strip brackets before the float conversion.
                if not getattr(_shap_ubj, "_bracket_fix_applied", False):
                    _orig_decode = _shap_ubj.decode_ubjson_buffer
                    def _fixed_decode(fd, _orig=_orig_decode):
                        result = _orig(fd)
                        try:
                            lmp = result["learner"]["learner_model_param"]
                            bs = lmp.get("base_score", "")
                            if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                                lmp["base_score"] = bs[1:-1]
                        except Exception:
                            pass
                        return result
                    _shap_ubj.decode_ubjson_buffer = _fixed_decode
                    _shap_tree.decode_ubjson_buffer = _fixed_decode
                    _shap_ubj._bracket_fix_applied = True

                # Stratified subsample of test rows to explain: mean |SHAP| per
                # feature is a row average, so a per-stratum-capped subsample is
                # an unbiased estimate at a fraction of the TreeExplainer cost.
                _sub_idx = _stratified_shap_subsample(
                    strata_arr,
                    SHAP_MAX_ROWS_PER_STRATUM,
                    SHAP_SUBSAMPLE_RANDOM_STATE + fold_idx,
                )
                _X_shap = X_test[_sub_idx]
                _strata_shap = strata_arr[_sub_idx]

                _explainer = _shap.TreeExplainer(model.model.get_booster())
                _shap_vals = _explainer.shap_values(_X_shap)  # (n_sub, n_features)
                fold_shap_abs["overall"].append(np.abs(_shap_vals).mean(axis=0))
                for _sl in np.unique(_strata_shap):
                    if _sl == "":
                        continue
                    _mask = _strata_shap == _sl
                    if _mask.sum() > 0:
                        fold_shap_abs[_sl].append(np.abs(_shap_vals[_mask]).mean(axis=0))
                if shap_interactions and fold_idx == total_folds - 1:
                    _ivals = _explainer.shap_interaction_values(_X_shap)  # (n_sub, n_feat, n_feat)
                    last_shap_interactions["overall"] = _ivals.mean(axis=0)
                    for _sl in np.unique(_strata_shap):
                        if _sl == "":
                            continue
                        _mask = _strata_shap == _sl
                        if _mask.sum() > 0:
                            last_shap_interactions[_sl] = _ivals[_mask].mean(axis=0)
                    last_X_test_raw = _X_shap
                    last_y_test_raw = y_test[_sub_idx]
            except Exception as _e:
                print(f"  [{pipeline_name}] SHAP skipped (fold {fold_idx + 1}): {_e}")

        if features_save_path is not None:
            all_X_test_chunks.append(X_test)
            all_y_test_chunks.append(y_test)
            all_strata_chunks.append(strata_arr)

        last_model = model
        last_test_df = test
        last_X_test = X_test
        last_test_strata = test_strata

    print()

    if shap_full and shap_save_path is not None and fold_shap_abs:
        shap_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, np.ndarray] = {
            "feature_names": np.array(feature_cols),
        }
        for _sl, _arrs in fold_shap_abs.items():
            save_dict[f"shap_{_sl}"] = np.stack(_arrs)
        for _sl, _mat in last_shap_interactions.items():
            save_dict[f"interact_{_sl}"] = _mat
        if last_X_test_raw is not None:
            save_dict["last_X_test_raw"] = last_X_test_raw
        if last_y_test_raw is not None:
            save_dict["last_y_test_raw"] = last_y_test_raw
        np.savez_compressed(shap_save_path, **save_dict)
        print(f"  [{pipeline_name}] SHAP saved to {shap_save_path}")

    if features_save_path is not None and all_X_test_chunks:
        features_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            features_save_path,
            feature_names=np.array(feature_cols),
            X_test_all=np.concatenate(all_X_test_chunks, axis=0),
            y_test_all=np.concatenate(all_y_test_chunks, axis=0),
            strata_all=np.concatenate(all_strata_chunks, axis=0),
        )
        print(f"  [{pipeline_name}] Feature values saved to {features_save_path}")

    return dict(fold_metrics), last_model, last_X_test, last_test_df, feature_cols, last_test_strata


def summarise_cv(fold_metrics: dict[str, list], pipeline_name: str) -> dict[str, dict]:
    """
    Aggregate per-fold metric lists into mean ± 95% CI (t-distribution).
    Returns a dict keyed by stratum label (plus "overall").
    """
    from scipy import stats

    _CORE_METRICS = (
        "auprc", "auprc_lift", "auroc", "brier_skill_score", "n_pos_pct",
    )
    summary = {}
    for key, folds in fold_metrics.items():
        all_metrics = {m for f in folds for m in f}
        ordered_metrics = [m for m in _CORE_METRICS if m in all_metrics] + sorted(
            m for m in all_metrics if m not in _CORE_METRICS
        )
        for metric in ordered_metrics:
            vals = np.array([f[metric] for f in folds if metric in f and not np.isnan(f[metric])])
            if len(vals) == 0:
                continue
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ci = stats.t.ppf(0.975, df=len(vals) - 1) * se
            summary.setdefault(key, {})[metric] = {"mean": mean, "ci": ci}

    o = summary.get("overall", {})
    print(
        f"\n[{pipeline_name}] overall  "
        + "  ".join(f"{m.upper()}={v['mean']:.3f}±{v['ci']:.3f}" for m, v in o.items())
    )
    for s in sorted(k for k in summary if k != "overall"):
        vals = summary[s]
        print(
            f"  {s}  "
            + "  ".join(f"{m.upper()}={v['mean']:.3f}±{v['ci']:.3f}" for m, v in vals.items())
        )
    return summary


def save_cv_results(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
) -> None:
    """
    Write a tidy CSV with one row per pipeline × stratum, columns:
      pipeline, run_timestamp, stratum, auprc_mean, auprc_ci, auprc_lift_mean, auprc_lift_ci,
      auroc_mean, auroc_ci, brier_skill_score_mean, brier_skill_score_ci,
      n_pos_pct_mean, n_pos_pct_ci, avg_indicated_vars_pct_mean, …
    """
    _CORE_METRICS = ("auprc", "auprc_lift", "auroc", "brier_skill_score", "n_pos_pct")

    # Collect all metric keys across all summaries so every column is present.
    all_metric_keys: set[str] = set()
    for _, summary in pipeline_summaries:
        for s, s_dict in summary.items():
            if not s.startswith("_"):
                all_metric_keys.update(s_dict.keys())
    metrics = [m for m in _CORE_METRICS if m in all_metric_keys] + sorted(
        m for m in all_metric_keys if m not in _CORE_METRICS
    )

    rows = []
    for pipeline_name, summary in pipeline_summaries:
        run_ts = summary.get("_run_timestamp", "")
        strata = [s for s in _STRATUM_ORDER if s in summary] + sorted(
            s for s in summary if s not in _STRATUM_ORDER and not s.startswith("_")
        )
        for stratum in strata:
            row: dict = {"pipeline": pipeline_name, "run_timestamp": run_ts, "stratum": stratum}
            for m in metrics:
                v = summary.get(stratum, {}).get(m)
                row[f"{m}_mean"] = v["mean"] if v else float("nan")
                row[f"{m}_ci"] = v["ci"] if v else float("nan")
            rows.append(row)

    df = pl.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(save_path)
    print(f"CV results saved to {save_path}")


def load_cv_results(
    load_path: Path,
) -> list[tuple[str, dict[str, dict]]]:
    """
    Reconstruct pipeline_summaries from a CSV written by save_cv_results.

    Returns a list of (pipeline_name, summary) tuples in the same format
    consumed by plot_auprc_by_stratum / plot_auroc_by_stratum /
    plot_auprc_lift_by_stratum, so plots can be rebuilt from disk without
    re-running CV.

    _run_timestamp (if present in the CSV) is restored as summary["_run_timestamp"].
    """
    df = pl.read_csv(load_path)

    metric_cols = [
        c[: -len("_mean")]
        for c in df.columns
        if c.endswith("_mean") and c not in ("pipeline", "run_timestamp", "stratum")
    ]

    pipelines: dict[str, dict] = {}
    pipeline_order: list[str] = []

    for row in df.iter_rows(named=True):
        name = row["pipeline"]
        if name not in pipelines:
            pipelines[name] = {}
            pipeline_order.append(name)
            ts = row.get("run_timestamp")
            if ts:
                pipelines[name]["_run_timestamp"] = ts

        stratum = row["stratum"]
        pipelines[name][stratum] = {}
        for m in metric_cols:
            mean_val = row.get(f"{m}_mean")
            ci_val = row.get(f"{m}_ci")
            if mean_val is not None and not (isinstance(mean_val, float) and mean_val != mean_val):
                pipelines[name][stratum][m] = {"mean": mean_val, "ci": ci_val}

    return [(name, pipelines[name]) for name in pipeline_order]


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
        stay_df.select(["label"] + feature_cols)
        .unpivot(
            index=["label"],
            on=feature_cols,
            variable_name="feature",
            value_name="value",
        )
        .drop_nulls("value")
    )
    if long_df.height == 0:
        print("Skipping feature-by-outcome table: all selected feature values are null.")
        return

    grouped = long_df.group_by(["label", "feature"]).agg(
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

        outcome_val = row["label"]
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


def compute_feature_distribution_by_quadrant(
    features_npz_path: Path,
    save_path: Path,
    npz_data=None,
) -> pl.DataFrame | None:
    """Compute per-stratum feature distribution table from a features .npz file.

    The .npz must contain feature_names, X_test_all, y_test_all, strata_all
    (as saved by run_cv when features_save_path is provided).

    For each (feature, stratum) pair computes mean ± 95% CI across all held-out
    test rows in that stratum (pooled across folds), plus outcome_prevalence_mean
    / outcome_prevalence_ci columns consumed by plot_results.py's wide table.

    Returns the tidy DataFrame and writes it as CSV to save_path.
    Returns None if the .npz is missing or lacks required keys.
    ``npz_data`` may be a pre-loaded NpzFile to avoid re-reading the file.
    """
    from scipy import stats

    if npz_data is None:
        if not features_npz_path.exists():
            print(f"compute_feature_distribution_by_quadrant: {features_npz_path} not found")
            return None
        data = np.load(features_npz_path, allow_pickle=True)
    else:
        data = npz_data
    for key in ("feature_names", "X_test_all", "y_test_all", "strata_all"):
        if key not in data.files:
            print(f"compute_feature_distribution_by_quadrant: key {key!r} missing in {features_npz_path}")
            return None

    feature_names: list[str] = data["feature_names"].tolist()
    X_all: np.ndarray = data["X_test_all"].astype(np.float64)
    y_all: np.ndarray = data["y_test_all"].astype(np.int8)
    strata_all: np.ndarray = data["strata_all"]

    unique_strata = [s for s in np.unique(strata_all) if s != ""]

    rows: list[dict] = []
    for stratum in unique_strata:
        mask = strata_all == stratum
        X_s = X_all[mask]
        y_s = y_all[mask]
        n_s = mask.sum()

        # outcome prevalence for this stratum
        if n_s > 1:
            prev_mean = float(y_s.mean())
            prev_se = float(np.std(y_s, ddof=1)) / np.sqrt(n_s)
            prev_ci = float(stats.t.ppf(0.975, df=n_s - 1) * prev_se)
        else:
            prev_mean = float(y_s.mean()) if n_s == 1 else float("nan")
            prev_ci = float("nan")

        for fi, feat in enumerate(feature_names):
            vals = X_s[:, fi]
            valid = vals[np.isfinite(vals)]
            n = len(valid)
            if n == 0:
                rows.append({
                    "stratum": stratum,
                    "feature": feat,
                    "feature_group": _feature_group(feat),
                    "n": 0,
                    "mean": float("nan"),
                    "ci": float("nan"),
                    "outcome_prevalence_mean": prev_mean,
                    "outcome_prevalence_ci": prev_ci,
                })
                continue
            mean = float(valid.mean())
            se = float(valid.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
            ci = float(stats.t.ppf(0.975, df=n - 1) * se) if n > 1 else float("nan")
            rows.append({
                "stratum": stratum,
                "feature": feat,
                "feature_group": _feature_group(feat),
                "n": n,
                "mean": mean,
                "ci": ci,
                "outcome_prevalence_mean": prev_mean,
                "outcome_prevalence_ci": prev_ci,
            })

    if not rows:
        print("compute_feature_distribution_by_quadrant: no rows produced")
        return None

    out_df = pl.DataFrame(rows).sort(["stratum", "feature"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(save_path)
    print(f"Feature distribution by quadrant saved to {save_path}")
    return out_df


def compute_feature_distribution_by_outcome(
    features_npz_path: Path,
    save_path: Path,
    npz_data=None,
) -> pl.DataFrame | None:
    """Compute per-outcome feature distribution table from a features .npz file.

    The .npz must contain feature_names, X_test_all, y_test_all
    (as saved by run_cv when features_save_path is provided).

    For each (feature, outcome) pair computes mean ± 95% CI across all held-out
    test rows pooled across folds, plus median/quantile summaries.

    Returns the tidy DataFrame and writes it as CSV to save_path.
    Returns None if the .npz is missing or lacks required keys.
    ``npz_data`` may be a pre-loaded NpzFile to avoid re-reading the file.
    """
    from scipy import stats

    if npz_data is None:
        if not features_npz_path.exists():
            print(f"compute_feature_distribution_by_outcome: {features_npz_path} not found")
            return None
        data = np.load(features_npz_path, allow_pickle=True)
    else:
        data = npz_data
    for key in ("feature_names", "X_test_all", "y_test_all"):
        if key not in data.files:
            print(f"compute_feature_distribution_by_outcome: key {key!r} missing in {features_npz_path}")
            return None

    feature_names: list[str] = data["feature_names"].tolist()
    X_all: np.ndarray = data["X_test_all"].astype(np.float64)
    y_all: np.ndarray = data["y_test_all"].astype(np.int8)

    rows: list[dict] = []
    for outcome_val in np.unique(y_all):
        mask = y_all == outcome_val
        X_o = X_all[mask]

        for fi, feat in enumerate(feature_names):
            vals = X_o[:, fi]
            valid = vals[np.isfinite(vals)]
            n = len(valid)
            if n == 0:
                rows.append({
                    "feature": feat,
                    "feature_group": _feature_group(feat),
                    "outcome": int(outcome_val),
                    "n_non_null": 0,
                    "mean": float("nan"),
                    "ci": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "q25": float("nan"),
                    "q75": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                })
                continue
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if n > 1 else float("nan")
            se = std / np.sqrt(n) if n > 1 else float("nan")
            ci = float(stats.t.ppf(0.975, df=n - 1) * se) if n > 1 else float("nan")
            rows.append({
                "feature": feat,
                "feature_group": _feature_group(feat),
                "outcome": int(outcome_val),
                "n_non_null": n,
                "mean": mean,
                "ci": ci,
                "std": std,
                "median": float(np.median(valid)),
                "q25": float(np.quantile(valid, 0.25)),
                "q75": float(np.quantile(valid, 0.75)),
                "min": float(valid.min()),
                "max": float(valid.max()),
            })

    if not rows:
        print("compute_feature_distribution_by_outcome: no rows produced")
        return None

    out_df = pl.DataFrame(rows).sort(["feature", "outcome"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(save_path)
    print(f"Feature distribution by outcome saved to {save_path}")
    return out_df


def print_information_gain_ratio(
    baseline_summary: dict[str, dict],
    candidate_summary: dict[str, dict],
    baseline_name: str,
    candidate_name: str,
    metric: str = "auprc",
    test_set_name: str = "overall",
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


def summarise_shap(npz_path: Path) -> pl.DataFrame:
    """Read a SHAP .npz and return mean ± std of mean |SHAP| per stratum × feature."""
    data = np.load(npz_path, allow_pickle=True)
    feature_names: list[str] = data["feature_names"].tolist()
    rows: list[dict] = []
    for key in data.files:
        if not key.startswith("shap_"):
            continue
        stratum = key[len("shap_"):]
        mat = data[key]  # (n_folds, n_features)
        for i, feat in enumerate(feature_names):
            col = mat[:, i]
            rows.append(
                {
                    "stratum": stratum,
                    "feature": feat,
                    "feature_group": feature_group(feat),
                    "mean_abs_shap_mean": float(col.mean()),
                    "mean_abs_shap_std": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                }
            )
    return pl.DataFrame(rows)


def save_shap_importance_csv(
    npz_paths: list[tuple[str, Path]],
    save_path: Path,
) -> None:
    """Merge per-pipeline SHAP summaries into a tidy CSV.

    Columns: pipeline, stratum, feature, feature_group, mean_abs_shap_mean, mean_abs_shap_std
    """
    frames = []
    for pipeline_name, path in npz_paths:
        if not path.exists():
            print(f"  SHAP .npz not found, skipping: {path}")
            continue
        df = summarise_shap(path).with_columns(pl.lit(pipeline_name).alias("pipeline"))
        frames.append(df)
    if not frames:
        print("save_shap_importance_csv: no valid .npz files found.")
        return
    out = pl.concat(frames).select(
        ["pipeline", "stratum", "feature", "feature_group", "mean_abs_shap_mean", "mean_abs_shap_std"]
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(save_path)
    print(f"SHAP importance CSV saved to {save_path}")