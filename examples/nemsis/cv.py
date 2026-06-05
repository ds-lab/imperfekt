import json
from pathlib import Path

from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
import polars as pl

import numpy as np
import polars as pl
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
    data_fingerprint,
    data_fingerprint_tag,
)
from examples.nemsis.features import feature_group, is_structural_feature
from examples.utils.models import XGBoostModel  # noqa: E402

_STRATA_CACHE_ROOT = RESULTS_DIR / "intervariable_strata_cache"
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


def run_cv(
    stay_df: pl.DataFrame,
    case_metrics: pl.DataFrame,
    axes: tuple[str, str],
    pipeline_name: str,
    feature_distribution_save_path: Path | None = None,
    shap_save_path: Path | None = None,
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

    Irregularity quadrant thresholds (Q_alpha/Q_beta/Q_gamma/Q_delta) are derived strictly from
    the *train* fold on each iteration, then applied to test stays, so no
    test-set information leaks into the stratum evaluation boundaries.

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
      last_test_strata - id/intervariable_stratum for the final fold's
                         test set (train-derived thresholds)
    """
    feature_cols = [c for c in stay_df.columns if c not in ("id", "label")]
    collect_feature_dist = feature_distribution_save_path is not None
    fold_feature_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    fold_outcome_prevalence: dict[str, list[float]] = defaultdict(list)
    fold_shap_abs: dict[str, list[np.ndarray]] = defaultdict(list)
    last_shap_interactions: dict[str, np.ndarray] = {}
    last_X_test_raw: np.ndarray | None = None
    last_y_test_raw: np.ndarray | None = None

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
        strata_input_cols = ["id", axis_x, axis_y]
        if "avg_indicated_vars_pct" not in strata_input_cols:
            strata_input_cols.append("avg_indicated_vars_pct")
        # Do NOT drop null-axis rows before assign_strata: Q_complete cases (no
        # missingness) have null co-missingness/entropy axes by construction, and
        # assign_strata labels them Q_complete *before* its null-axis check. We
        # instead drop unassignable rows (null intervariable_stratum) afterwards,
        # which keeps Q_complete while still excluding genuinely null-axis cases.
        test_strata = (
            IntervariableImperfection.assign_strata(
                case_metrics.filter(pl.col("id").is_in(test_ids.to_list())).select(
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

        # Join test strata onto test rows to get a per-row stratum label aligned
        # with y_test/y_proba, then group by stratum without repeated is_in scans.
        # Also join cv/qcod/adherence_rate for irregularity characterisation per stratum.
        imperfekt_cols = [
            "avg_indicated_vars_pct",
            "co_missingness_concentration",
            "missing_variable_breadth",
            "pattern_entropy",
            "max_pairwise_co_missingness",
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

        if collect_feature_dist:
            overall_prevalence = test["label"].mean()
            if overall_prevalence is not None and not np.isnan(overall_prevalence):
                fold_outcome_prevalence["overall"].append(float(overall_prevalence))

            overall_means = (
                test.select(feature_cols)
                .unpivot(
                    on=feature_cols,
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
                test.select(["id", "label"] + feature_cols)
                .join(test_strata, on="id", how="left")
                .drop_nulls("intervariable_stratum")
            )
            if test_with_strata.height > 0:
                stratum_prevalence = test_with_strata.group_by("intervariable_stratum").agg(
                    pl.col("label").mean().alias("fold_prevalence")
                )
                for row in stratum_prevalence.iter_rows(named=True):
                    prev = row["fold_prevalence"]
                    if prev is None or np.isnan(prev):
                        continue
                    fold_outcome_prevalence[row["intervariable_stratum"]].append(float(prev))

                stratum_means = (
                    test_with_strata.unpivot(
                        index=["intervariable_stratum"],
                        on=feature_cols,
                        variable_name="feature",
                        value_name="value",
                    )
                    .drop_nulls("value")
                    .group_by(["intervariable_stratum", "feature"])
                    .agg(pl.col("value").mean().alias("fold_mean"))
                )
                for row in stratum_means.iter_rows(named=True):
                    fold_feature_means[(row["intervariable_stratum"], row["feature"])].append(
                        float(row["fold_mean"])
                    )

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

                _explainer = _shap.TreeExplainer(model.model.get_booster())
                _shap_vals = _explainer.shap_values(X_test)  # (n_test, n_features)
                fold_shap_abs["overall"].append(np.abs(_shap_vals).mean(axis=0))
                for _sl in np.unique(strata_arr):
                    if _sl == "":
                        continue
                    _mask = strata_arr == _sl
                    if _mask.sum() > 0:
                        fold_shap_abs[_sl].append(np.abs(_shap_vals[_mask]).mean(axis=0))
                if shap_interactions and fold_idx == total_folds - 1:
                    _ivals = _explainer.shap_interaction_values(X_test)  # (n_test, n_feat, n_feat)
                    last_shap_interactions["overall"] = _ivals.mean(axis=0)
                    for _sl in np.unique(strata_arr):
                        if _sl == "":
                            continue
                        _mask = strata_arr == _sl
                        if _mask.sum() > 0:
                            last_shap_interactions[_sl] = _ivals[_mask].mean(axis=0)
                    last_X_test_raw = X_test
                    last_y_test_raw = y_test
            except Exception as _e:
                print(f"  [{pipeline_name}] SHAP skipped (fold {fold_idx + 1}): {_e}")

        last_model = model
        last_test_df = test
        last_X_test = X_test
        last_test_strata = test_strata

    print()

    if collect_feature_dist:
        _write_feature_distribution_by_quadrant(
            fold_feature_means,
            fold_outcome_prevalence,
            feature_distribution_save_path,
        )

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

    return dict(fold_metrics), last_model, last_X_test, last_test_df, feature_cols, last_test_strata


def summarise_cv(fold_metrics: dict[str, list], pipeline_name: str) -> dict[str, dict]:
    """
    Aggregate per-fold metric lists into mean ± 95% CI (t-distribution).
    Returns a dict keyed by stratum label (plus "overall").
    """
    from scipy import stats

    summary = {}
    for key, folds in fold_metrics.items():
        for metric in (
            "auprc",
            "auprc_lift",
            "auroc",
            "brier_skill_score",
            "n_pos_pct",
            "avg_indicated_vars_pct",
            "co_missingness_concentration",
            "missing_variable_breadth",
            "pattern_entropy",
            "max_pairwise_co_missingness",
        ):
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
    metrics = (
        "auprc",
        "auprc_lift",
        "auroc",
        "brier_skill_score",
        "n_pos_pct",
        "avg_indicated_vars_pct",
        "co_missingness_concentration",
        "missing_variable_breadth",
        "pattern_entropy",
        "max_pairwise_co_missingness",
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


def _write_feature_distribution_by_quadrant(
    fold_feature_means: dict[tuple[str, str], list[float]],
    fold_outcome_prevalence: dict[str, list[float]],
    save_path: Path,
) -> None:
    """
    Aggregate per-fold feature means and outcome prevalence (collected during
    run_cv) into a tidy CSV of mean ± 95% CI per (feature, stratum).
    """
    from scipy import stats

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