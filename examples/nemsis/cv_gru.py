"""GRU cross-validation loop for the NEMSIS/MC-MED sequence experiments.

Mirrors the structure of cv.py's run_cv() but operates on the raw long-format
cohort DataFrame rather than stay-level aggregated features. Each fold converts
train/test subsets to (N, T, D) 3D arrays via df_to_3d_array, then trains a
GRUModel. SHAP is computed via shap.GradientExplainer when shap_full=True,
producing per-feature value importance, per-feature mask-channel importance
(when use_mask=True), and a per-feature temporal importance profile for both
the value and mask channels — all saved as .npz.

Shared utilities (_compute_metrics, prior_correct_probs, _stratified_shap_subsample)
are imported directly from cv.py to avoid drift.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
from imperfekt.analysis.intravariable.intravariable import IntravariableImperfection
from sklearn.model_selection import RepeatedStratifiedKFold

from config import (
    APPLY_PRIOR_CORRECTION,
    APPLY_UNDERSAMPLING,
    CV_N_REPEATS,
    CV_N_SPLITS,
    RANDOM_STATE,
    SHAP_MAX_ROWS_PER_STRATUM,
    SHAP_SUBSAMPLE_RANDOM_STATE,
    TRAIN_NEG_POS_RATIO,
    UNDERSAMPLE_RANDOM_STATE,
    VITAL_COLS,
)
from cv import (
    _compute_metrics,
    _stratified_shap_subsample,
    prior_correct_probs,
)
from examples.utils.gru_model import GRUModel
from seq_array import df_to_3d_array


class _GRUShapWrapper(nn.Module):
    """Single-tensor wrapper around _GRUNet for shap.GradientExplainer.

    GradientExplainer requires a callable f(tensor) -> tensor. The GRU's
    forward() needs lengths derived from the input, so we recompute them here.
    Works for both use_mask=True (D_input=2D, mask in last D cols) and
    use_mask=False (D_input=D, nonzero treated as observed).

    When the model was trained with static features (static_size > 0) the
    fc layer expects hidden_size + static_size inputs. We hold the static
    tensor fixed (background mean) so GradientExplainer only attributes over
    the sequence input — this avoids the shape mismatch while keeping the
    forward pass valid.
    """

    def __init__(self, net: nn.Module, D: int, use_mask: bool, static: torch.Tensor | None = None):
        super().__init__()
        self.net = net
        self.D = D
        self.use_mask = use_mask
        # Held-out static features (background mean); None when static_size == 0.
        if static is not None:
            self.register_buffer("static", static)
        else:
            self.static = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if self.use_mask:
            any_obs = x[:, :, self.D:].any(dim=2)  # (N, T)
        else:
            any_obs = (x != 0).any(dim=2)           # (N, T)
        flipped = any_obs.flip(dims=[1])
        last_from_end = flipped.long().argmax(dim=1)  # (N,)
        lengths = (T - last_from_end).clamp(min=1)
        static = self.static.expand(x.shape[0], -1) if self.static is not None else None
        return self.net(x, lengths, static).unsqueeze(1)  # (N, 1) — GradientExplainer requires 2D output


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
    stratification_mode: str = "intervariable",
    shap_save_path: Path | None = None,
    shap_full: bool = True,
    features_save_path: Path | None = None,
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
        cohort_df:           Long-format DataFrame (id, clock, vitals..., label) after
                             plausibility filtering and imputation by ConfigBuilder.
        case_metrics:        Combined per-case metrics from combine_case_metrics()
                             (intervariable + intravariable joined on id).
        axes:                (axis_x_name, axis_y_name) for the active stratification mode.
        pipeline_name:       Label used in progress output and result keys.
        use_mask:            When True the GRU input includes the binary observed-mask
                             channel; when False it is values-only (mask ablation).
                             Comparing the two per stratum shows where the missingness
                             indicator adds predictive information.
        stratification_mode: "intervariable" (co-missingness structure) or
                             "intravariable" (per-variable missingness burden). Controls
                             which axis space and assign_strata implementation are used.

    Returns:
        fold_metrics     - dict mapping "overall" and each stratum label to a
                           list of metric dicts, one per fold.
        last_model       - trained GRUModel from the final fold.
        None             - placeholder for SHAP compatibility with run_cv signature.
        last_test_df     - exact test DataFrame (long-format) from the final fold.
        feature_cols     - VITAL_COLS used as sequence features.
        last_test_strata - id/active_stratum for the final fold's test set.
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

    _stratum_meta_cols = {
        "id", "axis_x", "axis_y", "axis_pair_corr",
        "axis_x_median_threshold", "axis_y_median_threshold",
        "active_stratum", "intervariable_stratum", "imperfection_stratum",
    }
    imperfekt_cols = [
        c for c in case_metrics.columns
        if c not in _stratum_meta_cols and case_metrics.schema[c].is_numeric()
    ]

    fold_metrics: dict[str, list] = defaultdict(list)
    fold_shap_abs: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_shap_mask_abs: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_shap_time_feat_abs: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_shap_time_feat_mask_abs: dict[str, list[np.ndarray]] = defaultdict(list)
    all_X_test_chunks: list[np.ndarray] = []
    all_y_test_chunks: list[np.ndarray] = []
    all_strata_chunks: list[np.ndarray] = []
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
        # Build processed train array now (scaler already fitted); used as SHAP background.
        X_train_combined, _ = model._build_input(X_train_us.astype(np.float32))
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

        test_case_rows = case_metrics.filter(pl.col("id").is_in(list(test_subjects)))

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

        if features_save_path is not None:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                X_test_agg = np.concatenate(
                    [
                        np.nanmean(X_test_3d, axis=1),
                        np.nanmin(X_test_3d, axis=1),
                        np.nanmax(X_test_3d, axis=1),
                    ],
                    axis=1,
                )  # (N, D*3): per-stay mean/min/max over observed timesteps
            all_X_test_chunks.append(X_test_agg)
            all_y_test_chunks.append(y_test)
            all_strata_chunks.append(strata_arr)

        ids_arr = np.asarray(test_ids)
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
                D = len(VITAL_COLS)
                X_test_combined, _ = model._build_input(X_test_3d.astype(np.float32))
                _sub_idx = _stratified_shap_subsample(
                    strata_arr, SHAP_MAX_ROWS_PER_STRATUM, SHAP_SUBSAMPLE_RANDOM_STATE + fold_idx
                )
                X_shap_np = X_test_combined[_sub_idx]   # (N_sub, T, D_input)
                _strata_shap = strata_arr[_sub_idx]

                rng_bg = np.random.default_rng(SHAP_SUBSAMPLE_RANDOM_STATE + fold_idx)
                bg_idx = rng_bg.choice(
                    len(X_train_combined), size=min(100, len(X_train_combined)), replace=False
                )
                bg_tensor = torch.tensor(
                    X_train_combined[bg_idx], dtype=torch.float32, device=model.device
                )
                X_shap_tensor = torch.tensor(
                    X_shap_np, dtype=torch.float32, device=model.device
                )

                # When a static head is present, hold static at its training
                # mean so GradientExplainer only attributes over sequence dims.
                if model._static_mean is not None:
                    _static_bg = torch.tensor(
                        model._static_mean[None, :], dtype=torch.float32, device=model.device
                    )
                else:
                    _static_bg = None
                wrapper = _GRUShapWrapper(model.model, D, use_mask, _static_bg).to(model.device)
                wrapper.eval()
                with torch.backends.cudnn.flags(enabled=False):
                    explainer = _shap.GradientExplainer(wrapper, bg_tensor)
                    # shap_values returns (N_sub, T, D_input, 1) because the wrapper
                    # outputs (N, 1); [..., 0] drops the trailing output dimension.
                    shap_vals = np.array(explainer.shap_values(X_shap_tensor))[..., 0]  # (N_sub, T, D_input)
                abs_shap = np.abs(shap_vals)
                D_in = abs_shap.shape[2]

                def _agg(arr: np.ndarray):
                    val_imp = arr[:, :, :D].mean(axis=(0, 1))                          # (D,)
                    mask_imp = arr[:, :, D:].mean(axis=(0, 1)) if D_in > D else None   # (D,) or None
                    time_feat_imp = arr[:, :, :D].mean(axis=0)                          # (T, D)
                    time_feat_mask_imp = arr[:, :, D:].mean(axis=0) if D_in > D else None  # (T, D) or None
                    return val_imp, mask_imp, time_feat_imp, time_feat_mask_imp

                v, mk, ti, ti_mk = _agg(abs_shap)
                fold_shap_abs["overall"].append(v)
                if mk is not None:
                    fold_shap_mask_abs["overall"].append(mk)
                fold_shap_time_feat_abs["overall"].append(ti)
                if ti_mk is not None:
                    fold_shap_time_feat_mask_abs["overall"].append(ti_mk)

                for _sl in np.unique(_strata_shap):
                    if _sl == "":
                        continue
                    _sl_mask = _strata_shap == _sl
                    if _sl_mask.sum() > 0:
                        v_s, mk_s, ti_s, ti_mk_s = _agg(abs_shap[_sl_mask])
                        fold_shap_abs[_sl].append(v_s)
                        if mk_s is not None:
                            fold_shap_mask_abs[_sl].append(mk_s)
                        fold_shap_time_feat_abs[_sl].append(ti_s)
                        if ti_mk_s is not None:
                            fold_shap_time_feat_mask_abs[_sl].append(ti_mk_s)
            except Exception as _e:
                print(f"  [{pipeline_name}] GRU SHAP skipped (fold {fold_idx + 1}): {_e}")

        last_model = model
        last_test_df = test_df
        last_test_strata = test_strata

    print()

    if shap_full and shap_save_path is not None and fold_shap_abs:
        shap_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, np.ndarray] = {"feature_names": np.array(list(VITAL_COLS))}
        for _sl, _arrs in fold_shap_abs.items():
            save_dict[f"shap_{_sl}"] = np.stack(_arrs)
        for _sl, _arrs in fold_shap_mask_abs.items():
            save_dict[f"shap_mask_{_sl}"] = np.stack(_arrs)
        for _sl, _arrs in fold_shap_time_feat_abs.items():
            save_dict[f"shap_time_feat_{_sl}"] = np.stack(_arrs)          # (n_folds, T, D)
        for _sl, _arrs in fold_shap_time_feat_mask_abs.items():
            save_dict[f"shap_time_feat_mask_{_sl}"] = np.stack(_arrs)     # (n_folds, T, D)
        np.savez_compressed(shap_save_path, **save_dict)
        print(f"  [{pipeline_name}] GRU SHAP saved to {shap_save_path}")

    if features_save_path is not None and all_X_test_chunks:
        agg_feature_names = [f"{c}_{stat}" for stat in ("mean", "min", "max") for c in VITAL_COLS]
        features_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            features_save_path,
            feature_names=np.array(agg_feature_names),
            X_test_all=np.concatenate(all_X_test_chunks, axis=0),
            y_test_all=np.concatenate(all_y_test_chunks, axis=0),
            strata_all=np.concatenate(all_strata_chunks, axis=0),
        )
        print(f"  [{pipeline_name}] Feature values saved to {features_save_path}")

    return dict(fold_metrics), last_model, None, last_test_df, list(VITAL_COLS), last_test_strata
