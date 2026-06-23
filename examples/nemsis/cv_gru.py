"""GRU cross-validation loop for the NEMSIS/MC-MED sequence experiments.

Mirrors the structure of cv.py's run_cv() but operates on the raw long-format
cohort DataFrame rather than stay-level aggregated features. Each fold converts
train/test subsets to (N, T, D) 3D arrays via df_to_3d_array, then trains a
GRUModel. SHAP is computed via shap.GradientExplainer when shap_full=True,
producing per-feature value importance, per-feature mask-channel importance
(when use_mask=True), and a per-feature temporal importance profile for both
the value and mask channels — all saved as .npz.

Shared utilities (_compute_metrics, prior_correct_probs,
_stratified_shap_subsample, _assign_fold_strata, _accumulate_stratum_metrics)
are imported directly from cv.py to avoid drift. Like cv.run_cv, run_cv_gru
trains once per fold and evaluates all stratification modes from that single
pass.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
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
    _accumulate_stratum_metrics,
    _assign_fold_strata,
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


def _make_fold_gru_explainer(
    model: GRUModel,
    X_train_combined: np.ndarray,
    X_test_3d: np.ndarray,
    use_mask: bool,
    D: int,
    fold_idx: int,
    pipeline_name: str,
):
    """Build a SHAP GradientExplainer for one fold's GRU (or None on failure).

    Returns ``(explainer, X_test_combined)`` where ``X_test_combined`` is the
    fold's processed test input (N, T, D_input) that callers subsample per mode.
    Both are mode-independent, so building them once per fold lets every
    SHAP-emitting mode share the (expensive) explainer. Returns ``(None, None)``
    if SHAP setup fails.
    """
    try:
        import shap as _shap
        X_test_combined, _ = model._build_input(X_test_3d.astype(np.float32))

        rng_bg = np.random.default_rng(SHAP_SUBSAMPLE_RANDOM_STATE + fold_idx)
        bg_idx = rng_bg.choice(
            len(X_train_combined), size=min(100, len(X_train_combined)), replace=False
        )
        bg_tensor = torch.tensor(
            X_train_combined[bg_idx], dtype=torch.float32, device=model.device
        )

        # When a static head is present, hold static at its training mean so
        # GradientExplainer only attributes over sequence dims.
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
        return explainer, X_test_combined
    except Exception as _e:
        print(f"  [{pipeline_name}] GRU SHAP skipped (fold {fold_idx + 1}): {_e}")
        return None, None


def run_cv_gru(
    cohort_df: pl.DataFrame,
    case_metrics: pl.DataFrame,
    modes: dict[str, tuple[str, str]],
    pipeline_name: str,
    use_mask: bool = True,
    shap_save_paths: dict[str, Path] | None = None,
    shap_full: bool = True,
    features_save_paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, list]]:
    """Repeated stratified k-fold CV using a GRU sequence model, over 1+ modes.

    Mirrors :func:`cv.run_cv`: the GRU is trained once per fold and reused across
    every stratification mode in ``modes`` — the modes only differ in how the
    held-out test stays are bucketed into strata — so evaluating N modes costs
    ~1× a single pass rather than N×. SHAP (GradientExplainer) is likewise
    computed once per fold and re-bucketed per mode. Pass a single-entry
    ``modes`` for the ordinary one-mode run.

    Args:
        cohort_df:           Long-format DataFrame (id, clock, vitals..., label) after
                             plausibility filtering and imputation by ConfigBuilder.
        case_metrics:        Combined per-case metrics from combine_case_metrics()
                             (intervariable + intravariable joined on id).
        modes:               mode name (e.g. "intervariable") → its (axis_x, axis_y)
                             pair. Drives which axis space buckets strata.
        pipeline_name:       Label used in progress output and result keys.
        use_mask:            When True the GRU input includes the binary observed-mask
                             channel; when False it is values-only (mask ablation).
                             Comparing the two per stratum shows where the missingness
                             indicator adds predictive information.
        shap_save_paths:     mode name → SHAP .npz output path (only modes present
                             are written).
        features_save_paths: mode name → features .npz output path (only modes
                             present are written).

    Returns a dict mapping each mode name to that mode's ``fold_metrics`` dict
    (mapping "overall" and each stratum label to a list of per-fold metric
    dicts). Feed each value to :func:`summarise_cv` / :func:`save_cv_results`.
    """
    shap_save_paths = shap_save_paths or {}
    features_save_paths = features_save_paths or {}
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

    # Per-mode accumulators, keyed by mode name.
    fold_metrics_by_mode: dict[str, dict[str, list]] = {m: defaultdict(list) for m in modes}
    fold_shap_abs_by_mode: dict[str, dict[str, list[np.ndarray]]] = {
        m: defaultdict(list) for m in modes
    }
    fold_shap_mask_abs_by_mode: dict[str, dict[str, list[np.ndarray]]] = {
        m: defaultdict(list) for m in modes
    }
    fold_shap_time_feat_abs_by_mode: dict[str, dict[str, list[np.ndarray]]] = {
        m: defaultdict(list) for m in modes
    }
    fold_shap_time_feat_mask_abs_by_mode: dict[str, dict[str, list[np.ndarray]]] = {
        m: defaultdict(list) for m in modes
    }
    feat_chunks_by_mode: dict[str, dict[str, list[np.ndarray]]] = {
        m: {"X": [], "y": [], "strata": []} for m in modes
    }

    total_folds = CV_N_SPLITS * CV_N_REPEATS

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(subjects, subject_outcomes)):
        print(f"  [{pipeline_name}] fold {fold_idx + 1}/{total_folds}", end="\r")

        # Full train fold (pre-undersampling). Strata median thresholds are
        # derived from this set so they match cv.run_cv, where undersampling
        # touches only the model's feature matrix, not the stratum boundaries.
        train_subjects_full = set(subjects[train_idx].tolist())
        test_subjects = set(subjects[test_idx].tolist())

        # Train-only negative undersampling at the *subject* level, applied to the
        # id list *before* building any dense array. Mirrors cv.run_cv, where
        # undersampling precedes feature materialization — building the full
        # (N_train, T, D) array first and discarding ~90% of rows afterwards would
        # needlessly allocate ~10× the memory and stall on large cohorts.
        if APPLY_UNDERSAMPLING:
            train_subj_arr = subjects[train_idx]
            train_subj_outcomes = subject_outcomes[train_idx]
            keep_subj_idx = _undersample_indices(
                train_subj_outcomes,
                ratio=TRAIN_NEG_POS_RATIO,
                random_state=UNDERSAMPLE_RANDOM_STATE + fold_idx,
            )
            train_subjects = set(train_subj_arr[keep_subj_idx].tolist())
        else:
            train_subjects = train_subjects_full

        train_df = cohort_df.filter(pl.col("id").is_in(train_subjects))
        test_df = cohort_df.filter(pl.col("id").is_in(test_subjects))

        # Build (N, T, D) arrays. df_to_3d_array sorts by id then clock and
        # pads shorter sequences with NaN up to the longest in that split.
        train_ids, X_train_3d, _, _ = df_to_3d_array(train_df, "id", "clock", list(VITAL_COLS))
        test_ids, X_test_3d, _, _ = df_to_3d_array(test_df, "id", "clock", list(VITAL_COLS))

        # Static features: one row per subject, aligned to df_to_3d_array's id order.
        # A left-join onto the id ordering keeps alignment in a single vectorized
        # pass — a per-id .filter() loop here is O(N²) and stalls on large cohorts.
        static_cols = ["age"] if "age" in cohort_df.columns else []
        if static_cols:
            def _static_array(src_df: pl.DataFrame, ids: list) -> np.ndarray:
                per_id = src_df.group_by("id").agg([pl.col(c).first() for c in static_cols])
                ordered = (
                    pl.DataFrame({"id": ids})
                    .join(per_id, on="id", how="left")
                    .select(static_cols)
                )
                return ordered.to_numpy().astype(np.float32)

            X_train_static = _static_array(train_df, train_ids)
            X_test_static = _static_array(test_df, test_ids)
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

        # train_ids/X_train_3d already reflect the undersampled subject set (the
        # negatives were dropped from the id list before df_to_3d_array), so the
        # labels built here are the undersampled training labels.
        y_train_us = np.array([id_to_label_train[i] for i in train_ids], dtype=np.int8)
        y_test = np.array([id_to_label_test[i] for i in test_ids], dtype=np.int8)

        X_train_us = X_train_3d
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

        # Overall metrics are mode-independent (same y_proba); each mode keeps
        # its own "overall" list so per-mode summarise_cv is unchanged.
        m = _compute_metrics(y_test, y_proba)
        ids_arr = np.asarray(test_ids)
        # Strata thresholds use the full (pre-undersampling) train fold, matching
        # cv.run_cv — see train_subjects_full above.
        train_id_list = list(train_subjects_full)
        test_id_list = list(test_subjects)

        # Per-stay aggregated features are mode-independent — build once if any
        # mode requests the features .npz.
        X_test_agg = None
        if features_save_paths:
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

        # Build the GradientExplainer once for this fold's model (the expensive
        # step) and reuse it across every SHAP-emitting mode.
        explainer = None
        D = len(VITAL_COLS)
        X_test_combined = None
        if shap_full and shap_save_paths:
            explainer, X_test_combined = _make_fold_gru_explainer(
                model, X_train_combined, X_test_3d, use_mask, D,
                fold_idx, pipeline_name,
            )

        for mode_name, axes in modes.items():
            strata_arr, imperfekt_lookup, imperfekt_cols, _ = _assign_fold_strata(
                test_ids, train_id_list, test_id_list, case_metrics, axes, mode_name,
            )
            fm = fold_metrics_by_mode[mode_name]
            if m:
                fm["overall"].append(dict(m))
            _accumulate_stratum_metrics(
                fm, strata_arr, ids_arr, y_test, y_proba,
                imperfekt_lookup, imperfekt_cols,
            )

            if mode_name in features_save_paths and X_test_agg is not None:
                ch = feat_chunks_by_mode[mode_name]
                ch["X"].append(X_test_agg)
                ch["y"].append(y_test)
                ch["strata"].append(strata_arr)

            if explainer is None or mode_name not in shap_save_paths:
                continue

            # Per-stratum SHAP subsample uses this mode's strata, matching a
            # single-mode run so the .npz is identical.
            _sub_idx = _stratified_shap_subsample(
                strata_arr, SHAP_MAX_ROWS_PER_STRATUM, SHAP_SUBSAMPLE_RANDOM_STATE + fold_idx
            )
            X_shap_tensor = torch.tensor(
                X_test_combined[_sub_idx], dtype=torch.float32, device=model.device
            )
            _strata_shap = strata_arr[_sub_idx]
            with torch.backends.cudnn.flags(enabled=False):
                # shap_values returns (N_sub, T, D_input, 1) because the wrapper
                # outputs (N, 1); [..., 0] drops the trailing output dimension.
                shap_vals = np.array(explainer.shap_values(X_shap_tensor))[..., 0]  # (N_sub, T, D_input)
            abs_shap = np.abs(shap_vals)
            D_in = abs_shap.shape[2]

            def _agg(arr: np.ndarray, _D=D, _D_in=D_in):
                val_imp = arr[:, :, :_D].mean(axis=(0, 1))                            # (D,)
                mask_imp = arr[:, :, _D:].mean(axis=(0, 1)) if _D_in > _D else None   # (D,) or None
                time_feat_imp = arr[:, :, :_D].mean(axis=0)                            # (T, D)
                time_feat_mask_imp = arr[:, :, _D:].mean(axis=0) if _D_in > _D else None  # (T, D) or None
                return val_imp, mask_imp, time_feat_imp, time_feat_mask_imp

            v, mk, ti, ti_mk = _agg(abs_shap)
            fold_shap_abs_by_mode[mode_name]["overall"].append(v)
            if mk is not None:
                fold_shap_mask_abs_by_mode[mode_name]["overall"].append(mk)
            fold_shap_time_feat_abs_by_mode[mode_name]["overall"].append(ti)
            if ti_mk is not None:
                fold_shap_time_feat_mask_abs_by_mode[mode_name]["overall"].append(ti_mk)

            for _sl in np.unique(_strata_shap):
                if _sl == "":
                    continue
                _sl_mask = _strata_shap == _sl
                if _sl_mask.sum() > 0:
                    v_s, mk_s, ti_s, ti_mk_s = _agg(abs_shap[_sl_mask])
                    fold_shap_abs_by_mode[mode_name][_sl].append(v_s)
                    if mk_s is not None:
                        fold_shap_mask_abs_by_mode[mode_name][_sl].append(mk_s)
                    fold_shap_time_feat_abs_by_mode[mode_name][_sl].append(ti_s)
                    if ti_mk_s is not None:
                        fold_shap_time_feat_mask_abs_by_mode[mode_name][_sl].append(ti_mk_s)

    print()

    for mode_name in modes:
        shap_path = shap_save_paths.get(mode_name)
        fold_shap_abs = fold_shap_abs_by_mode[mode_name]
        if shap_full and shap_path is not None and fold_shap_abs:
            shap_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict: dict[str, np.ndarray] = {"feature_names": np.array(list(VITAL_COLS))}
            for _sl, _arrs in fold_shap_abs.items():
                save_dict[f"shap_{_sl}"] = np.stack(_arrs)
            for _sl, _arrs in fold_shap_mask_abs_by_mode[mode_name].items():
                save_dict[f"shap_mask_{_sl}"] = np.stack(_arrs)
            for _sl, _arrs in fold_shap_time_feat_abs_by_mode[mode_name].items():
                save_dict[f"shap_time_feat_{_sl}"] = np.stack(_arrs)          # (n_folds, T, D)
            for _sl, _arrs in fold_shap_time_feat_mask_abs_by_mode[mode_name].items():
                save_dict[f"shap_time_feat_mask_{_sl}"] = np.stack(_arrs)     # (n_folds, T, D)
            np.savez_compressed(shap_path, **save_dict)
            print(f"  [{pipeline_name}] GRU SHAP saved to {shap_path}")

        feat_path = features_save_paths.get(mode_name)
        ch = feat_chunks_by_mode[mode_name]
        if feat_path is not None and ch["X"]:
            agg_feature_names = [f"{c}_{stat}" for stat in ("mean", "min", "max") for c in VITAL_COLS]
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feat_path,
                feature_names=np.array(agg_feature_names),
                X_test_all=np.concatenate(ch["X"], axis=0),
                y_test_all=np.concatenate(ch["y"], axis=0),
                strata_all=np.concatenate(ch["strata"], axis=0),
            )
            print(f"  [{pipeline_name}] Feature values saved to {feat_path}")

    return {m: dict(fm) for m, fm in fold_metrics_by_mode.items()}
