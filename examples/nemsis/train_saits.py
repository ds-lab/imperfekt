"""Standalone SAITS pre-training.
Decouples SAITS training from the experiment pipeline. Trains one SAITS model
per data-preparation branch (``plaus_keep`` and ``plaus_remove``) and writes the
weights to disk so the experiment pipeline can load them for fast inference.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from config import (
    COHORT_MAX_READINGS,
    COHORT_MIN_READINGS,
    COHORT_PATH,
    COHORT_WINDOW_MINUTES,
    DATASET_NAME,
    RANDOM_STATE,
    REQUIRED_VITAL_COLS,
    data_fingerprint,
    saits_model_path,
)
from prep import (
    _apply_plausibility,
    make_plausibility_mask,
)
from seq_array import df_to_3d_array

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_saits")

ID_COL = "id"
TIME_COL = "clock"
PLAUS_METHOD = "mad"

APPLY_SAITS_UNDERSAMPLING = True
SAITS_NEG_POS_RATIO = 10  # target 1:10 pos:neg
LABEL_COL = "label"

# Held-out validation for SAITS. 
SAITS_VAL_FRACTION = 0.1
# Rate of additional MCAR masking applied to the val sequences to create the
# artificial holes whose reconstruction error pypots scores.
SAITS_VAL_ARTIFICIAL_MISSING_RATE = 0.2

# Early-stopping patience: number of consecutive epochs with no val-metric
# improvement before training halts. Set to None to disable early stopping (train the full `epochs`).
SAITS_PATIENCE: int | None = 25
# Optional learning-rate override (None -> use the per-regime default)
SAITS_LR_OVERRIDE: float | None = None


def _select_val_stay_ids(df: pl.DataFrame, val_fraction: float) -> tuple[set, dict]:
    """Pick the held-out validation stay ids for a long-format cohort.

    Stratified by stay label (max-over-rows) so both classes are represented in
    the validation set even at the cohort's low positive prevalence; at least
    one stay per present class is held out. Deterministic given RANDOM_STATE.

    Returns (set of val ids, stats). The caller does the actual train/val split
    by array membership, so only the ids are needed here.
    """
    stay_labels = (
        df.group_by(ID_COL).agg(pl.col(LABEL_COL).max().alias(LABEL_COL)).sort(ID_COL)
    )
    n_total = stay_labels.height
    val_ids: set = set()
    for lbl in (1, 0):
        ids = stay_labels.filter(pl.col(LABEL_COL) == lbl).select(ID_COL)
        if ids.height == 0:
            continue
        # round() can never exceed ids.height for val_fraction <= 1, so the only
        # adjustment needed is the floor: guarantee at least one stay per class.
        n_val = max(int(round(ids.height * val_fraction)), 1)
        sampled = ids.sample(n=n_val, shuffle=True, seed=RANDOM_STATE)
        val_ids.update(sampled[ID_COL].to_list())

    stats = {
        "val_fraction": val_fraction,
        "seed": RANDOM_STATE,
        "n_train_stays": n_total - len(val_ids),
        "n_val_stays": len(val_ids),
    }
    logger.info(
        "Train/val split: %d train stays, %d val stays (val_fraction=%g)",
        stats["n_train_stays"],
        stats["n_val_stays"],
        val_fraction,
    )
    return val_ids, stats


def _locf_impute_np(X: np.ndarray) -> np.ndarray:
    """Last-observation-carried-forward over the time axis of a (N, T, D) array.

    NaNs are filled forward within each (sequence, feature). Any leading NaNs
    (no prior observation) are then back-filled from the first later observation;
    a sequence/feature that is entirely NaN is left as NaN. Mirrors the LOCF
    strategy used at inference so the val baseline is the same method.
    """
    out = X.copy()
    t = out.shape[1]
    # Forward fill: carry the last seen value along T.
    for step in range(1, t):
        prev = out[:, step - 1, :]
        cur = out[:, step, :]
        take_prev = np.isnan(cur)
        out[:, step, :] = np.where(take_prev, prev, cur)
    # Back fill leading gaps: sweep from the end so the earliest obs propagates up.
    for step in range(t - 2, -1, -1):
        nxt = out[:, step + 1, :]
        cur = out[:, step, :]
        take_next = np.isnan(cur)
        out[:, step, :] = np.where(take_next, nxt, cur)
    return out


def _mse_on_holes(
    pred: np.ndarray, truth: np.ndarray, holes: np.ndarray
) -> float:
    """Mean squared error over exactly the cells flagged True in ``holes``.

    Cells where the prediction is NaN (e.g. LOCF left a fully-missing series
    empty) are dropped from the average so a baseline isn't credited or charged
    for cells it cannot fill. Returns NaN if no scorable holes remain.
    """
    sel = holes & ~np.isnan(pred) & ~np.isnan(truth)
    if not sel.any():
        return float("nan")
    diff = pred[sel] - truth[sel]
    return float(np.mean(diff * diff))


def _baseline_val_mse(
    val_X: np.ndarray, val_ori: np.ndarray, holes: np.ndarray
) -> dict:
    """Mean- and LOCF-imputation MSE on the same artificial holes as SAITS.

    ``val_X`` is the normalized val tensor with artificial holes already punched;
    ``val_ori`` is the normalized truth; ``holes`` marks the artificial holes.
    Both baselines see exactly ``val_X`` (natural + artificial missingness) and
    are scored only on ``holes`` against ``val_ori`` — identical to how SAITS is
    scored. Because the data is z-scored, mean imputation (predict 0) is expected
    to land near MSE 1.0; the measured value documents that the scoring is sane.
    """
    mean_pred = np.where(np.isnan(val_X), 0.0, val_X)  # z-scored mean == 0
    locf_pred = _locf_impute_np(val_X)
    return {
        "mean": _mse_on_holes(mean_pred, val_ori, holes),
        "locf": _mse_on_holes(locf_pred, val_ori, holes),
        "n_holes": int((holes & ~np.isnan(val_ori)).sum()),
    }


def _saits_regime(n_samples: int, n_steps: int, n_features: int) -> tuple[str, dict]:
    """Return the (regime_name, hyperparameter dict) sized to the data shape.
    """
    # --- Model capacity: driven by D (features) and T (sequence length). ---
    if n_features <= 6 and n_steps <= 32:
        shape_regime = "smallD_shortT"
        d_model, n_heads, d_k, d_v, d_ffn = 32, 4, 8, 8, 64
        n_layers = 1 if n_steps <= 12 else 2
        dropout = 0.1
    elif n_features <= 16:
        shape_regime = "midD"
        d_model, n_heads, d_k, d_v, d_ffn = 64, 4, 16, 16, 128
        n_layers = 2
        dropout = 0.2
    else:
        shape_regime = "largeD"
        d_model, n_heads, d_k, d_v, d_ffn = 128, 4, 32, 32, 256
        n_layers = 2
        dropout = 0.25

    # --- Training budget: driven by N (number of stays). ---
    if n_samples < 500:
        size_regime = "N<500"
        epochs, batch_size, lr = 200, 64, 1e-4
    elif n_samples < 5000:
        size_regime = "N<5000"
        epochs, batch_size, lr = 200, 128, 3e-4
    else:
        size_regime = "N>=5000"
        epochs, batch_size, lr = 200, 512, 3e-4

    regime = f"{shape_regime}|{size_regime}"
    hp = {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_k": d_k,
        "d_v": d_v,
        "d_ffn": d_ffn,
        "n_layers": n_layers,
        "dropout": dropout,
        "epochs": epochs,
        "patience": SAITS_PATIENCE,
        "batch_size": batch_size,
        "lr": SAITS_LR_OVERRIDE if SAITS_LR_OVERRIDE is not None else lr,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
    }
    return regime, hp


def _build_saits(n_steps: int, n_features: int, n_samples: int) -> tuple[object, str, dict]:
    """Construct a SAITS model + optimizer sized to the cohort.

    Returns the model, the regime name, and the hyperparameter dict so the
    caller can persist the exact configuration alongside the weights.
    """
    from pypots.imputation import SAITS
    from pypots.optim import AdamW

    regime, hp = _saits_regime(n_samples=n_samples, n_steps=n_steps, n_features=n_features)

    logger.info(
        "SAITS config: n_samples=%d, n_steps=%d, n_features=%d, regime=%s",
        n_samples,
        n_steps,
        n_features,
        regime,
    )
    logger.info(
        "SAITS hyperparams: d_model=%d, d_ffn=%d, lr=%g, batch_size=%d, epochs=%d, patience=%s",
        hp["d_model"],
        hp["d_ffn"],
        hp["lr"],
        hp["batch_size"],
        hp["epochs"],
        hp["patience"],
    )

    optimizer = AdamW(lr=hp["lr"], weight_decay=hp["weight_decay"])
    saits = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=hp["n_layers"],
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        d_k=hp["d_k"],
        d_v=hp["d_v"],
        d_ffn=hp["d_ffn"],
        dropout=hp["dropout"],
        epochs=hp["epochs"],
        patience=hp["patience"],
        batch_size=hp["batch_size"],
        optimizer=optimizer,
        # device="cuda:0",
        verbose=True,
    )
    return saits, regime, hp


SAITS_TRAIN_SEED = 0  # passed to pypots.set_random_seed before each fit


def _capture_env() -> dict:
    """Record library/runtime versions for reproducibility."""
    import platform

    import torch

    try:
        import pypots

        pypots_version = pypots.__version__
    except Exception:  # pragma: no cover - version attr should exist
        pypots_version = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "pypots": pypots_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def train_branch(
    df: pl.DataFrame,
    branch: str,
    cols: list[str],
    model_path: Path,
    provenance: dict,
) -> None:
    """Train and persist a single SAITS model for one preparation branch.

    The normalization statistics (per-feature mean/std) are saved alongside the
    weights so inference uses the exact same scaling. A ``.meta.json`` sidecar
    records the full hyperparameters, cohort/undersampling provenance, training
    dynamics, and environment so the run is reproducible for publication.
    """
    from pygrinder import mcar
    from pypots.utils.random import set_random_seed

    set_random_seed(SAITS_TRAIN_SEED)

    # Split off a held-out validation set at the stay level
    if SAITS_VAL_FRACTION > 0:
        val_id_set, split_stats = _select_val_stay_ids(df, SAITS_VAL_FRACTION)
        split_stats = {"applied": True, **split_stats}
    else:
        split_stats = {"applied": False}
        val_id_set = set()

    # Convert the full branch once so train and val share the same padded n_steps
    all_ids, data, _mask, seq_lengths = df_to_3d_array(
        df, id_col=ID_COL, time_col=TIME_COL, feature_cols=cols
    )
    n_samples, n_steps, n_features = data.shape
    n_observations = int(seq_lengths.sum())

    if COHORT_MAX_READINGS is not None and n_steps < COHORT_MAX_READINGS:
        pad = np.full((n_samples, COHORT_MAX_READINGS - n_steps, n_features), np.nan)
        data = np.concatenate([data, pad], axis=1)
        n_steps = COHORT_MAX_READINGS

    is_val = np.array([i in val_id_set for i in all_ids], dtype=bool)
    train_data = data[~is_val]
    val_data = data[is_val]

    flat_train = train_data.reshape(-1, n_features)
    feature_means = np.nanmean(flat_train, axis=0)
    feature_stds = np.nanstd(flat_train, axis=0)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # avoid div-by-zero

    train_norm = (train_data - feature_means) / feature_stds
    train_set = {"X": train_norm}

    # Build the val_set: X_ori is the held-out truth; X has additional MCAR
    # holes punched in, and pypots scores reconstruction of exactly those holes.
    val_set = None
    val_ori = val_X = val_holes = None
    if val_data.shape[0] > 0:
        val_ori = (val_data - feature_means) / feature_stds
        val_X = mcar(val_ori, p=SAITS_VAL_ARTIFICIAL_MISSING_RATE)
        # Artificial holes = observed in truth but missing in the masked input.
        val_holes = ~np.isnan(val_ori) & np.isnan(val_X)
        val_set = {"X": val_X, "X_ori": val_ori}

    saits, regime, hp = _build_saits(
        n_steps=n_steps, n_features=n_features, n_samples=train_data.shape[0]
    )

    logger.info(
        "Training SAITS for branch '%s' (%d train / %d val stays)…",
        branch,
        train_data.shape[0],
        val_data.shape[0],
    )
    start = time.time()
    saits.fit(train_set, val_set=val_set)
    elapsed = time.time() - start
    logger.info("SAITS branch '%s' trained in %.2f seconds (%.2f min)", branch, elapsed, elapsed / 60)

    # Score SAITS and the trivial baselines
    val_mse = None
    if val_set is not None:
        saits_pred = np.asarray(saits.impute({"X": val_X}))
        if saits_pred.ndim == 4:
            saits_pred = saits_pred.squeeze(1)
        val_mse = {
            "saits": _mse_on_holes(saits_pred, val_ori, val_holes),
            **_baseline_val_mse(val_X, val_ori, val_holes),
        }
        logger.info(
            "Val MSE on artificial holes (normalized) — SAITS=%.4f mean=%.4f locf=%.4f (n=%d)",
            val_mse["saits"],
            val_mse["mean"],
            val_mse["locf"],
            val_mse["n_holes"],
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    saits.save(str(model_path), overwrite=True)

    # Persist normalization stats + the feature order/n_steps next to the model.
    meta_path = model_path.with_suffix(model_path.suffix + ".norm.npz")
    np.savez(
        meta_path,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_cols=np.array(cols, dtype=object),
        n_steps=np.array([n_steps]),
        train_elapsed_seconds=np.array([elapsed]),
    )

    # Human/machine-readable provenance sidecar
    json_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    meta = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "branch": branch,
        "dataset": DATASET_NAME,
        "model": "SAITS",
        "feature_cols": list(cols),
        "n_steps": n_steps,
        "n_features": n_features,
        "shape": {"n_samples": n_samples, "n_observations": n_observations},
        "regime": regime,
        "hyperparameters": hp,
        "normalization": {
            "method": "per_feature_zscore_nanignore_trainsplit",
            "fit_on": "train_split_only",
            "feature_means": [float(x) for x in feature_means],
            "feature_stds": [float(x) for x in feature_stds],
        },
        "validation": {
            **split_stats,
            "artificial_missing_rate": (
                SAITS_VAL_ARTIFICIAL_MISSING_RATE if val_set is not None else None
            ),
            "metric": "MSE_on_artificial_holes" if val_set is not None else None,
            "mse_on_holes_normalized": val_mse,
        },
        "training": {
            "train_seed": SAITS_TRAIN_SEED,
            "elapsed_seconds": elapsed,
            "training_loss": type(getattr(saits, "training_loss", None)).__name__,
            "validation_metric": type(getattr(saits, "validation_metric", None)).__name__,
            "best_loss": float(getattr(saits, "best_loss", float("nan"))),
            "best_loss_is_validation": val_set is not None,
            "best_loss_metric": (
                type(getattr(saits, "validation_metric", None)).__name__
                if val_set is not None
                else type(getattr(saits, "training_loss", None)).__name__
            ),
            "best_epoch": int(getattr(saits, "best_epoch", -1)),
            "device": str(getattr(saits, "device", None)),
        },
        "provenance": provenance,
        "env": _capture_env(),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Saved SAITS model -> %s", model_path)
    logger.info("Saved SAITS normalization meta -> %s", meta_path)
    logger.info("Saved SAITS provenance meta -> %s", json_path)


def _stay_label_counts(df: pl.DataFrame) -> tuple[int, int]:
    """Return (n_positive_stays, n_negative_stays) using max-over-rows labels."""
    stay_labels = df.group_by(ID_COL).agg(pl.col(LABEL_COL).max().alias(LABEL_COL))
    n_pos = stay_labels.filter(pl.col(LABEL_COL) == 1).height
    n_neg = stay_labels.height - n_pos
    return n_pos, n_neg


def _undersample_stays(df: pl.DataFrame, neg_pos_ratio: int) -> tuple[pl.DataFrame, dict]:
    """Keep all positive stays, subsample negative stays to 1:``neg_pos_ratio``.

    Returns the filtered DataFrame and a stats dict for provenance.
    """
    stay_labels = (
        df.group_by(ID_COL).agg(pl.col(LABEL_COL).max().alias(LABEL_COL)).sort(ID_COL)
    )
    pos_ids = stay_labels.filter(pl.col(LABEL_COL) == 1).select(ID_COL)
    neg = stay_labels.filter(pl.col(LABEL_COL) != 1)

    n_neg_target = min(neg.height, pos_ids.height * neg_pos_ratio)
    neg_ids = neg.sample(n=n_neg_target, shuffle=True, seed=RANDOM_STATE).select(ID_COL)

    keep_ids = pl.concat([pos_ids, neg_ids], how="vertical")
    out = df.join(keep_ids, on=ID_COL, how="inner")
    logger.info(
        "Undersampled stays: %d positive + %d negative (target 1:%d) -> %d stays, %d observations",
        pos_ids.height,
        neg_ids.height,
        neg_pos_ratio,
        out[ID_COL].n_unique(),
        len(out),
    )
    stats = {
        "applied": True,
        "neg_pos_ratio": neg_pos_ratio,
        "seed": RANDOM_STATE,
        "n_pos_stays": pos_ids.height,
        "n_neg_stays": neg_ids.height,
    }
    return out, stats


def main() -> None:
    cohort_path = Path(COHORT_PATH)
    logger.info("Loading cohort from %s", cohort_path)
    df = pl.read_parquet(cohort_path)
    logger.info("Cohort loaded: %d stays, %d observations", df[ID_COL].n_unique(), len(df))

    n_pos_full, n_neg_full = _stay_label_counts(df)

    if APPLY_SAITS_UNDERSAMPLING:
        df, undersampling = _undersample_stays(df, neg_pos_ratio=SAITS_NEG_POS_RATIO)
    else:
        undersampling = {"applied": False}

    mask = make_plausibility_mask(df, method=PLAUS_METHOD)

    provenance = {
        "cohort": {
            "path": str(cohort_path),
            "window_minutes": COHORT_WINDOW_MINUTES,
            "min_readings": COHORT_MIN_READINGS,
            "plaus_method": PLAUS_METHOD,
            "n_stays_full": df[ID_COL].n_unique() if not APPLY_SAITS_UNDERSAMPLING else None,
            "n_pos_stays_full": n_pos_full,
            "n_neg_stays_full": n_neg_full,
        },
        "undersampling": undersampling,
        "data_fingerprint": data_fingerprint(cohort_path),
    }

    branches = {
        "keep": _apply_plausibility(df, mask, "keep"),
        "remove": _apply_plausibility(df, mask, "remove"),
    }

    for plaus, branch_df in branches.items():
        model_path = saits_model_path(plaus)
        train_branch(
            branch_df,
            branch=f"plaus_{plaus}",
            cols=list(REQUIRED_VITAL_COLS),
            model_path=model_path,
            provenance=provenance,
        )

    logger.info("All SAITS branches trained.")


if __name__ == "__main__":
    main()
