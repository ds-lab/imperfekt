"""Standalone SAITS pre-training.

Decouples SAITS training from the experiment pipeline. Trains one SAITS model
per data-preparation branch (``plaus_keep`` and ``plaus_remove``) and writes the
weights to disk so the experiment pipeline can load them for fast inference.

Run with::

    python -m examples.nemsis.train_saits

The models are written to ``models/saits/<dataset>_<branch>.pypots``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from config import (
    COHORT_PATH,
    DATASET_NAME,
    PATH,
    VITAL_COLS,
)
from prep import (
    _apply_plausibility,
    filter_cohort,
    make_plausibility_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_saits")

# Where the trained model directories are written.
SAITS_MODEL_DIR = PATH / "models" / "saits"

ID_COL = "id"
TIME_COL = "clock"

# The plausibility method used to flag implausible values for the
# ``plaus_remove`` branch. ``prep.make_plausibility_mask`` supports "iqr"/"mad";
# the task specifies an L1 IQR mask.
PLAUS_METHOD = "iqr"


def _df_to_3d_array(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a long-format Polars DF into a ``(N, T, D)`` numpy array and mask.

    N: number of ids
    T: max number of time steps across all ids (shorter sequences padded with NaN)
    D: number of features

    Returns:
        ids: list of unique ids in sorted order
        data: (N, T, D) float64 array with NaN for missing/padded values
        mask: (N, T, D) bool array, True = observed, False = missing/padded
        seq_lengths: (N,) array of actual sequence lengths per id
    """
    df_sorted = df.sort([id_col, time_col])

    ids = df_sorted.select(pl.col(id_col)).unique().sort(id_col)[id_col].to_list()
    n_ids = len(ids)
    n_features = len(feature_cols)

    counts = df_sorted.group_by(id_col).agg(pl.len().alias("count")).sort(id_col)
    seq_lengths = counts["count"].to_numpy()
    n_steps = int(seq_lengths.max())  # pad shorter sequences up to the max length

    data = np.full((n_ids, n_steps, n_features), np.nan, dtype=np.float64)

    values_2d = df_sorted.select(feature_cols).with_columns([pl.all().cast(pl.Float64)]).to_numpy()

    row_idx = 0
    for i, length in enumerate(seq_lengths):
        data[i, :length, :] = values_2d[row_idx : row_idx + length, :]
        row_idx += length

    mask = ~np.isnan(data)
    return ids, data, mask, seq_lengths


def _build_saits(n_steps: int, n_features: int, n_samples: int):
    """Construct a SAITS model + optimizer sized to the cohort.

    The hyperparameter regimes mirror the previous in-pipeline training so the
    pre-trained models match the behaviour the experiments expect.
    """
    from pypots.imputation import SAITS
    from pypots.optim import AdamW

    if n_samples < 50:
        d_model, n_heads, d_k, d_v, d_ffn, n_layers = 32, 2, 16, 16, 64, 1
        epochs, batch_size, lr = 500, 32, 1e-3
    elif n_samples < 500:
        d_model, n_heads, d_k, d_v, d_ffn, n_layers = 64, 2, 32, 32, 128, 1
        epochs, batch_size, lr = 200, 32, 5e-4
    elif n_samples < 5000:
        d_model, n_heads, d_k, d_v, d_ffn, n_layers = 64, 2, 32, 32, 128, 2
        epochs, batch_size, lr = 100, 64, 1e-4
    else:
        # Large cohort, short sequences, few features: stable transformer regime.
        d_model, n_heads, d_k, d_v, d_ffn, n_layers = 128, 4, 32, 32, 256, 2
        epochs, batch_size, lr = 100, 512, 3e-4

    logger.info(
        "SAITS config: n_samples=%d, n_steps=%d, n_features=%d", n_samples, n_steps, n_features
    )
    logger.info(
        "SAITS hyperparams: d_model=%d, d_ffn=%d, lr=%g, batch_size=%d, epochs=%d",
        d_model,
        d_ffn,
        lr,
        batch_size,
        epochs,
    )

    optimizer = AdamW(lr=lr, weight_decay=1e-4)
    saits = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ffn=d_ffn,
        dropout=0.25,
        epochs=epochs,
        patience=10,
        batch_size=batch_size,
        optimizer=optimizer,
        verbose=True,
    )
    return saits


def train_branch(
    df: pl.DataFrame,
    branch: str,
    cols: list[str],
    model_path: Path,
) -> None:
    """Train and persist a single SAITS model for one preparation branch.

    The normalization statistics (per-feature mean/std) are saved alongside the
    weights so inference uses the exact same scaling.
    """
    from pypots.utils.random import set_random_seed

    set_random_seed(0)

    _ids, data, _mask, _seq_lengths = _df_to_3d_array(
        df, id_col=ID_COL, time_col=TIME_COL, feature_cols=cols
    )
    n_samples, n_steps, n_features = data.shape

    # Per-feature z-score normalization, ignoring NaN. Critical for stable
    # transformer training. Persist the stats so inference can de/normalize.
    flat = data.reshape(-1, n_features)
    feature_means = np.nanmean(flat, axis=0)
    feature_stds = np.nanstd(flat, axis=0)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # avoid div-by-zero

    data_normalized = (data - feature_means) / feature_stds
    train_set = {"X": data_normalized}

    saits = _build_saits(n_steps=n_steps, n_features=n_features, n_samples=n_samples)

    logger.info("Training SAITS for branch '%s' (%d stays)…", branch, n_samples)
    start = time.time()
    saits.fit(train_set)
    elapsed = time.time() - start
    logger.info("SAITS branch '%s' trained in %.2f seconds (%.2f min)", branch, elapsed, elapsed / 60)

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
    logger.info("Saved SAITS model -> %s", model_path)
    logger.info("Saved SAITS normalization meta -> %s", meta_path)


def main() -> None:
    cohort_path = Path(COHORT_PATH)
    logger.info("Loading cohort from %s", cohort_path)
    df_raw = pl.read_parquet(cohort_path)

    # Apply the same cohort filtering the experiment pipeline uses before any
    # plausibility handling, so the trained model sees identical sequences.
    df = filter_cohort(df_raw)
    logger.info("Cohort after filtering: %d stays, %d observations", df[ID_COL].n_unique(), len(df))

    mask = make_plausibility_mask(df, method=PLAUS_METHOD)

    branches = {
        "plaus_keep": _apply_plausibility(df, mask, "keep"),
        "plaus_remove": _apply_plausibility(df, mask, "remove"),
    }

    for branch, branch_df in branches.items():
        model_path = SAITS_MODEL_DIR / f"{DATASET_NAME}_{branch}.pypots"
        train_branch(branch_df, branch=branch, cols=list(VITAL_COLS), model_path=model_path)

    logger.info("All SAITS branches trained.")


if __name__ == "__main__":
    main()
