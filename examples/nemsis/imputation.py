import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def impute(
    df: pl.DataFrame,
    cols: list,
    strategy: str = "locf",
    id_col: str = "id",
    time_col: str = "clock_no",
    saits_model_path: str = None,
) -> tuple[pl.DataFrame, float]:
    """Imputes missing values in specified columns using the given strategy.

    Parameters:
        df (pl.DataFrame): Input DataFrame with missing values.
        cols (list): List of columns to impute.
        strategy (str): Imputation strategy ('locf' for last observation carried forward,
                       'mean' for mean imputation, 'saits' for SAITS deep learning imputation).
        id_col (str): Name of the ID column (default: "id").
        time_col (str): Name of the time column (default: "clock_no").
        saits_model_path (str): Path to a pre-trained SAITS model (required for the
                       'saits' strategy). The model is loaded from disk for inference;
                       no training happens here. See examples/nemsis/train_saits.py.

    Returns:
        tuple[pl.DataFrame, float]: Tuple of (DataFrame with imputed values, elapsed time in seconds).
    """
    start_time = time.time()
    imputed_df = df.clone()

    if strategy == "locf":
        # Sort by id and time, then forward fill within each id group
        imputed_df = (
            imputed_df.sort([id_col, time_col])
            .with_columns([pl.col(col).fill_nan(None).forward_fill().over(id_col) for col in cols])
            .with_columns(
                [
                    # Fill any remaining nulls (at the start of a sequence) with the per-id mean
                    pl.col(col).fill_null(pl.col(col).mean().over(id_col))
                    for col in cols
                ]
            )
        )
    elif strategy == "mean":
        for col in cols:
            mean_value = imputed_df.select(pl.col(col).mean()).item()
            imputed_df = imputed_df.with_columns(pl.col(col).fill_null(mean_value).alias(col))
    elif strategy == "saits":
        if saits_model_path is None:
            raise ValueError(
                "strategy='saits' requires `saits_model_path` to a pre-trained model "
                "(train one with examples/nemsis/train_saits.py)."
            )
        imputed_df = _impute_with_saits(
            imputed_df,
            cols,
            saits_model_path=saits_model_path,
            id_col=id_col,
            time_col=time_col,
        )
    else:
        raise ValueError(f"Imputation strategy '{strategy}' not recognized.")

    elapsed_time = time.time() - start_time
    logger.info(f"Imputation strategy '{strategy}' completed in {elapsed_time:.2f} seconds")

    return imputed_df, elapsed_time


def export_imputation_elapsed_time(save_results_path: Path, imputation_times: dict) -> None:
    imputation_times_extracted = {}
    for strategy, time_seconds in imputation_times.items():
        imputation_times_extracted[f"{strategy}_seconds"] = time_seconds
    with open(save_results_path / "imputation_times.json", "w") as f:
        json.dump(imputation_times_extracted, f, indent=4)
    return None


def _df_to_3d_array(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
):
    """
    Convert a long-format Polars DF into a (N, T, D) numpy array and mask.

    N: number of ids
    T: max number of time steps across all ids (shorter sequences are padded with NaN)
    D: number of features

    Returns:
        ids: list of unique ids in order
        data: (N, T, D) numpy array with NaN for missing/padded values
        mask: (N, T, D) boolean array, True = observed, False = missing/padded
        seq_lengths: (N,) array of actual sequence lengths per id
    """
    # sort by id, time
    df_sorted = df.sort([id_col, time_col])

    # unique ids in deterministic order
    ids = df_sorted.select(pl.col(id_col)).unique().sort(id_col)[id_col].to_list()
    n_ids = len(ids)
    n_features = len(feature_cols)

    # Get the number of time steps per id
    counts = df_sorted.group_by(id_col).agg(pl.len().alias("count")).sort(id_col)
    seq_lengths = counts["count"].to_numpy()
    n_steps = int(seq_lengths.max())  # use max length, pad shorter ones

    # Initialize with NaN (will be treated as missing)
    data = np.full((n_ids, n_steps, n_features), np.nan, dtype=np.float64)

    # Fill in actual values per id
    values_2d = df_sorted.select(feature_cols).with_columns([pl.all().cast(pl.Float64)]).to_numpy()

    row_idx = 0
    for i, length in enumerate(seq_lengths):
        data[i, :length, :] = values_2d[row_idx : row_idx + length, :]
        row_idx += length

    # mask: True = observed, False = missing/padded
    mask = ~np.isnan(data)

    return ids, data, mask, seq_lengths


def _impute_with_saits(
    df: pl.DataFrame,
    cols: list[str],
    saits_model_path: str,
    id_col: str = "id",
    time_col: str = "clock",
) -> pl.DataFrame:
    """
    Impute using a PRE-TRAINED SAITS model from pypots (inference only).

    The model is loaded from `saits_model_path` (trained by
    examples/nemsis/train_saits.py). This function does NOT fit a model; it
    converts the incoming frame to the required 3D array, runs `model.impute()`,
    de-normalizes, and maps the imputed values back into the polars frame.

    Crucial rule: if a variable is 100% missing for a specific patient id, SAITS
    may still produce a value, but we revert that variable back to NaN for that
    patient. We rely on XGBoost downstream to treat a fully-missing variable as a
    distinct clinical phenotype.

    Parameters:
        df: Input DataFrame with missing values
        cols: Columns to impute
        saits_model_path: Path to the pre-trained SAITS model on disk
        id_col: Name of the ID column
        time_col: Name of the time column
    """
    from pypots.imputation import SAITS

    model_path = Path(saits_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pre-trained SAITS model not found at {model_path}. "
            "Train it first with examples/nemsis/train_saits.py."
        )

    # Load the normalization stats persisted alongside the model weights.
    meta_path = model_path.with_suffix(model_path.suffix + ".norm.npz")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"SAITS normalization meta not found at {meta_path}. "
            "Re-train with examples/nemsis/train_saits.py to regenerate it."
        )
    meta = np.load(meta_path, allow_pickle=True)
    feature_means = meta["feature_means"]
    feature_stds = meta["feature_stds"]
    trained_cols = [str(c) for c in meta["feature_cols"].tolist()]
    if trained_cols != list(cols):
        raise ValueError(
            f"SAITS model was trained on feature order {trained_cols} "
            f"but impute() was called with {list(cols)}. Order must match."
        )

    ids, data, mask, seq_lengths = _df_to_3d_array(
        df, id_col=id_col, time_col=time_col, feature_cols=cols
    )
    n_samples, n_steps, n_features = data.shape

    # Per-patient fully-missing mask, computed BEFORE imputation.
    # missing_all[i, d] == True  ->  feature d was never observed for patient i.
    # `mask` is True where observed; collapse over the time axis within each
    # sequence's real length (padded steps are already False in `mask`).
    observed_any = mask.any(axis=1)  # (N, D): True if observed at least once
    missing_all = ~observed_any  # (N, D): True if 100% missing for that patient

    # Normalize using the SAME stats the model was trained with.
    data_normalized = (data - feature_means) / feature_stds

    # In pypots, `load` is an instance method that restores the saved model
    # (architecture + weights) into the constructed shell. The placeholder
    # n_steps/n_features are overwritten by the checkpoint on load.
    saits = SAITS(n_steps=n_steps, n_features=n_features)
    saits.load(str(model_path))

    imputed_data_normalized = saits.impute({"X": data_normalized})
    # pypots may return an extra leading dim depending on version; squeeze to (N, T, D).
    imputed_data_normalized = np.asarray(imputed_data_normalized)
    if imputed_data_normalized.ndim == 4:
        imputed_data_normalized = imputed_data_normalized.squeeze(1)

    # De-normalize back to original scale.
    imputed_data = imputed_data_normalized * feature_stds + feature_means

    # --- Crucial masking rule ---------------------------------------------
    # Revert variables that were 100% missing for a patient back to NaN, so
    # XGBoost downstream treats them as a distinct (never-recorded) phenotype.
    for i in range(n_samples):
        for d in range(n_features):
            if missing_all[i, d]:
                imputed_data[i, :, d] = np.nan
    # ----------------------------------------------------------------------

    # Extract only the actual (non-padded) values for each sequence (un-pad).
    imputed_values_list = []
    for i, length in enumerate(seq_lengths):
        imputed_values_list.append(imputed_data[i, :length, :])

    imputed_flat = np.vstack(imputed_values_list)

    df_sorted = df.sort([id_col, time_col])
    imputed_cols = [pl.Series(name=col, values=imputed_flat[:, i]) for i, col in enumerate(cols)]

    df_imputed_sorted = df_sorted.with_columns(imputed_cols)

    # Restore original row order
    return (
        df_imputed_sorted.join(
            df.select(id_col, time_col).with_row_index("row_idx"),
            on=[id_col, time_col],
            how="inner",
        )
        .sort("row_idx")
        .drop("row_idx")
    )
