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
    save_training_log: str = None,
) -> tuple[pl.DataFrame, float]:
    """Imputes missing values in specified columns using the given strategy.

    Parameters:
        df (pl.DataFrame): Input DataFrame with missing values.
        cols (list): List of columns to impute.
        strategy (str): Imputation strategy ('locf' for last observation carried forward,
                       'mean' for mean imputation, 'saits' for SAITS deep learning imputation).
        id_col (str): Name of the ID column (default: "id").
        time_col (str): Name of the time column (default: "clock_no").
        save_training_log (str): Optional path to save SAITS training log (only used with 'saits' strategy).

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
        imputed_df = _impute_with_saits(
            imputed_df, cols, id_col=id_col, time_col=time_col, save_training_log=save_training_log
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
    id_col: str = "id",
    time_col: str = "clock",
    save_training_log: str = None,
) -> pl.DataFrame:
    """
    Impute using SAITS from pypots.

    Parameters:
        df: Input DataFrame with missing values
        cols: Columns to impute
        id_col: Name of the ID column
        time_col: Name of the time column
        save_training_log: Optional path to save training summary (as .txt)
    """
    from pathlib import Path

    from pypots.imputation import SAITS
    from pypots.utils.random import set_random_seed

    set_random_seed(0)

    ids, data, mask, seq_lengths = _df_to_3d_array(
        df, id_col=id_col, time_col=time_col, feature_cols=cols
    )
    n_samples, n_steps, n_features = data.shape

    # Normalize data (per-feature z-score normalization, ignoring NaN)
    # This is critical for stable neural network training
    feature_means = np.nanmean(data.reshape(-1, n_features), axis=0)
    feature_stds = np.nanstd(data.reshape(-1, n_features), axis=0)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # Avoid division by zero

    data_normalized = (data - feature_means) / feature_stds

    train_set = {"X": data_normalized}

    # Scale model size based on number of samples
    # For few features (<= 8), use smaller model to avoid overfitting
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
        # Large cohort, short sequences, few features: stable transformer regime
        d_model, n_heads, d_k, d_v, d_ffn, n_layers = 128, 4, 32, 32, 256, 2
        epochs, batch_size, lr = 100, 512, 3e-4

    if save_training_log:
        log_path = Path(save_training_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"SAITS config: n_samples={n_samples}, n_steps={n_steps}, n_features={n_features}")
    logger.info(
        f"SAITS hyperparams: d_model={d_model}, d_ffn={d_ffn}, lr={lr}, batch_size={batch_size}"
    )

    from pypots.optim import AdamW

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
        saving_path=str(log_path) if save_training_log else None,
    )

    saits.fit(train_set)

    # Impute on the FULL dataset (normalized)
    full_set = {"X": data_normalized}
    imputed_data_normalized = saits.impute(full_set)

    # Denormalize back to original scale
    imputed_data = imputed_data_normalized * feature_stds + feature_means

    # Extract only the actual (non-padded) values for each sequence
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
