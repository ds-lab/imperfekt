import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from seq_array import df_to_3d_array

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

    trained_n_steps = int(np.asarray(meta["n_steps"]).reshape(-1)[0])

    ids, data, mask, seq_lengths = df_to_3d_array(
        df, id_col=id_col, time_col=time_col, feature_cols=cols
    )
    n_samples, n_steps, n_features = data.shape

    # SAITS is fixed-length: it cannot process a sequence longer than the T it was
    # trained on. 
    n_truncated = int((seq_lengths > trained_n_steps).sum())
    if n_steps > trained_n_steps:
        logger.warning(
            "SAITS: %d/%d inference stay(s) longer than trained n_steps=%d; "
            "truncating to the first %d readings (later readings left un-imputed).",
            n_truncated,
            n_samples,
            trained_n_steps,
            trained_n_steps,
        )
        data = data[:, :trained_n_steps, :]
        mask = mask[:, :trained_n_steps, :]
        n_steps = trained_n_steps

    observed_any = mask.any(axis=1)  # (N, D): True if observed at least once
    missing_all = ~observed_any  # (N, D): True if 100% missing for that patient

    # Normalize using the SAME stats the model was trained with.
    data_normalized = (data - feature_means) / feature_stds
    
    if n_steps < trained_n_steps:
        pad = np.full((n_samples, trained_n_steps - n_steps, n_features), np.nan)
        model_input = np.concatenate([data_normalized, pad], axis=1)
    else:
        model_input = data_normalized

    json_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"SAITS architecture meta not found at {json_path}. "
            "Re-train with examples/nemsis/train_saits.py to regenerate it."
        )
    import torch

    hp = json.loads(json_path.read_text())["hyperparameters"]
    inference_batch_size = 131072
    saits = SAITS(
        n_steps=trained_n_steps,
        n_features=n_features,
        n_layers=hp["n_layers"],
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        d_k=hp["d_k"],
        d_v=hp["d_v"],
        d_ffn=hp["d_ffn"],
        dropout=hp["dropout"],
        batch_size=inference_batch_size,
    )
    saits.load(str(model_path))
    saits.model.eval()

    n_batches = (n_samples + inference_batch_size - 1) // inference_batch_size
    print(f"SAITS inference: {n_samples} samples in {n_batches} batches of {inference_batch_size}")
    batch_results = []
    with torch.no_grad():
        for i, start in enumerate(range(0, n_samples, inference_batch_size)):
            batch_np = model_input[start : start + inference_batch_size]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=saits.device)
            missing_mask = (~torch.isnan(batch_tensor)).float()
            batch_tensor = torch.nan_to_num(batch_tensor, nan=0.0)
            inputs = {"X": batch_tensor, "missing_mask": missing_mask}
            result = saits.model(inputs)
            batch_results.append(result["imputation"].cpu().numpy())
            del batch_tensor, missing_mask, inputs, result
            torch.cuda.empty_cache()
            print(f"SAITS inference: batch {i+1}/{n_batches}")

    del saits
    torch.cuda.empty_cache()

    print("SAITS inference complete; post-processing results…")
    imputed_data_normalized = np.concatenate(batch_results, axis=0)
    if imputed_data_normalized.ndim == 4:
        imputed_data_normalized = imputed_data_normalized.squeeze(1)
    # Drop the padding steps so the array lines up with the (truncated) time axis.
    imputed_data_normalized = imputed_data_normalized[:, :n_steps, :]

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

    # Un-pad
    imputed_values_list = []
    for i, length in enumerate(seq_lengths):
        n_imp = min(int(length), n_steps)  # rows the model actually imputed
        imputed_values_list.append(imputed_data[i, :n_imp, :])
    imputed_flat = np.vstack(imputed_values_list)

    df_sorted = df.sort([id_col, time_col])
    if n_truncated:
        original_flat = (
            df_sorted.select(cols).with_columns([pl.all().cast(pl.Float64)]).to_numpy()
        )
        keep = np.ones(original_flat.shape[0], dtype=bool)
        row = 0
        for length in seq_lengths:
            n_imp = min(int(length), n_steps)
            keep[row : row + n_imp] = False  # these get overwritten by SAITS
            row += int(length)
        merged = original_flat.copy()
        merged[~keep] = imputed_flat
        imputed_flat = merged

    imputed_cols = [pl.Series(name=col, values=imputed_flat[:, i]) for i, col in enumerate(cols)]

    df_imputed_sorted = df_sorted.with_columns(imputed_cols)
    
    print(f"SAITS imputation applied to {len(cols)} columns: {cols}")

    return (
        df_imputed_sorted.join(
            df.select(id_col, time_col).with_row_index("row_idx"),
            on=[id_col, time_col],
            how="inner",
        )
        .sort("row_idx")
        .drop("row_idx")
    )
