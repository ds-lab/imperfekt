"""Shared long-format <-> 3D sequence-array conversion.

Both SAITS training (``train_saits.py``) and SAITS inference (``imputation.py``)
need to turn a long-format Polars cohort into the ``(N, T, D)`` array.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def df_to_3d_array(
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
