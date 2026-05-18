import warnings
from pathlib import Path

import plotly.graph_objects as go
import polars as pl


def create_reference_range_mask(
    df: pl.DataFrame,
    reference_ranges: dict[str, tuple[int | float, int | float]],
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    clock_col: str = "clock",
    cols: list[str] | None = None,
    missing_as: str = "ignore",
) -> pl.DataFrame:
    """
    Create a plausibility mask using user-supplied fixed reference ranges.

    A value is flagged as implausible (1) when it falls outside the
    ``[lo, hi]`` bounds specified for its column. Columns absent from
    ``reference_ranges`` are not flagged (all zeros).

    Parameters:
        df (pl.DataFrame): Input DataFrame.
        reference_ranges (dict): Mapping of column name to ``(lo, hi)`` where
            either bound may be ``None`` (unbounded on that side).
            Example: ``{"heart_rate": (0, 300), "spo2": (0, 100)}``.
        id_col (str): ID column name.
        clock_no_col (str): Clock-number column name.
        clock_col (str): Clock/datetime column name.
        cols (list[str]): Columns to include in the output mask. If None,
            all non-ID columns are used.
        missing_as (str): How to treat originally-missing values.
            ``"ignore"`` (default) — flag as 0.
            ``"flag"``  — flag as 1.
            ``"null"``  — flag as null.

    Returns:
        pl.DataFrame: Same shape as df (restricted to cols + id cols),
            each data cell is 1 (implausible), 0 (plausible), or null (unknown).
    """
    if missing_as not in {"ignore", "flag", "null"}:
        raise ValueError(f"missing_as must be 'ignore', 'flag', or 'null', got {missing_as!r}")

    if cols is None:
        cols = [c for c in df.columns if c not in {id_col, clock_no_col, clock_col}]

    unmapped = [
        c
        for c in cols
        if c in df.select(pl.selectors.numeric()).columns and c not in reference_ranges
    ]
    if unmapped:
        warnings.warn(
            f"No reference range provided for columns {unmapped}. "
            "These columns will not be flagged.",
            stacklevel=2,
        )

    flag_series: list[pl.Series] = []
    for c in cols:
        was_null = df[c].is_null()
        lo, hi = reference_ranges.get(c, (None, None))

        below = (df[c] < lo) if lo is not None else pl.Series(c, [False] * len(df))
        above = (df[c] > hi) if hi is not None else pl.Series(c, [False] * len(df))
        outlier = (~was_null) & (below | above)

        if missing_as == "flag":
            flag_series.append((outlier | was_null).cast(pl.Int8).rename(c))
        elif missing_as == "null":
            flag_series.append(outlier.cast(pl.Int8).set(was_null, None).rename(c))
        else:
            flag_series.append(outlier.cast(pl.Int8).rename(c))

    mask = pl.DataFrame(flag_series)
    id_cols_to_add = [
        c for c in [id_col, clock_col, clock_no_col] if c in df.columns and c not in mask.columns
    ]
    if id_cols_to_add:
        mask = pl.concat([df.select(id_cols_to_add), mask], how="horizontal")
    return mask


def create_plausibility_mask(
    df: pl.DataFrame,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    clock_col: str = "clock",
    cols: list[str] | None = None,
    method: str | None = "iqr",
    threshold: float = 1.5,
    scope: str = "global",
    missing_as: str = "ignore",
    reference_ranges: dict[str, tuple[int | float, int | float]] | None = None,
) -> pl.DataFrame:
    """
    Create a plausibility mask, optionally combining reference range checks
    with a statistical outlier method.

    When ``reference_ranges`` is provided, it is applied first as a hard
    filter. The statistical method (if given) then runs only on values that
    passed the reference range check — so the bounds are not inflated by
    hard implausible values. The final mask is the union of both stages.

    When only ``reference_ranges`` is provided (``method=None``), the mask
    is purely range-based. When only ``method`` is provided, the mask is
    purely statistical.

    Two statistical methods are available:

    **IQR** (Tukey fences) — flags values outside
    [Q1 - threshold * IQR, Q3 + threshold * IQR].
    Typical thresholds: 1.5 (standard outlier), 3.0 (extreme outlier).

    **MAD** (modified Z-score, Iglewicz & Hoaglin 1993) — flags values where
    |0.6745 * (x - median) / MAD| > threshold.
    Handles skewed distributions better than IQR.
    Typical threshold: 3.5. When MAD = 0 (>50% identical values), the method
    is undefined and no values are flagged for that column/group.

    Parameters:
        df (pl.DataFrame): Input DataFrame.
        id_col (str): ID column name.
        clock_no_col (str): Clock-number column name.
        clock_col (str): Clock/datetime column name.
        cols (list[str]): Columns to evaluate. Non-numeric columns are skipped.
            If None, all non-ID columns are used.
        method (str | None): ``"iqr"`` (default), ``"mad"``, or ``None``
            (skip statistical step, use reference ranges only).
        threshold (float): Multiplier for IQR (default 1.5) or cutoff for the
            modified Z-score (default 3.5 for MAD).
        scope (str): ``"global"`` — bounds computed over the whole dataset;
            ``"per_id"`` — bounds computed per ID group.
        missing_as (str): How to treat originally-missing values in the mask.
            ``"ignore"`` (default) — flag as 0; pure outlier signal.
            ``"flag"``  — flag as 1; combines missingness and implausibility.
            ``"null"``  — flag as null; unknown plausibility.
        reference_ranges (dict | None): Optional mapping of column name to
            ``(lo, hi)`` hard domain bounds. Applied before the statistical
            method; the statistical method then sees only range-clean values (this avoids inflating the bounds with hard implausible values).
            Example: ``{"heart_rate": (0, 300), "spo2": (0, 100)}``.

    Returns:
        pl.DataFrame: Same shape as df (restricted to cols + id cols),
            each data cell is 1 (implausible), 0 (plausible), or null (unknown).
    """
    if method is not None and method not in {"iqr", "mad"}:
        raise ValueError(f"method must be 'iqr', 'mad', or None, got {method!r}")
    if scope not in {"global", "per_id"}:
        raise ValueError(f"scope must be 'global' or 'per_id', got {scope!r}")
    if missing_as not in {"ignore", "flag", "null"}:
        raise ValueError(f"missing_as must be 'ignore', 'flag', or 'null', got {missing_as!r}")
    if method is None and reference_ranges is None:
        raise ValueError("At least one of 'method' or 'reference_ranges' must be provided.")

    if cols is None:
        cols = [c for c in df.columns if c not in {id_col, clock_no_col, clock_col}]

    numeric_cols = df.select(pl.selectors.numeric()).columns
    analysis_cols = [c for c in cols if c in numeric_cols]

    def _apply_missing_as(outlier: pl.Series, was_null: pl.Series) -> pl.Series:
        name = outlier.name
        if missing_as == "flag":
            return (outlier | was_null).cast(pl.Int8).rename(name)
        if missing_as == "null":
            return outlier.cast(pl.Int8).set(was_null, None).rename(name)
        return outlier.cast(pl.Int8).rename(name)

    def _iqr_bounds_global(series: pl.Series) -> tuple[float, float] | None:
        q1 = series.quantile(0.25, interpolation="linear")
        q3 = series.quantile(0.75, interpolation="linear")
        if q1 is None or q3 is None:
            return None
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr

    def _mad_bounds_global(series: pl.Series) -> tuple[float, float] | None:
        med = series.drop_nulls().median()
        if not isinstance(med, (int, float)):
            return None
        mad_val = (series - med).abs().drop_nulls().median()
        if not isinstance(mad_val, (int, float)) or mad_val == 0:
            warnings.warn(
                "MAD=0 (global): more than 50% of values are identical, "
                "modified Z-score is undefined. No values will be flagged.",
                stacklevel=3,
            )
            return None
        half_width = threshold * mad_val / 0.6745
        return float(med - half_width), float(med + half_width)

    def _iqr_bounds_per_id(series_df: pl.DataFrame, c: str) -> pl.DataFrame:
        return (
            series_df.group_by(id_col)
            .agg(
                pl.col(c).quantile(0.25, interpolation="linear").alias("_q1"),
                pl.col(c).quantile(0.75, interpolation="linear").alias("_q3"),
            )
            .with_columns((pl.col("_q3") - pl.col("_q1")).alias("_iqr"))
            .with_columns(
                (pl.col("_q1") - threshold * pl.col("_iqr")).alias("_lo"),
                (pl.col("_q3") + threshold * pl.col("_iqr")).alias("_hi"),
            )
            .select([id_col, "_lo", "_hi"])
        )

    def _mad_bounds_per_id(series_df: pl.DataFrame, c: str) -> pl.DataFrame:
        med_df = series_df.group_by(id_col).agg(pl.col(c).median().alias("_med"))
        with_med = series_df.join(med_df, on=id_col, how="left")
        mad_df = (
            with_med.with_columns((pl.col(c) - pl.col("_med")).abs().alias("_dev"))
            .group_by(id_col)
            .agg(pl.col("_dev").median().alias("_mad"))
        )
        bounds = med_df.join(mad_df, on=id_col, how="left")
        zero_mad_ids = bounds.filter(pl.col("_mad") == 0)[id_col].to_list()
        if zero_mad_ids:
            warnings.warn(
                f"MAD=0 for column '{c}' in groups {zero_mad_ids}: "
                "more than 50% of values are identical, modified Z-score is undefined. "
                "No values will be flagged for these groups.",
                stacklevel=3,
            )
        return (
            bounds.with_columns(
                pl.when(pl.col("_mad") == 0)
                .then(pl.lit(None))
                .otherwise(threshold * pl.col("_mad") / 0.6745)
                .alias("_hw")
            )
            .with_columns(
                (pl.col("_med") - pl.col("_hw")).alias("_lo"),
                (pl.col("_med") + pl.col("_hw")).alias("_hi"),
            )
            .select([id_col, "_lo", "_hi"])
        )

    flag_series: list[pl.Series] = []
    for c in cols:
        was_null = df[c].is_null()

        if c not in analysis_cols:
            null_flag = was_null.cast(pl.Int8).rename(c)
            flag_series.append(
                null_flag.set(was_null, None).rename(c) if missing_as == "null" else null_flag
            )
            continue

        # --- Stage 1: reference range ---
        ref_outlier = pl.Series(c, [False] * len(df))
        if reference_ranges and c in reference_ranges:
            lo, hi = reference_ranges[c]
            below = (df[c] < lo) if lo is not None else pl.Series(c, [False] * len(df))
            above = (df[c] > hi) if hi is not None else pl.Series(c, [False] * len(df))
            ref_outlier = (~was_null) & (below | above)

        # --- Stage 2: statistical method on reference-range-clean values only ---
        stat_outlier = pl.Series(c, [False] * len(df))
        if method is not None:
            # Null out values that failed the reference range check before computing bounds
            clean_series = df[c].set(ref_outlier, None)
            clean_df = df.with_columns(clean_series.alias(c))

            if scope == "global":
                bounds = (
                    _iqr_bounds_global(clean_series)
                    if method == "iqr"
                    else _mad_bounds_global(clean_series)
                )
                if bounds is not None:
                    lo_s, hi_s = bounds
                    stat_outlier = (~was_null) & (~ref_outlier) & ((df[c] < lo_s) | (df[c] > hi_s))
            else:  # per_id
                bounds_df = (
                    _iqr_bounds_per_id(clean_df, c)
                    if method == "iqr"
                    else _mad_bounds_per_id(clean_df, c)
                )
                joined = df.join(bounds_df, on=id_col, how="left")
                stat_outlier = (
                    (~was_null)
                    & (~ref_outlier)
                    & (
                        (joined[c] < joined["_lo"]).fill_null(False)
                        | (joined[c] > joined["_hi"]).fill_null(False)
                    )
                )

        outlier = ref_outlier | stat_outlier
        flag_series.append(_apply_missing_as(outlier.rename(c), was_null))

    mask = pl.DataFrame(flag_series)
    id_cols_to_add = [
        c for c in [id_col, clock_col, clock_no_col] if c in df.columns and c not in mask.columns
    ]
    if id_cols_to_add:
        mask = pl.concat([df.select(id_cols_to_add), mask], how="horizontal")
    return mask


def create_missingness_mask(
    df: pl.DataFrame,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    clock_col: str = "clock",
    cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Create a missingness masking matrix.
    This function returns a Polars DataFrame where each cell is 1 if the value was missing (null) and 0 otherwise.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame to analyze for missing values.

        cols (list[str], optional): List of columns to include in the missingness matrix.
                                    If None, all columns will be used.

    Returns:
        pl.DataFrame: A Polars DataFrame with the same shape as the input (or selected columns),
                      where each cell is either 1 (missing) or 0 (not missing).
    """
    if cols is None:
        cols = [c for c in df.columns if c not in [id_col, clock_no_col, clock_col]]

    missing_mask = df.select(pl.col(c).is_null().cast(pl.Int8).alias(c) for c in cols)

    # Add columns that were not in `cols` but are needed for joins/identification
    id_cols_to_add = [
        c
        for c in [id_col, clock_col, clock_no_col]
        if c in df.columns and c not in missing_mask.columns
    ]
    if id_cols_to_add:
        missing_mask = pl.concat([df.select(id_cols_to_add), missing_mask], how="horizontal")

    return missing_mask


def create_missingness_mask_long_table(
    df: pl.DataFrame,
    cols: list | None = None,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
) -> pl.DataFrame:
    """
    Create a missingness matrix for the DataFrame ordered by id (rows) and showing clock_no as columns.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame to analyze for missing values.
        cols (list): List of columns to include in the missingness matrix. If None, all columns will be used.
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".

    Returns:
        pl.DataFrame: A Polars DataFrame with the missingness matrix, where each cell is 1 if the value was missing (null) and 0 otherwise.
                        The DataFrame will have the following structure:
                        - Index: [id_col, variable]
                        - Columns: clock_no_col
                        - Values: 1 for missing, 0 for present.

    Notes:
        - Caveat: Time-series data must be ordered by clock_no for each id and visualizes time-series as *equi-distant* points.
    """
    if cols is None:
        cols = df.columns
    long = df.unpivot(
        index=[id_col, clock_no_col],
        on=cols,
        variable_name="variable",
        value_name="value",
    )

    # Add missingness indicator
    long = long.with_columns((pl.col("value").is_null().cast(pl.Int8)).alias("is_missing"))
    long = long.drop("value")  # Drop the value column as we only need the missingness indicator

    # Pivot to matrix format
    missing_matrix = long.pivot(
        index=[id_col, "variable"],
        on=clock_no_col,
        values="is_missing",
    )

    return missing_matrix


def create_missingness_mask_per_col_long_table(
    df, col: str, id_col: str = "id", clock_no_col: str = "clock_no"
) -> pl.DataFrame:
    """
    Plot a missingness matrix for a specific column in the DataFrame ordered by id (rows) and showing clock_no as columns.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame to analyze for missing values.
        col (str): The column name to create the missingness mask for.
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".

    Returns:
        pl.DataFrame: A Polars DataFrame with the missingness matrix, where each cell is 1 if the value was missing (null) and 0 otherwise.
    """
    # Create a boolean mask for missing values in the specified column
    missing_mask = df.select(pl.col(col).is_null().cast(pl.Int8).alias("is_missing"))

    # Add clock_no and id columns to the mask
    missing_mask = missing_mask.with_columns(df.select([id_col, clock_no_col]))

    # Pivot the DataFrame to create a matrix format
    missing_matrix = missing_mask.pivot(index=id_col, on=clock_no_col, values="is_missing")

    return missing_matrix


def plot_missingness_mask(
    missing_matrix: pl.DataFrame,
    title: str = "Missingness Heatmap",
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    renderer: str | None = "browser",
    save_path: str | Path | None = None,
    save_results: bool = True,
) -> None:
    """
    Plot a heatmap of the missingness matrix using Plotly.
    Visualizes the missingness of each variable over time (order not datetime) for each ID.
    Allows filtering by ID using a dropdown menu.

    Parameters:
        missing_matrix (pl.DataFrame): The Polars DataFrame containing the missingness matrix.
        title (str): The title of the heatmap plot. Defaults to "Missingness Heatmap".
        id_col (str): The name of the column representing the unique identifier for each row. Defaults to "id".
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). Defaults to "clock_no".
        renderer (str): The renderer to use for displaying the plot. Defaults to "browser".
        save_path (str | Path, optional): Path to save the plot as an HTML file. If None, the plot will not be saved.
        save_results (bool): Whether to save the plot as an HTML file. Defaults to True

    Returns:
        None: Displays the heatmap plot in the browser.
    """
    # 1) Convert Polars DataFrame to pandas so we can easily extract a numpy array
    pdf = missing_matrix.to_pandas()
    if "variable" in pdf.columns:
        # If the DataFrame has a "variable" column, we can use it to create a multi-index
        pdf.set_index([id_col, "variable"], inplace=True)
        y_labels = [f"{pid} | {var}" for (pid, var) in pdf.index]
    else:
        # If not, we just set id_col as the index
        pdf.set_index(id_col, inplace=True)
        y_labels = pdf.index.astype(str).tolist()

    clock_no = pdf.columns.astype(str).tolist()
    data_matrix = pdf.fillna(0.5).values  # shape = (n_rows, n_clock_no)

    # 3) Build a Heatmap with Plotly

    fig = go.Figure(
        go.Heatmap(
            z=data_matrix,
            x=clock_no,
            y=y_labels,
            colorscale=[
                [0.0, "darkolivegreen"],
                [0.5, "rgb(211, 211, 211)"],
                [1.0, "firebrick"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title="Missingness", tickvals=[0, 1], ticktext=["Present", "Missing"]),
        )
    )

    if "variable" in pdf.index.names:
        # Extract unique IDs
        dropdown_buttons = []
        unique_ids = pdf.index.get_level_values(id_col).unique().sort_values().tolist()
        for pid in unique_ids:
            # Boolean mask for rows with this ID
            mask = pdf.index.get_level_values(id_col) == pid
            # y_labels_id: just the subset of ["pid | variable"] for this ID
            y_labels_id = [f"{_pid} | {var}" for (_pid, var) in pdf.index[mask]]
            # z_id: the sub‐matrix of shape (n_vars_for_pid, n_clock_no)
            z_id = data_matrix[mask, :]

            dropdown_buttons.append(
                {
                    "label": str(pid),
                    "method": "update",
                    "args": [
                        {
                            "y": [y_labels_id],  # override y to just this ID’s rows
                            "z": [z_id],  # override z to just this ID’s submatrix
                        },
                        {
                            "yaxis": {"title": f"ID: {pid} | Variable"},
                        },
                    ],
                }
            )

        # Finally, add an “All” button at the top
        dropdown_buttons.insert(
            0,
            {
                "label": "All",
                "method": "update",
                "args": [
                    {
                        "y": [y_labels],
                        "z": [data_matrix],
                    },
                    {
                        "yaxis": {"title": "(id | variable)"},
                    },
                ],
            },
        )

        # Attach updatemenus
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": dropdown_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.15,
                    "y": 0.9,
                    "xanchor": "left",
                    "yanchor": "top",
                }
            ]
        )

    # 5) (Optional) Tweak layout
    fig.update_layout(
        title=title,
        xaxis_title=clock_no_col,
        yaxis_title=id_col,
        xaxis_nticks=len(clock_no),
        yaxis_autorange="reversed",  # so the first ID appears at the top
    )
    if renderer:
        fig.show(renderer=renderer)  # Open in browser for better visibility

    if save_path and save_results:
        fig.write_html(save_path)
        print(f"Missingness heatmap saved to {save_path}")


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "clock_no": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "clock": ["2024-01-01", "2024-01-02", "2024-01-03"] * 3,
            # id=1: normal; id=2: one hard outlier (999), one missing; id=3: one soft outlier (130)
            "heartrate": [70, 72, 74, 999, None, 80, 75, 130, 78],
            # id=1: normal; id=2: all missing; id=3: one below hard lower bound (-5)
            "resprate": [16, 18, 17, None, None, None, -5, 22, 21],
        }
    )

    reference_ranges: dict[str, tuple[int | float, int | float]] = {
        "heartrate": (0.0, 300.0),
        "resprate": (0.0, 60.0),
    }

    print("=== Missingness mask ===")
    print(create_missingness_mask(df, cols=["heartrate", "resprate"]))

    print("\n=== Reference range mask (missing_as='ignore') ===")
    print(create_reference_range_mask(df, reference_ranges=reference_ranges))

    print("\n=== Reference range mask (missing_as='flag') ===")
    print(create_reference_range_mask(df, reference_ranges=reference_ranges, missing_as="flag"))

    print("\n=== Reference range mask (missing_as='null') ===")
    print(create_reference_range_mask(df, reference_ranges=reference_ranges, missing_as="null"))

    print("\n=== Plausibility mask: reference ranges only (method=None) ===")
    print(create_plausibility_mask(df, method=None, reference_ranges=reference_ranges))

    print("\n=== Plausibility mask: IQR only, global ===")
    print(create_plausibility_mask(df, method="iqr", threshold=1.5, scope="global"))

    print("\n=== Plausibility mask: IQR + reference ranges, global ===")
    print(
        create_plausibility_mask(
            df, method="iqr", threshold=1.5, scope="global", reference_ranges=reference_ranges
        )
    )

    print("\n=== Plausibility mask: IQR + reference ranges, per_id ===")
    print(
        create_plausibility_mask(
            df, method="iqr", threshold=1.5, scope="per_id", reference_ranges=reference_ranges
        )
    )

    print("\n=== Plausibility mask: MAD + reference ranges, global ===")
    print(
        create_plausibility_mask(
            df, method="mad", threshold=3.5, scope="global", reference_ranges=reference_ranges
        )
    )

    missing_matrix = create_missingness_mask_per_col_long_table(df, "heartrate")
    print("\n=== Missingness long table (heartrate) ===")
    print(missing_matrix)

    missing_matrix = create_missingness_mask_long_table(df, cols=["heartrate", "resprate"])
    print("\n=== Missingness long table (all cols) ===")
    print(missing_matrix)
