import plotly.graph_objects as go
import polars as pl


def create_plausibility_mask(
    df: pl.DataFrame,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    clock_col: str = "clock",
    cols: list[str] | None = None,
    iqr_multiplier: float = 1.5,
    scope: str = "global",
    missing_as: str = "ignore",
) -> pl.DataFrame:
    """
    Create a plausibility mask using the IQR (Tukey fence) method.

    A value is flagged as implausible (1) when it falls outside
    [Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR].

    Parameters:
        df (pl.DataFrame): Input DataFrame.
        id_col (str): ID column name.
        clock_no_col (str): Clock-number column name.
        clock_col (str): Clock/datetime column name.
        cols (list[str]): Numeric columns to evaluate. Non-numeric columns
            are silently skipped. If None, all non-ID columns are used.
        iqr_multiplier (float): Tukey fence multiplier (default 1.5;
            use 3.0 for "extreme outlier" fences).
        scope (str): ``"global"`` — IQR computed over the whole dataset;
            ``"per_id"`` — IQR computed per ID group.
        missing_as (str): How to treat originally-missing values in the mask.
            ``"ignore"`` (default) — flag as 0 (plausible); pure outlier signal.
            ``"flag"``  — flag as 1; combines missingness and implausibility.
            ``"null"``  — flag as null; unknown plausibility.

    Returns:
        pl.DataFrame: Same shape as df (restricted to cols + id cols),
            each data cell is 1 (implausible), 0 (plausible), or null (unknown).
    """
    if missing_as not in {"ignore", "flag", "null"}:
        raise ValueError(f"missing_as must be 'ignore', 'flag', or 'null', got {missing_as!r}")
    if cols is None:
        cols = [c for c in df.columns if c not in {id_col, clock_no_col, clock_col}]

    numeric_cols = df.select(pl.selectors.numeric()).columns
    analysis_cols = [c for c in cols if c in numeric_cols]

    if scope not in {"global", "per_id"}:
        raise ValueError(f"scope must be 'global' or 'per_id', got {scope!r}")

    def _apply_missing_as(outlier: pl.Series, was_null: pl.Series) -> pl.Series:
        """Combine outlier flag with original-null handling per missing_as policy."""
        name = outlier.name
        if missing_as == "flag":
            return (outlier | was_null).cast(pl.Int8)
        if missing_as == "null":
            return outlier.cast(pl.Int8).set(was_null, None).rename(name)
        # "ignore": missing → 0, only real outliers → 1
        return outlier.cast(pl.Int8).rename(name)

    flag_series: list[pl.Series] = []
    for c in cols:
        was_null = df[c].is_null()

        if c not in analysis_cols:
            # Non-numeric: no IQR check possible; apply missing_as directly
            null_flag = was_null.cast(pl.Int8).rename(c)
            if missing_as == "null":
                flag_series.append(null_flag.set(was_null, None).rename(c))
            else:
                flag_series.append(null_flag)
            continue

        if scope == "global":
            q1 = df[c].quantile(0.25, interpolation="linear")
            q3 = df[c].quantile(0.75, interpolation="linear")
            if q1 is None or q3 is None:
                flag_series.append(_apply_missing_as(was_null, was_null))
                continue
            iqr = q3 - q1
            lo = q1 - iqr_multiplier * iqr
            hi = q3 + iqr_multiplier * iqr
            outlier = (~was_null) & ((df[c] < lo) | (df[c] > hi))
            flag_series.append(_apply_missing_as(outlier.rename(c), was_null))
        else:  # per_id — evaluate eagerly so _lo/_hi are resolved before they are gone
            bounds = (
                df.group_by(id_col)
                .agg(
                    pl.col(c).quantile(0.25, interpolation="linear").alias("_q1"),
                    pl.col(c).quantile(0.75, interpolation="linear").alias("_q3"),
                )
                .with_columns((pl.col("_q3") - pl.col("_q1")).alias("_iqr"))
                .with_columns(
                    (pl.col("_q1") - iqr_multiplier * pl.col("_iqr")).alias("_lo"),
                    (pl.col("_q3") + iqr_multiplier * pl.col("_iqr")).alias("_hi"),
                )
                .select([id_col, "_lo", "_hi"])
            )
            joined = df.join(bounds, on=id_col, how="left")
            outlier = (~was_null) & (
                (joined[c] < joined["_lo"]) | (joined[c] > joined["_hi"])
            )
            flag_series.append(_apply_missing_as(outlier.rename(c), was_null))

    mask = pl.DataFrame(flag_series)

    id_cols_to_add = [
        c for c in [id_col, clock_col, clock_no_col]
        if c in df.columns and c not in mask.columns
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
    renderer: str = "browser",
    save_path: str | None = None,
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
        save_path (str, optional): Path to save the plot as an HTML file. If None, the plot will not be saved.
        save_result (bool): Whether to save the plot as an HTML file. Defaults to True

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
    # Example usage
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "clock_no": [1, 2, 1, 2, 1, 2],
            "heartrate": [70, None, None, 80, 75, None],
            "resprate": [16, 18, None, None, 20, 22],
        }
    )
    mask = create_missingness_mask(df, cols=["heartrate"])
    print(mask)

    missing_matrix = create_missingness_mask_per_col_long_table(df, "heartrate")
    print(missing_matrix)
    plot_missingness_mask(missing_matrix)

    missing_matrix = create_missingness_mask_long_table(df, cols=["heartrate", "resprate"])
    print(missing_matrix)
    plot_missingness_mask(missing_matrix)
