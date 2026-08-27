import numpy as np
import polars as pl
from scipy.stats import spearmanr

from imperfekt.analysis.utils import pretty_printing

# Schema of the pairwise correlation table
_CORR_SCHEMA = {
    "axis_1": pl.Utf8,
    "axis_2": pl.Utf8,
    "corr": pl.Float64,
    "abs_corr": pl.Float64,
    "n_complete_cases": pl.Int64,
}


def assign_strata(
    df: pl.DataFrame,
    axis_x: str,
    axis_y: str,
    x_median: float,
    y_median: float,
    stratum_col: str = "stratum",
    inverted_axes: frozenset[str] | set[str] = frozenset(),
    complete_col: str | None = None,
    complete_value: float = 0.0,
    complete_label: str = "Q_complete",
    labels: tuple[str, str, str, str] = ("Q_alpha", "Q_beta", "Q_gamma", "Q_delta"),
) -> pl.DataFrame:
    """
    Assign each row to a quadrant by median-bisecting two axes.

    Quadrants, where "high" means more imperfect on that axis:

        Q_alpha  low x,  low y
        Q_beta   high x, low y
        Q_gamma  low x,  high y
        Q_delta  high x, high y

    Rows with a null on either axis get a null stratum — they cannot be placed.

    Thresholds are passed in rather than computed here, which is what makes the
    function usable for leakage-free cross-validation: fit medians on a training
    fold and apply them to a held-out fold, or fit once on a pooled cohort and apply
    to each subgroup so the labels stay comparable across groups.

    Parameters:
        df (pl.DataFrame): Frame containing the axis columns.
        axis_x (str): Column name for the x-axis metric.
        axis_y (str): Column name for the y-axis metric.
        x_median (float): Threshold bisecting the x-axis.
        y_median (float): Threshold bisecting the y-axis.
        stratum_col (str): Name of the output column.
        inverted_axes (frozenset[str]): Axes where a *lower* value means *more*
            imperfect (e.g. an adherence rate). For these, "high" is ``<= median``.
        complete_col (str | None): Column identifying rows with no imperfection at
            all, which are labelled ``complete_label`` instead of being placed in a
            quadrant. Pass None when the concept does not apply — an observation
            rhythm, for instance, always exists.
        complete_value (float): Value of ``complete_col`` marking such rows.
        complete_label (str): Label given to them.
        labels (tuple[str, str, str, str]): Quadrant labels, in the order
            (low/low, high/low, low/high, high/high).

    Returns:
        pl.DataFrame: df with the stratum column added.
    """
    x_high = pl.col(axis_x) <= x_median if axis_x in inverted_axes else pl.col(axis_x) > x_median
    y_high = pl.col(axis_y) <= y_median if axis_y in inverted_axes else pl.col(axis_y) > y_median

    q_alpha, q_beta, q_gamma, q_delta = labels
    unplaceable = pl.col(axis_x).is_null() | pl.col(axis_y).is_null()

    # The complete check comes first: such rows have no imperfection to stratify, and
    # their axis values are typically null, which would otherwise leave them unlabelled.
    if complete_col is not None:
        expr = (
            pl.when(pl.col(complete_col) == complete_value)
            .then(pl.lit(complete_label))
            .when(unplaceable)
            .then(pl.lit(None))
        )
    else:
        expr = pl.when(unplaceable).then(pl.lit(None))

    return df.with_columns(
        expr.when(~x_high & ~y_high)
        .then(pl.lit(q_alpha))
        .when(x_high & ~y_high)
        .then(pl.lit(q_beta))
        .when(~x_high & y_high)
        .then(pl.lit(q_gamma))
        .when(x_high & y_high)
        .then(pl.lit(q_delta))
        .otherwise(pl.lit(None))
        .alias(stratum_col)
    )


def pair_corr(df: pl.DataFrame, col_x: str, col_y: str) -> tuple[float, int]:
    """
    Spearman rank correlation between two columns over their complete pairs.

    Returns ``(nan, n_complete)`` when fewer than 3 complete pairs remain or when
    either column is constant — in both cases the correlation is undefined and the
    axis pair must not be selected.

    Parameters:
        df (pl.DataFrame): Frame containing both columns.
        col_x (str): First column name.
        col_y (str): Second column name.

    Returns:
        tuple[float, int]: (Spearman rho, number of complete pairs).
    """
    pair_df = df.select([col_x, col_y]).drop_nulls([col_x, col_y])
    n_complete = pair_df.height
    if n_complete < 3:
        return float("nan"), n_complete

    x = pair_df[col_x].to_numpy()
    y = pair_df[col_y].to_numpy()
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), n_complete
    return float(spearmanr(x, y).statistic), n_complete


def is_discriminating(df: pl.DataFrame, col: str, max_at_median: float = 0.5) -> bool:
    """
    Whether a candidate axis can meaningfully bisect the population.

    An axis where more than ``max_at_median`` of the values sit exactly on the
    median cannot split the cases into two halves — median-bisecting it produces a
    degenerate, near-single-quadrant assignment. Such axes are rejected before
    axis selection.

    Parameters:
        df (pl.DataFrame): Frame containing the column.
        col (str): Candidate axis column name.
        max_at_median (float): Maximum tolerated fraction of values equal to the median.

    Returns:
        bool: True if the axis discriminates, False if it is near-constant.
    """
    s = df[col].cast(pl.Float64).drop_nulls()
    if len(s) < 3:
        return False
    median = s.median()
    frac_at_median = (s == median).sum() / len(s)
    return float(frac_at_median) <= max_at_median


def pairwise_axis_correlations(
    df: pl.DataFrame,
    candidate_axes: list[str],
    discriminating_only: bool = False,
    max_at_median: float = 0.5,
) -> pl.DataFrame:
    """
    Correlation table for every pair of candidate axes, sorted most-orthogonal first.

    Parameters:
        df (pl.DataFrame): Frame containing the candidate axis columns.
        candidate_axes (list[str]): Axis column names to consider.
        discriminating_only (bool): If True, drop near-constant axes via is_discriminating().
        max_at_median (float): Threshold passed through to is_discriminating().

    Returns:
        pl.DataFrame: Columns axis_1, axis_2, corr, abs_corr, n_complete_cases,
                      sorted by abs_corr ascending (ties broken by more complete cases).
    """
    present_axes = [a for a in candidate_axes if a in df.columns]
    if discriminating_only:
        present_axes = [a for a in present_axes if is_discriminating(df, a, max_at_median)]

    corr_rows = []
    for i, ax_x in enumerate(present_axes):
        for ax_y in present_axes[i + 1 :]:
            corr, n_complete = pair_corr(df, ax_x, ax_y)
            corr_rows.append(
                {
                    "axis_1": ax_x,
                    "axis_2": ax_y,
                    "corr": corr,
                    "abs_corr": float(abs(corr)) if not np.isnan(corr) else float("nan"),
                    "n_complete_cases": n_complete,
                }
            )

    if not corr_rows:
        return pl.DataFrame(schema=_CORR_SCHEMA)

    return pl.DataFrame(corr_rows).sort(
        ["abs_corr", "n_complete_cases"], descending=[False, True], nulls_last=True
    )


def select_axis_pair(
    corr_table: pl.DataFrame,
    fallback_x: str,
    fallback_y: str,
    context: str = "",
) -> tuple[str, str, float]:
    """
    Pick the least-correlated (most orthogonal) axis pair from a correlation table.

    Falls back to the supplied default pair — with a warning — when no pair has a
    defined correlation, which happens with too few complete cases or when every
    candidate axis is constant.

    Parameters:
        corr_table (pl.DataFrame): Output of pairwise_axis_correlations().
        fallback_x (str): Default x-axis if no valid pair exists.
        fallback_y (str): Default y-axis if no valid pair exists.
        context (str): Label (e.g. a variable name) prefixed onto the warning.

    Returns:
        tuple[str, str, float]: (axis_x, axis_y, Spearman rho of the selected pair).
    """
    if corr_table.height > 0:
        valid_pairs = corr_table.filter(
            pl.col("corr").is_not_null() & pl.col("corr").is_not_nan()
        )
        if valid_pairs.height > 0:
            selected = valid_pairs.row(0, named=True)
            return selected["axis_1"], selected["axis_2"], float(selected["corr"])

    prefix = f"[{context}] " if context else ""
    pretty_printing.rich_warning(
        f"{prefix}Could not compute pairwise Spearman correlations for axis selection "
        "(too few complete cases or zero-variance metrics). "
        f"Falling back to default axes: {fallback_x} × {fallback_y}."
    )
    return fallback_x, fallback_y, float("nan")


def attach_axis_metadata(
    df: pl.DataFrame,
    axis_x: str,
    axis_y: str,
    corr: float,
    x_median: float,
    y_median: float,
) -> pl.DataFrame:
    """
    Add the five cohort-constant axis provenance columns to every row.

    These record which axes were selected, how orthogonal they were, and the median
    thresholds used to bisect them — so a stratified frame is self-documenting and
    the thresholds can be re-applied to another cohort.

    Parameters:
        df (pl.DataFrame): Frame to annotate.
        axis_x (str): Selected x-axis column name.
        axis_y (str): Selected y-axis column name.
        corr (float): Spearman rho between the selected axes.
        x_median (float): Median threshold applied to the x-axis.
        y_median (float): Median threshold applied to the y-axis.

    Returns:
        pl.DataFrame: df with axis_x, axis_y, axis_pair_corr,
                      axis_x_median_threshold, axis_y_median_threshold added.
    """
    return df.with_columns(
        pl.lit(axis_x).alias("axis_x"),
        pl.lit(axis_y).alias("axis_y"),
        pl.lit(corr).cast(pl.Float64).alias("axis_pair_corr"),
        pl.lit(x_median).cast(pl.Float64).alias("axis_x_median_threshold"),
        pl.lit(y_median).cast(pl.Float64).alias("axis_y_median_threshold"),
    )


def null_axis_metadata(
    df: pl.DataFrame,
    axis_x: str,
    axis_y: str,
    stratum_col: str,
) -> pl.DataFrame:
    """
    Annotate a frame that could not be stratified (fewer than 2 complete cases).

    The axis names are recorded so the output is still self-documenting, but the
    thresholds, correlation and stratum are all null — nothing was fitted.

    Parameters:
        df (pl.DataFrame): Frame to annotate.
        axis_x (str): Axis name that would have been used.
        axis_y (str): Axis name that would have been used.
        stratum_col (str): Name of the module's stratum column.

    Returns:
        pl.DataFrame: df with the axis columns and a null stratum column added.
    """
    return df.with_columns(
        pl.lit(axis_x).alias("axis_x"),
        pl.lit(axis_y).alias("axis_y"),
        pl.lit(None).cast(pl.Float64).alias("axis_pair_corr"),
        pl.lit(None).cast(pl.Float64).alias("axis_x_median_threshold"),
        pl.lit(None).cast(pl.Float64).alias("axis_y_median_threshold"),
        pl.lit(None).cast(pl.Utf8).alias(stratum_col),
    )
