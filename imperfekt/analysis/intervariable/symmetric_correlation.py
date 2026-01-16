import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import polars as pl
import scipy.cluster.hierarchy as sch
from scipy.stats import chi2_contingency

############################################################
#               CO-IMPERFECTION STATISTICS                  #
############################################################


def chi2_intervariable_imperfection_statistics(
    mask_df: pl.DataFrame, col_i: str, col_j: str
) -> float:
    """
    Calculate chi-squared statistics for imperfection association between two columns.
    Computes the chi-squared statistic, p-value, degrees of freedom, observed counts,
    expected counts, and Cramér's V statistic for the association of imperfection
    between two columns in a DataFrame.

    Parameters:
        mask_df (pl.DataFrame): DataFrame with binary values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected).
        col_i (str): The first column name to check for imperfection.
        col_j (str): The second column name to check for imperfection.

    Returns:
        dict: A dictionary containing the chi-squared statistic, p-value, degrees of freedom, observed counts, expected counts, and Cramér's V statistic.
            - chi2: The chi-squared statistic.
            - p_value: The p-value of the chi-squared test. Is considered significant if it is less than the significance level (commonly 0.05).
            - dof: The degrees of freedom of the chi-squared test.
            - observed: The observed counts of imperfection in a 2x2 contingency table.
            - expected: The expected counts of imperfection in a 2x2 contingency table.
            - cramer_v: Cramér's V statistic, which measures the association between two categorical variables.
                - Cramér's V can be categorizesd as follows (https://www.scribbr.de/statistik/cramers-v/):
                    - 0.00: Negligible association
                    - 0.10: Weak association
                    - 0.30: Moderate association
                    - 0.50: Strong association
                    - 1.00: Perfect association
    """
    df2 = mask_df.with_columns(
        [
            pl.col(col_i).alias("Mi"),
            pl.col(col_j).alias("Mj"),
        ]
    )
    cont = (
        df2.group_by(["Mi", "Mj"])
        .agg(pl.len().alias("n"))
        .to_pandas()
        .pivot(index="Mi", columns="Mj", values="n")
        .fillna(0)
    )
    # Ensure 2x2 shape
    for idx in [0, 1]:
        if idx not in cont.index:
            cont.loc[idx] = 0
    for col in [0, 1]:
        if col not in cont.columns:
            cont[col] = 0
    cont = cont.sort_index().sort_index(axis=1)
    observed = cont.loc[[0, 1], [0, 1]].values

    chi2, p, dof, expected = chi2_contingency(observed)

    n_total = observed.sum()
    cramer_v = np.sqrt(chi2 / n_total)

    return {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "observed": observed,
        "expected": expected,
        "cramer_v": cramer_v,
    }


def chi2_intervariable_imperfection_matrix(
    mask_df: pl.DataFrame,
    cols: list[str] = None,
    save_path: str = None,
    save_results: bool = True,
) -> pl.DataFrame:
    """
    Create a matrix of chi-squared statistics for imperfection association between variables.

    Parameters:
        mask_df (pl.DataFrame): DataFrame with binary values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected).
        cols (list): List of column names to include in the matrix.
        save_path (str): Path to save the resulting DataFrame as a CSV file. If None, the DataFrame will not be saved.
        save_results (bool): Whether to save the results to the specified path.

    Returns:
        pl.DataFrame: A DataFrame containing the chi-squared statistics for imperfection association.
                      The statistics are stored in a struct for each pair of variables.
    """
    if cols is None:
        cols = mask_df.columns
    data = []
    for i in cols:
        for j in cols:
            if i == j:
                continue
            else:
                stats = chi2_intervariable_imperfection_statistics(mask_df, i, j)
                data.append(
                    {
                        "col_i": i,
                        "col_j": j,
                        "chi2": stats["chi2"],
                        "p_value": stats["p_value"],
                        "dof": stats["dof"],
                        # "observed": stats["observed"].tolist(),
                        # "expected": stats["expected"].tolist(),
                        "cramer_v": stats["cramer_v"],
                    }
                )
    data = pl.DataFrame(data)

    if save_path and save_results:
        # Save the results to a CSV file if a path is provided
        data.write_csv(save_path)
        print(f"Chi-squared matrix saved to {save_path}")

    return data


def corr_matrix(
    mask_df: pl.DataFrame,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    clock_col: str = "clock",
) -> pl.DataFrame:
    """
    Computes the correlation matrix of imperfection across columns in a DataFrame.

    Parameters:
        mask_df (pl.DataFrame): with boolean values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected).

    Returns:
        pl.DataFrame containing the correlation matrix of imperfection.
    """
    mask_df = mask_df.select(
        [pl.col(c) for c in mask_df.columns if c not in [id_col, clock_no_col, clock_col]]
    )
    return mask_df.corr().fill_null(0)


def lagged_cross_correlation(
    mask_df: pl.DataFrame,
    col_x: str,
    col_y: str,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    max_lag: int = 10,
):
    """
    Compute lagged cross-correlation between imperfection indicators of two variables.
    The function computes the cross-correlation for lags from -max_lag to +max_lag, where negative lags indicate
    that col_x leads col_y, and positive lags indicate that col_y leads col_x.
    Order of col_x and col_y does not matter; however the order of the cross-correlation values will be flipped.

    Example:
    If col_x leads col_y by 2 time points, the cross-correlation at lag -2 will be positive.
    Meaning that if col_x is missing, col_y is likely to be missing 2 time points later.
    Or: At a positive lag, the cross-correlation indicates how likely col_y is to be missing after col_x is missing.

    Parameters:
        mask_df (pl.DataFrame): DataFrame with binary values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected).
        col_x (str): First variable.
        col_y (str): Second variable.
        id_col (str): ID column.
        clock_no_col (str): Time ordering column.
        max_lag (int): Maximum lag (both positive and negative) to compute.
        save_path (str): Path to save the results CSV (optional).
        save_results (bool): Whether to save the results (default: True).

    Returns:
        lags (np.ndarray): Array of lag values (negative to positive).
        crosscorrs (np.ndarray): Cross-correlation at each lag.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    elif max_lag > mask_df.height:
        max_lag = mask_df.height - 1

    lags = np.arange(-max_lag, max_lag + 1)

    # Create imperfection indicators once
    mask_df = mask_df.sort([id_col, clock_no_col])

    # Check for constant series (zero variance)
    stats = mask_df.select(
        pl.var(col_x).alias("var_x"),
        pl.var(col_y).alias("var_y"),
    )

    if stats["var_x"][0] == 0 or stats["var_y"][0] == 0:
        raise ValueError(
            "One or both series are constant (all indicated or no indicated values). Cross-correlation is undefined."
        )

    # Calculate all correlations in a single expression
    # This is much more efficient than a Python loop
    corr_results = mask_df.select(
        [
            pl.corr(col_x, pl.col(col_y).shift(lag).over(id_col)).alias(f"corr_lag_{lag}")
            for lag in lags
        ]
    )

    # Extract results into a numpy array
    crosscorrs = corr_results.row(0)

    ccfs = pl.DataFrame({"lag": lags, "crosscorr": crosscorrs})

    return ccfs


############################################################
#      CO-IMPERFECTION EXPLORATION AND VISUALIZATION        #
############################################################


def co_imperfection_heatmap(
    corr_matrix: pl.DataFrame,
    renderer: str = "browser",
    save_path: str = None,
    save_results: bool = True,
) -> None:
    """
    Visualizes the correlation matrix of imperfection using Plotly.

    Parameters:
        corr_matrix (pl.DataFrame): DataFrame containing the correlation matrix of imperfection extracted by `symmetric_correlation`.
        renderer (str): Renderer to use for displaying the plot. Default is "browser".
        save_path (str): Path to save the figure. If None, the figure will not be saved.
        save_results (bool): Whether to save the figure to the specified path. Default is False.

    Returns:
        None: Displays the heatmap visualization.
    """
    cols = corr_matrix.columns
    corr_matrix_values = corr_matrix.to_pandas().values
    text_labels = corr_matrix_values.round(4).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix_values,
            x=cols,
            y=cols,
            text=text_labels,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Phi coefficient (Φ)"),
        )
    )

    fig.update_layout(
        title="Imperfection Correlation Heatmap",
        xaxis=dict(side="top"),
        width=700,
        height=700,
        yaxis_autorange="reversed",  # So that the first row is at the top
        template="plotly_white",
    )

    if renderer:
        fig.show(renderer=renderer)

    if save_results and save_path:
        fig.write_image(save_path)
        print(f"Heatmap saved to {save_path}")

    return fig


def co_imperfection_dendogram(
    corr_matrix: pl.DataFrame,
    renderer: str = "browser",
    save_path: str = None,
    save_results: bool = True,
) -> None:
    """
    Visualizes the co-imperfection dendrogram using scipy and plotly.

    Parameters:
        corr_matrix (pl.DataFrame): DataFrame containing the correlation matrix of imperfection extracted by `symmetric_correlation`.
        renderer (str): Renderer to use for displaying the plot. Default is "browser".
        save_path (str): Path to save the figure. If None, the figure will not be saved.
        save_results (bool): Whether to save the figure to the specified path. Default is False.
    Returns:
        None: Displays the dendrogram visualization.
    """
    corr_matrix = corr_matrix.to_pandas()

    # 1) Compute the raw distance matrix
    dist = 1.0 - corr_matrix.values

    # 2) Force exact symmetry (average with its transpose)
    dist = (dist + dist.T) / 2.0

    # 3) Zero‐out the diagonal (distance to itself must be 0)
    np.fill_diagonal(dist, 0.0)

    # 4) Convert to “condensed” form and run linkage
    condensed = sch.distance.squareform(dist)
    linkage = sch.linkage(condensed, method="average")

    # 5) Create the dendrogram
    fig = ff.create_dendrogram(
        corr_matrix.values,
        orientation="bottom",
        labels=corr_matrix.columns.tolist(),
        linkagefun=lambda x: linkage,
    )
    fig.update_layout(
        title="Dendrogram of Imperfection Patterns (1 – Pearson r)",
        xaxis_title="Variable",
        yaxis_title="Distance",
        template="plotly_white",
    )
    if renderer:
        fig.show(renderer=renderer)
    if save_results and save_path:
        fig.write_image(save_path)
        print(f"Dendrogram saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    # Create a sample DataFrame with imperfect values
    data = {
        "id": [1, 2, 3, 4],
        "A": [1, 2, None, 4],
        "B": [None, 2, 3, 4],
        "C": [1, None, None, 4],
        "D": [1, 2, 3, None],
    }
    df = pl.DataFrame(data)

    from imperfekt.analysis.utils import masking

    mask_df = masking.create_missingness_mask(df)

    cm = corr_matrix(mask_df)
    print(cm)
    co_imperfection_heatmap(cm)
    co_imperfection_dendogram(cm)

    chi2 = chi2_intervariable_imperfection_statistics(mask_df, "A", "B")
    print(chi2)

    # Example usage
    df = pl.DataFrame(
        {
            "id": ["b", "b", "b", "b", "b", "b"],
            "clock_no": [1, 2, 3, 4, 5, 6],
            "var1": [1.0, None, 2.0, None, 4.5, None],
            "var2": [None, 4.0, None, 5.0, None, 2.0],
        }
    )
    mask_df = df.select(
        pl.col("id"),
        pl.col("clock_no"),
        (pl.col("var1").is_null().cast(pl.Int8)).alias("var1"),
        (pl.col("var2").is_null().cast(pl.Int8)).alias("var2"),
    )
    print(mask_df)

    ccfs = lagged_cross_correlation(mask_df, "var1", "var2")
    print(ccfs)
