from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import kruskal

from imperfekt.analysis.utils.statistics_utils import (
    mwu_effect_size_ci,
)


def perform_statistical_analysis(
    analysis_pd: pd.DataFrame,
    c: str,
    group_col: str,
    posthoc_method: str,
    renderer: str,
    save_path: str,
    save_results: bool,
    analyzed_col: str = None,
):
    """Helper function to perform statistical analysis for a given column."""
    # Prepare data for the test: a list of arrays, one for each diagnosis group
    groups = [
        group_df[c].dropna().values
        for name, group_df in analysis_pd.groupby(group_col, observed=True)
    ]

    # 4. Perform Kruskal-Wallis test and if significant Dunn's post-hoc test
    if len(groups) > 1:
        stat, p_value = kruskal(*groups)
        kw_result = kruskal_wallis_effect_size_ci(groups)
        kw_result["statistic"] = stat
        kw_result["p_value"] = p_value

        if renderer:
            print(f"\nKruskal-Wallis Test for {c} by {group_col}:")
            print(f"Statistic: {stat:.4f}, p-value: {p_value:.4g}")
            print(f"Effect Size (Eta-squared): {kw_result['effect_size']:.4f}")
            print(f"95% CI: [{kw_result['ci_lower']:.4f}, {kw_result['ci_upper']:.4f}]")

        if save_path and save_results:
            with open(save_path / f"kw_results_{c}.txt", "w") as f:
                f.write(f"H-Statistic: {stat:.4f}\n")
                f.write(f"p-value: {p_value:.4g}\n")
                f.write(f"Eta-squared = {kw_result['effect_size']:.4f}\n")
                f.write(f"95% CI: [{kw_result['ci_lower']:.4f}, {kw_result['ci_upper']:.4f}]\n")
                f.write(
                    "For more details on the analyzed groups, see <col>_gap_return_summary.csv\n"
                )

        # If the result is significant, perform the post-hoc tests
        if p_value < 0.05:
            pval_heatmap_fig, es_heatmap_fig = _perform_posthoc_tests(
                analysis_pd,
                c,
                group_col,
                posthoc_method,
                renderer,
                save_path,
                save_results,
            )
        else:
            pval_heatmap_fig, es_heatmap_fig = None, None

        return kw_result, pval_heatmap_fig, es_heatmap_fig

    else:
        print(f" {analyzed_col}: Not enough groups to perform the test.")
        return None, None, None


def kruskal_wallis_effect_size_ci(
    groups: list,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 42,
):
    """
    Calculate Kruskal-Wallis effect size (eta-squared) with confidence interval

    Parameters:
        groups (list): List of arrays/lists, each representing a group of data.
        confidence_level (float): Confidence level for the bootstrap CI.
        n_bootstrap (int): Number of bootstrap resamples to compute the CI.

    Returns:
        dict: Contains the observed effect size, lower and upper bounds of the confidence interval, and the confidence level.
    """

    def eta_squared_kw(*groups):
        h_stat, _ = stats.kruskal(*groups)
        n_total = sum(len(group) for group in groups)
        k_groups = len(groups)
        eta_sq = (h_stat - k_groups + 1) / (n_total - k_groups)
        return eta_sq

    # Calculate observed effect size
    eta_observed = eta_squared_kw(*groups)

    # Bootstrap to get confidence interval
    rng = np.random.default_rng(random_state)
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        resampled_groups = [rng.choice(group, size=len(group), replace=True) for group in groups]
        eta_bootstrap = eta_squared_kw(*resampled_groups)
        bootstrap_estimates.append(eta_bootstrap)

    bootstrap_estimates = np.array(bootstrap_estimates)
    lower = np.percentile(bootstrap_estimates, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(bootstrap_estimates, (1 + confidence_level) / 2 * 100)

    return {
        "effect_size": eta_observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
    }


def _perform_posthoc_tests(
    analysis_pd: pd.DataFrame,
    c: str,
    group_col: str,
    posthoc_method: str,
    renderer: str,
    save_path: str,
    save_results: bool,
):
    """Helper function to perform post-hoc tests."""
    analysis_pd = analysis_pd[analysis_pd[c].notna()]  # Ensure no NaN values in the column
    if posthoc_method == "dunn":
        posthoc_results = sp.posthoc_dunn(
            analysis_pd,
            val_col=c,
            group_col=group_col,
            p_adjust="holm",  # Adjust p-values using Holm's method
        )

    elif posthoc_method == "dscf":
        # Pairwise comparisons were performed using the Dwass-Steel-Critchlow-Fligner (DSCF) test does not assume equal variances and is robust to unequal group sizes.
        posthoc_results = sp.posthoc_dscf(
            analysis_pd,
            val_col=c,
            group_col=group_col,
        )
        # Calculate effect sizes for DSCF
        effect_size_results, effect_size_matrix = calculate_dcsf_effect_sizes(
            df=analysis_pd,
            val_col=c,
            group_col=group_col,
        )

        # Visualize effect sizes with a heatmap
        es_heatmap_fig = posthoc_heatmap(
            effect_size_matrix,
            col=c,
            type="posthoc_effect_sizes",
            renderer=renderer,
            save_path=save_path,
            save_results=save_results,
        )
        # Save effect size results if required
        if save_results and save_path:
            pd.DataFrame(effect_size_results).to_csv(
                save_path / f"posthoc_effect_sizes_rbc_{c}.csv"
            )
            pd.DataFrame(posthoc_results).to_csv(save_path / f"posthoc_dscf_p_{c}.csv")

    else:
        raise ValueError(f"Unsupported post-hoc method: {posthoc_method}. Use 'dunn' or 'dscf'.")

    # A value < 0.05 in the table means the two groups are significantly different
    if renderer:
        print(f"\n{posthoc_method.upper()} Test Results (p-values):")
        print(posthoc_results.round(5))
    # Visualize the posthoc p-value results with a heatmap
    pval_heatmap_fig = posthoc_heatmap(
        posthoc_results,
        col=c,
        type="posthoc_p_values",
        renderer=renderer,
        save_path=save_path,
        save_results=save_results,
    )

    return pval_heatmap_fig, es_heatmap_fig


def calculate_dcsf_effect_sizes(
    df: pd.DataFrame, val_col: str, group_col: str
) -> tuple[list, pd.DataFrame]:
    """
    Calculate the effect sizes for all pairwise group comparisons.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        val_col (str): Column name for the values to compare.
        group_col (str): Column name for the grouping variable.

    Returns:
        tuple: A list of dictionaries with effect size results and a DataFrame with the effect sizes
        for each pair of groups.
    """
    groups = df[group_col].unique()
    effect_sizes = pd.DataFrame(index=groups, columns=groups, dtype=np.float64)
    results = []
    for group1, group2 in combinations(groups, 2):
        data1 = df[df[group_col] == group1][val_col]
        data2 = df[df[group_col] == group2][val_col]

        mwu_result = mwu_effect_size_ci(data1, data2)
        results.append(
            {
                "group1": group1,
                "group2": group2,
                "effect_size": mwu_result["effect_size"],
                "ci_lower": mwu_result["ci_lower"],
                "ci_upper": mwu_result["ci_upper"],
            }
        )
        effect_sizes.loc[group1, group2] = mwu_result["effect_size"]
        effect_sizes.loc[group2, group1] = mwu_result["effect_size"]
        effect_sizes.fillna(0, inplace=True)  # Fill diagonal with 0s

    return results, effect_sizes


def posthoc_heatmap(
    posthoc_results: pl.DataFrame,
    col: str = None,
    type: str = "dunn_p_values",
    renderer: str = None,
    save_path: str = None,
    save_results: bool = False,
) -> None:
    """Create a heatmap for post-hoc test results.
    Parameters:
        posthoc_results (pl.DataFrame): DataFrame containing the results of Dunn's post-hoc test.
    Returns:
        None: Displays the heatmap using Plotly.
    """
    if type == "posthoc_p_values":
        title_text = f"<b>Post-Hoc Test Pairwise p-values for {col.title()}</b>"
        colorbar_title = "p-value"
    elif type == "posthoc_effect_sizes":
        title_text = f"<b>Post-Hoc Test Pairwise Effect Sizes for {col.title()}</b>"
        colorbar_title = "r"
    else:
        raise ValueError("Type must be either 'posthoc_p_values' or 'posthoc_effect_sizes'.")

    if col is None:
        raise ValueError("Column name for the heatmap must be provided.")

    fig = go.Figure(
        data=go.Heatmap(
            z=posthoc_results.values,
            x=posthoc_results.columns,
            y=posthoc_results.index,
            colorscale="Blues_r",  # Low p-values are dark blue
            colorbar=dict(title=colorbar_title),
            # Create a matrix of string-formatted p-values for annotations
            text=posthoc_results.map("{:.4g}".format).astype(str).values,
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        width=800,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_tickangle=-45,  # Rotate labels to prevent overlap
    )

    if renderer:
        fig.show(renderer=renderer)

    if save_results and save_path:
        fig.write_html(save_path / f"{type}_heatmap_{col}.html")
        print(f"Saved {type} heatmap for {col} to {save_path / f'{type}_heatmap_{col}.html'}")

    return fig
