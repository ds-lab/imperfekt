import numpy as np
import polars as pl
import scipy.stats as stats
from scipy.stats import levene, shapiro


def check_ttest_assumptions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    col1: str,
    col2: str = None,
    alpha: float = 0.05,
    print_info: bool = True,
) -> dict:
    """
    Check assumptions for independent two-sample t-test:
    1. Normality of each group (Shapiro-Wilk)
    2. Homogeneity of variances (Levene's test)

    Parameters:
        df1, df2 (pl.DataFrame): The two groups to compare.
        col1 (str): Column in df1.
        col2 (str): Column in df2. Defaults to col1.
        alpha (float): Significance level.
        print_info (bool): Whether to print results.

    Returns:
        dict: P-values and results of normality and variance homogeneity tests.
                Interpretation:
                - If p-value > alpha, the assumption is met (normality or equal variance).
                - If p-value <= alpha, the assumption is violated.
                - 'normality' key contains results for normality tests and if True the assumption is met.
                - 'equal_variance' key contains results for variance homogeneity and if 'equal_var' is True the assumption is met.

    """
    if col2 is None:
        col2 = col1

    x = df1[col1].drop_nulls().to_numpy()
    y = df2[col2].drop_nulls().to_numpy()

    results = {}

    # Normality tests
    sw_x_stat, sw_x_p = shapiro(x)
    sw_y_stat, sw_y_p = shapiro(y)
    normal_x = sw_x_p > alpha
    normal_y = sw_y_p > alpha

    results["normality"] = {
        "group1_p": sw_x_p,
        "group2_p": sw_y_p,
        "group1_normal": normal_x,
        "group2_normal": normal_y,
    }

    # Variance homogeneity (Levene's test)
    lev_stat, lev_p = levene(x, y)
    equal_var = lev_p > alpha

    results["equal_variance"] = {"levene_p": lev_p, "equal_var": equal_var}

    if print_info:
        print("Normality (Shapiro-Wilk):")
        print(f"  Group 1 p-value = {sw_x_p:.4f} -> {'OK' if normal_x else 'Violated'}")
        print(f"  Group 2 p-value = {sw_y_p:.4f} -> {'OK' if normal_y else 'Violated'}")
        print("Variance Homogeneity (Levene's test):")
        print(f"  Levene p-value = {lev_p:.4f} -> {'OK' if equal_var else 'Violated'}")

    return results


def t_test_two_subgroups(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    col1: str = None,
    col2: str = None,
    print_info: bool = True,
) -> tuple:
    if col1 is None:
        raise ValueError("col1 must be specified")
    if col2 is None:
        col2 = col1
    if col2 not in df2.columns:
        raise ValueError(f"col2 '{col2}' not found in DataFrame df2")

    df1_filtered = df1.filter(pl.col(col1).is_not_null())
    df2_filtered = df2.filter(pl.col(col2).is_not_null())
    if print_info:
        print(len(df1_filtered[col1].to_numpy()), len(df2_filtered[col2].to_numpy()))

    stdd1 = df1[col1].std()
    stdd2 = df2[col2].std()
    if print_info:
        print(f"Standard Deviation of {col1} result_means:", stdd1)
        print(f"Standard Deviation of {col2} vitals_means:", stdd2)

    if stdd1 == 0 or stdd2 == 0:
        raise ValueError("Standard deviation of one or both groups is zero, cannot perform t-test.")
    if stdd1 == stdd2:
        equal_var = True
    else:
        equal_var = False
    stat = stats.ttest_ind(
        df1_filtered[col1].to_numpy(),
        df2_filtered[col2].to_numpy(),
        equal_var=equal_var,
    )
    t_stat = stat.statistic
    p_val = stat.pvalue
    ci = stat.confidence_interval()
    ci_low, ci_high = ci.low, ci.high

    # Calculate effect size (Cohen's d)
    x1 = df1[col1].drop_nulls().to_numpy()
    x2 = df2[col2].drop_nulls().to_numpy()
    std1 = np.std(x1, ddof=1)
    std2 = np.std(x2, ddof=1)
    if equal_var:
        n1, n2 = len(x1), len(x2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = (np.mean(x1) - np.mean(x2)) / pooled_std
    else:
        avg_std = np.sqrt((std1**2 + std2**2) / 2)
        effect_size = (np.mean(x1) - np.mean(x2)) / avg_std

    if print_info:
        print(f"T-statistic: {t_stat}, P-value: {p_val}")
        print(f"Confidence Interval: [{ci_low}, {ci_high}]")
    return t_stat, p_val, stdd1, stdd2, effect_size, ci_low, ci_high


def mwu_two_subgroups(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    col1: str = None,
    col2: str = None,
    alpha: float = 0.05,
    print_info: bool = True,
    save_path: str = None,
    save_results: bool = True,
) -> tuple:
    if col1 is None:
        raise ValueError("col1 must be specified")
    if col2 is None:
        col2 = col1
    if col2 not in df2.columns:
        raise ValueError(f"col2 '{col2}' not found in DataFrame df2")

    df1_filtered = df1.filter(pl.col(col1).is_not_null())
    df2_filtered = df2.filter(pl.col(col2).is_not_null())
    if print_info:
        print(len(df1_filtered[col1].to_numpy()), len(df2_filtered[col2].to_numpy()))

    stat = stats.mannwhitneyu(
        df1_filtered[col1].to_numpy(),
        df2_filtered[col2].to_numpy(),
        alternative="two-sided",
    )
    u_stat = stat.statistic
    p_val = stat.pvalue

    # Calculate effect size (r)
    r = mwu_effect_size_ci(
        df1_filtered[col1].to_numpy(),
        df2_filtered[col2].to_numpy(),
        confidence_level=0.95,
        n_bootstrap=1000,
    )

    if print_info:
        print(f"Mann-Whitney U test results for {col1}/{col2}:")
        print(f"U-statistic: {u_stat}, P-value: {p_val}")
        print(
            f"Rank-biserial correlation: {r['effect_size']}, CI {r['confidence_level']}: [{r['ci_lower']}, {r['ci_upper']}]"
        )

    results = {
        "u_stat": u_stat,
        "p_val": p_val,
        "significance": p_val < alpha,
        "effect_size": r["effect_size"],
        "confidence_level": r["confidence_level"],
        "ci_lower": r["ci_lower"],
        "ci_upper": r["ci_upper"],
        "size_group_1": len(df1_filtered[col1].to_numpy()),
        "size_group_2": len(df2_filtered[col2].to_numpy()),
        "mean_group_1": df1_filtered[col1].mean(),
        "mean_group_2": df2_filtered[col2].mean(),
        "stdd_group_1": df1_filtered[col1].std(),
        "stdd_group_2": df2_filtered[col2].std(),
    }

    if save_path and save_results:
        with open(save_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    return results


def rank_biserial_correlation(x, y):
    """Calculate rank-biserial correlation from Mann-Whitney U test"""
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    n1, n2 = len(x), len(y)
    # RBC = (2*U1 - n1*n2) / (n1*n2)
    rbc = (2 * u_stat - n1 * n2) / (n1 * n2)
    return rbc


def mwu_effect_size_ci(
    x1: np.ndarray,
    x2: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap the rank-biserial correlation between two samples.

    Parameters:
        x1, x2: Arrays of data for the two groups.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level for the interval.
        random_state: For reproducibility.

    Returns:
        dict with point estimate, lower and upper CI.
    """
    rank_biserial_observed = rank_biserial_correlation(x1, x2)

    rng = np.random.default_rng(random_state)
    estimates = []

    n1, n2 = len(x1), len(x2)
    for _ in range(n_bootstrap):
        sample1 = rng.choice(x1, size=n1, replace=True)
        sample2 = rng.choice(x2, size=n2, replace=True)
        r_rb = rank_biserial_correlation(sample1, sample2)
        estimates.append(r_rb)

    lower = np.percentile(estimates, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(estimates, (1 + confidence_level) / 2 * 100)
    # point_estimate = np.median(estimates)

    return {
        "effect_size": rank_biserial_observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
    }
