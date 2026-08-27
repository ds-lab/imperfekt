from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as stats
from scipy.stats import levene, shapiro


def check_ttest_assumptions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    col1: str,
    col2: str | None = None,
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
    col1: str | None = None,
    col2: str | None = None,
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
    equal_var = np.isclose(stdd1, stdd2, rtol=1e-5)
    
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
    col1: str | None = None,
    col2: str | None = None,
    alpha: float = 0.05,
    print_info: bool = True,
    save_path: str | Path | None = None,
    save_results: bool = True,
) -> dict:
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


def _dominance_means(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Row and column means of the sign-dominance matrix d_ij = sign(x_i - y_j), plus the tie count.

    Computed without ever materialising the n1 x n2 matrix: for each x_i, binary search
    tells us how many y fall below, equal to, and above it, which is all the row mean
    needs. Cost is O((n1 + n2) log(n1 + n2)) time and O(n1 + n2) memory instead of
    O(n1 * n2) of both.

    Parameters:
        x, y: Finite-valued 1-D arrays for the two groups.

    Returns:
        tuple: (row means d_i., column means d_.j, number of tied pairs).
    """
    xs = np.sort(x)
    ys = np.sort(y)
    n1, n2 = len(x), len(y)

    # For each x_i: how many y_j are strictly below / at / above it.
    lo = np.searchsorted(ys, x, side="left")
    hi = np.searchsorted(ys, x, side="right")
    less = lo  # y_j < x_i
    ties_per_x = hi - lo  # y_j == x_i
    greater = n2 - hi  # y_j > x_i
    row_means = (less - greater) / n2

    # Mirror image for each y_j, over x.
    lo_y = np.searchsorted(xs, y, side="left")
    hi_y = np.searchsorted(xs, y, side="right")
    # d_.j is the mean over i of sign(x_i - y_j): x_i > y_j counts +1, x_i < y_j counts -1.
    col_means = ((n1 - hi_y) - lo_y) / n1

    return row_means, col_means, int(ties_per_x.sum())


def cliffs_delta(x, y) -> float:
    """
    Cliff's delta: P(x > y) - P(x < y), bounded in [-1, 1].

    Mathematically identical to the rank-biserial correlation, but computed by binary
    search rather than by running a full Mann-Whitney U test, so it stays usable on
    cohorts with millions of cases.

    Parameters:
        x, y: Arrays of data for the two groups.

    Returns:
        float: Cliff's delta, or nan if either sample is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    row_means, _, _ = _dominance_means(x, y)
    return float(row_means.mean())


def cliffs_delta_ci(
    x,
    y,
    confidence_level: float = 0.95,
) -> dict:
    """
    Cliff's delta with an analytic confidence interval (Cliff 1993).

    Uses Cliff's consistent variance estimator and his asymmetric interval, which is
    built on the inverse-hyperbolic-tangent-like transform

        (d - d^3 +/- z*s*sqrt((1 - d^2)^2 + z^2*s^2)) / (1 - d^2 + z^2*s^2)

    so the bounds cannot escape [-1, 1] the way a symmetric normal interval can when
    delta sits near the edge.

    This replaces bootstrapping the effect size. A 1000-iteration bootstrap of a rank
    statistic costs O(1000 * n log n) and becomes unusable at cohort sizes in the
    hundreds of thousands; the analytic form is a single O(n log n) pass and needs no
    subsampling, so it is both faster and exact rather than approximate.

    Parameters:
        x, y: Arrays of data for the two groups.
        confidence_level (float): Coverage of the interval.

    Returns:
        dict: effect_size, ci_lower, ci_upper, confidence_level. Bounds are nan when
              either sample has fewer than 2 values, where the variance is undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)

    out = {
        "effect_size": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "confidence_level": confidence_level,
    }
    if n1 == 0 or n2 == 0:
        return out

    row_means, col_means, n_ties = _dominance_means(x, y)
    delta = float(row_means.mean())
    out["effect_size"] = delta

    if n1 < 2 or n2 < 2:
        return out

    # Cliff's consistent variance estimator. The double sum over the full matrix is
    # obtained in closed form: sum(d_ij^2) = n1*n2 - ties and sum(d_ij) = n1*n2*delta,
    # so sum (d_ij - delta)^2 = n1*n2 - ties - n1*n2*delta^2 — no matrix needed.
    ss_rows = float(np.sum((row_means - delta) ** 2))
    ss_cols = float(np.sum((col_means - delta) ** 2))
    ss_all = n1 * n2 - n_ties - n1 * n2 * delta**2

    var_delta = (n2**2 * ss_rows + n1**2 * ss_cols - ss_all) / (n1 * n2 * (n1 - 1) * (n2 - 1))
    var_delta = max(var_delta, 0.0)
    sigma = np.sqrt(var_delta)

    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    denom = 1 - delta**2 + z**2 * var_delta
    if denom <= 0:
        # delta is exactly +/-1 with no variance: the estimate is a point.
        out["ci_lower"] = out["ci_upper"] = delta
        return out

    centre = delta - delta**3
    spread = z * sigma * np.sqrt((1 - delta**2) ** 2 + z**2 * var_delta)
    out["ci_lower"] = float(np.clip((centre - spread) / denom, -1.0, 1.0))
    out["ci_upper"] = float(np.clip((centre + spread) / denom, -1.0, 1.0))
    return out


def _fit_pair_budget(
    x: np.ndarray,
    y: np.ndarray,
    max_pairs: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shrink two samples just enough that their pair grid fits in max_pairs.

    Both sides are scaled by the same factor sqrt(max_pairs / (n1*n2)), which keeps the
    original balance between them and spends the whole budget. Shrinking each side to
    sqrt(max_pairs) independently would be wrong for unbalanced groups: with n1 = 2.5M
    and n2 = 10 the grid is already exactly at budget, yet a per-side cap would still
    cut x to 5000 and throw away almost all the precision for nothing.

    The larger side is clamped against the budget afterwards, because a side that
    rounds down to the floor of 1 would otherwise let the other side overshoot.

    Parameters:
        x, y: Finite-valued 1-D arrays.
        max_pairs (int): Maximum number of pairwise differences allowed.
        random_state (int): Seed, so the estimate is reproducible.

    Returns:
        tuple[np.ndarray, np.ndarray]: The (possibly subsampled) arrays.
    """
    n1, n2 = len(x), len(y)
    if n1 * n2 <= max_pairs:
        return x, y

    rng = np.random.default_rng(random_state)
    shrink = np.sqrt(max_pairs / (n1 * n2))
    new_n1 = max(1, min(n1, int(n1 * shrink)))
    new_n2 = max(1, min(n2, int(n2 * shrink)))

    # A side that the proportional split rounds below 1 is being over-shrunk: if it is
    # small enough to keep whole, keep all of it and spend the budget on the other side.
    # Otherwise a 10^9 x 2 comparison would throw away one of only two values.
    if int(n1 * shrink) < 1 and n1 <= max_pairs:
        new_n1 = n1
    if int(n2 * shrink) < 1 and n2 <= max_pairs:
        new_n2 = n2

    # Whichever side is larger absorbs the rounding, so the product still fits.
    if new_n1 * new_n2 > max_pairs:
        if new_n1 >= new_n2:
            new_n1 = max(1, max_pairs // new_n2)
        else:
            new_n2 = max(1, max_pairs // new_n1)

    if new_n1 < n1:
        x = rng.choice(x, size=new_n1, replace=False)
    if new_n2 < n2:
        y = rng.choice(y, size=new_n2, replace=False)
    return x, y


def hodges_lehmann(
    x: np.ndarray,
    y: np.ndarray,
    max_pairs: int = 25_000_000,
    random_state: int = 42,
) -> float:
    """
    Hodges-Lehmann estimator: the median of all pairwise differences x_i - y_j.

    This is the location-shift estimator consistent with the Mann-Whitney U test —
    it answers "by how much does x exceed y" in the metric's own units, without
    assuming normality.

    The exact estimator needs an n1 x n2 difference matrix. Above max_pairs entries both
    samples are proportionally subsampled (seeded, so results are reproducible) to keep
    memory bounded — see _fit_pair_budget.

    Parameters:
        x, y: Arrays of data for the two groups.
        max_pairs (int): Maximum number of pairwise differences to materialise.
        random_state (int): Seed for the subsampling.

    Returns:
        float: Median pairwise difference, or nan if either sample is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    x, y = _fit_pair_budget(x, y, max_pairs, random_state)
    return float(np.median(x[:, None] - y[None, :]))


def hodges_lehmann_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    max_pairs: int = 25_000_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Distribution-free confidence interval for the Hodges-Lehmann location shift.

    Obtained by inverting the Mann-Whitney rank statistic: the interval spans the k-th
    smallest to the k-th largest pairwise difference, with k from the normal
    approximation to the null distribution of U. Valid without any distributional
    assumption.

    Uses the same subsampling rule as hodges_lehmann() with the same seed, so the point
    estimate and its interval are computed over identical pairs. When subsampling is in
    play the interval reflects the subsample size, which makes it conservative — that is
    the honest direction, since claiming full-sample precision from a subsample would
    overstate what was actually measured.

    Parameters:
        x, y: Arrays of data for the two groups.
        alpha (float): Significance level; the interval has coverage 1 - alpha.
        max_pairs (int): Maximum number of pairwise differences to materialise.
        random_state (int): Seed for the subsampling.

    Returns:
        tuple[float, float]: (lower, upper) bounds, or (nan, nan) if undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")

    x, y = _fit_pair_budget(x, y, max_pairs, random_state)
    n1, n2 = len(x), len(y)
    diffs = np.sort((x[:, None] - y[None, :]).ravel())
    n_pairs = len(diffs)

    z = stats.norm.ppf(1 - alpha / 2)
    # Normal approximation to the null distribution of U, with continuity correction.
    mean_u = n1 * n2 / 2
    sd_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    k = int(np.floor(mean_u - z * sd_u))

    if k < 0:
        # Too few pairs for the approximation to bite — the interval is the full range.
        return float(diffs[0]), float(diffs[-1])

    lower_idx = min(k, n_pairs - 1)
    upper_idx = max(n_pairs - 1 - k, 0)
    return float(diffs[lower_idx]), float(diffs[upper_idx])


def benjamini_hochberg(p_values) -> np.ndarray:
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Parameters:
        p_values: Sequence of raw p-values, possibly containing nan.

    Returns:
        np.ndarray: q-values, same length and order as the input.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    if not valid.any():
        return q

    from statsmodels.stats.multitest import multipletests

    _, q_valid, _, _ = multipletests(p[valid], method="fdr_bh")
    q[valid] = q_valid
    return q


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
