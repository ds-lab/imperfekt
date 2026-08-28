import numpy as np
import polars as pl
import scipy.stats as stats

from imperfekt.analysis.utils import pretty_printing
from imperfekt.analysis.utils.kruskal_wallis import kruskal_wallis_effect_size_ci
from imperfekt.analysis.utils.statistics_utils import (
    cliffs_delta_ci,
    hodges_lehmann,
    hodges_lehmann_ci,
)

# Minimum defined values a metric needs in every group to be testable
MIN_DEFINED_PER_GROUP = 3

# Schema of the pairwise post-hoc frame, used to build a correctly-typed empty result
POSTHOC_SCHEMA = {
    "aspect": pl.Utf8,
    "variable": pl.Utf8,
    "metric": pl.Utf8,
    "group_1": pl.Utf8,
    "group_2": pl.Utf8,
    "p_value": pl.Float64,
    "cliffs_delta": pl.Float64,
    "ci_lower": pl.Float64,
    "ci_upper": pl.Float64,
    "direction": pl.Utf8,
}


def _facet_values(frames: dict[str, pl.DataFrame], facet_col: str | None) -> list:
    """Sorted union of facet values across all group frames; [None] when unfaceted."""
    if facet_col is None:
        return [None]
    values = set()
    for df in frames.values():
        if facet_col in df.columns:
            values.update(df[facet_col].drop_nulls().unique().to_list())
    return sorted(values)


def _slice(df: pl.DataFrame, facet_col: str | None, facet_value) -> pl.DataFrame:
    """Restrict a group frame to one facet value (a no-op when unfaceted)."""
    if facet_col is None or facet_value is None:
        return df
    return df.filter(pl.col(facet_col) == facet_value)


def _defined_values(df: pl.DataFrame, metric: str) -> np.ndarray:
    """Non-null, non-nan values of a metric as a float array."""
    if metric not in df.columns or df.height == 0:
        return np.array([], dtype=float)
    v = df[metric].cast(pl.Float64).drop_nulls().to_numpy()
    return v[~np.isnan(v)]


def describe_groups(
    frames: dict[str, pl.DataFrame],
    metric_cols: list[str],
    facet_col: str | None = None,
    aspect: str = "",
) -> pl.DataFrame:
    """
    Descriptives: one row per (aspect, facet_value, metric, group).

    Reports median [q25, q75], mean (std), n_defined / pct_defined, and the total n in each group.

    Parameters:
        frames (dict[str, pl.DataFrame]): Per-case metric frame keyed by group label.
        metric_cols (list[str]): Metric columns to describe.
        facet_col (str | None): Column to facet by (e.g. "variable"), or None.
        aspect (str): Aspect label recorded in the output. (e.g. "intravariable", "intervariable", "irregularity")

    Returns:
        pl.DataFrame: aspect, variable, metric, group, n, n_defined, pct_defined,
                      mean, std, median, q25, q75.
    """
    rows = []
    for facet_value in _facet_values(frames, facet_col):
        for metric in metric_cols:
            for group, df in frames.items():
                sub = _slice(df, facet_col, facet_value)
                values = _defined_values(sub, metric)
                n_total = sub.height
                n_defined = len(values)
                rows.append(
                    {
                        "aspect": aspect,
                        "variable": facet_value,
                        "metric": metric,
                        "group": str(group),
                        "n": n_total,
                        "n_defined": n_defined,
                        "pct_defined": (n_defined / n_total * 100) if n_total else float("nan"),
                        "mean": float(np.mean(values)) if n_defined else float("nan"),
                        "std": float(np.std(values, ddof=1)) if n_defined > 1 else float("nan"),
                        "median": float(np.median(values)) if n_defined else float("nan"),
                        "q25": float(np.percentile(values, 25)) if n_defined else float("nan"),
                        "q75": float(np.percentile(values, 75)) if n_defined else float("nan"),
                    }
                )

    if not rows:
        return pl.DataFrame(
            schema={
                "aspect": pl.Utf8,
                "variable": pl.Utf8,
                "metric": pl.Utf8,
                "group": pl.Utf8,
                "n": pl.Int64,
                "n_defined": pl.Int64,
                "pct_defined": pl.Float64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "median": pl.Float64,
                "q25": pl.Float64,
                "q75": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


def _definedness_p_value(
    frames: dict[str, pl.DataFrame], metric: str, facet_col: str | None, facet_value
) -> float:
    """
    Chi-square p-value for "is this metric computable equally often in every group?".

    A metric that is defined in 90% of one group and 60% of another yields an effect
    size computed on a self-selected subsample — an artefact of differential
    computability rather than a real difference in the metric. Returns nan when the
    contingency table is degenerate (all-defined or all-missing everywhere).
    """
    table = []
    for df in frames.values():
        sub = _slice(df, facet_col, facet_value)
        n_defined = len(_defined_values(sub, metric))
        table.append([n_defined, sub.height - n_defined])

    arr = np.array(table, dtype=float)
    if arr.sum() == 0 or (arr[:, 0].sum() == 0) or (arr[:, 1].sum() == 0):
        return float("nan")
    try:
        return float(stats.chi2_contingency(arr).pvalue)
    except ValueError:
        return float("nan")


def compare_groups(
    frames: dict[str, pl.DataFrame],
    metric_cols: list[str],
    alpha: float = 0.05,
    facet_col: str | None = None,
    aspect: str = "",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Omnibus test of every metric across groups.

    Two groups   -> Mann-Whitney U; effect size is Cliff's delta with a bootstrap CI,
                    plus the Hodges-Lehmann median difference and its distribution-free
                    CI, which express the direction in the metric's own units.
    k > 2 groups -> Kruskal-Wallis; effect size is eta-squared (H-based) with a
                    bootstrap CI. Pairwise detail comes from posthoc_pairwise(), which
                    the caller runs only for metrics surviving FDR correction.

    Metrics with fewer than MIN_DEFINED_PER_GROUP defined values in any group, or with
    no variance once pooled, are recorded with a skipped_reason rather than dropped —
    a silently absent row reads as "tested and null", which is not what happened.

    p-values are returned uncorrected; the caller applies Benjamini-Hochberg across
    the full family of tests.

    Parameters:
        frames (dict[str, pl.DataFrame]): Per-case metric frame keyed by group label.
        metric_cols (list[str]): Metric columns to test.
        alpha (float): Significance level (recorded; the caller decides on q).
        facet_col (str | None): Column to facet by (e.g. "variable"), or None.
        aspect (str): Aspect label recorded in the output.
        n_bootstrap (int): Bootstrap resamples for the Kruskal-Wallis eta-squared CI.
            Unused for two groups, where Cliff's delta gets an analytic interval.
        random_state (int): Seed for the bootstrap and for Hodges-Lehmann subsampling.

    Returns:
        pl.DataFrame: One row per (aspect, variable, metric) — see the module docstring
                      of imperfekt.analysis.imperfekt for the full column list.
    """
    group_labels = sorted(str(g) for g in frames)
    n_groups = len(group_labels)
    ordered = {g: frames[k] for g, k in zip(group_labels, sorted(frames, key=str))}

    rows = []
    for facet_value in _facet_values(
        frames, facet_col
    ):  # intravarialbe has "variable", other aspects have None
        for metric in metric_cols:
            row = {
                "aspect": aspect,
                "variable": facet_value,
                "metric": metric,
                "n_groups": n_groups,
                "test": None,
                "statistic": float("nan"),
                "p_value": float("nan"),
                "effect_size": float("nan"),
                "effect_size_name": None,
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "direction": None,
                "group_high": None,
                "group_low": None,
                "hodges_lehmann": float("nan"),
                "hl_ci_lower": float("nan"),
                "hl_ci_upper": float("nan"),
                "definedness_p_value": _definedness_p_value(
                    ordered, metric, facet_col, facet_value
                ),
                "skipped_reason": None,
            }

            samples = {
                g: _defined_values(_slice(df, facet_col, facet_value), metric)
                for g, df in ordered.items()
            }

            too_small = [g for g, v in samples.items() if len(v) < MIN_DEFINED_PER_GROUP]
            if too_small:
                row["skipped_reason"] = (
                    f"fewer than {MIN_DEFINED_PER_GROUP} defined values in group(s): "
                    + ", ".join(too_small)
                )
                rows.append(row)
                continue

            pooled = np.concatenate(list(samples.values()))
            if np.nanstd(pooled) == 0:
                row["skipped_reason"] = "zero variance across the pooled sample"
                rows.append(row)
                continue

            # Direction by median, which is meaningful for every group count.
            medians = {g: float(np.median(v)) for g, v in samples.items()}
            row["group_high"] = max(medians, key=lambda g: medians[g])
            row["group_low"] = min(medians, key=lambda g: medians[g])

            if n_groups == 2:
                g1, g2 = group_labels
                x, y = samples[g1], samples[g2]

                mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
                row["test"] = "mannwhitneyu"
                row["statistic"] = float(mwu.statistic)
                row["p_value"] = float(mwu.pvalue)

                es = cliffs_delta_ci(x, y)
                row["effect_size"] = float(es["effect_size"])
                row["effect_size_name"] = "cliffs_delta"
                row["ci_lower"] = float(es["ci_lower"])
                row["ci_upper"] = float(es["ci_upper"])

                row["hodges_lehmann"] = hodges_lehmann(x, y, random_state=random_state)
                hl_lo, hl_hi = hodges_lehmann_ci(x, y, alpha=alpha, random_state=random_state)
                row["hl_ci_lower"] = hl_lo
                row["hl_ci_upper"] = hl_hi

                # Cliff's delta is signed relative to the first (alphabetical) group.
                delta = row["effect_size"]
                if delta > 0:
                    row["direction"] = f"higher in {g1}"
                elif delta < 0:
                    row["direction"] = f"higher in {g2}"
                else:
                    row["direction"] = "no difference"
            else:
                groups_list = [samples[g] for g in group_labels]
                h_stat, p_value = stats.kruskal(*groups_list)
                row["test"] = "kruskal"
                row["statistic"] = float(h_stat)
                row["p_value"] = float(p_value)

                # eta-squared from H: (H - k + 1) / (n - k)
                n_total = sum(len(v) for v in groups_list)
                row["effect_size"] = float(h_stat - n_groups + 1) / (n_total - n_groups)
                row["effect_size_name"] = "eta_squared_h"
                try:
                    es = kruskal_wallis_effect_size_ci(
                        groups_list, n_bootstrap=n_bootstrap, random_state=random_state
                    )
                    row["ci_lower"] = float(es["ci_lower"])
                    row["ci_upper"] = float(es["ci_upper"])
                except ValueError:
                    pretty_printing.rich_warning(
                        f"Bootstrap CI unavailable for {aspect}/{facet_value}/{metric} "
                        "(a resample had no variance); reporting the point estimate only."
                    )
                row["direction"] = f"{row['group_high']} > {row['group_low']}"

            rows.append(row)

    return pl.DataFrame(rows, infer_schema_length=None)


def posthoc_pairwise(
    frames: dict[str, pl.DataFrame],
    metric: str,
    facet_col: str | None = None,
    facet_value=None,
    aspect: str = "",
    random_state: int = 42,
) -> pl.DataFrame:
    """
    All-pairs post-hoc comparison for one metric, for the k > 2 group case.

    Uses the Dwass-Steel-Critchlow-Fligner test, which controls the family-wise error
    rate across the pairwise comparisons directly rather than adjusting Dunn's
    rank-sum tests after the fact. Each pair also gets a Cliff's delta with a
    bootstrap CI, since DSCF returns only p-values.

    Run this only for metrics whose Kruskal-Wallis omnibus survived FDR correction —
    unconditional post-hoc testing inflates the error rate the omnibus was there to
    contain.

    Parameters:
        frames (dict[str, pl.DataFrame]): Per-case metric frame keyed by group label.
        metric (str): The metric to compare.
        facet_col (str | None): Column to facet by, or None.
        facet_value: The facet value to restrict to.
        aspect (str): Aspect label recorded in the output.
        random_state (int): Reserved for reproducibility of any sampling.

    Returns:
        pl.DataFrame: aspect, variable, metric, group_1, group_2, p_value,
                      cliffs_delta, ci_lower, ci_upper, direction.
    """
    import pandas as pd
    import scikit_posthocs as sp

    group_labels = sorted(str(g) for g in frames)
    ordered = {g: frames[k] for g, k in zip(group_labels, sorted(frames, key=str))}

    samples = {
        g: _defined_values(_slice(df, facet_col, facet_value), metric) for g, df in ordered.items()
    }
    usable = [g for g in group_labels if len(samples[g]) >= MIN_DEFINED_PER_GROUP]
    if len(usable) < 2:
        return pl.DataFrame(schema=POSTHOC_SCHEMA)

    long_df = pd.DataFrame(
        {
            "value": np.concatenate([samples[g] for g in usable]),
            "group": np.concatenate([[g] * len(samples[g]) for g in usable]),
        }
    )
    try:
        dscf = sp.posthoc_dscf(long_df, val_col="value", group_col="group")
    except Exception as exc:  # noqa: BLE001 — degenerate input should not abort the run
        pretty_printing.rich_warning(
            f"DSCF post-hoc failed for {aspect}/{facet_value}/{metric}: {exc}"
        )
        return pl.DataFrame(schema=POSTHOC_SCHEMA)

    rows = []
    for i, g1 in enumerate(usable):
        for g2 in usable[i + 1 :]:
            x, y = samples[g1], samples[g2]
            es = cliffs_delta_ci(x, y)
            delta = float(es["effect_size"])
            rows.append(
                {
                    "aspect": aspect,
                    "variable": facet_value,
                    "metric": metric,
                    "group_1": g1,
                    "group_2": g2,
                    "p_value": float(dscf.loc[g1, g2]),
                    "cliffs_delta": delta,
                    "ci_lower": float(es["ci_lower"]),
                    "ci_upper": float(es["ci_upper"]),
                    "direction": (
                        f"higher in {g1}"
                        if delta > 0
                        else f"higher in {g2}"
                        if delta < 0
                        else "no difference"
                    ),
                }
            )

    if not rows:
        return pl.DataFrame(schema=POSTHOC_SCHEMA)
    return pl.DataFrame(rows, infer_schema_length=None)
