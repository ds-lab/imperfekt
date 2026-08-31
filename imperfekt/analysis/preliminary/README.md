# Preliminary Analysis Module

This module provides a comprehensive suite of preliminary statistical analyses for exploratory data analysis (EDA). It is designed for time-series or longitudinal data with unique identifiers and temporal ordering.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Class Structure](#class-structure)
4. [Analysis Methods](#analysis-methods)
   - [Descriptive Statistics](#1-descriptive-statistics-describe_df)
   - [Shapiro-Wilk Normality Test](#2-shapiro-wilk-normality-test-shapiro_wilk)
   - [Multivariate Normality Test](#3-multivariate-normality-test-multivariate_normality)
   - [Correlation Analysis](#4-correlation-analysis-correlation)
   - [Autocorrelation Analysis](#5-autocorrelation-analysis-autocorrelation)
   - [Case-Level Observed-Value Metrics](#6-case-level-observed-value-metrics-case_metrics)
5. [Usage Example](#usage-example)
6. [References](#references)

---

## Overview

The `Preliminary` class performs foundational statistical analyses to:
- Characterize data distributions
- Assess normality assumptions (univariate and multivariate)
- Quantify relationships between variables
- Detect temporal dependencies within variables



---

## Class Structure

```
Preliminary
├── Parameters (constructor)
│   ├── df: pl.DataFrame              # Input dataframe
│   ├── id_col: str                   # Unique identifier column (default: "id")
│   ├── clock_col: str                # Temporal ordering column (default: "clock")
│   ├── clock_no_col: str             # Clock number column (default: "clock_no")
│   ├── cols: list                    # Columns to analyze (default: all except id/clock)
│   ├── alpha: float                  # Significance level (default: 0.05)
│   ├── save_path: Path               # Output directory (optional)
│   ├── plot_library: str             # "matplotlib" or "plotly"
│   └── renderer: str                 # Plotly renderer (e.g., "notebook_connected")
│
├── Methods
│   ├── describe_df()                 # Descriptive statistics + violin plots
│   ├── shapiro_wilk()                # Univariate normality test + Q-Q plots
│   ├── multivariate_normality()     # Multivariate normality (Henze-Zirkler)
│   ├── correlation(use=)             # Spearman correlation matrix + heatmap
│   ├── autocorrelation(lags=)        # Lagged autocorrelation + scatter plots
│   ├── case_metrics()                # Per-case summary measures of observed values
│   ├── run()                         # Execute all analyses
│   └── generate_html_report()        # Create HTML summary report
│
└── results: PreliminaryResults
    ├── descriptive_stats: pl.DataFrame
    ├── cm_case_metrics: pl.DataFrame | None
    ├── shapiro_wilk: pl.DataFrame
    ├── multivariate_normality: pl.DataFrame
    ├── correlation: pl.DataFrame
    ├── autocorrelation: dict[str, pl.DataFrame]
    └── plots: PreliminaryPlots
        ├── violin: dict[str, Figure]
        ├── qq_plot: dict[str, Figure]
        ├── correlation_heatmap: Figure
        └── autocorrelation_lag_plot: dict[str, Figure]
```

### Method Chaining

All analysis methods return `self`, enabling fluent chaining:

```python
preliminary = (
    Preliminary(df, id_col="patient", clock_col="time")
    .describe_df()
    .shapiro_wilk()
    .correlation(use="pairwise")
    .autocorrelation(lags=10)
)
```

---

## Analysis Methods

### 1. Descriptive Statistics (`describe_df`)

Computes summary statistics for all columns in the dataset using Polars' built-in `DataFrame.describe()` method.

**Library**: [Polars](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.describe.html)

#### Metrics Computed
Count, Mean, Standard Deviation, Min, Max, Percentiles

#### Visualization

**Violin Plot**: Combines a box plot with a kernel density estimate (KDE) to show the probability density of the data at different values. Generated via Plotly or Matplotlib.

---

### 2. Shapiro-Wilk Normality Test (`shapiro_wilk`)

Tests whether a univariate sample comes from a normally distributed population.

**Library**: [`scipy.stats.shapiro`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html) [[1]](#references)

#### Hypotheses

$$
\begin{aligned}
H_0 &: \text{The data follows a normal distribution} \\
H_1 &: \text{The data does not follow a normal distribution}
\end{aligned}
$$

#### Test Statistic

The Shapiro-Wilk test statistic $W$ is defined as [[1]](#references):

$$
W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}
$$

Where:
- $x_{(i)}$ are the ordered sample values (order statistics)
- $a_i = (a_1, \ldots, a_n) = \frac{m^\top V^{-1}}{(m^\top V^{-1} V^{-1} m)^{1/2}}$ where $m$ and $V$ are the expected values and covariance matrix of standard normal order statistics [[1]](#references)
- $\bar{x}$ is the sample mean

#### Interpretation

| Condition | Interpretation |
|-----------|----------------|
| $p > \alpha$ | Fail to reject $H_0$; data is consistent with normality |
| $p \leq \alpha$ | Reject $H_0$; evidence against normality |

#### Visualization

**Q-Q Plot (Quantile-Quantile Plot)**: Plots the quantiles of the observed data against the theoretical quantiles of a normal distribution. Points falling along the diagonal line suggest normality. Generated via Matplotlib.

---

### 3. Multivariate Normality Test (`multivariate_normality`)

Assesses whether a multivariate dataset follows a multivariate normal distribution using the **Henze-Zirkler test**[[2]](#references).

**Library**: [`pingouin.multivariate_normality`](https://pingouin-stats.org/build/html/generated/pingouin.multivariate_normality.html) [[5]](#references)

#### Hypotheses

$$
\begin{aligned}
H_0 &: \text{The variables jointly follow a multivariate normal distribution — i.e.,  any linear combination of the variables is normally distributed.} \\
H_1 &: \text{The variables do not follow a multivariate normal distribution.}
\end{aligned}
$$

Where $\mathcal{N}_p$ denotes a $p$-dimensional multivariate normal distribution.

#### Interpretation

| Condition | Interpretation |
|-----------|----------------|
| $p > \alpha$ | Fail to reject $H_0$; multivariate normality is plausible |
| $p \leq \alpha$ | Reject $H_0$; evidence against multivariate normality |

---

### 4. Correlation Analysis (`correlation`)

Computes the **Spearman rank correlation coefficient** [[3]](#references) ($\rho_s \in [-1,1]$ ) between all pairs of numeric variables.

**Library**: [`polars.corr`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.corr.html) with `method="spearman"`



#### Missing Data Handling

| Method | Description |
|--------|-------------|
| **Listwise** | Removes any row containing at least one null value across selected columns. Ensures consistent sample across all correlations. |
| **Pairwise** | Computes correlation for each pair using all available observations for that pair. Maximizes data usage but may result in different sample sizes per pair. |

#### Interpretation
| $\rho_s$ Value | Interpretation |
|----------------|----------------|
| $\rho_s = 1$ | Perfect positive monotonic relationship |
| $\rho_s = 0$ | No monotonic relationship |
| $\rho_s = -1$ | Perfect negative relationship (symmetric interpretation) |



#### Visualization

**Correlation Heatmap**: A color-coded matrix where each cell represents the correlation coefficient between two variables. The color scale ranges from -1 (negative correlation, blue) to +1 (positive correlation, red). Generated via Plotly.

---

### 5. Autocorrelation Analysis (`autocorrelation`)

Measures the correlation of a variable with its own lagged values over time, useful for detecting temporal dependencies.

**Library**: Custom implementation in `imperfekt.analysis.intravariable.autocorrelation`

**Source**: [Wikipedia: Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) [[4]](#references)

#### Autocorrelation Estimator

The implementation follows the sample autocorrelation estimator from Wikipedia [[4]](#references):

$$
\hat{R}(k) = \frac{1}{n \cdot \hat{\sigma}^2} \sum_{t=1}^{n-k}(x_t - \bar{x})(x_{t+k} - \bar{x})
$$

Where:
- $k$ is the lag
- $\bar{x}$ is the sample mean
- $\hat{\sigma}^2$ is the sample variance

#### Panel Data Adaptation

For grouped/longitudinal data, the implementation:
- Computes lags **within each group** (via `pl.shift().over(id_col)`)
- Excludes invalid (null) lag pairs from the summation
- Normalizes by the number of valid pairs at each lag

#### Interpretation

| Pattern | Interpretation |
|---------|----------------|
| Rapid decay to zero | Short-term memory |
| Slow decay | Long-term dependencies |
| Alternating signs | Oscillatory behavior |
| Spikes at specific lags | Seasonal/periodic patterns |

---

### 6. Case-Level Observed-Value Metrics (`case_metrics`)

Reduces each case's series to **one row per (case, variable)** summarizing the *observed values*.

Unlike the metrics of the intravariable, intervariable and irregularity modules, these are
plain **descriptive summary measures of the data itself**, not constructed imperfection
indices. They share the `case_metrics` name and frame shape so the observed values can enter
the same [group comparison](../README.md#group-comparison) as the imperfection metrics.

#### Why per case, and not per timestamp

Observations within a case are correlated, so pooling raw timestamps across groups and testing
them treats each observation as independent. That inflates $n$ by one to two orders of magnitude,
makes p-values meaningless, and lets cases with the most observations dominate the result.

The standard remedy is the **summary-measures** (or *two-stage*) approach [[6]](#references):
reduce each case's series to a single number per variable, then apply an ordinary test across
cases. The metrics below are the summary measures; the group comparison is the second stage.

#### Metrics

For a case with observed values $v_1, \dots, v_n$ at times $t_1 \le \dots \le t_n$
(nulls excluded), where $n$ is the number of *observed* values of that variable:

| Metric | Formula | Units | Defined when |
|--------|---------|-------|--------------|
| `value_mean` | $\bar{v} = \frac{1}{n}\sum_i v_i$ | variable's own | $n \ge 1$ |
| `value_min` | $\min_i v_i$ | variable's own | $n \ge 1$ |
| `value_max` | $\max_i v_i$ | variable's own | $n \ge 1$ |
| `value_iqr` | $Q_{75}(v) - Q_{25}(v)$ | variable's own | $n \ge$ `min_obs_spread` |
| `value_slope` | $\hat{\beta} = \dfrac{\mathrm{Cov}(t, v)}{\mathrm{Var}(t)}$ | variable's own **per hour** | $n \ge$ `min_obs_slope` and $t_n > t_1$ |
| `value_first` | $v_1$ | variable's own | $n \ge 1$ |


The time axis is seconds from `clock_col`, converted to the unit given by `slope_time_unit`. When
`clock_col` is not temporal the `clock_no_col` index is used instead and the rate is per
observation, with `slope_time_unit` ignored.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_obs_spread` | 2 | Minimum observations for `value_iqr` |
| `min_obs_slope` | 2 | Minimum observations for `value_slope` |
| `slope_time_unit` | `"hour"` | Unit of `value_slope`: `"second"`, `"minute"`, `"hour"` or `"day"` |
| `save_results` | `True` | Write `case_metrics/case_metrics.csv` to `save_path` |


#### Output

**`cm_case_metrics`** — one row per (case, variable), with the `id` column, `variable`, and the six
metric columns above. A case with **no** observed values of a variable yields an all-null row
rather than no row, so that differential availability stays visible rather than silently
shrinking the sample.

#### Interpretation caveats

| Caveat | Consequence |
|--------|-------------|
| Computed on observed values only | Under MNAR, a case's mean is a biased estimate of its true mean, and the bias may differ by group. This is why the observed-value and imperfection comparisons belong side by side. |
| Unweighted over irregular samples | `value_mean` over-weights densely sampled stretches. Cross-read against the [irregularity](../irregularity/README.md) aspect. |
| `value_slope` is noisy on short records | The rate is unbiased at any record length, but its variance grows as the observed span shrinks. Raise `min_obs_slope` if that matters. |
| Availability is itself a finding | In the group comparison, `pct_defined` reads as "% of cases in which this variable was ever recorded", and `definedness_q_value` tests whether that differs by group. |

---

## Usage Example

```python
import polars as pl
from pathlib import Path
from imperfekt.analysis.preliminary import Preliminary

# Load your data
df = pl.DataFrame(
    {
        "patient": ["a", "a", "a", "b", "b"],
        "time": [
            "2023-01-01 00:00",
            "2023-01-01 00:05",
            "2023-01-01 00:10",
            "2023-01-02 00:00",
            "2023-01-02 00:05",
        ],
        "heartrate": [60, 65, 70, 80, 85],
        "blood_pressure": [120, 125, None, 130, 135],
    }
).with_columns(pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M"))

# Initialize and run preliminary analysis
preliminary = Preliminary(
    df=df,
    id_col="patient",
    clock_col="time",
    clock_no_col="clock_no",
    cols=["heartrate", "blood_pressure"],
    alpha=0.05,
    save_path=Path("results/preliminary"),
    plot_library="plotly",
    renderer="notebook_connected",
)

# Run all analyses
preliminary.run(lags=10, save_results=True, use="pairwise")

# Or just the per-case summary measures of the observed values
preliminary.case_metrics(save_results=True)
print(preliminary.results.cm_case_metrics)  # one row per (case, variable)

# Generate HTML report
preliminary.generate_html_report(
    report_path="preliminary_report.html", title="Preliminary Analysis Report"
)
```

---

## References

1. **Shapiro, S. S., & Wilk, M. B.** (1965). An analysis of variance test for normality (complete samples). *Biometrika*, 52(3/4), 591-611. https://doi.org/10.2307/2333709
   - Used for: Shapiro-Wilk test statistic definition and $a_i$ coefficients formula.
   - SciPy implementation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

2. **Henze, N., & Zirkler, B.** (1990). A class of invariant consistent tests for multivariate normality. *Communications in Statistics - Theory and Methods*, 19(10), 3595-3617. https://doi.org/10.1080/03610929008830400
   - Used for: Henze-Zirkler test statistic formula, smoothing parameter $\beta$, and Mahalanobis distance definitions.

3. **Spearman, C.** (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72-101. https://doi.org/10.2307/1412159
   - Used for: Spearman rank correlation formula (both simplified and Pearson-on-ranks formulations).

4. **Wikipedia contributors.** Autocorrelation. *Wikipedia, The Free Encyclopedia*. https://en.wikipedia.org/wiki/Autocorrelation
   - Used for: Sample autocorrelation coefficient estimator formula.

5. **Vallat, R.** (2018). Pingouin: statistics in Python. *Journal of Open Source Software*, 3(31), 1026. https://doi.org/10.21105/joss.01026
   - Used for: `pingouin.multivariate_normality` implementation of the Henze-Zirkler test.
   - Documentation: https://pingouin-stats.org/build/html/generated/pingouin.multivariate_normality.html

6. **Matthews, J. N. S., Altman, D. G., Campbell, M. J., & Royston, P.** (1990). Analysis of serial measurements in medical research. *BMJ*, 300(6719), 230-235. https://doi.org/10.1136/bmj.300.6719.230
   - Used for: the summary-measures ("two-stage") rationale behind `case_metrics` — reducing each case's series to one value per variable before testing across cases, rather than pooling correlated within-case observations.
