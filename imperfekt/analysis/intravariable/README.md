# Intravariable Imperfection Analysis Module

This module provides a comprehensive suite of analyses for examining **intravariable imperfection patterns** in time-series data. "Imperfection" refers to missingness, noise, or any anomaly that can be indicated using a binary mask.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Class Structure](#class-structure)
4. [Analysis Methods](#analysis-methods)
   - [Column Statistics](#1-column-statistics-column_statistics)
   - [Gap Statistics](#2-gap-statistics-gap_statistics)
   - [Markov Chain Summary](#3-markov-chain-summary-markov_chain_summary)
   - [Autocorrelation](#4-autocorrelation-autocorrelation)
   - [Windowed Significance](#5-windowed-significance-windowed_significance)
   - [DateTime Statistics](#6-datetime-statistics-date_time_statistics)
5. [Usage Example](#usage-example)
6. [References](#references)

---

## Overview

The `IntravariableImperfection` class analyzes imperfection patterns **within individual variables** over time. It addresses questions such as:

- How prevalent is imperfection in each variable?
- Do imperfect values cluster together (temporal autocorrelation)?
- What is the transition behavior between observed and imperfect states?
- Are there systematic patterns by time of day, month, or weekday?
- Do values after gaps differ systematically from other values (MNAR patterns)?

---

## Class Structure

```
IntravariableImperfection
├── Parameters (constructor)
│   ├── df: pl.DataFrame              # Original data
│   ├── imperfection: str             # Type of imperfection (optional, default: "missingness")
│   ├── mask_df: pl.DataFrame         # Custom binary mask (optional, can be calculated for missingness)
│   ├── id_col: str                   # Unique identifier column name (optional, default: "id")
│   ├── clock_col: str                # Temporal ordering column name (optional, default: "clock")
│   ├── clock_no_col: str             # Integer time index column name (optional, default: "clock_no", column will be generated based on clock_col)
│   ├── cols: list                    # Columns to analyze (optional)
│   ├── alpha: float                  # Significance level (optional, default: 0.05)
│   ├── save_path: Path               # Output directory (optional)
│   ├── plot_library: str             # "matplotlib" or "plotly" (optional, default: "matplotlib")
│   └── renderer: str                 # Plotly renderer (optional, default: "notebook_connected")
│
├── Methods
│   ├── column_statistics()           # Imperfection prevalence per column
│   ├── gap_statistics()              # Gap lengths and return values
│   ├── markov_chain_summary()        # Transition probabilities
│   ├── autocorrelation()             # Temporal autocorrelation of imperfection
│   ├── windowed_significance()       # Values near imperfect instances
│   ├── date_time_statistics()        # Temporal distribution patterns
│   ├── run()                         # Execute all analyses
│   └── generate_html_report()        # Create HTML summary
│
└── results: IntravariableResults
    ├── cs_overall_statistics: pl.DataFrame
    ├── cs_case_level_statistics: pl.DataFrame
    ├── gs_gaps_observation_runs: pl.DataFrame
    ├── gs_gaps_df: dict[str, pl.DataFrame]
    ├── gr_gap_returns: pl.DataFrame
    ├── gr_gap_kruskal: dict
    ├── mc_markov_summary: dict
    ├── ac_autocorrelation: dict
    ├── ws_observations_around_indicated: dict
    ├── ws_mwu_result: pl.DataFrame
    ├── dt_date_time_statistics: dict
    └── plots: IntravariablePlots
```

---

## Analysis Methods

### 1. Column Statistics (`column_statistics`)

Quantifies the prevalence of imperfection for each variable at both overall and case (ID) levels.

**Library**: [Polars](https://docs.pola.rs/)

#### Metrics Computed

| Metric | Description |
|--------|-------------|
| `indicated_count` | Number of imperfect values |
| `indicated_pct` | Percentage of imperfect values: $\frac{indicated\_count}{n} \times 100$ |
| `non_indicated_count` | Number of non-imperfect values |
| `above_threshold` | Boolean flag if imperfection exceeds threshold (default: 5%) |

#### Case-Level Analysis

Computes the same metrics grouped by `id_col`, useful for identifying cases with unusually high imperfection rates.

#### Visualization

- **Histogram**: Distribution of imperfection percentages across cases
- **Boxplot**: Summary of imperfection rates per variable

---

### 2. Gap Statistics (`gap_statistics`)

Analyzes the temporal structure of gaps (consecutive imperfect values) and the values observed after gaps.

**Library**: [Polars](https://docs.pola.rs/), [SciPy](https://docs.scipy.org/) [[1]](#references)

#### Gap Length Analysis

Computes the duration between consecutive observed values:

$$
\text{gaplength}_i = t_{\text{next}} - t_{\text{prev}}
$$

Where $t$ represents timestamps of observed (non-imperfect) values.

#### Gap Return Analysis (MNAR Investigation)

Investigates whether values **after gaps** differ systematically from values after shorter gaps — a potential indicator of **Missing Not At Random (MNAR)** patterns.

1. **Binning**: Gap lengths are divided into quantile-based bins (default: 8 bins at 0.125 quantile intervals)
2. **Kruskal-Wallis Test**: Non-parametric test comparing return values across gap-length bins

##### Kruskal-Wallis H-Statistic

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
$$

Where [[1]](#references):
- $k$ = number of groups (gap bins)
- $n_i$ = sample size of group $i$
- $R_i$ = sum of ranks in group $i$
- $N$ = total sample size

##### Effect Size (Eta-squared)

$$
\eta^2 = \frac{H - k + 1}{N - k}
$$

##### Post-hoc Testing

If the Kruskal-Wallis test is significant ($p < \alpha$), pairwise comparisons are performed using the **Dwass-Steel-Critchlow-Fligner (DSCF)** test [[2]](#references).
#### Interpretation

- **$H_0$**: Return values are identically distributed across all gap-length bins
- **$H_1$**: At least one gap-length bin has a different distribution of return values
- Significant results suggest MNAR: the value after a gap depends on the gap duration

#### Visualization

- **Violin Plot**: Gap length distributions per variable
- **Boxplot**: Return values by gap-length bin
- **Heatmaps**: Post-hoc p-values and effect sizes

---

### 3. Markov Chain Summary (`markov_chain_summary`)

Models imperfection as a **two-state Markov chain** to quantify transition dynamics between observed and imperfect states.

**Library**: [NumPy](https://numpy.org/) (eigenvalue computation)

#### States

| State | Value | Description |
|-------|-------|-------------|
| 0 | Observed | Value is present/normal |
| 1 | Imperfect | Value is missing/noisy/indicated |

#### Transition Matrix

The transition probability matrix $\mathbf{P}$ is estimated from observed state sequences:

$$
\mathbf{P} = \begin{pmatrix} P_{00} & P_{01} \\ P_{10} & P_{11} \end{pmatrix}
$$

Where $P_{ij} = P(X_{t+1} = j \mid X_t = i)$ is estimated as:

$$
\hat{P}_{ij} = \frac{n_{ij}}{\sum_{k} n_{ik}}
$$

- $n_{ij}$ = count of transitions from state $i$ to state $j$

#### Steady-State Distribution

The long-run proportion of time spent in each state, computed as the left eigenvector of $\mathbf{P}$ corresponding to eigenvalue 1:

$$
\boldsymbol{\pi} \mathbf{P} = \boldsymbol{\pi}, \quad \sum_i \pi_i = 1
$$

#### Interpretation

| Metric | Interpretation |
|--------|----------------|
| $P_{00}$ high | Observed values tend to persist |
| $P_{11}$ high | Imperfect values cluster together (bursty imperfection) |
| $\pi_1$ | Long-run imperfection rate |

#### Visualization

**Transition Matrix Heatmap**: Visual representation of transition probabilities

---

### 4. Autocorrelation (`autocorrelation`)

Measures the temporal correlation of imperfection with its own lagged values.

**Library**: Custom implementation (see `autocorrelation.py`)

**Source**: [Wikipedia: Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) [[3]](#references)

#### Autocorrelation Estimator

$$
\hat{R}(k) = \frac{1}{n \cdot \hat{\sigma}^2} \sum_{t=1}^{n-k}(m_t - \bar{m})(m_{t+k} - \bar{m})
$$

Where:
- $m_t \in \{0, 1\}$ is the imperfection indicator at time $t$
- $\bar{m}$ is the mean imperfection rate
- $\hat{\sigma}^2$ is the sample variance of the indicator

#### Panel Data Adaptation

- Lags are computed **within each ID** (via `pl.shift().over(id_col)`)
- Invalid (cross-ID) lag pairs are excluded

#### Interpretation

| Pattern | Interpretation |
|---------|----------------|
| High positive ACF at lag 1 | Imperfection clusters (bursty) |
| Rapid decay | Short-term memory only |
| Periodic spikes | Systematic patterns (e.g., every $k$ observations) |

#### Visualization

**Lag Plot**: Autocorrelation coefficient vs. lag number

---

### 5. Windowed Significance (`windowed_significance`)

Extracts observed values within a temporal window around imperfect instances to investigate **local context effects**.

**Library**: [Polars](https://docs.pola.rs/)

#### Method

For each imperfect instance at time $t^*$:

1. Define a window: $[t^* - \Delta t, t^* + \Delta t]$ (default $\Delta t = 5$ minutes)
2. Collect all **observed** values of the same variable within this window
3. Compare the distribution of "near-imperfect" values to the overall distribution

#### Window Location Options

| Option | Window |
|--------|--------|
| `"before"` | $[t^* - \Delta t, t^*]$ |
| `"after"` | $[t^* , t^* + \Delta t]$ |
| `"both"` | $[t^* - \Delta t, t^* + \Delta t]$ |

#### Use Case

Helps identify **MNAR patterns**: if values near imperfect instances differ systematically (e.g., extreme values are more likely to be followed by missingness), this suggests the imperfection mechanism depends on the underlying value.

#### Visualization

- **Overlay Histogram**: Distribution of values near imperfect instances vs. all values
- **Multi-boxplot**: Comparison across variables

---

### 6. DateTime Statistics (`date_time_statistics`)

Analyzes imperfection patterns by calendar/clock time to detect **provider-level or system-level** patterns.

**Library**: [Polars](https://docs.pola.rs/), [Plotly](https://plotly.com/python/)

#### Temporal Groupings

| Grouping | Purpose |
|----------|---------|
| **Month** | Seasonal patterns, system updates |
| **Weekday** | Workflow differences (weekday vs. weekend) |
| **Hour** | Shift changes, workload variations |

#### Metrics per Group

- **Mean imperfection rate**: $\bar{m}_g = \frac{1}{n_g} \sum_{i \in g} m_i$
- **Count**: Total imperfect instances in group

#### Visualization

**Month × Hour Heatmap**: Two-dimensional view of imperfection rates across months and hours of day

---

## Usage Example

```python
import polars as pl
from pathlib import Path
from datetime import timedelta
from imperfekt.analysis.intravariable import IntravariableImperfection

# Load your data
df = pl.DataFrame({
    "patient": ["a", "a", "a", "a", "b", "b", "b"],
    "time": [
        "2023-01-01 08:00", "2023-01-01 08:05", "2023-01-01 08:10", "2023-01-01 08:15",
        "2023-01-02 12:00", "2023-01-02 12:05", "2023-01-02 12:10"
    ],
    "heartrate": [60, None, 70, None, 80, 85, None],
    "blood_pressure": [120, 125, None, None, 130, None, 140],
}).with_columns(
    pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M")
)

# Initialize analysis
analysis = IntravariableImperfection(
    df=df,
    imperfection="missingness",
    id_col="patient",
    clock_col="time",
    clock_no_col="clock_no",
    cols=["heartrate", "blood_pressure"],
    alpha=0.05,
    save_path=Path("results/intravariable"),
    plot_library="plotly",
    renderer="notebook_connected",
)

# Run all analyses
analysis.run(
    save_results=True,
    window_size=timedelta(minutes=5),
    window_location="both",
)

# Generate HTML report
analysis.generate_html_report(
    report_path="intravariable_report.html",
    title="Intravariable Imperfection Analysis"
)
```

---

## References

1. **Kruskal, W. H., & Wallis, W. A.** (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583-621. https://doi.org/10.1080/01621459.1952.10483441
   - Used for: Kruskal-Wallis H-statistic formula for comparing gap-return distributions.
   - Implementation: [scipy.stats](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)

2. **Dwass, M.** (1960). Some k-sample rank-order tests. In *Contributions to Probability and Statistics* (pp. 198-202). Stanford University Press.
   - Used for: Dwass-Steel-Critchlow-Fligner post-hoc test for pairwise comparisons.
   - Implementation: [scikit-posthocs](https://scikit-posthocs.readthedocs.io/)

3. **Wikipedia contributors.** Autocorrelation. *Wikipedia, The Free Encyclopedia*. https://en.wikipedia.org/wiki/Autocorrelation
   - Used for: Sample autocorrelation coefficient estimator formula.
