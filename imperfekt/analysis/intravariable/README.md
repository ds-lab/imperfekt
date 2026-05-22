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
   - [Composite Score](#7-composite-score-composite_score)
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
│   ├── composite_score()             # Per-(case × variable) quadrant stratification
│   ├── run()                         # Execute all analyses
│   └── generate_html_report()        # Create HTML summary
│
└── results: IntravariableResults
    ├── cs_overall_statistics: pl.DataFrame
    ├── cs_case_level_statistics: pl.DataFrame
    ├── gs_gaps_observation_runs: pl.DataFrame
    ├── gs_gaps_df: dict[str, pl.DataFrame]
    ├── gs_gap_dominant: dict[str, pl.DataFrame]     # per-column dominant gap length summary
    ├── gs_gap_burstiness: dict[str, pl.DataFrame]   # per-column gap burstiness summary
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

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_resolution_seconds` | `60.0` | Bin width in seconds for dominant gap length detection |
| `adherence_tolerance` | `0.5` | Fractional tolerance around the dominant gap for adherence rate |

#### Gap Length Analysis

Computes the duration between consecutive observed values:

$$
\text{gaplength}_i = t_{\text{next}} - t_{\text{prev}}
$$

Where $t$ represents timestamps of observed (non-imperfect) values.

#### Dominant Gap Length

Identifies the most frequently occurring gap duration using the same binning approach as the Irregularity module, enabling direct comparison between the two.

Only rows with `count_clock_no > 0` are used — i.e. intervals where at least one imperfect value sits between two observations. Rows with `count_clock_no == 0` are adjacent observations with no imperfection between them; those are equivalent to the inter-observation intervals analyzed by the Irregularity module and are excluded here to avoid overlap.

Gap lengths are discretised into bins of width `bin_resolution_seconds`. The bin with the highest count is the **dominant gap**.

| Metric | Description |
|--------|-------------|
| `dominant_gap_seconds` | Centre of the most frequent gap-length bin |
| `gap_adherence_rate` | Fraction of all gaps within $[\text{dominant} \times (1 - \text{tol}),\ \text{dominant} \times (1 + \text{tol})]$ |
| `gap_normalized_entropy` | $H / \log_2(n_{\text{bins}}) \in [0, 1]$: 0 = all gaps equal length, 1 = maximally spread |

#### Gap Burstiness

Quantifies whether gaps arrive in clusters or are evenly spaced, using the B coefficient [[4]](#references):

$$
B = \frac{\sigma - \mu}{\sigma + \mu}
$$

Where $\mu$ and $\sigma$ are the mean and standard deviation of all gap lengths for the variable. Range: $[-1, 1]$.

| Value | Interpretation |
|-------|----------------|
| $B = -1$ | Perfectly regular gaps (all equal length) |
| $B = 0$ | Poisson-like (random arrival) |
| $B > 0$ | Bursty gaps (clusters separated by long silences) |

Requires at least 3 gaps; returns `null` otherwise.

**Comparability with Irregularity module**: `gap_adherence_rate`, `gap_normalized_entropy`, and `gap_burstiness_coeff` use the same underlying computations as the Irregularity module's `adherence_rate`, `normalized_entropy`, and `burstiness_coeff`. This allows direct comparison of the observation-time rhythm (Irregularity) with the gap-length rhythm (Gap Statistics).

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

### 7. Composite Score (`composite_score`)

Assigns each (case × variable) pair to one of five imperfection strata, enabling subgroup analysis of model performance broken down by missingness pattern per variable.

#### Strata

| Stratum | Meaning |
|---------|---------|
| `Q_complete` | No imperfection for this case and variable |
| `Q_alpha` | Low irregularity on both selected axes |
| `Q_beta` | High on axis X, low on axis Y |
| `Q_gamma` | Low on axis X, high on axis Y |
| `Q_delta` | High irregularity on both axes |

#### Candidate Axes (per case, per variable)

All metrics below are computed per (case × variable). Only a subset is eligible for axis selection (see [Axis Selection](#axis-selection) below).

| Metric | Captures | Min. gaps required |
|--------|----------|--------------------|
| `indicated_pct` | Overall missingness burden | — |
| `gap_cv` | CV of gap lengths (std / mean) | ≥ 2 |
| `gap_qcod` | Quartile CoD of gap lengths — robust analog to CV: $(Q_{75}-Q_{25})/(Q_{75}+Q_{25})$ | ≥ 4 |
| `gap_burstiness_coeff` | Goh & Barabási burstiness of gap lengths [[4]](#references) | ≥ 3 |
| `gap_adherence_rate` | Fraction of gaps near the case's own dominant gap length (inverted: lower = more imperfect) | ≥ 1 |
| `gap_normalized_entropy` | Shannon entropy of the gap length distribution | ≥ 1 |
| `max_gap_fraction` | Largest single gap as fraction of total observation window | ≥ 1 |
| `gap_onset_cv` | CV of the spacing between consecutive gap start times | ≥ 3 |
| `gap_missing_centroid` | Mean clock-position of missing ticks on the case–variable's normalized $[0, 1]$ observation timeline ($\approx 0$ = front-loaded, $\approx 0.5$ = symmetric, $\approx 1$ = back-loaded) | — |
| `mc_p11` | Markov $P(1 \to 1)$: probability that imperfection persists into the next time step | — |

> **Limitation — gaps at the start or end of a case's timeline**: `time_length` (gap duration in seconds) is `null` for any gap that has no preceding or following observation within the same case, i.e. the imperfect run starts at the very first record or ends at the very last record. Because there is no bracketing observation to anchor the gap boundary, its duration is undefined. All metrics that rely on `time_length` (`gap_cv`, `gap_qcod`, `gap_burstiness_coeff`, `gap_adherence_rate`, `gap_normalized_entropy`, `max_gap_fraction`, `gap_onset_cv`) therefore require at least one gap with a finite duration — those metrics are `null` for a case that has only boundary gaps or only a single gap with an undefined length.

#### Axis Selection

Only the five axes that are structurally defined for every imperfect case are eligible for axis selection:

- `indicated_pct`, `gap_adherence_rate`, `gap_normalized_entropy`, `max_gap_fraction`, `gap_missing_centroid`

Axes derived from gap length distributions (`gap_cv`, `gap_qcod`, `gap_burstiness_coeff`, `gap_onset_cv`, `mc_p11`) require multiple gaps and produce `null` for cases with only one gap, making them unsuitable for axis selection across the full imperfect population. They are included in `iv_composite_scores` for reference but not used to split cases into quadrants.

Before computing correlations, axes that are near-constant (more than 50 % of values equal to the median) are excluded for that variable, as they cannot meaningfully bisect the population.

All pairwise Spearman rank correlations are computed across the eligible axes. The pair with the **lowest absolute correlation** (most orthogonal) is selected as the two stratification axes. Selection is performed independently per variable. The full correlation table is stored in `results.iv_pairwise_correlations`.

#### Median Bisection

The selected axes are split at their medians to produce the four quadrants. Medians are passed as parameters to `assign_strata()`, enabling leakage-free cross-validation: fit medians on the training fold, apply to the held-out test fold.

#### Output

- `results.iv_composite_scores` — one row per (case × variable) with all metrics, selected axes, thresholds, and `imperfection_stratum`
- `results.iv_pairwise_correlations` — dict keyed by variable name, each a correlation table used for axis selection

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
    bin_resolution_seconds=60.0,   # bin width for dominant gap detection
    adherence_tolerance=0.5,       # tolerance around dominant gap for adherence rate
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

4. **Goh, K.-I., & Barabási, A.-L.** (2008). Burstiness and memory in complex systems. *EPL (Europhysics Letters)*, 81(4), 48002. https://doi.org/10.48550/arXiv.physics/0610233
   - Used for: Burstiness coefficient $B = (\sigma - \mu) / (\sigma + \mu)$ applied to gap length distributions.
