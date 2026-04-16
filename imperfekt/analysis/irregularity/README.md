# Irregularity Analysis Module

This module provides a suite of analyses for quantifying the **irregularity of the observation time grid** in panel data. Irregularity refers to unevenness in the spacing of observations across time — entities may be observed densely, sparsely, or in bursts, and this structure is meaningful independently of any variable-level imperfection.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Class Structure](#class-structure)
4. [Analysis Methods](#analysis-methods)
   - [Interval Statistics](#1-interval-statistics-interval_statistics)
   - [Dominant Frequency](#2-dominant-frequency-dominant_frequency)
   - [Burstiness](#3-burstiness-burstiness)
   - [Interval Autocorrelation](#4-interval-autocorrelation-interval_autocorrelation)
   - [Case Entropy & Adherence](#5-case-entropy--adherence-case_entropy_adherence)
   - [Composite Irregularity Score](#6-composite-irregularity-score-composite_score)
5. [Usage Example](#usage-example)
6. [References](#references)

---

## Overview

The `Irregularity` class analyzes the structure of the **inter-observation interval** (delta_t): the time between consecutive observations for each case. It addresses questions such as:

- How variable is the spacing between observations, per case and globally?
- Is there a dominant observation rhythm, and how closely does the data adhere to it?
- Do observations cluster in bursts (dense periods followed by long gaps)?
- Are successive gaps correlated — does a long gap tend to be followed by another long gap?
- How can individual entities be ranked and stratified by their overall irregularity?

The clock column (`clock_col`) may be a `Datetime` column or a numeric column (integer/float representing seconds). All interval computations produce a `Float64` `interval_seconds` column regardless of input type.

---

## Class Structure

```
Irregularity
├── Parameters (constructor)
│   ├── df: pl.DataFrame              # Input data
│   ├── id_col: str                   # Case identifier column (default: "id")
│   ├── clock_col: str                # Time/clock column — Datetime or numeric seconds (default: "clock")
│   ├── save_path: Path               # Output directory (optional)
│   ├── plot_library: str             # "matplotlib" or "plotly" (default: "matplotlib")
│   └── renderer: str                 # Plotly renderer (default: "notebook_connected")
│
├── Methods
│   ├── interval_statistics()         # Per-case and global interval summary statistics
│   ├── dominant_frequency()          # Modal interval, adherence rate, entropy (global)
│   ├── burstiness()                  # Burstiness coefficient per case and globally
│   ├── interval_autocorrelation()    # Autocorrelation of the interval sequence
│   ├── case_entropy_adherence()    # Per-case Shannon entropy and adherence rate
│   ├── composite_score()             # Composite irregularity score and Q1–Qn strata
│   └── run()                         # Execute all analyses
│
└── results: IrregularityResults
    ├── ins_case_statistics: pl.DataFrame
    ├── ins_global_statistics: pl.DataFrame
    ├── domf_frequency_summary: pl.DataFrame
    ├── domf_bin_counts: pl.DataFrame
    ├── bu_case_burstiness: pl.DataFrame
    ├── bu_global_burstiness: pl.DataFrame
    ├── ia_autocorrelation: pl.DataFrame
    ├── ea_case_entropy_adherence: pl.DataFrame
    ├── cs_case_scores: pl.DataFrame
    └── plots: IrregularityPlots
        ├── ins_cv_violin
        ├── domf_interval_frequency_bar
        ├── bu_burstiness_violin
        └── ia_acf_plot
```

---

## Analysis Methods

### 1. Interval Statistics (`interval_statistics`)

Computes per-case and global summary statistics of inter-observation intervals. The **coefficient of variation (CV)** is the primary irregularity score.


#### Metrics Computed

| Metric | Description |
|--------|-------------|
| `n_intervals` | Number of inter-observation intervals for this case |
| `mean_seconds` | Mean interval length |
| `median_seconds` | Median interval length |
| `std_seconds` | Standard deviation of interval lengths |
| `cv` | Coefficient of variation: $\text{CV} = \sigma / \mu$. CV = 0 for a perfectly regular grid; higher values indicate increasing irregularity |
| `iqr_seconds` | Interquartile range (Q75 − Q25) of interval lengths |
| `min_seconds` | Shortest interval |
| `max_seconds` | Longest interval |

Global statistics are computed over all pooled intervals via `describe()`.

Entities with only one observation (zero intervals after differencing) are preserved in the output with NaN for all statistics.

#### Visualization

**Violin plot**: distribution of per-case CV values across the cohort.

---

### 2. Dominant Frequency (`dominant_frequency`)

Identifies the **modal inter-observation interval** globally and quantifies how consistently the data follows it, along with the overall spread of the interval distribution.


#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_resolution_seconds` | 60.0 | Bin width for discretizing intervals. Reduce for sub-minute data |
| `adherence_tolerance` | 0.5 | Fractional tolerance around the mode for adherence rate |

#### Method

1. Discretize intervals: $b_i = \text{round}(\Delta t_i / r)$ where $r$ is `bin_resolution_seconds`
2. Find the mode bin (highest count) → **dominant interval**
3. Compute adherence rate and Shannon entropy (see below)

#### Adherence Rate

Fraction of all intervals within
                  [dominant_interval * (1 - tolerance), dominant_interval * (1 + tolerance)].

Answers "how often does the data follow its own dominant rhythm?"
 — values near 1 indicate a consistent schedule even under small jitter.

#### Shannon Entropy [[1]](#references)

$H_{\text{norm}} \in [0, 1]$: 0 when all intervals fall in a single bin (perfectly regular), 1 when uniformly spread across all bins (maximally irregular). Unlike the adherence rate, entropy captures how chaotic the non-dominant portion of the distribution is.

#### Output

| Field | Description |
|-------|-------------|
| `dominant_interval_seconds` | Center of the most frequent bin |
| `adherence_rate` | Fraction of intervals within the tolerance band around the mode |
| `n_total_intervals` | Total number of intervals |
| `n_adhering_intervals` | Count of intervals within the adherence band |
| `interval_entropy_bits` | Raw Shannon entropy in bits |
| `normalized_entropy` | Entropy rescaled to [0, 1] |
| `n_unique_bins` | Number of distinct interval bins observed |

#### Visualization

**Bar chart**: frequency of the top-20 interval bins, with the dominant bin highlighted.

---

### 3. Burstiness (`burstiness`)

Quantifies whether observations cluster in **bursts** — dense periods of activity separated by long gaps.


#### Burstiness Coefficient

$$
B = \frac{\sigma - \mu}{\sigma + \mu}
$$

Where $\mu$ and $\sigma$ are the mean and standard deviation of inter-observation intervals for an case (Goh & Barabasi, 2008) [[2]](#references).

| Value | Interpretation |
|-------|----------------|
| $B = -1$ | Perfectly regular (all intervals equal) |
| $B = 0$ | Poisson process (random, memoryless) |
| $B > 0$ | Bursty (clusters of activity separated by long gaps) |
| $B = 1$ | Maximally bursty (single burst, otherwise silent) |

Entities with fewer than 3 intervals receive NaN for `std_interval` and `burstiness_coeff`.

Global burstiness is computed over all pooled intervals (again entities with fewer than 3 intervals are removed).

#### Visualization

**Violin plot**: distribution of per-case burstiness coefficients across the cohort.

---

### 4. Interval Autocorrelation (`interval_autocorrelation`)

Tests whether **successive inter-observation gaps are correlated** — i.e., whether a long gap tends to be followed by another long gap (or short by short) $k$ steps later.

Custom implementation (reuses `autocorrelation.acf()` from `intravariable`)

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lags` | 20 | Maximum number of lags to compute |

#### Interpretation

| Pattern | Interpretation |
|---------|----------------|
| High positive ACF at lag 1 | Long gaps tend to follow long gaps (persistent clustering) |
| Near-zero ACF | Gap lengths are approximately uncorrelated (Poisson-like) |
| Negative ACF | Alternating short/long gaps (anti-persistent) |
| Periodic spikes | Systematic rhythm in gap structure |

#### Visualization

**Scatter/line plot**: autocorrelation coefficient vs. lag number.

---

### 5. Case Entropy & Adherence (`case_entropy_adherence`)

Computes **per-case** Shannon entropy and adherence rate — the local-level counterpart to the global `dominant_frequency` analysis.

#### Key design choice: adherence to own dominant

Adherence is measured against each case's **own** dominant interval, not the dataset-wide dominant. A patient with a unique but perfectly consistent 45-minute rhythm scores `adherence_rate = 1.0`. Using the global dominant would conflate dataset-level and case-level irregularity.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_resolution_seconds` | 60.0 | Bin width for discretizing intervals |
| `adherence_tolerance` | 0.5 | Fractional tolerance around each case's own dominant interval |
| `min_intervals` | 2 | Minimum intervals required; entities below this threshold receive NaN |

#### Method

For each case $i$ with $N_i$ intervals and bin resolution $r$:

**Empirical distribution:**

$$p_{i,j} = \frac{n_{i,j}}{N_i}$$

where $n_{i,j}$ is the count of case $i$'s intervals falling in bin $j$.

**Shannon entropy:**

$$H_i = -\sum_{j} p_{i,j} \log_2 p_{i,j}, \qquad \tilde{H}_i = \frac{H_i}{\log_2 K_i} \in [0, 1]$$

where $K_i$ is the number of distinct bins for case $i$. Defined as 0 when $K_i = 1$.

**Adherence rate:**

$$A_i = \frac{1}{N_i} \sum_{k} \mathbf{1}\!\left[\Delta t_{i,k} \in \left[\delta_i^* (1 - \tau),\; \delta_i^* (1 + \tau)\right]\right]$$

where $\delta_i^* = b_i^* \cdot r$ is the center of case $i$'s own dominant bin and $\tau$ is `adherence_tolerance`.

#### Output

| Field | Description |
|-------|-------------|
| `entropy_bits` | Raw Shannon entropy $H_i$ in bits |
| `normalized_entropy` | $\tilde{H}_i \in [0, 1]$: 0 = perfectly regular, 1 = maximally irregular |
| `adherence_rate` | $A_i \in [0, 1]$: fraction of intervals near this case's own dominant rhythm |
| `n_adhering_intervals` | Count of intervals within the adherence band |

---

### 6. Composite Irregularity Score (`composite_score`)

Combines three complementary per-case irregularity metrics into a single **composite score**, then assigns each case to a quantile-based irregularity stratum (Q1 = least irregular, Qn = most irregular).

Requires `interval_statistics()` and `burstiness()` to have been run first. Reuses `case_entropy_adherence()` results if already computed.

#### The three features

| Feature | Symbol | Range | Captures |
|---------|--------|-------|----------|
| Coefficient of Variation | $\text{CV}_i = \sigma_{\Delta t} / \mu_{\Delta t}$ | $[0, \infty)$ | Magnitude of interval dispersion |
| Burstiness Coefficient | $B_i = (\sigma - \mu)/(\sigma + \mu)$ | $[-1, 1]$ | Shape — clustered vs. regular |
| Normalized Entropy | $\tilde{H}_i$ | $[0, 1]$ | Distributional complexity and multimodality |

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_resolution_seconds` | 60.0 | Bin width for entropy/adherence computation |
| `adherence_tolerance` | 0.5 | Fractional tolerance for adherence rate |
| `min_intervals` | 2 | Minimum intervals for entropy/adherence |
| `weights` | None | Feature weights, e.g. `{"cv": 1.0, "burstiness_coeff": 1.0, "normalized_entropy": 1.0}`. None = equal weights |
| `n_quantiles` | 4 | Number of strata (default 4 → Q1…Q4) |

#### Method

**Step 1 — Z-score standardization** (computed over entities with all three features non-null):

$$z_{i,f} = \frac{f_i - \mu_f}{\sigma_f}, \quad f \in \{\text{CV},\, B,\, \tilde{H}\}$$

**Step 2 — Weighted composite score:**

$$S_i = \frac{w_{\text{CV}} \cdot z_{i,\text{CV}} + w_B \cdot z_{i,B} + w_{\tilde{H}} \cdot z_{i,\tilde{H}}}{w_{\text{CV}} + w_B + w_{\tilde{H}}}$$

Default: $w_{\text{CV}} = w_B = w_{\tilde{H}} = 1$.

**Step 3 — Quantile stratification:**

$$\text{stratum}_i = Q_q \quad \text{if} \quad q_{q-1} < S_i \leq q_q$$

where $q_1, \ldots, q_{n-1}$ are the $\tfrac{1}{n}, \tfrac{2}{n}, \ldots, \tfrac{n-1}{n}$ quantiles of $\{S_i\}$.

Entities with NaN for any feature receive NaN for `composite_score` and `irregularity_stratum`.

#### Output (`cs_case_scores`)

| Field | Description |
|-------|-------------|
| `cv` | Per-case coefficient of variation |
| `burstiness_coeff` | Per-case burstiness coefficient |
| `normalized_entropy` | Per-case normalized Shannon entropy |
| `adherence_rate` | Per-case adherence to own dominant rhythm |
| `composite_score` | $S_i$: weighted Z-score average of the three standardized features |
| `irregularity_stratum` | Q1 (least irregular) … Qn (most irregular) |

---

## Usage Example

```python
import polars as pl
from pathlib import Path
from imperfekt.analysis.irregularity import Irregularity

df = pl.DataFrame({
    "patient": ["a", "a", "a", "a", "a", "b", "b", "b", "b"],
    "time": [0, 60, 120, 180, 240, 0, 60, 500, 560],  # seconds (numeric)
})

analysis = Irregularity(
    df=df,
    id_col="patient",
    clock_col="time",
    save_path=Path("results/irregularity"),
    plot_library="matplotlib",
    renderer="notebook_connected",
)

# Run all analyses (including composite score and stratification)
analysis.run(
    save_results=True,
    bin_resolution_seconds=60.0,
    adherence_tolerance=0.5,
    autocorrelation_lags=20,
)

# Access results
print(analysis.results.ins_case_statistics)       # CV and interval stats per case
print(analysis.results.bu_case_burstiness)        # Burstiness per case
print(analysis.results.ea_case_entropy_adherence) # Entropy and adherence per case
print(analysis.results.cs_case_scores)            # Composite score and Q1–Q4 strata
print(analysis.results.domf_frequency_summary)      # Global dominant frequency
print(analysis.results.ia_autocorrelation)          # Global interval autocorrelation

# Run individual analyses and chain
analysis.interval_statistics().burstiness().case_entropy_adherence().composite_score()

# Customize composite score
analysis.composite_score(
    weights={"cv": 2.0, "burstiness_coeff": 1.0, "normalized_entropy": 1.0},
    n_quantiles=5,
)
```

---

## References

1. **Shannon, C. E.** (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
   - Used for: Shannon entropy formula for quantifying spread of the interval distribution (global in `dominant_frequency`, per-case in `case_entropy_adherence`).

2. **Goh, K.-I., & Barabási, A.-L.** (2008). Burstiness and memory in complex systems. *EPL (Europhysics Letters)*, 81(4), 48002. https://doi.org/10.1209/0295-5075/81/48002
   - Used for: Burstiness coefficient $B = (\sigma - \mu) / (\sigma + \mu)$.

3. **Wikipedia contributors.** Autocorrelation. *Wikipedia, The Free Encyclopedia*. https://en.wikipedia.org/wiki/Autocorrelation
   - Used for: Sample autocorrelation coefficient estimator formula.
