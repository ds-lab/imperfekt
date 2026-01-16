# Intervariable Imperfection Analysis Module

This module provides a comprehensive suite of analyses for examining **intervariable imperfection patterns** — how imperfection in one variable relates to imperfection or observed values in other variables. It addresses missingness mechanism classification (MCAR, MAR, MNAR) and co-imperfection structures.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Class Structure](#class-structure)
4. [Analysis Methods](#analysis-methods)
   - [Row Statistics](#1-row-statistics-row_statistics)
   - [MCAR Test](#2-mcar-test-mcar_test)
   - [MAR/MNAR Test](#3-marmnar-test-mar_mnar_test)
   - [Symmetric Correlation](#4-symmetric-correlation-symmetric_correlation)
   - [Symmetric Lagged Cross-Correlation](#5-symmetric-lagged-cross-correlation-symmetric_lagged_cross_correlation)
   - [Asymmetric Correlation](#6-asymmetric-correlation-asymmetric_correlation)
5. [Missingness Mechanism Classification](#missingness-mechanism-classification)
6. [Usage Example](#usage-example)
7. [References](#references)

---

## Overview

The `IntervariableImperfection` class analyzes imperfection patterns **across multiple variables**. It addresses questions such as:

- Do variables tend to be imperfect together (co-imperfection)?
- Is imperfection **completely at random** (MCAR), or does it depend on other variables?
- Does the imperfection of one variable predict the **observed values** of another (asymmetric analysis)?
- Are there temporal lead-lag relationships between imperfection patterns?

---

## Class Structure

```
IntervariableImperfection
├── Parameters (constructor)
│   ├── df: pl.DataFrame              # Original data
│   ├── imperfection: str             # Type of imperfection (default: "missingness")
│   ├── mask_df: pl.DataFrame         # Custom binary mask (optional)
│   ├── id_col: str                   # Unique identifier column
│   ├── clock_col: str                # Temporal ordering column
│   ├── clock_no_col: str             # Integer time index column
│   ├── cols: list                    # Columns to analyze
│   ├── alpha: float                  # Significance level (default: 0.05)
│   ├── save_path: Path               # Output directory
│   ├── plot_library: str             # "matplotlib" or "plotly"
│   └── renderer: str                 # Plotly renderer
│
├── Methods
│   ├── row_statistics()              # Row-wise imperfection analysis
│   ├── mcar_test()                   # Little's MCAR test
│   ├── mar_mnar_test()               # Temporal MAR/MNAR test
│   ├── symmetric_correlation()       # Co-imperfection correlation
│   ├── symmetric_lagged_cross_correlation()  # Lagged co-imperfection
│   ├── asymmetric_correlation()      # Missing vs observed values
│   ├── run()                         # Execute all analyses
│   └── generate_html_report()        # Create HTML summary
│
└── results: IntervariableResults
    ├── rs_overall_statistics: pl.DataFrame
    ├── rs_case_level_statistics: pl.DataFrame
    ├── rs_empty_statistics: pl.DataFrame
    ├── mcar_results: pl.DataFrame
    ├── mar_mnar_results: pl.DataFrame
    ├── sc_symmetric_correlation: pl.DataFrame
    ├── sc_chi2_intervariable_matrix: pl.DataFrame
    ├── sc_symmetric_crosscorrelation: dict
    ├── ac_asymmetric_statistical_results: dict
    ├── ac_asymmetric_crosscorrelation: dict
    └── plots: IntervariablePlots
```

---

## Analysis Methods

### 1. Row Statistics (`row_statistics`)

Analyzes row-level imperfection patterns: how many variables are imperfect per observation.

**Library**: [Polars](https://docs.pola.rs/)

#### Metrics Computed

| Metric | Description |
|--------|-------------|
| `all_null_rows` | Rows where all analyzed variables are imperfect |
| `all_null_pct` | Percentage of *completely* imperfect rows |
| `indicated_vars` | Count of imperfect variables per row |
| `indicated_vars_pct` | Percentage of variables imperfect per row, e.g. 2 of 4 variables are missing at t=2 ➡️ 50% |

#### Case-Level Analysis

Computes the same metrics grouped by `id_col` to identify cases with unusually high row-level imperfection.

#### Visualization

- **Histogram**: Distribution of imperfect-variable counts per row
- **Boxplot**: Summary across cases

---

### 2. MCAR Test (`mcar_test`)

Tests whether imperfection is **Missing Completely At Random (MCAR)** using Little's MCAR test [[1]](#references).

**Library**: [SciPy](https://docs.scipy.org/) (`scipy.stats.chi2`)

#### Hypotheses

- **$H_0$ (Null)**: Imperfection is MCAR — the probability of imperfection does not depend on observed or unobserved data.
- **$H_1$ (Alternative)**: Imperfection is not MCAR — there is systematic structure to imperfection patterns.

#### Little's MCAR Test Statistic

The test compares subgroup means (defined by imperfection patterns) to the overall mean using a chi-squared statistic ($\chi^2$)[[1]](#references):

$$
d^2 = \sum_{j=1}^{J} n_j (\bar{\mathbf{x}}_j - \hat{\boldsymbol{\mu}})^\top \hat{\boldsymbol{\Sigma}}_{j}^{-1} (\bar{\mathbf{x}}_j - \hat{\boldsymbol{\mu}})
$$

Where:
- $J$ = number of distinct imperfection patterns
- $n_j$ = sample size of pattern $j$
- $\bar{\mathbf{x}}_j$ = mean of observed variables in pattern $j$
- $\hat{\boldsymbol{\mu}}$ = estimated global mean (from complete cases)
- $\hat{\boldsymbol{\Sigma}}_j$ = estimated covariance matrix for observed variables in pattern $j$

Under MCAR, $d^2 \sim \chi^2_{df}$ where $df = \sum_j k_j - p$ ($k_j$ = number of observed variables in pattern $j$, $p$ = total variables).

#### P-Value Calculation

The p-value is computed from the chi-squared distribution:

$$
p = 1 - F_{\chi^2}(d^2; df)
$$

Where $F_{\chi^2}$ is the cumulative distribution function of the chi-squared distribution with $df$ degrees of freedom.

#### Interpretation

| Condition | Interpretation |
|-----------|----------------|
| $p > \alpha$ | Fail to reject $H_0$; **MCAR is plausible** — no evidence that imperfection depends on data values |
| $p \leq \alpha$ | Reject $H_0$; **MCAR is rejected** — imperfection patterns differ systematically across subgroups (proceed to MAR/MNAR testing) |

> **Caution**: A non-significant result does not prove MCAR — it only means there is insufficient evidence to reject it. The test has limited power with small samples or few patterns.

#### Effect Size Measures

These are **adapted effect sizes** for Little's MCAR test (not classical contingency table measures):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Phi ($\phi$) | $\sqrt{d^2 / n}$ | Sample-size normalized effect; larger values indicate stronger deviation from MCAR |
| Cramér's V | $\sqrt{d^2 / (n \cdot \min(J-1, m-1))}$ | Bounded version (0 to 1); where $J$ = patterns, $m$ = variables |

#### Visualization

**UpSet Plot**: Visualizes the frequency of each imperfection pattern (which combinations of variables are imperfect together).

---

### 3. MAR/MNAR Test (`mar_mnar_test`)

Tests whether imperfection is **Missing At Random (MAR)** or **Missing Not At Random (MNAR)** using a temporal logistic regression approach.

**Library**: [scikit-learn](https://scikit-learn.org/) (`LogisticRegression`)

#### Concept

This test compares two nested logistic regression models per variable, as inspired by [[4]](#references):

1. **MAR Model**: Predicts imperfection in variable $Y$ using:
   - Other variables at time $t$
   - Lagged values of other variables ($t-1$)
   - Severity indicators (abnormal count, trend)

2. **MNAR Model**: MAR predictors + **own lagged value** ($Y_{t-1}$)

If including $Y_{t-1}$ significantly improves prediction, this suggests MNAR — the imperfection depends on the variable's own (unobserved) value.

#### Likelihood Ratio Test

$$
LRT = 2 \cdot (\ell_{\text{MNAR}} - \ell_{\text{MAR}}) \sim \chi^2_1
$$

Where $\ell$ denotes the log-likelihood. Under MAR, the LRT follows a chi-squared distribution with 1 degree of freedom.

#### Output Metrics

| Metric | Description |
|--------|-------------|
| `lrt_statistic` | Likelihood ratio test statistic |
| `p_value` | p-value for the LRT |
| `decision` | "Likely MNAR" if $p < \alpha$, else "No strong evidence against MAR" |
| `coef_lag1_mnar` | Coefficient for own lag in MNAR model (positive = higher value → more imperfection) |
| `auc_mar`, `auc_mnar` | ROC AUC for each model |

#### Interpretation

- **Significant LRT ($p < \alpha$)**: Evidence of MNAR — imperfection probability depends on the variable's own lagged value.
- **Positive `coef_lag1_mnar`**: Higher previous values predict more imperfection.
- **Negative `coef_lag1_mnar`**: Lower previous values predict more imperfection.

---

### 4. Symmetric Correlation (`symmetric_correlation`)

Analyzes **co-imperfection**: whether imperfection in one variable correlates with imperfection in another. This method provides two complementary analyses:

**Library**: [Polars](https://docs.pola.rs/) (`pl.corr`), [SciPy](https://docs.scipy.org/) (`chi2_contingency`)

#### Two Analyses in One Method

| Analysis | Function | Purpose | Output |
|----------|----------|---------|--------|
| **Correlation Matrix** | `corr_matrix()` | Quick overview of co-imperfection strength | `sc_symmetric_correlation` |
| **Chi-Squared Test** | `chi2_intervariable_imperfection_matrix()` | Statistical significance + effect size | `sc_chi2_intervariable_matrix` |

---

#### A. Correlation Matrix (Phi Coefficient)

Computes **Pearson correlation** between binary imperfection indicators using `pl.corr()`. For binary variables, Pearson correlation equals the **phi coefficient** ($\phi$).

**Range**: $-1 \leq \phi \leq 1$

| Value | Interpretation |
|-------|----------------|
| $\phi = 1$ | Perfect positive co-imperfection (always imperfect together) |
| $\phi = 0$ | No linear association |
| $\phi = -1$ | Perfect negative (when one is imperfect, the other is observed) |

**Visualizations**:
- **Heatmap**: Matrix showing $\phi$ for all variable pairs
- **Dendrogram**: Hierarchical clustering by co-imperfection similarity

---

#### B. Chi-Squared Test for Independence

Tests whether imperfection in two variables is **statistically independent** using `scipy.stats.chi2_contingency` on a 2×2 contingency table.

##### Test Statistic

$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where $O_{ij}$ = observed count, $E_{ij}$ = expected count under independence.

##### Hypotheses

- **$H_0$**: Imperfection in variable $i$ is independent of imperfection in variable $j$
- **$H_1$**: There is an association between imperfection in variables $i$ and $j$

##### Effect Size: Cramér's V

For a 2×2 table, Cramér's V equals $|\phi|$:

$$
V = \sqrt{\frac{\chi^2}{n}}
$$


---

#### Why Both?

- **Phi/Correlation Matrix**: Fast, shows direction (positive vs negative), good for visualization
- **Chi-Squared Test**: Provides p-values for statistical significance and standardized effect size

---

### 5. Symmetric Lagged Cross-Correlation (`symmetric_lagged_cross_correlation`)

Analyzes temporal lead-lag relationships between imperfection patterns of different variables.

**Library**: [Polars](https://docs.pola.rs/)

#### Cross-Correlation Function

$$
\rho_{XY}(k) = \text{Corr}(M_X(t), M_Y(t+k))
$$

Where:
- $M_X(t)$, $M_Y(t)$ are imperfection indicators at time $t$
- $k$ is the lag (negative = $X$ leads $Y$; positive = $Y$ leads $X$)

#### Interpretation

| Pattern | Interpretation |
|---------|----------------|
| Peak at $k < 0$ | Imperfection in $X$ precedes imperfection in $Y$ |
| Peak at $k > 0$ | Imperfection in $Y$ precedes imperfection in $X$ |
| Peak at $k = 0$ | Simultaneous co-imperfection |

---

### 6. Asymmetric Correlation (`asymmetric_correlation`)

Analyzes relationships between **imperfection in one variable** and **observed values in another** — a key indicator of MAR/MNAR mechanisms.

**Library**: Custom implementation using rank-biserial correlation

#### Rank-Biserial Correlation

Measures the correlation between a binary indicator (imperfection) and a continuous variable (observed values) [[2]](#references):

$$
r_{rb} = \frac{2U}{n_1 n_0} - 1
$$

Where:
- $U$ = Mann-Whitney U statistic
- $n_1$ = count of imperfect observations
- $n_0$ = count of non-imperfect observations

#### Interpretation

| $r_{rb}$ | Interpretation |
|----------|----------------|
| $r_{rb} > 0$ | Higher values of variable $Y$ are associated with imperfection in variable $X$ |
| $r_{rb} < 0$ | Lower values of variable $Y$ are associated with imperfection in variable $X$ |
| $r_{rb} = 0$ | No association |

#### Lagged Asymmetric Analysis

Extends the analysis to temporal relationships:

$$
r_{rb}(k) = \text{RankBiserial}(M_X(t), Y(t+k))
$$

This answers: "When $X$ is imperfect at time $t$, what were the values of $Y$ at time $t \pm k$?"

---

## Missingness Mechanism Classification

The module helps classify imperfection into three categories [[3]](#references):

| Mechanism | Definition | Test |
|-----------|------------|------|
| **MCAR** | Imperfection is independent of all data (observed and unobserved) | Little's MCAR test |
| **MAR** | Imperfection depends only on observed data | MAR/MNAR test (non-significant) |
| **MNAR** | Imperfection depends on unobserved data | MAR/MNAR test (significant) |

### Decision Flow

```
                    Little's MCAR Test
                           │
              ┌────────────┴────────────┐
              │                         │
         p > α                      p ≤ α
      (MCAR plausible)          (Not MCAR)
              │                         │
              ▼                         ▼
         Stop here              MAR/MNAR Test
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                     p > α                      p ≤ α
                   (MAR likely)              (MNAR likely)
```

---

## Usage Example

```python
import polars as pl
from pathlib import Path
from imperfekt.analysis.intervariable import IntervariableImperfection

# Load your data
df = pl.DataFrame({
    "patient": ["a", "a", "a", "a", "b", "b", "b"],
    "time": [
        "2023-01-01 08:00", "2023-01-01 08:05", "2023-01-01 08:10", "2023-01-01 08:15",
        "2023-01-02 12:00", "2023-01-02 12:05", "2023-01-02 12:10"
    ],
    "heartrate": [60, None, 70, None, 80, 85, None],
    "blood_pressure": [120, 125, None, None, 130, None, 140],
    "resprate": [12, 14, None, 16, 18, None, 20],
}).with_columns(
    pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M")
)

# Initialize analysis
analysis = IntervariableImperfection(
    df=df,
    imperfection="missingness",
    id_col="patient",
    clock_col="time",
    clock_no_col="clock_no",
    cols=["heartrate", "blood_pressure", "resprate"],
    alpha=0.05,
    save_path=Path("results/intervariable"),
    plot_library="plotly",
    renderer="notebook_connected",
)

# Run all analyses
analysis.run(
    save_results=True,
    lagged_crosscorr_max_lag=10,
)

# Or run individual analyses
analysis.row_statistics()
analysis.mcar_test()
analysis.mar_mnar_test()
analysis.symmetric_correlation()
analysis.asymmetric_correlation()

# Generate HTML report
analysis.generate_html_report(
    report_path="intervariable_report.html",
    title="Intervariable Imperfection Analysis"
)
```

---

## References

1. **Little, R. J. A.** (1988). A test of missing completely at random for multivariate data with missing values. *Journal of the American Statistical Association*, 83(404), 1198-1202. https://doi.org/10.1080/01621459.1988.10478722
   - Used for: Little's MCAR test formulation and chi-squared statistic.

2. **Wendt, H. W.** (1972). Dealing with a common problem in social science: A simplified rank-biserial coefficient of correlation based on the U statistic. *European Journal of Social Psychology*, 2(4), 463-465.
   - Used for: Rank-biserial correlation formula for asymmetric analysis.
   - Wikipedia: https://en.wikipedia.org/wiki/Mann–Whitney_U_test#Rank-biserial_correlation

3. **Rubin, D. B.** (1976). Inference and missing data. *Biometrika*, 63(3), 581-592. https://doi.org/10.1093/biomet/63.3.581
   - Used for: MCAR/MAR/MNAR framework definitions.

4. **Alasal, L. M., Hammarlund, E. U., Pienta, K. J., Rönnstrand, L. & Kazi, J. U.** (2025). XeroGraph: enhancing data integrity in the presence of missing values with statistical and predictive analysis. Bioinformatics Advances, 5(1). https://doi.org/10.1093/bioadv/vbaf035
    - Used for: MAR-MNAR LRT
  
