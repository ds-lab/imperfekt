# Features Module

This module generates features from imperfection (missingness/noise) patterns for downstream machine learning tasks.

## Structure

| File | Purpose |
|------|---------|
| `core.py` | `FeatureGenerator` class orchestrating all feature generation |
| `temporal.py` | Time-based features: lags, consecutive counts, time-since |
| `window.py` | Rolling statistics: sum, variance, exponential moving average |
| `interaction.py` | Cross-variable features: pairwise interactions, row-level statistics |

---

## FeatureGenerator

The main class that coordinates feature generation from imperfection masks.

```python
from imperfekt.features import FeatureGenerator

fg = FeatureGenerator(
    df,
    id_col="patient_id",      # Entity identifier
    clock_col="timestamp",    # Datetime column
    variable_cols=["hr", "sbp", "rr"],  # Columns to analyze
    imperfection="missingness"  # Type of imperfection
)

# Generate all features
df_features = fg.generate_all_features()
```

### Mask Generation

For `imperfection="missingness"`, creates binary masks:

$$mask_{i,t} = \begin{cases} 1 & \text{if } x_{i,t} \text{ is null} \\ 0 & \text{otherwise} \end{cases}$$

---

## Feature Categories

### 1. Binary Masks

Base imperfection indicators for each variable.

| Output Column | Description |
|---------------|-------------|
| `{var}_mask` | Binary indicator (1 = imperfect, 0 = present) |

---

### 2. Circular Features

Encodes cyclical time patterns using sine/cosine transformation to preserve continuity (e.g., hour 23 is close to hour 0).

`hour_sin` $= \sin\left(\frac{2\pi \cdot h}{24}\right)$

`hour_cos` $= \cos\left(\frac{2\pi \cdot h}{24}\right)$

| Output Column | Description |
|---------------|-------------|
| `hour_sin` | Sine component of hour |
| `hour_cos` | Cosine component of hour |

---

### 3. Temporal Features

#### Lag Mask

Previous imperfection state, shifted by `lag` observations:

`mask_lag_{n}` $= mask_{t-\text{lag}}$

| Output Column | Description |
|---------------|-------------|
| `{var}_mask_lag_{n}`|  Mask value from `n` observations ago |

#### Consecutive Count

Running count of consecutive imperfections within each block:

| Mask Sequence | → | Count |
|---------------|---|-------|
| `[0, 1, 1, 1, 0, 1]` | → | `[0, 1, 2, 3, 0, 1]` |

| Output Column | Description |
|---------------|-------------|
| `{var}_mask_consecutive` | Count of consecutive imperfections |

#### Time Since

Elapsed time (in seconds) since last imperfect or non-imperfect observation:

| Output Column | Description |
|---------------|-------------|
| `{var}_time_since_imperfect` | Seconds since last missing value |
| `{var}_time_since_non_imperfect` | Seconds since last present value |

---

### 4. Window Features

#### Rolling Statistics

Sliding window aggregations over the past `w` observations:

`rolling_sum`$_t = \sum_{i=t-w+1}^{t}$ `mask`$_i$

`rolling_var`$_t = \text{Var}($`mask`$_{t-w+1}, \ldots,$ `mask`$_t)$

| Output Column | Description |
|---------------|-------------|
| `{var}_mask_rolling_sum_{w}` | Count of imperfections in window |
| `{var}_mask_rolling_var_{w}` | Variance (volatility) of imperfections |

#### Exponential Moving Average (EWMA)

Weighted average giving more importance to recent observations:

$$\text{EWMA}_t = \alpha \cdot \text{mask}_t + (1 - \alpha) \cdot \text{EWMA}_{t-1}$$

where $\alpha \in (0, 1)$ is the smoothing factor.

| Output Column | Description |
|---------------|-------------|
| `{var}_mask_ewma_{α}` | EWMA with smoothing factor α |

---

### 5. Interaction Features

#### Pairwise Interactions

For each ordered pair of variables $(A, B)$, generates 4 interaction types:

| Type | Formula | Description |
|------|---------|-------------|
| Concurrent value | $x_{A,t} \cdot mask_{B,t}$ | Value of A when B is missing |
| Concurrent mask | $mask_{A,t} \cdot mask_{B,t}$ | Both missing simultaneously |
| Predictive value | $x_{A,t-1} \cdot mask_{B,t}$ | Previous value of A before B is missing |
| Predictive mask | $mask_{A,t-1} \cdot mask_{B,t}$ | A was missing before B is missing |

**Feature count:** $4 \times N \times (N-1)$ for $N$ variables.

| Output Column | Description |
|---------------|-------------|
| `inter_{var_a}_t_x_{var_b}_mask` | Concurrent value interaction |
| `inter_{var_a}_mask_t_x_{var_b}_mask` | Concurrent mask interaction |
| `inter_{var_a}_t-1_x_{var_b}_mask` | Predictive value interaction |
| `inter_{var_a}_mask_t-1_x_{var_b}_mask` | Predictive mask interaction |

#### Row-Level Features

Aggregate imperfection statistics across all variables at each timestamp:

| Output Column | Description |
|---------------|-------------|
| `row_imperfection_pct` | Fraction of variables missing in this row |

---

## Quick Reference

| Method | Features Added |
|--------|----------------|
| `add_binary_masks()` | `{var}_mask` |
| `add_circular_features()` | `hour_sin`, `hour_cos` |
| `add_temporal_features()` | `{var}_mask_lag_*`, `{var}_mask_consecutive`, `{var}_time_since_*` |
| `add_window_features()` | `{var}_mask_rolling_*`, `{var}_mask_ewma_*` |
| `add_interaction_features()` | `inter_*` pairwise features |
| `add_row_imperfection_pct()` | `row_imperfection_pct` |
| `generate_all_features()` | All of the above |

---

## Example

```python
from imperfekt.features import FeatureGenerator
import polars as pl

df = pl.DataFrame({
    "patient": ["a", "a", "a", "a"],
    "time": ["2023-01-01 00:00", "2023-01-01 00:05", 
             "2023-01-01 00:10", "2023-01-01 00:15"],
    "hr": [80, None, None, 85],
    "sbp": [None, 120, None, 125],
}).with_columns(pl.col("time").str.to_datetime())

fg = FeatureGenerator(df, id_col="patient", clock_col="time")
df_features = fg.generate_all_features()

# Result: Original columns + ~30 new feature columns
```
