# Features Module

This module generates features from imperfection (missingness/noise) patterns for downstream machine learning tasks.

## Structure

| File | Purpose |
|------|---------|
| `core.py` | `FeatureGenerator` class orchestrating all feature generation |
| `temporal.py` | Time-based features: lags, consecutive counts, time-since |
| `window.py` | Rolling statistics: sum, variance, exponential moving average |
| `interaction.py` | Cross-variable features: pairwise interactions, row-level statistics |
| `irregularity.py` | Sampling-rhythm features: interval gaps, z-scores, local CV, acceleration |

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

```python
fg.add_binary_masks()                        # all variable_cols
fg.add_binary_masks(cols=["hr", "sbp"])      # subset only
```

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

```python
fg.add_temporal_features()                   # all variable_cols
fg.add_temporal_features(cols=["hr"])        # subset only
```

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

```python
fg.add_window_features(rolling_window_sizes=[2], ewma_alphas=[0.3])               # all variable_cols
fg.add_window_features(rolling_window_sizes=[2], ewma_alphas=[0.3], cols=["sbp"]) # subset only
```

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

### 5. Irregularity Features

```python
fg.add_irregularity_features()                          # default window_size=5
fg.add_irregularity_features(acceleration_window_size=10)
```

#### Interval Features

Row-level features derived from inter-observation gaps per entity:

`interval_z_score` $= \dfrac{\Delta t - \mu}{\sigma}$

`interval_cv_local` $= \dfrac{\text{rolling\_std}_5(\Delta t)}{\text{rolling\_mean}_5(\Delta t)}$

| Output Column | Description |
|---------------|-------------|
| `interval_seconds` | Gap to the previous observation (null for first row per entity) |
| `interval_z_score` | Z-score of the gap relative to entity-level mean and std; null when σ = 0 |
| `interval_cv_local` | Rolling coefficient of variation (std/mean) over the last 5 intervals; captures local rhythm irregularity |

#### Windowed Acceleration Features

First-order differences of the interval sequence — how fast the sampling rhythm is changing:

`interval_acceleration`$_i = \Delta t_i - \Delta t_{i-1}$

Positive values mean gaps are growing (spacing out); negative values mean gaps are shrinking (bunching together).

| Output Column | Description |
|---------------|-------------|
| `interval_acceleration` | Raw Δ(interval); null for the first two observations per entity |
| `rolling_mean_acceleration_{n}` | Smoothed trend of acceleration over window `n` |
| `rolling_abs_acceleration_{n}` | Rolling mean of \|acceleration\| — magnitude of rhythm change |
| `rolling_std_acceleration_{n}` | Rolling std of acceleration — volatility of rhythm change |

---

### 6. Interaction Features

```python
fg.add_interaction_features()                       # all variable_cols
fg.add_interaction_features(cols=["hr", "sbp"])     # subset only
fg.add_row_imperfection_pct()                       # all variable_cols
fg.add_row_imperfection_pct(cols=["hr", "sbp"])     # subset only
```

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

All `cols` parameters accept a list of column names from `variable_cols`. When omitted, the full `variable_cols` list is used.

| Method | `cols` param | Features Added |
|--------|:------------:|----------------|
| `add_binary_masks(cols)` | ✓ | `{var}_mask` |
| `add_circular_features()` | — | `hour_sin`, `hour_cos` |
| `add_temporal_features(cols, lag, …)` | ✓ | `{var}_mask_lag_*`, `{var}_mask_consecutive`, `{var}_time_since_*` |
| `add_window_features(cols, rolling_window_sizes, ewma_alphas, …)` | ✓ | `{var}_mask_rolling_*`, `{var}_mask_ewma_*` |
| `add_interaction_features(cols)` | ✓ | `inter_*` pairwise features |
| `add_row_imperfection_pct(cols)` | ✓ | `row_imperfection_pct` |
| `add_irregularity_features(acceleration_window_size)` | — | `interval_seconds`, `interval_z_score`, `interval_cv_local`, `interval_acceleration`, `rolling_*_acceleration_*` |
| `generate_all_features(…)` | ✓ (per step) | All of the above |

### `generate_all_features` parameters

Parameters are prefixed by feature set so the call site is self-documenting.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `masks_cols` | `None` | Columns for binary mask features |
| `temporal_cols` | `None` | Columns for temporal features |
| `temporal_lag` | `1` | Lag size for lag-mask features |
| `temporal_lag_mask_replace_nulls_with_zero` | `True` | Replace nulls in lagged mask columns |
| `temporal_time_since_upper_bound` | `3600` | Cap in seconds for time-since features |
| `window_cols` | `None` | Columns for window features |
| `window_rolling_window_sizes` | `[2]` | Rolling window sizes |
| `window_ewma_alphas` | `[0.3, 0.5]` | EWMA smoothing factors |
| `window_replace_nulls_with_zero` | `True` | Replace nulls in window features |
| `interaction_cols` | `None` | Columns for pairwise interaction features |
| `row_imperfection_pct_cols` | `None` | Columns for row-level imperfection percentage |
| `irregularity_window_size` | `5` | Rolling window size for acceleration features |

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

# Generate all features with defaults
df_features = fg.generate_all_features()

# Customise per feature set
df_features = fg.generate_all_features(
    masks_cols=["hr", "sbp"],
    temporal_cols=["hr"],
    temporal_lag=2,
    temporal_time_since_upper_bound=7200,
    window_cols=["sbp"],
    window_rolling_window_sizes=[3, 5],
    window_ewma_alphas=[0.2, 0.5],
    interaction_cols=["hr", "sbp"],
)

# Or build up the pipeline step by step
df_features = (
    fg.add_binary_masks(cols=["hr"])
    .add_temporal_features(cols=["hr"], lag=2)
    .add_window_features(rolling_window_sizes=[2], ewma_alphas=[0.3], cols=["sbp"])
    .add_interaction_features(cols=["hr", "sbp"])
    .add_row_imperfection_pct()
    .df
)
```
