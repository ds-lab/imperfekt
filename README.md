# Imperfekt - Understanding Data Imperfections in Time-Series

[![PyPI version](https://img.shields.io/pypi/v/imperfekt.svg)](https://pypi.org/project/imperfekt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive analysis toolkit for studying "imperfect" data patterns in time-series datasets.
Imperfection refers to missingness, implausible values, irregular sampling, and other data quality issues that can be indicated using a binary mask.

## Overview

This library provides tools to analyze data quality issues in time-series data, including:
- **Preliminary analysis** of the observed values (description, normality, correlation)
- **Irregularity analysis** of the observation time grid (intervals, burstiness, dominant frequency)
- **Intravariable analysis** of imperfection patterns for individual variables
- **Intervariable analysis** of co-occurring imperfections across multiple parameters
- **Group- and event-based analysis** to compare imperfection between cohorts or around events
- **Case-level metrics and stratification** to rank and bucket individual cases by imperfection
- **Feature generation** based on imperfection patterns for downstream ML tasks

Two imperfection types are supported: `"missingness"` (nulls) and `"plausibility"` (values flagged as
implausible via IQR/MAD bounds or explicit reference ranges).

## Installation

Install the library using `pip`:

```bash
pip install imperfekt
```

> Note:
> Plots are drawn with `matplotlib` by default. If you switch to Plotly (`plot_library="plotly"`) and export
> figures as static images (`save_results=True`), some environments may raise a `plotly_get_chrome`/Kaleido
> error at runtime. This happens because Kaleido needs a Chrome/Chromium binary.
> Install Chrome manually, or run:
>
> ```bash
> plotly_get_chrome
> ```

## Quick Start

```python
import polars as pl
from imperfekt import Imperfekt, FeatureGenerator

# Load your time-series data
df = pl.read_parquet("your_data.parquet")

# Configure Analyzer Setup
analyzer = Imperfekt(
    df=df,
    id_col="id",  # Unique identifier column
    clock_col="clock",  # Timestamp column
    cols=["var1", "var2"],  # Variables to analyze
    save_path="./results",
    imperfection="missingness",  # or "plausibility"
)

# Simple intravariable imperfection stats
analyzer.intravariable.column_statistics(save_results=True)
print(analyzer.intravariable.results.cs_overall_statistics)
print(analyzer.intravariable.results.cs_case_level_statistics)

# Run full imperfection analysis (preliminary, irregularity, intra- and intervariable analyses)
analyzer.run()  # cheap_mode=True for a faster, reduced set of analyses

# Compare imperfection between groups/labels, or around events
analyzer.run_grouped_analysis(annotation_col="age")
analyzer.run_event_based_analysis(events_df=events_df, window_size=300)

# Or generate imperfection-aware features for ML
fg = FeatureGenerator(df=df, id_col="id", clock_col="clock", variable_cols=["var1", "var2"])
features_df = fg.add_binary_masks().add_temporal_features().df

# Or restrict individual steps to a subset of variables
features_df = (
    fg.add_binary_masks(cols=["var1"])
    .add_temporal_features(cols=["var1"])
    .add_window_features(rolling_window_sizes=[2], ewma_alphas=[0.3], cols=["var1", "var2"])
    .df
)

# Or generate everything at once
features_df = fg.generate_all_features()
```

Runnable end-to-end scripts live in [examples/scripts/](examples/scripts/).

## Library Structure

```
imperfekt/
├── analysis/
│   ├── preliminary/     # Basic data exploration
│   ├── irregularity/    # Observation-grid irregularity
│   ├── intravariable/   # Single variable analysis
│   ├── intervariable/   # Multi-variable patterns
│   └── utils/           # Shared utilities (masking, statistics, stratification, plotting)
├── features/            # Feature engineering
│   ├── core.py          # FeatureGenerator class
│   ├── temporal.py      # Time-based features
│   ├── window.py        # Rolling and EWMA features
│   ├── irregularity.py  # Sampling-rhythm features
│   └── interaction.py   # Variable interactions
└── config/              # Default settings
```

Module-level documentation:  [analysis/intravariable/README.md](imperfekt/analysis/intravariable/README.md),  [analysis/intervariable/README.md](imperfekt/analysis/intervariable/README.md), [analysis/irregularity/README.md](imperfekt/analysis/irregularity/README.md), [features/README.md](imperfekt/features/README.md).

## Data Format

The library expects time-series data with the following structure:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each time-series (e.g., patient, sensor) |
| `clock` | Timestamp for each observation (optional) |
| `var1`, `var2`, ... | Variables to analyze |

## Key Dependencies

- **polars**: High-performance data processing
- **matplotlib** / **plotly**: Static and interactive visualizations
- **statsmodels**, **pingouin**, **scikit-posthocs**: Statistical computations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## AI Disclaimer

We used AI (Google Gemini 3.1 Pro, Anthropic Claude Opus 5) during the development of this repository.
All AI-generated output was reviewed by the authors, who take full responsibility for the code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
