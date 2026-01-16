# Imperfekt - Understanding Data Imperfections in Time-Series

[![PyPI version](https://img.shields.io/pypi/v/imperfekt.svg)](https://pypi.org/project/imperfekt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive analysis toolkit for studying "imperfect" data patterns in time-series datasets.
Imperfection refers to missingness, noise, and other data quality issues that can be indicated using a binary mask.

## Overview

This library provides tools to analyze data quality issues in time-series data, including:
- **Intravariable analysis** of imperfection patterns for individual variables
- **Intervariable analysis** of co-occurring imperfections across multiple parameters
- **Feature generation** based on missingness patterns for downstream ML tasks

## Installation

Install the library using `pip`:

```bash
pip install imperfekt
```

## Quick Start

```python
import polars as pl
from imperfekt import Imperfekt, FeatureGenerator

# Load your time-series data
df = pl.read_parquet("your_data.parquet")

# Run simple

# Configure Analyzer Setup
analyzer = Imperfekt(
    df=df,
    id_col="id",           # Unique identifier column
    clock_col="clock",     # Timestamp column
    cols=["var1", "var2"], # Variables to analyze
    save_path="./results"
)

# Simple intravariable missingness stats
analyzer.intravariable.column_statistics(save_results=True)
print(analyzer.intravariable.results.cs_overall_statistics)
print(analyzer.intravariable.results.cs_case_level_statistics)

# Run full imperfection analysis (preliminary correlations, intra- and intervariable analyses)
results = analyzer.run()

# Or generate missingness-aware features for ML
fg = FeatureGenerator(
    df=df,
    id_col="id",
    clock_col="clock",
    variable_cols=["var1", "var2"]
)
features_df = fg.add_binary_masks().add_temporal_features().df
```

## Library Structure

```
imperfekt/
├── analysis/
│   ├── preliminary/     # Basic data exploration
│   ├── intravariable/      # Single variable analysis
│   ├── intervariable/    # Multi-variable patterns
│   └── utils/           # Shared utilities
├── features/            # Feature engineering
│   ├── core.py          # FeatureGenerator class
│   ├── temporal.py      # Time-based features
│   └── interaction.py   # Variable interactions
└── config/              # Default settings
```

## Data Format

The library expects time-series data with the following structure:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each time-series (e.g., patient, sensor) |
| `clock` | Timestamp for each observation |
| `var1`, `var2`, ... | Variables to analyze |

## Key Dependencies

- **polars**: High-performance data processing
- **plotly**: Interactive visualizations
- **scipy**: Statistical computations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

