# Analysis Module

This module provides statistical analysis tools for characterizing imperfection (missingness and noise) in time-series data.

## Structure

| Submodule | Purpose |
|-----------|---------|
| `preliminary/` | Descriptive statistics, normality tests, correlation, and autocorrelation |
| `intravariable/` | Within-column analysis: gap patterns, Markov chains, windowed significance |
| `intervariable/` | Between-column analysis: MCAR tests, MAR/MNAR detection, symmetric/asymmetric correlation |
| `utils/` | Shared utilities for statistics, visualization, and HTML reporting |

## Usage

```python
from imperfekt.analysis import Imperfekt

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

analyzer = Imperfekt(
    df=df,
    id_col="patient",
    clock_col="time",
    clock_no_col="time_no",
    save_path="/path",
    plot_library="matplotlib",
    renderer="notebook_connected",
)
# Run all analyses
analyzer.run()

# Run preliminary analysis only
analyzer.preliminary.run()
```

## Detailed Documentation

Each submodule contains its own README with further details:

- [Preliminary Analysis](preliminary/README.md)
- [Intravariable Analysis](intravariable/README.md)
- [Intervariable Analysis](intervariable/README.md)

### Overview Figure
![Imperfekt Analysis Matrix](imperfekt.png "Imperfekt Analysis Matrix")


