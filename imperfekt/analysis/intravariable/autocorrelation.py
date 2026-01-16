from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.tsa.seasonal import STL


def acf(
    mask_df: pl.DataFrame,
    col: str,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    max_lag: int = 20,
    seasonal_trend_decomposition: bool = False,
    stl_period: int = 7,
    save_path: str = None,
    save_results: bool = True,
):
    """
    Compute lagged autocorrelation estimate of the imperfection indicator for a variable.
    Only uses positive lags (lag>0), because autocorrelation is symmetric.
    Ressource: https://en.wikipedia.org/wiki/Autocorrelation#:~:text=an%20estimate%20of%20the%20autocorrelation%20coefficient%20may%20be%20obtained%20as
    Example:
    If col is imperfect at time t, the autocorrelation at lag 1 will be positive if col is likely to also be imperfect at time t+1.

    Parameters:
        mask_df (pl.DataFrame): DataFrame with a mask indicating imperfect values (1=missing/noisy/indicated, 0=observed/normal).
        col (str): Column to analyze.
        id_col (str): ID column.
        clock_no_col (str): Time ordering column.
        max_lag (int): Maximum lag to compute.
        seasonal_trend_decomposition (bool): Whether to perform seasonal trend decomposition (default: False).
        stl_period (int): Seasonal period for decomposition (default: 7).
        save_path (str): Path to save the results CSV and STL plot (optional).
        save_results (bool): Whether to save the results (default: True).

    Returns:
        autocorrs (pl.DataFrame): DataFrame with columns 'lag' and 'autocorr' containing the autocorrelation values for each lag.
    """
    # Create a lazy frame with the imperfection indicator, sorted correctly.
    mask_df = mask_df.sort([id_col, clock_no_col])

    # Get the overall mean and var of m
    stats = mask_df.select([pl.mean(col).alias("mean"), pl.var(col).alias("var")])
    mean_m = stats["mean"][0]
    var_m = stats["var"][0]

    if var_m == 0:
        raise ValueError("Constant series. Autocorrelation is undefined.")

    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    elif max_lag > mask_df.height:
        max_lag = mask_df.height - 1

    # For each lag, shift within the group and mask boundaries
    lags = np.arange(1, max_lag + 1)
    acfs = []
    for lag in lags:
        shifted = mask_df.with_columns(
            pl.col(col)
            .shift(lag)
            .over(id_col)  # shift within each group
            .alias("m_lag")
        )

        # Drop rows where m_lag is null and compute the numerator
        numer = (
            shifted.drop_nulls("m_lag")
            .select((pl.col(col) - mean_m) * (pl.col("m_lag") - mean_m))
            .sum()
        ).row(0)[0]

        # Calculate the denominator (var_m * (T-k), where T-k is the number of valid lag-k pairs)
        n_pairs = (shifted.filter(pl.col("m_lag").is_not_null()).select(pl.len())).row(0)[0]

        if n_pairs == 0:
            break

        denom = var_m * n_pairs
        acfs.append(numer / denom)

    # Cut lags to fit the length of acfs
    lags = lags[: len(acfs)]
    acf_df = pl.DataFrame({"lag": lags, "autocorr": acfs})

    if seasonal_trend_decomposition:
        print(acfs)
        stl = STL(acf_df["autocorr"].to_numpy(), period=stl_period, robust=True)
        res = stl.fit()
        fig = res.plot()

        if save_path and save_results:
            fig.savefig(Path(save_path) / "stl_plot.png")
            print(f"STL plot saved to {Path(save_path) / 'stl_plot.png'}")

    if save_results and save_path:
        acf_df.write_csv(Path(save_path) / "autocorrelation.csv")
        print(f"Autocorrelation results saved to {Path(save_path) / 'autocorrelation.csv'}")
    return acf_df


if __name__ == "__main__":
    # Example usage
    data = {
        "id": [
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
        ],
        "clock_no": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        "vital": [
            None,
            2.0,
            None,
            4.0,
            None,
            4.0,
            None,
            5.0,
            None,
            2.0,
            None,
            4.0,
            None,
            4.0,
            None,
            5.0,
        ],
    }
    df = pl.DataFrame(data)
    mask_df = df.with_columns(vital=pl.col("vital").is_null().cast(pl.Int8).alias("vital"))

    autocorrs = acf(mask_df, col="vital")
    print(autocorrs)
