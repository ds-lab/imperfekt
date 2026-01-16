import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from imperfekt.analysis.utils import pretty_printing
from imperfekt.config.global_settings import VITALS


def create_lagged_features(df: pl.DataFrame, id_col: str, cols: list) -> pl.DataFrame:
    """
    Adds t-1 to each row
    Parameters:
       df (pl.DataFrame): Input DataFrame
       id_col (str): Column to group by (e.g., patient ID)
       cols (list): List of columns to lag
    Returns:
       pl.DataFrame: DataFrame with lagged features
    """
    lagged_df = df.with_columns(
        [pl.col(col).shift(1).over(id_col).alias(f"{col}_lag1") for col in cols]
    )
    return lagged_df


def compute_severity_indicators(df: pl.DataFrame, cols: list) -> pl.DataFrame:
    """
    Simple severity indicators:
    - Count of abnormal values per timepoint
    - Trend direction (improving/worsening)

    Parameters:
       df (pl.DataFrame): Input DataFrame
       cols (list): List of columns to analyze
    Returns:
       pl.DataFrame: DataFrame with severity indicators
    """
    # Define normal ranges (based on National Early Warning Score)
    normal_ranges = VITALS.NORMAL_RANGES_MAR_MNAR_TEST

    severity_df = df.with_columns(
        [
            # Count abnormal
            pl.sum_horizontal(
                [
                    (
                        (pl.col(col) < normal_ranges[col][0])
                        | (pl.col(col) > normal_ranges[col][1])
                    ).cast(pl.Int8)
                    for col in cols
                    if col in normal_ranges
                ]
            ).alias("abnormal_count"),
            # Simple trend: current vs previous
            pl.sum_horizontal(
                [
                    (pl.col(col) - pl.col(f"{col}_lag1") > 0).cast(pl.Int8)
                    for col in cols
                    if f"{col}_lag1" in df.columns
                ]
            ).alias("improving_count"),
        ]
    )

    return severity_df


def _impute(X, id_series, strategy: str):
    """
    Impute predictors.
    strategy in {'zero','mean','median','ffill_within_id'}.
    """
    if strategy == "zero":
        return X.fillna(0)
    if strategy == "mean":
        return X.fillna(X.mean(numeric_only=True))
    if strategy == "median":
        return X.fillna(X.median(numeric_only=True))
    if strategy == "ffill_within_id":
        tmp = X.copy()
        tmp[id_series.name] = id_series.values
        tmp = tmp.groupby(id_series.name, sort=False).ffill()
        # backstop any remaining NaNs (e.g., all-missing within an id)
        return tmp.fillna(X.mean(numeric_only=True))
    raise ValueError(
        f"impute_strategy must be one of zero|mean|median|ffill_within_id, got {strategy}"
    )


def _standardize(X):
    """
    Z-score standardize columns of X. Returns (X_std, means, stds).
    """
    means = X.mean(numeric_only=True)
    stds = X.std(ddof=0, numeric_only=True).replace(0, 1.0)
    Xs = (X - means) / stds
    return Xs, means, stds


def _safe_auc(y_true, y_prob):
    """Return ROC AUC or NaN if only one class is present."""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def temporal_mar_mnar_test(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    id_col: str = "id",
    clock_col: str = "clock",
    cols: list = None,
    alpha: float = 0.05,
    impute_strategy: str = "mean",
    standardize: bool = True,
    save_path: str = None,
    save_results: bool = False,
) -> pl.DataFrame:
    """
    Time-series aware MAR/MNAR test based on
    Parameters:
       df (pl.DataFrame): Input DataFrame
       id_col (str): Column to group by (e.g., patient ID)
       clock_col (str): Column representing time (e.g., timestamp)
       cols (list): List of columns to analyze
       alpha (float): Significance level for hypothesis testing
    Returns:
       polars.DataFrame: Results of the MAR/MNAR test
    """
    # make sure df and mask_df are sorted
    df = df.sort(by=[id_col, clock_col])
    mask_df = mask_df.sort(by=[id_col, clock_col])
    if cols is None:
        cols = [c for c in df.columns if c not in {id_col, clock_col}]
    # Step 1: Create lagged features
    df_with_lags = create_lagged_features(df, id_col, cols)

    for c in cols:
        plot_missingness_vs_lag_quantiles(
            df_with_lags,
            mask_df,
            var=c,
            id_col=id_col,
            save_path=save_path,
            save_results=save_results,
        )

    # Step 2: Add severity indicators
    df_enhanced = compute_severity_indicators(df_with_lags, cols)

    # Step 3: Convert to pandas for sklearn
    df_pd = df_enhanced.to_pandas()

    results = []

    for c1 in cols:
        # Skip if no missingness / nothing indicated
        if mask_df[c1].sum() == 0:
            continue

        # Imperfection indicator, y is the dependent variable
        dependent_variable = mask_df[c1].to_pandas()

        # MAR predictors: other columns + severity indicators
        mar_predictors = []
        for c2 in cols:
            if c2 != c1:
                mar_predictors.append(c2)
                if f"{c2}_lag1" in df_pd.columns:
                    mar_predictors.append(f"{c2}_lag1")

        mar_predictors.extend(["abnormal_count", "improving_count"])
        mnar_predictors = mar_predictors.copy()
        own_lag = f"{c1}_lag1"
        if own_lag in df_pd.columns:
            mnar_predictors.append(own_lag)

        # Get predictors
        X_mar = df_pd[mar_predictors].copy()
        X_mnar = df_pd[mnar_predictors].copy()

        # Impute
        X_mar = _impute(X_mar, df_pd[id_col], impute_strategy)
        X_mnar = _impute(X_mnar, df_pd[id_col], impute_strategy)

        # standardize (store SD of own_lag for back-transform)
        if standardize:
            means_union = X_mnar.mean(numeric_only=True)
            stds_union = X_mnar.std(ddof=0, numeric_only=True).replace(0, 1.0)

            # apply to MNAR (full set)
            X_mnar_std = (X_mnar - means_union) / stds_union
            # apply to MAR (subset uses aligned stats)
            X_mar_std = (X_mar - means_union[X_mar.columns]) / stds_union[X_mar.columns]

            # keep stds for back-transform of own_lag
            stds_mnar = stds_union
        else:
            X_mar_std, X_mnar_std = X_mar, X_mnar
            stds_mnar = None

        if len(dependent_variable) < 2:
            continue

        # Fit MAR model
        try:
            # Fit MAR
            mar_model = LogisticRegression(max_iter=1000, C=1e6)
            mar_model.fit(X_mar_std, dependent_variable)
            p_mar = mar_model.predict_proba(X_mar_std)[:, 1]
            ll_mar = _compute_log_likelihood(mar_model, X_mar_std, dependent_variable)
            auc_mar = _safe_auc(dependent_variable, p_mar)

            # Fit MNAR (MAR + own lag)
            mnar_model = LogisticRegression(max_iter=1000, C=1e6)
            mnar_model.fit(X_mnar_std, dependent_variable)
            p_mnar = mnar_model.predict_proba(X_mnar_std)[:, 1]
            ll_mnar = _compute_log_likelihood(mnar_model, X_mnar_std, dependent_variable)
            auc_mnar = _safe_auc(dependent_variable, p_mnar)

            # LRT (1 extra parameter: own lag)
            lrt = 2 * (ll_mnar - ll_mar)
            p_value = 1 - chi2.cdf(lrt, df=1)

            # Extract coefficient for own lag from MNAR model
            coef_map = dict(zip(X_mnar_std.columns, mnar_model.coef_[0]))
            coef_std = float(coef_map.get(own_lag, float("nan")))

            # Back-transform to raw units if standardized
            if standardize and stds_mnar is not None and own_lag in stds_mnar:
                s = float(stds_mnar[own_lag])
                coef_lag1_raw = float(coef_std / s) if s > 0 else float("nan")
            else:
                coef_lag1_raw = coef_std

            results.append(
                {
                    "column": c1,
                    "lrt_statistic": float(lrt),
                    "p_value": float(p_value),
                    "decision": "Likely MNAR"
                    if p_value < alpha
                    else "No strong evidence against MAR",
                    "sample_size": int(len(dependent_variable)),
                    "missing_count": int(dependent_variable.sum()),
                    "coef_lag1_mnar": coef_lag1_raw,  # positive = higher value at lag1 means higher missingness probability
                    "auc_mar": float(auc_mar),
                    "auc_mnar": float(auc_mnar),
                    "loglik_mar": float(ll_mar),  # higher is better
                    "loglik_mnar": float(ll_mnar),
                }
            )

        except Exception as e:
            results.append({"column": c1, "error": str(e)})

    if len(results) == 0:
        pretty_printing.rich_warning("No results for MAR-MNAR testing.")
        return {"temporal_mar_mnar_results": []}

    return pl.DataFrame(results)


def _compute_log_likelihood(model, X, y):
    """Helper to compute log-likelihood"""
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def plot_missingness_vs_lag_quantiles(
    df: pl.DataFrame,
    mask_df: pl.DataFrame,
    var: str,
    id_col="id",
    q=10,
    save_path: str = None,
    save_results: bool = True,
):
    # convert polars -> pandas if needed
    if hasattr(df, "to_pandas"):
        df = df.sort([id_col, "clock"]).to_pandas()
    if hasattr(mask_df, "to_pandas"):
        mask_df = mask_df.sort([id_col, "clock"]).to_pandas()

    lag_col = f"{var}_lag1"
    if lag_col not in df.columns:
        raise ValueError(f"{lag_col} not found. Create lag first.")

    # build a working frame
    work = pd.DataFrame(
        {id_col: df[id_col], "lag": df[lag_col], "missing": mask_df[var].astype(int)}
    ).dropna(subset=["lag"])  # only rows with a valid lag

    # global quantile bins
    work["lag_q"] = pd.qcut(work["lag"], q=q, duplicates="drop")

    # overall aggregation
    overall = (
        work.groupby("lag_q")
        .agg(missing_rate=("missing", "mean"), n=("missing", "size"))
        .reset_index()
    )
    # approximate 95% CI (normal approx)
    z = 1.96
    overall["se"] = np.sqrt(overall["missing_rate"] * (1 - overall["missing_rate"]) / overall["n"])
    overall["ci_low"] = (overall["missing_rate"] - z * overall["se"]).clip(0, 1)
    overall["ci_high"] = (overall["missing_rate"] + z * overall["se"]).clip(0, 1)
    # use bin centers for plotting
    overall["bin_center"] = overall["lag_q"].apply(lambda r: r.mid)

    # plot overall
    fig = plt.figure(figsize=(7, 4))
    plt.errorbar(
        overall["bin_center"],
        overall["missing_rate"],
        yerr=[
            overall["missing_rate"] - overall["ci_low"],
            overall["ci_high"] - overall["missing_rate"],
        ],
        fmt="o-",
        capsize=3,
    )
    plt.xlabel(f"{var} (t-1) — quantile bin centers")
    plt.ylabel("P(missing at t)")
    plt.title(f"Missingness vs {var}_lag1 (overall, q={q})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    if save_results and save_path is not None:
        fig.savefig(f"{save_path}/missingness_vs_{var}_lag1.png")


if __name__ == "__main__":
    # Example usage
    df = pl.DataFrame(
        {
            "id": ["a"] * 10 + ["b"] * 10,
            "clock": [f"2023-10-01 12:{i:02d}:00" for i in range(10)] * 2,
            "clock_no": list(range(1, 11)) * 2,
            "heartrate": [1.0, 2.0, None, 4.0, 5.0, 6.0, None, 8.0, 9.0, 10.0] * 2,
            "resprate": [100.0, None, 10.0, 10.0, 140.0, None, 10.0, 10.0, 10.0, 10.0] * 2,
            "sbp": [20.0, 221.0, 22.0, None, 24.0, 252.0, 26.0, None, 28.0, 29.0] * 2,
            "o2sat": [30.0, 31.0, 32.0, None, 34.0, 35.0, 36.0, None, 38.0, 39.0] * 2,
        }
    ).with_columns(
        [
            pl.col("clock").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )

    mask_df = df.with_columns(
        [
            pl.col("heartrate").is_null().cast(pl.Int8).alias("heartrate"),
            pl.col("resprate").is_null().cast(pl.Int8).alias("resprate"),
            pl.col("sbp").is_null().cast(pl.Int8).alias("sbp"),
            pl.col("o2sat").is_null().cast(pl.Int8).alias("o2sat"),
        ]
    )

    # Run the MAR/MNAR test
    results = temporal_mar_mnar_test(df, mask_df, cols=["heartrate", "resprate", "sbp", "o2sat"])
    print(results)
