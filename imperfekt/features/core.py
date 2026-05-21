# in imperfekt/features/core.py
import numpy as np
import polars as pl

from imperfekt.analysis.utils import masking, pretty_printing
from imperfekt.features import interaction, irregularity, temporal, window


class FeatureGenerator:
    def __init__(
        self,
        df: pl.DataFrame,
        id_col: str = "id",
        clock_col: str = "clock",
        clock_no_col: str = "clock_no",
        variable_cols: list | None = None,
        imperfection: str = "missingness",
        plausibility_method: str | None = "iqr",
        plausibility_threshold: float = 1.5,
        plausibility_scope: str = "global",
        plausibility_missing_as: str = "ignore",
        plausibility_reference_ranges: dict | None = None,
    ):
        self.imperfection = imperfection
        self.df = df
        self.id_col = id_col
        self.clock_col = clock_col
        self.variable_cols = variable_cols or [
            c for c in self.df.columns if c not in {id_col, clock_col, clock_no_col}
        ]
        self.clock_no_col = clock_no_col
        self._generate_clock_no_col()
        self._generate_mask(
            plausibility_method=plausibility_method,
            plausibility_threshold=plausibility_threshold,
            plausibility_scope=plausibility_scope,
            plausibility_missing_as=plausibility_missing_as,
            plausibility_reference_ranges=plausibility_reference_ranges,
        )

    def _generate_clock_no_col(self):
        """Generates a clock number column based on ordered timestamps (clock_col)."""
        self.df = self.df.sort([self.id_col, self.clock_col])
        if self.clock_no_col not in self.df.columns:
            self.df = self.df.with_columns(
                pl.cum_count(self.id_col).over(self.id_col).alias(self.clock_no_col)
            )

    def _generate_mask(
        self,
        plausibility_method: str | None = "iqr",
        plausibility_threshold: float = 1.5,
        plausibility_scope: str = "global",
        plausibility_missing_as: str = "ignore",
        plausibility_reference_ranges: dict | None = None,
    ):
        """
        Generates a binary mask DataFrame based on the imperfection type.
        The mask includes the id and clock columns and is stored in self.mask.
        """
        if self.imperfection == "missingness":
            self.mask = masking.create_missingness_mask(
                df=self.df,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                cols=self.variable_cols,
            )
        elif self.imperfection == "plausibility":
            self.mask = masking.create_plausibility_mask(
                df=self.df,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                cols=self.variable_cols,
                method=plausibility_method,
                threshold=plausibility_threshold,
                scope=plausibility_scope,
                missing_as=plausibility_missing_as,
                reference_ranges=plausibility_reference_ranges,
            )
        else:
            raise ValueError(f"Unknown imperfection type: {self.imperfection}")

    def add_binary_masks(self, cols: list | None = None):
        """
        Joins the pre-generated binary mask columns to the main DataFrame.

        Parameters:
            cols: Subset of variable_cols to include. Defaults to all variable_cols.
        """
        cols = cols or self.variable_cols
        if self.mask is not None:
            mask_subset = self.mask.select(
                [self.id_col, self.clock_col, self.clock_no_col]
                + [f"{c}_mask" for c in cols if f"{c}_mask" in self.mask.columns]
            )
            mask_cols = [c for c in mask_subset.columns if c not in [self.id_col, self.clock_col]]
            self.df = self.df.drop(mask_cols, strict=False)
            self.df = self.df.join(mask_subset, on=[self.id_col, self.clock_col], how="inner")
        return self

    def add_circular_features(self):
        """
        Adds circular features for time-based columns like hour of the day.
        This uses sine and cosine transformations to represent cyclical data smoothly.
        """
        hour = pl.col(self.clock_col).dt.hour()

        self.df = self.df.with_columns(
            (2 * np.pi * hour / 24).sin().alias("hour_sin"),
            (2 * np.pi * hour / 24).cos().alias("hour_cos"),
        )
        return self

    def add_temporal_features(
        self,
        lag: int = 1,
        lag_mask_replace_nulls_with_zero: bool = True,
        time_since_upper_bound: int = 3600,
        cols: list | None = None,
    ):
        """
        Adds features like lags, consecutive counts, and time-since.

        Parameters:
            lag: The lag to apply. Default is 1.
            lag_mask_replace_nulls_with_zero: Whether to replace nulls with zero in the lagged columns. If True, we assume that time before the first observation was not imperfect.
            time_since_upper_bound: Optional upper bound for time since features in seconds. Default is 3600 (1 hour).
            cols: Subset of variable_cols to include. Defaults to all variable_cols.
        """
        cols = cols or self.variable_cols
        self.df = temporal.add_lag_mask(
            self.df,
            self.mask,
            cols,
            self.id_col,
            self.clock_col,
            lag=lag,
            replace_nulls_with_zero=lag_mask_replace_nulls_with_zero,
        )
        self.df = temporal.add_consecutive_counts(
            self.df, self.mask, cols, self.id_col, self.clock_col
        )
        self.df = temporal.add_time_since(
            self.df,
            self.mask,
            cols,
            self.id_col,
            self.clock_col,
            cap_seconds=time_since_upper_bound,  # e.g. cap at 1h
        )
        return self

    def add_window_features(
        self,
        rolling_window_sizes: list | None = None,
        ewma_alphas: list | None = None,
        replace_nulls_with_zero: bool = True,
        cols: list | None = None,
    ):
        """
        Adds rolling window statistics (of imperfect timestamps per variable):
        - Rolling count
        - Rolling variance
        - Exponential moving average

        Parameters:
            rolling_window_sizes(list of int): The sizes of the rolling windows to apply.
            ewma_alphas (list of float): The alphas for the exponential moving average.
            replace_nulls_with_zero (bool): Whether to replace nulls with zero in the rolling window features.
            cols: Subset of variable_cols to include. Defaults to all variable_cols.
        """
        cols = cols or self.variable_cols
        rolling_window_sizes = rolling_window_sizes or [5]
        ewma_alphas = ewma_alphas or [0.3]
        for window_size in rolling_window_sizes:
            self.df = window.add_rolling_window_features(
                self.df,
                self.mask,
                cols,
                self.id_col,
                self.clock_col,
                window_size=window_size,
                replace_nulls_with_zero=replace_nulls_with_zero,
            )
        for alpha in ewma_alphas:
            self.df = window.add_exponential_moving_average(
                self.df,
                self.mask,
                cols,
                self.id_col,
                self.clock_col,
                alpha=alpha,
            )
        return self

    def add_interaction_features(self, cols: list | None = None):
        """
        Adds pairwise cross-variable interaction features.

        Generates four types of interactions for each pair of variables (A, B):
        1. Concurrent value: var_a_t * mask_b_t
        2. Concurrent mask: mask_a_t * mask_b_t
        3. Predictive value: var_a_t-1 * mask_b_t
        4. Predictive mask: mask_a_t-1 * mask_b_t
        Results in 4*N*(N-1) new features.

        Parameters:
            cols: Subset of variable_cols to include. Defaults to all variable_cols.
        """
        cols = cols or self.variable_cols
        # if cols len is 1 or less, skip interaction features, pretty print a warning
        if len(cols) <= 1:
            pretty_cols = ", ".join(cols)
            pretty_printing.rich_warning(
                "⚠️ Not enough columns for interaction features. Skipping interaction feature generation. Columns provided: "
                f"{pretty_cols}"
            )
            return self
        self.df = interaction.add_pairwise_interactions(
            self.df, self.mask, cols, self.id_col, self.clock_col
        )
        return self

    def add_irregularity_features(self, acceleration_window_size: int = 5):
        """
        Adds all irregularity features derived from inter-observation intervals.

        Interval features:
            - ``interval_seconds``: gap to the previous observation (null for first row per entity).
            - ``interval_z_score``: z-score of the gap relative to the entity's own mean/std.
            - ``interval_cv_local``: rolling CV (std/mean) of the last 5 intervals per entity.

        Windowed acceleration features:
            - ``interval_acceleration``: Δ(interval) = interval_i − interval_{i-1}.
            - ``rolling_mean_acceleration_{n}``: smoothed trend — are gaps steadily growing or shrinking?
            - ``rolling_abs_acceleration_{n}``: rolling mean of |acceleration| — magnitude of rhythm change.
            - ``rolling_std_acceleration_{n}``: rolling std of acceleration — volatility of rhythm change.

        Parameters:
            acceleration_window_size: Rolling window size for acceleration statistics. Default 5.
        """
        self.df = irregularity.add_interval_features(self.df, self.id_col, self.clock_col)
        self.df = irregularity.add_windowed_acceleration(
            self.df, self.id_col, self.clock_col, window_size=acceleration_window_size
        )
        return self

    def add_row_imperfection_pct(self, cols: list | None = None):
        """
        Adds the percentage of imperfect (missing) values for each row.

        Parameters:
            cols: Subset of variable_cols to include. Defaults to all variable_cols.
        """
        cols = cols or self.variable_cols
        self.df = interaction.add_row_level_features(
            self.df, self.mask, cols, self.id_col, self.clock_col
        )
        return self

    def generate_all_features(
        self,
        # binary masks
        masks_cols: list | None = None,
        # temporal features
        temporal_cols: list | None = None,
        temporal_lag: int = 1,
        temporal_lag_mask_replace_nulls_with_zero: bool = True,
        temporal_time_since_upper_bound: int = 3600,
        # window features
        window_cols: list | None = None,
        window_rolling_window_sizes: list | None = None,
        window_ewma_alphas: list | None = None,
        window_replace_nulls_with_zero: bool = True,
        # interaction features
        interaction_cols: list | None = None,
        # row imperfection percentage
        row_imperfection_pct_cols: list | None = None,
        # irregularity features
        irregularity_window_size: int = 5,
    ):
        """
        Convenience method to run all feature generation steps.

        Parameters are grouped by feature set. For each step that accepts a `cols`
        argument, omitting it (or passing ``None``) falls back to all ``variable_cols``.

        Parameters:
            masks_cols: Columns for binary mask features.
            temporal_cols: Columns for temporal features (lag, consecutive count, time-since).
            temporal_lag: Lag size for lag-mask features. Default 1.
            temporal_lag_mask_replace_nulls_with_zero: Replace nulls with zero in lagged mask columns. Default True.
            temporal_time_since_upper_bound: Cap (seconds) for time-since features. Default 3600.
            window_cols: Columns for window features (rolling statistics, EWMA).
            window_rolling_window_sizes: List of rolling window sizes. Default [2].
            window_ewma_alphas: List of EWMA smoothing factors. Default [0.3, 0.5].
            window_replace_nulls_with_zero: Replace nulls with zero in window features. Default True.
            interaction_cols: Columns for pairwise interaction features.
            row_imperfection_pct_cols: Columns used to compute the row-level imperfection percentage.
            irregularity_window_size: Rolling window size for acceleration features. Default 5.
        """
        if window_rolling_window_sizes is None:
            window_rolling_window_sizes = [2]
        if window_ewma_alphas is None:
            window_ewma_alphas = [0.3, 0.5]

        self.add_binary_masks(cols=masks_cols)
        self.add_circular_features()
        self.add_temporal_features(
            cols=temporal_cols,
            lag=temporal_lag,
            lag_mask_replace_nulls_with_zero=temporal_lag_mask_replace_nulls_with_zero,
            time_since_upper_bound=temporal_time_since_upper_bound,
        )
        self.add_window_features(
            cols=window_cols,
            rolling_window_sizes=window_rolling_window_sizes,
            ewma_alphas=window_ewma_alphas,
            replace_nulls_with_zero=window_replace_nulls_with_zero,
        )
        self.add_interaction_features(cols=interaction_cols)
        self.add_row_imperfection_pct(cols=row_imperfection_pct_cols)
        self.add_irregularity_features(acceleration_window_size=irregularity_window_size)
        return self.df


if __name__ == "__main__":
    # Example usage
    # pl.Config.set_tbl_cols(34)
    df = pl.DataFrame(
        {
            "patient": ["a", "a", "a", "a", "a", "c", "c"],
            "time": [
                "2023-01-01 00:00:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:10:00",
                "2023-01-01 00:15:00",
                "2023-01-01 00:20:00",
                "2023-02-02 03:55:00",
                "2023-02-01 04:00:00",
            ],
            "value1": [1, None, None, 4, 5, 6, 7],
            "value2": [None, 1, None, 3, None, None, None],
        }
    ).with_columns(
        [
            pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )
    fg = FeatureGenerator(
        df, id_col="patient", clock_col="time", variable_cols=["value1", "value2"]
    )
    print(fg.mask)
    df_features = fg.generate_all_features()  # -> 30 new features
    print(df_features)
