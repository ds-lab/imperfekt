# in imperfekt/features/core.py
import numpy as np
import polars as pl

from imperfekt.features import interaction, temporal, window


class FeatureGenerator:
    def __init__(
        self,
        df: pl.DataFrame,
        id_col: str = "id",
        clock_col: str = "clock",
        clock_no_col: str = "clock_no",
        variable_cols: list = None,
        imperfection: str = "missingness",
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
        self._generate_mask()

    def _generate_clock_no_col(self):
        """Generates a clock number column based on ordered timestamps (clock_col)."""
        self.df = self.df.sort([self.id_col, self.clock_col])
        if self.clock_no_col not in self.df.columns:
            self.df = self.df.with_columns(
                pl.cum_count(self.id_col).over(self.id_col).alias(self.clock_no_col)
            )

    def _generate_mask(self):
        """
        Generates a binary mask DataFrame based on the imperfection type.
        The mask includes the id and clock columns and is stored in self.mask.
        """
        if self.imperfection == "missingness":
            # Create a new DataFrame that contains the identifiers and the mask columns
            self.mask = self.df.select(
                self.id_col,
                self.clock_col,
                self.clock_no_col,
                pl.col(self.variable_cols).is_null().cast(pl.Int32).name.suffix("_mask"),
            )
        # In the future, other imperfection definitions can be defined here
        # elif self.imperfection == "unplausible_values":
        #     # e.g., self.mask = self.df.select(...)
        else:
            raise ValueError(f"Unknown imperfection type: {self.imperfection}")

    def add_binary_masks(self):
        """
        Joins the pre-generated binary mask columns to the main DataFrame.
        """
        if self.mask is not None:
            # Drop mask columns if they already exist to avoid duplicates
            mask_cols = [c for c in self.mask.columns if c not in [self.id_col, self.clock_col]]
            self.df = self.df.drop(mask_cols, strict=False)
            self.df = self.df.join(self.mask, on=[self.id_col, self.clock_col], how="inner")
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
    ):
        """
        Adds features like lags, consecutive counts, and time-since.

        Parameters:
            lag: The lag to apply. Default is 1.
            lag_mask_replace_nulls_with_zero: Whether to replace nulls with zero in the lagged columns. If True, we assume that time before the first observation was not imperfect.
            time_since_upper_bound: Optional upper bound for time since features in seconds. Default is 3600 (1 hour).
        """
        # This method can call functions from your temporal.py module
        self.df = temporal.add_lag_mask(
            self.df,
            self.mask,
            self.variable_cols,
            self.id_col,
            self.clock_col,
            lag=lag,
            replace_nulls_with_zero=lag_mask_replace_nulls_with_zero,
        )
        self.df = temporal.add_consecutive_counts(
            self.df, self.mask, self.variable_cols, self.id_col, self.clock_col
        )
        self.df = temporal.add_time_since(
            self.df,
            self.mask,
            self.variable_cols,
            self.id_col,
            self.clock_col,
            cap_seconds=time_since_upper_bound,  # e.g. cap at 1h
        )
        return self

    def add_window_features(
        self,
        rolling_window_sizes: list,
        ewma_alphas: float,
        replace_nulls_with_zero: bool = True,
    ):
        """
        Adds rolling window statistics (of imperfect timestamps per variable):
        - Rolling count
        - Rolling variance
        - Exponential moving average

        Parameters:
            rolling_window_sizes(list of int): The sizes of the rolling windows to apply.
            ewma_spans (list of int): The spans for the exponential moving average.
            replace_nulls_with_zero (bool): Whether to replace nulls with zero in the rolling window features.
        """
        for window_size in rolling_window_sizes:
            self.df = window.add_rolling_window_features(
                self.df,
                self.mask,
                self.variable_cols,
                self.id_col,
                self.clock_col,
                window_size=window_size,
                replace_nulls_with_zero=replace_nulls_with_zero,
            )
        for alpha in ewma_alphas:
            self.df = window.add_exponential_moving_average(
                self.df,
                self.mask,
                self.variable_cols,
                self.id_col,
                self.clock_col,
                alpha=alpha,
            )
        return self

    def add_interaction_features(self):
        """
        Adds pairwisecross-variable interaction features.

        Generates four types of interactions for each pair of variables (A, B):
        1. Concurrent value: var_a_t * mask_b_t
        2. Concurrent mask: mask_a_t * mask_b_t
        3. Predictive value: var_a_t-1 * mask_b_t
        4. Predictive mask: mask_a_t-1 * mask_b_t
        Results in 4*N*(N-1) new features.

        """
        self.df = interaction.add_pairwise_interactions(
            self.df, self.mask, self.variable_cols, self.id_col, self.clock_col
        )
        return self

    def add_row_imperfection_pct(self):
        """
        Adds the percentage of imperfect (missing) values for each row.
        """
        self.df = interaction.add_row_level_features(
            self.df, self.mask, self.variable_cols, self.id_col, self.clock_col
        )
        return self

    def generate_all_features(self):
        """A convenience method to run all feature generation steps."""
        self.add_binary_masks()
        self.add_circular_features()
        self.add_temporal_features()
        self.add_window_features(rolling_window_sizes=[2], ewma_alphas=[0.3, 0.5])
        self.add_interaction_features()
        self.add_row_imperfection_pct()
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
