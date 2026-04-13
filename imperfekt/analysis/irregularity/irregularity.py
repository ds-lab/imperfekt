import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from imperfekt.analysis.intravariable import autocorrelation
from imperfekt.analysis.irregularity import burstiness as burstiness_module
from imperfekt.analysis.irregularity import interval_statistics as interval_statistics_module
from imperfekt.analysis.utils import pretty_printing, visualization_utils


class IrregularityPlots:
    def __init__(self):
        self.ins_cv_violin = None
        self.domf_interval_frequency_bar = None
        self.bu_burstiness_violin = None
        self.ia_acf_plot = None


class IrregularityResults:
    def __init__(self):
        # Interval statistics
        self.ins_entity_statistics: pl.DataFrame = None
        self.ins_global_statistics: pl.DataFrame = None
        # Dominant frequency
        self.domf_frequency_summary: pl.DataFrame = None
        self.domf_bin_counts: pl.DataFrame = None
        # Burstiness
        self.bu_entity_burstiness: pl.DataFrame = None
        self.bu_global_burstiness: pl.DataFrame = None
        # Interval autocorrelation
        self.ia_autocorrelation: pl.DataFrame = None
        # Plots
        self.plots = IrregularityPlots()


class Irregularity:
    def __init__(
        self,
        df: pl.DataFrame,
        id_col: str = "id",
        clock_col: str = "clock",
        save_path: Path = None,
        plot_library: str = "matplotlib",
        renderer: str = "notebook_connected",
    ):
        """
        Initializes the Irregularity analysis class.

        Parameters:
            df (pl.DataFrame): The dataframe to analyze.
            id_col (str): The column representing unique identifiers.
            clock_col (str): The column representing time or clock. May be a Datetime
                             column or a numeric column (integer/float representing seconds).
            save_path (Path): Path to save results. If None, results will not be saved.
            plot_library (str): The plotting library to use ('matplotlib' or 'plotly').
            renderer (str): The renderer for Plotly visualizations.
        """
        if not renderer and not save_path:
            pretty_printing.rich_warning(
                "⚠️ No renderer or save_path provided. "
                "Visualizations will not be displayed or saved."
            )
        # Dataframe that will be analyzed
        self.df: pl.DataFrame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df) # if pandas DataFrame is passed, convert to Polars
        # Relevant columns for the analysis
        self.id_col = id_col
        self.clock_col = clock_col

        # Result persistence
        self.save_path = save_path

        # Plotting library
        if plot_library not in ["matplotlib", "plotly"]:
            raise ValueError(
                f"Unsupported plot library: {plot_library}. Supported libraries: 'matplotlib', 'plotly'."
            )
        self.plot_library = plot_library
        # Plotly rendering
        self.renderer = renderer

        # Results
        self.results = IrregularityResults()

        # Cached inter-observation interval DataFrame (computed lazily)
        self._delta_t_df: pl.DataFrame = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_intervals(self) -> pl.DataFrame:
        """
        Compute and cache per-entity inter-observation intervals.

        Handles both Datetime clock columns (intervals converted to seconds via
        .dt.total_seconds()) and numeric clock columns (intervals computed as
        a plain numeric difference, cast to Float64).

        The resulting "interval_seconds" column represents the gap between
        consecutive observations. Null values (first observation per entity)
        and zero-length intervals (duplicate timestamps) are filtered out.

        Returns:
            pl.DataFrame: Sorted DataFrame with an additional "interval_seconds" column.
        """
        if self._delta_t_df is None:
            sorted_df = self.df.sort([self.id_col, self.clock_col])
            clock_dtype = sorted_df[self.clock_col].dtype

            is_temporal = (
                clock_dtype == pl.Datetime
                or clock_dtype == pl.Date
                or clock_dtype == pl.Duration
                or str(clock_dtype).startswith("Datetime")
            )

            if is_temporal:
                intervals = (
                    pl.col(self.clock_col)
                    .diff()
                    .over(self.id_col)
                    .dt.total_seconds()
                    .alias("interval_seconds")
                )
            else:
                intervals = (
                    pl.col(self.clock_col)
                    .diff()
                    .over(self.id_col)
                    .cast(pl.Float64)
                    .alias("interval_seconds")
                )

            self._delta_t_df = (
                sorted_df
                .with_columns(intervals)
                .filter(pl.col("interval_seconds").is_not_null())
                .filter(pl.col("interval_seconds") > 0)
            )
        return self._delta_t_df

    def _path(self, subpath: str) -> Path:
        """Generates a full path for saving results."""
        if self.save_path:
            return self.save_path / subpath
        return None

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def interval_statistics(self, save_results: bool = True) -> "Irregularity":
        """
        Compute per-entity and global summary statistics of inter-observation intervals.

        The coefficient of variation (CV = std / mean) per entity is the primary
        irregularity score: CV = 0 for a perfectly regular time grid, increasing
        values indicate increasing irregularity.

        Results stored in:
            self.results.ins_entity_statistics  — one row per entity
            self.results.ins_global_statistics  — describe()-style global summary
            self.results.plots.ins_cv_violin    — violin plot of per-entity CV values

        Parameters:
            save_results (bool): Whether to save CSVs and plots to save_path.

        Returns:
            self: Supports method chaining.
        """
        new_path_level_name = "interval_statistics"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        delta_t_df = self._compute_intervals()

        self.results.ins_entity_statistics = (
            interval_statistics_module.compute_entity_interval_statistics(
                delta_t_df, id_col=self.id_col
            )
        )
        self.results.ins_global_statistics = (
            interval_statistics_module.compute_global_interval_statistics(delta_t_df)
        )

        if self.renderer:
            pretty_printing.rich_info(
                "Interval Statistics — Entity Level: "
                "cv = std/mean per entity (0 = perfectly regular, higher = more irregular); "
                "iqr = spread of interval lengths."
            )
            print(self.results.ins_entity_statistics.describe(interpolation="linear"))
            pretty_printing.rich_info(
                "Interval Statistics — Global: pooled summary over all inter-observation intervals."
            )
            print(self.results.ins_global_statistics)

        if save_results and path:
            self.results.ins_entity_statistics.write_csv(path / "entity_statistics.csv")
            self.results.ins_global_statistics.write_csv(path / "global_statistics.csv")

        # Violin of CV values across entities (drop entities with NaN CV)
        cv_df = self.results.ins_entity_statistics.filter(pl.col("cv").is_not_null())
        if cv_df.height > 0:
            cv_violin = visualization_utils.plot_violin(
                cv_df,
                y="cv",
                title="Per-Entity Coefficient of Variation of Inter-Observation Intervals",
                yaxis_title="CV (std / mean)",
                library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/cv_violin.png"),
                save_results=save_results,
            )
            self.results.plots.ins_cv_violin = cv_violin

        return self

    def dominant_frequency(
        self,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        save_results: bool = True,
    ) -> "Irregularity":
        """
        Identify the modal inter-observation interval and quantify how much of the data
        adheres to it, along with the Shannon entropy of the interval distribution.

        Adherence rate: fraction of all intervals within
            [dominant * (1 - tolerance), dominant * (1 + tolerance)].
        Normalized entropy: 0 = perfectly regular (single interval), 1 = maximally spread.

        Results stored in:
            self.results.domf_frequency_summary          — single-row summary
            self.results.domf_bin_counts                 — full frequency table
            self.results.plots.domf_interval_frequency_bar — bar chart of top bins

        Parameters:
            bin_resolution_seconds (float): Bin width in seconds for discretizing intervals.
                                            Default 60.0 (1-minute bins). Adjust for sub-minute data.
            adherence_tolerance (float): Fractional tolerance around the dominant interval.
            save_results (bool): Whether to save CSVs and plots to save_path.

        Returns:
            self: Supports method chaining.
        """
        new_path_level_name = "dominant_frequency"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        delta_t_df = self._compute_intervals()

        self.results.domf_frequency_summary, self.results.domf_bin_counts = (
            interval_statistics_module.compute_dominant_frequency(
                delta_t_df,
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
            )
        )

        if self.renderer:
            pretty_printing.rich_info(
                "Dominant Frequency Summary: "
                "dominant_interval_seconds = most common gap length (the modal rhythm); "
                "adherence_rate = fraction of intervals within ±tolerance of that mode (1 = consistent schedule); "
                "normalized_entropy = spread of the interval distribution (0 = all intervals equal, 1 = maximally irregular)."
            )
            print(self.results.domf_frequency_summary)

        if save_results and path:
            self.results.domf_frequency_summary.write_csv(path / "frequency_summary.csv")
            self.results.domf_bin_counts.write_csv(path / "bin_counts.csv")

        dominant_bin = self.results.domf_frequency_summary["dominant_interval_bin"][0]
        freq_bar = interval_statistics_module.plot_interval_frequency_bar(
            bin_counts=self.results.domf_bin_counts,
            dominant_bin=dominant_bin,
            library=self.plot_library,
            renderer=self.renderer,
            save_path=self._path(f"{new_path_level_name}/interval_frequency_bar.png"),
            save_results=save_results,
        )
        self.results.plots.domf_interval_frequency_bar = freq_bar

        return self

    def burstiness(self, save_results: bool = True) -> "Irregularity":
        """
        Compute the burstiness coefficient B per entity and globally.

        B = (std - mean) / (std + mean)  [Goh & Barabasi, 2008]
        Range [-1, 1]: B = -1 perfectly regular, B = 0 Poisson, B > 0 bursty.

        Entities with fewer than 3 intervals receive NaN for burstiness_coeff.

        Results stored in:
            self.results.bu_entity_burstiness    — one row per entity
            self.results.bu_global_burstiness    — single-row global summary
            self.results.plots.bu_burstiness_violin — violin of entity B values

        Parameters:
            save_results (bool): Whether to save CSVs and plots to save_path.

        Returns:
            self: Supports method chaining.
        """
        new_path_level_name = "burstiness"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        delta_t_df = self._compute_intervals()

        self.results.bu_entity_burstiness = burstiness_module.compute_burstiness_coefficient(
            delta_t_df, id_col=self.id_col
        )
        self.results.bu_global_burstiness = burstiness_module.compute_global_burstiness(
            delta_t_df, id_col=self.id_col
        )

        if self.renderer:
            pretty_printing.rich_info("Burstiness — Entity Level (B=1: bursty, B=-1: perfectly periodic):")
            print(self.results.bu_entity_burstiness.describe(interpolation="linear"))
            pretty_printing.rich_info("Burstiness — Global (B=1: bursty, B=-1: perfectly periodic):")
            print(self.results.bu_global_burstiness)

        if save_results and path:
            self.results.bu_entity_burstiness.write_csv(path / "entity_burstiness.csv")
            self.results.bu_global_burstiness.write_csv(path / "global_burstiness.csv")

        b_df = self.results.bu_entity_burstiness.filter(
            pl.col("burstiness_coeff").is_not_null()
        )
        if b_df.height > 0:
            b_violin = visualization_utils.plot_violin(
                b_df,
                y="burstiness_coeff",
                title="Per-Entity Burstiness Coefficient",
                yaxis_title="Burstiness Coefficient B",
                library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/burstiness_violin.png"),
                save_results=save_results,
            )
            self.results.plots.bu_burstiness_violin = b_violin

        return self

    def interval_autocorrelation(
        self, lags: int = 20, save_results: bool = True
    ) -> "Irregularity":
        """
        Compute the autocorrelation of inter-observation intervals across lags.

        A positive autocorrelation at lag k means that a long gap tends to be
        followed by another long gap k steps later (and vice versa for short gaps).

        Reuses the existing acf() function from intravariable.autocorrelation,
        treating the interval sequence as the signal.

        Results stored in:
            self.results.ia_autocorrelation    — DataFrame with columns lag, autocorr
            self.results.plots.ia_acf_plot     — scatter/line plot of lag vs autocorr

        Parameters:
            lags (int): Maximum number of lags to compute.
            save_results (bool): Whether to save CSVs and plots to save_path.

        Returns:
            self: Supports method chaining.
        """
        new_path_level_name = "interval_autocorrelation"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        delta_t_df = self._compute_intervals()

        # Add a sequential integer index per entity to serve as clock_no_col for acf()
        interval_no_col = "_interval_no"
        acf_input = delta_t_df.with_columns(
            pl.int_range(pl.len()).over(self.id_col).alias(interval_no_col)
        )

        self.results.ia_autocorrelation = autocorrelation.acf(
            mask_df=acf_input,
            col="interval_seconds",
            id_col=self.id_col,
            clock_no_col=interval_no_col,
            max_lag=lags,
            save_path=self._path(new_path_level_name),
            save_results=save_results,
            addition_to_save_path="interval_autocorrelation.csv",
        )

        if self.renderer:
            pretty_printing.rich_info(
                "Interval Autocorrelation: correlation of each gap with the gap k steps later. "
                "Positive = long gaps tend to follow long gaps; "
                "near-zero = gaps are uncorrelated (Poisson-like); "
                "negative = gaps alternate short/long."
            )
            print(self.results.ia_autocorrelation)

        acf_fig = visualization_utils.plot_scatter(
            x=self.results.ia_autocorrelation["lag"].to_numpy(),
            y=self.results.ia_autocorrelation["autocorr"].to_numpy(),
            title="Autocorrelation of Inter-Observation Intervals",
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
            save_path=self._path(f"{new_path_level_name}/interval_acf_plot.png"),
            save_results=save_results,
            renderer=self.renderer,
            library=self.plot_library,
        )
        self.results.plots.ia_acf_plot = acf_fig

        return self

    def run(
        self,
        save_results: bool = True,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        autocorrelation_lags: int = 20,
    ) -> "Irregularity":
        """
        Run all irregularity analyses in sequence.

        Parameters:
            save_results (bool): Whether to save results to save_path.
            bin_resolution_seconds (float): Bin width for dominant_frequency().
            adherence_tolerance (float): Tolerance for dominant_frequency() adherence rate.
            autocorrelation_lags (int): Number of lags for interval_autocorrelation().

        Returns:
            self: Supports method chaining.
        """
        try:
            self.interval_statistics(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in interval_statistics: {e}")

        try:
            self.dominant_frequency(
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
                save_results=save_results,
            )
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in dominant_frequency: {e}")

        try:
            self.burstiness(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in burstiness: {e}")

        try:
            self.interval_autocorrelation(lags=autocorrelation_lags, save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in interval_autocorrelation: {e}")

        return self


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "a", "a", "c", "c"],
            "clock": [
                "2023-01-01 00:00:00",
                "2023-01-01 00:02:00",
                "2023-01-01 00:10:00",
                "2023-01-01 00:15:00",
                "2023-01-01 00:20:00",
                "2023-02-02 00:25:00",
                "2023-02-02 00:30:00",
            ],
            "heartrate": [60, None, 70, 65, None, 80, None],
            "blood_pressure": [120, 130, None, None, None, 135, None],
        }
    ).with_columns(
        [
            pl.col("clock").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )
    
    print(df)
    irregularity_analysis = Irregularity(df, save_path=Path("results/irregularity_example"), renderer="notebook_connected")
    irregularity_analysis.run(save_results=True)