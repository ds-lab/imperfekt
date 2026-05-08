import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from imperfekt.analysis.intravariable import (
    autocorrelation,
    column_statistics,
    date_time_statistics,
    gap_statistics,
    markov_chain_summary,
    windowed_significance
)
from imperfekt.analysis.utils import masking, pretty_printing, statistics_utils, visualization_utils


class IntravariablePlots:
    def __init__(self):
        self.cs_imperfection_histogram: dict = {}
        self.cs_imperfection_boxplot: dict = {}
        self.gs_gap_lengths_violin: dict = {}
        self.gr_gap_returns_boxplot: dict = {}
        self.gr_posthoc_pval_heatmap: dict = {}
        self.gr_posthoc_es_heatmap: dict = {}
        self.mc_heatmap: dict = {}
        self.ac_lag_plot: dict = {}
        self.ws_overlay_histogram: dict = {}
        self.ws_multi_boxplot: dict = {}
        self.dt_month_daytime_heatmap: dict = {}


class IntravariableResults:
    def __init__(self):
        # Analytical results
        self.cs_overall_statistics: pl.DataFrame = None
        self.cs_case_level_statistics: pl.DataFrame = None
        self.gs_gaps_observation_runs: pl.DataFrame = None
        self.gs_gaps_df: dict[str, pl.DataFrame] = {}
        self.gs_gap_dominant: dict[str, pl.DataFrame] = {}
        self.gs_gap_burstiness: dict[str, pl.DataFrame] = {}
        self.gr_gap_returns: pl.DataFrame = None
        self.gr_gap_kruskal: dict = {}
        self.mc_markov_summary: dict = {}
        self.ac_autocorrelation: dict = {}
        self.ws_observations_around_indicated: dict = {}
        self.ws_mwu_result: pl.DataFrame = None
        self.dt_date_time_statistics: dict = {}
        self.iv_composite_scores: pl.DataFrame = None
        self.iv_pairwise_correlations: dict[str, pl.DataFrame] = {}
        # Plots
        self.plots = IntravariablePlots()


class IntravariableImperfection:
    """
    A class for performing intravariable "imperfection" analysis on a Polars DataFrame. Imperfection refers to missingness, noise etc. things that can be indicated using a binary mask.
    This class provides methods to analyze column completeness, gaps and observation lengths,
    gaps and returns, Markov chain summaries, autocorrelation, temporal analysis, and datetime correlation
    from a intravariable perspective.
    Attributes:
        df (pl.DataFrame): The input DataFrame to analyze.
        imperfection (str): The type of imperfection to analyze (e.g., "missingness").
        mask_df (pl.DataFrame): A DataFrame with binary values indicating imperfection (1 for imperfect/noisy/indicated, 0 for present/normal/expected). Can be used for custom imperfection analysis.
        id_col (str): The name of the column representing the unique identifier for each row.
        clock_col (str): The name of the column representing the clock time.
        clock_no_col (str): The name of the column representing the clock number (integer index that orders time-series). This column is generated if not present.
        cols (list): List of columns to analyze for imperfection. If None, all columns except id_col, clock_col, and clock_no_col are considered.
        alpha (float): The significance level for hypothesis testing.
        save_path (Path): Path to save the results. If None, results are not saved.
        renderer (str): Renderer for visualizations. Defaults to "notebook_connected".
    Methods:
        column_statistics(threshold: float = 5, save_results: bool = True):
            Analyzes the completeness of each column in the DataFrame and generates visualizations.
        gap_statistics(save_results: bool = True, gap_and_return_bins: list = None):
            Analyzes gaps and observation lengths, and extracts gap and return values for each column.
        markov_chain_summary(save_results: bool = True):
            Summarizes Markov chain properties for each column and generates visualizations.
        autocorrelation(lags: int = 20, save_results: bool = True):
            Computes and visualizes the autocorrelation of imperfection for each column.
        windowed_significance(save_results: bool = True, window_size: timedelta = timedelta(minutes=5), window_location: str = "both"):
            Analyzes and visualizes observations around imperfect values for each column within a specified temporal window.
        date_time_statistics(save_results: bool = True):
            Analyzes and visualizes datetime distributions for each column.
        run(save_results: bool = True, gap_and_return_bins: list = None, window_size: timedelta = timedelta(minutes=5), window_location: str = "both"):
            Runs all analyses in sequence.
    Usage:
        intravariable_imperfection = IntravariableImperfection(
            df=your_dataframe,
            id_col="your_id_column",
            clock_col="your_clock_column",
            clock_no_col="your_clock_no_column",
            cols=["col1", "col2"],  # Specify columns to analyze, or leave as None to analyze all except id_col, clock_col, and clock_no_col
            save_path=Path("path/to/save/results"),
            renderer="notebook_connected"  # Specify the renderer for visualizations
        )
        intravariable_imperfection.run(save_results=True)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        imperfection: str = "missingness",
        mask_df: pl.DataFrame = None,
        id_col: str = "id",
        clock_col: str = "clock",
        clock_no_col: str = "clock_no",
        cols: list = None,
        alpha: float = 0.05,
        save_path: Path = None,
        plot_library: str = "matplotlib",
        renderer: str = "notebook_connected",
    ):
        if not renderer and not save_path:
            pretty_printing.rich_warning(
                "⚠️ No renderer or save_path provided. "
                "Visualizations will not be displayed or saved."
            )
        # Dataframe that will be analyzed
        self.df = df
        # Relevant columns for the analysis
        self.id_col = id_col
        self.clock_col = clock_col
        self.clock_no_col = clock_no_col
        self._generate_clock_no_col()
        self.cols = cols or [c for c in df.columns if c not in {id_col, clock_col, clock_no_col}]
        self.alpha = alpha

        # Binary indicator mask for imperfection
        self.imperfection = imperfection
        if imperfection == "missingness" and mask_df is None:
            self.mask = masking.create_missingness_mask(
                df=self.df,
                id_col=id_col,
                clock_col=clock_col,
                clock_no_col=clock_no_col,
                cols=self.cols,
            )
        else:
            if mask_df is not None:
                self.mask = mask_df
            else:
                raise ValueError(
                    f"Unsupported imperfection type: {imperfection}. Supported types: 'missingness'."
                )

        # Result persistence
        self.save_path = Path(save_path) if save_path else None

        # Plotting library
        if plot_library not in ["matplotlib", "plotly"]:
            raise ValueError(
                f"Unsupported plot library: {plot_library}. Supported libraries: 'matplotlib', 'plotly'."
            )
        self.plot_library = plot_library
        # Plotly rendering
        self.renderer = renderer

        # Results
        self.results = IntravariableResults()

        # Analysis parameters init
        self.ws_window_size = None
        self.ws_window_location = None
        self.gr_gap_and_return_bins = None
        self.ac_autocorrelation_lags = None
        self.gs_bin_resolution_seconds = None
        self.gs_adherence_tolerance = None

    def column_statistics(self, threshold: float = 5, save_results: bool = True):
        """Analyzes the completeness of each column in the DataFrame and generate visualizations.

        Parameters:
            threshold (float): The threshold percentage for imperfection to consider a column as incomplete.
            save_results (bool): Whether to save the results to files.

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining.
        """
        new_path_level_name = "column_statistics"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        self.results.cs_overall_statistics = column_statistics.analyze_column_imperfection(
            mask_df=self.mask,
            cols=self.cols,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
            save_path=self._path(f"{new_path_level_name}/column_statistics.csv"),
            save_results=save_results,
        )
        self.results.cs_case_level_statistics = (
            column_statistics.analyze_column_imperfection_per_id(
                mask_df=self.mask,
                cols=self.cols,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                threshold=threshold,
                save_path=self._path(f"{new_path_level_name}/column_statistics_per_id.csv"),
                save_results=save_results,
            )
        )

        for c in self.cols:
            if self.renderer:
                print(f"Imperfection distribution for {c} per ID:")

            hist_fig = visualization_utils.plot_histogram(
                self.results.cs_case_level_statistics,
                x=f"{c}_indicated_pct",
                title=f"{c} Imperfection Distribution (per ID)",
                xaxis_title=f"{c} Imperfection Percentage",
                yaxis_title="#Cases",
                library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(
                    f"{new_path_level_name}/{c}_imperfection_distribution_per_id.png"
                ),
                save_results=save_results,
            )
            self.results.plots.cs_imperfection_histogram[c] = hist_fig

            box_fig = visualization_utils.plot_boxplot(
                self.results.cs_case_level_statistics,
                y=f"{c}_indicated_pct",
                title=f"{c} Imperfection Boxplot (per ID)",
                yaxis_title=f"{c} Imperfection Percentage",
                library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/{c}_imperfection_boxplot_per_id.png"),
                save_results=save_results,
            )
            self.results.plots.cs_imperfection_boxplot[c] = box_fig

        return self

    def gap_statistics(
        self,
        save_results: bool = True,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
    ):
        """Analyzes gaps and observation lengths.

        Parameters:
            save_results (bool): Whether to save the results to files.
            bin_resolution_seconds (float): Bin width in seconds for dominant gap length detection.
            adherence_tolerance (float): Fractional tolerance around the dominant gap for adherence rate.

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining. Updates the following attributes:
                results.gs_gaps_observation_runs (pl.DataFrame): DataFrame containing gaps and observation lengths.
                results.gs_gap_dominant (dict): Per-column dominant gap length summary.
                results.gs_gap_burstiness (dict): Per-column gap burstiness summary.
        """
        self.gs_bin_resolution_seconds = bin_resolution_seconds
        self.gs_adherence_tolerance = adherence_tolerance

        # Get gap and observation runs (and lengths) for each column, shape: (id_col, variable, count_clock_no, time_length, run_start_clock, run_end_clock)
        self.results.gs_gaps_observation_runs = gap_statistics.analyze_gap_lengths(
            mask_df=self.mask,
            cols=self.cols,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
        )

        # Result persistence and plotting
        for c in self.cols:
            new_path_level_name = f"gap_statistics/{c}"
            if self.save_path and save_results:
                (self.save_path / new_path_level_name).mkdir(parents=True, exist_ok=True)

            if self.renderer:
                print(f"Gap and observation lengths for {c}:")
            gaps_df, gap_length_violin = gap_statistics.gap_lengths(
                lengths_df=self.results.gs_gaps_observation_runs,
                col=c,
                save_path=self._path(new_path_level_name),
                save_results=save_results,
                renderer=self.renderer,
                plot_library=self.plot_library,
            )
            self.results.gs_gaps_df[c] = gaps_df
            self.results.plots.gs_gap_lengths_violin[c] = gap_length_violin

            self.results.gs_gap_dominant[c] = gap_statistics.compute_gap_dominant_length(
                gaps_df=gaps_df,
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
            )
            self.results.gs_gap_burstiness[c] = gap_statistics.compute_gap_burstiness(
                gaps_df=gaps_df,
                id_col=self.id_col,
            )

        return self

    def gap_returns(
        self,
        save_results: bool = True,
        gap_and_return_bins: list = None,
    ):
        """Analyzes gaps and their corresponding return values for each column.

        Parameters:
            save_results (bool): Whether to save the results to files.
            gap_and_return_bins (list): Bins for gap and return analysis.

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining.
        """
        self.gr_gap_and_return_bins = gap_and_return_bins
        # Get the return value for each gap and column, shape: same as gs_gaps_observation_runs + return_value, return_time, clock_no of return
        self.results.gr_gap_returns = gap_statistics.extract_gap_return_values(
            df=self.df,
            mask_df=self.mask,
            cols=self.cols,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
        )

        # Result persistence and plotting
        for c in self.cols:
            new_path_level_name = f"gap_returns/{c}"
            if self.save_path and save_results:
                (self.save_path / new_path_level_name).mkdir(parents=True, exist_ok=True)

            if self.renderer:
                print(f"Gap and return values for {c}:")
            (
                kw_result,
                gap_return_boxfig,
                posthoc_pval_heatmap_fig,
                posthoc_es_heatmap_fig,
            ) = gap_statistics.gap_returns(
                spans=self.results.gr_gap_returns,
                col=c,
                bins=gap_and_return_bins,
                plot_library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(new_path_level_name),
                save_results=save_results,
            )
            self.results.gr_gap_kruskal[c] = kw_result

            # Plots that visualize the kruskal results and provide more insights
            self.results.plots.gr_gap_returns_boxplot[c] = gap_return_boxfig
            self.results.plots.gr_posthoc_pval_heatmap[c] = posthoc_pval_heatmap_fig
            self.results.plots.gr_posthoc_es_heatmap[c] = posthoc_es_heatmap_fig

        return self

    def markov_chain_summary(self, save_results: bool = True):
        """Summarizes Markov chain properties for each column and generates visualizations.

        Parameters:
            save_results (bool): Whether to save the results to files.

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining.
        """
        new_path_level_name = "markov_chain_summary"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        for c in self.cols:
            self.results.mc_markov_summary[c] = markov_chain_summary.markov_chain_summary(
                mask_df=self.mask,
                col=c,
                id_col=self.id_col,
                clock_no_col=self.clock_no_col,
                save_path=self._path(f"{new_path_level_name}/{c}_markov_chain_summary.csv"),
                save_results=save_results,
            )

            if self.renderer:
                print(f"Markov Chain Summary for {c}:")
                print("Transition Matrix:")
                print(self.results.mc_markov_summary[c]["transition_matrix"])
                print("Transition Counts:")
                print(self.results.mc_markov_summary[c]["transition_counts"])
                print("Steady State Distribution:")
                print(self.results.mc_markov_summary[c]["steady_state"])

            heatmap_fig = markov_chain_summary.plot_markov_heatmap(
                probs=self.results.mc_markov_summary[c]["transition_matrix"],
                labels=self.results.mc_markov_summary[c]["labels"],
                title=f"Markov Chain Transition Matrix for {c}",
                save_path=self._path(f"{new_path_level_name}/{c}_markov_chain_heatmap.png"),
                save_results=save_results,
                renderer=self.renderer,
            )
            self.results.plots.mc_heatmap[c] = heatmap_fig

        return self

    def autocorrelation(
        self,
        lags: int = 20,
        save_results: bool = True,
        seasonal_trend_decomposition: bool = False,
        stl_period: int = 7,
    ):
        """Computes and visualizes the autocorrelation of imperfection for each column.

        Parameters:
            lags (int): The number of lags to compute for autocorrelation.
            save_results (bool): Whether to save the results to files.
            seasonal_trend_decomposition (bool): Whether to perform seasonal trend decomposition (default: False).
            stl_period (int): Seasonal period for decomposition (default: 7).

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining.
        """
        self.ac_autocorrelation_lags = lags
        new_path_level_name = "autocorrelation"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        for c in self.cols:
            if path is not None:
                (path / c).mkdir(parents=True, exist_ok=True)
            self.results.ac_autocorrelation[c] = autocorrelation.acf(
                mask_df=self.mask,
                col=c,
                id_col=self.id_col,
                clock_no_col=self.clock_no_col,
                max_lag=lags,
                seasonal_trend_decomposition=seasonal_trend_decomposition,
                stl_period=stl_period,
                save_path=self._path(f"{new_path_level_name}/{c}"),
                save_results=save_results,
            )

            if self.renderer:
                print(f"Autocorrelation for {c}:")
                print(self.results.ac_autocorrelation[c])

            acf_fig = visualization_utils.plot_scatter(
                x=self.results.ac_autocorrelation[c]["lag"].to_numpy(),
                y=self.results.ac_autocorrelation[c]["autocorr"].to_numpy(),
                title=f"Lagged Autocorrelation of Imperfection: {c}",
                xaxis_title="Lag",
                yaxis_title="Autocorrelation",
                save_path=self._path(f"{new_path_level_name}/{c}/{c}_autocorrelation_plot.png"),
                save_results=save_results,
                renderer=self.renderer,
                library=self.plot_library,
            )
            self.results.plots.ac_lag_plot[c] = acf_fig

        return self

    def windowed_significance(
        self,
        save_results: bool = True,
        window_size: timedelta = timedelta(minutes=5),
        window_location: str = "both",
    ):
        """Analyzes and visualizes observations around imperfect values for each column within a specified temporal window.

        Parameters:
            save_results (bool): Whether to save the results to files.
            window_size (timedelta): Size of the temporal window for analysis.
            window_location (str): Location of the temporal window ('before', 'after', 'both').

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining.
        """
        self.ws_window_size = window_size
        self.ws_window_location = window_location
        new_path_level_name = "windowed_significance"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        # Pre-cast the clock column to a consistent timezone-aware type to avoid repeated casting in the loop.
        self.df = self.df.with_columns(pl.col(self.clock_col).cast(pl.Datetime("ms", "UTC")))

        numeric_cols = self.df.select(pl.selectors.numeric()).columns
        mwu_res = {}
        for c in self.cols:
            if c not in numeric_cols:
                pretty_printing.rich_warning(
                    f"⚠️ Column '{c}' is not numeric. Skipping temporal analysis for this column."
                )
                continue

            around_indicated_df = windowed_significance.extract_values_near_indicated(
                df=self.df,
                mask_df=self.mask,
                col=c,
                id_col=self.id_col,
                clock_col=self.clock_col,
                window=window_size,
                window_location=window_location,
            )

            if around_indicated_df.is_empty():
                print(f"No temporal data found for {c} around imperfect values.")
                continue

            # Cast the extracted dataframe's clock column to match self.df
            self.results.ws_observations_around_indicated[c] = around_indicated_df.with_columns(
                pl.col(self.clock_col).cast(pl.Datetime("ms", "UTC"))
            )

            # Use an anti-join to find the remaining rows efficiently.
            remaining_df = self.df.join(
                self.results.ws_observations_around_indicated[c],
                on=[self.id_col, self.clock_col],
                how="anti",
            )

            if remaining_df.is_empty():
                print(f"No remaining data found for {c} after removing 'around imperfect' values.")
                continue

            mwu_res[c] = statistics_utils.mwu_two_subgroups(
                df1=self.results.ws_observations_around_indicated[c],
                df2=remaining_df,
                col1=c,
                col2=c,
                alpha=self.alpha,
                print_info=bool(self.renderer),
                save_path=self._path(f"{new_path_level_name}/{c}_mwu_results.csv"),
                save_results=save_results,
            )

            hist_overlay_fig = visualization_utils.plot_overlay_histograms(
                dfs=[remaining_df, self.results.ws_observations_around_indicated[c]],
                x=c,
                group_names=[f"{c} Remaining", f"{c} Around Imperfect Values"],
                title=f"Overlay Histogram of {c}",
                xaxis_title=f"{c}",
                yaxis_title="Frequency",
                histnorm="probability",
                library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/{c}_overlay_histogram.png"),
                save_results=save_results,
            )
            self.results.plots.ws_overlay_histogram[c] = hist_overlay_fig

            multi_box_fig = visualization_utils.plot_multi_boxplot(
                dfs=[remaining_df, self.results.ws_observations_around_indicated[c]],
                y=c,
                group_names=[f"{c} Remaining", f"{c} Around Imperfect Values"],
                title=f"Boxplots of {c}: Around Imperfect vs Remaining",
                yaxis_title=f"{c}",
                library=self.plot_library,
                renderer=self.renderer,
                save_path=self._path(
                    f"{new_path_level_name}/{c}_boxplot_around_indicated_vs_remaining.png"
                ),
                save_results=save_results,
            )
            self.results.plots.ws_multi_boxplot[c] = multi_box_fig
        rows = [{"column": k, **v} for k, v in mwu_res.items()]
        self.results.ws_mwu_result = pl.from_dicts(rows)
        return self

    def date_time_statistics(self, save_results: bool = True):
        """Analyzes and visualizes datetime distributions for each column.
        Parameters:
            save_results (bool): Whether to save the results to files.

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining.
        """
        new_path_level_name = "date_time_statistics"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        for c in self.cols:
            m, w, h = date_time_statistics.extract_datetime_distribution(
                mask_df=self.mask,
                col=c,
                clock_col=self.clock_col,
                save_path=self._path(f"{new_path_level_name}"),
                save_results=save_results,
            )
            self.results.dt_date_time_statistics[c] = {
                "monthly": m,
                "weekly": w,
                "hourly": h,
            }

            if self.renderer:
                print(f"Datetime distribution for {c}:")
                print(f"Monthly Distribution of {c}:")
                print(m)
                print(f"Weekly Distribution of {c}:")
                print(w)
                print(f"Hourly Distribution of {c}:")
                print(h)

            heatmap_fig = date_time_statistics.visualize_month_daytime_heatmap(
                mask_df=self.mask,
                col=c,
                clock_col=self.clock_col,
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/{c}_month_daytime_heatmap.png"),
                save_results=save_results,
            )
            self.results.plots.dt_month_daytime_heatmap[c] = heatmap_fig
        return self

    # Axes where a *lower* value means *more* imperfect (all others: higher = more imperfect)
    INVERTED_AXES: frozenset = frozenset({"gap_adherence_rate"})

    @staticmethod
    def assign_strata(
        df: pl.DataFrame,
        axis_x: str,
        axis_y: str,
        x_median: float,
        y_median: float,
    ) -> pl.DataFrame:
        """
        Assign each row to an imperfection quadrant by median-bisecting two axes.

        Returns df with an added "imperfection_stratum" column
        (Q_alpha / Q_beta / Q_gamma / Q_delta, or null for rows with nulls on either axis).

        Axis direction:
            gap_adherence_rate — lower = more imperfect (inverted)
            all other axes    — higher = more imperfect

        Parameters:
            df (pl.DataFrame): Input DataFrame containing the axis columns.
            axis_x (str): Column name for the x-axis metric.
            axis_y (str): Column name for the y-axis metric.
            x_median (float): Median threshold for the x-axis.
            y_median (float): Median threshold for the y-axis.
        """
        x_high = (
            pl.col(axis_x) <= x_median
            if axis_x in IntravariableImperfection.INVERTED_AXES
            else pl.col(axis_x) > x_median
        )
        y_high = (
            pl.col(axis_y) <= y_median
            if axis_y in IntravariableImperfection.INVERTED_AXES
            else pl.col(axis_y) > y_median
        )
        return df.with_columns(
            pl.when(pl.col("indicated_pct") == 0)
            .then(pl.lit("Q_complete"))
            .when(pl.col(axis_x).is_null() | pl.col(axis_y).is_null())
            .then(pl.lit(None))
            .when(~x_high & ~y_high).then(pl.lit("Q_alpha"))
            .when(x_high & ~y_high).then(pl.lit("Q_beta"))
            .when(~x_high & y_high).then(pl.lit("Q_gamma"))
            .when(x_high & y_high).then(pl.lit("Q_delta"))
            .otherwise(pl.lit(None))
            .alias("imperfection_stratum")
        )

    def composite_score(
        self,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        min_observations: int = 10,
        save_results: bool = True,
    ) -> "IntravariableImperfection":
        """
        Assign each (case, variable) pair to one of four imperfection quadrants.

        For each variable independently:
          1. Collect per-case imperfection metrics as candidate axes.
          2. Select the axis pair with the lowest absolute Spearman correlation
             (most orthogonal dimensions of imperfection for that variable).
          3. Median-bisect the selected axes to assign Q_alpha / Q_beta / Q_gamma / Q_delta.

        Candidate axes (per case, per variable):
            indicated_pct       : overall missingness burden
            gap_cv              : CV of gap lengths
            gap_qcod            : quartile CoD of gap lengths
            gap_burstiness_coeff: Goh & Barabási burstiness
            gap_adherence_rate  : fraction near dominant gap length (inverted axis)
            gap_normalized_entropy: entropy of gap length distribution
            max_gap_fraction    : max gap / observation window
            gap_onset_cv        : CV of inter-onset intervals
            mc_p11              : Markov P(1→1) persistence probability

        Requires column_statistics() and gap_statistics() to have been run.
        Runs per-case Markov P(1→1) internally.

        Results stored in:
            self.results.iv_composite_scores        — one row per (case × variable)
            self.results.iv_pairwise_correlations   — dict keyed by variable name

        Parameters:
            bin_resolution_seconds (float): Bin width for entropy/adherence computation.
            adherence_tolerance (float): Fractional tolerance for gap adherence rate.
            min_observations (int): Min observations per case for Markov P(1→1).
            save_results (bool): Whether to save CSVs to save_path.

        Returns:
            self: Supports method chaining.
        """
        new_path_level_name = "composite_score"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        if self.results.cs_case_level_statistics is None:
            self.column_statistics(save_results=save_results)
        if self.results.gs_gaps_observation_runs is None:
            self.gap_statistics(
                save_results=save_results,
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
            )

        # --- indicated_pct per (case, variable) ---
        case_stats = self.results.cs_case_level_statistics
        indicated_long = pl.concat([
            case_stats.select([self.id_col, pl.lit(c).alias("variable"),
                               pl.col(f"{c}_indicated_pct").alias("indicated_pct")])
            for c in self.cols
        ])

        # --- per-case gap metrics ---
        gap_metrics = gap_statistics.compute_case_gap_metrics(
            gaps_df=self.results.gs_gaps_observation_runs,
            mask_df=self.mask,
            id_col=self.id_col,
            clock_col=self.clock_col,
            bin_resolution_seconds=bin_resolution_seconds,
            adherence_tolerance=adherence_tolerance,
        )

        # --- per-case Markov P(1→1) ---
        p11 = markov_chain_summary.compute_case_markov_p11(
            mask_df=self.mask,
            cols=self.cols,
            id_col=self.id_col,
            clock_no_col=self.clock_no_col,
            min_observations=min_observations,
        )

        # --- join all metrics ---
        base = (
            indicated_long
            .join(gap_metrics, on=[self.id_col, "variable"], how="left")
            .join(p11, on=[self.id_col, "variable"], how="left")
        )

        candidate_axes = [
            "indicated_pct",
            "gap_cv",
            "gap_qcod",
            "gap_burstiness_coeff",
            "gap_adherence_rate",
            "gap_normalized_entropy",
            "max_gap_fraction",
            "gap_onset_cv",
            "mc_p11",
        ]

        def _pair_corr(df: pl.DataFrame, col_x: str, col_y: str) -> tuple:
            pair_df = df.select([col_x, col_y]).drop_nulls([col_x, col_y])
            n = pair_df.height
            if n < 3:
                return float("nan"), n
            x = pair_df[col_x].to_numpy()
            y = pair_df[col_y].to_numpy()
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                return float("nan"), n
            return float(spearmanr(x, y).statistic), n

        all_scores = []
        all_corr_tables = {}

        for var in self.cols:
            var_df = base.filter(pl.col("variable") == var)

            # Build pairwise correlation table for this variable
            corr_rows = []
            present_axes = [a for a in candidate_axes if a in var_df.columns]
            for i, ax_x in enumerate(present_axes):
                for ax_y in present_axes[i + 1:]:
                    corr, n_complete = _pair_corr(var_df, ax_x, ax_y)
                    corr_rows.append({
                        "axis_1": ax_x,
                        "axis_2": ax_y,
                        "corr": corr,
                        "abs_corr": float(abs(corr)) if not np.isnan(corr) else float("nan"),
                        "n_complete_cases": n_complete,
                    })

            corr_table = pl.DataFrame(corr_rows).sort(
                ["abs_corr", "n_complete_cases"], descending=[False, True], nulls_last=True
            )
            all_corr_tables[var] = corr_table

            # Select least-correlated axis pair
            valid_pairs = corr_table.filter(pl.col("corr").is_not_null())
            if valid_pairs.height > 0:
                selected = valid_pairs.row(0, named=True)
                axis_x = selected["axis_1"]
                axis_y = selected["axis_2"]
                selected_corr = float(selected["corr"])
            else:
                axis_x = "indicated_pct"
                axis_y = "gap_cv"
                selected_corr = float("nan")
                pretty_printing.rich_warning(
                    f"[{var}] Could not compute pairwise correlations for axis selection. "
                    f"Falling back to default axes: {axis_x} × {axis_y}."
                )

            complete_mask = pl.col(axis_x).is_not_null() & pl.col(axis_y).is_not_null()
            complete_df = var_df.filter(complete_mask)

            scores = var_df.clone()
            if complete_df.height < 2:
                scores = scores.with_columns(
                    pl.lit(axis_x).alias("axis_x"),
                    pl.lit(axis_y).alias("axis_y"),
                    pl.lit(None).cast(pl.Float64).alias("axis_pair_corr"),
                    pl.lit(None).cast(pl.Float64).alias("axis_x_median_threshold"),
                    pl.lit(None).cast(pl.Float64).alias("axis_y_median_threshold"),
                    pl.lit(None).cast(pl.Utf8).alias("imperfection_stratum"),
                )
            else:
                x_median = float(complete_df[axis_x].median())
                y_median = float(complete_df[axis_y].median())
                scores = self.assign_strata(scores, axis_x, axis_y, x_median, y_median)
                scores = scores.with_columns(
                    pl.lit(axis_x).alias("axis_x"),
                    pl.lit(axis_y).alias("axis_y"),
                    pl.lit(selected_corr).alias("axis_pair_corr"),
                    pl.lit(x_median).alias("axis_x_median_threshold"),
                    pl.lit(y_median).alias("axis_y_median_threshold"),
                )

            scores = scores.select([
                self.id_col, "variable",
                "indicated_pct",
                "gap_cv", "gap_qcod", "gap_burstiness_coeff",
                "gap_normalized_entropy", "gap_adherence_rate",
                "max_gap_fraction", "gap_onset_cv", "mc_p11",
                "axis_x", "axis_y", "axis_pair_corr",
                "axis_x_median_threshold", "axis_y_median_threshold",
                "imperfection_stratum",
            ])
            all_scores.append(scores)

            if self.renderer:
                total = len(scores)
                prevalence = (
                    scores.filter(pl.col("imperfection_stratum").is_not_null())
                    .group_by("imperfection_stratum")
                    .agg(pl.len().alias("n"))
                    .with_columns((pl.col("n") / total * 100).round(1).alias("pct"))
                    .sort("imperfection_stratum")
                )
                pretty_printing.rich_info(
                    f"[{var}] selected axes: {axis_x} × {axis_y} "
                    f"(corr={selected_corr:.3f})"
                )
                print(prevalence)

        self.results.iv_composite_scores = pl.concat(all_scores)
        self.results.iv_pairwise_correlations = all_corr_tables

        if save_results and path:
            self.results.iv_composite_scores.write_csv(path / "case_scores.csv")
            for var, tbl in all_corr_tables.items():
                tbl.write_csv(path / f"{var}_pairwise_axis_correlations.csv")

        return self

    def run(
        self,
        save_results: bool = True,
        gap_and_return_bins: list = None,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        window_size: timedelta = timedelta(minutes=5),
        window_location: str = "both",
        autocorrelation_lags: int = 20,
    ):
        """
        Run all analyses in sequence, then write a wide summary CSV
        (``intravariable_summary.csv``) consolidating key quantified results
        across all sub-analyses.

        Parameters:
            save_results (bool): Whether to save the results to files.
            gap_and_return_bins (list): Bins for gap and return analysis.
            bin_resolution_seconds (float): Bin width for dominant gap length detection.
            adherence_tolerance (float): Fractional tolerance for gap adherence rate.
            window_size (timedelta): Size of the temporal window for the analysis of observations around Imperfect values.
            window_location (str): Location of the temporal window ('before', 'after', 'both').
        """
        try:
            self.column_statistics(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in column analysis: {e}")
        try:
            self.gap_statistics(
                save_results=save_results,
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
            )
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in gap analysis: {e}")
        try:
            self.gap_returns(save_results=save_results, gap_and_return_bins=gap_and_return_bins)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in gap returns analysis: {e}")
        try:
            self.markov_chain_summary(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in Markov chain summary: {e}")
        try:
            self.windowed_significance(
                save_results=save_results,
                window_size=window_size,
                window_location=window_location,
            )
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in temporal analysis: {e}")
        try:
            self.date_time_statistics(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in datetime correlation analysis: {e}")
        try:
            self.autocorrelation(save_results=save_results, lags=autocorrelation_lags)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in autocorrelation analysis: {e}")

        # --- Final consolidated summary ---
        try:
            summary = self._build_summary()
            if summary is not None:
                if self.renderer:
                    pretty_printing.rich_info(
                        "Intravariable Summary — consolidated quantified results across all analyses."
                    )
                    print(summary)
                if save_results and self.save_path:
                    self.save_path.mkdir(parents=True, exist_ok=True)
                    summary.write_csv(self.save_path / "intravariable_summary.csv")
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error building intravariable summary: {e}")

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------

    def _build_summary(self) -> pl.DataFrame:
        """
        Assemble a wide summary DataFrame: one row per metric, one column per
        analyzed variable, plus a leading 'name' column.

        Rows (metric names):
            Column statistics:
                cs_indicated_count, cs_indicated_pct,
                cs_entity_pct_mean, cs_entity_pct_std,
                cs_entity_pct_min, cs_entity_pct_max,
                cs_n_entities_above_threshold
            Gap statistics:
                gs_gap_length_mean_seconds, gs_gap_length_median_seconds,
                gs_gap_length_min_seconds, gs_gap_length_max_seconds,
                gs_dominant_gap_seconds, gs_gap_adherence_rate,
                gs_gap_normalized_entropy, gs_gap_burstiness_coeff
            Gap returns / KW:
                gr_kw_statistic, gr_kw_p_value, gr_kw_effect_size,
                gr_kw_ci_lower, gr_kw_ci_upper
            Markov chain:
                mc_p_obs_to_obs, mc_p_obs_to_imp,
                mc_p_imp_to_obs, mc_p_imp_to_imp,
                mc_steady_state_observed, mc_steady_state_imperfect
            Windowed significance / MWU:
                ws_mwu_u_stat, ws_mwu_p_val, ws_mwu_significance,
                ws_mwu_effect_size,
                ws_mwu_mean_near_imperfect, ws_mwu_mean_outside
            Metadata (shared across all columns):
                meta_bin_resolution_seconds, meta_adherence_tolerance,
                meta_window_size_seconds, meta_window_location

        Returns:
            pl.DataFrame with columns ['name', col1, col2, ...], values as strings.
        """
        # metric_name -> {col: value}
        rows: dict[str, dict] = {}

        def _set(metric: str, col: str, value):
            rows.setdefault(metric, {})[col] = str(value) if value is not None else None

        # --- Column statistics ---
        if self.results.cs_overall_statistics is not None:
            for row in self.results.cs_overall_statistics.to_dicts():
                c = row["column"]
                if c not in self.cols:
                    continue
                _set("cs_indicated_count", c, row.get("indicated_count"))
                _set("cs_indicated_pct", c, row.get("indicated_pct"))

        if self.results.cs_case_level_statistics is not None:
            cs = self.results.cs_case_level_statistics
            for c in self.cols:
                pct_col = f"{c}_indicated_pct"
                thresh_col = next(
                    (col for col in cs.columns if col.startswith(f"{c}_above_") and col.endswith("_threshold")),
                    None,
                )
                if pct_col in cs.columns:
                    vals = cs[pct_col].drop_nulls()
                    _set("cs_entity_pct_mean", c, float(vals.mean()) if vals.len() > 0 else None)
                    _set("cs_entity_pct_std", c, float(vals.std()) if vals.len() > 0 else None)
                    _set("cs_entity_pct_min", c, float(vals.min()) if vals.len() > 0 else None)
                    _set("cs_entity_pct_max", c, float(vals.max()) if vals.len() > 0 else None)
                if thresh_col and thresh_col in cs.columns:
                    _set("cs_n_entities_above_threshold", c, int(cs[thresh_col].sum()))

        # --- Gap statistics ---
        if self.results.gs_gaps_df:
            for c in self.cols:
                gaps_frame = self.results.gs_gaps_df.get(c)
                col_gaps = gaps_frame["time_length"].drop_nulls() if gaps_frame is not None else pl.Series("time_length", [], dtype=pl.Float64)
                if col_gaps.len() > 0:
                    _set("gs_gap_length_mean_seconds", c, float(col_gaps.mean()))
                    _set("gs_gap_length_median_seconds", c, float(col_gaps.median()))
                    _set("gs_gap_length_min_seconds", c, float(col_gaps.min()))
                    _set("gs_gap_length_max_seconds", c, float(col_gaps.max()))

        for c in self.cols:
            dom = self.results.gs_gap_dominant.get(c)
            if dom is not None:
                d = dom.row(0, named=True)
                _set("gs_dominant_gap_seconds", c, d.get("dominant_gap_seconds"))
                _set("gs_gap_adherence_rate", c, d.get("gap_adherence_rate"))
                _set("gs_gap_normalized_entropy", c, d.get("gap_normalized_entropy"))
            bu = self.results.gs_gap_burstiness.get(c)
            if bu is not None:
                b = bu.row(0, named=True)
                _set("gs_gap_burstiness_coeff", c, b.get("gap_burstiness_coeff"))

        # --- Gap returns / KW ---
        for c, kw in self.results.gr_gap_kruskal.items():
            if kw is None:
                continue
            _set("gr_kw_statistic", c, kw.get("statistic"))
            _set("gr_kw_p_value", c, kw.get("p_value"))
            _set("gr_kw_effect_size", c, kw.get("effect_size"))
            _set("gr_kw_ci_lower", c, kw.get("ci_lower"))
            _set("gr_kw_ci_upper", c, kw.get("ci_upper"))

        # --- Markov chain ---
        for c, mc in self.results.mc_markov_summary.items():
            if mc is None:
                continue
            tm = mc.get("transition_matrix")
            ss = mc.get("steady_state")
            if tm is not None:
                _set("mc_p_obs_to_obs", c, float(tm[0, 0]))
                _set("mc_p_obs_to_imp", c, float(tm[0, 1]))
                _set("mc_p_imp_to_obs", c, float(tm[1, 0]))
                _set("mc_p_imp_to_imp", c, float(tm[1, 1]))
            if ss is not None:
                _set("mc_steady_state_observed", c, float(ss[0]))
                _set("mc_steady_state_imperfect", c, float(ss[1]))

        # --- Windowed significance / MWU ---
        if self.results.ws_mwu_result is not None and not self.results.ws_mwu_result.is_empty():
            mwu_df = self.results.ws_mwu_result
            for row in mwu_df.to_dicts():
                c = row.get("column")
                if c not in self.cols:
                    continue
                _set("ws_mwu_u_stat", c, row.get("u_stat"))
                _set("ws_mwu_p_val", c, row.get("p_val"))
                _set("ws_mwu_significance", c, row.get("significance"))
                _set("ws_mwu_effect_size", c, row.get("effect_size"))
                _set("ws_mwu_mean_near_imperfect", c, row.get("mean_group_1"))
                _set("ws_mwu_mean_outside", c, row.get("mean_group_2"))

        # --- Metadata (same value broadcast across all columns) ---
        for metric, value in [
            ("meta_bin_resolution_seconds", self.gs_bin_resolution_seconds),
            ("meta_adherence_tolerance", self.gs_adherence_tolerance),
            (
                "meta_window_size_seconds",
                self.ws_window_size.total_seconds() if self.ws_window_size else None,
            ),
            ("meta_window_location", self.ws_window_location),
        ]:
            for c in self.cols:
                _set(metric, c, value)

        if not rows:
            return None

        names = list(rows.keys())
        data = {"name": names}
        for c in self.cols:
            data[c] = [rows[m].get(c) for m in names]

        return pl.DataFrame(data)

    def generate_html_report(
        self, report_path: str = "intravariable_report.html", title: str = None
    ):
        """Generates an HTML report from the analysis results."""
        if not self.save_path:
            pretty_printing.rich_warning("⚠️ Cannot generate report without a save_path.")
            return
        else:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        from imperfekt.analysis.intravariable.html_report_generator import IntravariableHTMLReportGenerator

        # Create report generator and generate report
        report_generator = IntravariableHTMLReportGenerator(self)
        full_report_path = report_generator.generate_report(report_path, title=title)

        pretty_printing.rich_info(f"✅ Report generated at [green]{full_report_path}[/green]")

    def _path(self, subpath: str) -> Path:
        """Generates a full path for saving results."""
        if self.save_path:
            return self.save_path / subpath
        return None

    def _generate_clock_no_col(self):
        """Generates the clock_no_col if it does not exist in the DataFrame."""
        self.df = self.df.sort([self.id_col, self.clock_col])
        if self.clock_no_col not in self.df.columns:
            self.df = self.df.with_columns(
                pl.cum_count(self.id_col).over(self.id_col).alias(self.clock_no_col)
            )


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Verification dataset — all expected values are hand-calculable.
    #
    # Layout: 2 patients (A, B), 10 observations each, timestamps every 60 s.
    # clock_no is the 0-based integer index within each patient.
    #
    # hr missingness (mask = 1 where None):
    #   patient A: indices 2, 3, 7  → 3/10 = 30%
    #     - gap at idx 2-3: 2 consecutive missing → gap_length = 3*60 = 180 s
    #       (from obs at idx 1 to obs at idx 4, spanning 2 missing clock steps)
    #     - gap at idx 7:   1 missing            → gap_length = 2*60 = 120 s
    #   patient B: indices 1, 5, 6  → 3/10 = 30%
    #     - gap at idx 1:   1 missing            → gap_length = 2*60 = 120 s
    #     - gap at idx 5-6: 2 consecutive missing → gap_length = 3*60 = 180 s
    #   hr true gaps (count_clock_no > 0): lengths = [180, 120, 120, 180] s
    #     mean = 150 s, median = 150 s, min = 120 s, max = 180 s
    #
    # bp missingness:
    #   patient A: index 5   → 1/10 = 10%  (gap_length = 2*60 = 120 s)
    #   patient B: index 8   → 1/10 = 10%  (gap_length = 2*60 = 120 s)
    #   bp true gaps: lengths = [120, 120] s  → mean = median = 120 s
    #
    # Markov chain for hr (A): sequence = 0,0,1,1,0,0,0,1,0,0
    #   transitions 0→0: 5, 0→1: 2, 1→0: 1, 1→1: 1
    #   P00 = 5/7, P01 = 2/7, P10 = 1/2, P11 = 1/2
    #   steady state: π1 = P01/(P01+P10) = (2/7)/(2/7+1/2) = 4/11 ≈ 0.364
    #
    # cs_indicated_pct for hr: (3+3)/(10+10) = 6/20 = 30%
    # cs_indicated_pct for bp: (1+1)/(10+10) = 2/20 = 10%
    # -----------------------------------------------------------------------

    from datetime import datetime

    base_A = datetime(2023, 1, 1, 8, 0, 0)
    base_B = datetime(2023, 1, 2, 8, 0, 0)
    times_A = [base_A + timedelta(seconds=60 * i) for i in range(10)]
    times_B = [base_B + timedelta(seconds=60 * i) for i in range(10)]

    # hr: None at A[2,3,7] and B[1,5,6]
    hr_A = [70.0, 72.0, None, None, 68.0, 74.0, 71.0, None, 69.0, 73.0]
    hr_B = [65.0, None, 66.0, 67.0, 64.0, None, None, 63.0, 68.0, 65.0]

    # bp: None at A[5] and B[8]
    bp_A = [120.0, 118.0, 122.0, 119.0, 121.0, None, 117.0, 123.0, 120.0, 119.0]
    bp_B = [115.0, 116.0, 114.0, 117.0, 115.0, 118.0, 116.0, 119.0, None, 114.0]

    rows = (
        [("A", times_A[i], hr_A[i], bp_A[i], i) for i in range(10)]
        + [("B", times_B[i], hr_B[i], bp_B[i], i) for i in range(10)]
    )

    df = pl.DataFrame(
        rows,
        schema=["patient", "time", "heartrate", "blood_pressure", "clock_no"],
        orient="row",
    )

    intravariable_imperfection = IntravariableImperfection(
        df=df,
        id_col="patient",
        clock_col="time",
        clock_no_col="clock_no",
        renderer=None,
        save_path=Path("test"),
        cols=["heartrate", "blood_pressure"],
    )

    intravariable_imperfection.run(save_results=True)

    print("\n=== intravariable_summary.csv ===")
    print(pl.read_csv("test/intravariable_summary.csv"))

    # Generate HTML report
    intravariable_imperfection.generate_html_report("test_report.html")
