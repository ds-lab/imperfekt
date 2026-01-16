import traceback
from datetime import timedelta
from pathlib import Path

import polars as pl

from imperfekt.analysis.intravariable import (
    autocorrelation,
    column_statistics,
    date_time_statistics,
    gap_statistics,
    markov_chain_summary,
    windowed_significance,
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
        self.gr_gap_returns: pl.DataFrame = None
        self.gr_gap_kruskal: dict = {}
        self.mc_markov_summary: dict = {}
        self.ac_autocorrelation: dict = {}
        self.ws_observations_around_indicated: dict = {}
        self.ws_mwu_result: pl.DataFrame = None
        self.dt_date_time_statistics: dict = {}
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
        self.results = IntravariableResults()

        # Analysis parameters init
        self.ws_window_size = None
        self.ws_window_location = None
        self.gr_gap_and_return_bins = None
        self.ac_autocorrelation_lags = None

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

    def gap_statistics(self, save_results: bool = True):
        """Analyzes gaps and observation lengths.

        Parameters:
            save_results (bool): Whether to save the results to files.
            gap_and_return_bins (list): Bins for gap and return analysis.

        Returns:
            self: Returns the instance of IntravariableImperfection for method chaining. Updates the following attributes:
                results.gs_gaps_observation_runs (pl.DataFrame): DataFrame containing gaps and observation lengths.
                results.gap_returns (pl.DataFrame): DataFrame containing gaps and return values.
        """
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
                P=self.results.mc_markov_summary[c]["transition_matrix"],
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
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        for c in self.cols:
            (path / c).mkdir(parents=True, exist_ok=True)
            self.results.ac_autocorrelation[c] = autocorrelation.acf(
                mask_df=self.mask,
                col=c,
                id_col=self.id_col,
                clock_no_col=self.clock_no_col,
                max_lag=lags,
                seasonal_trend_decomposition=seasonal_trend_decomposition,
                stl_period=stl_period,
                save_path=self._path(f"{path}/{c}"),
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
                save_path=self._path(f"{path}/{c}/{c}_autocorrelation_plot.png"),
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

    def run(
        self,
        save_results: bool = True,
        gap_and_return_bins: list = None,
        window_size: timedelta = timedelta(minutes=5),
        window_location: str = "both",
        autocorrelation_lags: int = 20,
    ):
        """
        Run all analyses in sequence.

        Parameters:
            save_results (bool): Whether to save the results to files.
            gap_and_return_bins (list): Bins for gap and return analysis.
            window_size (timedelta): Size of the temporal window for the analysis of observations around Imperfect values.
            window_location (str): Location of the temporal window ('before', 'after', 'both').
            visualize_gaps_and_obs_lengths (bool): Whether to visualize gaps and observation lengths.
            visualize_returns (bool): Whether to visualize gap and return values.
        """
        try:
            self.column_statistics(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in column analysis: {e}")
        try:
            self.gap_statistics(save_results=save_results)
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

        from .html_report_generator import IntravariableHTMLReportGenerator

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
    df = pl.DataFrame(
        {
            "patient": ["a", "a", "a", "a", "a", "c", "c"],
            "time": [
                "2023-01-01 00:00:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:10:00",
                "2023-01-01 00:15:00",
                "2023-01-01 00:20:00",
                "2023-02-02 00:25:00",
                "2023-02-01 00:30:00",
            ],
            "heartrate": [60, None, 70, 65, None, 80, None],
            "blood_pressure": [120, 130, None, None, None, 135, None],
        }
    ).with_columns(
        [
            pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )

    intravariable_imperfection = IntravariableImperfection(
        df=df,
        id_col="patient",
        clock_col="time",
        clock_no_col="time_no",
        renderer=None,
        save_path="test",
        cols=["heartrate", "blood_pressure"],
    )

    intravariable_imperfection.run(save_results=True)

    # Generate HTML report
    intravariable_imperfection.generate_html_report("test_report.html")
