import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl

from imperfekt.analysis.intervariable import (
    asymmetric_analysis,
    marmnar,
    mcar,
    row_statistics,
    symmetric_correlation,
)
from imperfekt.analysis.utils import masking, pretty_printing, visualization_utils
from imperfekt.config.global_settings import VITALS


class IntervariablePlots:
    def __init__(self):
        self.rs_case_level_histogram: go.Figure | plt.Figure = None
        self.rs_case_level_boxplot: go.Figure | plt.Figure = None
        self.mcar_upset_plot: go.Figure = None
        self.sc_lag_scatter_plot: dict[str, go.Figure | plt.Figure] = {}
        self.sc_correlation_heatmap: go.Figure | plt.Figure = None
        self.sc_correlation_dendogram: go.Figure | plt.Figure = None
        self.ac_multi_histogram: dict[str, go.Figure | plt.Figure] = {}
        self.ac_multi_boxplot: dict[str, go.Figure | plt.Figure] = {}
        self.ac_lag_scatter_plot: dict[str, go.Figure | plt.Figure] = {}


class IntervariableResults:
    def __init__(self):
        # Analytical results
        self.rs_overall_statistics: pl.DataFrame = None
        self.rs_case_level_statistics: pl.DataFrame = None
        self.rs_empty_statistics: pl.DataFrame = None
        self.rs_empty_case_level_statistics: pl.DataFrame = None
        self.mcar_results: pl.DataFrame = None
        self.mar_mnar_results: pl.DataFrame = None
        self.sc_symmetric_correlation: pl.DataFrame = None
        self.sc_chi2_intervariable_matrix: pl.DataFrame = None
        self.sc_symmetric_crosscorrelation: dict[str, pl.DataFrame] = {}
        self.ac_asymmetric_statistical_results: dict[str, pl.DataFrame] = {}
        self.ac_asymmetric_crosscorrelation: dict[str, pl.DataFrame] = {}
        # Plots
        self.plots = IntervariablePlots()


class IntervariableImperfection:
    """
    A class to analyze intervariable "imperfection" on a Polars DataFrame. Imperfection refers to missingness, noise etc. things that can be indicated using a binary mask.
    It provides methods to analyze row completeness, co-imperfection, cross-correlation and asymmetric observations on imperfect data.
    It also includes the visualization of imperfection over time per subject.
    The results can be saved to a specified path.
    Attributes:
        df (pl.DataFrame): The DataFrame to analyze.
        imperfection (str): The type of imperfection to analyze (e.g., "missingness").
        mask_df (pl.DataFrame): A DataFrame with binary values indicating imperfection (1 for missing/noisy/indicated, 0 for present/normal/expected). Can be used for custom imperfection analysis.
        id_col (str): The column name for the ID.
        clock_col (str): The column name for the clock.
        clock_no_col (str): The column name for the clock number.
        cols (list): The list of columns to analyze for imperfection.
        alpha (float): The significance level for hypothesis testing.
        save_path (Path): The path to save the results.
        renderer (str): The renderer to use for visualizations.
    Methods:
        __init__(df, id_col, clock_col, clock_no_col, cols, save_path, renderer):
            Initializes the IntervariableImperfection class with the given DataFrame and parameters.
        row_statistics(save_results):
            Analyzes row completeness in the DataFrame, including all-null rows and row-wise imperfection statistics.
        symmetric_correlation(save_results):
            Analyzes co-imperfection in the DataFrame, including pairwise imperfection statistics and correlation.
        mcar_test(save_results):
            Performs Little's MCAR test on the DataFrame to check if the imperfect data is completely at random (MCAR).
        mar_mnar_test(save_results):
            Performs MAR/MNAR test on the DataFrame to check if the missing data is missing at random (MAR) or not missing at random (MNAR).
        symmetric_lagged_cross_correlation(save_results, max_lag):
            Analyzes lagged cross-correlation in the DataFrame, including the calculation of cross-correlations between pairs of columns.
        asymmetric_correlation(save_results, max_lag):
            Analyzes asymmetric missing-observation correlations, comparing missing patterns of one variable with observed values of another.
        asymmetric_lagged_cross_correlation(save_results, max_lag):
            Analyzes asymmetric lagged cross-correlation between pairs of columns, focusing on missing vs observed values.
        visualize_missingness(save_results):
            Visualizes the missingness over time per subject, showing the percentage of imperfect values for each
            variable over time.
        run(save_results,lagged_crosscorr_max_lag,asymmetric_analysis_window_size,asymmetric_analysis_window_location,
                asymmetric_analysis_window_exclude_imperfect_ts,asymmetric_analysis_window_get_unique):
            Runs all analyses in the IntervariableImperfection class, including row completeness, co-imperfection,
            Little's MCAR test, asymmetric observations on imperfect data, lagged cross-correlation,
            and visualization of missingness.

        _path(sub_path):
            Helper method to generate a full path for saving results.
        _generate_clock_no_col():
            Helper method that generates a clock number column if it does not exist in the DataFrame.
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

        # Binary indicator mask for imperfection
        self.imperfection = imperfection
        if imperfection == "missingness" or mask_df is None:
            self.mask = masking.create_missingness_mask(
                df=self.df,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                cols=self.cols,
            )
        else:
            if mask_df is not None:
                self.mask = mask_df
            else:
                raise ValueError(
                    f"Unsupported imperfection type: {imperfection}. Supported types: 'missingness'."
                )

        self.alpha = alpha
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
        self.results = IntervariableResults()

        # Analysis parameters init
        self.sc_max_lag: int = 10
        self.ac_max_lag: int = 10

    def row_statistics(self, save_results: bool = True, analyze_all_null_rows: bool = True):
        """Analyzes row completeness in the DataFrame, including all-null rows and row-wise imperfection statistics.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.

        Returns:
            self: Returns the instance of IntervariableImperfection for method chaining.
        """
        new_path_level_name = "row_statistics"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        if analyze_all_null_rows:
            # 1. All-null rows overall
            self.results.rs_empty_statistics = row_statistics.analyze_all_null_rows(
                mask_df=self.mask,
                cols=self.cols,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                save_path=self._path(f"{new_path_level_name}/all_null_rows_stats.csv"),
                save_results=save_results,
            )
            if self.renderer:
                pretty_printing.rich_info(
                    f"All null rows: {self.results.rs_empty_statistics[0]}, Percentage: {self.results.rs_empty_statistics[1]:.4f}%\n"
                )

            # 2. All-null rows per ID
            self.results.rs_empty_case_level_statistics = (
                row_statistics.analyze_all_null_rows_per_id(
                    mask_df=self.mask,
                    cols=self.cols,
                    id_col=self.id_col,
                    clock_col=self.clock_col,
                    clock_no_col=self.clock_no_col,
                    save_path=self._path(f"{new_path_level_name}/all_null_rows_per_id_stats.csv"),
                    save_results=save_results,
                )
            )
            if self.renderer:
                print("All null rows per ID:")
                print(
                    self.results.rs_empty_case_level_statistics.describe(interpolation="linear"),
                    "\n",
                )

            visualization_utils.plot_histogram(
                self.results.rs_empty_case_level_statistics,
                x="null_vitals_pct",
                title="All Null Vitals Percentage (per ID)",
                xaxis_title="All Null Vitals Percentage",
                yaxis_title="#Cases",
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/null_vitals_pct_per_id_histogram.png"),
                library=self.plot_library,
                save_results=save_results,
            )
            visualization_utils.plot_boxplot(
                self.results.rs_empty_case_level_statistics,
                y="null_vitals_pct",
                title="All Null Vitals Boxplot (per ID)",
                yaxis_title="All Null Vitals Percentage",
                boxpoints="false",
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/null_vitals_pct_per_id_boxplot.png"),
                save_results=save_results,
                library=self.plot_library,
            )

        # 3. Row-wise imperfect-variable completeness
        self.results.rs_overall_statistics = row_statistics.analyze_row_imperfection(
            mask_df=self.mask,
            cols=self.cols,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
            save_path=self._path(f"{new_path_level_name}/row_completeness_stats.csv"),
            save_results=save_results,
        )

        if self.renderer:
            print("Row Completeness Stats:")
            print(
                self.results.rs_overall_statistics.describe(interpolation="linear"),
                "\n",
            )

        visualization_utils.plot_histogram(
            self.results.rs_overall_statistics,
            x="indicated_vars_pct",
            title="Imperfect Variables Per Row Percentage",
            xaxis_title="Imperfect Variables Percentage",
            yaxis_title="#Cases",
            renderer=self.renderer,
            save_path=self._path(f"{new_path_level_name}/indicated_vars_pct_histogram.png"),
            save_results=save_results,
            library=self.plot_library,
        )

        # 4. Imperfect-variable stats per ID
        self.results.rs_case_level_statistics = row_statistics.analyze_row_imperfection_per_id(
            mask_df=self.mask,
            cols=self.cols,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
            save_path=self._path(f"{new_path_level_name}/row_completeness_per_id_stats.csv"),
            save_results=save_results,
        )

        if self.renderer:
            print("Row Completeness Stats (per ID):")
            print(
                self.results.rs_case_level_statistics.describe(interpolation="linear"),
                "\n",
            )

        case_level_histogram = visualization_utils.plot_histogram(
            self.results.rs_case_level_statistics,
            x="avg_indicated_vars_pct",
            title="Average Imperfect Variables Per Row Percentage (per ID)",
            xaxis_title="Average Imperfect Variables Percentage",
            yaxis_title="#Cases",
            renderer=self.renderer,
            save_path=self._path(
                f"{new_path_level_name}/avg_indicated_vars_pct_per_id_histogram.png"
            ),
            save_results=save_results,
            library=self.plot_library,
        )
        case_level_boxplot = visualization_utils.plot_boxplot(
            self.results.rs_case_level_statistics,
            y="avg_indicated_vars_pct",
            title="Average Imperfect Variables Per Row Boxplot (per ID)",
            yaxis_title="Average Imperfect Variables Percentage",
            renderer=self.renderer,
            save_path=self._path(
                f"{new_path_level_name}/avg_indicated_vars_pct_per_id_boxplot.png"
            ),
            save_results=save_results,
            library=self.plot_library,
        )

        self.results.plots.rs_case_level_histogram = case_level_histogram
        self.results.plots.rs_case_level_boxplot = case_level_boxplot

        return self

    def mcar_test(self, save_results: bool = True):
        """Performs Little's MCAR test on the DataFrame to check if the imperfect data is completely at random (MCAR).
        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.

        Returns:
            dict: A dictionary containing the results of Little's MCAR test.
        """
        new_path_level_name = "mcar_test"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        # Perform Little's MCAR test for imperfection association across variables
        self.results.mcar_results = mcar.mcar_test(
            df=self.df,
            mask_df=self.mask,
            id_col=self.id_col,
            clock_col=self.clock_col,
            cols=self.cols,
            alpha=self.alpha,
        )

        if self.renderer:
            print(
                "Little's MCAR Test Results - Observed vs. Imperfect Association across Variables:"
            )
            print("What patterns occur?")
            print(self.results.mcar_results["patterns"])
            print("Little's MCAR Test Summary:")
            print(
                "Across all variables together, is there any systematic structure to who is imperfect?"
            )
            print(self.results.mcar_results["little_mcar_test"])

        self.results.plots.mcar_upset_plot = mcar.upset(
            mask_df=self.mask,
            cols=self.cols,
            renderer=self.renderer,
            save_path=self._path(f"{new_path_level_name}/co_imperfection_upset_plot.png"),
            save_results=save_results,
        )

        if self.save_path and save_results:
            # Save the results to a CSV file if a path is provided
            patterns_csv = self.results.mcar_results["patterns"].write_csv(None, separator="\t")
            with open(self._path(f"{new_path_level_name}/little_mcar_test_results.csv"), "w") as f:
                f.write("mcar_test:\n")
                f.write(str(self.results.mcar_results["little_mcar_test"]) + "\n")
                f.write("Patterns:\n")
                f.write(patterns_csv)

        return self

    def mar_mnar_test(
        self,
        save_results: bool = True,
        impute_strategy: str = VITALS.IMPUTATION_VALUE_MAR_MNAR_TEST,
        standardize: bool = VITALS.STANDARDIZE_MAR_MNAR_TEST,
    ):
        """
        Performs MAR/MNAR test on the DataFrame to check if the imperfect data is missing at random (MAR) or not at random (MNAR).
        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
            impute_strategy (str): The imputation strategy to use. Defaults to "mean". Options are: zero, mean, median, ffill_within_id.
            standardize (bool): Whether to standardize the data. Defaults to True.
        Returns:
            self: Returns the instance of IntervariableImperfection for method chaining.
        """
        new_path_level_name = "mar_mnar_test"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        self.results.mar_mnar_results = marmnar.temporal_mar_mnar_test(
            self.df,
            self.mask,
            self.id_col,
            self.clock_col,
            self.cols,
            self.alpha,
            impute_strategy=impute_strategy,
            standardize=standardize,
            save_path=self._path(new_path_level_name),
            save_results=save_results,
        )

        if self.save_path and save_results:
            self.results.mar_mnar_results.write_csv(
                self._path(f"{new_path_level_name}/mar_mnar_test_results.csv")
            )

        return self

    def symmetric_correlation(
        self,
        save_results: bool = True,
        heatmap: bool = True,
        dendrogram: bool = True,
        chi_squared_test: bool = True,
    ):
        """Analyzes co-imperfection in the DataFrame, including pairwise imperfection statistics and correlation.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
            upset_plot (bool): Whether to create an UpSet plot for co-imperfection. Defaults to True.
            heatmap (bool): Whether to create a heatmap for co-imperfection correlation. Defaults to True.
            dendrogram (bool): Whether to create a dendrogram for co-imperfection correlation. Defaults to True.
            chi_squared_test (bool): Whether to perform Chi-squared test for imperfection association between variables. Defaults to True.

        Returns:
            self: Returns the instance of IntervariableImperfection for method chaining.
        """
        new_path_level_name = "symmetric"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        # Correlation of imperfection, plot heatmap and dendrogram
        if heatmap or dendrogram:
            self.results.sc_symmetric_correlation = symmetric_correlation.corr_matrix(
                self.mask,
                id_col=self.id_col,
                clock_no_col=self.clock_no_col,
                clock_col=self.clock_col,
            )
            if heatmap:
                self.results.plots.sc_correlation_heatmap = (
                    symmetric_correlation.co_imperfection_heatmap(
                        corr_matrix=self.results.sc_symmetric_correlation,
                        renderer=self.renderer,
                        save_path=self._path(f"{new_path_level_name}/co_imperfection_heatmap.png"),
                        save_results=save_results,
                    )
                )

            if dendrogram:
                self.results.plots.sc_correlation_dendogram = (
                    symmetric_correlation.co_imperfection_dendogram(
                        corr_matrix=self.results.sc_symmetric_correlation,
                        renderer=self.renderer,
                        save_path=self._path(
                            f"{new_path_level_name}/co_imperfection_dendrogram.png"
                        ),
                        save_results=save_results,
                    )
                )

        # Chi-squared test for imperfection association between variables
        if chi_squared_test:
            self.results.sc_chi2_intervariable_matrix = (
                symmetric_correlation.chi2_intervariable_imperfection_matrix(
                    mask_df=self.mask,
                    cols=self.cols,
                    save_path=self._path(f"{new_path_level_name}/chi2_intervariable.csv"),
                    save_results=save_results,
                )
            )
            if self.renderer:
                print(self.results.sc_chi2_intervariable_matrix)

        return self

    def symmetric_lagged_cross_correlation(self, save_results: bool = True, max_lag: int = 10):
        """Analyzes lagged cross-correlation in the DataFrame, including the calculation of cross-correlations between pairs of columns.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
            max_lag (int): The maximum lag to consider for cross-correlation. Defaults to 10.

        Returns:
            self: Returns the instance of IntervariableImperfection for method chaining.
        """
        self.sc_max_lag = max_lag
        new_path_level_name = "symmetric"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        for col_x in self.cols:
            for col_y in self.cols:
                if col_x == col_y:
                    continue
                try:
                    pair_key = f"{col_x}_vs_{col_y}"
                    self.results.sc_symmetric_crosscorrelation[pair_key] = (
                        symmetric_correlation.lagged_cross_correlation(
                            mask_df=self.mask,
                            col_x=col_x,
                            col_y=col_y,
                            id_col=self.id_col,
                            clock_no_col=self.clock_no_col,
                            max_lag=max_lag,
                        )
                    )
                    if self.renderer:
                        print(f"Cross-correlation between {col_x} and {col_y}:")
                        print(
                            "Lags:",
                            self.results.sc_symmetric_crosscorrelation[pair_key]["lag"],
                        )
                        print(
                            "Cross-correlations:",
                            self.results.sc_symmetric_crosscorrelation[pair_key]["crosscorr"],
                            "\n",
                        )

                    self.results.plots.sc_lag_scatter_plot[pair_key] = (
                        visualization_utils.plot_scatter(
                            x=self.results.sc_symmetric_crosscorrelation[pair_key]["lag"],
                            y=self.results.sc_symmetric_crosscorrelation[pair_key]["crosscorr"],
                            title=f"{col_x} - {col_y} Cross-Correlation",
                            xaxis_title="Lags",
                            yaxis_title="Phi coefficient",
                            renderer=self.renderer,
                            library=self.plot_library,
                            save_path=self._path(
                                f"{new_path_level_name}/{col_x}_vs_{col_y}_cross_correlation.png"
                            ),
                            save_results=save_results,
                        )
                    )

                except ValueError as e:
                    print(f"Skipping {col_x} vs {col_y}: {e}")

        if save_results and self.save_path:
            all_ccfs = pl.DataFrame(
                {"col_pair": [], "lag": [], "crosscorr": []},
                schema=pl.Schema(
                    {
                        "col_pair": pl.String(),
                        "lag": pl.Int64(),
                        "crosscorr": pl.Float64(),
                    }
                ),
            )
            for k, v in self.results.sc_symmetric_crosscorrelation.items():
                all_ccfs = pl.concat(
                    [
                        all_ccfs,
                        pl.DataFrame(
                            {
                                "col_pair": k,
                                "lag": v["lag"],
                                "crosscorr": v["crosscorr"],
                            }
                        ),
                    ],
                    how="vertical",
                )
            all_ccfs.write_csv(self._path(f"{new_path_level_name}/symmetric_crosscorrelation.csv"))

        return self

    def asymmetric_correlation(self, save_results: bool = True):
        """Analyzes asymmetric missing-observation correlations, comparing missing patterns of one variable with observed values of another.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
            max_lag (int): Maximum lag for asymmetric correlation analysis. Defaults to 10.

        Returns:
            self: Returns the instance of IntervariableImperfection for method chaining.
        """
        new_path_level_name = "asymmetric"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        # Ensure datetime format for temporal analysis
        numeric_cols = self.df.select(pl.selectors.numeric()).columns
        analysis_cols = [c for c in self.cols if c in numeric_cols]

        if not analysis_cols:
            pretty_printing.rich_warning("No numeric columns found for asymmetric analysis.")
            return self

        # Detailed statistical comparison for each missing-observation pair (asymmetric correlation and MWU)
        for indicated_col in analysis_cols:
            for obs_col in analysis_cols:
                if indicated_col == obs_col:
                    continue
                if self.renderer:
                    print(f"Analyzing: Missing {indicated_col} vs Observed {obs_col}")

                try:
                    # Comprehensive statistical comparison
                    stats_result, histogram_fig, boxplot_fig = (
                        asymmetric_analysis.asymmetric_statistical_comparison(
                            df=self.df,
                            mask_df=self.mask,
                            indicated_col=indicated_col,
                            observation_col=obs_col,
                            id_col=self.id_col,
                            clock_col=self.clock_col,
                            clock_no_col=self.clock_no_col,
                            statistical_tests=True,
                            save_path=self._path(new_path_level_name),
                            save_results=save_results,
                            renderer=self.renderer,
                            plot_library=self.plot_library,
                        )
                    )

                    pair_key = f"{indicated_col}_imperfect_vs_{obs_col}_observed"
                    self.results.ac_asymmetric_statistical_results[pair_key] = stats_result
                    self.results.plots.ac_multi_histogram[pair_key] = histogram_fig
                    self.results.plots.ac_multi_boxplot[pair_key] = boxplot_fig

                    if self.renderer:
                        print(f"  Correlation (lag=0): {stats_result['correlation']:.4f}")
                        if "conditional_stats" in stats_result:
                            print(
                                f"  When {indicated_col} missing: {obs_col} mean = {stats_result['conditional_stats']['missing_mean']:.3f}"
                            )
                            print(
                                f"  When {indicated_col} observed: {obs_col} mean = {stats_result['conditional_stats']['observed_mean']:.3f}"
                            )

                        if "mannwhitney" in stats_result:
                            print(
                                f"  Mann-Whitney p-value: {stats_result['mannwhitney']['p_value']:.4f}, significant = {stats_result['mannwhitney']['p_value'] < self.alpha}"
                            )
                        print()

                except Exception as e:
                    if self.renderer:
                        print(f"  Error analyzing {indicated_col} vs {obs_col}: {e}")
                    continue

        return self

    def asymmetric_lagged_cross_correlation(self, save_results: bool = True, max_lag: int = 10):
        """Analyzes asymmetric lagged cross-correlation between pairs of columns, focusing on missing vs observed values.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
            max_lag (int): The maximum lag to consider for cross-correlation. Defaults to 10.

        Returns:
            self: Returns the instance of IntervariableImperfection for method chaining.
        """
        self.ac_max_lag = max_lag
        new_path_level_name = "asymmetric"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        # Ensure datetime format for temporal analysis
        numeric_cols = self.df.select(pl.selectors.numeric()).columns
        analysis_cols = [c for c in self.cols if c in numeric_cols]

        # Generate lagged correlation
        for indicated_col in analysis_cols:
            for obs_col in analysis_cols:
                if indicated_col == obs_col:
                    continue

                try:
                    # Generate lagged correlation for visualization
                    pair_key = f"{indicated_col}_imperfect_vs_{obs_col}_observed"
                    self.results.ac_asymmetric_crosscorrelation[pair_key] = (
                        asymmetric_analysis.asymmetric_missing_observation_lagged_correlation(
                            df=self.df,
                            mask_df=self.mask,
                            indicated_col=indicated_col,
                            observation_col=obs_col,
                            id_col=self.id_col,
                            clock_no_col=self.clock_no_col,
                            max_lag=max_lag,
                        )
                    )

                    # Visualize lagged correlations
                    self.results.plots.ac_lag_scatter_plot[pair_key] = (
                        visualization_utils.plot_scatter(
                            x=self.results.ac_asymmetric_crosscorrelation[pair_key]["lag"],
                            y=self.results.ac_asymmetric_crosscorrelation[pair_key]["crosscorr"],
                            title=f"Asymmetric Correlation: Missing {indicated_col} vs Observed {obs_col}",
                            xaxis_title="Lag",
                            yaxis_title="Rank-Biserial Correlation",
                            renderer=self.renderer,
                            save_path=self._path(
                                f"{new_path_level_name}/lagged_correlation_{indicated_col}_missing_vs_{obs_col}_observed.png"
                            ),
                            save_results=save_results,
                            library=self.plot_library,
                        )
                    )
                except Exception as e:
                    if self.renderer:
                        print(f"    Could not generate visualization: {e}")

        if save_results and self.save_path:
            # Save all asymmetric correlation results to a single CSV file
            all_ccfs = pl.DataFrame(
                {"col_pair": [], "lag": [], "crosscorr": []},
                schema=pl.Schema(
                    {
                        "col_pair": pl.String(),
                        "lag": pl.Int64(),
                        "crosscorr": pl.Float64(),
                    }
                ),
            )
            for k, v in self.results.ac_asymmetric_crosscorrelation.items():
                # replace all nulls in v with 0
                all_ccfs = pl.concat(
                    [
                        all_ccfs,
                        pl.DataFrame(
                            {
                                "col_pair": k,
                                "lag": v["lag"],
                                "crosscorr": v["crosscorr"],
                            }
                        ),
                    ],
                    how="vertical",
                )
            all_ccfs.write_csv(self._path(f"{new_path_level_name}/asymmetric_crosscorrelation.csv"))
        return self

    def visualize_missingness(self, save_results: bool = True) -> None:
        """Visualizes the missingness over time per subject, showing the percentage of imperfect values for each variable over time.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
        Returns:
            None: This method does not return anything, it saves the visualizations to files.
        """
        new_path_level_name = "visualize_missingness"
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        matrix_full = masking.create_missingness_mask_long_table(
            df=self.df,
            cols=self.cols,
            id_col=self.id_col,
            clock_no_col=self.clock_no_col,
        )
        masking.plot_missingness_mask(
            matrix_full,
            title="Overall Missingness Heatmap",
            id_col=self.id_col,
            clock_no_col=self.clock_no_col,
            renderer=self.renderer,
            save_path=self._path(f"{new_path_level_name}/overall_imperfection_heatmap.html"),
            save_results=save_results,
        )

        for c in self.cols:
            matrix = masking.create_missingness_mask_per_col_long_table(
                self.df, col=c, id_col=self.id_col, clock_no_col=self.clock_no_col
            )
            masking.plot_missingness_mask(
                matrix,
                title=f"Missingness Heatmap for {c}",
                id_col=self.id_col,
                clock_no_col=self.clock_no_col,
                renderer=self.renderer,
                save_path=self._path(f"{new_path_level_name}/imperfection_heatmap_{c}.html"),
                save_results=save_results,
            )

    def run(
        self,
        save_results: bool = True,
        lagged_crosscorr_max_lag: int = 10,
        asymmetric_analysis_max_lag: int = 10,
    ):
        """Runs all analyses in the IntervariableImperfection class, including row completeness, co-imperfection, Little's MCAR test, asymmetric imperfection, lagged cross-correlation, and visualization of missingness.

        Parameters:
            save_results (bool): Whether to save the results to files. Defaults to True.
            lagged_crosscorr_max_lag (int): The maximum lag to consider for cross-correlation. Defaults to 10.
            asymmetric_analysis_max_lag (int): The maximum lag to consider for asymmetric missing-observation correlation. Defaults to 10.
        Returns:
            None: This method does not return anything, it runs all analyses and saves the results to files.
        """
        try:
            self.row_statistics(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in row completeness analysis: {e}")
        try:
            self.symmetric_correlation(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in co-imperfection analysis: {e}")
        try:
            self.symmetric_lagged_cross_correlation(
                save_results=save_results, max_lag=lagged_crosscorr_max_lag
            )
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in cross-correlation analysis: {e}")
        try:
            self.mcar_test(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in Little's MCAR test: {e}")
        try:
            self.mar_mnar_test(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in MAR/MNAR test: {e}")
        try:
            self.asymmetric_correlation(save_results=save_results)
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in asymmetric correlation analysis: {e}")
        try:
            self.asymmetric_lagged_cross_correlation(
                save_results=save_results, max_lag=asymmetric_analysis_max_lag
            )
        except Exception as e:
            print(traceback.format_exc())
            print(f"🚩Error in asymmetric lagged cross-correlation analysis: {e}")

    def generate_html_report(
        self, report_path: str = "intervariable_report.html", title: str = None
    ):
        """Generates an HTML report from the analysis results."""
        if not self.save_path:
            pretty_printing.rich_warning("⚠️ Cannot generate report without a save_path.")
            return
        else:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        from .html_report_generator import IntervariableHTMLReportGenerator

        # Create report generator and generate report
        report_generator = IntervariableHTMLReportGenerator(self)
        full_report_path = report_generator.generate_report(report_path, title=title)

        pretty_printing.rich_info(f"✅ Report generated at [green]{full_report_path}[/green]")

    def _path(self, subpath: str) -> Path:
        if self.save_path:
            return self.save_path / subpath
        return None

    def _generate_clock_no_col(self):
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

    # Binary missingness mask
    mask_df = df.with_columns(
        [
            pl.col("heartrate").is_null().cast(pl.Int8).alias("heartrate"),
            pl.col("blood_pressure").is_null().cast(pl.Int8).alias("blood_pressure"),
        ]
    )

    intervariable_imperfection = IntervariableImperfection(
        df=df,
        id_col="patient",
        clock_col="time",
        clock_no_col="clock_no",
        cols=["heartrate", "blood_pressure"],
        renderer=None,
        save_path="test",
    )
    intervariable_imperfection.run(
        save_results=True, lagged_crosscorr_max_lag=5, asymmetric_analysis_max_lag=5
    )
