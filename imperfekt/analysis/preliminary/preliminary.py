from pathlib import Path

import matplotlib.pyplot as plt
import pingouin as pg
import plotly.graph_objects as go
import polars as pl
from scipy.stats import shapiro

from imperfekt.analysis.intravariable import autocorrelation
from imperfekt.analysis.utils import pretty_printing, visualization_utils


class PreliminaryPlots:
    def __init__(self):
        self.violin: dict[str, go.Figure] = {}
        self.correlation_heatmap: go.Figure = go.Figure()
        self.autocorrelation_lag_plot: dict[str, go.Figure] = {}
        self.qq_plot: dict[str, go.Figure | plt.Figure] = {}


class PreliminaryResults:
    def __init__(self):
        self.correlation: pl.DataFrame = pl.DataFrame()
        self.autocorrelation: dict[str, pl.DataFrame] = {}
        self.multivariate_normality: pl.DataFrame = pl.DataFrame()
        self.shapiro_wilk: pl.DataFrame = pl.DataFrame()
        self.descriptive_stats: pl.DataFrame = pl.DataFrame()
        self.plots = PreliminaryPlots()


class Preliminary:
    def __init__(
        self,
        df: pl.DataFrame,
        id_col: str = "id",
        clock_col: str = "clock",
        clock_no_col: str = "clock_no",
        cols: list = None,
        alpha: float = 0.05,
        save_path: Path = None,
        plot_library: str = "matplotlib",
        renderer: str = "notebook_connected",
    ):
        """
        Initializes the Preliminary analysis class.

        Parameters:
            df (pl.DataFrame): The dataframe to analyze.
            id_col (str): The column representing unique identifiers.
            clock_col (str): The column representing time or clock.
            clock_no_col (str): The column representing the clock number.
            cols (list): List of columns to analyze. If None, all columns except id_col and clock_col will be used.
            alpha (float): The significance level for hypothesis testing.
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
        self.df = df
        # Relevant columns for the analysis
        self.id_col = id_col
        self.clock_col = clock_col
        self.clock_no_col = clock_no_col
        self._generate_clock_no_col()
        self.cols = cols or [c for c in df.columns if c not in {id_col, clock_col, clock_no_col}]
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
        self.results = PreliminaryResults()

    def describe_df(self, save_results: bool = True):
        """
        Returns a description of the dataframe.

        Returns:
            pl.DataFrame: Description of the dataframe.
        """
        self.results.descriptive_stats = self.df.describe()

        if self.save_path and save_results:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.results.descriptive_stats.write_csv(self.save_path / "descriptive_stats.csv")

        # Plot violin plots for each column
        for col in self.cols:
            self.results.plots.violin[col] = visualization_utils.plot_violin(
                self.df,
                y=col,
                title=f"Violin Plot of {col}",
                save_path=self._path(f"{col}_violin_plot.png"),
                save_results=save_results,
                renderer=self.renderer,
                library=self.plot_library,
            )

        return self

    def multivariate_normality(self, save_results: bool = True):
        """
        Performs multivariate normality tests using pingouin.

        Parameters:
            save_results (bool): Whether to save the results.
        Returns:
            self: Returns the instance for method chaining.
        """
        # Get numeric columns only
        numeric_cols = [
            c for c in self.cols if self.df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        if len(numeric_cols) < 2:
            pretty_printing.rich_warning(
                "Need at least 2 numeric columns for multivariate normality testing."
            )
            return self

        # Convert to pandas for pingouin (remove nulls)
        df_pandas = self.df.select(numeric_cols).drop_nulls().to_pandas()

        if df_pandas.empty:
            pretty_printing.rich_warning("No data remaining after removing nulls.")
            return self

        # Henze-Zirkler test
        try:
            hz_result = pg.multivariate_normality(df_pandas, alpha=self.alpha)
            self.results.multivariate_normality = pl.DataFrame(
                {
                    "hz": [hz_result.hz],
                    "pval": [hz_result.pval],
                    "normal": [hz_result.normal],
                }
            )
            if self.renderer:
                print("\nHenze-Zirkler Multivariate Normality Test:")
                print(hz_result)
        except Exception as e:
            pretty_printing.rich_warning(f"Henze-Zirkler test failed: {e}")
            return self

        if save_results and self.save_path:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.results.multivariate_normality.write_csv(
                self.save_path / "multivariate_normality_results.csv"
            )
            print(
                f"Multivariate normality results saved to {self.save_path / 'multivariate_normality_results.csv'}"
            )
        return self

    def shapiro_wilk(self, save_results: bool = True):
        """
        Performs Shapiro-Wilk test for normality on the specified columns.

        Parameters:
            save_results (bool): Whether to save the results.
        Returns:
            self: Returns the instance for method chaining.
        """
        # Perform Shapiro-Wilk test
        results = {}

        for col in self.cols:
            # Check if column is numeric
            if self.df[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                pretty_printing.rich_warning(
                    f"Skipping non-numeric column for Shapiro-Wilk test: {col}"
                )
                continue

            values_np = self.df[col].to_numpy()
            shapiro_result = shapiro(values_np, nan_policy="omit")  # Handle NaNs

            results[col] = shapiro_result
            self.results.plots.qq_plot[col] = visualization_utils.plot_qq(
                values_np,
                title=f"Q-Q Plot of {col}",
                library="matplotlib",
                save_path=self._path(f"{col}_qq_plot.png"),
                save_results=save_results,
                renderer=self.renderer,
            )

        # Convert results to DataFrame
        shapiro_results = pl.DataFrame(
            {
                "Column": list(results.keys()),
                "W": [res.statistic for res in results.values()],
                "p-value": [res.pvalue for res in results.values()],
                f"Significance_{self.alpha}": [res.pvalue < self.alpha for res in results.values()],
            }
        )
        self.results.shapiro_wilk = shapiro_results

        if save_results and self.save_path:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            shapiro_results.write_csv(self.save_path / "shapiro_wilk_results.csv")
            print(f"Shapiro-Wilk results saved to {self.save_path / 'shapiro_wilk_results.csv'}")

        return self

    def correlation(self, use: str = "listwise", save_results: bool = True):
        """
        Calculates and plots the correlation matrix for the dataframe.
        `use` can be 'listwise' or 'pairwise'.
        listwise = Drop rows with any null values in the selected columns.
        pairwise = Compute correlation for each pair of columns. More rows can be included this way.

        Parameters:
            use (str): Method for correlation ('listwise' or 'pairwise').
            save_results (bool): Whether to save the results.
        Returns:
            self: Returns the instance for method chaining.
        """
        if self.save_path and save_results:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
        if use not in ["listwise", "pairwise"]:
            raise ValueError("`use` must be either 'listwise' or 'pairwise'.")

        if not self.cols or len(self.cols) == 1:
            pretty_printing.rich_warning(
                "Not sufficient columns for correlation analysis. "
                "Please provide multiple columns for a meaningful correlation analysis."
            )
            return self
        else:
            filtered_df = self.df.select(self.cols)

        # Ensure the dataframe has numeric columns for correlation
        numeric_cols = filtered_df.select(
            pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ).columns
        if len(numeric_cols) < 2:
            pretty_printing.rich_warning(
                "Not enough numeric columns for correlation analysis. "
                "Please provide at least two numeric columns."
            )
            self.results.correlation = pl.DataFrame()
            return self

        # Remove irrelevant columns from the dataframe
        filtered_df = filtered_df.select(
            [
                pl.col(c)
                for c in filtered_df.columns
                if c not in [self.id_col, self.clock_no_col, self.clock_col]
            ]
        )
        numeric_cols = [
            c for c in numeric_cols if c not in [self.id_col, self.clock_no_col, self.clock_col]
        ]

        if use == "listwise":
            # Drop rows where null values are present in any of the selected columns
            filtered_df = filtered_df.drop_nulls()
            self.results.correlation = filtered_df.select(numeric_cols).corr()
        elif use == "pairwise":
            # Pairwise correlation
            corr_data = []
            for col1 in numeric_cols:
                row_data = {" ": col1}
                for col2 in numeric_cols:
                    if col1 == col2:
                        correlation = 1.0
                    else:
                        # Select the two columns and drop rows with nulls for the pair
                        pair_df = filtered_df.select([col1, col2]).drop_nulls()
                        # Calculate correlation for the pair
                        if pair_df.height > 1:
                            correlation = pair_df.select(
                                pl.corr(col1, col2, method="spearman")
                            ).item()
                        else:
                            correlation = None  # Not enough data
                    row_data[col2] = correlation
                corr_data.append(row_data)

            self.results.correlation = pl.from_dicts(corr_data)

        self.results.plots.correlation_heatmap = self._plot_heatmap(
            self.results.correlation,
            save_path=self._path(f"{use}_correlation_heatmap.png"),
            save_results=save_results,
        )

        return self

    def autocorrelation(self, lags: int = 20, save_results: bool = True):
        """
        Calculates and plots the autocorrelation for specified columns.
        Parameters:
            lags (int): Number of lags for autocorrelation.
            save_results (bool): Whether to save the results.
        Returns:
            self: Returns the instance for method chaining.
        """
        if self.save_path and save_results:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
        for c in self.cols:
            # check if column is numeric
            if self.df[c].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                print(f"Skipping non-numeric column: {c}")
                continue
            self.results.autocorrelation[c] = autocorrelation.acf(
                mask_df=self.df,
                col=c,
                id_col=self.id_col,
                clock_no_col=self.clock_no_col,
                max_lag=lags,
                save_path=self._path(f"{c}_autocorrelation.csv"),
                save_results=save_results,
            )

            if self.renderer:
                print(f"Autocorrelation for {c}:")
                print(self.results.autocorrelation[c])

            self.results.plots.autocorrelation_lag_plot[c] = visualization_utils.plot_scatter(
                x=self.results.autocorrelation[c]["lag"],
                y=self.results.autocorrelation[c]["autocorr"],
                title=f"Lagged Autocorrelation of Imperfection: {c}",
                xaxis_title="Lag",
                yaxis_title="Autocorrelation",
                save_path=self._path(f"{c}_autocorrelation_plot.png"),
                save_results=save_results,
                renderer=self.renderer,
                library=self.plot_library,
            )

        return self

    def run(self, lags: int = 20, save_results: bool = True, use: str = "pairwise"):
        """
        Runs all preliminary analyses: autocorrelation and correlation.

        Parameters:
            lags (int): Number of lags for autocorrelation.
            save_results (bool): Whether to save the results.
            use (str): Method for correlation ('listwise' or 'pairwise').
        Returns:
            self: Returns the instance for method chaining.
        """
        self.describe_df(save_results=save_results)
        self.shapiro_wilk(save_results=save_results)
        self.intervariable_normality(save_results=save_results)
        self.autocorrelation(lags=lags, save_results=save_results)
        self.correlation(use=use, save_results=save_results)
        return self

    def generate_html_report(self, report_path: str = "preliminary_report.html", title: str = None):
        """Generates an HTML report from the analysis results."""
        if not self.save_path:
            pretty_printing.rich_warning("⚠️ Cannot generate report without a save_path.")
            return
        else:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        from .html_report_generator import PreliminaryHTMLReportGenerator

        # Create report generator and generate report
        report_generator = PreliminaryHTMLReportGenerator(self)
        full_report_path = report_generator.generate_report(report_path, title=title)

        pretty_printing.rich_info(f"✅ Report generated at [green]{full_report_path}[/green]")

    def _plot_heatmap(
        self, matrix: pl.DataFrame, save_path: Path, save_results: bool = True
    ) -> go.Figure:
        # Handle the case where the correlation matrix has a label column (from pairwise correlation)
        if " " in matrix.columns:
            # Remove the label column and get only numeric correlation values
            cols = [c for c in matrix.columns if c != " "]
            corr_matrix_values = matrix.select(cols).to_pandas().values
        else:
            cols = matrix.columns
            corr_matrix_values = matrix.to_pandas().values

        # Replace None values with NaN for proper handling
        import numpy as np

        corr_matrix_values = np.array(corr_matrix_values, dtype=float)

        # Create text labels, handling NaN values
        text_labels = np.where(
            np.isnan(corr_matrix_values),
            "N/A",
            np.round(corr_matrix_values, 4).astype(str),
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix_values,
                x=cols,
                y=cols,
                text=text_labels,
                texttemplate="%{text}",
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Spearman (ρ)"),
            )
        )

        fig.update_layout(
            title="Correlation Heatmap",
            xaxis=dict(side="top"),
            width=700,
            height=700,
            yaxis_autorange="reversed",  # So that the first row is at the top
            template="plotly_white",
        )

        if self.renderer:
            fig.show(renderer=self.renderer)

        if save_results and save_path:
            fig.write_image(save_path)
            print(f"Heatmap saved to {save_path}")

        return fig

    def _generate_clock_no_col(self):
        self.df = self.df.sort([self.id_col, self.clock_col])
        if self.clock_no_col not in self.df.columns:
            self.df = self.df.with_columns(
                pl.cum_count(self.id_col).over(self.id_col).alias(self.clock_no_col)
            )

    def _path(self, subpath: str) -> Path:
        """Generates a full path for saving results."""
        if self.save_path:
            return self.save_path / subpath
        return None


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
    preliminary = Preliminary(
        df=df,
        id_col="patient",
        clock_col="time",
        clock_no_col="clock_no",
        cols=["heartrate", "blood_pressure"],
        alpha=0.05,
        save_path=Path("results"),
        plot_library="plotly",
        renderer="notebook_connected",
    )

    print("Running preliminary analysis...")
    print(preliminary.run(lags=5, save_results=True, use="pairwise"))
