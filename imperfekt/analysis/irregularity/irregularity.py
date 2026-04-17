import traceback
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        self.ins_case_statistics: pl.DataFrame = None
        self.ins_global_statistics: pl.DataFrame = None
        # Dominant frequency
        self.domf_frequency_summary: pl.DataFrame = None
        self.domf_bin_counts: pl.DataFrame = None
        # Burstiness
        self.bu_case_burstiness: pl.DataFrame = None
        self.bu_global_burstiness: pl.DataFrame = None
        # Interval autocorrelation
        self.ia_autocorrelation: pl.DataFrame = None
        # Case entropy and adherence
        self.ea_case_entropy_adherence: pl.DataFrame = None
        # Composite score
        self.cs_case_scores: pl.DataFrame = None
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
        Compute and cache per-case inter-observation intervals.

        Handles both Datetime clock columns (intervals converted to seconds via
        .dt.total_seconds()) and numeric clock columns (intervals computed as
        a plain numeric difference, cast to Float64).

        The resulting "interval_seconds" column represents the gap between
        consecutive observations. Null values (first observation per case)
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
        Compute per-case and global summary statistics of inter-observation intervals.

        The coefficient of variation (CV = std / mean) per case is the primary
        irregularity score: CV = 0 for a perfectly regular time grid, increasing
        values indicate increasing irregularity.

        Results stored in:
            self.results.ins_case_statistics  — one row per case
            self.results.ins_global_statistics  — describe()-style global summary
            self.results.plots.ins_cv_violin    — violin plot of per-case CV values

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

        self.results.ins_case_statistics = (
            interval_statistics_module.compute_case_interval_statistics(
                delta_t_df, id_col=self.id_col
            )
        )
        self.results.ins_global_statistics = (
            interval_statistics_module.compute_global_interval_statistics(delta_t_df)
        )

        if self.renderer:
            pretty_printing.rich_info(
                "Interval Statistics — Case Level:"
                "cv = std/mean per case (0 = perfectly regular, higher = more irregular); "
                "iqr = spread of interval lengths."
            )
            print(self.results.ins_case_statistics.describe(interpolation="linear"))
            pretty_printing.rich_info(
                "Interval Statistics — Global: pooled summary over all inter-observation intervals."
            )
            print(self.results.ins_global_statistics)

        if save_results and path:
            self.results.ins_case_statistics.write_csv(path / "case_statistics.csv")
            self.results.ins_global_statistics.write_csv(path / "global_statistics.csv")

        # Violin of CV values across entities (drop entities with NaN CV)
        cv_df = self.results.ins_case_statistics.filter(pl.col("cv").is_not_null())
        if cv_df.height > 0:
            cv_violin = visualization_utils.plot_violin(
                cv_df,
                y="cv",
                title="Per-Case Coefficient of Variation of Inter-Observation Intervals",
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
        Compute the burstiness coefficient B per case and globally.

        B = (std - mean) / (std + mean)  [Goh & Barabasi, 2008]
        Range [-1, 1]: B = -1 perfectly regular, B = 0 Poisson, B > 0 bursty.

        Entities with fewer than 3 intervals receive NaN for burstiness_coeff.

        Results stored in:
            self.results.bu_case_burstiness    — one row per case
            self.results.bu_global_burstiness    — single-row global summary
            self.results.plots.bu_burstiness_violin — violin of case B values

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

        self.results.bu_case_burstiness = burstiness_module.compute_burstiness_coefficient(
            delta_t_df, id_col=self.id_col
        )
        self.results.bu_global_burstiness = burstiness_module.compute_global_burstiness(
            delta_t_df, id_col=self.id_col
        )

        if self.renderer:
            pretty_printing.rich_info("Burstiness — Case Level (B=1: bursty, B=-1: perfectly periodic):")
            print(self.results.bu_case_burstiness.describe(interpolation="linear"))
            pretty_printing.rich_info("Burstiness — Global (B=1: bursty, B=-1: perfectly periodic):")
            print(self.results.bu_global_burstiness)

        if save_results and path:
            self.results.bu_case_burstiness.write_csv(path / "case_burstiness.csv")
            self.results.bu_global_burstiness.write_csv(path / "global_burstiness.csv")

        b_df = self.results.bu_case_burstiness.filter(
            pl.col("burstiness_coeff").is_not_null()
        )
        if b_df.height > 0:
            b_violin = visualization_utils.plot_violin(
                b_df,
                y="burstiness_coeff",
                title="Per-Case Burstiness Coefficient",
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

        # Add a sequential integer index per case to serve as clock_no_col for acf()
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

    def case_entropy_adherence(
        self,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        min_intervals: int = 2,
        save_results: bool = True,
    ) -> "Irregularity":
        """
        Compute per-case Shannon entropy and adherence rate of the interval distribution.

        Entropy measures how spread out each case's own interval lengths are (0 = all
        intervals in one bin = perfectly regular; 1 = maximally spread = maximally irregular).

        Adherence measures how consistently each case follows *its own* dominant interval
        rhythm — not the dataset-wide dominant. A patient with a unique but perfectly
        consistent rhythm scores adherence = 1.0.

        Results stored in:
            self.results.ea_case_entropy_adherence  — one row per case

        Parameters:
            bin_resolution_seconds (float): Bin width in seconds for discretizing intervals.
            adherence_tolerance (float): Fractional tolerance around each case's own dominant
                                         interval. E.g. 0.5 means within [dominant*0.5, dominant*1.5].
            min_intervals (int): Minimum intervals required to compute metrics. Entities below
                                 this threshold receive NaN. Default 2.
            save_results (bool): Whether to save CSV to save_path.

        Returns:
            self: Supports method chaining.
        """
        new_path_level_name = "case_entropy_adherence"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        delta_t_df = self._compute_intervals()

        self.results.ea_case_entropy_adherence = (
            interval_statistics_module.compute_case_interval_entropy_adherence(
                delta_t_df,
                id_col=self.id_col,
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
                min_intervals=min_intervals,
            )
        )

        if self.renderer:
            pretty_printing.rich_info(
                "Case Entropy & Adherence:"
                "normalized_entropy = 0 (all intervals in one bin = regular) → 1 (maximally spread = irregular); "
                "adherence_rate = fraction of intervals near each case's own dominant rhythm."
            )
            print(self.results.ea_case_entropy_adherence.describe(interpolation="linear"))

        if save_results and path:
            self.results.ea_case_entropy_adherence.write_csv(
                path / "case_entropy_adherence.csv"
            )

        return self

    def composite_score(
        self,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        min_intervals: int = 2,
        n_quantiles: int = 4,
        save_results: bool = True,
    ) -> "Irregularity":
        """
        Compute a PCA-based composite irregularity score per case and assign quantile strata.

        Combines three complementary per-case metrics:
            - cv: magnitude of interval dispersion (from interval_statistics)
            - burstiness_coeff: direction of dispersion — clustered vs. regular (from burstiness)
            - normalized_entropy: distributional complexity (from case_entropy_adherence)

        Metrics are scaled with StandardScaler, then compressed to a single principal
        component (PC1) via PCA. The PC1 axis is anchored so that high cv corresponds
        to a high (more irregular) composite score. Entities are assigned to
        Q1 (least irregular) … Qn (most irregular) quantile strata.

        Requires interval_statistics() and burstiness() to have been run first.
        If case_entropy_adherence() has already been run, reuses those results.

        Results stored in:
            self.results.cs_case_scores  — one row per case

        Parameters:
            bin_resolution_seconds (float): Bin width for entropy/adherence computation.
            adherence_tolerance (float): Fractional tolerance for adherence_rate.
            min_intervals (int): Minimum intervals for entropy/adherence computation.
            n_quantiles (int): Number of strata (default 4 → Q1…Q4).
            save_results (bool): Whether to save case_scores.csv to save_path.

        Returns:
            self: Supports method chaining.
        """
        if self.results.ins_case_statistics is None:
            raise ValueError(
                "interval_statistics() must be run before composite_score(). "
                "Call .interval_statistics() first or use .run()."
            )
        if self.results.bu_case_burstiness is None:
            raise ValueError(
                "burstiness() must be run before composite_score(). "
                "Call .burstiness() first or use .run()."
            )

        new_path_level_name = "composite_score"
        path = None
        if self.save_path and save_results:
            path = self.save_path / new_path_level_name
            path.mkdir(parents=True, exist_ok=True)

        # Reuse entropy/adherence if already computed, otherwise compute now
        if self.results.ea_case_entropy_adherence is not None:
            entropy_df = self.results.ea_case_entropy_adherence
        else:
            delta_t_df = self._compute_intervals()
            entropy_df = interval_statistics_module.compute_case_interval_entropy_adherence(
                delta_t_df,
                id_col=self.id_col,
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
                min_intervals=min_intervals,
            )

        id_col = self.id_col

        # Join the three feature sources on id
        scores = (
            self.results.ins_case_statistics
            .select([id_col, "cv"])
            .join(
                self.results.bu_case_burstiness.select([id_col, "burstiness_coeff"]),
                on=id_col,
                how="left",
            )
            .join(
                entropy_df.select([id_col, "normalized_entropy", "adherence_rate"]),
                on=id_col,
                how="left",
            )
        )

        # PCA composite score using complete rows only
        feature_cols = ["cv", "burstiness_coeff", "normalized_entropy"]
        complete_mask = (
            pl.col("cv").is_not_null()
            & pl.col("burstiness_coeff").is_not_null()
            & pl.col("normalized_entropy").is_not_null()
        )
        complete_df = scores.filter(complete_mask)

        if len(complete_df) < 2:
            scores = scores.with_columns(pl.lit(None).cast(pl.Float64).alias("composite_score"))
        else:
            X = complete_df.select(feature_cols).to_numpy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Guard: if any feature has zero variance after scaling, PCA is meaningless
            if np.any(scaler.scale_ == 0):
                scores = scores.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("composite_score")
                )
            else:
                pca = PCA(n_components=1)
                pc1_scores = pca.fit_transform(X_scaled).ravel()

                # Directionality fix: anchor so that high cv → high score (more irregular)
                cv_idx = feature_cols.index("cv")
                if pca.components_[0, cv_idx] < 0:
                    pc1_scores = -pc1_scores

                explained = float(pca.explained_variance_ratio_[0])
                if self.renderer:
                    pretty_printing.rich_info(
                        f"PCA composite score: PC1 explains {explained:.1%} of variance "
                        "across cv, burstiness_coeff, normalized_entropy."
                    )

                # Map PC1 scores back to full row order (incomplete rows get None)
                id_to_pc1 = dict(zip(complete_df[id_col].to_list(), pc1_scores.tolist()))
                composite_values = [id_to_pc1.get(v) for v in scores[id_col].to_list()]
                scores = scores.with_columns(
                    pl.Series(name="composite_score", values=composite_values, dtype=pl.Float64)
                )

        # Quantile stratification — clamp n_quantiles to available scored entities
        non_null_scores = (
            scores.filter(pl.col("composite_score").is_not_null())["composite_score"].to_numpy()
        )

        effective_quantiles = min(n_quantiles, len(non_null_scores))

        if effective_quantiles >= 2:
            quantile_bounds = np.quantile(
                non_null_scores,
                q=[i / effective_quantiles for i in range(1, effective_quantiles)],
            )

            def assign_stratum(score_val) -> str:
                if score_val is None or (isinstance(score_val, float) and np.isnan(score_val)):
                    return None
                for q_idx, bound in enumerate(quantile_bounds):
                    if score_val <= bound:
                        return f"Q{q_idx + 1}"
                return f"Q{effective_quantiles}"

            stratum_values = [assign_stratum(v) for v in scores["composite_score"].to_list()]
        else:
            stratum_values = [
                "Q1" if v is not None else None
                for v in scores["composite_score"].to_list()
            ]

        scores = scores.with_columns(
            pl.Series(name="irregularity_stratum", values=stratum_values, dtype=pl.Utf8)
        )

        scores = scores.select([
            id_col,
            "cv",
            "burstiness_coeff",
            "normalized_entropy",
            "adherence_rate",
            "composite_score",
            "irregularity_stratum",
        ])

        self.results.cs_case_scores = scores

        if self.renderer:
            pretty_printing.rich_info(
                "Composite Irregularity Score: "
                "composite_score = PC1 from PCA on StandardScaler-normalized cv, burstiness_coeff, normalized_entropy; "
                f"irregularity_stratum = Q1 (least irregular) … Q{n_quantiles} (most irregular)."
            )
            print(self.results.cs_case_scores.describe(interpolation="linear"))
            total = len(scores)
            prevalence = (
                scores.group_by("irregularity_stratum")
                .agg(pl.len().alias("n"))
                .with_columns((pl.col("n") / total * 100).round(1).alias("pct"))
                .sort("irregularity_stratum")
            )
            print(prevalence)

        if save_results and path:
            self.results.cs_case_scores.write_csv(path / "case_scores.csv")

        return self

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        bin_resolution_seconds: float,
        adherence_tolerance: float,
    ) -> pl.DataFrame:
        """
        Assemble a single-row summary DataFrame from all completed analyses.

        Columns (all global / cross-case):
            Interval statistics (pooled):
                mean_seconds, median_seconds, q25_seconds, q75_seconds,
                min_seconds, max_seconds
            Dominant frequency:
                dominant_interval_seconds, adherence_rate, n_adhering_intervals,
                n_total_intervals, interval_entropy_bits, normalized_entropy,
                n_unique_bins
            Burstiness:
                burstiness_coeff_global
            Metadata:
                bin_resolution_seconds, adherence_tolerance

        Returns:
            pl.DataFrame: One-row summary, or None if no results are available.
        """
        row: dict = {}

        # --- Interval statistics (global describe table) ---
        if self.results.ins_global_statistics is not None:
            stats = self.results.ins_global_statistics
            # describe() returns rows keyed by "statistic" column
            stat_map = {
                r["statistic"]: r["interval_seconds"]
                for r in stats.to_dicts()
            }
            for stat_key, col_name in [
                ("mean",   "mean_seconds"),
                ("50%",    "median_seconds"),
                ("25%",    "q25_seconds"),
                ("75%",    "q75_seconds"),
                ("min",    "min_seconds"),
                ("max",    "max_seconds"),
            ]:
                row[col_name] = stat_map.get(stat_key)

        # --- Dominant frequency ---
        if self.results.domf_frequency_summary is not None:
            domf = self.results.domf_frequency_summary.row(0, named=True)
            for key in [
                "dominant_interval_seconds",
                "adherence_rate",
                "n_adhering_intervals",
                "n_total_intervals",
                "interval_entropy_bits",
                "normalized_entropy",
                "n_unique_bins",
            ]:
                row[key] = domf.get(key)

        # --- Burstiness (global) ---
        if self.results.bu_global_burstiness is not None:
            bu = self.results.bu_global_burstiness.row(0, named=True)
            row["burstiness_coeff_global"] = bu.get("burstiness_coeff")

        # --- Composite score (global aggregates) ---
        if self.results.cs_case_scores is not None:
            cs = self.results.cs_case_scores
            non_null_cs = cs.filter(pl.col("composite_score").is_not_null())["composite_score"]
            row["mean_composite_score"] = float(non_null_cs.mean()) if non_null_cs.len() > 0 else None
            row["std_composite_score"] = float(non_null_cs.std()) if non_null_cs.len() > 1 else None
            row["n_quantiles"] = int(
                cs.filter(pl.col("irregularity_stratum").is_not_null())["irregularity_stratum"].n_unique()
            )

        # --- Metadata ---
        row["bin_resolution_seconds"] = bin_resolution_seconds
        row["adherence_tolerance"] = adherence_tolerance

        if not row:
            return None

        return pl.DataFrame({
            "name": list(row.keys()),
            "value": [str(v) if v is not None else None for v in row.values()],
        })

    def run(
        self,
        save_results: bool = True,
        bin_resolution_seconds: float = 60.0,
        adherence_tolerance: float = 0.5,
        min_intervals: int = 2,
        n_strata_quantiles: int = 4,
        autocorrelation_lags: int = 20,
    ) -> "Irregularity":
        """
        Run all irregularity analyses in sequence, then write a single-row
        summary CSV (``irregularity_summary.csv``) that consolidates the key
        quantified results across all sub-analyses.

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

        try:
            self.case_entropy_adherence(
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
                save_results=save_results,
            )
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in case_entropy_adherence: {e}")

        try:
            self.composite_score(
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
                min_intervals=min_intervals,
                n_quantiles=n_strata_quantiles,
                save_results=save_results,
            )
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error in composite_score: {e}")

        # --- Final consolidated summary ---
        try:
            summary = self._build_summary(
                bin_resolution_seconds=bin_resolution_seconds,
                adherence_tolerance=adherence_tolerance,
            )
            if summary is not None:
                if self.renderer:
                    pretty_printing.rich_info(
                        "Irregularity Summary — consolidated quantified results across all analyses."
                    )
                    print(summary)
                if save_results and self.save_path:
                    self.save_path.mkdir(parents=True, exist_ok=True)
                    summary.write_csv(self.save_path / "irregularity_summary.csv")
        except Exception as e:
            print(traceback.format_exc())
            pretty_printing.rich_error(f"Error building irregularity summary: {e}")

        return self


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "id": [
                "a", "a", "a", "a", "a",
                "b", "b", "b", "b", "b",
                "c", "c", "c", "c", "c",
                "d", "d",
            ],
            "clock": [
                # a — irregular gaps (bursty)
                "2023-01-01 00:00:00",
                "2023-01-01 00:02:00",
                "2023-01-01 00:10:00",
                "2023-01-01 00:15:00",
                "2023-01-01 00:20:00",
                # b — very regular, 5-minute cadence
                "2023-01-01 00:00:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:10:00",
                "2023-01-01 00:15:00",
                "2023-01-01 00:20:00",
                # c — moderately irregular (evenly spaced but not perfectly so)
                "2023-01-01 00:00:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:07:00",
                "2023-01-01 00:11:00",
                "2023-01-01 00:16:00",
                # d — only 2 observations, too few for burstiness → no composite score
                "2023-02-02 00:25:00",
                "2023-02-02 00:30:00",
            ],
            "heartrate": [
                60, None, 70, 65, None,
                72, 74, 71, 73, 70,
                80, None, 85, 82, None,
                90, None,
            ],
            "blood_pressure": [
                120, 130, None, None, None,
                118, 120, 119, 121, 117,
                135, 140, None, None, None,
                125, None,
            ],
        }
    ).with_columns(
        [
            pl.col("clock").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        ]
    )
    
    print(df)
    irregularity_analysis = Irregularity(df, save_path=Path("results/irregularity_example"), renderer="notebook_connected")
    irregularity_analysis.run(save_results=True)