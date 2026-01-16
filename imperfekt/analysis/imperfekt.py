from pathlib import Path

import polars as pl

from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
from imperfekt.analysis.intravariable.intravariable import IntravariableImperfection
from imperfekt.analysis.preliminary.preliminary import Preliminary
from imperfekt.analysis.utils import masking, pretty_printing
from imperfekt.analysis.utils.events import (
    calculate_event_percentage,
    create_event_mask,
    filter_events_df,
)


class Imperfekt:
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
        self.cols = cols or [c for c in df.columns if c not in {id_col, clock_col, clock_no_col}]
        self._initial_check()
        self._generate_clock_no_col()

        self.missingness_mask = masking.create_missingness_mask(
            df=self.df,
            id_col=id_col,
            clock_col=clock_col,
            clock_no_col=clock_no_col,
            cols=self.cols,
        )

        self.alpha = alpha

        # Result persistence
        self.save_path = save_path
        if self.save_path:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        # Plotting library
        if plot_library not in ["matplotlib", "plotly"]:
            raise ValueError(
                f"Unsupported plot library: {plot_library}. Supported libraries: 'matplotlib', 'plotly'."
            )
        self.plot_library = plot_library

        # Plot rendering
        self.renderer = renderer

        self.preliminary = Preliminary(
            df=self.df,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
            cols=self.cols,
            alpha=self.alpha,
            save_path=self.save_path / "preliminary" if self.save_path else None,
            plot_library=self.plot_library,
            renderer=self.renderer,
        )

        self.intravariable = IntravariableImperfection(
            df=self.df,
            mask_df=self.missingness_mask,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
            cols=self.cols,
            alpha=self.alpha,
            save_path=self.save_path / "intravariable" if self.save_path else None,
            plot_library=self.plot_library,
            renderer=self.renderer,
        )

        self.intervariable = IntervariableImperfection(
            df=self.df,
            mask_df=self.missingness_mask,
            id_col=self.id_col,
            clock_col=self.clock_col,
            clock_no_col=self.clock_no_col,
            cols=self.cols,
            alpha=self.alpha,
            save_path=self.save_path / "intervariable" if self.save_path else None,
            plot_library=self.plot_library,
            renderer=self.renderer,
        )

        # Result Persistence
        self.group_results = {}
        self.event_results = {}

    def run(
        self,
        save_results: bool = True,
        generate_html: bool = True,
        addition_to_title: str = None,
        cheap_mode: bool = False,
    ):
        """
        Runs the preliminary analysis, intravariable analysis, and intervariable analysis.
        To run each analysis individually and define parameters, call their respective run methods, e.g.: <imperfect_object>.preliminary.run(save_results=True,lags=10,use='pairwise').
        Access results using <imperfect_object>.preliminary.results, <imperfect_object>.intravariable.results, and <imperfect_object>.intervariable.results.

        Parameters:
            save_results (bool): If True, saves the results of the analysis.
            generate_html (bool): If True, generates HTML reports for the analysis.
            addition_to_title (str): Additional string to add to the title of the HTML reports.
            cheap_mode (bool): If True, runs a faster, less comprehensive analysis. Includes: DataFrame description, Observed Value Correlation, Column-level imperfection statistics, Gap statistics, Datetime statistics, Row-level imperfection statistics, MCAR test, MAR/MNAR test, Symmetric Correlation.
        """
        pretty_printing.rich_info("Starting analysis...")
        if cheap_mode:
            self.preliminary.describe_df(save_results=save_results).correlation(
                save_results=save_results
            )
            self.intravariable.column_statistics(save_results=save_results).gap_statistics(
                save_results=save_results
            ).date_time_statistics(save_results=save_results)
            self.intervariable.row_statistics(save_results=save_results).mcar_test(
                save_results=save_results
            ).mar_mnar_test(save_results=save_results).symmetric_correlation(
                save_results=save_results
            )
            pretty_printing.rich_info("Cheap mode analysis complete.")
        else:
            self.preliminary.run(save_results=save_results)
            self.intravariable.run(save_results=save_results)
            self.intervariable.run(save_results=save_results)
            pretty_printing.rich_info("Full analysis complete.")
        if generate_html:
            self.generate_html_reports(addition_to_title=addition_to_title)

    def run_grouped_analysis(
        self,
        annotation_col: str,
        annotation_df: pl.DataFrame = None,
        save_results: bool = True,
        top_n_groups: int = None,
        addition_to_title: str = None,
        cheap_mode: bool = False,
    ):
        """
        Runs a grouped analysis based on an annotation dataframe.
        RQ: Does the imperfection analysis reveal any differences between groups?

        For each group in the annotation column, it filters the main dataframe
        and runs the complete Imperfekt analysis. Results are saved in
        subdirectories named after each group and can be retrieved under <imperfect_object>.group_results.
        To limit the number of classes/groups analyzed, use the top_n_groups parameter.

        Parameters:
            annotation_df (pl.DataFrame, optional): Dataframe with at least two columns:
                                          the id_col and the annotation_col. If None, the main dataframe is used.
            annotation_col (str): The column name in annotation_df to group by.
            save_results (bool): If True, saves the results of the analysis.
            top_n_groups (int, optional): If set, limits the analysis to the
                                          top N largest groups. Defaults to None.

        Returns:
            None, updates the dict under <imperfect_object>.group_results.
        """
        pretty_printing.rich_info(f"Starting grouped analysis on column '{annotation_col}'...")

        if annotation_df is not None:
            if self.id_col not in annotation_df.columns:
                raise ValueError(
                    f"ID column '{self.id_col}' not found in the annotation dataframe."
                )
            if annotation_col not in annotation_df.columns:
                raise ValueError(
                    f"Annotation column '{annotation_col}' not found in the annotation dataframe."
                )

            # Keep only relevant columns and drop duplicates
            annotation_df = annotation_df.select([self.id_col, annotation_col]).unique(
                subset=[self.id_col]
            )

            merged_df = self.df.join(annotation_df, on=self.id_col, how="inner")
        else:
            if annotation_col not in self.df.columns:
                raise ValueError(
                    f"Annotation column '{annotation_col}' not found in the main dataframe."
                )
            merged_df = self.df

        # Calculate group sizes and filter for top N if requested
        group_counts = (
            merged_df.group_by(annotation_col)
            .agg(pl.n_unique(self.id_col).alias("count"))
            .sort("count", descending=True)
        )

        if top_n_groups:
            pretty_printing.rich_info(
                f"Limiting analysis to the top {top_n_groups} largest groups."
            )
            group_counts = group_counts.head(top_n_groups)

        groups = group_counts[annotation_col].to_list()

        if not groups:
            pretty_printing.rich_warning(
                "No common cases found between the main dataframe and the annotation dataframe. "
                "Grouped analysis cannot be performed."
            )
            return

        for group in groups:
            if group is None:
                pretty_printing.rich_warning(
                    "Skipping group with `None` value in annotation column."
                )
                continue

            pretty_printing.rich_info(f"Running analysis for group: {group}")
            group_df = merged_df.filter(pl.col(annotation_col) == group)

            if self.save_path:
                group_save_path = self.save_path / str(group)
                group_save_path.mkdir(parents=True, exist_ok=True)
            else:
                group_save_path = None

            self.group_results[group] = Imperfekt(
                df=group_df,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                cols=self.cols,
                alpha=self.alpha,
                save_path=group_save_path,
                plot_library=self.plot_library,
                renderer=self.renderer,
            )

            if not cheap_mode:
                self.group_results[group].run(save_results=save_results)
            else:
                self.group_results[group].preliminary.describe_df(
                    save_results=save_results
                ).correlation(save_results=save_results)
                self.group_results[group].intravariable.column_statistics(
                    save_results=save_results
                ).gap_statistics(save_results=save_results).date_time_statistics(
                    save_results=save_results
                )
                self.group_results[group].intervariable.row_statistics(
                    save_results=save_results
                ).mcar_test(save_results=save_results).mar_mnar_test(
                    save_results=save_results
                ).symmetric_correlation(save_results=save_results)
            if group_save_path:
                self.group_results[group].generate_html_reports(
                    addition_to_title=addition_to_title + f" - Group: {group}"
                )
        pretty_printing.rich_info("Grouped analysis complete.")

    def run_event_based_analysis(
        self,
        events_df: pl.DataFrame,
        event_name_col: str = None,
        included_event_names: list = None,
        window_size: int = 0,
        window_location: str = "both",  # 'both', 'before', 'after'
        remove_ids_without_events: bool = True,
        save_results: bool = True,
    ):
        """
        Runs event-based analysis on the specified timestamps.
        RQ: Do on-scene events (interventions) impact data imperfection?

        This method extracts a binary event mask from a event_df to split the initial df into event and non-event subsets.
        Subsequently, all analyzes that do not depend on ordered timestamps are performed and their results stored in <imperfect_object>.event_results.

        Included Analyses are:
        - Preliminary: DataFrame description, Observed Value Correlation, Normality tests.
        - Intravariable Imperfection: Column-level imperfection statistics, Datetime statistics.
        - Intervariable Imperfection: Row-level imperfection statistics, MCAR test, Symmetric and Asymmetric Correlation.

        Parameters:
            events_df (pl.DataFrame): A DataFrame containing event data with at least the ID and clock columns.
            event_name_col (str): The column name in events_df that contains event names.
            included_event_names (list): A list of event names to include in the analysis. Requires event_name_col to be set.
            window_size (int): The size of the time window to consider for each event.
            window_location (str): The location of the time window relative to the event ('both', 'before', 'after').
            remove_ids_without_events (bool): Whether to remove IDs that do not have any associated events.
            save_results (bool): Whether to save the results of the analysis.

        Returns:
            None, updates <imperfect_object>.event_results
        """
        pretty_printing.rich_info("Starting event-based analysis...")

        if self.id_col not in events_df.columns:
            raise ValueError(f"ID column '{self.id_col}' not found in the event dataframe.")
        if self.clock_col not in events_df.columns:
            raise ValueError(f"Clock column '{self.clock_col}' not found in the event dataframe.")

        # Create a mask based on events_df and window_size/location
        events_df = filter_events_df(
            events_df,
            event_name_col=event_name_col,
            included_event_names=included_event_names,
        )
        _, events_mask = create_event_mask(
            df=self.df,
            events_df=events_df,
            id_col=self.id_col,
            clock_col=self.clock_col,
            window_size=window_size,
            window_location=window_location,
            remove_ids_without_events=remove_ids_without_events,
        )

        event_percentage = calculate_event_percentage(events_mask)
        if self.renderer:
            pretty_printing.rich_info(
                f"Percentage of timestamps that could be affected by events: {event_percentage:.2f}%"
            )

        # Split the main dataframe into event and non-event subsets
        joined_df = self.df.join(
            events_mask.select([self.id_col, self.clock_col, "event_mask"]),
            on=[self.id_col, self.clock_col],
            how="inner",
        ).with_columns(pl.col("event_mask").fill_null(False))

        event_df = joined_df.filter(pl.col("event_mask")).drop("event_mask")
        non_event_df = joined_df.filter(~pl.col("event_mask")).drop("event_mask")

        if self.save_path:
            event_save_path = self.save_path / "events"
            event_save_path.mkdir(parents=True, exist_ok=True)
        else:
            event_save_path = None

        for df, label in zip([event_df, non_event_df], ["event", "non_event"]):
            self.event_results[label] = Imperfekt(
                df=df,
                id_col=self.id_col,
                clock_col=self.clock_col,
                clock_no_col=self.clock_no_col,
                cols=self.cols,
                alpha=self.alpha,
                save_path=event_save_path / label,
                plot_library=self.plot_library,
                renderer=self.renderer,
            )

            # Perform a speced-down run that only runs analysis that don't depend on ordered continuous data (for example gap statistics breaks by the split)
            self.event_results[label].preliminary = (
                self.event_results[label]
                .preliminary.describe_df(save_results=save_results)
                .intervariable_normality(save_results=save_results)
                .shapiro_wilk(save_results=save_results)
                .correlation(save_results=save_results)
            )
            self.event_results[label].intravariable = (
                self.event_results[label]
                .intravariable.column_statistics(save_results=save_results)
                .date_time_statistics(save_results=save_results)
            )
            self.event_results[label].intervariable = (
                self.event_results[label]
                .intervariable.row_statistics(save_results=save_results)
                .symmetric_correlation(save_results=save_results)
                .mcar_test(save_results=save_results)
                .asymmetric_correlation(save_results=save_results)
            )

            if event_save_path:
                if label == "event":
                    title = f"Event Timestamps - {event_percentage:.2f}%"
                else:
                    title = f"Non-event Timestamps - {(100 - event_percentage):.2f}%"
                self.event_results[label].generate_html_reports(addition_to_title=title)

    def generate_html_reports(self, addition_to_title: str = None):
        """
        Generates HTML reports for the analysis results.
        """
        self.preliminary.generate_html_report(title=addition_to_title)
        self.intravariable.generate_html_report(title=addition_to_title)
        self.intervariable.generate_html_report(title=addition_to_title)
        pretty_printing.rich_info("HTML reports generated.")

    def _generate_clock_no_col(self):
        """Generates a clock number column based on ordered timestamps (clock_col)."""
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

    def _initial_check(self):
        """
        Performs initial checks on the dataframe to ensure it has the necessary columns.
        Raises:
            ValueError: If required columns are missing.
        """
        if self.df.height < 50:
            pretty_printing.rich_warning(
                "The dataframe has less than 50 rows. "
                "This may not be sufficient for a meaningful analysis."
            )
        if self.id_col not in self.df.columns:
            pretty_printing.rich_warning(
                f"Column '{self.id_col}' not found in the dataframe. "
                "This column is required for analysis."
            )
        if self.clock_col not in self.df.columns:
            pretty_printing.rich_warning(
                f"Column '{self.clock_col}' not found in the dataframe. "
                "A time column is required for analysis."
            )
        else:
            ## Check if the clock_col has null values, if so remove them
            clock_null_rows = self.df.filter(pl.col(self.clock_col).is_null())
            if clock_null_rows.height > 0:
                pretty_printing.rich_warning(
                    f"Time column '{self.clock_col}' contains {clock_null_rows.height} null values. "
                    f"{clock_null_rows.height} rows with null values will be removed."
                )
                self.df = self.df.filter(pl.col(self.clock_col).is_not_null())
        ## Check if only one column is provided for analysis
        if len(self.cols) == 1:
            pretty_printing.rich_warning(
                "Only one column specified for analysis. "
                "Consider providing more columns for a comprehensive analysis."
            )
        ## Check if all values are null in a column of cols and raise a warning
        for col in self.cols:
            if col not in self.df.columns:
                pretty_printing.rich_warning(
                    f"Column '{col}' not found in the dataframe. "
                    "This column was passed in the Preliminary init. Check cols parameter."
                )
                self.cols.remove(col)
            elif self.df[col].is_null().all():
                pretty_printing.rich_warning(
                    f"Column '{col}' contains all null values / no observed values."
                )
            ## Check if all values are observed / no missing data
            elif not self.df[col].is_null().any():
                pretty_printing.rich_warning(
                    f"Column '{col}' has no missing data. Missingness analysis won't be feasible."
                )
