from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import plotly.graph_objects as go


############################################################
#        Per-Case and Global Interval Statistics           #
############################################################


def compute_case_interval_statistics(
    delta_t_df: pl.DataFrame,
    id_col: str = "id",
) -> pl.DataFrame:
    """
    Compute per-case summary statistics of inter-observation intervals.

    Parameters:
        delta_t_df (pl.DataFrame): DataFrame with columns [id_col, ..., "interval_seconds"],
                                   one row per interval (already filtered: not-null, > 0).
        id_col (str): Case identifier column.

    Returns:
        pl.DataFrame: One row per case with columns:
            id, n_intervals, mean_seconds, median_seconds, std_seconds,
            cv, iqr_seconds, min_seconds, max_seconds.
    """
    case_stats = (
        delta_t_df
        .group_by(id_col)
        .agg(
            pl.len().alias("n_intervals"),
            pl.col("interval_seconds").mean().alias("mean_seconds"),
            pl.col("interval_seconds").median().alias("median_seconds"),
            pl.col("interval_seconds").std().alias("std_seconds"),
            pl.col("interval_seconds").quantile(0.25, interpolation="nearest").alias("q25_seconds"),
            pl.col("interval_seconds").quantile(0.75, interpolation="nearest").alias("q75_seconds"),
            pl.col("interval_seconds").min().alias("min_seconds"),
            pl.col("interval_seconds").max().alias("max_seconds"),
        )
        .with_columns(
            (pl.col("q75_seconds") - pl.col("q25_seconds")).alias("iqr_seconds"),
            ((pl.col("q75_seconds") - pl.col("q25_seconds")) / (pl.col("q75_seconds") + pl.col("q25_seconds"))).alias("qcod"),
            pl.when(pl.col("mean_seconds") != 0)
            .then(pl.col("std_seconds") / pl.col("mean_seconds"))
            .otherwise(None)
            .alias("cv"),
        )
        .drop(["q25_seconds", "q75_seconds"])
        .select([
            id_col,
            "n_intervals",
            "mean_seconds",
            "median_seconds",
            "std_seconds",
            "cv",
            "iqr_seconds",
            "qcod",
            "min_seconds",
            "max_seconds",
        ])
        .sort(id_col)
    )

    # Preserve cases that have 0 intervals (single-observation cases)
    all_cases = delta_t_df.select(pl.col(id_col).unique()).sort(id_col)
    case_stats = all_cases.join(case_stats, on=id_col, how="left")

    return case_stats


def compute_global_interval_statistics(
    delta_t_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute summary statistics over all pooled inter-observation intervals.

    Parameters:
        delta_t_df (pl.DataFrame): DataFrame with an "interval_seconds" column.

    Returns:
        pl.DataFrame: describe()-style summary over the interval_seconds column.
    """
    return delta_t_df.select("interval_seconds").describe(interpolation="linear")


############################################################
#         Dominant Frequency Analysis                      #
############################################################


def compute_dominant_frequency(
    delta_t_df: pl.DataFrame,
    bin_resolution_seconds: float = 60.0,
    adherence_tolerance: float = 0.5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Identify the modal inter-observation interval and quantify adherence and entropy.

    Parameters:
        delta_t_df (pl.DataFrame): DataFrame with an "interval_seconds" column.
        bin_resolution_seconds (float): Bin width in seconds for discretizing intervals.
        adherence_tolerance (float): Fraction tolerance around the mode for adherence rate
                                     (e.g. 0.5 means within [mode*0.5, mode*1.5]).

    Returns:
        tuple:
            frequency_summary (pl.DataFrame): Single-row summary with columns:
                - dominant_interval_seconds: center of the most frequent interval bin (the modal interval)
                - dominant_interval_bin: integer bin index of the mode
                - adherence_rate: fraction of all intervals within
                  [dominant * (1 - tolerance), dominant * (1 + tolerance)].
                  Answers "how often does the data follow its own dominant rhythm?"
                  — values near 1.0 indicate a consistent schedule, even with small jitter.
                - n_total_intervals: total number of inter-observation intervals
                - n_adhering_intervals: count of intervals within the adherence band
                - interval_entropy_bits: Shannon entropy H = -sum(p_i * log2(p_i)) over all bins.
                  Measures how spread out the interval distribution is, regardless of which bin
                  dominates. H = 0 when all intervals fall in one bin (perfectly regular).
                - normalized_entropy: H / log2(n_unique_bins). Rescaled to [0, 1]:
                  0 = perfectly regular (single bin), 1 = maximally irregular (uniform distribution).
                  Unlike adherence_rate, captures how chaotic the non-dominant portion is.
                - n_unique_bins: number of distinct interval bins observed
                - bin_resolution_seconds: bin width used for discretization
                - adherence_tolerance: fractional tolerance used for adherence_rate

            bin_counts (pl.DataFrame): Frequency table of all interval bins, sorted by count desc.
                Columns: interval_bin, interval_seconds_bin_center, count, fraction.
    """
    binned = delta_t_df.with_columns(
        (pl.col("interval_seconds") / bin_resolution_seconds)
        .round(0)
        .cast(pl.Int64)
        .alias("interval_bin")
    )

    bin_counts = (
        binned
        .group_by("interval_bin")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("interval_bin") * bin_resolution_seconds).alias("interval_seconds_bin_center"),
        )
        .sort(["count", "interval_bin"], descending=[True, False])
    )

    n_total = bin_counts["count"].sum()
    bin_counts = bin_counts.with_columns(
        (pl.col("count") / n_total).alias("fraction")
    ).select(["interval_bin", "interval_seconds_bin_center", "count", "fraction"])

    # Dominant bin = mode
    dominant_row = bin_counts.row(0, named=True)
    dominant_bin = dominant_row["interval_bin"]
    dominant_interval_seconds = dominant_row["interval_seconds_bin_center"]

    # Adherence: fraction of all intervals within [mode*(1-tol), mode*(1+tol)]
    lo = dominant_interval_seconds * (1 - adherence_tolerance)
    hi = dominant_interval_seconds * (1 + adherence_tolerance)
    n_adhering = (
        delta_t_df
        .filter(pl.col("interval_seconds").is_between(lo, hi))
        .height
    )
    adherence_rate = n_adhering / n_total if n_total > 0 else None

    # Shannon entropy of bin distribution
    fractions = bin_counts["fraction"].to_numpy()
    fractions = fractions[fractions > 0]
    entropy_bits = float(-np.sum(fractions * np.log2(fractions)))
    n_unique_bins = len(fractions)
    normalized_entropy = (entropy_bits / np.log2(n_unique_bins)) if n_unique_bins > 1 else 0.0

    frequency_summary = pl.DataFrame({
        "dominant_interval_seconds": [dominant_interval_seconds],
        "dominant_interval_bin": [dominant_bin],
        "adherence_rate": [adherence_rate],
        "n_total_intervals": [n_total],
        "n_adhering_intervals": [n_adhering],
        "interval_entropy_bits": [entropy_bits],
        "normalized_entropy": [normalized_entropy],
        "n_unique_bins": [n_unique_bins],
        "bin_resolution_seconds": [bin_resolution_seconds],
        "adherence_tolerance": [adherence_tolerance],
    })

    return frequency_summary, bin_counts


############################################################
#        Per-Case Entropy and Adherence                    #
############################################################


def compute_case_interval_entropy_adherence(
    delta_t_df: pl.DataFrame,
    id_col: str = "id",
    bin_resolution_seconds: float = 60.0,
    adherence_tolerance: float = 0.5,
    min_intervals: int = 2,
) -> pl.DataFrame:
    """
    Compute per-case Shannon entropy and adherence rate of the interval distribution.

    Entropy measures how spread out each case's own interval lengths are.
    Entropy = 0 means all intervals fall in one bin (perfectly regular for that case,
    regardless of what that bin length is or what the dataset average looks like).

    Adherence measures how consistently each case follows *its own* dominant interval
    (not the dataset-wide dominant). A case with a unique but perfectly consistent
    rhythm scores adherence = 1.0.

    Uses the same binning logic as compute_dominant_frequency() but applied
    independently to each case's interval sequence.

    Parameters:
        delta_t_df (pl.DataFrame): DataFrame with columns [id_col, ..., "interval_seconds"],
                                   one row per interval (not-null, > 0).
        id_col (str): Case identifier column.
        bin_resolution_seconds (float): Bin width in seconds for discretizing intervals.
        adherence_tolerance (float): Fractional tolerance around each case's own dominant
                                     interval for adherence_rate computation. E.g. 0.5 means
                                     within [dominant * 0.5, dominant * 1.5].
        min_intervals (int): Minimum number of intervals required to compute metrics.
                             Cases with fewer intervals receive NaN for all metrics.

    Returns:
        pl.DataFrame: One row per case with columns:
            id, entropy_bits, normalized_entropy, adherence_rate, n_adhering_intervals.
            Cases with fewer than min_intervals intervals receive NaN for all metrics.
    """
    # All cases — needed to preserve those with insufficient intervals
    all_cases = delta_t_df.select(pl.col(id_col).unique()).sort(id_col)

    # Step 1: bin every interval row
    binned = delta_t_df.with_columns(
        (pl.col("interval_seconds") / bin_resolution_seconds)
        .round(0)
        .cast(pl.Int64)
        .alias("interval_bin")
    )

    # Step 2: per-case per-bin counts
    case_bin_counts = (
        binned
        .group_by([id_col, "interval_bin"])
        .agg(pl.len().alias("bin_count"))
    )

    # Step 3: per-case total intervals
    case_totals = (
        binned
        .group_by(id_col)
        .agg(pl.len().alias("n_total"))
    )

    # Step 4: join totals back and compute per-bin fraction
    case_bin_counts = (
        case_bin_counts
        .join(case_totals, on=id_col, how="left")
        .with_columns(
            (pl.col("bin_count").cast(pl.Float64) / pl.col("n_total")).alias("fraction")
        )
        .with_columns(
            # entropy contribution: -p * log2(p); safe because fraction > 0 by construction
            (-pl.col("fraction") * pl.col("fraction").log(base=2.0)).alias("entropy_contrib")
        )
    )

    # Step 5: per-case entropy, dominant bin, and n_unique_bins
    case_entropy = (
        case_bin_counts
        .group_by(id_col)
        .agg(
            pl.col("entropy_contrib").sum().alias("entropy_bits"),
            pl.col("interval_bin").count().cast(pl.Int64).alias("n_unique_bins"),
            # dominant bin = the bin_count argmax (sort descending, take first)
            pl.col("interval_bin")
              .sort_by(["bin_count", "interval_bin"], descending=True)
              .first()
              .alias("dominant_bin"),
        )
        .join(case_totals, on=id_col, how="left")
    )

    # Step 6: normalized entropy — scale to [0, 1]; 0 when n_unique_bins == 1
    case_entropy = case_entropy.with_columns(
        pl.when(pl.col("n_unique_bins") > 1)
        .then(
            pl.col("entropy_bits")
            / pl.col("n_unique_bins").cast(pl.Float64).log(base=2.0)
        )
        .otherwise(0.0)
        .alias("normalized_entropy")
    )

    # Step 7: per-case adherence against each case's own dominant interval
    case_entropy = case_entropy.with_columns(
        (pl.col("dominant_bin") * bin_resolution_seconds).alias("dominant_interval_seconds")
    )

    binned_with_dominant = binned.join(
        case_entropy.select([id_col, "dominant_interval_seconds"]),
        on=id_col,
        how="left",
    )

    adherence_df = (
        binned_with_dominant
        .with_columns(
            pl.col("interval_seconds")
            .is_between(
                pl.col("dominant_interval_seconds") * (1.0 - adherence_tolerance),
                pl.col("dominant_interval_seconds") * (1.0 + adherence_tolerance),
            )
            .cast(pl.Int32)
            .alias("adheres")
        )
        .group_by(id_col)
        .agg(
            pl.col("adheres").sum().alias("n_adhering_intervals"),
            pl.len().alias("n_total_for_adherence"),
        )
        .with_columns(
            (pl.col("n_adhering_intervals").cast(pl.Float64) / pl.col("n_total_for_adherence"))
            .alias("adherence_rate")
        )
    )

    # Step 8: assemble and left-join all_cases to preserve those with 0 intervals
    result = (
        all_cases
        .join(
            case_entropy.select([id_col, "entropy_bits", "normalized_entropy"]),
            on=id_col,
            how="left",
        )
        .join(
            adherence_df.select([id_col, "adherence_rate", "n_adhering_intervals"]),
            on=id_col,
            how="left",
        )
        .join(case_totals, on=id_col, how="left")
    )

    # Step 9: null out cases with fewer than min_intervals intervals
    result = (
        result
        .with_columns(
            pl.when(pl.col("n_total").fill_null(0) < min_intervals)
            .then(None)
            .otherwise(pl.col("entropy_bits"))
            .alias("entropy_bits"),
            pl.when(pl.col("n_total").fill_null(0) < min_intervals)
            .then(None)
            .otherwise(pl.col("normalized_entropy"))
            .alias("normalized_entropy"),
            pl.when(pl.col("n_total").fill_null(0) < min_intervals)
            .then(None)
            .otherwise(pl.col("adherence_rate"))
            .alias("adherence_rate"),
            pl.when(pl.col("n_total").fill_null(0) < min_intervals)
            .then(None)
            .otherwise(pl.col("n_adhering_intervals").cast(pl.Float64))
            .alias("n_adhering_intervals"),
        )
        .drop("n_total")
        .select([id_col, "entropy_bits", "normalized_entropy", "adherence_rate", "n_adhering_intervals"])
        .sort(id_col)
    )

    return result


############################################################
#        Interval Frequency Bar Chart                      #
############################################################


def plot_interval_frequency_bar(
    bin_counts: pl.DataFrame,
    dominant_bin: int,
    top_n: int = 20,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Bar chart of interval bin frequencies, with the dominant bin highlighted.

    Parameters:
        bin_counts (pl.DataFrame): Output of compute_dominant_frequency(), sorted by count desc.
        dominant_bin (int): The bin index of the dominant (modal) interval.
        top_n (int): Number of top bins to display.
        library (str): "matplotlib" or "plotly".
        renderer (str): Plotly renderer. None disables display.
        save_path (str): File path to save the figure. None disables saving.
        save_results (bool): Whether to save the figure.

    Returns:
        Figure object (matplotlib or plotly).
    """
    top = bin_counts.head(top_n).sort("interval_seconds_bin_center")
    x_vals = top["interval_seconds_bin_center"].to_numpy()
    y_vals = top["count"].to_numpy()
    colors = ["#d62728" if b == dominant_bin else "#1f77b4" for b in top["interval_bin"].to_list()]
    labels = [f"{v:.0f}s" for v in x_vals]

    title = "Inter-Observation Interval Frequency (top bins)"
    xaxis_title = "Interval (seconds)"
    yaxis_title = "Count"

    if library.lower() == "plotly":
        fig = go.Figure(
            go.Bar(
                x=x_vals,
                y=y_vals,
                marker_color=colors,
                text=[f"{v:.3f}" for v in top["fraction"].to_list()],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template="plotly_white",
            xaxis=dict(tickvals=x_vals.tolist(), ticktext=labels),
        )
        if renderer:
            fig.show(renderer=renderer)
        if save_results and save_path:
            save_path = Path(save_path)
            fig.write_image(save_path)
            print(f"Interval frequency bar chart saved to {save_path}")
        return fig

    elif library.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.6), 6))
        ax.bar(range(len(x_vals)), y_vals, color=colors)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel(xaxis_title)
        ax.set_ylabel(yaxis_title)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        if save_results and save_path:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Interval frequency bar chart saved to {save_path}")
        if renderer:
            plt.show()
        plt.close(fig)
        return fig
    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")
