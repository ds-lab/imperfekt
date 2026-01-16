import numpy as np
import plotly.graph_objects as go
import polars as pl


def markov_chain_summary(
    mask_df: pl.DataFrame,
    col: str,
    id_col: str = "id",
    clock_no_col: str = "clock_no",
    labels: list = ["Observed", "Imperfect"],
    save_path: str = None,
    save_results: bool = True,
) -> dict:
    """
    Summarize the imperfection pattern of a variable as a two-state Markov chain.
    Example:
    Computes the transition probabilities between observed and imperfect values
    in a time-ordered sequence for each ID, and returns the transition matrix,
    transition counts, steady-state distribution, and labels for the states.

    Parameters:
        mask_df (pl.DataFrame): DataFrame with a mask indicating imperfect values (1=missing/noisy/indicated, 0=observed/normal).
        col (str): Column to analyze.
        id_col (str): ID column.
        clock_no_col (str): Time ordering column.
        save_path (str, optional): Path to save the results. If None, results are not saved.

    Returns:
        dict: A dictionary containing:
            - transition_matrix: Transition probabilities between states.
                Transition Matrix (rows: from, cols: to):
                    [[0.95 0.05]
                    [0.20 0.80]]
                    0.95: If a value is observed, there’s a 95% chance the next value is also observed.
                    0.05: If a value is observed, there’s a 5% chance the next value is imperfect.
                    0.20: If a value is imperfect, there’s a 20% chance the next value is observed.
                    0.80: If a value is imperfect, there’s an 80% chance the next value is also imperfect.
            - transition_counts: Counts of transitions between states.
            - steady_state: Steady-state distribution of the Markov chain.
                Steady-state probabilities (Observed, Imperfect):
                    [0.85, 0.15]
                    0.85: Over the long run, 85% of values are observed.
                    0.15: 15% of values are imperfect.
            - labels: Labels for the states (e.g., "Observed", "Imperfect").
    """
    # Create current and next state columns
    transitions_df = (
        mask_df.sort(id_col, clock_no_col)
        .with_columns(
            current_state=pl.col(col),
        )
        .with_columns(
            next_state=pl.col("current_state").shift(-1).over(id_col),
        )
        .filter(pl.col("next_state").is_not_null())
    )

    counts_df = (
        transitions_df.group_by("current_state", "next_state")
        .agg(pl.len().alias("n"))
        .pivot(values="n", index="current_state", on="next_state")
        .fill_null(0)
        .sort("current_state")
    )

    # Ensure columns 0 and 1 exist
    for c in [0, 1]:
        if str(c) not in counts_df.columns:
            counts_df = counts_df.with_columns(pl.lit(0).alias(str(c)))
    counts_df = counts_df.select(["0", "1"])

    transition_counts = counts_df.to_numpy().astype(int)  # shape (2,2)

    # Normalize rows -> transition probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = transition_counts / row_sums
    transition_matrix = np.nan_to_num(probs)

    # Calculate steady-state distribution
    try:
        eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
        idx = np.isclose(eigvals, 1)
        if not idx.any():
            steady_state = np.full(2, np.nan)
        else:
            v = np.real(eigvecs[:, idx][:, 0])
            steady_state = v / v.sum()
    except np.linalg.LinAlgError:
        steady_state = np.full(2, np.nan)

    # Ensure the save_path directory exists
    if save_path and save_results:
        with open(save_path, "w") as f:
            f.write("Transition Matrix:\n")
            np.savetxt(f, transition_matrix, fmt="%.3f", delimiter=", ")
            f.write("\nTransition Counts:\n")
            np.savetxt(f, transition_counts, fmt="%d", delimiter=", ")
            f.write("\nSteady State Distribution:\n")
            np.savetxt(f, steady_state.reshape(1, -1), fmt="%.3f", delimiter=", ")

    return {
        "transition_matrix": transition_matrix,
        "transition_counts": transition_counts,
        "steady_state": steady_state,
        "labels": labels,
    }


def plot_markov_heatmap(
    probs,
    labels,
    title="Markov Chain Transition Matrix",
    renderer="browser",
    save_path: str = None,
    save_results: bool = True,
) -> go.Figure:
    """
    Plot a Markov chain transition matrix as a Plotly heatmap.

    Parameters:
        probs (np.ndarray): Transition matrix.
        labels (list): Labels for the states.
        title (str): Title of the heatmap.
        renderer (str): Renderer for Plotly (e.g., "browser", "notebook").
        save_path (str, optional): Path to save the heatmap image. If None, the image is not saved.
        save_results (bool): Whether to save the figure. Defaults to False.

    Returns:
        None: Displays the heatmap.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=probs,
            x=labels,
            y=labels,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in probs],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title=title, xaxis_title="To State", yaxis_title="From State")

    if renderer:
        fig.show(renderer=renderer)

    if save_results and save_path:
        if not save_path.suffix == ".png":
            save_path += ".png"

        fig.write_image(save_path)
        print(f"Markov heatmap saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    data = {
        "id": [
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
        ],
        "clock_no": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        "clock": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        "vital": [
            None,
            2.0,
            None,
            4.0,
            None,
            4.0,
            None,
            5.0,
            None,
            None,
            4.0,
            None,
            None,
            4.0,
            None,
            None,
        ],
    }
    df = pl.DataFrame(data)
    mask_df = df.with_columns(vital=pl.col("vital").is_null().cast(pl.Int8).alias("vital"))

    result = markov_chain_summary(mask_df, col="vital")
    print("Transition Matrix:\n", result["transition_matrix"])
    print("Steady State Distribution:\n", result["steady_state"])
    print("Labels:", result["labels"])
