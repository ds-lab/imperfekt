from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl


def plot_histogram(
    df: pl.DataFrame,
    x: str,
    title: str | None = None,
    nbins: int = 50,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic histogram function for visualizing the distribution of a column.

    Parameters:
        df (pl.DataFrame): DataFrame containing the data to plot.
        x (str): Column name to plot on the x-axis.
        title (str): Title of the plot.
        nbins (int): Number of bins for the histogram, default is 50.
        xaxis_title (str): Title for the x-axis.
        yaxis_title (str): Title for the y-axis.
        library (str): Visualization library to use, default is "matplotlib". Other options include "matplotlib".
        renderer (str): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
                        Available renderers:
                        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                        'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                        'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                        'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                        'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        save_path (str): Path to save the plot image, default is None.
        save_results (bool): Whether to save the plot image, default is True.

    Returns:
        None: Displays the histogram.
    """
    # Input validation
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame")
    if df[x].dtype not in [pl.Float64, pl.Int64]:
        raise TypeError(f"Column '{x}' must be of type Float64 or Int64, got {df[x].dtype}")

    if library.lower() == "plotly":
        # Create histogram
        fig = go.Figure(go.Histogram(x=df[x].to_numpy(), nbinsx=nbins))

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template="plotly_white",
        )

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.write_image(save_path)
            print(f"Histogram saved to {save_path}")

        return fig

    elif library.lower() == "matplotlib":
        # Create matplotlib histogram using the object-oriented API
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[x].to_numpy(), bins=nbins, alpha=0.7, edgecolor="black")

        if title:
            ax.set_title(title)
        if xaxis_title:
            ax.set_xlabel(xaxis_title)
        if yaxis_title:
            ax.set_ylabel(yaxis_title)
        else:
            ax.set_ylabel("Frequency")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Histogram saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)  # Close the figure to free up memory

        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def plot_boxplot(
    df: pl.DataFrame,
    y: str,
    x: str | None = None,
    title: str | None = None,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
    category_order: list[str] | None = None,
    boxpoints: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic boxplot function for visualizing distributions of a column.

    Parameters:
        df (pl.DataFrame): DataFrame containing the data to plot.
        y (str): Column name to plot on the y-axis.
        x (str | None): Column name to plot on the x-axis (for grouped boxplots), default is None.
        title (str | None): Title of the plot.
        yaxis_title (str | None): Title for the y-axis.
        xaxis_title (str | None): Title for the x-axis.
        category_order (list[str] | None): Order of categories for the x-axis, default is None.
        boxpoints (str | None): Type of boxpoints to show, default is None. Options include 'all', 'outliers', 'suspectedoutliers', 'false'.
                        If set to None, no boxpoints will be shown.
        library (str): Visualization library to use, default is "matplotlib". Other options include "matplotlib".
        renderer (str | None): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
                        Available renderers:
                        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                        'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                        'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                        'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                        'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        save_path (str | Path | None): Path to save the plot image, default is None.
        save_results (bool): Whether to save the plot image, default is True.

    Returns:
        None: Displays the boxplot.
    """
    # Input validation
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame")
    if x and x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame")

    if library.lower() == "plotly":
        fig = go.Figure(
            go.Box(
                x=df[x].to_numpy() if x else None,
                y=df[y].to_numpy(),
                boxpoints=boxpoints,  # Show all points
                jitter=0.3,  # Add some jitter to the points
                pointpos=-1.8,  # Position of the points relative to the box
                boxmean=True,  # Show mean line
            )
        )

        fig.update_layout(
            title=title,
            yaxis_title=yaxis_title,
            xaxis_title=xaxis_title,
            template="plotly_white",
        )

        if category_order is not None and x is not None:
            fig.update_xaxes(categoryorder="array", categoryarray=category_order)

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.write_image(save_path)
            print(f"Boxplot saved to {save_path}")

        return fig

    elif library.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=(10, 6))

        if x is None:
            # Simple boxplot for single variable
            box_data = [df[y].drop_nulls().to_numpy()]
            box_plot = ax.boxplot(box_data, patch_artist=True, showmeans=True, meanline=True)

            # Customize colors
            for patch in box_plot["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
        else:
            # Grouped boxplot
            groups = df[x].unique().to_list()
            if category_order:
                # Reorder groups if category_order is provided
                groups = [g for g in category_order if g in groups]

            box_data = []
            labels = []
            for group in groups:
                group_data = df.filter(pl.col(x) == group)[y].drop_nulls().to_numpy()
                if len(group_data) > 0:
                    box_data.append(group_data)
                    labels.append(str(group))

            box_plot = ax.boxplot(
                box_data,
                tick_labels=labels,
                patch_artist=True,
                showmeans=True,
                meanline=True,
            )

            # Customize colors
            colors = [
                "lightblue",
                "lightgreen",
                "lightcoral",
                "lightyellow",
                "lightpink",
            ]
            for i, patch in enumerate(box_plot["boxes"]):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)

        if title:
            ax.set_title(title)
        if yaxis_title:
            ax.set_ylabel(yaxis_title)
        if xaxis_title:
            ax.set_xlabel(xaxis_title)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Boxplot saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)
        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def plot_violin(
    df: pl.DataFrame,
    y: str,
    x: str | None = None,
    title: str | None = None,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
    category_order: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic violin plot function for visualizing the distribution of a column.

    Parameters:
        df (pl.DataFrame): DataFrame containing the data to plot.
        y (str): Column name to plot on the y-axis.
        x (str | None): Column name to plot on the x-axis (for grouped violin plots), default is None.
        title (str | None): Title of the plot.
        yaxis_title (str | None): Title for the y-axis.
        xaxis_title (str | None): Title for the x-axis.
        category_order (list[str] | None): Order of categories for the x-axis, default is None.
        library (str): Visualization library to use, default is "matplotlib". Other options include "plotly".
        renderer (str | None): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
    """
    # Input validation
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame")
    if x and x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame")

    if library.lower() == "plotly":
        fig = go.Figure(
            go.Violin(
                x=df[x].to_numpy() if x else None,
                y=df[y].to_numpy(),
                box_visible=True,
                line_color="black",
                meanline_visible=True,
            )
        )

        fig.update_layout(
            title=title,
            yaxis_title=yaxis_title,
            xaxis_title=xaxis_title,
            template="plotly_white",
        )

        if category_order is not None and x is not None:
            fig.update_xaxes(categoryorder="array", categoryarray=category_order)

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.write_image(save_path)
            print(f"Boxplot saved to {save_path}")
        return fig

    elif library.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=(10, 6))

        if x is None:
            # Simple violin plot for single variable
            data = [df[y].drop_nulls().to_numpy()]
            ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
        else:
            # Grouped violin plot
            groups = df[x].unique().to_list()
            if category_order:
                # Reorder groups if category_order is provided
                groups = [g for g in category_order if g in groups]

            data = []
            labels = []
            for group in groups:
                group_data = df.filter(pl.col(x) == group)[y].drop_nulls().to_numpy()
                if len(group_data) > 0:
                    data.append(group_data)
                    labels.append(str(group))

            ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)

        if title:
            ax.set_title(title)
        if yaxis_title:
            ax.set_ylabel(yaxis_title)
        if xaxis_title:
            ax.set_xlabel(xaxis_title)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Violin plot saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)
        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "lines+markers",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic scatter plot function for visualizing the relationship between two columns.
    Parameters:
        x (np.ndarray): Data to plot on the x-axis.
        y (np.ndarray): Data to plot on the y-axis.
        mode (str): Mode for the scatter plot, default is 'lines+markers'. Other options include 'markers', 'lines', etc.
        title (str | None): Title of the plot.
        xaxis_title (str | None): Title for the x-axis.
        yaxis_title (str): Title for the y-axis.
        library (str): Visualization library to use, default is "matplotlib". Other options include "matplotlib".
        renderer (str): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
                        Available renderers:
                        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                        'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                        'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                        'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                        'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        save_path (str): Path to save the plot image, default is None.
        save_results (bool): Whether to save the plot image, default is True.
    Returns:
        None: Displays the scatter plot.
    """
    if library.lower() == "plotly":
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=y,
                mode=mode,  # Use markers for scatter plot
                marker=dict(size=5, color="blue", opacity=0.6),  # Customize marker appearance
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template="plotly_white",
        )

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.write_image(save_path)
            print(f"Scatter plot saved to {save_path}")
        return fig

    elif library.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=(10, 6))

        if mode == "markers" or mode == "lines+markers":
            ax.scatter(x, y, alpha=0.6, s=20, color="blue")

        if mode == "lines" or mode == "lines+markers":
            # Sort data for line plot
            sorted_indices = np.argsort(x)
            ax.plot(x[sorted_indices], y[sorted_indices], color="blue", alpha=0.8)

        if title:
            ax.set_title(title)
        if xaxis_title:
            ax.set_xlabel(xaxis_title)
        if yaxis_title:
            ax.set_ylabel(yaxis_title)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Scatter plot saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)
        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def plot_overlay_histograms(
    dfs: list[pl.DataFrame],
    x: str,
    group_names: list[str] | None = None,
    title: str | None = None,
    nbins: int = 50,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    histnorm: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Overlay histograms of two DataFrames for comparison.
    Parameters:
        dfs (list[pl.DataFrame]): List of DataFrames containing the data to plot.
        x (str): Column name to plot on the x-axis.
        group_names (list[str]): Names for each group, default is None which will generate generic names.
        title (str): Title of the plot.
        nbins (int): Number of bins for the histogram, default is 50.
        xaxis_title (str): Title for the x-axis.
        yaxis_title (str): Title for the y-axis.
        histnorm (str): Normalization method for the histogram, default is None.
        library (str): Visualization library to use, default is "matplotlib". Other options include "matplotlib".
        renderer (str): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
                        Available renderers:
                        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                        'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                        'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                        'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                        'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        save_path (str): Path to save the plot image, default is None.
        save_results (bool): Whether to save the plot image, default is True.

    Returns:
        None: Displays the overlayed histograms.
    """
    if group_names is None:
        group_names = [f"Group {i + 1}" for i in range(len(dfs))]

    # Collect all data points across DataFrames
    all_data = []
    for df in dfs:
        if x not in df.columns:
            raise ValueError(f"Column '{x}' not found in DataFrame")
        if df[x].dtype not in [pl.Float64, pl.Int64]:
            raise TypeError(f"Column '{x}' must be of type Float64 or Int64, got {df[x].dtype}")
        col_data = df[x].to_numpy()
        if len(col_data) > 0:  # Only extend if there's data
            all_data.extend(col_data)

    if len(all_data) == 0:
        raise ValueError("No data available for column")

    if library.lower() == "plotly":
        min_val = min(all_data)
        max_val = max(all_data)
        range_size = max_val - min_val

        # Add padding (5% of range on each side)
        padding = 0.05 * range_size
        min_val_padded = min_val - padding
        max_val_padded = max_val + padding

        # Calculate fixed bin width based on padded range
        if max_val_padded == min_val_padded:
            # Handle case where all values are identical (avoid division by zero)
            bin_width = 1.0
        else:
            bin_width = (max_val_padded - min_val_padded) / nbins

        colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta"]
        hists = []
        for group_name, df in zip(group_names, dfs):
            if x not in df.columns:
                raise ValueError(f"Column '{x}' not found in DataFrame")
            hist = go.Histogram(
                x=df[x].to_numpy(),
                opacity=0.6,
                name=group_name,
                xbins=dict(size=bin_width),  # Fixed bin width
                histnorm=histnorm,
                marker=dict(color=colors[len(hists) % len(colors)]),
            )
            hists.append(hist)

        # Create layout with consistent x-axis range and overlay mode
        layout = go.Layout(
            title=title,
            barmode="overlay",
            xaxis=dict(title=xaxis_title, range=[min_val_padded, max_val_padded]),
            yaxis=dict(title=yaxis_title),
            template="plotly_white",
        )

        # Create figure and plot
        fig = go.Figure(data=hists, layout=layout)

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.write_image(save_path)
            print(f"Overlayed histograms saved to {save_path}")
        return fig

    elif library.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors for different groups
        colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta"]

        # Determine common bin range
        # Filter out NaN values from all_data
        all_data_filtered = [d for d in all_data if not np.isnan(d)]
        if not all_data_filtered:
            # If all data was NaN, there's nothing to plot.
            # We can return an empty figure or raise an error.
            # For now, let's just print a warning and return the empty figure.
            print(f"Warning: All data for column '{x}' is NaN. Cannot plot histogram.")
            return fig

        min_val = min(all_data_filtered)
        max_val = max(all_data_filtered)
        if min_val == max_val:
            bins = [min_val - 0.5, min_val + 0.5]
        else:
            bins = np.linspace(min_val, max_val, nbins + 1)

        # Create overlaid histograms
        for i, (df, group_name) in enumerate(zip(dfs, group_names)):
            data = df[x].drop_nulls().to_numpy()
            if len(data) == 0:
                continue

            # Apply normalization if specified
            if histnorm == "probability":
                weights = np.ones_like(data) / len(data)
            elif histnorm == "density":
                weights = None  # matplotlib handles this with density=True
            else:
                weights = None

            if histnorm == "density":
                ax.hist(
                    data,
                    bins=bins,
                    alpha=0.6,
                    label=group_name,
                    color=colors[i % len(colors)],
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                )
            else:
                ax.hist(
                    data,
                    bins=bins,
                    alpha=0.6,
                    label=group_name,
                    color=colors[i % len(colors)],
                    weights=weights,
                    edgecolor="black",
                    linewidth=0.5,
                )

        if title:
            ax.set_title(title)
        if xaxis_title:
            ax.set_xlabel(xaxis_title)
        if yaxis_title:
            ax.set_ylabel(yaxis_title)
        else:
            if histnorm == "probability":
                ax.set_ylabel("Probability")
            elif histnorm == "density":
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("Frequency")

        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Overlayed histograms saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)

        return fig
    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def plot_multi_boxplot(
    dfs: list[pl.DataFrame],
    y: str,
    group_names: list[str] | None = None,
    title: str | None = None,
    yaxis_title: str | None = None,
    boxpoints: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Overlay boxplots of two DataFrames for comparison.

    Parameters:
        dfs (list[pl.DataFrame]): List of DataFrames containing the data to plot.
        y (str): Column name to plot on the y-axis.
        group_names (list[str]): Names for each group, default is None which will generate generic names.
        title (str): Title of the plot.
        yaxis_title (str): Title for the y-axis.
        boxpoints (str): Type of boxpoints to show, default is None. Options include 'all', 'outliers', 'suspectedoutliers', 'false'.
        library (str): Visualization library to use, default is "matplotlib". Other options include "matplotlib".
        renderer (str): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
                        Available renderers:
                        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
                        'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
                        'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
                        'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
                        'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
        save_path (str | Path | None): Path to save the plot image, default is None.
        save_results (bool): Whether to save the plot image, default is True.

    Returns:
        None: Displays the overlayed boxplots.
    """
    if group_names is None:
        group_names = [f"Group {i + 1}" for i in range(len(dfs))]

    # Input validation
    for df in dfs:
        if y not in df.columns:
            raise ValueError(f"Column '{y}' not found in DataFrame")

    if library.lower() == "plotly":
        colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta"]

        boxs = []
        for group_name, df in zip(group_names, dfs):
            if boxpoints == "all":
                jitter = 0.3  # Add some jitter to the points
                pointpos = -1.8
            else:
                jitter = None
                pointpos = None

            box = go.Box(
                y=df[y].to_numpy(),
                name=group_name,
                boxpoints=boxpoints,  # Show all points
                jitter=jitter,  # Add some jitter to the points
                pointpos=pointpos,  # Position of the points relative to the box
                marker=dict(color=colors[len(boxs) % len(colors)]),  # Cycle through colors
                boxmean=True,  # Show mean line
            )
            boxs.append(box)

        # Create layout with overlay mode
        layout = go.Layout(
            title=title,
            yaxis=dict(title=yaxis_title),
            template="plotly_white",
        )

        # Create figure and plot
        fig = go.Figure(data=boxs, layout=layout)

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.write_image(save_path)
            print(f"Multi-boxplot saved to {save_path}")
        return fig

    elif library.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for matplotlib boxplot
        box_data = []
        labels = []
        colors = [
            "lightblue",
            "lightgreen",
            "lightcoral",
            "lightyellow",
            "lightpink",
            "lightgray",
            "lightcyan",
        ]

        for group_name, df in zip(group_names, dfs):
            data = df[y].drop_nulls().to_numpy()
            if len(data) > 0:
                box_data.append(data)
                labels.append(group_name)

        if not box_data:
            raise ValueError("No valid data found for any group")

        # Create boxplots
        box_plot = ax.boxplot(
            box_data,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        # Customize colors
        for i, patch in enumerate(box_plot["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)

        # Add individual points if requested
        if boxpoints == "all":
            for i, data in enumerate(box_data):
                # Add jitter to x-coordinates
                x_pos = i + 1  # matplotlib boxplot positions start at 1
                x_jitter = np.random.normal(x_pos, 0.04, size=len(data))
                ax.scatter(x_jitter, data, alpha=0.4, s=10, color="black")

        if title:
            ax.set_title(title)
        if yaxis_title:
            ax.set_ylabel(yaxis_title)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Multi-boxplot saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)
        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def plot_qq(
    data: np.ndarray,
    dist: str = "norm",
    title: str = "QQ Plot",
    xaxis_title: str = "Theoretical Quantiles",
    yaxis_title: str = "Sample Quantiles",
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "plt.Figure":
    """
    Generic QQ plot function for visualizing quantile-quantile plots.

    Parameters:
        data (np.ndarray): Data to compare against the theoretical distribution.
        dist (str): Theoretical distribution to compare against, default is "norm" (normal distribution).
        title (str | None): Title of the plot.
        xaxis_title (str | None): Title for the x-axis.
        yaxis_title (str | None): Title for the y-axis.
        library (str): Visualization library to use, default is "matplotlib".
        renderer (str | None): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
        save_path (str | Path | None): Path to save the plot image, default is None.
        save_results (bool): Whether to save the plot image, default is True.

    Returns:
        None: Displays the QQ plot.
    """
    import scipy.stats as stats

    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(data, dist=dist, plot=ax)

    if title:
        ax.set_title(title)
    if xaxis_title:
        ax.set_xlabel(xaxis_title)
    if yaxis_title:
        ax.set_ylabel(yaxis_title)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_results and save_path:
        save_path = Path(save_path)
        if save_path.suffix != ".png":
            save_path = save_path.with_suffix(".png")

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"QQ plot saved to {save_path}")

    if renderer:
        plt.show()

    plt.close(fig)
    return fig


# Colour cycle shared by the group-comparison figures, one colour per group.
_GROUP_COLORS = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
]


def plot_descriptives_boxplot(
    df: pl.DataFrame,
    group_col: str = "group",
    metric_col: str = "metric",
    facet_col: str | None = "variable",
    title: str | None = None,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Box plots of per-group distributions, drawn from precomputed descriptives.

    One panel per metric, since the metrics live on different scales and sharing a y-axis
    would flatten all but the widest of them. Within a panel the groups sit side by side at
    each facet value, which is the arrangement that makes a between-group shift readable.

    The boxes are built from the five-number summary in the descriptives frame
    (``q25``/``median``/``q75`` with Tukey whisker ends), not from the raw values, so the
    figure can be redrawn from a saved ``descriptives.csv`` alone. Outliers are therefore
    not drawn — they are past the whiskers by construction, and the ``min``/``max`` columns
    still carry the full range. The mean is marked with a dashed line, since these metrics
    are skewed often enough that the gap between mean and median is itself informative.

    Groups whose metric is undefined everywhere in a facet (``n_defined == 0``) are left out
    of that panel rather than drawn as an empty slot.

    Parameters:
        df (pl.DataFrame): Descriptives, one row per (facet value, metric, group). Requires
            the columns written by group_comparison.describe_groups.
        group_col (str): Column holding the group label — one box colour per value.
        metric_col (str): Column holding the metric name — one panel per value.
        facet_col (str | None): Column spread along the x-axis (e.g. "variable"). Ignored
            when absent or all-null, which is the unfaceted single-slot case.
        title (str): Suptitle of the figure.
        library (str): "matplotlib" or "plotly".
        renderer (str): Renderer for displaying the plot. None disables rendering.
        save_path (str): Path to save the plot image.
        save_results (bool): Whether to save the plot image.

    Returns:
        go.Figure | plt.Figure: The figure object.
    """
    required = [group_col, metric_col, "median", "q25", "q75", "whisker_low", "whisker_high"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    plot_df = df.filter(pl.col("median").is_not_null() & pl.col("median").is_not_nan())
    if "n_defined" in plot_df.columns:
        plot_df = plot_df.filter(pl.col("n_defined") > 0)
    if plot_df.height == 0:
        raise ValueError("No rows with a defined median to plot.")

    faceted = (
        facet_col is not None
        and facet_col in plot_df.columns
        and plot_df[facet_col].null_count() < plot_df.height
    )
    facet_values = sorted(plot_df[facet_col].drop_nulls().unique().to_list()) if faceted else [None]
    metrics = sorted(plot_df[metric_col].unique().to_list())
    groups = sorted(plot_df[group_col].unique().to_list())
    color_of = {g: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(groups)}

    def _rows_for(metric, group):
        sub = plot_df.filter((pl.col(metric_col) == metric) & (pl.col(group_col) == group))
        by_facet = {}
        for row in sub.iter_rows(named=True):
            by_facet[row[facet_col] if faceted else None] = row
        return by_facet

    if library.lower() == "plotly":
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=metrics,
            vertical_spacing=min(0.08, 0.6 / max(len(metrics), 1)),
        )
        for m_idx, metric in enumerate(metrics, start=1):
            for group in groups:
                by_facet = _rows_for(metric, group)
                present = [f for f in facet_values if f in by_facet]
                if not present:
                    continue
                rows = [by_facet[f] for f in present]
                fig.add_trace(
                    go.Box(
                        x=[str(f) if faceted else str(group) for f in present],
                        q1=[r["q25"] for r in rows],
                        median=[r["median"] for r in rows],
                        q3=[r["q75"] for r in rows],
                        lowerfence=[r["whisker_low"] for r in rows],
                        upperfence=[r["whisker_high"] for r in rows],
                        mean=[r.get("mean") for r in rows],
                        sd=[r.get("std") for r in rows],
                        name=str(group),
                        legendgroup=str(group),
                        showlegend=m_idx == 1,
                        boxmean=True,
                        marker={"color": color_of[group]},
                        # The per-box n varies, so it goes in the hover text rather than
                        # the axis labels, where it would only fit for the smallest facets.
                        text=[f"n defined {r.get('n_defined')}/{r.get('n')}" for r in rows],
                    ),
                    row=m_idx,
                    col=1,
                )

        fig.update_layout(
            title=title,
            boxmode="group",
            template="plotly_white",
            height=max(320, 260 * len(metrics)),
            legend_title_text=group_col,
        )

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")
            fig.write_image(save_path)
            print(f"Descriptives box plot saved to {save_path}")

        return fig

    elif library.lower() == "matplotlib":
        # Faceted: one slot per facet value, groups side by side within it. Unfaceted:
        # one slot per group, which reads better than a single crowded slot.
        n_slots = len(facet_values) if faceted else len(groups)
        slot_labels = [str(f) for f in facet_values] if faceted else [str(g) for g in groups]
        width = min(24, max(7, 1.4 * n_slots * (len(groups) if faceted else 1) + 3))
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(width, max(3.2, 2.9 * len(metrics))),
            squeeze=False,
        )
        axes = [ax for (ax,) in axes]

        box_width = 0.8 / len(groups) if faceted else 0.5

        for ax, metric in zip(axes, metrics):
            for g_idx, group in enumerate(groups):
                by_facet = _rows_for(metric, group)
                stats_list, positions = [], []
                for f_idx, facet_value in enumerate(facet_values):
                    row = by_facet.get(facet_value)
                    if row is None:
                        continue
                    stats_list.append(
                        {
                            "med": row["median"],
                            "q1": row["q25"],
                            "q3": row["q75"],
                            "whislo": row["whisker_low"],
                            "whishi": row["whisker_high"],
                            "mean": row.get("mean"),
                            "fliers": [],
                            "label": "",
                        }
                    )
                    if faceted:
                        positions.append(f_idx + (g_idx - (len(groups) - 1) / 2) * box_width)
                    else:
                        positions.append(g_idx)
                if not stats_list:
                    continue
                artists = ax.bxp(
                    stats_list,
                    positions=positions,
                    widths=box_width * 0.8,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in artists["boxes"]:
                    patch.set_facecolor(color_of[group])
                    patch.set_alpha(0.55)
                    patch.set_edgecolor(color_of[group])
                for median in artists["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.4)
                for mean in artists["means"]:
                    mean.set_color("black")
                    mean.set_linestyle(":")
                    mean.set_linewidth(1.0)

            ax.set_ylabel(metric)
            ax.set_xticks(range(n_slots))
            ax.set_xticklabels(
                slot_labels,
                rotation=45 if n_slots > 4 else 0,
                ha="right" if n_slots > 4 else "center",
            )
            ax.set_xlim(-0.5, n_slots - 0.5)
            ax.grid(True, alpha=0.3, axis="y")

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=9,
                alpha=0.55,
                color=color_of[g],
                label=str(g),
            )
            for g in groups
        ]
        if faceted:
            axes[0].legend(handles=handles, title=group_col, loc="best", fontsize=8)
        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Descriptives box plot saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)
        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


def _format_q_value(value) -> str:
    """q-value as text; very small values get a bound rather than a run of zeros."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _format_estimate_ci(estimate, lower, upper) -> str:
    """Point estimate with its interval, or the estimate alone when no interval is given."""

    def defined(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    def fmt(v):
        # A bound that came out of a rank-based search on tied values lands on cancellation
        # noise rather than on zero; printing "-1.11e-16" would suggest a precision the
        # estimate does not have.
        return f"{0.0 if abs(v) < 1e-12 else v:.3g}"

    if not defined(estimate):
        return ""
    if defined(lower) and defined(upper):
        return f"{fmt(estimate)} [{fmt(lower)}, {fmt(upper)}]"
    return fmt(estimate)


def _format_hl_column(
    plot_df: pl.DataFrame, hl_col: str | None, hl_ci_cols: tuple[str, str]
) -> list[str] | None:
    """Hodges-Lehmann annotation per row, or None when the frame does not carry it."""
    if not hl_col or hl_col not in plot_df.columns:
        return None
    estimates = plot_df[hl_col].to_list()
    lo_col, hi_col = hl_ci_cols
    lowers = plot_df[lo_col].to_list() if lo_col in plot_df.columns else [None] * len(estimates)
    uppers = plot_df[hi_col].to_list() if hi_col in plot_df.columns else [None] * len(estimates)
    texts = [_format_estimate_ci(e, lo, hi) for e, lo, hi in zip(estimates, lowers, uppers)]
    # All-blank means k > 2 groups throughout: the column would be an empty header.
    return texts if any(texts) else None


def plot_effect_size_forest(
    df: pl.DataFrame,
    effect_col: str = "effect_size",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    label_col: str = "metric",
    facet_col: str | None = None,
    significant_col: str | None = "significant",
    q_col: str | None = "q_value",
    hl_col: str | None = "hodges_lehmann",
    hl_ci_cols: tuple[str, str] = ("hl_ci_lower", "hl_ci_upper"),
    reference_line: float = 0.0,
    title: str | None = None,
    xaxis_title: str | None = None,
    sort_by_magnitude: bool = True,
    library: str = "matplotlib",
    renderer: str | None = None,
    save_path: str | Path | None = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Forest plot of effect sizes with confidence intervals — one row per comparison.

    The standard presentation for many outcomes compared at once: the point estimate
    with its interval carries the magnitude and its precision, and the reference line
    shows at a glance which intervals exclude "no effect". Rows are sorted by absolute
    effect so the substantive findings sit together, regardless of their p-values.

    Non-significant rows are drawn with hollow markers rather than omitted — a metric
    tested and found null is information, and dropping it would misrepresent the
    breadth of what was examined.

    When the columns are present, each row carries two further readouts in a text column
    to the right: the FDR-corrected q-value, and the Hodges-Lehmann median difference with
    its confidence interval. The plotted effect size is unit-free by design, which is what
    makes the rows comparable but also what makes a large delta impossible to translate
    back into the metric — the Hodges-Lehmann estimate is that translation, in the metric's
    own units. It is defined for two groups only, so it is left blank for a k > 2 omnibus.

    Parameters:
        df (pl.DataFrame): One row per comparison.
        effect_col (str): Column holding the point estimate.
        ci_lower_col (str): Column holding the lower interval bound.
        ci_upper_col (str): Column holding the upper interval bound.
        label_col (str): Column holding the row label.
        facet_col (str | None): Optional column prefixed onto the label (e.g. "variable").
        significant_col (str | None): Boolean column controlling filled vs hollow markers.
        q_col (str | None): Column holding the corrected q-value. None, or absent from the
            frame, drops the q column from the annotations.
        hl_col (str | None): Column holding the Hodges-Lehmann median difference. None, or
            absent, drops the Hodges-Lehmann column.
        hl_ci_cols (tuple[str, str]): Columns holding its interval bounds. Absent bounds
            print the point estimate on its own.
        reference_line (float): x position of the dashed null-effect line.
        title (str): Title of the plot.
        xaxis_title (str): Title for the x-axis.
        sort_by_magnitude (bool): Sort rows by absolute effect size descending.
        library (str): "matplotlib" or "plotly".
        renderer (str): Renderer for displaying the plot. None disables rendering.
        save_path (str): Path to save the plot image.
        save_results (bool): Whether to save the plot image.

    Returns:
        go.Figure | plt.Figure: The figure object.
    """
    for col in (effect_col, ci_lower_col, ci_upper_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    plot_df = df.filter(pl.col(effect_col).is_not_null() & pl.col(effect_col).is_not_nan())
    if plot_df.height == 0:
        raise ValueError("No rows with a defined effect size to plot.")

    if sort_by_magnitude:
        plot_df = plot_df.with_columns(pl.col(effect_col).abs().alias("_abs_effect")).sort(
            "_abs_effect", descending=False
        )

    if facet_col is not None and facet_col in plot_df.columns:
        labels = [
            f"{f} · {m}" if f is not None else str(m)
            for f, m in zip(plot_df[facet_col].to_list(), plot_df[label_col].to_list())
        ]
    else:
        labels = [str(v) for v in plot_df[label_col].to_list()]

    effects = plot_df[effect_col].cast(pl.Float64).to_numpy()
    lowers = plot_df[ci_lower_col].cast(pl.Float64).to_numpy()
    uppers = plot_df[ci_upper_col].cast(pl.Float64).to_numpy()

    if significant_col and significant_col in plot_df.columns:
        significant = [bool(v) for v in plot_df[significant_col].fill_null(False).to_list()]
    else:
        significant = [True] * len(effects)

    q_texts = (
        [_format_q_value(v) for v in plot_df[q_col].to_list()]
        if q_col and q_col in plot_df.columns
        else None
    )
    hl_texts = _format_hl_column(plot_df, hl_col, hl_ci_cols)

    y_positions = np.arange(len(effects))
    # Error bars are offsets from the point estimate, clipped at 0 so an inverted
    # interval (possible with a bootstrap on a degenerate sample) cannot raise.
    err_low = np.clip(effects - lowers, 0, None)
    err_high = np.clip(uppers - effects, 0, None)

    if library.lower() == "plotly":
        fig = go.Figure()
        for is_sig, marker_name in ((True, "significant"), (False, "not significant")):
            idx = [i for i, s in enumerate(significant) if s == is_sig]
            if not idx:
                continue
            hover_extra = [
                "<br>".join(
                    part
                    for part in (
                        f"q = {q_texts[i]}" if q_texts and q_texts[i] else "",
                        f"Hodges-Lehmann {hl_texts[i]}" if hl_texts and hl_texts[i] else "",
                    )
                    if part
                )
                for i in idx
            ]
            fig.add_trace(
                go.Scatter(
                    x=effects[idx],
                    y=[labels[i] for i in idx],
                    mode="markers",
                    name=marker_name,
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": err_high[idx],
                        "arrayminus": err_low[idx],
                        "thickness": 1.2,
                    },
                    marker={
                        "size": 9,
                        "color": "#1f77b4" if is_sig else "white",
                        "line": {"color": "#1f77b4", "width": 1.5},
                    },
                    customdata=hover_extra,
                    hovertemplate="%{y}<br>%{x:.3g}<br>%{customdata}<extra></extra>",
                )
            )
        fig.add_vline(x=reference_line, line_dash="dash", line_color="grey")

        # Text columns to the right of the plotting area, in paper coordinates so they
        # stay put whatever the x range does.
        annotations, right_margin = [], 40
        columns = [(c, t) for c, t in (("q", q_texts), ("Hodges-Lehmann [95% CI]", hl_texts)) if t]
        for col_idx, (header, texts) in enumerate(columns):
            x_paper = 1.02 + col_idx * 0.16
            annotations.append(
                {
                    "text": f"<b>{header}</b>",
                    "x": x_paper,
                    "y": 1.0,
                    "xref": "paper",
                    "yref": "paper",
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "showarrow": False,
                    "font": {"size": 10},
                }
            )
            annotations += [
                {
                    "text": text,
                    "x": x_paper,
                    "y": labels[i],
                    "xref": "paper",
                    "yref": "y",
                    "xanchor": "left",
                    "showarrow": False,
                    "font": {"size": 10},
                }
                for i, text in enumerate(texts)
                if text
            ]
        if columns:
            right_margin = 120 + 130 * len(columns)

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title or effect_col,
            yaxis_title="",
            template="plotly_white",
            height=max(400, 26 * len(effects) + 160),
            annotations=annotations,
            margin={"r": right_margin},
        )

        if renderer:
            fig.show(renderer=renderer)

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")
            fig.write_image(save_path)
            print(f"Forest plot saved to {save_path}")

        return fig

    elif library.lower() == "matplotlib":
        n_annotation_cols = sum(1 for t in (q_texts, hl_texts) if t)
        fig, ax = plt.subplots(
            figsize=(10 + 2.2 * n_annotation_cols, max(6, 0.32 * len(effects) + 2))
        )

        ax.errorbar(
            effects,
            y_positions,
            xerr=[err_low, err_high],
            fmt="none",
            ecolor="#4c72b0",
            elinewidth=1.2,
            capsize=3,
        )
        for i, (effect, is_sig) in enumerate(zip(effects, significant)):
            ax.plot(
                effect,
                i,
                marker="o",
                markersize=7,
                color="#4c72b0",
                markerfacecolor="#4c72b0" if is_sig else "white",
            )

        ax.axvline(reference_line, linestyle="--", color="grey", linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xaxis_title or effect_col)
        if title:
            ax.set_title(title)

        ax.grid(True, alpha=0.3, axis="x")

        # Text columns to the right of the axes: x in axes fraction, y in data
        # coordinates, so a row's annotation tracks its marker.
        columns = [(c, t) for c, t in (("q", q_texts), ("Hodges-Lehmann [95% CI]", hl_texts)) if t]
        if columns:
            ax.set_ylim(-0.8, len(effects) - 0.2)
            transform = ax.get_yaxis_transform()
            for col_idx, (header, texts) in enumerate(columns):
                x_frac = 1.03 + col_idx * 0.14
                ax.text(
                    x_frac,
                    len(effects) - 0.5,
                    header,
                    transform=transform,
                    fontsize=8,
                    fontweight="bold",
                    va="center",
                    clip_on=False,
                )
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    ax.text(
                        x_frac,
                        i,
                        text,
                        transform=transform,
                        fontsize=8,
                        va="center",
                        clip_on=False,
                    )
            # Leave room for the columns; tight_layout only measures the axes themselves.
            fig.tight_layout(rect=(0, 0, 1 - 0.16 * len(columns), 1))
        else:
            fig.tight_layout()

        if save_results and save_path:
            save_path = Path(save_path)
            if save_path.suffix != ".png":
                save_path = save_path.with_suffix(".png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Forest plot saved to {save_path}")

        if renderer:
            plt.show()

        plt.close(fig)
        return fig

    else:
        raise ValueError(f"Library '{library}' is not supported. Choose 'plotly' or 'matplotlib'.")


# test matplotlib plotting
if __name__ == "__main__":
    # Example usage
    df1 = pl.DataFrame({"value": np.random.normal(0, 1, 100)})
    df2 = pl.DataFrame({"value": np.random.normal(1, 1, 100)})

    fig1 = plot_histogram(
        df1,
        "value",
        title="Histogram Example",
        library="matplotlib",
        save_path="histogram_example.png",
        save_results=True,
        renderer="browser",
    )
    fig2 = plot_boxplot(
        df1,
        "value",
        title="Boxplot Example",
        library="matplotlib",
        save_path="boxplot_example.png",
        save_results=True,
    )
    fig3 = plot_scatter(
        df1["value"].to_numpy(),
        df2["value"].to_numpy(),
        title="Scatter Example",
        library="matplotlib",
        save_path="scatter_example.png",
        save_results=True,
    )
    fig4 = plot_multi_boxplot(
        [df1, df2],
        "value",
        group_names=["Group 1", "Group 2"],
        title="Multi Boxplot Example",
        library="matplotlib",
        save_path="multi_boxplot_example.png",
        save_results=True,
    )
    fig5 = plot_overlay_histograms(
        [df1, df2],
        "value",
        group_names=["Group 1", "Group 2"],
        title="Overlayed Histograms Example",
        library="matplotlib",
        save_path="overlayed_histograms_example.png",
        save_results=True,
    )
    # QQ plot example
    fig6 = plot_qq(
        df1["value"].to_numpy(),
        title="QQ Plot Example",
        library="matplotlib",
        save_path="qq_plot_example.png",
        save_results=True,
    )
