from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl


def plot_histogram(
    df: pl.DataFrame,
    x: str,
    title: str = None,
    nbins: int = 50,
    xaxis_title: str = None,
    yaxis_title: str = None,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
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
    x: str = None,
    title: str = None,
    yaxis_title: str = None,
    xaxis_title: str = None,
    category_order: str = None,
    boxpoints: str = None,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic boxplot function for visualizing distributions of a column.

    Parameters:
        df (pl.DataFrame): DataFrame containing the data to plot.
        y (str): Column name to plot on the y-axis.
        x (str): Column name to plot on the x-axis (for grouped boxplots), default is None.
        title (str): Title of the plot.
        yaxis_title (str): Title for the y-axis.
        xaxis_title (str): Title for the x-axis.
        category_order (list[str]): Order of categories for the x-axis, default is None.
        boxpoints (str): Type of boxpoints to show, default is None. Options include 'all', 'outliers', 'suspectedoutliers', 'false'.
                        If set to None, no boxpoints will be shown.
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
                labels=labels,
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
    x: str = None,
    title: str = None,
    yaxis_title: str = None,
    xaxis_title: str = None,
    category_order: str = None,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic violin plot function for visualizing the distribution of a column.

    Parameters:
        df (pl.DataFrame): DataFrame containing the data to plot.
        y (str): Column name to plot on the y-axis.
        x (str): Column name to plot on the x-axis (for grouped violin plots), default is None.
        title (str): Title of the plot.
        yaxis_title (str): Title for the y-axis.
        xaxis_title (str): Title for the x-axis.
        category_order (list[str]): Order of categories for the x-axis, default is None.
        library (str): Visualization library to use, default is "matplotlib". Other options include "plotly".
        renderer (str): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
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
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
    save_results: bool = False,
) -> "go.Figure | plt.Figure":
    """
    Generic scatter plot function for visualizing the relationship between two columns.
    Parameters:
        x (np.ndarray): Data to plot on the x-axis.
        y (np.ndarray): Data to plot on the y-axis.
        mode (str): Mode for the scatter plot, default is 'lines+markers'. Other options include 'markers', 'lines', etc.
        title (str): Title of the plot.
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
    group_names: list[str] = None,
    title: str = None,
    nbins: int = 50,
    xaxis_title: str = None,
    yaxis_title: str = None,
    histnorm: str = None,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
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
    group_names: list[str] = None,
    title: str = None,
    yaxis_title: str = None,
    boxpoints: str = None,
    library: str = "matplotlib",
    renderer: str = None,
    save_path: str = None,
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
        save_path (str): Path to save the plot image, default is None.
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
    renderer: str = None,
    save_path: str = None,
    save_results: bool = False,
) -> "plt.Figure":
    """
    Generic QQ plot function for visualizing quantile-quantile plots.

    Parameters:
        data (np.ndarray): Data to compare against the theoretical distribution.
        dist (str): Theoretical distribution to compare against, default is "norm" (normal distribution).
        title (str): Title of the plot.
        xaxis_title (str): Title for the x-axis.
        yaxis_title (str): Title for the y-axis.
        library (str): Visualization library to use, default is "matplotlib".
        renderer (str): Renderer for displaying the plot, default is "notebook_connected". Set to None to disable rendering.
        save_path (str): Path to save the plot image, default is None.
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
