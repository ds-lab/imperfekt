import base64
import io

import matplotlib
import plotly.graph_objects as go
import polars as pl


def matplotlib_fig_to_html(fig) -> str:
    """Converts a Matplotlib figure to an HTML img tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;"/>'


def fig_to_html(fig) -> str:
    """Converts a plotly or matplotlib figure to HTML."""
    if isinstance(fig, go.Figure):
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    elif isinstance(fig, matplotlib.figure.Figure):
        return matplotlib_fig_to_html(fig)
    return ""


def dataframe_to_html_table(df: pl.DataFrame, title: str = "") -> str:
    """Converts a Polars DataFrame to an HTML table."""
    if df.is_empty():
        return f"<p>No data available for {title}</p>"

    html = "<table>"

    # Table header
    html += "<tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    # Table rows
    for row in df.iter_rows():
        html += "<tr>"
        for val in row:
            # Format numbers nicely
            if isinstance(val, float):
                formatted_val = f"{val:.4g}" if val is not None else "N/A"
            else:
                formatted_val = str(val) if val is not None else "N/A"
            html += f"<td>{formatted_val}</td>"
        html += "</tr>"

    html += "</table>"
    return html


def get_apple_style_css() -> str:
    """Returns Apple-inspired CSS styles."""
    return """
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1d1d1f;
        background: #f4f6f9;
        min-height: 100vh;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    .header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .header h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 15px;
        color: #1A4481;
    }

    .metadata {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    .badge {
        background-color: #1A4481;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
    }

    .section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        overflow: hidden;
    }

    .section-header {
        padding: 25px 30px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .section-header:hover {
        background: rgba(26, 68, 129, 0.05);
    }

    .section-title {
        font-size: 1.4em;
        font-weight: 600;
        color: #1d1d1f;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-content {
        padding: 30px;
        display: none;
    }

    .section.expanded .section-content {
        display: block;
        animation: fadeIn 0.3s ease;
    }

    .chevron {
        font-size: 1.2em;
        transition: transform 0.3s ease;
        color: #1A4481;
    }

    .section.expanded .chevron {
        transform: rotate(180deg);
    }

    .subsection {
        margin: 25px 0;
        background: rgba(26, 68, 129, 0.02);
        border-radius: 12px;
        border-left: 4px solid #1A4481;
        overflow: hidden;
    }

    .subsection-header {
        padding: 15px 20px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .subsection-header:hover {
        background-color: rgba(26, 68, 129, 0.05);
    }

    .subsection h3 {
        font-size: 1.2em;
        font-weight: 600;
        margin: 0;
        color: #1A4481;
    }

    .subsection-chevron {
        font-size: 1em;
        transition: transform 0.3s ease;
        color: #1A4481;
    }

    .subsection.expanded .subsection-chevron {
        transform: rotate(180deg);
    }

    .subsection-content {
        padding: 0 20px 20px 20px;
        display: none;
    }

    .subsection.expanded .subsection-content {
        display: block;
    }

    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 15px;
    }

    .card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }

    .metric {
        text-align: center;
        padding: 10px;
    }

    .metric-value {
        font-size: 1.6em;
        font-weight: 700;
        color: #1A4481;
        margin-bottom: 5px;
    }

    .metric-label {
        font-size: 0.8em;
        color: #86868b;
        font-weight: 500;
    }

    .table-container {
        overflow-x: auto;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }

    table {
        width: 100%;
        border-collapse: collapse;
        background: white;
    }

    th {
        background: #1A4481;
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: 600;
    }

    td {
        padding: 12px 15px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }

    tr:hover {
        background: rgba(26, 68, 129, 0.02);
    }

    .plot-container {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }

    .plot-container img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
    }

    .plot-section {
        margin: 15px 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }

    .plot-header {
        background: rgba(26, 68, 129, 0.08);
        padding: 15px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }

    .plot-header:hover {
        background: rgba(26, 68, 129, 0.12);
    }

    .plot-header h4 {
        margin: 0;
        color: #1A4481;
        font-size: 1.1em;
        font-weight: 600;
    }

    .plot-chevron {
        font-size: 1em;
        transition: transform 0.3s ease;
        color: #1A4481;
    }

    .plot-section.expanded .plot-chevron {
        transform: rotate(180deg);
    }

    .collapsible-plot {
        display: none;
        padding: 20px;
        background: white;
        text-align: center;
    }

    .plot-section.expanded .collapsible-plot {
        display: block;
        animation: fadeIn 0.3s ease;
    }
    .footer {
        text-align: center;
        padding: 30px;
        color: #86868b;
        font-size: 0.9em;
    }
    .warning {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        border-left: 4px solid #ff6b6b;
    }
    .info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        border-left: 4px solid #20bf6b;
        font-style: italic; /* Alles innerhalb kursiv */
    }
    .info strong {
        font-style: normal; /* Headlines oder wichtige Begriffe bleiben normal */
        font-weight: 600;
    }
    .info ol {
        margin-left: 20px;
        padding-left: 20px;
        list-style-type: disc;
    }
    .info ul {
        margin-left: 20px;
        padding-left: 20px;
        list-style-type: disc;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 768px) {
        .container {
            padding: 15px;
        }
        .header h1 {
            font-size: 2em;
        }
        .metadata {
            flex-direction: column;
        }
        .grid {
            grid-template-columns: 1fr;
        }
    }
    """


def get_javascript() -> str:
    """Returns JavaScript for interactive functionality."""
    return """
    document.addEventListener('DOMContentLoaded', function() {
        // Toggle sections
        document.querySelectorAll('.section-header').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('expanded');
            });
        });

        // Toggle subsections
        document.querySelectorAll('.subsection-header').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('expanded');
            });
        });
        // Expand first section by default
        const firstSection = document.querySelector('.section');
        if (firstSection) {
            firstSection.classList.add('expanded');
        }
        // Smooth scrolling for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    });
    // Toggle plot function
    function togglePlot(plotHeader) {
        const plotSection = plotHeader.parentElement;
        plotSection.classList.toggle('expanded');
    }
    """
