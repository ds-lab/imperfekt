# %%
from pathlib import Path

import polars as pl

from imperfekt import Imperfekt

df = pl.read_parquet(Path("/workspaces/imperfekt/data/nemsis/destinations.parquet"))

# %%
imp = Imperfekt(
    df=df,
    id_col="PcrKey",
    clock_col="clock",
    cols=["sbp", "hr", "o2sat", "rr"],
    save_path=Path("/workspaces/imperfekt/data/nemsis/results"),
    plot_library="matplotlib",
)

# %%
imp.run(save_results=True, generate_html=False, cheap_mode=False, addition_to_title="NEMSIS 2024")

# %%
imp.run_grouped_analysis(
    annotation_col="label",
    save_results=True,
    addition_to_title="NEMSIS 2024",
    cheap_mode=True,
)
