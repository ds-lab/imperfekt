# %%
import polars as pl

from examples.mimic_iv_ed.extract_cohort import build_cohort
from imperfekt import Imperfekt

pl.Config.set_tbl_cols(25)
pl.Config.set_tbl_rows(24)
SAVE_RESULTS_PATH = "mimic_iv_ed_results"
RENDERER = None
SAVE_RESULTS = True
cols = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]

# %%
df = build_cohort(["mortality"])

imp = Imperfekt(
    df=df,
    id_col="stay_id",
    clock_col="charttime",
    clock_no_col="ct_no",
    cols=cols,
    save_path=SAVE_RESULTS_PATH,
    plot_library="matplotlib",
    renderer=RENDERER,
)

# %% Run the main analysis
# imp.preliminary.run(save_results=SAVE_RESULTS)
imp.irregularity.run(save_results=SAVE_RESULTS)

# %%
imp.irregularity.results.cs_case_scores

# %%
