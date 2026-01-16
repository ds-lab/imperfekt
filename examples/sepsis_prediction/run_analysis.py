# %%
import polars as pl

import examples.sepsis_prediction.config as settings
from imperfekt.analysis.imperfekt import Imperfekt
from imperfekt.config.global_settings import VITALS

pl.Config.set_tbl_cols(25)
pl.Config.set_tbl_rows(24)
path = settings.PATHS.get_nemsis_path(settings.VARIABLES.NEMSIS_YEAR_STR)
data_path = path / "model_input"
SAVE_RESULTS_PATH = path / "results" / "imperfekt_analysis"
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RENDERER = None
SAVE_RESULTS = True
CHEAP_MODE = True
cols = VITALS.PARAMS
vitals_df = pl.read_parquet(
    data_path / "sepsis_prediction_input_20minutes_20minutes_5count_604960.parquet"
)
print(vitals_df.height)
print(f"Unique IDs: {vitals_df.select(pl.col('id').n_unique()).item()}")

imp = Imperfekt(
    df=vitals_df,
    id_col="id",
    clock_col="clock",
    clock_no_col="clock_no",
    cols=cols,
    save_path=SAVE_RESULTS_PATH,
    plot_library="matplotlib",
    renderer=RENDERER,
)

# %% Run the main analysis
imp.run(save_results=SAVE_RESULTS, cheap_mode=CHEAP_MODE)
# %% Run the grouped analysis
# This will run the full analysis for both sepsis and non-sepsis groups.

imp.run_grouped_analysis(
    annotation_col="sepsis_outcome",
    save_results=SAVE_RESULTS,
    addition_to_title="Sepsis ED Diagnosis",
    cheap_mode=CHEAP_MODE,
)
