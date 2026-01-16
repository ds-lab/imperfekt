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
vitals_df = pl.read_parquet(data_path / "vitals.parquet")
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
# %%
# Run the grouped analysis
# This will run the full analysis for each unique value in the 'ed_icd_chapter' column.
# e.g., Results for the 'X' group will be in './analysis_results/X/'
diagnosis_df = pl.read_parquet(data_path / "ed_icd.parquet")
imp.run_grouped_analysis(
    annotation_df=diagnosis_df,
    annotation_col="ed_icd_chapter",
    save_results=SAVE_RESULTS,
    addition_to_title="ED ICD Chapter",
    cheap_mode=CHEAP_MODE,  # Cheap mode only performs a subset of the full analysis which includes basic statistics and correlation analyses
)
# %% Run event-based analysis, here: NEMSIS procedures
# DF will be split into windows around each event occurrence
interventions_df = pl.read_parquet(data_path / "interventions.parquet")

intervention_list = [
    "External ventricular defibrillation (procedure)",
    "Cardiopulmonary resuscitation (procedure)",
]

imp.run_event_based_analysis(
    events_df=interventions_df,
    event_name_col="intervention_title",
    included_event_names=intervention_list,
    window_size=0,
    window_location="both",  # is ignored if window_size=0, result: exact event time only
    remove_ids_without_events=True,
    save_results=False,
)
