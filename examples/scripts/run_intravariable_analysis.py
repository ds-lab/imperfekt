# %%
from datetime import timedelta

import examples.sepsis_prediction.config as settings
import polars as pl

from imperfekt.analysis.intravariable import intravariable
from imperfekt.config.global_settings import VITALS

pl.Config.set_tbl_cols(25)
pl.Config.set_tbl_rows(24)
path = settings.PATHS.get_nemsis_path(settings.VARIABLES.NEMSIS_YEAR_STR)
data_path = path / "cleaned"
SAVE_RESULTS_PATH = path / "results" / "intravariable"
RENDERER = None
SAVE_RESULTS = True
cols = VITALS.PARAMS  # or cols=['heartrate'] for single column analysis

vitals_df = pl.read_parquet(data_path / "vitals.parquet")
vitals_df = vitals_df.filter(pl.col("clock").is_not_null() & pl.col("id").is_not_null())
print(vitals_df.sample(5))

intravariable = intravariable.IntravariableImperfection(
    df=vitals_df,
    save_path=SAVE_RESULTS_PATH,
    renderer=RENDERER,
    cols=cols,
    # plot_library="plotly",
)
# %%
##############################
#     COLUMN COMPLETENESS    #
##############################
intravariable = intravariable.column_statistics(save_results=SAVE_RESULTS)
# %%
##############################
#         GAP LENGTH         #
##############################
intravariable = intravariable.gap_statistics(
    save_results=SAVE_RESULTS,
)
# %%
intravariable = intravariable.gap_returns(save_results=SAVE_RESULTS)
# %%
##############################
#    MARKOV CHAIN SUMMARY    #
##############################
intravariable = intravariable.markov_chain_summary(save_results=SAVE_RESULTS)
# %%
##############################
#          TEMPORAL          #
##############################
intravariable = intravariable.windowed_significance(
    save_results=SAVE_RESULTS,
    window_size=timedelta(minutes=5),
    window_location="before",
)

# %%
################################
#  IMPERFECTION AUTOCORRELATION #
################################
intravariable = intravariable.autocorrelation(
    save_results=SAVE_RESULTS, lags=20, seasonal_trend_decomposition=True
)
# %%
##############################
#   DATETIME CORRELATION     #
##############################
intravariable = intravariable.date_time_statistics(save_results=SAVE_RESULTS)

# %%
intravariable.generate_html_report()

# %%
df = intravariable.results.gs_gaps_observation_runs
max_val = df.select(pl.max("time_length")).item()  # scalar max ([DataFrame.max], [DataFrame.item])
row_at_max = df.filter(pl.col("time_length") == max_val)
print(row_at_max)
# %%
df.filter(pl.col("time_length") > (60 * 60 * 6)).n_unique(
    "id"
)  # number of unique patients with gaps > 6 hours -> these ids might contain unreliable data recordings
# %%
