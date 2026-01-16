# %%
import polars as pl

import examples.sepsis_prediction.config as settings
from imperfekt.analysis.intervariable import intervariable
from imperfekt.config.global_settings import VITALS

pl.Config.set_tbl_cols(25)
path = settings.PATHS.get_nemsis_path(settings.VARIABLES.NEMSIS_YEAR_STR)
data_path = path / "cleaned"
SAVE_RESULTS_PATH = path / "results" / "imperfekt_analysis" / "intervariable"
RENDERER = None
SAVE_RESULTS = True

vitals_df = pl.read_parquet(data_path / "vitals.parquet")
print(vitals_df.sample(5))

intervariable = intervariable.IntervariableImperfection(
    df=vitals_df,
    id_col="id",
    clock_col="clock",
    clock_no_col="clock_no",
    cols=VITALS.PARAMS,
    save_path=SAVE_RESULTS_PATH,
    renderer=RENDERER,
)
# %%
##############################
#       ROW COMPLETENESS     #
##############################
intervariable = intervariable.row_statistics(save_results=SAVE_RESULTS, analyze_all_null_rows=True)
# %%
##############################
#       CO-imperfection       #
##############################
intervariable = intervariable.symmetric_correlation(
    save_results=SAVE_RESULTS, dendrogram=False, heatmap=True
)
intervariable = intervariable.symmetric_lagged_cross_correlation(
    save_results=SAVE_RESULTS, max_lag=10
)

# %%
##############################
#    ASYMMETRIC ANALYSIS     #
##############################
# Is imperfection in variable X associated with observed values of variables Y, Z, etc.?
# Little's MCAR test for imperfection association across variables
intervariable = intervariable.mcar_test(save_results=SAVE_RESULTS)
intervariable = intervariable.mar_mnar_test(save_results=SAVE_RESULTS)
# %%
# Conditional distribution of a column based on the observations of another column
intervariable = intervariable.asymmetric_correlation(save_results=SAVE_RESULTS)
intervariable = intervariable.asymmetric_lagged_cross_correlation(
    save_results=SAVE_RESULTS, max_lag=10
)
# %%
##############################
#           MASKING          #
##############################
# intervariable.visualize_missingness(save_results=SAVE_RESULTS)
# %%
intervariable.generate_html_report()
