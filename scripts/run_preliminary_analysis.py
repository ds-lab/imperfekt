# %%
import polars as pl

import examples.sepsis_prediction.config as settings
from imperfekt.analysis.preliminary.preliminary import Preliminary
from imperfekt.config.global_settings import VITALS

pl.Config.set_tbl_cols(25)
pl.Config.set_tbl_rows(24)
path = settings.PATHS.get_nemsis_path(settings.VARIABLES.NEMSIS_YEAR_STR)
data_path = path / "cleaned"
SAVE_RESULTS_PATH = path / "results" / "preliminary"
RENDERER = None
SAVE_RESULTS = True
cols = VITALS.PARAMS

vitals_df = pl.read_parquet(data_path / "vitals.parquet", n_rows=10000)

# %%
prelim = Preliminary(df=vitals_df, save_path=SAVE_RESULTS_PATH, renderer=RENDERER, cols=cols)

prelim = prelim.describe_df(save_results=SAVE_RESULTS)
prelim = prelim.intervariable_normality(save_results=SAVE_RESULTS)
prelim = prelim.shapiro_wilk(save_results=SAVE_RESULTS)
prelim = prelim.autocorrelation(save_results=SAVE_RESULTS, lags=20)
prelim = prelim.correlation(save_results=SAVE_RESULTS, use="pairwise")

# %%
prelim.generate_html_report()
