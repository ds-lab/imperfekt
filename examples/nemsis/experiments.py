# %%
from examples.nemsis.features import make_feature_sets
from examples.nemsis.cv import compute_intervariable_missingness_strata, summarise_cv, run_cv
import polars as pl
from pathlib import Path

from config import COHORT_PATH, RESULTS_DIR, CV_N_REPEATS, CV_N_SPLITS
from prep import make_plausibility_mask, make_configs

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(Path(COHORT_PATH), n_rows=10000)

print(f"Cohort: {df['id'].n_unique()} stays, {len(df)} observations")
outcome = df.group_by("id").agg(pl.col("label").max()).select("label")
print(f"Outcome prevalence (stay-level): {outcome.mean()[0]['label']} ({outcome.sum()[0]['label']}/{len(outcome)})")

case_metrics, axes = compute_intervariable_missingness_strata(df, cohort_path=Path(COHORT_PATH))

mask_iqr = make_plausibility_mask(df, method="iqr")
mask_mad = make_plausibility_mask(df, method="mad")

configs = make_configs(df, mask_iqr, mask_mad)

print("=== Configs preview ===")
for name, cfg in configs.items():
    print(f"\nConfig: {name}")
    print(cfg.select(["id", "clock", "sbp", "hr", "o2sat", "rr"]).head(5))

print(case_metrics.columns)
# %% EXPERIMENTS
setups = make_feature_sets(
    configs["iq_pk_in"],
    config_name="iq_pk_in",
    cohort_path=Path(COHORT_PATH),
    case_metrics=case_metrics,
)
setup_0 = setups["base"]
print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline 0…")
folds_0, _, _, _, _, _ = run_cv(setup_0, case_metrics, axes, "Pipeline0")
summary_0 = summarise_cv(folds_0, "Pipeline0")