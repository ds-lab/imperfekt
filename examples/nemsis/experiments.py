# %%
from datetime import datetime
from plotting import plot_auprc_by_stratum, plot_auprc_lift_by_stratum, plot_auroc_by_stratum
from examples.nemsis.features import make_feature_sets
from examples.nemsis.cv import compute_intervariable_missingness_strata, save_cv_results, summarise_cv, run_cv
import polars as pl
from pathlib import Path

from config import AXES, COHORT_PATH, RESULTS_DIR, CV_N_REPEATS, CV_N_SPLITS, STAGE_3_CONFIGS, load_cohort, saits_model_path, data_fingerprint_tag
from prep import ConfigBuilder

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_fp_tag = data_fingerprint_tag(COHORT_PATH)
RUN_DIR = RESULTS_DIR / _fp_tag
RUN_DIR.mkdir(parents=True, exist_ok=True)

df = load_cohort()
print(f"Cohort: {df['id'].n_unique()} stays, {len(df)} observations")
outcome = df.group_by("id").agg(pl.col("label").max()).select("label")
print(f"Outcome prevalence (stay-level): {outcome.mean()[0]['label'][0]} ({outcome.sum()[0]['label'][0]}/{len(outcome)})")

case_metrics, axes = compute_intervariable_missingness_strata(df, cohort_path=Path(COHORT_PATH))
print(case_metrics.columns)

axes = AXES

# Masks and configs are built lazily by the ConfigBuilder: a fully cached
# feature-set run never touches them. On a cache miss, make_feature_sets calls
# the provider, which builds only the requested config (and only the mask that
# config needs), memoized for reuse across setups.
builder = ConfigBuilder(df)

setups = dict()
pipeline_summaries = []
_SHAP_FULL_RUNS = {"ma_pk_in/base+miss"} # {"ma_pk_in/base+miss"}
_FEAT_DIST_OUTCOME_RUN = "ma_pk_in/base+miss"

# %% EXPERIMENTS
for config_name, config in STAGE_3_CONFIGS.items():
    print(f"Config {config_name}: method={config['method']}, plaus={config['plaus']}, imp={config['imp']}")
    if config["imp"] == "saits" and not saits_model_path(config["plaus"]).exists():
        print(f"  -> skipped {config_name}: no SAITS model for plaus={config['plaus']}")
        continue
    feature_sets = make_feature_sets(
        lambda config_name=config_name: builder.config(config_name),
        config_name=config_name,
        cohort_path=Path(COHORT_PATH),
        case_metrics=case_metrics,
    )

    for setup_name, stay_df in feature_sets.items():
        run_name = f"{config_name}/{setup_name}"
        setups[run_name] = stay_df
        print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Setup {run_name}…")
        shap_path = RUN_DIR / "shap" / f"{run_name.replace('/', '__')}.npz"
        features_path = RUN_DIR / "features" / f"{run_name.replace('/', '__')}.npz"
        folds, _, _, _, _, _ = run_cv(stay_df, case_metrics, axes, run_name,
                                       shap_save_path=shap_path,
                                       features_save_path=features_path if run_name == _FEAT_DIST_OUTCOME_RUN else None,
                                       shap_full=run_name in _SHAP_FULL_RUNS)
        summary = summarise_cv(folds, run_name)
        summary["_run_timestamp"] = datetime.now().isoformat(timespec="seconds")
        pipeline_summaries.append((f"Setup {run_name}", summary))
        
        plot_auprc_lift_by_stratum(
            pipeline_summaries,
            RUN_DIR / "figures" / "auprc_lift_by_stratum.png",
        )
        save_cv_results(pipeline_summaries, RUN_DIR / "cv_results_temp_2.csv")
# %%
save_cv_results(pipeline_summaries, RUN_DIR / "cv_results.csv")
(RUN_DIR / "cv_results_temp.csv").unlink(missing_ok=True)

# %%
