# %%
from datetime import datetime
from plotting import plot_auprc_by_stratum, plot_auprc_lift_by_stratum, plot_auroc_by_stratum
from examples.nemsis.features import make_feature_sets
from examples.nemsis.cv import (
    combine_case_metrics,
    compute_intervariable_missingness_strata,
    compute_intravariable_missingness_strata,
    save_cv_results,
    summarise_cv,
    run_cv,
)
import polars as pl
from pathlib import Path

from config import (
    AXES_INTERVARIABLE,
    AXES_INTRAVARIABLE,
    COHORT_PATH,
    RESULTS_DIR,
    CV_N_REPEATS,
    CV_N_SPLITS,
    STAGE_3_CONFIGS,
    load_cohort,
    saits_model_path,
    data_fingerprint_tag,
)
from prep import ConfigBuilder

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_fp_tag = data_fingerprint_tag(COHORT_PATH)

df = load_cohort()
print(f"Cohort: {df['id'].n_unique()} stays, {len(df)} observations")
outcome = df.group_by("id").agg(pl.col("label").max()).select("label")
print(f"Outcome prevalence (stay-level): {outcome.mean()[0]['label'][0]} ({outcome.sum()[0]['label'][0]}/{len(outcome)})")

inter_metrics, _ = compute_intervariable_missingness_strata(df, cohort_path=Path(COHORT_PATH))
intra_metrics, _ = compute_intravariable_missingness_strata(df, cohort_path=Path(COHORT_PATH))
case_metrics = combine_case_metrics(inter_metrics, intra_metrics)
print(f"Combined case_metrics columns: {case_metrics.columns}")

# Masks and configs are built lazily by the ConfigBuilder: a fully cached
# feature-set run never touches them. On a cache miss, make_feature_sets calls
# the provider, which builds only the requested config (and only the mask that
# config needs), memoized for reuse across setups.
builder = ConfigBuilder(df)

_SHAP_FULL_RUNS = {"ma_pk_in/base+miss"} # {"ma_pk_in/base+miss"}
_FEAT_DIST_OUTCOME_RUN = "ma_pk_in/base+miss"

STRATIFICATION_MODES = {
    "intervariable": AXES_INTERVARIABLE,
    "intravariable": AXES_INTRAVARIABLE,
}

# %% EXPERIMENTS
# Each (config x setup) is evaluated under both stratification modes so the
# per-stratum patterns can be compared across the intra/inter decomposition,
# mirroring experiments_gru.py. run_cv takes all modes at once and trains the
# model once per fold, reusing it across modes (they only differ in how held-out
# test stays are bucketed into strata), so the two-mode sweep costs ~1× a single
# pass instead of 2×. Per-mode results are still written to the same
# RESULTS_DIR/<fp>/<mode>/ layout (cv_results.csv, shap/, features/, figures/)
# that plot_results.py expects.
_RUN_DIRS = {}
for strat_mode in STRATIFICATION_MODES:
    if strat_mode == "intravariable":
        a = AXES_INTRAVARIABLE
    else:
        a = AXES_INTERVARIABLE
    RUN_DIR = RESULTS_DIR / _fp_tag / strat_mode / f"{a[0]}_{a[1]}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _RUN_DIRS[strat_mode] = RUN_DIR

pipeline_summaries_by_mode = {m: [] for m in STRATIFICATION_MODES}

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
        print(
            f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV"
            f" — Setup {run_name} [{', '.join(STRATIFICATION_MODES)}]…"
        )
        shap_paths = (
            {
                m: _RUN_DIRS[m] / "shap" / f"{run_name.replace('/', '__')}.npz"
                for m in STRATIFICATION_MODES
            }
            if run_name in _SHAP_FULL_RUNS
            else {}
        )
        features_paths = (
            {
                m: _RUN_DIRS[m] / "features" / f"{run_name.replace('/', '__')}.npz"
                for m in STRATIFICATION_MODES
            }
            if run_name == _FEAT_DIST_OUTCOME_RUN
            else {}
        )
        folds_by_mode = run_cv(
            stay_df, case_metrics, STRATIFICATION_MODES, run_name,
            shap_save_paths=shap_paths,
            features_save_paths=features_paths,
            shap_full=run_name in _SHAP_FULL_RUNS,
        )
        run_ts = datetime.now().isoformat(timespec="seconds")
        for strat_mode in STRATIFICATION_MODES:
            summary = summarise_cv(folds_by_mode[strat_mode], run_name)
            summary["_run_timestamp"] = run_ts
            pipeline_summaries_by_mode[strat_mode].append((f"Setup {run_name}", summary))

            plot_auprc_lift_by_stratum(
                pipeline_summaries_by_mode[strat_mode],
                _RUN_DIRS[strat_mode] / "figures" / "auprc_lift_by_stratum.png",
            )
            save_cv_results(
                pipeline_summaries_by_mode[strat_mode],
                _RUN_DIRS[strat_mode] / "cv_results_temp.csv",
            )

for strat_mode in STRATIFICATION_MODES:
    save_cv_results(
        pipeline_summaries_by_mode[strat_mode],
        _RUN_DIRS[strat_mode] / "cv_results.csv",
    )
    (_RUN_DIRS[strat_mode] / "cv_results_temp.csv").unlink(missing_ok=True)

# %%
