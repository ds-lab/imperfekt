# %%
from datetime import datetime
from pathlib import Path

import polars as pl

from config import (
    AXES_INTERVARIABLE,
    AXES_INTRAVARIABLE,
    COHORT_PATH,
    RESULTS_DIR,
    CV_N_REPEATS,
    CV_N_SPLITS,
    STAGE_3_CONFIGS,
    data_fingerprint_tag,
    load_cohort,
    saits_model_path,
)
from cv import (
    combine_case_metrics,
    compute_intervariable_missingness_strata,
    compute_intravariable_missingness_strata,
    save_cv_results,
    summarise_cv,
)
from cv_gru import run_cv_gru
from plotting import plot_auprc_lift_by_stratum
from prep import ConfigBuilder

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_fp_tag = data_fingerprint_tag(COHORT_PATH)

df = load_cohort()
print(f"Cohort: {df['id'].n_unique()} stays, {len(df)} observations")
outcome = df.group_by("id").agg(pl.col("label").max()).select("label")
print(
    f"Outcome prevalence (stay-level): "
    f"{outcome.mean()[0]['label'][0]} "
    f"({outcome.sum()[0]['label'][0]}/{len(outcome)})"
)

inter_metrics, _ = compute_intervariable_missingness_strata(df, cohort_path=Path(COHORT_PATH))
intra_metrics, _ = compute_intravariable_missingness_strata(df, cohort_path=Path(COHORT_PATH))
case_metrics = combine_case_metrics(inter_metrics, intra_metrics)

builder = ConfigBuilder(df)

# %% GRU EXPERIMENTS
# GRU uses raw vitals + (optionally) a binary missingness mask at each time step,
# so no STAGE_4 feature-set loop is needed. Each config is run twice — with and
# without the mask channel (MASK_ARMS) — so the per-stratum gap between the two
# arms quantifies how much information the missingness indicator adds, and where.
# Each (config × mask arm) is evaluated under both stratification modes so the
# per-stratum patterns can be compared across the intra/inter decomposition.
MASK_ARMS = {"mask": True, "nomask": False}
STRATIFICATION_MODES = {
    "intervariable": AXES_INTERVARIABLE,
    "intravariable": AXES_INTRAVARIABLE,
}
_FEAT_DIST_RUN = "ma_pk_in/mask"

# run_cv_gru takes all modes at once and trains the GRU once per fold, reusing
# it across modes (they only differ in how held-out test stays are bucketed into
# strata), so the two-mode sweep costs ~1× a single pass instead of 2×. Per-mode
# results are still written to the same RESULTS_DIR/<fp>/<mode>/ layout
# (cv_results_gru.csv, shap/, features/, figures/) that plot_results.py expects.
_RUN_DIRS = {}
for strat_mode in STRATIFICATION_MODES:
    RUN_DIR = RESULTS_DIR / _fp_tag / strat_mode
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _RUN_DIRS[strat_mode] = RUN_DIR

pipeline_summaries_by_mode = {m: [] for m in STRATIFICATION_MODES}

for config_name, config in STAGE_3_CONFIGS.items():
    print(f"Config {config_name}: method={config['method']}, plaus={config['plaus']}, imp={config['imp']}")
    if config["imp"] == "saits" and not saits_model_path(config["plaus"]).exists():
        print(f"  -> skipped {config_name}: no SAITS model for plaus={config['plaus']}")
        continue

    cohort_df = builder.config(config_name)
    if cohort_df is None:
        print(f"  -> skipped {config_name}: config builder returned None")
        continue

    for arm_name, use_mask in MASK_ARMS.items():
        run_name = f"gru/{config_name}/{arm_name}"
        print(
            f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV"
            f" — Setup {run_name} [{', '.join(STRATIFICATION_MODES)}]…"
        )

        do_shap = config_name == "ma_pk_in" and arm_name == "masky"
        shap_paths = (
            {m: _RUN_DIRS[m] / "shap" / f"{config_name}_{arm_name}.npz" for m in STRATIFICATION_MODES}
            if do_shap
            else {}
        )
        do_feat_dist = f"{config_name}/{arm_name}" == _FEAT_DIST_RUN
        features_paths = (
            {m: _RUN_DIRS[m] / "features" / f"{config_name}_{arm_name}.npz" for m in STRATIFICATION_MODES}
            if do_feat_dist
            else {}
        )
        folds_by_mode = run_cv_gru(
            cohort_df, case_metrics, STRATIFICATION_MODES, run_name,
            use_mask=use_mask,
            shap_full=do_shap,
            shap_save_paths=shap_paths,
            features_save_paths=features_paths,
        )
        run_ts = datetime.now().isoformat(timespec="seconds")
        for strat_mode in STRATIFICATION_MODES:
            summary = summarise_cv(folds_by_mode[strat_mode], run_name)
            summary["_run_timestamp"] = run_ts
            pipeline_summaries_by_mode[strat_mode].append((f"Setup {run_name}", summary))

            plot_auprc_lift_by_stratum(
                pipeline_summaries_by_mode[strat_mode],
                _RUN_DIRS[strat_mode] / "figures" / "auprc_lift_by_stratum_gru.png",
            )
            save_cv_results(
                pipeline_summaries_by_mode[strat_mode],
                _RUN_DIRS[strat_mode] / "cv_results_gru_temp.csv",
            )

for strat_mode in STRATIFICATION_MODES:
    save_cv_results(
        pipeline_summaries_by_mode[strat_mode],
        _RUN_DIRS[strat_mode] / "cv_results_gru.csv",
    )
    (_RUN_DIRS[strat_mode] / "cv_results_gru_temp.csv").unlink(missing_ok=True)

# %%
