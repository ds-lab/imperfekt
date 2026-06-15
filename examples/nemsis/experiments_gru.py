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

for strat_mode, axes in STRATIFICATION_MODES.items():
    RUN_DIR = RESULTS_DIR / _fp_tag / strat_mode
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_summaries = []

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
                f" — Setup {run_name} [{strat_mode}]…"
            )

            folds, _, _, _, _, _ = run_cv_gru(
                cohort_df, case_metrics, axes, run_name,
                use_mask=use_mask,
                stratification_mode=strat_mode,
            )
            summary = summarise_cv(folds, run_name)
            summary["_run_timestamp"] = datetime.now().isoformat(timespec="seconds")
            pipeline_summaries.append((f"Setup {run_name}", summary))

            plot_auprc_lift_by_stratum(
                pipeline_summaries,
                RUN_DIR / "figures" / "auprc_lift_by_stratum_gru.png",
            )
            save_cv_results(pipeline_summaries, RUN_DIR / "cv_results_gru_temp.csv")

    save_cv_results(pipeline_summaries, RUN_DIR / "cv_results_gru.csv")
    (RUN_DIR / "cv_results_gru_temp.csv").unlink(missing_ok=True)

# %%
