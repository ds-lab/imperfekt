"""
MIMIC-IV-ED irregularity experiment
====================================
Compares XGBoost pipelines for different prediction tasks (e.g. 30-day readmission, in-hospital mortality):

  Pipeline 0 – raw irregular timestamps, statistical aggregates of vital signs only
  Pipeline A – regular 30-min resampled grid, statistical aggregates only
  Pipeline B – raw irregular intervals, statistical + imperfekt irregularity aggregates
  Pipeline C – raw imperfekt irregularity features, then 30-min resample + fill,
               then statistical + imperfekt irregularity aggregates
  Pipeline D – Pipeline 0 + observation-count feature (timestamps per stay)

Performance is estimated with repeated stratified k-fold cross-validation
(5 folds × 10 repeats = 50 fits per pipeline).  Irregularity quadrant thresholds
(LL/HL/LH/HH) are derived from each fold's train set — no test leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from examples.mimic_iv_ed.extract_cohort import build_cohort  # noqa: E402
from examples.mimic_iv_ed.config import (  # noqa: E402
    RESULTS_DIR,
    OUTCOME_COL,
    WINDOW_HOURS,
    MIN_OBS,
    MAX_MISSINGNESS,
    CV_N_SPLITS,
    CV_N_REPEATS,
)
from examples.mimic_iv_ed.features import (  # noqa: E402
    pipeline_0_features,
    pipeline_d_features,
    pipeline_a_features,
    pipeline_b_features,
    pipeline_c_features,
    build_stay_level,
)
from examples.mimic_iv_ed.cv import (  # noqa: E402
    compute_irregularity_strata,
    run_cv,
    summarise_cv,
    print_information_gain_ratio,
)
from examples.mimic_iv_ed.plotting import (  # noqa: E402
    plot_auprc_by_stratum,
    plot_auprc_lift_by_stratum,
    run_shap_group_analysis,
)


def load_cohort() -> pl.DataFrame:
    return build_cohort(
        ["critical_outcome", "ed_stay_length"],
        min_observations=MIN_OBS,
        window_hours=WINDOW_HOURS,
        max_missingness=MAX_MISSINGNESS,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading cohort (first {WINDOW_HOURS} h of ED stay, ≥{MIN_OBS} observations)…")
    ts_df = load_cohort()
    print(f"Cohort: {ts_df['stay_id'].n_unique()} stays, {len(ts_df)} observations")
    print(f"Outcome prevalence: {ts_df[OUTCOME_COL].mean():.3f} ({ts_df[OUTCOME_COL].sum()}/{len(ts_df)})")

    print("\nComputing irregularity strata on full dataset (axes + raw metrics only)…")
    case_metrics, axes = compute_irregularity_strata(ts_df)

    print("\nBuilding stay-level feature frames…")
    stay_0 = build_stay_level(ts_df, pipeline_0_features)
    stay_d = build_stay_level(ts_df, pipeline_d_features)
    stay_a = build_stay_level(ts_df, pipeline_a_features)
    stay_b = build_stay_level(ts_df, pipeline_b_features)
    stay_c = build_stay_level(ts_df, pipeline_c_features)

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline 0…")
    folds_0, _, _, _, _, _ = run_cv(stay_0, case_metrics, axes, "Pipeline0")
    summary_0 = summarise_cv(folds_0, "Pipeline0")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline D…")
    folds_d, _, _, _, _, _ = run_cv(stay_d, case_metrics, axes, "PipelineD")
    summary_d = summarise_cv(folds_d, "PipelineD")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline A…")
    folds_a, _, _, _, _, _ = run_cv(stay_a, case_metrics, axes, "PipelineA")
    summary_a = summarise_cv(folds_a, "PipelineA")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline B…")
    folds_b, last_model_b, last_X_test_b, last_test_df_b, feat_cols_b, last_test_strata_b = run_cv(stay_b, case_metrics, axes, "PipelineB")
    summary_b = summarise_cv(folds_b, "PipelineB")

    print(f"\nRunning {CV_N_SPLITS}×{CV_N_REPEATS} repeated stratified k-fold CV — Pipeline C…")
    folds_c, _, _, _, _, _ = run_cv(stay_c, case_metrics, axes, "PipelineC")
    summary_c = summarise_cv(folds_c, "PipelineC")

    print_information_gain_ratio(summary_0, summary_d, "Pipeline0", "PipelineD", metric="auprc_lift", test_set_name="overall")
    print_information_gain_ratio(summary_0, summary_b, "Pipeline0", "PipelineB", metric="auprc_lift", test_set_name="overall")
    print_information_gain_ratio(summary_0, summary_c, "Pipeline0", "PipelineA", metric="auprc_lift", test_set_name="overall")

    print("\nPlotting AUPRC by stratum…")
    pipeline_summaries = [
        ("Pipeline 0 (raw stats)", summary_0),
        ("Pipeline D (raw stats + observation count)", summary_d),
        ("Pipeline A (resampled)", summary_a),
        ("Pipeline B (imperfekt)", summary_b),
        ("Pipeline C (resampled, imperfekt)", summary_c),
    ]
    plot_auprc_by_stratum(
        pipeline_summaries,
        RESULTS_DIR / "figures" / "auprc_by_stratum.svg",
        show_legend=False,
    )
    plot_auprc_lift_by_stratum(
        pipeline_summaries,
        RESULTS_DIR / "figures" / "auprc_lift_by_stratum.svg",
        show_legend=False,
    )

    print("\nComputing SHAP group-importance analysis for Pipeline B (last CV fold)…")
    if last_model_b is not None and last_X_test_b is not None and last_test_df_b is not None and last_test_strata_b is not None:
        run_shap_group_analysis(
            model=last_model_b,
            X_test=last_X_test_b,
            test_stay_df=last_test_df_b,
            strata=last_test_strata_b,
            feature_cols=feat_cols_b,
            pipeline_name="PipelineB",
        )
    else:
        print("Skipping SHAP for Pipeline B: no final-fold artifacts available.")

    print("\nDone.")


if __name__ == "__main__":
    main()
