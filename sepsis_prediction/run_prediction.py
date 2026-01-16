# %%
import json
from datetime import datetime

import config as settings
import polars as pl
from imputation import export_imputation_elapsed_time, impute
from models import XGBOOST_AVAILABLE, XGBoostModel, evaluate_and_compare_models
from utils import (
    compute_encounter_missingness,
    generate_baseline_non_seq_features,
    generate_cohort,
    generate_imperfekt_non_seq_features,
)

from imperfekt.config.global_settings import VITALS

target_col = "sepsis_outcome"
first_minutes = [30, 20, 10]  # [30,20,10] # How early can we predict sepsis
minimum_durations = [
    40,
    30,
    20,
    10,
]  # [40,30,20,10] # Changes if we only look at longer records only or if we include all kinds of durations?
minimum_counts = [5, 8, 10]  # [5,8,10] # How little data points are needed
imputation_strategies = ["locf", "saits"]

pl.Config.set_tbl_cols(80)
pl.Config.set_tbl_rows(40)
path = settings.PATHS.get_nemsis_path(settings.VARIABLES.NEMSIS_YEAR_STR)
data_path = path / "model_input"
cols = VITALS.PARAMS
random_state = 42
store_additional_info = True
model_input_df = pl.read_parquet(data_path / "sepsis_prediction_input.parquet")

# %%
for first_minute in first_minutes:
    for minimum_dur in minimum_durations:
        if minimum_dur < first_minute:
            continue
        for minimum_count in minimum_counts:
            print(
                f"\n\n===== EXPERIMENT: First {first_minute} minutes, Minimum Duration {minimum_dur} minutes, Minimum Count {minimum_count} ====="
            )

            # DATA
            current_model_input_df = generate_cohort(
                model_input_df.clone(),
                first_minute=first_minute,
                minimum_dur=minimum_dur,
                minimum_count=minimum_count,
            )

            if current_model_input_df.filter(pl.col(target_col) == 1).n_unique("id") < 100:
                print("Not enough positive cases, skipping this configuration.")
                continue

            # CREATE SAVE PATH
            SAVE_RESULTS_PATH = (
                path
                / "results"
                / "imperfekt"
                / f"{datetime.now().strftime('%Y%m%d')}"
                / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{first_minute}minutes_{minimum_dur}minutes_{minimum_count}count"
            )
            SAVE_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            current_model_input_df.write_parquet(
                data_path
                / f"sepsis_prediction_input_{first_minute}minutes_{minimum_dur}minutes_{minimum_count}count_{current_model_input_df.shape[0]}.parquet"
            )

            # COMPUTE ENCOUNTER-LEVEL MISSINGNESS
            missingness_df = compute_encounter_missingness(
                current_model_input_df, cols=cols, id_col="id"
            )
            median_missingness = missingness_df["missingness_rate"].median()
            mean_missingness = missingness_df["missingness_rate"].mean()
            print(
                f"\nMissingness stats: median={median_missingness:.3f}, mean={mean_missingness:.3f}"
            )

            #  IMPUTATION OF MISSING VALUES
            # Impute missing values with forward fill per id
            model_input_imputed_df, imputation_times = {}, {}
            for strategy in imputation_strategies:
                model_input_imputed_df[strategy], imputation_times[strategy] = impute(
                    current_model_input_df,
                    cols=cols,
                    strategy=strategy,
                    save_training_log=SAVE_RESULTS_PATH / "imputation_logs",  # for SAITS training
                )

            # Save imputation times
            export_imputation_elapsed_time(SAVE_RESULTS_PATH, imputation_times)

            # FEATURE GENERATION
            imperfekt_features = generate_imperfekt_non_seq_features(
                current_model_input_df,
                target_col=target_col,
                cols=cols,
                fill_nulls=False,
                feature_name_filepath=SAVE_RESULTS_PATH / "imperfekt_feature_names.txt",
            )
            baseline_features = generate_baseline_non_seq_features(
                current_model_input_df,
                target_col=target_col,
                cols=cols,
                fill_nulls=False,
                feature_name_filepath=SAVE_RESULTS_PATH / "baseline_feature_names.txt",
            )
            locf_baseline_features = generate_baseline_non_seq_features(
                model_input_imputed_df["locf"], target_col=target_col, cols=cols, fill_nulls=False
            )
            saits_baseline_features = generate_baseline_non_seq_features(
                model_input_imputed_df["saits"], target_col=target_col, cols=cols, fill_nulls=False
            )
            union_features = imperfekt_features.join(
                baseline_features, on=["id", target_col], how="inner"
            )
            locf_union_features = imperfekt_features.join(
                locf_baseline_features, on=["id", target_col], how="inner"
            )
            saits_union_features = imperfekt_features.join(
                saits_baseline_features, on=["id", target_col], how="inner"
            )
            # MODELS
            # Create a dictionary to store model instances
            models = {
                "Base": {
                    "XGBoost": XGBoostModel(
                        feature_mode="baseline",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="baseline", save_path=SAVE_RESULTS_PATH),
                },
                "Miss": {
                    "XGBoost": XGBoostModel(
                        feature_mode="imperfekt",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="imperfekt", save_path=SAVE_RESULTS_PATH),
                },
                "Base+Miss": {
                    "XGBoost": XGBoostModel(
                        feature_mode="baseline+imperfekt",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="baseline+imperfekt", save_path=SAVE_RESULTS_PATH),
                },
                "Base-LOCF": {
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="imputed+baseline", save_path=SAVE_RESULTS_PATH, target_sensitivity=0.95),
                    "XGBoost": XGBoostModel(
                        feature_mode="locf+baseline",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                },
                "Base+Miss-LOCF": {
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="imputed+imperfekt", save_path=SAVE_RESULTS_PATH, target_sensitivity=0.95),
                    "XGBoost": XGBoostModel(
                        feature_mode="locf+baseline+imperfekt",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                },
                "Base-SAITS": {
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="imputed+baseline", save_path=SAVE_RESULTS_PATH, target_sensitivity=0.95),
                    "XGBoost": XGBoostModel(
                        feature_mode="saits+baseline",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                },
                "Base+Miss-SAITS": {
                    # "RandomForest": RandomForestModel(random_state=random_state, feature_mode="imputed+imperfekt", save_path=SAVE_RESULTS_PATH, target_sensitivity=0.95),
                    "XGBoost": XGBoostModel(
                        feature_mode="saits+baseline+imperfekt",
                        random_state=random_state,
                        calibrate=True,
                        save_path=SAVE_RESULTS_PATH,
                        target_sensitivity=0.95,
                    )
                    if XGBOOST_AVAILABLE
                    else None,
                },
            }

            # Train and evaluate models on different feature sets
            metrics_dfs = {}
            for feature_type, feature_df in {
                "Miss": imperfekt_features,
                "Base": baseline_features,
                "Base+Miss": union_features,
                "Base-LOCF": locf_baseline_features,
                "Base+Miss-LOCF": locf_union_features,
                "Base-SAITS": saits_baseline_features,
                "Base+Miss-SAITS": saits_union_features,
            }.items():
                print(f"\n\n===== MODELS TRAINED ON {feature_type.upper()} =====")
                metrics_df = evaluate_and_compare_models(
                    feature_df,
                    target_col,
                    models[feature_type],
                    missingness_df=missingness_df,  # Pass missingness for stratified evaluation
                )
                metrics_dfs[feature_type] = metrics_df

            # Combined results
            if metrics_dfs:
                for feature_type, metrics_df in metrics_dfs.items():
                    metrics_df = metrics_df.with_columns(
                        pl.lit(feature_type).alias("feature_type"),
                        pl.lit(current_model_input_df.n_unique("id")).alias("cohort_size"),
                    )
                    metrics_dfs[feature_type] = metrics_df

                # Combine results
                all_metrics_df = pl.concat([metrics for metrics in metrics_dfs.values()])

                # Display and save complete comparison
                print("\n--- Complete Model Comparison ---")
                print(f"Target variable: {target_col}\n")
                print(
                    all_metrics_df.sort(
                        ["model", "feature_type", "subset"], descending=[False, True, False]
                    )
                )

                # Save results
                all_metrics_df.write_csv(
                    SAVE_RESULTS_PATH
                    / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

                if store_additional_info:
                    # For each model, extract and save feature importance and SHAP values
                    for feature, feature_models in models.items():
                        for model_name, m in feature_models.items():
                            if m is not None:
                                if feature == "Miss":
                                    feature_names = imperfekt_features.drop(
                                        ["id", target_col]
                                    ).columns
                                elif feature == "Base":
                                    feature_names = baseline_features.drop(
                                        ["id", target_col]
                                    ).columns
                                elif feature == "Base+Miss":
                                    feature_names = union_features.drop(["id", target_col]).columns
                                elif feature == "Base-LOCF":
                                    feature_names = locf_baseline_features.drop(
                                        ["id", target_col]
                                    ).columns
                                elif feature == "Base+Miss-LOCF":
                                    feature_names = locf_union_features.drop(
                                        ["id", target_col]
                                    ).columns
                                elif feature == "Base-SAITS":
                                    feature_names = saits_baseline_features.drop(
                                        ["id", target_col]
                                    ).columns
                                elif feature == "Base+Miss-SAITS":
                                    feature_names = saits_union_features.drop(
                                        ["id", target_col]
                                    ).columns
                                else:
                                    feature_names = []
                                feature_importance = m.get_feature_importance(feature_names)
                                if feature_importance is not None:
                                    print(
                                        f"\n--- Feature Importance for {model_name} with {feature} ---"
                                    )
                                    print(feature_importance.head(10))
                                    feature_importance.write_csv(
                                        SAVE_RESULTS_PATH
                                        / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{feature}.csv"
                                    )

                                m.compute_shap(
                                    X=m.X_test,
                                    feature_names=feature_names,
                                    save_dir=SAVE_RESULTS_PATH,
                                    prefix=f"{feature}_{model_name}",
                                    file_format="png",
                                )

                                with open(
                                    SAVE_RESULTS_PATH
                                    / f"best_model_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{feature}.json",
                                    "w",
                                ) as f:
                                    json.dump(m.params, f, indent=4)

                # Filter only XGBoost models if available
                full_metrics_df = all_metrics_df.filter(
                    (pl.col("model") == "XGBoost") & (pl.col("subset") == "full")
                )
                # Extract feature importance for the best performing model if available
                best_model_idx = full_metrics_df["auc_pr_ci_lower"].arg_max()
                best_model_type = full_metrics_df[best_model_idx, "model"]
                best_feature_type = full_metrics_df[best_model_idx, "feature_type"]

                print(f"\nBest model: {best_model_type} with {best_feature_type}")
