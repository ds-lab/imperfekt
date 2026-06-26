# %%
# Scenario evaluation: cross-dataset plots and tables for all datasets,
# models, and stratification modes.
#
# Usage:
#   python examples/nemsis/plot_results.py
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from plotting import (
    plot_auprc_by_stratum,
    plot_auprc_lift_by_stratum,
    plot_auroc_by_stratum,
    plot_cross_dataset_delta_heatmap,
    plot_delta_auprc_heatmap,
    plot_shap_importance_bar,
    plot_shap_stability_scatter,
    plot_spearman_orthogonality,
    render_stratum_axis_table,
)
from examples.nemsis.cv import (
    load_cv_results,
    save_shap_importance_csv,
    compute_feature_distribution_by_quadrant,
    compute_feature_distribution_by_outcome,
)
from config import AXES_INTRAVARIABLE, AXES_INTERVARIABLE

# ── Toggle flags ──────────────────────────────────────────────────────────────
ENABLE_SHAP = True  # Set to False to skip SHAP analysis (expensive)

# ── Pipeline filters for Scenario 5 by-stratum plots ─────────────────────────
# Set to None to include all pipelines found in CSV.
XGB_PIPELINES: list[str] | None = [
    "Setup ma_pk_in/base",
    "Setup ma_pk_in/base+miss",
    "Setup ma_pk_is/base",
    "Setup ma_pk_is/base+miss",
    "Setup ma_pk_il/base",
    "Setup ma_pk_il/base+miss",
    "Setup ma_pr_in/base",
    "Setup ma_pr_in/base+miss",
    "Setup ma_pr_il/base",
    "Setup ma_pr_il/base+miss",
]
GRU_PIPELINES: list[str] | None = [
    "Setup gru/ma_pk_in/mask",
]

# ── Constants ─────────────────────────────────────────────────────────────────
_STRATUM_ORDER = ["Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]
_HEATMAP_ROWS = ["overall", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]

_AXIS_METRICS = {
    "intervariable": [
        "avg_indicated_vars_pct",
        "co_missingness_concentration",
        "missing_variable_breadth",
        "pattern_entropy",
        "max_pairwise_co_missingness",
    ],
    "intravariable": [
        "sbp_indicated_pct",
        "rr_indicated_pct",
        "hr_indicated_pct",
        "o2sat_indicated_pct",
    ],
}

# ── Dataset paths ─────────────────────────────────────────────────────────────
_SCENARIO_ROOT = ROOT / "examples" / "data" / "scenario_evaluation"
_SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)

_DATASET_BASES = {
    "nemsis": ROOT / "examples" / "data" / "nemsis" / "post_publication_results" / "65260d86861d",
    "mcmed": ROOT / "examples" / "data" / "mcmed" / "post_publication_results" / "75927eb4c6de",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_as_dict(path: Path) -> dict[str, dict]:
    if not path.exists():
        print(f"WARNING: {path} not found")
        return {}
    return {label: s for label, s in load_cv_results(path)}


def _filter_pipelines(
    data: dict[str, dict], allowed: list[str] | None,
) -> list[tuple[str, dict]]:
    if allowed is None:
        return list(data.items())
    return [(l, s) for l, s in data.items() if l in allowed]


def _gru_shap_filename(label: str) -> str:
    _, config_name, arm_name = label.removeprefix("Setup ").split("/")
    return f"{config_name}_{arm_name}.npz"


def _build_feature_distribution_tables(
    feat_npz_path: Path, output_dir: Path, prefix: str,
) -> None:
    if not feat_npz_path.exists():
        return
    npz_data = np.load(feat_npz_path, allow_pickle=True)
    quad_path = output_dir / f"{prefix}_feature_distribution_by_quadrant.csv"
    compute_feature_distribution_by_quadrant(
        feat_npz_path, quad_path, npz_data=npz_data,
    )
    outcome_path = output_dir / f"{prefix}_feature_distribution_by_outcome.csv"
    compute_feature_distribution_by_outcome(
        feat_npz_path, outcome_path, npz_data=npz_data,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load all datasets
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("Loading both datasets × both stratification modes")
print("=" * 72)

_strat_data: dict[str, dict] = {}
for mode in ("intervariable", "intravariable"):
    _axes = AXES_INTERVARIABLE if mode == "intervariable" else AXES_INTRAVARIABLE
    _axes_tag = f"{_axes[0]}_{_axes[1]}"
    _per_mode: dict[str, dict] = {}
    for ds_name, ds_base in _DATASET_BASES.items():
        ds_run = ds_base / mode / _axes_tag
        _per_mode[f"{ds_name}_xgb"] = _load_as_dict(ds_run / "cv_results.csv")
        _per_mode[f"{ds_name}_gru"] = _load_as_dict(ds_run / "cv_results_gru.csv")
    _strat_data[mode] = _per_mode
    print(
        f"  [{mode}] "
        + " | ".join(
            f"{ds.upper()} XGB={len(_per_mode[f'{ds}_xgb'])} "
            f"GRU={len(_per_mode[f'{ds}_gru'])}"
            for ds in _DATASET_BASES
        )
    )


# %%
# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

for _mode in ("intervariable", "intravariable"):
    _d = _strat_data[_mode]
    nemsis_xgb, nemsis_gru = _d["nemsis_xgb"], _d["nemsis_gru"]
    mcmed_xgb, mcmed_gru = _d["mcmed_xgb"], _d["mcmed_gru"]

    _mode_axes = AXES_INTERVARIABLE if _mode == "intervariable" else AXES_INTRAVARIABLE
    _axes_tag = f"{_mode_axes[0]}_{_mode_axes[1]}"
    _mode_dir = _SCENARIO_ROOT / _mode / _axes_tag
    _xgb_datasets = {"NEMSIS": nemsis_xgb, "MC-MED": mcmed_xgb}
    _gru_datasets = {"NEMSIS": nemsis_gru, "MC-MED": mcmed_gru}
    _xgb_baselines = {"NEMSIS": "Setup ma_pk_in/base", "MC-MED": "Setup ma_pk_in/base"}
    _gru_baselines = {
        "NEMSIS": "Setup gru/ma_pk_in/nomask",
        "MC-MED": "Setup gru/ma_pk_in/nomask",
    }

    print(f"\n{'─' * 72}")
    print(f"  Stratification mode: {_mode}")
    print(f"{'─' * 72}")

    # ── SCENARIO 1: Stratum Characterisation ──────────────────────────────────
    print(f"  [{_mode}] Scenario 1: Stratum Characterisation Tables")
    _s1_dir = _mode_dir / "scenario_1"
    _s1_dir.mkdir(parents=True, exist_ok=True)

    for ds_label, xgb_data in [("nemsis", nemsis_xgb), ("mcmed", mcmed_xgb)]:
        _baseline = xgb_data.get("Setup ma_pk_in/base", {})
        if _baseline:
            render_stratum_axis_table(
                _baseline, _AXIS_METRICS[_mode],
                _s1_dir / f"{ds_label}_stratum_axis_metrics.csv",
            )

    # ── SCENARIO 2: Imputation ────────────────────────────────────────────────
    print(f"  [{_mode}] Scenario 2: Imputation Delta Heatmaps")
    _s2_dir = _mode_dir / "scenario_2"
    _s2_dir.mkdir(parents=True, exist_ok=True)

    plot_cross_dataset_delta_heatmap(
        _xgb_datasets, _s2_dir / "xgboost_imputation_delta_heatmap.png",
        baselines=_xgb_baselines,
        variants=[
            ("LOCF", {"NEMSIS": "Setup ma_pk_il/base", "MC-MED": "Setup ma_pk_il/base"}),
            ("SAITS", {"NEMSIS": "Setup ma_pk_is/base", "MC-MED": "Setup ma_pk_is/base"}),
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    plot_cross_dataset_delta_heatmap(
        _gru_datasets, _s2_dir / "gru_imputation_delta_heatmap.png",
        baselines=_gru_baselines,
        variants=[
            ("LOCF", {"NEMSIS": "Setup gru/ma_pk_il/nomask", "MC-MED": "Setup gru/ma_pk_il/nomask"}),
            ("SAITS", {"NEMSIS": "Setup gru/ma_pk_is/nomask", "MC-MED": "Setup gru/ma_pk_is/nomask"}),
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    _s2_xgb_pipes = [
        "Setup ma_pk_in/base", "Setup ma_pk_il/base", "Setup ma_pk_is/base",
    ]
    _s2_gru_pipes = [
        "Setup gru/ma_pk_in/nomask",
        "Setup gru/ma_pk_il/nomask",
        "Setup gru/ma_pk_is/nomask",
    ]
    for ds_name, xgb_data, gru_data in [
        ("nemsis", nemsis_xgb, nemsis_gru),
        ("mcmed", mcmed_xgb, mcmed_gru),
    ]:
        xgb_sums = [(l, s) for l, s in xgb_data.items() if l in _s2_xgb_pipes]
        if xgb_sums:
            plot_auprc_lift_by_stratum(
                xgb_sums,
                _s2_dir / f"{ds_name}_xgboost_imputation_by_stratum.png",
            )
        gru_sums = [(l, s) for l, s in gru_data.items() if l in _s2_gru_pipes]
        if gru_sums:
            plot_auprc_lift_by_stratum(
                gru_sums,
                _s2_dir / f"{ds_name}_gru_imputation_by_stratum.png",
            )

    # ── SCENARIO 3: Plausibility Handling ─────────────────────────────────────
    print(f"  [{_mode}] Scenario 3: Plausibility Handling Heatmaps")
    _s3_dir = _mode_dir / "scenario_3"
    _s3_dir.mkdir(parents=True, exist_ok=True)

    # 3a. XGBoost: outlier removal vs absolute baseline
    plot_cross_dataset_delta_heatmap(
        _xgb_datasets, _s3_dir / "xgboost_outlier_vs_baseline_delta_heatmap.png",
        baselines=_xgb_baselines,
        variants=[
            ("pr none", {"NEMSIS": "Setup ma_pr_in/base", "MC-MED": "Setup ma_pr_in/base"}),
            ("pr LOCF", {"NEMSIS": "Setup ma_pr_il/base", "MC-MED": "Setup ma_pr_il/base"}),
            ("pr SAITS", {"NEMSIS": "Setup ma_pr_is/base", "MC-MED": "Setup ma_pr_is/base"}),
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    # 3a. GRU: outlier removal vs absolute baseline
    plot_cross_dataset_delta_heatmap(
        _gru_datasets, _s3_dir / "gru_outlier_vs_baseline_delta_heatmap.png",
        baselines=_gru_baselines,
        variants=[
            ("pr none", {"NEMSIS": "Setup gru/ma_pr_in/nomask", "MC-MED": "Setup gru/ma_pr_in/nomask"}),
            ("pr LOCF", {"NEMSIS": "Setup gru/ma_pr_il/nomask", "MC-MED": "Setup gru/ma_pr_il/nomask"}),
            ("pr SAITS", {"NEMSIS": "Setup gru/ma_pr_is/nomask", "MC-MED": "Setup gru/ma_pr_is/nomask"}),
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    # 3b. XGBoost: marginal effect of outlier removal (pr − pk, per imputation)
    plot_cross_dataset_delta_heatmap(
        _xgb_datasets, _s3_dir / "xgboost_outlier_marginal_delta_heatmap.png",
        baselines=_xgb_baselines,
        variants=[
            ("none:\npr−pk", {"NEMSIS": "Setup ma_pr_in/base", "MC-MED": "Setup ma_pr_in/base"}),
            ("LOCF:\npr−pk", {"NEMSIS": "Setup ma_pr_il/base", "MC-MED": "Setup ma_pr_il/base"}),
            ("SAITS:\npr−pk", {"NEMSIS": "Setup ma_pr_is/base", "MC-MED": "Setup ma_pr_is/base"}),
        ],
        variant_baselines=[
            {"NEMSIS": "Setup ma_pk_in/base", "MC-MED": "Setup ma_pk_in/base"},
            {"NEMSIS": "Setup ma_pk_il/base", "MC-MED": "Setup ma_pk_il/base"},
            {"NEMSIS": "Setup ma_pk_is/base", "MC-MED": "Setup ma_pk_is/base"},
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    # 3b. GRU: marginal effect of outlier removal (pr − pk, per imputation)
    plot_cross_dataset_delta_heatmap(
        _gru_datasets, _s3_dir / "gru_outlier_marginal_delta_heatmap.png",
        baselines=_gru_baselines,
        variants=[
            ("none:\npr−pk", {"NEMSIS": "Setup gru/ma_pr_in/nomask", "MC-MED": "Setup gru/ma_pr_in/nomask"}),
            ("LOCF:\npr−pk", {"NEMSIS": "Setup gru/ma_pr_il/nomask", "MC-MED": "Setup gru/ma_pr_il/nomask"}),
            ("SAITS:\npr−pk", {"NEMSIS": "Setup gru/ma_pr_is/nomask", "MC-MED": "Setup gru/ma_pr_is/nomask"}),
        ],
        variant_baselines=[
            {"NEMSIS": "Setup gru/ma_pk_in/nomask", "MC-MED": "Setup gru/ma_pk_in/nomask"},
            {"NEMSIS": "Setup gru/ma_pk_il/nomask", "MC-MED": "Setup gru/ma_pk_il/nomask"},
            {"NEMSIS": "Setup gru/ma_pk_is/nomask", "MC-MED": "Setup gru/ma_pk_is/nomask"},
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    # 3c. By-stratum line plots
    _s3_xgb_pipes = [
        "Setup ma_pk_in/base", "Setup ma_pk_il/base", "Setup ma_pk_is/base",
        "Setup ma_pr_in/base", "Setup ma_pr_il/base", "Setup ma_pr_is/base",
    ]
    _s3_gru_pipes = [
        "Setup gru/ma_pk_in/nomask", "Setup gru/ma_pk_il/nomask", "Setup gru/ma_pk_is/nomask",
        "Setup gru/ma_pr_in/nomask", "Setup gru/ma_pr_il/nomask", "Setup gru/ma_pr_is/nomask",
    ]
    for ds_name, xgb_data, gru_data in [
        ("nemsis", nemsis_xgb, nemsis_gru),
        ("mcmed", mcmed_xgb, mcmed_gru),
    ]:
        xgb_sums = [(l, s) for l, s in xgb_data.items() if l in _s3_xgb_pipes]
        if xgb_sums:
            plot_auprc_lift_by_stratum(
                xgb_sums,
                _s3_dir / f"{ds_name}_xgboost_plausibility_by_stratum.png",
            )
        gru_sums = [(l, s) for l, s in gru_data.items() if l in _s3_gru_pipes]
        if gru_sums:
            plot_auprc_lift_by_stratum(
                gru_sums,
                _s3_dir / f"{ds_name}_gru_plausibility_by_stratum.png",
            )

    # ── SCENARIO 4: Imperfection-Aware Features ──────────────────────────────
    print(f"  [{_mode}] Scenario 4: Imperfection-Aware Features")
    _s4_dir = _mode_dir / "scenario_4"
    _s4_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Cross-model heatmap (XGBoost +miss vs GRU +mask)
    _s4_combined_nemsis = {**nemsis_xgb, **nemsis_gru}
    _s4_combined_mcmed = {**mcmed_xgb, **mcmed_gru}
    _s4_datasets = {"NEMSIS": _s4_combined_nemsis, "MC-MED": _s4_combined_mcmed}

    plot_cross_dataset_delta_heatmap(
        _s4_datasets, _s4_dir / "miss_features_delta_heatmap.png",
        baselines={"NEMSIS": "Setup ma_pk_in/base", "MC-MED": "Setup ma_pk_in/base"},
        variants=[
            ("XGBoost\n+miss", {"NEMSIS": "Setup ma_pk_in/base+miss", "MC-MED": "Setup ma_pk_in/base+miss"}),
            ("GRU\n+mask", {"NEMSIS": "Setup gru/ma_pk_in/mask", "MC-MED": "Setup gru/ma_pk_in/mask"}),
        ],
        variant_baselines=[
            {"NEMSIS": "Setup ma_pk_in/base", "MC-MED": "Setup ma_pk_in/base"},
            {"NEMSIS": "Setup gru/ma_pk_in/nomask", "MC-MED": "Setup gru/ma_pk_in/nomask"},
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    # 4a. XGBoost structural feature variants
    plot_cross_dataset_delta_heatmap(
        _xgb_datasets, _s4_dir / "xgboost_miss_features_delta_heatmap.png",
        baselines=_xgb_baselines,
        variants=[
            ("+miss", {"NEMSIS": "Setup ma_pk_in/base+miss", "MC-MED": "Setup ma_pk_in/base+miss"}),
            ("+plaus", {"NEMSIS": "Setup ma_pk_in/base+plaus", "MC-MED": "Setup ma_pk_in/base+plaus"}),
            ("+miss+plaus", {"NEMSIS": "Setup ma_pk_in/base+miss+plaus", "MC-MED": "Setup ma_pk_in/base+miss+plaus"}),
        ],
        metric="auprc_lift",
        row_order=_HEATMAP_ROWS,
    )

    # 4b. By-stratum line plots
    _s4_xgb_pipes = [
        "Setup ma_pk_in/base", "Setup ma_pk_in/base+miss",
        "Setup ma_pk_in/base+plaus", "Setup ma_pk_in/base+miss+plaus",
    ]
    _s4_gru_pipes = ["Setup gru/ma_pk_in/nomask", "Setup gru/ma_pk_in/mask"]

    for ds_name, xgb_data, gru_data in [
        ("nemsis", nemsis_xgb, nemsis_gru),
        ("mcmed", mcmed_xgb, mcmed_gru),
    ]:
        xgb_sums = [(l, s) for l, s in xgb_data.items() if l in _s4_xgb_pipes]
        if xgb_sums:
            plot_auprc_lift_by_stratum(
                xgb_sums,
                _s4_dir / f"{ds_name}_xgboost_features_by_stratum.png",
            )
        gru_sums = [(l, s) for l, s in gru_data.items() if l in _s4_gru_pipes]
        if gru_sums:
            plot_auprc_lift_by_stratum(
                gru_sums,
                _s4_dir / f"{ds_name}_gru_mask_by_stratum.png",
            )

    # ── 4c: SHAP Interpretability ─────────────────────────────────────────────
    if ENABLE_SHAP:
        print(f"  [{_mode}] Scenario 4c: SHAP Interpretability")
        _s4c_dir = _s4_dir / "shap"

        for ds_name in _DATASET_BASES:
            ds_run_dir = _DATASET_BASES[ds_name] / _mode / _axes_tag
            shap_dir = ds_run_dir / "shap"
            features_dir = ds_run_dir / "features"
            if not shap_dir.exists():
                print(f"    [{ds_name}] No shap/ directory — skipping")
                continue

            ds_shap_dir = _s4c_dir / ds_name
            ds_shap_dir.mkdir(parents=True, exist_ok=True)

            xgb_data = nemsis_xgb if ds_name == "nemsis" else mcmed_xgb
            gru_data = nemsis_gru if ds_name == "nemsis" else mcmed_gru

            # XGBoost SHAP: importance bars, stability scatter, Spearman
            xgb_npz = [
                (label, shap_dir / f"{label.removeprefix('Setup ').replace('/', '__')}.npz")
                for label in xgb_data
            ]
            xgb_npz_found = [(l, p) for l, p in xgb_npz if p.exists()]

            if xgb_npz_found:
                shap_csv = ds_shap_dir / "xgboost_shap_importance.csv"
                save_shap_importance_csv(xgb_npz_found, shap_csv)
                shap_df = pl.read_csv(shap_csv)
                strata = shap_df["stratum"].unique().to_list()
                for stratum in ["overall"] + [
                    s for s in _STRATUM_ORDER if s in strata
                ]:
                    tag = stratum.lower().replace("_", "")
                    plot_shap_importance_bar(
                        shap_df,
                        ds_shap_dir / f"xgboost_shap_importance_bar_{tag}.png",
                        stratum=stratum,
                    )
                    plot_shap_stability_scatter(
                        shap_df,
                        ds_shap_dir / f"xgboost_shap_stability_scatter_{tag}.png",
                        stratum=stratum,
                    )
                for label, npz_path in xgb_npz_found:
                    run_name = label.removeprefix("Setup ").replace("/", "__")
                    feat_npz = features_dir / f"{run_name}.npz"
                    plot_spearman_orthogonality(
                        npz_path, feat_npz, label, ds_shap_dir,
                    )
            else:
                print(f"    [{ds_name}] No XGBoost SHAP .npz files found")

            # GRU SHAP: importance bars, stability scatter, temporal
            gru_labels = [l for l in gru_data if l.startswith("Setup gru/")]
            gru_npz = [
                (l, shap_dir / _gru_shap_filename(l)) for l in gru_labels
            ]
            gru_npz_found = [(l, p) for l, p in gru_npz if p.exists()]

            if gru_npz_found:
                gru_shap_csv = ds_shap_dir / "gru_shap_importance.csv"
                save_shap_importance_csv(gru_npz_found, gru_shap_csv)
                gru_shap_df = pl.read_csv(gru_shap_csv)
                strata = gru_shap_df["stratum"].unique().to_list()
                for stratum in ["overall"] + [
                    s for s in _STRATUM_ORDER if s in strata
                ]:
                    tag = stratum.lower().replace("_", "")
                    plot_shap_importance_bar(
                        gru_shap_df,
                        ds_shap_dir / f"gru_shap_importance_bar_{tag}.png",
                        stratum=stratum,
                    )
                    plot_shap_stability_scatter(
                        gru_shap_df,
                        ds_shap_dir / f"gru_shap_stability_scatter_{tag}.png",
                        stratum=stratum,
                    )

                # GRU temporal SHAP
                for label, npz_path in gru_npz_found:
                    data = np.load(npz_path, allow_pickle=True)
                    run_name = label.removeprefix("Setup ").replace("/", "__")
                    feature_names = data["feature_names"].tolist()
                    time_keys = [
                        k for k in data.files
                        if k.startswith("shap_time_feat_")
                        and not k.startswith("shap_time_feat_mask_")
                    ]
                    if not time_keys:
                        continue
                    strata_present = sorted(
                        k[len("shap_time_feat_"):] for k in time_keys
                    )
                    colors = plt.cm.tab10(
                        np.linspace(0, 1, len(feature_names))
                    )

                    for stratum_label in strata_present:
                        val_arr = data[f"shap_time_feat_{stratum_label}"]
                        mask_key = f"shap_time_feat_mask_{stratum_label}"
                        mask_arr = (
                            data[mask_key]
                            if mask_key in data.files
                            else None
                        )

                        fig, ax = plt.subplots(figsize=(10, 5))
                        mean_val = val_arr.mean(axis=0)
                        for i, feat in enumerate(feature_names):
                            ax.plot(
                                mean_val[:, i],
                                label=feat,
                                color=colors[i],
                                linestyle="-",
                            )
                        if mask_arr is not None:
                            mean_mask = mask_arr.mean(axis=0)
                            for i, feat in enumerate(feature_names):
                                ax.plot(
                                    mean_mask[:, i],
                                    color=colors[i],
                                    linestyle="--",
                                    alpha=0.7,
                                )

                        ax.set_xlabel("Time step (clock index)")
                        ax.set_ylabel("Mean |SHAP|")
                        handles = [
                            plt.Line2D(
                                [0], [0],
                                color=colors[i],
                                linestyle="-",
                                label=feat,
                            )
                            for i, feat in enumerate(feature_names)
                        ]
                        if mask_arr is not None:
                            handles += [
                                plt.Line2D([0], [0], color="black", linestyle="-", label="value"),
                                plt.Line2D([0], [0], color="black", linestyle="--", label="mask"),
                            ]
                        ax.legend(
                            handles=handles,
                            fontsize=7,
                            ncol=2,
                            loc="upper right",
                        )
                        fig.tight_layout()
                        tag = stratum_label.lower().replace("_", "")
                        out_path = ds_shap_dir / f"gru_shap_time_{run_name}_{tag}.png"
                        fig.savefig(out_path, dpi=150)
                        plt.close(fig)
                        print(f"    GRU temporal SHAP saved to {out_path}")
            else:
                print(f"    [{ds_name}] No GRU SHAP .npz files found")

    # ── SCENARIO 5: Miscellaneous Evaluations ─────────────────────────────────
    print(f"  [{_mode}] Scenario 5: Miscellaneous Evaluations")
    _s5_dir = _mode_dir / "scenario_5"
    _s5_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in _DATASET_BASES:
        xgb_data = nemsis_xgb if ds_name == "nemsis" else mcmed_xgb
        gru_data = nemsis_gru if ds_name == "nemsis" else mcmed_gru
        ds_run_dir = _DATASET_BASES[ds_name] / _mode / _axes_tag
        ds_dir = _s5_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # 5a. Stratum prevalence
        for model_name, model_data in [("xgboost", xgb_data), ("gru", gru_data)]:
            prev: dict[str, dict] = {}
            for summary in model_data.values():
                for stratum, metrics in summary.items():
                    if stratum.startswith("_") or not isinstance(metrics, dict):
                        continue
                    v = metrics.get("n_pos_pct")
                    if v and stratum not in prev:
                        prev[stratum] = {"mean": v["mean"], "ci": v["ci"]}
            if prev:
                prev_path = ds_dir / f"{model_name}_stratum_prevalence.json"
                prev_path.write_text(json.dumps(
                    {s: {k: round(v, 6) for k, v in vals.items()}
                     for s, vals in sorted(prev.items())},
                    indent=2,
                ))
                print(f"    [{ds_name}] {model_name} prevalence → {prev_path.name}")

        # 5b. By-stratum line plots (AUPRC, AUPRC lift, AUROC)
        xgb_filtered = _filter_pipelines(xgb_data, XGB_PIPELINES)
        gru_filtered = _filter_pipelines(gru_data, GRU_PIPELINES)

        if xgb_filtered:
            plot_auprc_by_stratum(xgb_filtered, ds_dir / "xgboost_auprc_by_stratum.png")
            plot_auprc_lift_by_stratum(xgb_filtered, ds_dir / "xgboost_auprc_lift_by_stratum.png")
            plot_auroc_by_stratum(xgb_filtered, ds_dir / "xgboost_auroc_by_stratum.png")
        if gru_filtered:
            plot_auprc_by_stratum(gru_filtered, ds_dir / "gru_auprc_by_stratum.png")
            plot_auprc_lift_by_stratum(gru_filtered, ds_dir / "gru_auprc_lift_by_stratum.png")
            plot_auroc_by_stratum(gru_filtered, ds_dir / "gru_auroc_by_stratum.png")

        # 5c. Single-dataset delta heatmaps (variant vs baseline per quadrant)
        xgb_all = list(xgb_data.items())
        gru_all = list(gru_data.items())

        if xgb_all:
            plot_delta_auprc_heatmap(
                xgb_all, ds_dir / "xgboost_delta_auprc_heatmap.png",
                metric="auprc",
            )
            plot_delta_auprc_heatmap(
                xgb_all, ds_dir / "xgboost_delta_auprc_lift_heatmap.png",
                metric="auprc_lift",
            )

        _gru_baseline = "Setup gru/ma_pk_in/nomask"
        _gru_swaps = [
            ("imp → LOCF", "Setup gru/ma_pk_il/nomask"),
            ("imp → SAITS", "Setup gru/ma_pk_is/nomask"),
            ("plaus → remove", "Setup gru/ma_pr_in/nomask"),
            ("plaus → remove, imp → LOCF", "Setup gru/ma_pr_il/nomask"),
            ("plaus → remove, imp → SAITS", "Setup gru/ma_pr_is/nomask"),
            ("+ mask channel", "Setup gru/ma_pk_in/mask"),
        ]
        if gru_all:
            plot_delta_auprc_heatmap(
                gru_all, ds_dir / "gru_delta_auprc_heatmap.png",
                baseline=_gru_baseline, swaps=_gru_swaps, metric="auprc",
            )
            plot_delta_auprc_heatmap(
                gru_all, ds_dir / "gru_delta_auprc_lift_heatmap.png",
                baseline=_gru_baseline, swaps=_gru_swaps, metric="auprc_lift",
            )

        # 5d. Feature distribution tables
        features_dir = ds_run_dir / "features"
        _build_feature_distribution_tables(
            features_dir / "ma_pk_in__base+miss.npz", ds_dir, "xgboost",
        )
        _build_feature_distribution_tables(
            features_dir / "ma_pk_in_mask.npz", ds_dir, "gru",
        )

print("\n" + "=" * 72)
print("SCENARIO EVALUATION COMPLETE")
print(f"Results in: {_SCENARIO_ROOT}")
print("=" * 72)
# %%
