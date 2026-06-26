# %%
# Rebuild AUPRC / AUROC plots from a saved cv_results.csv — no CV re-run needed.
#
# Usage:
#   python examples/nemsis/plot_results.py
#   python examples/nemsis/plot_results.py path/to/cv_results.csv
#
# With no argument, reads RESULTS_DIR / cv_results.csv from config.py.
import json
import sys
from pathlib import Path
import glob

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from plotting import (
    plot_auprc_by_stratum,
    plot_auprc_lift_by_stratum,
    plot_auroc_by_stratum,
    plot_delta_auprc_heatmap,
    plot_shap_importance_bar,
    plot_shap_stability_scatter,
    plot_spearman_orthogonality,
)
from examples.nemsis.cv import load_cv_results, save_shap_importance_csv, compute_feature_distribution_by_quadrant, compute_feature_distribution_by_outcome
from config import RESULTS_DIR, STRATIFICATION_MODE, data_fingerprint_tag, COHORT_PATH, AXES_INTRAVARIABLE, AXES_INTERVARIABLE
import polars as pl

_fp_tag = data_fingerprint_tag(COHORT_PATH)
if STRATIFICATION_MODE == "intravariable":
    AXES = AXES_INTRAVARIABLE
else:
    AXES = AXES_INTERVARIABLE

print(f"AXES: {AXES}")

_RUN_DIR = RESULTS_DIR / _fp_tag / STRATIFICATION_MODE / f"{AXES[0]}_{AXES[1]}"
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _RUN_DIR / "cv_results.csv"
# experiments_gru.py writes GRU summaries to a sibling file, not cv_results.csv.
gru_csv_path = csv_path.parent / "cv_results_gru.csv"

# ── Filter ────────────────────────────────────────────────────────────────────
# Comment out any pipelines you don't want plotted.
# Set PIPELINES = None to plot everything in the CSV.
PIPELINES = [
    "Setup ma_pk_in/base",
    "Setup ma_pk_in/base+miss",
#    "Setup ma_pk_in/base+plaus",
#    "Setup ma_pk_in/base+miss+plaus",
    "Setup ma_pk_is/base",
    "Setup ma_pk_is/base+miss",
    "Setup ma_pk_il/base",
    "Setup ma_pk_il/base+miss",
 #   "Setup ma_pk_il/base+plaus",
 #   "Setup ma_pk_il/base+miss+plaus",
    "Setup ma_pr_in/base",
    "Setup ma_pr_in/base+miss",
   # "Setup ma_pr_in/base+plaus",
  #  "Setup ma_pr_in/base+miss+plaus",
    "Setup ma_pr_il/base",
    "Setup ma_pr_il/base+miss",
 #   "Setup ma_pr_il/base+plaus",
  #  "Setup ma_pr_il/base+miss+plaus",
]

# GRU pipelines (from experiments_gru.py: label = f"Setup gru/{config_name}/{arm_name}").
# Set GRU_PIPELINES = None to plot every GRU run found in the CSV.
GRU_PIPELINES = [
    "Setup gru/ma_pk_in/mask",
]
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading CV results from {csv_path}")
all_summaries = load_cv_results(csv_path)

if PIPELINES is not None:
    pipeline_summaries = [(label, s) for label, s in all_summaries if label in PIPELINES]
    missing = set(PIPELINES) - {label for label, _ in pipeline_summaries}
    if missing:
        print(f"WARNING: pipelines not found in CSV: {sorted(missing)}")
else:
    pipeline_summaries = all_summaries

print(f"Plotting {len(pipeline_summaries)}/{len(all_summaries)} pipelines:")
for label, summary in pipeline_summaries:
    ts = summary.get("_run_timestamp", "unknown")
    print(f"  {label}  (run at {ts})")

if gru_csv_path.exists():
    print(f"Loading GRU CV results from {gru_csv_path}")
    all_gru_summaries = load_cv_results(gru_csv_path)
else:
    print(f"No GRU CV results found at {gru_csv_path}")
    all_gru_summaries = []

if GRU_PIPELINES is not None:
    gru_pipeline_summaries = [(label, s) for label, s in all_gru_summaries if label in GRU_PIPELINES]
    gru_missing = set(GRU_PIPELINES) - {label for label, _ in gru_pipeline_summaries}
    if gru_missing:
        print(f"WARNING: GRU pipelines not found in {gru_csv_path.name}: {sorted(gru_missing)}")
else:
    gru_pipeline_summaries = all_gru_summaries

print(f"Plotting {len(gru_pipeline_summaries)} GRU pipelines:")
for label, summary in gru_pipeline_summaries:
    ts = summary.get("_run_timestamp", "unknown")
    print(f"  {label}  (run at {ts})")

# %%
figures_dir = csv_path.parent / "figures"

prevalence = {}
for _, summary in pipeline_summaries:
    for stratum, metrics in summary.items():
        if stratum.startswith("_") or not isinstance(metrics, dict):
            continue
        v = metrics.get("n_pos_pct")
        if v and stratum not in prevalence:
            prevalence[stratum] = {"mean": v["mean"], "ci": v["ci"]}

prev_path = csv_path.parent / "xgboost_stratum_prevalence.json"
prev_path.write_text(json.dumps(
    {s: {k: round(v, 6) for k, v in vals.items()} for s, vals in sorted(prevalence.items())},
    indent=2,
))
print(f"Stratum prevalence saved to {prev_path}")
for stratum, vals in sorted(prevalence.items()):
    print(f"  {stratum}: mean={vals['mean']:.4%}  ci={vals['ci']:.4%}")

gru_prevalence = {}
for _, summary in gru_pipeline_summaries:
    for stratum, metrics in summary.items():
        if stratum.startswith("_") or not isinstance(metrics, dict):
            continue
        v = metrics.get("n_pos_pct")
        if v and stratum not in gru_prevalence:
            gru_prevalence[stratum] = {"mean": v["mean"], "ci": v["ci"]}

gru_prev_path = csv_path.parent / "gru_stratum_prevalence.json"
gru_prev_path.write_text(json.dumps(
    {s: {k: round(v, 6) for k, v in vals.items()} for s, vals in sorted(gru_prevalence.items())},
    indent=2,
))
print(f"GRU stratum prevalence saved to {gru_prev_path}")
for stratum, vals in sorted(gru_prevalence.items()):
    print(f"  {stratum}: mean={vals['mean']:.4%}  ci={vals['ci']:.4%}")

# %%
plot_auprc_by_stratum(pipeline_summaries, figures_dir / "xgboost_auprc_by_stratum.png")
plot_auprc_lift_by_stratum(pipeline_summaries, figures_dir / "xgboost_auprc_lift_by_stratum.png")
plot_auroc_by_stratum(pipeline_summaries, figures_dir / "xgboost_auroc_by_stratum.png")

# Δ AUPRC heatmaps: variant − baseline (ma_pk_in/base) per quadrant.
# Uses all_summaries (not the PIPELINES-filtered set) so baseline + swaps are visible.
plot_delta_auprc_heatmap(all_summaries, figures_dir / "xgboost_delta_auprc_heatmap.png", metric="auprc")
plot_delta_auprc_heatmap(all_summaries, figures_dir / "xgboost_delta_auprc_lift_heatmap.png", metric="auprc_lift")

# By-stratum plots for GRU, analogous to the XGBoost ones above.
if gru_pipeline_summaries:
    plot_auprc_by_stratum(gru_pipeline_summaries, figures_dir / "gru_auprc_by_stratum.png")
    plot_auprc_lift_by_stratum(gru_pipeline_summaries, figures_dir / "gru_auprc_lift_by_stratum.png")
    plot_auroc_by_stratum(gru_pipeline_summaries, figures_dir / "gru_auroc_by_stratum.png")
else:
    print("\nNo GRU pipelines to plot — skipping GRU by-stratum plots.")

# Δ AUPRC heatmaps for GRU: variant − baseline (gru/ma_pk_in/mask) per quadrant.
# Uses all_gru_summaries so swaps not yet run just print a warning and are skipped.
_GRU_BASELINE = "Setup gru/ma_pk_in/mask"
_GRU_SWAPS = [
    ("imp → LOCF", "Setup gru/ma_pk_il/mask"),
    ("imp → SAITS", "Setup gru/ma_pk_is/mask"),
    ("plaus → remove", "Setup gru/ma_pr_in/mask"),
    ("plaus → remove, imp → LOCF", "Setup gru/ma_pr_il/mask"),
    ("plaus → remove, imp → SAITS", "Setup gru/ma_pr_is/mask"),
    ("no mask channel", "Setup gru/ma_pk_in/nomask"),
]
plot_delta_auprc_heatmap(
    all_gru_summaries, figures_dir / "gru_delta_auprc_heatmap.png",
    baseline=_GRU_BASELINE, swaps=_GRU_SWAPS, metric="auprc",
)
plot_delta_auprc_heatmap(
    all_gru_summaries, figures_dir / "gru_delta_auprc_lift_heatmap.png",
    baseline=_GRU_BASELINE, swaps=_GRU_SWAPS, metric="auprc_lift",
)

# %%
# ── Stratum characterisation table (missingness axes) ────────────────────────
# Find the most recently written case_metrics.parquet under the strata cache.
_STRATUM_ORDER = ["Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]

_cache_name = (
    "intravariable_strata_cache" if STRATIFICATION_MODE == "intravariable"
    else "intervariable_strata_cache"
)
_stratum_col = (
    "imperfection_stratum" if STRATIFICATION_MODE == "intravariable"
    else "intervariable_stratum"
)
cache_root = RESULTS_DIR / _cache_name
parquet_candidates = sorted(
    glob.glob(str(cache_root / "*" / "case_metrics.parquet")),
    key=lambda p: Path(p).stat().st_mtime,
)
if parquet_candidates:
    metrics_path = parquet_candidates[-1]
    print(f"\nBuilding stratum characterisation table from {metrics_path}")
    cm = pl.read_parquet(metrics_path)

    # Aggregate all numeric columns present in the cache — works for both modes.
    _meta_cols = {"id", "axis_x", "axis_y", "axis_pair_corr",
                  "axis_x_median_threshold", "axis_y_median_threshold",
                  "imperfection_stratum", "intervariable_stratum"}
    numeric_cols = [
        c for c in cm.columns
        if c not in _meta_cols and cm.schema[c].is_numeric()
    ]
    agg_exprs = [pl.len().alias("n")] + [
        expr
        for c in numeric_cols
        for expr in (
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
        )
    ]
    stratum_table = (
        cm.group_by(_stratum_col)
        .agg(agg_exprs)
        .with_columns(
            pl.col(_stratum_col).cast(pl.Enum(_STRATUM_ORDER)).alias("_sort_key")
        )
        .sort("_sort_key")
        .drop("_sort_key")
    )

    table_path = csv_path.parent / "stratum_characterisation.csv"
    stratum_table.write_csv(table_path)
    print(f"Stratum characterisation table saved to {table_path}")
    print(stratum_table)
else:
    print(f"\nNo case_metrics.parquet found under {_cache_name} — skipping stratum table.")
    cm = None

# %%
# ── Feature distribution by stratum + by outcome (wide tables) ───────────────
# Shared between XGBoost (stay-level aggregated features) and GRU (per-stay
# mean/min/max of raw vitals) — both write features .npz in the same schema
# (feature_names, X_test_all, y_test_all, strata_all), so the same builder works.
def _build_feature_distribution_tables(
    feat_dist_npz_path: Path, prefix: str, feat_dist_run: str,
) -> None:
    feat_quad_path = csv_path.parent / f"{prefix}_feature_distribution_by_quadrant.csv"
    feat_npz_data = (
        np.load(feat_dist_npz_path, allow_pickle=True) if feat_dist_npz_path.exists() else None
    )
    if not feat_quad_path.exists() and feat_npz_data is not None:
        compute_feature_distribution_by_quadrant(feat_dist_npz_path, feat_quad_path, npz_data=feat_npz_data)
    if feat_quad_path.exists() and cm is not None:
        print(f"\nBuilding {prefix} feature-by-stratum wide table from {feat_quad_path}")
        fq = pl.read_csv(feat_quad_path)

        # n per stratum from case_metrics → proportion of all stratified cases
        n_by_stratum = (
            cm.filter(pl.col(_stratum_col).is_not_null())
            .group_by(_stratum_col)
            .len()
        )
        total_cases = n_by_stratum["len"].sum()
        n_by_stratum = n_by_stratum.with_columns(
            (pl.col("len") / total_cases * 100).alias("pct")
        )

        # Format mean ± ci for each feature × stratum cell
        fq = fq.with_columns(
            pl.concat_str(
                pl.col("mean").round(2).cast(pl.Utf8),
                pl.lit(" ± "),
                pl.col("ci").round(2).cast(pl.Utf8),
            ).alias("mean_ci")
        )

        # Pivot: rows = (feature_group, feature), columns = strata
        strata_present = [s for s in _STRATUM_ORDER if s in fq["stratum"].unique().to_list()]
        wide = (
            fq.select(["feature", "feature_group", "stratum", "mean_ci"])
            .pivot(index=["feature_group", "feature"], on="stratum", values="mean_ci")
            .sort(["feature_group", "feature"])
        )
        # Reorder stratum columns to canonical order
        col_order = ["feature_group", "feature"] + [s for s in strata_present]
        wide = wide.select([c for c in col_order if c in wide.columns])

        # Outcome prevalence row (from fold summaries stored in fq)
        if "outcome_prevalence_mean" in fq.columns:
            prev_row = (
                fq.group_by("stratum")
                .agg(pl.col("outcome_prevalence_mean").mean(), pl.col("outcome_prevalence_ci").mean())
                .with_columns(
                    pl.concat_str(
                        (pl.col("outcome_prevalence_mean") * 100).round(2).cast(pl.Utf8),
                        pl.lit(" ± "),
                        (pl.col("outcome_prevalence_ci") * 100).round(2).cast(pl.Utf8),
                    ).alias("mean_ci")
                )
            )
            prev_pivot = (
                prev_row.select(["stratum", "mean_ci"])
                .with_columns(pl.lit(0).alias("_row"))
                .pivot(index="_row", on="stratum", values="mean_ci")
                .drop("_row")
            )
            # Proportion of cases row
            prop_pivot = (
                n_by_stratum.with_columns(
                    pl.concat_str(
                        pl.col("len").cast(pl.Utf8),
                        pl.lit(" ("),
                        pl.col("pct").round(1).cast(pl.Utf8),
                        pl.lit("%)"),
                    ).alias("mean_ci")
                )
                .select([_stratum_col, "mean_ci"])
                .rename({_stratum_col: "stratum"})
                .with_columns(pl.lit(0).alias("_row"))
                .pivot(index="_row", on="stratum", values="mean_ci")
                .drop("_row")
            )

            header_rows = pl.concat([prop_pivot, prev_pivot], how="diagonal_relaxed").with_columns(
                pl.lit("").alias("feature_group"),
                pl.lit("").alias("feature"),
            )
            wide = pl.concat([header_rows.select(wide.columns), wide], how="diagonal_relaxed")

        wide_path = csv_path.parent / f"{prefix}_feature_distribution_by_quadrant_wide.csv"
        wide.write_csv(wide_path)
        print(f"{prefix} feature-by-quadrant wide table saved to {wide_path}")
    elif not feat_quad_path.exists():
        print(f"\nNo {prefix} feature_distribution_by_quadrant.csv found — re-run with features_save_path enabled for {feat_dist_run!r}.")

    feat_outcome_path = csv_path.parent / f"{prefix}_feature_distribution_by_outcome.csv"
    if feat_npz_data is not None:
        compute_feature_distribution_by_outcome(feat_dist_npz_path, feat_outcome_path, npz_data=feat_npz_data)
    if feat_outcome_path.exists():
        print(f"\n{prefix} feature distribution by outcome from {feat_outcome_path}")
        fo = pl.read_csv(feat_outcome_path)
        # Show mean ± std for each feature grouped by outcome, sorted by feature_group then feature
        fo_wide = (
            fo.with_columns(
                pl.concat_str(
                    pl.col("mean").round(3).cast(pl.Utf8),
                    pl.lit(" ± "),
                    pl.col("ci").round(3).cast(pl.Utf8),
                ).alias("mean_ci")
            )
            .select(["feature_group", "feature", "outcome", "mean_ci"])
            .pivot(index=["feature_group", "feature"], on="outcome", values="mean_ci")
            .sort(["feature_group", "feature"])
        )
        fo_wide_path = csv_path.parent / f"{prefix}_feature_distribution_by_outcome_wide.csv"
        fo_wide.write_csv(fo_wide_path)
        print(f"{prefix} feature-by-outcome wide table saved to {fo_wide_path}")
    else:
        print(f"\nNo {prefix} feature_distribution_by_outcome.csv found and no features .npz at {feat_dist_npz_path}.")


# XGBoost: stay-level feature-set .npz uses "__" to join config/setup in the filename.
_feat_dist_run = "ma_pk_in/base+miss"
_feat_dist_npz = _RUN_DIR / "features" / f"{_feat_dist_run.replace('/', '__')}.npz"
_build_feature_distribution_tables(_feat_dist_npz, "xgboost", _feat_dist_run)

# GRU: experiments_gru.py saves to features/{config_name}_{arm_name}.npz (single "_").
_gru_feat_dist_run = "ma_pk_in/mask"
_gru_feat_dist_config, _gru_feat_dist_arm = _gru_feat_dist_run.split("/")
_gru_feat_dist_npz = _RUN_DIR / "features" / f"{_gru_feat_dist_config}_{_gru_feat_dist_arm}.npz"
_build_feature_distribution_tables(_gru_feat_dist_npz, "gru", _gru_feat_dist_run)

# %%
# ── SHAP Interpretability & Stability ─────────────────────────────────────────
# SHAP files always live in _RUN_DIR/shap/, independent of where csv_path came from.
shap_dir = _RUN_DIR / "shap"
features_dir = _RUN_DIR / "features"
npz_paths = [
    (label, shap_dir / f"{label.removeprefix('Setup ').replace('/', '__')}.npz")
    for label, _ in pipeline_summaries
]
npz_paths_found = [(label, p) for label, p in npz_paths if p.exists()]

if npz_paths_found:
    shap_csv_path = csv_path.parent / "xgboost_shap_importance.csv"
    save_shap_importance_csv(npz_paths_found, shap_csv_path)
    shap_df = pl.read_csv(shap_csv_path)

    strata_in_shap = shap_df["stratum"].unique().to_list()
    for stratum in ["overall"] + [s for s in _STRATUM_ORDER if s in strata_in_shap]:
        tag = stratum.lower().replace("_", "")
        plot_shap_importance_bar(
            shap_df, figures_dir / f"xgboost_shap_importance_bar_{tag}.png", stratum=stratum
        )
        plot_shap_stability_scatter(
            shap_df, figures_dir / f"xgboost_shap_stability_scatter_{tag}.png", stratum=stratum
        )

    for label, shap_npz_path in npz_paths_found:
        run_name = label.removeprefix("Setup ").replace("/", "__")
        features_npz_path = features_dir / f"{run_name}.npz"
        plot_spearman_orthogonality(shap_npz_path, features_npz_path, label, figures_dir)
else:
    print("\nNo SHAP .npz files found — re-run experiments.py to generate them.")

# %%
# ── GRU SHAP — temporal importance & mask channel ─────────────────────────────
import matplotlib.pyplot as plt  # noqa: E402

# experiments_gru.py saves to RUN_DIR/shap/{config_name}_{arm_name}.npz, where
# label = f"Setup gru/{config_name}/{arm_name}" — derive the same filename here.
def _gru_shap_filename(label: str) -> str:
    _, config_name, arm_name = label.removeprefix("Setup ").split("/")
    return f"{config_name}_{arm_name}.npz"

gru_shap_paths = [
    (label, shap_dir / _gru_shap_filename(label))
    for label, _ in gru_pipeline_summaries
]
gru_shap_found = [(label, p) for label, p in gru_shap_paths if p.exists()]

if gru_shap_found:
    gru_shap_csv_path = csv_path.parent / "gru_shap_importance.csv"
    save_shap_importance_csv(gru_shap_found, gru_shap_csv_path)
    gru_shap_df = pl.read_csv(gru_shap_csv_path)

    strata_in_gru_shap = gru_shap_df["stratum"].unique().to_list()
    for stratum in ["overall"] + [s for s in _STRATUM_ORDER if s in strata_in_gru_shap]:
        tag = stratum.lower().replace("_", "")
        plot_shap_importance_bar(
            gru_shap_df,
            figures_dir / f"gru_shap_importance_bar_{tag}.png",
            stratum=stratum,
        )
        plot_shap_stability_scatter(
            gru_shap_df,
            figures_dir / f"gru_shap_stability_scatter_{tag}.png",
            stratum=stratum,
        )

    # Per-quadrant temporal importance, one line per vital sign (value channel,
    # solid) and per mask channel (dashed, same color) — shap_time_feat_{stratum}
    # and shap_time_feat_mask_{stratum} both have shape (n_folds, T, D).
    for label, shap_npz_path in gru_shap_found:
        data = np.load(shap_npz_path, allow_pickle=True)
        run_name = label.removeprefix("Setup ").replace("/", "__")
        feature_names = data["feature_names"].tolist()
        time_keys = [k for k in data.files if k.startswith("shap_time_feat_") and not k.startswith("shap_time_feat_mask_")]
        if not time_keys:
            continue
        strata_present = sorted(k[len("shap_time_feat_"):] for k in time_keys)
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))

        for stratum_label in strata_present:
            val_arr = data[f"shap_time_feat_{stratum_label}"]              # (n_folds, T, D)
            mask_key = f"shap_time_feat_mask_{stratum_label}"
            mask_arr = data[mask_key] if mask_key in data.files else None  # (n_folds, T, D) or None

            fig, ax = plt.subplots(figsize=(10, 5))
            mean_val = val_arr.mean(axis=0)  # (T, D)
            for i, feat in enumerate(feature_names):
                ax.plot(mean_val[:, i], label=feat, color=colors[i], linestyle="-")
            if mask_arr is not None:
                mean_mask = mask_arr.mean(axis=0)  # (T, D)
                for i, feat in enumerate(feature_names):
                    ax.plot(mean_mask[:, i], color=colors[i], linestyle="--", alpha=0.7)

            ax.set_xlabel("Time step (clock index)")
            ax.set_ylabel("Mean |SHAP|")
            handles = [
                plt.Line2D([0], [0], color=colors[i], linestyle="-", label=feat)
                for i, feat in enumerate(feature_names)
            ]
            if mask_arr is not None:
                handles += [
                    plt.Line2D([0], [0], color="black", linestyle="-", label="value"),
                    plt.Line2D([0], [0], color="black", linestyle="--", label="mask"),
                ]
            ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")
            fig.tight_layout()
            tag = stratum_label.lower().replace("_", "")
            out_path = figures_dir / f"gru_shap_time_{run_name}_{tag}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"GRU temporal SHAP saved to {out_path}")
else:
    print("\nNo GRU SHAP .npz files found — re-run experiments with shap_full=True.")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO EVALUATION — Cross-dataset plots and tables
# ══════════════════════════════════════════════════════════════════════════════
from plotting import (
    plot_cross_dataset_delta_heatmap,
    render_stratum_axis_table,
)

_SCENARIO_ROOT = ROOT / "examples" / "data" / "scenario_evaluation"
_SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)

# ── Load both datasets (intervariable + intravariable) ────────────────────────
_NEMSIS_BASE = ROOT / "examples" / "data" / "nemsis" / "post_publication_results" / "65260d86861d"
_MCMED_BASE = ROOT / "examples" / "data" / "mcmed" / "post_publication_results" / "75927eb4c6de"

def _load_as_dict(path: Path) -> dict[str, dict]:
    """Load cv_results CSV and return {pipeline_label: summary_dict}."""
    if not path.exists():
        print(f"WARNING: {path} not found")
        return {}
    summaries = load_cv_results(path)
    return {label: s for label, s in summaries}

print("\n" + "=" * 72)
print("SCENARIO EVALUATION — Loading both datasets × both stratification modes")
print("=" * 72)

_strat_data: dict[str, dict] = {}
for mode in ("intervariable", "intravariable"):
    if mode == "intravariable":
        _axes = AXES_INTRAVARIABLE
    else:
        _axes = AXES_INTERVARIABLE
    _axes_tag = f"{_axes[0]}_{_axes[1]}"
    nemsis_xgb = _load_as_dict(_NEMSIS_BASE / mode / _axes_tag / "cv_results.csv")
    nemsis_gru = _load_as_dict(_NEMSIS_BASE / mode / _axes_tag / "cv_results_gru.csv")
    mcmed_xgb = _load_as_dict(_MCMED_BASE / mode / _axes_tag / "cv_results.csv")
    mcmed_gru = _load_as_dict(_MCMED_BASE / mode / _axes_tag / "cv_results_gru.csv")
    _strat_data[mode] = {
        "nemsis_xgb": nemsis_xgb, "nemsis_gru": nemsis_gru,
        "mcmed_xgb": mcmed_xgb, "mcmed_gru": mcmed_gru,
    }
    print(f"  [{mode}] NEMSIS XGB={len(nemsis_xgb)} GRU={len(nemsis_gru)} | MC-MED XGB={len(mcmed_xgb)} GRU={len(mcmed_gru)}")

# %%
# ── Generate scenario plots for BOTH stratification modes ─────────────────────

_AXIS_METRICS_INTERVARIABLE = [
    "avg_indicated_vars_pct",
    "co_missingness_concentration",
    "missing_variable_breadth",
    "pattern_entropy",
    "max_pairwise_co_missingness",
]
_AXIS_METRICS_INTRAVARIABLE = [
    "sbp_indicated_pct",
    "rr_indicated_pct",
    "hr_indicated_pct",
    "o2sat_indicated_pct",
]

for _mode in ("intervariable", "intravariable"):
    _d = _strat_data[_mode]
    nemsis_xgb = _d["nemsis_xgb"]
    nemsis_gru = _d["nemsis_gru"]
    mcmed_xgb = _d["mcmed_xgb"]
    mcmed_gru = _d["mcmed_gru"]

    if _mode == "intravariable":
        _mode_axes = AXES_INTRAVARIABLE
    else:
        _mode_axes = AXES_INTERVARIABLE
    _mode_dir = _SCENARIO_ROOT / _mode / f"{_mode_axes[0]}_{_mode_axes[1]}"
    _xgb_datasets = {"NEMSIS": nemsis_xgb, "MC-MED": mcmed_xgb}
    _gru_datasets = {"NEMSIS": nemsis_gru, "MC-MED": mcmed_gru}
    _xgb_baselines = {"NEMSIS": "Setup ma_pk_in/base", "MC-MED": "Setup ma_pk_in/base"}
    _gru_baselines = {"NEMSIS": "Setup gru/ma_pk_in/nomask", "MC-MED": "Setup gru/ma_pk_in/nomask"}
    _heatmap_rows = ["overall", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]

    print(f"\n{'─' * 72}")
    print(f"  Stratification mode: {_mode}")
    print(f"{'─' * 72}")

    # ── SCENARIO 1 ────────────────────────────────────────────────────────────
    print(f"  [{_mode}] Scenario 1: Stratum Characterisation Tables")
    _s1_dir = _mode_dir / "scenario_1"
    _s1_dir.mkdir(parents=True, exist_ok=True)

    _axis_metrics = _AXIS_METRICS_INTERVARIABLE if _mode == "intervariable" else _AXIS_METRICS_INTRAVARIABLE

    _baseline_nemsis = nemsis_xgb.get("Setup ma_pk_in/base", {})
    _baseline_mcmed = mcmed_xgb.get("Setup ma_pk_in/base", {})

    if _baseline_nemsis:
        render_stratum_axis_table(
            _baseline_nemsis, _axis_metrics,
            _s1_dir / "nemsis_stratum_axis_metrics.csv",
        )
    if _baseline_mcmed:
        render_stratum_axis_table(
            _baseline_mcmed, _axis_metrics,
            _s1_dir / "mcmed_stratum_axis_metrics.csv",
        )

    # ── SCENARIO 2 ────────────────────────────────────────────────────────────
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
        row_order=_heatmap_rows,
    )

    plot_cross_dataset_delta_heatmap(
        _gru_datasets, _s2_dir / "gru_imputation_delta_heatmap.png",
        baselines=_gru_baselines,
        variants=[
            ("LOCF", {"NEMSIS": "Setup gru/ma_pk_il/nomask", "MC-MED": "Setup gru/ma_pk_il/nomask"}),
            ("SAITS", {"NEMSIS": "Setup gru/ma_pk_is/nomask", "MC-MED": "Setup gru/ma_pk_is/nomask"}),
        ],
        metric="auprc_lift",
        row_order=_heatmap_rows,
    )

    _s2_xgb_pipes = ["Setup ma_pk_in/base", "Setup ma_pk_il/base", "Setup ma_pk_is/base"]
    _s2_gru_pipes = ["Setup gru/ma_pk_in/nomask", "Setup gru/ma_pk_il/nomask", "Setup gru/ma_pk_is/nomask"]

    for ds_name, xgb_data, gru_data in [("nemsis", nemsis_xgb, nemsis_gru), ("mcmed", mcmed_xgb, mcmed_gru)]:
        xgb_sums = [(l, s) for l, s in xgb_data.items() if l in _s2_xgb_pipes]
        if xgb_sums:
            plot_auprc_lift_by_stratum(xgb_sums, _s2_dir / f"{ds_name}_xgboost_imputation_by_stratum.png")
        gru_sums = [(l, s) for l, s in gru_data.items() if l in _s2_gru_pipes]
        if gru_sums:
            plot_auprc_lift_by_stratum(gru_sums, _s2_dir / f"{ds_name}_gru_imputation_by_stratum.png")

    # ── SCENARIO 3 ────────────────────────────────────────────────────────────
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
        row_order=_heatmap_rows,
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
        row_order=_heatmap_rows,
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
        row_order=_heatmap_rows,
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
        row_order=_heatmap_rows,
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

    for ds_name, xgb_data, gru_data in [("nemsis", nemsis_xgb, nemsis_gru), ("mcmed", mcmed_xgb, mcmed_gru)]:
        xgb_sums = [(l, s) for l, s in xgb_data.items() if l in _s3_xgb_pipes]
        if xgb_sums:
            plot_auprc_lift_by_stratum(xgb_sums, _s3_dir / f"{ds_name}_xgboost_plausibility_by_stratum.png")
        gru_sums = [(l, s) for l, s in gru_data.items() if l in _s3_gru_pipes]
        if gru_sums:
            plot_auprc_lift_by_stratum(gru_sums, _s3_dir / f"{ds_name}_gru_plausibility_by_stratum.png")

    # ── SCENARIO 4 ────────────────────────────────────────────────────────────
    print(f"  [{_mode}] Scenario 4: Imperfection-Aware Features Heatmaps")
    _s4_dir = _mode_dir / "scenario_4"
    _s4_dir.mkdir(parents=True, exist_ok=True)

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
        row_order=["overall", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"],
    )

    plot_cross_dataset_delta_heatmap(
        _xgb_datasets, _s4_dir / "xgboost_miss_features_delta_heatmap.png",
        baselines=_xgb_baselines,
        variants=[
            ("+miss", {"NEMSIS": "Setup ma_pk_in/base+miss", "MC-MED": "Setup ma_pk_in/base+miss"}),
            ("+plaus", {"NEMSIS": "Setup ma_pk_in/base+plaus", "MC-MED": "Setup ma_pk_in/base+plaus"}),
            ("+miss+plaus", {"NEMSIS": "Setup ma_pk_in/base+miss+plaus", "MC-MED": "Setup ma_pk_in/base+miss+plaus"}),
        ],
        metric="auprc_lift",
        row_order=_heatmap_rows,
    )

    # 4b. By-stratum line plots
    _s4_xgb_pipes = [
        "Setup ma_pk_in/base", "Setup ma_pk_in/base+miss",
        "Setup ma_pk_in/base+plaus", "Setup ma_pk_in/base+miss+plaus",
    ]
    _s4_gru_pipes = ["Setup gru/ma_pk_in/nomask", "Setup gru/ma_pk_in/mask"]

    for ds_name, xgb_data, gru_data in [("nemsis", nemsis_xgb, nemsis_gru), ("mcmed", mcmed_xgb, mcmed_gru)]:
        xgb_sums = [(l, s) for l, s in xgb_data.items() if l in _s4_xgb_pipes]
        if xgb_sums:
            plot_auprc_lift_by_stratum(xgb_sums, _s4_dir / f"{ds_name}_xgboost_features_by_stratum.png")
        gru_sums = [(l, s) for l, s in gru_data.items() if l in _s4_gru_pipes]
        if gru_sums:
            plot_auprc_lift_by_stratum(gru_sums, _s4_dir / f"{ds_name}_gru_mask_by_stratum.png")

# 4c. SHAP plots are kept from the existing code above — no changes needed.

print("\n" + "=" * 72)
print("SCENARIO EVALUATION COMPLETE")
print(f"Results in: {_SCENARIO_ROOT}")
print("=" * 72)
# %%
