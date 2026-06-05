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

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from plotting import plot_auprc_by_stratum, plot_auprc_lift_by_stratum, plot_auroc_by_stratum
from examples.nemsis.cv import load_cv_results
from config import RESULTS_DIR

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "cv_results.csv"

# ── Filter ────────────────────────────────────────────────────────────────────
# Comment out any pipelines you don't want plotted.
# Set PIPELINES = None to plot everything in the CSV.
PIPELINES = [
    "Setup ma_pk_in/base",
    "Setup ma_pk_in/base+miss",
#    "Setup ma_pk_in/base+plaus",
#    "Setup ma_pk_in/base+miss+plaus",
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

prev_path = csv_path.parent / "stratum_prevalence.json"
prev_path.write_text(json.dumps(
    {s: {k: round(v, 6) for k, v in vals.items()} for s, vals in sorted(prevalence.items())},
    indent=2,
))
print(f"Stratum prevalence saved to {prev_path}")
for stratum, vals in sorted(prevalence.items()):
    print(f"  {stratum}: mean={vals['mean']:.4%}  ci={vals['ci']:.4%}")

# %%
plot_auprc_by_stratum(pipeline_summaries, figures_dir / "auprc_by_stratum.svg")
plot_auprc_lift_by_stratum(pipeline_summaries, figures_dir / "auprc_lift_by_stratum.svg")
plot_auroc_by_stratum(pipeline_summaries, figures_dir / "auroc_by_stratum.svg")

# %%
# ── Stratum characterisation table (missingness axes) ────────────────────────
# Find the most recently written case_metrics.parquet under the strata cache.
import polars as pl

_STRATUM_ORDER = ["Q_complete", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]

cache_root = csv_path.parent / "intervariable_strata_cache"
parquet_candidates = sorted(
    glob.glob(str(cache_root / "*" / "case_metrics.parquet")),
    key=lambda p: Path(p).stat().st_mtime,
)
if parquet_candidates:
    metrics_path = parquet_candidates[-1]
    print(f"\nBuilding stratum characterisation table from {metrics_path}")
    cm = pl.read_parquet(metrics_path)

    stratum_table = (
        cm.group_by("intervariable_stratum")
        .agg(
            pl.len().alias("n"),
            pl.col("avg_indicated_vars_pct").mean().alias("avg_indicated_vars_pct_mean"),
            pl.col("avg_indicated_vars_pct").std().alias("avg_indicated_vars_pct_std"),
            pl.col("co_missingness_concentration").mean().alias("co_missingness_concentration_mean"),
            pl.col("co_missingness_concentration").std().alias("co_missingness_concentration_std"),
            pl.col("missing_variable_breadth").mean().alias("missing_variable_breadth_mean"),
            pl.col("missing_variable_breadth").std().alias("missing_variable_breadth_std"),
            pl.col("pattern_entropy").mean().alias("pattern_entropy_mean"),
            pl.col("pattern_entropy").std().alias("pattern_entropy_std"),
            pl.col("max_pairwise_co_missingness").mean().alias("max_pairwise_co_missingness_mean"),
            pl.col("max_pairwise_co_missingness").std().alias("max_pairwise_co_missingness_std"),
        )
        .with_columns(
            pl.col("intervariable_stratum").cast(pl.Enum(_STRATUM_ORDER)).alias("_sort_key")
        )
        .sort("_sort_key")
        .drop("_sort_key")
    )

    table_path = csv_path.parent / "stratum_characterisation.csv"
    stratum_table.write_csv(table_path)
    print(f"Stratum characterisation table saved to {table_path}")
    print(stratum_table)
else:
    print("\nNo case_metrics.parquet found under intervariable_strata_cache — skipping stratum table.")
    cm = None

# %%
# ── Feature distribution by stratum (wide table) ─────────────────────────────
# Requires feature_distribution_by_quadrant.csv produced by experiments.py.
feat_quad_path = csv_path.parent / "feature_distribution_by_quadrant.csv"
if feat_quad_path.exists() and cm is not None:
    print(f"\nBuilding feature-by-stratum wide table from {feat_quad_path}")
    fq = pl.read_csv(feat_quad_path)

    # n per stratum from case_metrics → proportion of all stratified cases
    n_by_stratum = (
        cm.filter(pl.col("intervariable_stratum").is_not_null())
        .group_by("intervariable_stratum")
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
            .pivot(index=pl.lit("Outcome Prevalence (%)"), on="stratum", values="mean_ci")
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
            .select(["intervariable_stratum", "mean_ci"])
            .rename({"intervariable_stratum": "stratum"})
            .pivot(index=pl.lit("N (% of stratified cases)"), on="stratum", values="mean_ci")
        )

        header_rows = pl.concat([prop_pivot, prev_pivot], how="diagonal_relaxed").with_columns(
            pl.lit("").alias("feature_group"),
        )
        header_rows = header_rows.rename({header_rows.columns[0]: "feature"})
        wide = pl.concat([header_rows.select(wide.columns), wide], how="diagonal_relaxed")

    wide_path = csv_path.parent / "feature_distribution_by_stratum_wide.csv"
    wide.write_csv(wide_path)
    print(f"Feature-by-stratum wide table saved to {wide_path}")
    print(wide.head(10))
else:
    if not feat_quad_path.exists():
        print(f"\nNo feature_distribution_by_quadrant.csv found — re-run experiments.py to generate it.")
# %%
