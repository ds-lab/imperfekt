# %%
from pathlib import Path

import polars as pl
from config import COHORT_MIN_READINGS, COHORT_PATH, COHORT_WINDOW_MINUTES

from imperfekt import Imperfekt

pl.Config.set_tbl_cols(8)
pl.Config.set_tbl_rows(25)

df = pl.read_parquet(Path(COHORT_PATH))
# %%
df_filtered = (
    df.with_columns(pl.col("clock").min().over("PcrKey").alias("_start_clock"))
    .with_columns(
        ((pl.col("clock") - pl.col("_start_clock")).dt.total_minutes()).alias("_minutes_from_start")
    )
    .filter(pl.col("_minutes_from_start") <= COHORT_WINDOW_MINUTES)
    .drop(["_start_clock", "_minutes_from_start"])
)

valid_keys = (
    df_filtered.group_by("PcrKey")
    .agg(pl.col("clock").count().alias("_num_vitals"))
    .filter(pl.col("_num_vitals") >= COHORT_MIN_READINGS)
    .select("PcrKey")
)

df_filtered = df_filtered.join(valid_keys, on="PcrKey", how="inner")

# label per PcrKey (constant within a case)
labels = df_filtered.select(["PcrKey", "label"]).unique("PcrKey")

print("=== Label prevalence per PcrKey ===")
print(
    df_filtered.select(["PcrKey", "label"])
    .unique("PcrKey")
    .group_by("label")
    .agg(pl.len().alias("n_cases"))
    .with_columns((pl.col("n_cases") / pl.col("n_cases").sum() * 100).round(1).alias("pct_cases"))
    .sort("label")
)
print(df_filtered["PcrKey"].n_unique(), "unique PcrKeys")

# %%
imp = Imperfekt(
    imperfection="missingness",
    df=df_filtered,
    id_col="PcrKey",
    clock_col="clock",
    cols=["sbp", "hr", "o2sat", "rr"],
    save_path=None,
    renderer=None,
    plot_library="matplotlib",
)

# ── Intravariable ────────────────────────────────────────────────────────────

# %%
# stratify=True is valid here: this is one pooled cohort, so the quadrant
# thresholds are fitted once and every case is labelled on the same scale.
imp.intravariable.case_metrics(stratify=True, save_results=False)

intra_scores = imp.intravariable.results.cm_case_metrics
assert intra_scores is not None

# Selected axis pair per variable
print("\n=== Intravariable: selected axis pairs ===")
print(
    intra_scores.select(
        [
            "variable",
            "axis_x",
            "axis_y",
            "axis_pair_corr",
            "axis_x_median_threshold",
            "axis_y_median_threshold",
        ]
    )
    .unique(["variable", "axis_x", "axis_y"])
    .sort("variable")
)

# All pairwise rho values (including non-selected pairs)
print("\n=== Intravariable: all axis-pair correlations (rho) ===")
for var, corr_tbl in imp.intravariable.results.cm_pairwise_correlations.items():
    print(f"\n-- variable: {var} --")
    print(corr_tbl.sort("abs_corr"))

print("\n=== Null stratum diagnosis ===")
null_strata = intra_scores.filter(
    pl.col("indicated_pct").gt(0) & pl.col("imperfection_stratum").is_null()
)
print(f"Imperfect cases with null stratum: {null_strata.height}")
if null_strata.height > 0:
    print(
        null_strata.select(
            [
                "PcrKey",
                "variable",
                "indicated_pct",
                "indicated_centroid",
                "gap_entropy",
                "gap_adh_rate",
                "axis_x",
                "axis_y",
                "imperfection_stratum",
            ]
        ).head(20)
    )

intra_scores = intra_scores.join(labels, on="PcrKey", how="left")


def intra_dist(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("imperfection_stratum").is_not_null())

    dist = (
        df.group_by(["variable", "imperfection_stratum"])
        .agg(
            pl.len().alias("n"),
            (pl.col("label").sum() / pl.len() * 100).round(1).alias("label1_pct"),
            pl.col("indicated_pct").mean().round(2).alias("mean_indicated_pct"),
            pl.col("indicated_centroid").mean().round(3).alias("mean_centroid"),
            pl.col("gap_adh_rate").mean().round(3).alias("mean_gap_adh_rate"),
        )
        .with_columns(
            (pl.col("n") / pl.col("n").sum().over("variable") * 100).round(1).alias("stratum_pct")
        )
        .sort(["variable", "imperfection_stratum"])
    )
    for var, var_df in dist.group_by("variable", maintain_order=True):
        var_total = var_df["n"].sum()
        print(f"\n=== Intravariable | variable={var[0]} (n={var_total}) ===")
        print(var_df.drop("variable"))


# %%
pl.Config.set_tbl_cols(100)
intra_dist(intra_scores)

# ── Intervariable ────────────────────────────────────────────────────────────

# %%
imp.intervariable.case_metrics(stratify=True, save_results=False)

inter_scores = imp.intervariable.results.cm_case_metrics
assert inter_scores is not None

# Selected axis pair (single pair for the whole cohort)
print("\n=== Intervariable: selected axis pair ===")
print(
    inter_scores.select(
        ["axis_x", "axis_y", "axis_pair_corr", "axis_x_median_threshold", "axis_y_median_threshold"]
    ).unique()
)

# All pairwise rho values (including non-selected pairs)
print("\n=== Intervariable: all axis-pair correlations (rho) ===")
inter_pair_corrs = imp.intervariable.results.cm_pairwise_correlations
if inter_pair_corrs is not None:
    print(inter_pair_corrs.sort("abs_corr"))

inter_scores = inter_scores.join(labels, on="PcrKey", how="left")


def inter_dist(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("intervariable_stratum").is_not_null())
    total = len(df)
    ax_x = str(df["axis_x"].drop_nulls().first())
    ax_y = str(df["axis_y"].drop_nulls().first())
    dist = (
        df.group_by("intervariable_stratum")
        .agg(
            pl.len().alias("n"),
            (pl.col("label").sum() / pl.len() * 100).round(1).alias("label1_pct"),
            pl.col(ax_x).mean().round(3).alias(f"mean_{ax_x}"),
            pl.col(ax_y).mean().round(3).alias(f"mean_{ax_y}"),
        )
        .with_columns((pl.col("n") / total * 100).round(1).alias("stratum_pct"))
        .sort("intervariable_stratum")
    )
    print(f"\n=== Intervariable (n={total} cases) ===")
    print(dist)


# %%
inter_dist(inter_scores)

# ── Group comparison: do the metrics differ by outcome label? ────────────────

# %%
# run_grouped_analysis(analysis_mode="metrics") computes the case-level metrics
# per group and then tests every metric across groups: Mann-Whitney with Cliff's
# delta and a Hodges-Lehmann median difference for two groups, Kruskal-Wallis
# with DSCF post-hoc for more, with Benjamini-Hochberg FDR across the whole family.
imp_grouped = Imperfekt(
    imperfection="missingness",
    df=df_filtered,
    id_col="PcrKey",
    clock_col="clock",
    cols=["sbp", "hr", "o2sat", "rr"],
    save_path=Path("results/nemsis_group_comparison"),
    renderer=None,
    plot_library="matplotlib",
)
imp_grouped.run_grouped_analysis(
    annotation_col="label",
    save_results=True,
    analysis_mode="metrics",
)

# %%
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(50)

print("\n=== Group comparison: effect sizes, largest first ===")
print(
    imp_grouped.group_comparison_results.filter(pl.col("skipped_reason").is_null())
    .with_columns(pl.col("effect_size").abs().alias("abs_effect"))
    .sort("abs_effect", descending=True)
    .select(
        [
            "aspect",
            "variable",
            "metric",
            "effect_size",
            "ci_lower",
            "ci_upper",
            "q_value",
            "significant",
            "direction",
            "hodges_lehmann",
        ]
    )
)

print("\n=== Group descriptives (median [IQR] per group) ===")
print(
    imp_grouped.group_comparison_descriptives.select(
        [
            "aspect",
            "variable",
            "metric",
            "group",
            "n",
            "pct_defined",
            "median",
            "q25",
            "q75",
            "mean",
            "std",
        ]
    )
)

skipped = imp_grouped.group_comparison_results.filter(pl.col("skipped_reason").is_not_null())
if skipped.height:
    print("\n=== Metrics that could not be tested ===")
    print(skipped.select(["aspect", "variable", "metric", "skipped_reason"]))
