# %%
from pathlib import Path

import polars as pl

from imperfekt import Imperfekt
from config import COHORT_PATH

pl.Config.set_tbl_cols(8)
pl.Config.set_tbl_rows(25)

df = pl.read_parquet(Path(COHORT_PATH))
# %%
df_filtered = df.with_columns(
    pl.col("clock").min().over("PcrKey").alias("_start_clock")
).with_columns(
    ((pl.col("clock") - pl.col("_start_clock")).dt.total_minutes()).alias("_minutes_from_start")
).filter(pl.col("_minutes_from_start") <= 30).drop(["_start_clock", "_minutes_from_start"])

valid_keys = (
    df_filtered.group_by("PcrKey")
    .agg(pl.col("clock").count().alias("_num_vitals"))
    .filter(pl.col("_num_vitals") >= 5)
    .select("PcrKey")
)

df_filtered = df_filtered.join(valid_keys, on="PcrKey", how="inner")

# label per PcrKey (constant within a case)
labels = df_filtered.select(["PcrKey", "label"]).unique("PcrKey")

print("=== Label prevalence per PcrKey ===")
print(df_filtered.select(["PcrKey", "label"]).unique("PcrKey").group_by("label").agg(pl.len().alias("n_cases")).with_columns(
    (pl.col("n_cases") / pl.col("n_cases").sum() * 100).round(1).alias("pct_cases")
).sort("label"))
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
imp.intravariable.composite_score(save_results=False)

intra_scores = imp.intravariable.results.iv_composite_scores
assert intra_scores is not None

# Selected axis pair per variable
print("\n=== Intravariable: selected axis pairs ===")
print(
    intra_scores.select(["variable", "axis_x", "axis_y", "axis_pair_corr",
                         "axis_x_median_threshold", "axis_y_median_threshold"])
    .unique(["variable", "axis_x", "axis_y"])
    .sort("variable")
)

# All pairwise rho values (including non-selected pairs)
print("\n=== Intravariable: all axis-pair correlations (rho) ===")
for var, corr_tbl in imp.intravariable.results.iv_pairwise_correlations.items():
    print(f"\n-- variable: {var} --")
    print(corr_tbl.sort("abs_corr"))

print("\n=== Null stratum diagnosis ===")
null_strata = intra_scores.filter(
    pl.col("indicated_pct").gt(0) & pl.col("imperfection_stratum").is_null()
)
print(f"Imperfect cases with null stratum: {null_strata.height}")
if null_strata.height > 0:
    print(null_strata.select([
        "PcrKey", "variable", "indicated_pct",
        "max_gap_fraction", "gap_missing_centroid",
        "gap_normalized_entropy", "gap_adherence_rate",
        "axis_x", "axis_y", "imperfection_stratum",
    ]).head(20))

intra_scores = intra_scores.join(labels, on="PcrKey", how="left")

def intra_dist(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("imperfection_stratum").is_not_null())

    dist = (
        df.group_by(["variable", "imperfection_stratum"])
        .agg(
            pl.len().alias("n"),
            (pl.col("label").sum() / pl.len() * 100).round(1).alias("label1_pct"),
            pl.col("indicated_pct").mean().round(2).alias("mean_indicated_pct"),
            pl.col("gap_missing_centroid").mean().round(3).alias("mean_centroid"),
            pl.col("max_gap_fraction").mean().round(3).alias("mean_max_gap_fraction"),
        )
        .with_columns(
            (pl.col("n") / pl.col("n").sum().over("variable") * 100)
            .round(1)
            .alias("stratum_pct")
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
imp.intervariable.composite_score(save_results=False)

inter_scores = imp.intervariable.results.iv_composite_scores
assert inter_scores is not None

# Selected axis pair (single pair for the whole cohort)
print("\n=== Intervariable: selected axis pair ===")
print(
    inter_scores.select(["axis_x", "axis_y", "axis_pair_corr",
                         "axis_x_median_threshold", "axis_y_median_threshold"])
    .unique()
)

# All pairwise rho values (including non-selected pairs)
print("\n=== Intervariable: all axis-pair correlations (rho) ===")
inter_pair_corrs = imp.intervariable.results.iv_pairwise_correlations
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
