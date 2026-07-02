# %%
import polars as pl

from imperfekt import Imperfekt
from config import VITAL_COLS, load_cohort, RESULTS_DIR

pl.Config.set_tbl_cols(8)
pl.Config.set_tbl_rows(25)

df = load_cohort()
print(f"Cohort: {df['id'].n_unique()} stays, {len(df)} observations")

# label per id (constant within a case)
labels = df.select(["id", "label"]).unique("id")

print("=== Label prevalence per id ===")
print(df.select(["id", "label"]).unique("id").group_by("label").agg(pl.len().alias("n_cases")).with_columns(
    (pl.col("n_cases") / pl.col("n_cases").sum() * 100).round(1).alias("pct_cases")
).sort("label"))

# %%
imp = Imperfekt(
    imperfection="missingness",
    df=df,
    id_col="id",
    clock_col="clock",
    cols=VITAL_COLS,
    save_path=RESULTS_DIR / "missingness_experiments",
    renderer=None,
    plot_library="matplotlib",
)

# ── Intravariable ────────────────────────────────────────────────────────────

# %%
imp.intravariable.composite_score(save_results=True)

intra_scores = imp.intravariable.results.iv_composite_scores
assert intra_scores is not None

# Selected axis pair (shared across all cases)
print("\n=== Intravariable: selected axis pair ===")
print(
    intra_scores.select(["axis_x", "axis_y", "axis_pair_corr",
                         "axis_x_median_threshold", "axis_y_median_threshold"])
    .unique(["axis_x", "axis_y"])
)

# All pairwise rho values for the candidate axes
print("\n=== Intravariable: all axis-pair correlations (rho) ===")
print(imp.intravariable.results.iv_pooled_corr_table)

indicated_cols = [c for c in intra_scores.columns if c.endswith("_indicated_pct")]

print("\n=== Null stratum diagnosis ===")
any_imperfect = pl.fold(
    acc=pl.lit(False),
    function=lambda acc, s: acc | (s > 0),
    exprs=[pl.col(c) for c in indicated_cols],
)
null_strata = intra_scores.filter(
    any_imperfect & pl.col("imperfection_stratum").is_null()
)
print(f"Imperfect cases with null stratum: {null_strata.height}")
if null_strata.height > 0:
    print(null_strata.select(
        ["id"] + indicated_cols + ["axis_x", "axis_y", "imperfection_stratum"]
    ).head(20))

intra_scores = intra_scores.join(labels, on="id", how="left")

def intra_dist(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("imperfection_stratum").is_not_null())
    indicated_cols = [c for c in df.columns if c.endswith("_indicated_pct")]

    agg_exprs = [
        pl.len().alias("n"),
        (pl.col("label").sum() / pl.len() * 100).round(1).alias("label1_pct"),
    ] + [pl.col(c).mean().round(2).alias(f"mean_{c}") for c in indicated_cols]

    dist = (
        df.group_by("imperfection_stratum")
        .agg(agg_exprs)
        .with_columns(
            (pl.col("n") / pl.col("n").sum() * 100).round(1).alias("stratum_pct")
        )
        .sort("imperfection_stratum")
    )
    total = dist["n"].sum()
    print(f"\n=== Intravariable (n={total}) ===")
    print(dist)

# %%
pl.Config.set_tbl_cols(100)
intra_dist(intra_scores)

# ── Intervariable ────────────────────────────────────────────────────────────

# %%
imp.intervariable.composite_score(save_results=True)

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

inter_scores = inter_scores.join(labels, on="id", how="left")

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
