# %%
from pathlib import Path

import polars as pl

pl.Config.set_tbl_cols(80)
pl.Config.set_tbl_rows(50)

experiment = "20251212"
folder = Path(f"/workspaces/prehosp-vitals-gap/data/nemsis2024/results/imperfekt/{experiment}")
csv_files = list(folder.rglob("model_comparison_*.csv"))
print(f"Found {len(csv_files)} model comparison files")

dfs = []
for file in csv_files:
    df = pl.read_csv(file)
    # Add metadata from folder structure
    df = df.with_columns(
        [
            pl.lit(file.name).alias("source_file"),
            pl.lit(file.parent.name).alias("experiment_folder"),
        ]
    )
    dfs.append(df)

# Combine everything
# Check if all dataframes have the same columns
column_sets = [set(df.columns) for df in dfs]
if not all(cols == column_sets[0] for cols in column_sets):
    # find df that does not fit
    for i, cols in enumerate(column_sets):
        if cols != column_sets[0]:
            print(f"DataFrame at index {i} has different columns: {cols}")
all_results = pl.concat(dfs, how="vertical", rechunk=True)

print(all_results.shape)
print(all_results.head())

# %%
# Find the index of the maximal value in column "recall"
max_idx = all_results.select(pl.col("brier").arg_max()).item()

# Retrieve the entire row at that index
row = all_results.row(max_idx, named=True)
print(row)
# %%
# Remove rows with model = "random forest"
all_results = all_results.filter(pl.col("model") != "RandomForest")
# %% # Find top 10 models by various (threshold-free) metrics and save to CSV
metrics = [
    "auc_pr_ci_lower",
    "auc_pr_ci_mean",
    "roc_auc_ci_mean",
    "brier_ci_mean",
    "subset_positive_cases",
]
for metric in metrics:
    best = all_results.sort(metric, descending=True).head(15)
    print(best)
    best.write_csv(folder / f"top10_model_comparisons_{metric}.csv")
# %%
