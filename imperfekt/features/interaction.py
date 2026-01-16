# In imperfekt/features/interaction.py
from itertools import permutations

import polars as pl


def add_pairwise_interactions(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
) -> pl.DataFrame:
    """
    Adds pairwise interaction features between variables and their masks.

    Generates four types of interactions for each pair of variables (A, B):
    1. Concurrent value: var_a_t * mask_b_t
    2. Concurrent mask: mask_a_t * mask_b_t
    3. Predictive value: var_a_t-1 * mask_b_t
    4. Predictive mask: mask_a_t-1 * mask_b_t
    Results in 4*N*(N-1) new features.

    Args:
        df: The main DataFrame with original values.
        mask: DataFrame with id, clock, and mask columns.
        variable_cols: List of original variable names to create interactions for.
        id_col: The column identifying individual time series.
        clock_col: The time column for joining.

    Returns:
        The DataFrame with added interaction features.
    """
    # Ensure we have the necessary columns to work with
    interaction_df = df.join(mask, on=[id_col, clock_col], how="left")

    interaction_exprs = []
    new_cols_names = []

    # Iterate through all ordered pairs of variables
    for var_a, var_b in permutations(variable_cols, 2):
        mask_a = f"{var_a}_mask"
        mask_b = f"{var_b}_mask"

        # 1. Concurrent var_a_t * mask_b_t
        col_name_1 = f"inter_{var_a}_t_x_{mask_b}"
        interaction_exprs.append((pl.col(var_a) * pl.col(mask_b)).alias(col_name_1))
        new_cols_names.append(col_name_1)

        # 2. Concurrent mask_a_t * mask_b_t
        col_name_2 = f"inter_{mask_a}_t_x_{mask_b}"
        interaction_exprs.append((pl.col(mask_a) * pl.col(mask_b)).alias(col_name_2))
        new_cols_names.append(col_name_2)

        # 3. Predictive var_a_t-1 * mask_b_t
        col_name_3 = f"inter_{var_a}_t-1_x_{mask_b}"
        interaction_exprs.append(
            (pl.col(var_a).shift(1).over(id_col) * pl.col(mask_b)).alias(col_name_3)
        )
        new_cols_names.append(col_name_3)

        # 4. Predictive mask_a_t-1 * mask_b_t
        col_name_4 = f"inter_{mask_a}_t-1_x_{mask_b}"
        interaction_exprs.append(
            (pl.col(mask_a).shift(1).over(id_col) * pl.col(mask_b)).alias(col_name_4)
        )
        new_cols_names.append(col_name_4)

    # Calculate all interaction features
    interaction_df = interaction_df.with_columns(interaction_exprs)

    # Select only the new columns and identifiers to join back
    final_interaction_cols = interaction_df.select(id_col, clock_col, *new_cols_names)

    # Join back to the original df
    df = df.drop(new_cols_names, strict=False)
    df = df.join(final_interaction_cols, on=[id_col, clock_col], how="left")

    return df


def add_row_level_features(
    df: pl.DataFrame,
    mask: pl.DataFrame,
    variable_cols: list,
    id_col: str,
    clock_col: str,
) -> pl.DataFrame:
    """
    Adds row-level features, like the percentage of missing values.

    Args:
        df: The main DataFrame.
        mask: DataFrame with id, clock, and mask columns.
        variable_cols: List of original variable names.
        id_col: The column identifying individual time series.
        clock_col: The time column for joining.

    Returns:
        The DataFrame with added row-level features.
    """
    mask_cols = [f"{col}_mask" for col in variable_cols]

    # Calculate percentage of imperfections
    percentage_col_name = "percentage_imperfect"
    row_level_df = mask.with_columns(
        (pl.sum_horizontal(mask_cols) / len(mask_cols)).alias(percentage_col_name)
    ).select(id_col, clock_col, percentage_col_name)

    # Join back to the main df
    df = df.drop(percentage_col_name, strict=False)
    df = df.join(row_level_df, on=[id_col, clock_col], how="left")

    return df
