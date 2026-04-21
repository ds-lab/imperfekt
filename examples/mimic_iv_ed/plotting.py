from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from examples.utils.models import XGBoostModel  # noqa: E402
from examples.mimic_iv_ed.config import (  # noqa: E402
    RESULTS_DIR,
    OUTCOME_COL,
    CV_N_SPLITS,
    CV_N_REPEATS,
    IREG_FEATURE_COLS,
    SPEARMAN_TOP_K_PHYS,
    SPEARMAN_TOP_K_STRUCT,
)


def _stratum_tick_labels(strata_keys: list[str], pipeline_summaries: list[tuple[str, dict[str, dict]]]) -> list[str]:
    """
    Build tick labels from mean ± CI prevalence aggregated across all pipelines'
    CV folds, e.g. "HH\n34%±2%". Uses n_pos_pct stored per fold in the summary.
    Falls back to the raw key if no prevalence data is available.
    """
    labels = []
    for k in strata_keys:
        prev_vals = []
        for _, summary in pipeline_summaries:
            v = summary.get(k, {}).get("n_pos_pct")
            if v:
                prev_vals.append(v["mean"])
        if prev_vals:
            mean_prev = float(np.mean(prev_vals))
            labels.append(f"{k}\n{mean_prev:.0%}")
        else:
            labels.append(k)
    return labels


def plot_auprc_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool = True,
) -> None:
    """
    Line plot: mean AUPRC per irregularity stratum for all provided pipelines.
    95% CI shown as dashed lines above and below the mean.
    Overall AUPRC shown as horizontal dashed reference lines.
    Tick labels show mean ± CI outcome prevalence derived from CV test folds.
    """
    strata_keys = sorted({
        k
        for _, summary in pipeline_summaries
        for k in summary
        if k != "overall"
    })

    def _vals(summary):
        means, lo, hi = [], [], []
        for k in strata_keys:
            v = summary.get(k, {}).get("auprc")
            if v:
                means.append(v["mean"])
                lo.append(v["mean"] - v["ci"])
                hi.append(v["mean"] + v["ci"])
            else:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
        return np.array(means), np.array(lo), np.array(hi)

    x = np.arange(len(strata_keys))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)

        overall = summary.get("overall", {}).get("auprc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(_stratum_tick_labels(strata_keys, pipeline_summaries))
    ax.set_xlabel("Irregularity stratum (mean outcome prevalence from CV test folds)")
    ax.set_ylabel("AUPRC (mean ± 95% CI)")
    ax.set_ylim(0, 1)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.set_title(f"AUPRC by irregularity stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout(rect=[0, 0, 0.78 if show_legend else 1, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_auprc_lift_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool = True,
) -> None:
    """
    Line plot: mean AUPRC lift (AUPRC / prevalence) per irregularity stratum.
    Lift > 1 means the model beats the no-skill baseline within that stratum.
    Reference line at lift = 1 (no-skill).
    Tick labels show mean ± CI outcome prevalence derived from CV test folds.
    """
    strata_keys = sorted({
        k
        for _, summary in pipeline_summaries
        for k in summary
        if k != "overall"
    })

    def _vals(summary):
        means, lo, hi = [], [], []
        for k in strata_keys:
            v = summary.get(k, {}).get("auprc_lift")
            if v:
                means.append(v["mean"])
                lo.append(v["mean"] - v["ci"])
                hi.append(v["mean"] + v["ci"])
            else:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
        return np.array(means), np.array(lo), np.array(hi)

    x = np.arange(len(strata_keys))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="No-skill baseline")

    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(_stratum_tick_labels(strata_keys, pipeline_summaries))
    ax.set_xlabel("Irregularity stratum (mean outcome prevalence from CV test folds)")
    ax.set_ylabel("AUPRC lift = AUPRC / prevalence (mean ± 95% CI)")
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.set_title(f"AUPRC lift by irregularity stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout(rect=[0, 0, 0.78 if show_legend else 1, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def _split_feature_groups(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Split aggregated feature names into physiological and structural groups.

    Structural features are the aggregated outputs derived from imperfekt
    interval/acceleration primitives. Physiological features are everything else.
    """
    struct_cols = [
        c
        for c in feature_cols
        if any(c == base or c.startswith(f"{base}_") for base in IREG_FEATURE_COLS)
    ]
    phys_cols = [c for c in feature_cols if c not in struct_cols]
    return phys_cols, struct_cols


def _compute_rfi_by_stratum(
    abs_shap: np.ndarray,
    stay_ids: np.ndarray,
    strata: pl.DataFrame,
    phys_idx: np.ndarray,
    struct_idx: np.ndarray,
) -> pl.DataFrame:
    """
    Relative Feature Importance (RFI) per stratum using summed absolute SHAP.

    For each stratum s:
      phys_sum  = sum over cases and physiology features of |SHAP|
      struct_sum = sum over cases and metadata features of |SHAP|
      phys_rfi  = phys_sum  / (phys_sum + struct_sum) * 100
      struct_rfi = struct_sum / (phys_sum + struct_sum) * 100

    Using the sum preserves the total attribution mass so that
    a metadata group with 5 features holding 30% of the sum is directly
    comparable to 50 physiology features — each metadata feature carries on
    average 10x the weight of a physiology feature.

    Also records per-feature-average importance for each group so callers can
    compute the per-feature multiplier (struct_per_feat / phys_per_feat).

    Returns DataFrame with columns:
      irregularity_stratum, phys_rfi, struct_rfi,
      phys_sum, struct_sum, n_phys, n_struct, per_feat_ratio
    """
    strata_map = {sid: s for sid, s in zip(
        strata["stay_id"].to_list(), strata["irregularity_stratum"].to_list()
    )}
    n_phys = len(phys_idx)
    n_struct = len(struct_idx)
    rows = []
    for stratum in sorted(set(strata["irregularity_stratum"].to_list())):
        mask = np.array([strata_map.get(sid) == stratum for sid in stay_ids])
        if mask.sum() == 0:
            continue
        phys_sum = float(abs_shap[mask][:, phys_idx].sum())
        struct_sum = float(abs_shap[mask][:, struct_idx].sum())
        total = phys_sum + struct_sum
        if total == 0:
            continue
        phys_rfi = phys_sum / total * 100.0
        struct_rfi = struct_sum / total * 100.0
        phys_per_feat = phys_sum / n_phys if n_phys > 0 else 0.0
        struct_per_feat = struct_sum / n_struct if n_struct > 0 else 0.0
        per_feat_ratio = struct_per_feat / phys_per_feat if phys_per_feat > 0 else float("nan")
        rows.append({
            "irregularity_stratum": stratum,
            "phys_rfi": phys_rfi,
            "struct_rfi": struct_rfi,
            "phys_sum": phys_sum,
            "struct_sum": struct_sum,
            "n_phys": n_phys,
            "n_struct": n_struct,
            "per_feat_ratio": per_feat_ratio,
        })
    return pl.DataFrame(rows)


def _plot_group_importance_by_stratum(
    abs_shap: np.ndarray,
    stay_ids: np.ndarray,
    strata: pl.DataFrame,
    phys_idx: np.ndarray,
    struct_idx: np.ndarray,
    save_path: Path,
    pipeline_name: str,
) -> None:
    rfi_df = _compute_rfi_by_stratum(abs_shap, stay_ids, strata, phys_idx, struct_idx)
    if rfi_df.height == 0:
        print(f"[{pipeline_name}] Skipping SHAP group plot: no overlapping stay_id rows.")
        return

    tick_labels = rfi_df["irregularity_stratum"].to_list()
    phys_pct = rfi_df["phys_rfi"].to_numpy()
    struct_pct = rfi_df["struct_rfi"].to_numpy()
    per_feat_ratio = rfi_df["per_feat_ratio"].to_numpy()
    n_phys = int(rfi_df["n_phys"][0])
    n_struct = int(rfi_df["n_struct"][0])

    x = np.arange(len(tick_labels))
    fig, (ax_stack, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_stack.bar(x, phys_pct, color="#4C72B0", label=f"Physiological ({n_phys} features)")
    ax_stack.bar(x, struct_pct, bottom=phys_pct, color="#DD8452", label=f"imperfekt metadata ({n_struct} features)")
    for i, (p, s) in enumerate(zip(phys_pct, struct_pct)):
        ax_stack.text(i, p + s / 2, f"{s:.1f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax_stack.set_ylim(0, 100)
    ax_stack.set_yticks(np.arange(0, 101, 20))
    ax_stack.set_ylabel("Share of total |SHAP| mass (%)")
    ax_stack.set_xlabel("Irregularity stratum")
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(tick_labels)
    ax_stack.set_title("Attention shift (summed |SHAP| by group)")
    ax_stack.legend(loc="upper right", fontsize=8)

    bar_colors = ["#c84b31" if r > 1 else "#4C72B0" for r in per_feat_ratio]
    ax_ratio.bar(x, per_feat_ratio, color=bar_colors)
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Equal influence (×1)")
    for i, r in enumerate(per_feat_ratio):
        if not np.isnan(r):
            ax_ratio.text(i, r + 0.05, f"×{r:.3f}", ha="center", va="bottom", fontsize=8)
    ax_ratio.set_ylabel("Per-feature influence multiplier\n(avg metadata SHAP / avg physiology SHAP)")
    ax_ratio.set_xlabel("Irregularity stratum")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(tick_labels)
    ax_ratio.set_title("Per-feature influence ratio (metadata vs physiology)")
    ax_ratio.legend(fontsize=8)

    fig.suptitle(f"{pipeline_name}: Structural floor analysis", fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"[{pipeline_name}] SHAP structural-floor plot saved to {save_path}")


def _spearman_structural_floor(
    abs_shap: np.ndarray,
    raw_features: np.ndarray,
    feature_cols: list[str],
    phys_cols: list[str],
    struct_cols: list[str],
    pipeline_name: str,
) -> pl.DataFrame | None:
    """
    Pairwise Spearman rho between top metadata and physiology raw features.

    Feature ranking is defined by mean |SHAP| over samples.
    We select the top-k features from each group by this ranking and compute
    pairwise Spearman correlations on aligned raw feature values only.

    Returns:
        A Polars DataFrame sorted by absolute rho (descending), or None if
        one of the groups is empty.
    """
    from scipy.stats import spearmanr

    if len(phys_cols) == 0 or len(struct_cols) == 0:
        print(
            f"[{pipeline_name}] Skipping Spearman structural-floor test: "
            f"phys={len(phys_cols)}, struct={len(struct_cols)}."
        )
        return None

    if raw_features.shape[0] != abs_shap.shape[0]:
        raise ValueError(
            f"[{pipeline_name}] Spearman alignment error: raw_features rows "
            f"({raw_features.shape[0]}) != abs_shap rows ({abs_shap.shape[0]})."
        )

    mean_abs = abs_shap.mean(axis=0)
    mean_raw = np.nanmean(np.where(np.isfinite(raw_features), raw_features, np.nan), axis=0)

    phys_idx_map = {c: feature_cols.index(c) for c in phys_cols}
    struct_idx_map = {c: feature_cols.index(c) for c in struct_cols}

    top_k_phys = min(SPEARMAN_TOP_K_PHYS, len(phys_idx_map))
    top_k_struct = min(SPEARMAN_TOP_K_STRUCT, len(struct_idx_map))

    top_phys = sorted(phys_idx_map, key=lambda c: mean_abs[phys_idx_map[c]], reverse=True)[:top_k_phys]
    top_struct = sorted(struct_idx_map, key=lambda c: mean_abs[struct_idx_map[c]], reverse=True)[:top_k_struct]

    rows: list[dict] = []
    for struct_feat in top_struct:
        struct_vals = raw_features[:, struct_idx_map[struct_feat]]
        for phys_feat in top_phys:
            phys_vals = raw_features[:, phys_idx_map[phys_feat]]

            valid_mask = np.isfinite(phys_vals) & np.isfinite(struct_vals)
            valid_n = int(valid_mask.sum())
            if valid_n < 3:
                rho, pval = float("nan"), float("nan")
            else:
                phys_valid = phys_vals[valid_mask]
                struct_valid = struct_vals[valid_mask]
                if np.nanstd(phys_valid) == 0 or np.nanstd(struct_valid) == 0:
                    rho, pval = float("nan"), float("nan")
                else:
                    rho, pval = spearmanr(phys_valid, struct_valid)

            if np.isnan(rho) or np.isnan(pval):
                significance = "undefined"
            elif pval < 0.001:
                significance = "*** (p<0.001)"
            elif pval < 0.01:
                significance = "** (p<0.01)"
            elif pval < 0.05:
                significance = "* (p<0.05)"
            else:
                significance = "ns"

            rows.append(
                {
                    "metadata_feature": struct_feat,
                    "physiology_feature": phys_feat,
                    "metadata_mean_abs_shap": float(mean_abs[struct_idx_map[struct_feat]]),
                    "physiology_mean_abs_shap": float(mean_abs[phys_idx_map[phys_feat]]),
                    "metadata_mean_raw": float(mean_raw[struct_idx_map[struct_feat]]),
                    "physiology_mean_raw": float(mean_raw[phys_idx_map[phys_feat]]),
                    "rho": float(rho),
                    "abs_rho": float(abs(rho)) if not np.isnan(rho) else float("nan"),
                    "p_value": float(pval),
                    "significance": significance,
                    "valid_n": valid_n,
                }
            )

    if len(rows) == 0:
        return None

    spearman_df = pl.DataFrame(rows).sort(["abs_rho", "p_value"], descending=[True, False])

    abs_rhos = np.array(spearman_df["abs_rho"].to_list(), dtype=float)
    pvals = np.array(spearman_df["p_value"].to_list(), dtype=float)
    mean_abs_rho = float(np.nanmean(abs_rhos)) if len(abs_rhos) > 0 else float("nan")
    sig_pairs = int(np.sum((~np.isnan(pvals)) & (pvals < 0.05)))

    top_phys_desc = ", ".join(
        f"{feat} ({mean_abs[phys_idx_map[feat]]:.4f})" for feat in top_phys
    )
    top_struct_desc = ", ".join(
        f"{feat} ({mean_abs[struct_idx_map[feat]]:.4f})" for feat in top_struct
    )

    print(
        f"[{pipeline_name}] Spearman pairwise structural-floor test: "
        f"{top_k_struct} metadata × {top_k_phys} physiology = {spearman_df.height} pairs\n"
        f"  top physiology features (mean |SHAP| for ordering): {top_phys_desc}\n"
        f"  top metadata features (mean |SHAP| for ordering): {top_struct_desc}\n"
        f"  mean |rho| across raw-feature pairs = {mean_abs_rho:.3f}; "
        f"significant pairs (p<0.05) = {sig_pairs}/{spearman_df.height}"
    )
    print(spearman_df)

    return spearman_df


def _plot_spearman_heatmap(
    spearman_df: pl.DataFrame,
    pipeline_name: str,
    save_path: Path,
) -> None:
    """Plot rho heatmap for metadata vs physiology feature pairs."""
    if spearman_df.height == 0:
        print(f"[{pipeline_name}] Skipping Spearman heatmap: empty pair table.")
        return

    meta_order = (
        spearman_df
        .select(["metadata_feature", "metadata_mean_abs_shap"])
        .unique(subset=["metadata_feature"], keep="first")
        .sort("metadata_mean_abs_shap", descending=True)["metadata_feature"]
        .to_list()
    )
    phys_order = (
        spearman_df
        .select(["physiology_feature", "physiology_mean_abs_shap"])
        .unique(subset=["physiology_feature"], keep="first")
        .sort("physiology_mean_abs_shap", descending=True)["physiology_feature"]
        .to_list()
    )

    rho_lookup = {
        (row["metadata_feature"], row["physiology_feature"]): row["rho"]
        for row in spearman_df.iter_rows(named=True)
    }
    sig_lookup = {
        (row["metadata_feature"], row["physiology_feature"]): row["significance"]
        for row in spearman_df.iter_rows(named=True)
    }

    rho_mat = np.full((len(meta_order), len(phys_order)), np.nan, dtype=float)
    for i, meta in enumerate(meta_order):
        for j, phys in enumerate(phys_order):
            rho_mat[i, j] = rho_lookup.get((meta, phys), float("nan"))

    fig_w = max(8.0, 1.2 * len(phys_order) + 3.0)
    fig_h = max(5.0, 0.8 * len(meta_order) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(rho_mat, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")

    ax.set_xticks(np.arange(len(phys_order)))
    ax.set_xticklabels(phys_order, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(meta_order)))
    ax.set_yticklabels(meta_order)
    ax.set_xlabel("Physiology features (SHAP-ranked)")
    ax.set_ylabel("Metadata features (SHAP-ranked)")
    ax.set_title(f"{pipeline_name}: Spearman heatmap (raw-feature pairs)")

    for i, meta in enumerate(meta_order):
        for j, phys in enumerate(phys_order):
            rho = rho_mat[i, j]
            sig = sig_lookup.get((meta, phys), "")
            if np.isnan(rho):
                text = "NA"
            else:
                star = "" if sig in ("", "ns", "undefined") else "*"
                text = f"{rho:.2f}{star}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"[{pipeline_name}] Spearman heatmap saved to {save_path}")


def run_shap_group_analysis(
    model: XGBoostModel,
    X_test: np.ndarray,
    test_stay_df: pl.DataFrame,
    strata: pl.DataFrame,
    feature_cols: list[str],
    pipeline_name: str,
) -> None:
    if X_test.shape[0] == 0 or test_stay_df.height == 0:
        print(f"[{pipeline_name}] Skipping SHAP: empty final-fold test data.")
        return

    n_rows = min(X_test.shape[0], test_stay_df.height)
    if X_test.shape[0] != test_stay_df.height:
        print(
            f"[{pipeline_name}] SHAP alignment warning: X_test has {X_test.shape[0]} rows, "
            f"test metadata has {test_stay_df.height} rows. Using first {n_rows} rows."
        )

    X_aligned = X_test[:n_rows]
    test_aligned = test_stay_df.head(n_rows).select(["stay_id"])

    explanation = model.compute_shap(
        X=X_aligned,
        feature_names=feature_cols,
        save_dir=None,
        prefix=f"{pipeline_name}_last_fold",
    )
    if explanation is None:
        print(f"[{pipeline_name}] Skipping SHAP: explanation backend unavailable.")
        return

    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    n_aligned = min(shap_values.shape[0], test_aligned.height)
    if shap_values.shape[0] != test_aligned.height:
        print(
            f"[{pipeline_name}] SHAP value alignment warning: explanation has {shap_values.shape[0]} rows, "
            f"metadata has {test_aligned.height} rows. Using first {n_aligned} rows."
        )

    abs_shap = np.abs(shap_values[:n_aligned])
    stay_ids = test_aligned["stay_id"].to_numpy()[:n_aligned]

    phys_cols, struct_cols = _split_feature_groups(feature_cols)
    if len(phys_cols) == 0 or len(struct_cols) == 0:
        print(
            f"[{pipeline_name}] Skipping SHAP group analysis: "
            f"phys={len(phys_cols)}, struct={len(struct_cols)} feature groups."
        )
        return

    phys_idx = np.array([feature_cols.index(c) for c in phys_cols], dtype=int)
    struct_idx = np.array([feature_cols.index(c) for c in struct_cols], dtype=int)

    _plot_group_importance_by_stratum(
        abs_shap=abs_shap,
        stay_ids=stay_ids,
        strata=strata,
        phys_idx=phys_idx,
        struct_idx=struct_idx,
        save_path=RESULTS_DIR / "figures" / f"shap_group_importance_{pipeline_name}.svg",
        pipeline_name=pipeline_name,
    )

    spearman_df = _spearman_structural_floor(
        abs_shap=abs_shap,
        raw_features=X_aligned[:n_aligned],
        feature_cols=feature_cols,
        phys_cols=phys_cols,
        struct_cols=struct_cols,
        pipeline_name=pipeline_name,
    )

    if spearman_df is not None and spearman_df.height > 0:
        _plot_spearman_heatmap(
            spearman_df=spearman_df,
            pipeline_name=pipeline_name,
            save_path=RESULTS_DIR / "figures" / f"spearman_heatmap_{pipeline_name}.svg",
        )

        spearman_path = RESULTS_DIR / "figures" / f"spearman_pairwise_{pipeline_name}.csv"
        spearman_path.parent.mkdir(parents=True, exist_ok=True)
        spearman_df.write_csv(spearman_path)
        print(f"[{pipeline_name}] Spearman pairwise table saved to {spearman_path}")
