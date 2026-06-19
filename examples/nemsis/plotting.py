from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CV_N_REPEATS,
    CV_N_SPLITS,
    PLOT_COLORS,
    SHOW_LEGEND,
)
from examples.nemsis.features import feature_group  # noqa: E402

_STRATUM_ORDER = ["Q_complete", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]

_PLAUS_LABEL    = {"pk": "plaus=keep", "pr": "plaus=remove"}
_IMP_LABEL      = {"in": "no imputation", "il": "LOCF", "is": "SAITS"}
_IMP_LINESTYLE  = {"in": "-", "il": ":", "is": "-."}


def _imp_linestyle(label: str) -> str:
    """Return linestyle for the imputation encoded in a pipeline label string.

    For GRU mask-ablation runs ('gru/<config>/<arm>') the linestyle instead
    distinguishes the mask arm (solid = mask, dashed = no mask) so the two arms
    of one config are visually paired.
    """
    name = label.removeprefix("Setup ")
    if name.startswith("gru/"):
        return "-" if name.rsplit("/", 1)[-1] != "nomask" else "--"
    parts = name.split("/")[0].split("_")  # e.g. ["ma", "pk", "in"]
    if len(parts) == 3:
        return _IMP_LINESTYLE.get(parts[2], "-")
    return "-"


def _decode_pipeline_label(label: str) -> str:
    """
    Convert internal pipeline names to human-readable legend labels.

    'Setup ma_pk_in/base+miss' → 'plaus=keep · no imputation\nbase+miss'
    Falls back to the raw label if the pattern is not recognised.
    """
    name = label.removeprefix("Setup ")
    # GRU mask-ablation: 'gru/<config>/<arm>' → '<config decoded> · GRU (mask|no mask)'.
    if name.startswith("gru/"):
        rest = name[len("gru/"):]
        config_key, arm = rest.rsplit("/", 1) if "/" in rest else (rest, "")
        parts = config_key.split("_")
        if len(parts) == 3:
            _, plaus_code, imp_code = parts
            plaus = _PLAUS_LABEL.get(plaus_code, plaus_code)
            imp = _IMP_LABEL.get(imp_code, imp_code)
            head = f"{plaus} · {imp}"
        else:
            head = config_key
        arm_label = {"mask": "GRU (mask)", "nomask": "GRU (no mask)"}.get(arm, f"GRU {arm}".strip())
        return f"{head}\n{arm_label}"
    if "/" in name:
        config_key, feature_set = name.split("/", 1)
    else:
        return label
    parts = config_key.split("_")  # e.g. ["ma", "pk", "in"]
    if len(parts) == 3:
        _, plaus_code, imp_code = parts
        plaus = _PLAUS_LABEL.get(plaus_code, plaus_code)
        imp   = _IMP_LABEL.get(imp_code, imp_code)
        return f"{plaus} · {imp}\n{feature_set}"
    return label


def _sort_strata(keys: set[str]) -> list[str]:
    known = [k for k in _STRATUM_ORDER if k in keys]
    other = sorted(keys - set(_STRATUM_ORDER))
    return known + other



def _data_ylim(all_vals: list[np.ndarray], pad: float = 0.1) -> tuple[float, float]:
    """Auto y-limits: data range expanded by pad fraction, rounded to 2 decimals."""
    finite = np.concatenate([v[np.isfinite(v)] for v in all_vals])
    if finite.size == 0:
        return 0.0, 1.0
    lo = finite.min()
    hi = finite.max()
    margin = max((hi - lo) * pad, 1e-4)
    return round(max(lo - margin, 0.0), 4), round(hi + margin, 4)


def plot_delta_auprc_heatmap(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    *,
    baseline: str = "Setup ma_pk_in/base",
    swaps: list[tuple[str, str]] | None = None,
    metric: str = "auprc",
) -> None:
    """
    Heatmap of Δmetric (variant − baseline) for single-design-choice swaps.

    Rows = swap variants, columns = quadrants (facet). Diverging colormap
    centered at 0 so the "helps here / hurts there" pattern is legible.
    ``metric`` is "auprc" or "auprc_lift". Skips (with a warning) any swap whose
    pipeline is absent from ``pipeline_summaries``, and skips the whole figure if
    the baseline is missing.
    """
    summ = dict(pipeline_summaries)
    if baseline not in summ:
        print(f"WARNING: baseline {baseline!r} not in CSV — skipping {save_path.name}")
        return

    if swaps is None:
        swaps = [
            ("F → base+miss", "Setup ma_pk_in/base+miss"),
            ("F → base+plaus", "Setup ma_pk_in/base+plaus"),
            ("imp → LOCF", "Setup ma_pk_il/base"),
            ("imp → SAITS", "Setup ma_pk_is/base"),
            ("plaus → remove", "Setup ma_pr_in/base"),
            ("plaus/F → remove & base+plaus", "Setup ma_pr_in/base+plaus"),
        ]

    present_swaps = []
    for row_label, pipe in swaps:
        if pipe in summ:
            present_swaps.append((row_label, pipe))
        else:
            print(f"WARNING: skipping swap {row_label!r} — pipeline {pipe!r} not in CSV")
    if not present_swaps:
        print(f"WARNING: no swap pipelines present — skipping {save_path.name}")
        return

    quad_keys = _sort_strata(
        {k for k, v in summ[baseline].items()
         if k != "overall" and k != "Q_complete" and not k.startswith("_") and isinstance(v, dict)}
    )

    def _mean(pipe: str, q: str) -> float:
        cell = summ.get(pipe, {}).get(q, {}).get(metric)
        return cell["mean"] if cell else float("nan")

    delta = np.full((len(present_swaps), len(quad_keys)), np.nan)
    for i, (_, pipe) in enumerate(present_swaps):
        for j, q in enumerate(quad_keys):
            base_v = _mean(baseline, q)
            var_v = _mean(pipe, q)
            if np.isfinite(base_v) and np.isfinite(var_v):
                delta[i, j] = var_v - base_v

    is_lift = metric == "auprc_lift"
    metric_label = "AUPRC lift" if is_lift else "AUPRC"
    fmt = "{:+.2f}" if is_lift else "{:+.4f}"

    fig_w = max(7.0, 1.3 * len(quad_keys) + 3.0)
    fig_h = max(2.5, 0.7 * len(present_swaps) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f2f2f2")
    vmax = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1.0
    vmax = vmax or 1.0
    im = ax.imshow(delta, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"Δ {metric_label} vs baseline")

    ax.set_xticks(np.arange(len(quad_keys)))
    ax.set_xticklabels(quad_keys, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(present_swaps)))
    ax.set_yticklabels([r for r, _ in present_swaps], fontsize=8)
    ax.set_xlabel("Quadrant")
    base_label = _decode_pipeline_label(baseline).replace("\n", " · ")
    ax.set_title(
        f"Δ {metric_label} vs baseline ({base_label})\n({CV_N_SPLITS}×{CV_N_REPEATS} CV)",
        fontsize=9,
    )

    for i in range(len(present_swaps)):
        for j in range(len(quad_keys)):
            v = delta[i, j]
            if np.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="#888888")
                continue
            color = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_auprc_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool | None = None,
    colors: list[str] | None = None,
) -> None:
    """
    Line plot: mean AUPRC per stratum for all provided pipelines.
    95% CI shown as dashed lines above and below the mean.
    Overall AUPRC shown as horizontal dashed reference lines.
    Y-axis auto-zoomed to the data range (not fixed 0–1).
    """
    if show_legend is None:
        show_legend = SHOW_LEGEND
    if colors is None:
        colors = PLOT_COLORS
    strata_keys = _sort_strata(
        {k for _, summary in pipeline_summaries for k in summary if k != "overall" and not k.startswith("_")}
    )

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

    all_lo, all_hi = [], []
    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        ls = _imp_linestyle(label)
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, linestyle=ls, label=_decode_pipeline_label(label))
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)
        all_lo.append(lo)
        all_hi.append(hi)

        overall = summary.get("overall", {}).get("auprc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=ls, linewidth=0.8, alpha=0.5)

    ax.set_ylim(*_data_ylim(all_lo + all_hi))
    ax.set_xticks(x)
    ax.set_xticklabels(strata_keys)
    ax.set_xlabel("Stratum")
    ax.set_ylabel("AUPRC (mean ± 95% CI)")
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=7)
    ax.set_title(f"AUPRC by stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_auroc_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool | None = None,
    colors: list[str] | None = None,
) -> None:
    """
    Line plot: mean AUROC per stratum for all provided pipelines.
    95% CI shown as dashed lines above and below the mean.
    Overall AUROC shown as horizontal dashed reference lines.
    Y-axis auto-zoomed to the data range (not fixed 0.5–1).
    """
    if show_legend is None:
        show_legend = SHOW_LEGEND
    if colors is None:
        colors = PLOT_COLORS
    strata_keys = _sort_strata(
        {k for _, summary in pipeline_summaries for k in summary if k != "overall" and not k.startswith("_")}
    )

    def _vals(summary):
        means, lo, hi = [], [], []
        for k in strata_keys:
            v = summary.get(k, {}).get("auroc")
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

    all_lo, all_hi = [], []
    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        ls = _imp_linestyle(label)
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, linestyle=ls, label=_decode_pipeline_label(label))
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)
        all_lo.append(lo)
        all_hi.append(hi)

        overall = summary.get("overall", {}).get("auroc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=ls, linewidth=0.8, alpha=0.5)

    y_lo, y_hi = _data_ylim(all_lo + all_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x)
    ax.set_xticklabels(strata_keys)
    ax.set_xlabel("Stratum")
    ax.set_ylabel("AUROC (mean ± 95% CI)")
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=7)
    ax.set_title(f"AUROC by stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_auprc_lift_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool | None = None,
    colors: list[str] | None = None,
) -> None:
    """
    Line plot: mean AUPRC lift (AUPRC / prevalence) per stratum.
    Lift > 1 means the model beats the no-skill baseline within that stratum.
    Reference line at lift = 1 (no-skill).
    """
    if show_legend is None:
        show_legend = SHOW_LEGEND
    if colors is None:
        colors = PLOT_COLORS
    strata_keys = _sort_strata(
        {k for _, summary in pipeline_summaries for k in summary if k != "overall" and not k.startswith("_")}
    )

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

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="No-skill baseline")

    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        ls = _imp_linestyle(label)
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, linestyle=ls, label=_decode_pipeline_label(label))
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(strata_keys)
    ax.set_xlabel("Stratum")
    ax.set_ylabel("AUPRC lift = AUPRC / prevalence (mean ± 95% CI)")
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=7)
    ax.set_title(f"AUPRC lift by stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# ── SHAP colour palette ───────────────────────────────────────────────────────
_SHAP_GROUP_COLORS = {
    "structural_miss":  "#D55E00",   # vermilion — key signal
    "structural_plaus": "#E69F00",   # orange
    "physiology":       "#999999",   # grey
    "metadata":         "#BBBBBB",   # light grey
    "mask":             "#0072B2",   # blue — GRU mask-channel importance
}


def plot_shap_importance_bar(
    shap_df: pl.DataFrame,
    save_path: Path,
    stratum: str = "overall",
    top_n: int = 30,
) -> None:
    """Horizontal bar chart: mean ± std of mean |SHAP| per feature, one subplot per pipeline.

    Bars are colored by feature_group. Structural miss/plaus features are highlighted.
    """
    df = shap_df.filter(pl.col("stratum") == stratum)
    pipelines = df["pipeline"].unique(maintain_order=True).to_list()
    if not pipelines:
        print(f"plot_shap_importance_bar: no data for stratum={stratum!r}")
        return

    fig, axes = plt.subplots(
        len(pipelines), 1,
        figsize=(10, max(4, 0.2 * top_n) * len(pipelines)),
        squeeze=False,
    )

    for ax, pipeline in zip(axes[:, 0], pipelines):
        sub = (
            df.filter(pl.col("pipeline") == pipeline)
            .sort("mean_abs_shap_mean", descending=True)
            .head(top_n)
        )
        features = sub["feature"].to_list()[::-1]
        means = sub["mean_abs_shap_mean"].to_list()[::-1]
        stds = sub["mean_abs_shap_std"].to_list()[::-1]
        groups = sub["feature_group"].to_list()[::-1]
        colors = [_SHAP_GROUP_COLORS.get(g, "#999999") for g in groups]

        y = np.arange(len(features))
        ax.barh(y, means, height=0.4, xerr=stds, color=colors, alpha=0.85, error_kw={"linewidth": 0.8})
        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel("Mean |SHAP| (mean ± std across folds)")
        ax.set_title(f"{pipeline}  [{stratum}]", fontsize=9)

        # legend patches
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=c, label=g) for g, c in _SHAP_GROUP_COLORS.items()
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_shap_stability_scatter(
    shap_df: pl.DataFrame,
    save_path: Path,
    stratum: str = "overall",
    top_n_label: int = 10,
) -> None:
    """Scatter: x = mean |SHAP|, y = std |SHAP| across folds, one subplot per pipeline.

    All features plotted; structural features and top-N by importance are annotated.
    Low std + high mean = stable & important (bottom-right).
    """
    df = shap_df.filter(pl.col("stratum") == stratum)
    pipelines = df["pipeline"].unique(maintain_order=True).to_list()
    if not pipelines:
        print(f"plot_shap_stability_scatter: no data for stratum={stratum!r}")
        return

    n_cols = min(2, len(pipelines))
    n_rows = (len(pipelines) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    for idx, pipeline in enumerate(pipelines):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = df.filter(pl.col("pipeline") == pipeline)
        means = sub["mean_abs_shap_mean"].to_numpy()
        stds = sub["mean_abs_shap_std"].to_numpy()
        groups = sub["feature_group"].to_list()
        features = sub["feature"].to_list()
        colors = [_SHAP_GROUP_COLORS.get(g, "#999999") for g in groups]

        ax.scatter(means, stds, c=colors, alpha=0.7, s=20, linewidths=0)

        # annotate top-N by importance and all structural features
        top_idx = set(np.argsort(means)[::-1][:top_n_label])
        structural_idx = {i for i, g in enumerate(groups) if g.startswith("structural_")}
        for i in top_idx | structural_idx:
            ax.annotate(
                features[i],
                (means[i], stds[i]),
                fontsize=5,
                textcoords="offset points",
                xytext=(3, 2),
            )

        ax.set_xlabel("Mean |SHAP| (mean across folds)")
        ax.set_ylabel("Std of mean |SHAP| across folds")
        ax.set_title(f"{pipeline}  [{stratum}]", fontsize=9)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=c, label=g) for g, c in _SHAP_GROUP_COLORS.items()
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=6)

    # hide unused subplots
    for idx in range(len(pipelines), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_spearman_orthogonality(
    shap_npz_path: Path,
    features_npz_path: Path,
    pipeline_name: str,
    figures_dir: Path,
    top_k_struct: int = 10,
    top_k_phys: int = 10,
) -> None:
    """Pairwise Spearman rho between top structural and physiology features.

    Feature selection: top-k by mean |SHAP| from shap_npz_path (shap_overall,
    n_folds × n_features).
    Correlation: computed on actual feature values from features_npz_path
    (X_test_all, all folds concatenated), so rho reflects true co-variation
    in the test population.
    Saves a heatmap PNG and a pairwise CSV to figures_dir.
    """
    from scipy.stats import spearmanr

    if not shap_npz_path.exists():
        print(f"[{pipeline_name}] Skipping Spearman: {shap_npz_path} not found")
        return
    if not features_npz_path.exists():
        print(f"[{pipeline_name}] Skipping Spearman: {features_npz_path} not found — re-run experiments.py")
        return

    shap_data = np.load(shap_npz_path, allow_pickle=True)
    feat_data = np.load(features_npz_path, allow_pickle=True)

    if "shap_overall" not in shap_data.files:
        print(f"[{pipeline_name}] Skipping Spearman: shap_overall not in {shap_npz_path}")
        return
    for key in ("X_test_all", "feature_names"):
        if key not in feat_data.files:
            print(f"[{pipeline_name}] Skipping Spearman: {key} not in {features_npz_path} — re-run experiments.py")
            return

    shap_mat: np.ndarray = shap_data["shap_overall"].astype(float)  # (n_folds, n_features)
    X_all: np.ndarray = feat_data["X_test_all"].astype(float)        # (n_cases, n_features)
    feature_names: list[str] = feat_data["feature_names"].tolist()

    mean_abs = shap_mat.mean(axis=0)  # (n_features,) — used for ranking only

    phys_idx_map = {f: i for i, f in enumerate(feature_names) if feature_group(f) == "physiology"}
    struct_groups = {"structural_miss"} | (
        {"structural_plaus"} if any(feature_group(f) == "structural_plaus" for f in feature_names) else set()
    )
    struct_idx_map = {
        f: i for i, f in enumerate(feature_names)
        if feature_group(f) in struct_groups
    }

    if not phys_idx_map or not struct_idx_map:
        print(f"[{pipeline_name}] Skipping Spearman: phys={len(phys_idx_map)}, struct={len(struct_idx_map)}")
        return

    top_phys = sorted(phys_idx_map, key=lambda f: mean_abs[phys_idx_map[f]], reverse=True)[:top_k_phys]
    top_struct = sorted(struct_idx_map, key=lambda f: mean_abs[struct_idx_map[f]], reverse=True)[:top_k_struct]

    rows: list[dict] = []
    for sf in top_struct:
        sv = X_all[:, struct_idx_map[sf]]
        for pf in top_phys:
            pv = X_all[:, phys_idx_map[pf]]
            valid = np.isfinite(pv) & np.isfinite(sv)
            n = int(valid.sum())
            if n < 3 or np.std(sv[valid]) == 0 or np.std(pv[valid]) == 0:
                rho, pval = float("nan"), float("nan")
            else:
                rho, pval = spearmanr(sv[valid], pv[valid])
            abs_rho = float(abs(rho)) if not np.isnan(rho) else float("nan")
            if np.isnan(abs_rho):
                effect = "undefined"
            elif abs_rho < 0.1:
                effect = "negligible_overlap"
            elif abs_rho < 0.3:
                effect = "small_overlap"
            elif abs_rho < 0.5:
                effect = "moderate_overlap"
            else:
                effect = "high_overlap"
            rows.append({
                "structural_feature": sf,
                "physiology_feature": pf,
                "structural_mean_abs_shap": float(mean_abs[struct_idx_map[sf]]),
                "physiology_mean_abs_shap": float(mean_abs[phys_idx_map[pf]]),
                "rho": float(rho) if not np.isnan(rho) else float("nan"),
                "abs_rho": abs_rho,
                "effect_size": effect,
                "valid_n": n,
            })

    if not rows:
        print(f"[{pipeline_name}] Spearman: no valid pairs computed.")
        return

    sp_df = pl.DataFrame(rows).sort("abs_rho", descending=True)

    safe = pipeline_name.replace(" ", "_").replace("/", "__")
    csv_path = figures_dir / f"spearman_pairwise_{safe}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sp_df.write_csv(csv_path)
    print(f"[{pipeline_name}] Spearman pairwise CSV saved to {csv_path}")

    # ── Interpretation summary ────────────────────────────────────────────────
    valid_rows = [r for r in sp_df.iter_rows(named=True) if not np.isnan(r["rho"])]
    n_pairs = len(valid_rows)
    abs_rhos = [r["abs_rho"] for r in valid_rows]
    mean_abs_rho = float(np.mean(abs_rhos)) if abs_rhos else float("nan")
    n_small    = sum(1 for v in abs_rhos if 0.1 <= v < 0.3)
    n_moderate = sum(1 for v in abs_rhos if 0.3 <= v < 0.5)
    n_high     = sum(1 for v in abs_rhos if v >= 0.5)
    top5 = valid_rows[:5]

    print(f"\n[{pipeline_name}] Spearman orthogonality interpretation")
    print(f"  {n_pairs} pairs · mean |rho| = {mean_abs_rho:.3f}")
    print(f"  Overlap with physiology: negligible (<0.1): {n_pairs - n_small - n_moderate - n_high} · "
          f"small (0.1–0.3): {n_small} · moderate (0.3–0.5): {n_moderate} · high (≥0.5): {n_high}")
    if mean_abs_rho < 0.1:
        verdict = "structural features are largely orthogonal to physiology — they provide independent information"
    elif mean_abs_rho < 0.3:
        verdict = "small average co-variation — structural features are mostly independent, with some overlap"
    elif mean_abs_rho < 0.5:
        verdict = "moderate co-variation — structural features partially overlap with physiology signal"
    else:
        verdict = "strong co-variation — structural features are highly correlated with physiology, suggesting redundancy"
    print(f"  Interpretation: {verdict}")
    if top5:
        print(f"  Top {len(top5)} pairs by |rho|:")
        for r in top5:
            direction = "positive" if r["rho"] > 0 else "negative"
            print(f"    {r['structural_feature']} × {r['physiology_feature']}: "
                  f"rho={r['rho']:+.3f} ({direction}, {r['effect_size']}) — "
                  f"when {r['structural_feature']} is high, "
                  f"{r['physiology_feature']} tends to be "
                  f"{'higher' if r['rho'] > 0 else 'lower'}")
    print()

    # heatmap
    rho_lookup = {(r["structural_feature"], r["physiology_feature"]): r["rho"] for r in sp_df.iter_rows(named=True)}
    effect_lookup = {(r["structural_feature"], r["physiology_feature"]): r["effect_size"] for r in sp_df.iter_rows(named=True)}
    struct_order = (
        sp_df.select(["structural_feature", "structural_mean_abs_shap"])
        .unique(subset=["structural_feature"], keep="first")
        .sort("structural_mean_abs_shap", descending=True)["structural_feature"]
        .to_list()
    )
    phys_order = (
        sp_df.select(["physiology_feature", "physiology_mean_abs_shap"])
        .unique(subset=["physiology_feature"], keep="first")
        .sort("physiology_mean_abs_shap", descending=True)["physiology_feature"]
        .to_list()
    )
    rho_mat = np.full((len(struct_order), len(phys_order)), np.nan)
    for i, sf in enumerate(struct_order):
        for j, pf in enumerate(phys_order):
            rho_mat[i, j] = rho_lookup.get((sf, pf), float("nan"))

    fig_w = max(8.0, 1.2 * len(phys_order) + 3.0)
    fig_h = max(5.0, 0.8 * len(struct_order) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(rho_mat, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman rho")
    ax.set_xticks(np.arange(len(phys_order)))
    ax.set_xticklabels(phys_order, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(struct_order)))
    ax.set_yticklabels(struct_order, fontsize=7)
    ax.set_xlabel("Physiology features (SHAP-ranked)")
    ax.set_ylabel("Structural features (SHAP-ranked)")
    ax.set_title(f"{pipeline_name}: Spearman orthogonality heatmap\n^ moderate overlap |rho|≥0.3  ^^ high overlap |rho|≥0.5", fontsize=9)
    for i, sf in enumerate(struct_order):
        for j, pf in enumerate(phys_order):
            v = rho_mat[i, j]
            e = effect_lookup.get((sf, pf), "")
            marker = {"moderate_overlap": "^", "high_overlap": "^^"}.get(e, "")
            ax.text(j, i, "NA" if np.isnan(v) else f"{v:.2f}{marker}",
                    ha="center", va="center", fontsize=7, color="black")

    heatmap_path = figures_dir / f"spearman_heatmap_{safe}.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, format="png", dpi=150)
    plt.close(fig)
    print(f"[{pipeline_name}] Spearman heatmap saved to {heatmap_path}")
