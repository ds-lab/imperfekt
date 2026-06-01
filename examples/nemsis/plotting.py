from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CV_N_REPEATS,
    CV_N_SPLITS,
    PLOT_COLORS,
    SHOW_LEGEND,
)

_STRATUM_ORDER = ["Q_complete", "Q_alpha", "Q_beta", "Q_gamma", "Q_delta"]


def _sort_strata(keys: set[str]) -> list[str]:
    known = [k for k in _STRATUM_ORDER if k in keys]
    other = sorted(keys - set(_STRATUM_ORDER))
    return known + other


def _stratum_tick_labels(
    strata_keys: list[str], pipeline_summaries: list[tuple[str, dict[str, dict]]]
) -> list[str]:
    """
    Build tick labels from mean ± CI prevalence aggregated across all pipelines'
    CV folds, e.g. "Q_delta\n34%±2%". Uses n_pos_pct stored per fold in the summary.
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
    show_legend: bool | None = None,
    colors: list[str] | None = None,
) -> None:
    """
    Line plot: mean AUPRC per irregularity stratum for all provided pipelines.
    95% CI shown as dashed lines above and below the mean.
    Overall AUPRC shown as horizontal dashed reference lines.
    Tick labels show mean ± CI outcome prevalence derived from CV test folds.

    show_legend defaults to the config SHOW_LEGEND constant when None; colors
    defaults to the config PLOT_COLORS palette when None.
    """
    if show_legend is None:
        show_legend = SHOW_LEGEND
    if colors is None:
        colors = PLOT_COLORS
    strata_keys = _sort_strata(
        {k for _, summary in pipeline_summaries for k in summary if k != "overall"}
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


def plot_auroc_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool | None = None,
    colors: list[str] | None = None,
) -> None:
    """
    Line plot: mean AUROC per irregularity stratum for all provided pipelines.
    95% CI shown as dashed lines above and below the mean.
    Overall AUROC shown as horizontal dashed reference lines.
    Tick labels show mean outcome prevalence derived from CV test folds.

    show_legend defaults to the config SHOW_LEGEND constant when None; colors
    defaults to the config PLOT_COLORS palette when None.
    """
    if show_legend is None:
        show_legend = SHOW_LEGEND
    if colors is None:
        colors = PLOT_COLORS
    strata_keys = _sort_strata(
        {k for _, summary in pipeline_summaries for k in summary if k != "overall"}
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

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Random baseline")

    for idx, (label, summary) in enumerate(pipeline_summaries):
        color = colors[idx % len(colors)]
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)

        overall = summary.get("overall", {}).get("auroc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(_stratum_tick_labels(strata_keys, pipeline_summaries))
    ax.set_xlabel("Irregularity stratum (mean outcome prevalence from CV test folds)")
    ax.set_ylabel("AUROC (mean ± 95% CI)")
    ax.set_ylim(0.5, 1)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.set_title(f"AUROC by irregularity stratum ({CV_N_SPLITS}×{CV_N_REPEATS} CV)")
    fig.tight_layout(rect=[0, 0, 0.78 if show_legend else 1, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_auprc_lift_by_stratum(
    pipeline_summaries: list[tuple[str, dict[str, dict]]],
    save_path: Path,
    show_legend: bool | None = None,
    colors: list[str] | None = None,
) -> None:
    """
    Line plot: mean AUPRC lift (AUPRC / prevalence) per irregularity stratum.
    Lift > 1 means the model beats the no-skill baseline within that stratum.
    Reference line at lift = 1 (no-skill).
    Tick labels show mean ± CI outcome prevalence derived from CV test folds.

    show_legend defaults to the config SHOW_LEGEND constant when None; colors
    defaults to the config PLOT_COLORS palette when None.
    """
    if show_legend is None:
        show_legend = SHOW_LEGEND
    if colors is None:
        colors = PLOT_COLORS
    strata_keys = _sort_strata(
        {k for _, summary in pipeline_summaries for k in summary if k != "overall"}
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
