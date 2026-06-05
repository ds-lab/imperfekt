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

_PLAUS_LABEL = {"pk": "plaus=keep", "pr": "plaus=remove"}
_IMP_LABEL   = {"in": "no imputation", "il": "LOCF", "is": "SAITS"}


def _decode_pipeline_label(label: str) -> str:
    """
    Convert internal pipeline names to human-readable legend labels.

    'Setup ma_pk_in/base+miss' → 'plaus=keep · no imputation\nbase+miss'
    Falls back to the raw label if the pattern is not recognised.
    """
    name = label.removeprefix("Setup ")
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
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=_decode_pipeline_label(label))
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)
        all_lo.append(lo)
        all_hi.append(hi)

        overall = summary.get("overall", {}).get("auprc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

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
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=_decode_pipeline_label(label))
        ax.plot(x, lo, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(x, hi, linestyle="--", color=color, linewidth=0.8, alpha=0.6)
        ax.fill_between(x, lo, hi, color=color, alpha=0.1)
        all_lo.append(lo)
        all_hi.append(hi)

        overall = summary.get("overall", {}).get("auroc")
        if overall:
            ax.axhline(overall["mean"], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

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
        mean, lo, hi = _vals(summary)
        ax.plot(x, mean, marker="o", color=color, label=_decode_pipeline_label(label))
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
    fig.savefig(save_path, format="svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}")
