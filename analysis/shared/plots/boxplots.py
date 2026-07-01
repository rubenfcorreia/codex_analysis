from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def plot_boxplot_series(
    values_by_series: Sequence[Sequence[float] | np.ndarray],
    labels: Sequence[str],
    series_names: Sequence[str],
    series_colors: Sequence[str],
    output_dir: Path | str,
    *,
    stem: str,
    title: str,
    ylabel: str,
    xlabel: str = "State",
    title_color: str = "#334155",
    label_color_fn: Callable[[str], str] | None = None,
    edge_color: str = "#334155",
) -> list[Path]:
    cleaned_values: list[np.ndarray] = []
    cleaned_labels: list[str] = []
    cleaned_series: list[str] = []
    cleaned_colors: list[str] = []
    for values, label, series_name, color in zip(values_by_series, labels, series_names, series_colors):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        cleaned_values.append(arr)
        cleaned_labels.append(label)
        cleaned_series.append(series_name)
        cleaned_colors.append(color)
    if not cleaned_values:
        return []

    fig_width = max(12.0, 1.25 * len(cleaned_values))
    fig, ax = plt.subplots(figsize=(fig_width, 8), constrained_layout=True)
    bp = ax.boxplot(
        cleaned_values,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 2.2},
        whiskerprops={"color": "#555555", "linewidth": 1.8},
        capprops={"color": "#555555", "linewidth": 1.8},
        boxprops={"linewidth": 2.0},
    )
    for patch, color in zip(bp.get("boxes", []), cleaned_colors):
        patch.set_facecolor(mcolors.to_rgba(color, 0.28))
        patch.set_edgecolor(edge_color)
    rng = np.random.default_rng(0)
    for xpos, (series_name, values, color) in enumerate(zip(cleaned_series, cleaned_values, cleaned_colors), start=1):
        jitter = rng.normal(0.0, 0.06, size=values.size)
        ax.scatter(
            np.full(values.shape, xpos, dtype=float) + jitter,
            values,
            s=20,
            alpha=0.55,
            color=color,
            edgecolors="none",
            zorder=3,
        )
    ax.set_title(title, fontsize=20, fontweight="bold", color=title_color, pad=12)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xticks(list(range(1, len(cleaned_labels) + 1)))
    ax.set_xticklabels(cleaned_labels, rotation=30, ha="right")
    for tick, series_name in zip(ax.get_xticklabels(), cleaned_series):
        if label_color_fn is not None:
            tick.set_color(label_color_fn(series_name))
        else:
            tick.set_color("#1f2937")
        tick.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]
