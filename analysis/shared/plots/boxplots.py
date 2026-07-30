from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from analysis.shared.statistics import is_significant_row


FIGURE_WIDTH_MM = 170.0
FIGURE_HEIGHT_MM = 75.0
FIGURE_TITLE_FS = 12
FIGURE_LABEL_FS = 11
FIGURE_TICK_FS = 9
FIGURE_NOTE_FS = 9


def _boxplot_significance_stars(p_value: Any) -> str:
    try:
        p = float(p_value)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _comparison_display_label(row: Mapping[str, Any]) -> str:
    def _clean(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text.replace("_", " ").replace("-", " ").strip().title()

    left = row.get("state_a_display") or row.get("state_a") or row.get("x1_label") or row.get("x1_state") or ""
    right = row.get("state_b_display") or row.get("state_b") or row.get("x2_label") or row.get("x2_state") or ""
    left_text = _clean(left)
    right_text = _clean(right)
    if left_text and right_text:
        return f"{left_text} vs {right_text}"
    if left_text or right_text:
        return left_text or right_text
    return ""


def _draw_boxplot_significance_annotations(
    ax: plt.Axes,
    annotation_rows: Sequence[Mapping[str, Any]],
    *,
    horizontal: bool,
) -> None:
    if not annotation_rows:
        return
    from analysis.dendrites_pipeline.dendrites_pipeline import _draw_boxplot_significance_annotations as _pipeline_draw_boxplot_significance_annotations

    _pipeline_draw_boxplot_significance_annotations(ax, annotation_rows, orientation="horizontal" if horizontal else "vertical")


def draw_boxplot_series(
    ax: plt.Axes,
    values_by_series: Sequence[Sequence[float] | np.ndarray],
    labels: Sequence[str],
    series_names: Sequence[str],
    series_colors: Sequence[str],
    *,
    title: str,
    ylabel: str,
    xlabel: str = "State",
    title_color: str = "#334155",
    label_color_fn: Callable[[str], str] | None = None,
    edge_color: str = "#334155",
    significance_flags: Sequence[bool] | None = None,
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    top_labels: Sequence[str] | None = None,
    horizontal: bool = False,
) -> bool:
    cleaned_values: list[np.ndarray] = []
    cleaned_labels: list[str] = []
    cleaned_series: list[str] = []
    cleaned_colors: list[str] = []
    cleaned_flags: list[bool] = []
    cleaned_top_labels: list[str] = []
    flags = list(significance_flags) if significance_flags is not None else None
    top_list = list(top_labels) if top_labels is not None else None

    for index, (values, label, series_name, color) in enumerate(zip(values_by_series, labels, series_names, series_colors)):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        cleaned_values.append(arr)
        cleaned_labels.append(str(label))
        cleaned_series.append(str(series_name))
        cleaned_colors.append(str(color))
        if top_list is not None:
            cleaned_top_labels.append(str(top_list[index]) if index < len(top_list) else "")
        if flags is not None:
            cleaned_flags.append(bool(flags[index]) if index < len(flags) else False)

    if not cleaned_values:
        return False

    bp = ax.boxplot(
        cleaned_values,
        patch_artist=True,
        showfliers=False,
        vert=not horizontal,
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
        if horizontal:
            ax.scatter(
                values,
                np.full(values.shape, xpos, dtype=float) + jitter,
                s=20,
                alpha=0.55,
                color=color,
                edgecolors="none",
                zorder=3,
            )
        else:
            ax.scatter(
                np.full(values.shape, xpos, dtype=float) + jitter,
                values,
                s=20,
                alpha=0.55,
                color=color,
                edgecolors="none",
                zorder=3,
            )

    if comparison_rows:
        annotation_rows: list[dict[str, Any]] = []
        position_lookup = {str(series_name).strip().lower(): idx for idx, series_name in enumerate(cleaned_series, start=1)}
        for row in comparison_rows:
            if not isinstance(row, Mapping):
                continue
            if not is_significant_row(dict(row), p_key="shuffle_p"):
                continue
            x1 = row.get("x1")
            x2 = row.get("x2")
            if x1 is None or x2 is None:
                state_a = str(row.get("state_a") or row.get("state_a_display") or row.get("x1_label") or row.get("x1_state") or "").strip().lower()
                state_b = str(row.get("state_b") or row.get("state_b_display") or row.get("x2_label") or row.get("x2_state") or "").strip().lower()
                if not state_a or not state_b:
                    continue
                x1 = position_lookup.get(state_a)
                x2 = position_lookup.get(state_b)
            if x1 is None or x2 is None:
                continue
            stars = _boxplot_significance_stars(row.get("shuffle_p"))
            annotation_rows.append({"x1": float(x1), "x2": float(x2), "shuffle_p": row.get("shuffle_p"), "label": f"{_comparison_display_label(row)} {stars}".strip()})
        _draw_boxplot_significance_annotations(ax, annotation_rows, horizontal=horizontal)
    elif flags is not None and any(cleaned_flags):
        finite = np.concatenate(cleaned_values)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            extent = float(np.nanmax(finite)) + max(0.05 * float(np.ptp(finite)), 0.05)
        else:
            extent = 1.0
        for xpos, is_sig in enumerate(cleaned_flags, start=1):
            if not is_sig:
                continue
            if horizontal:
                ax.text(extent, xpos, "*", ha="left", va="center", fontsize=FIGURE_NOTE_FS, color="#8b0000", fontweight="bold")
            else:
                ax.text(xpos, extent, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS, color="#8b0000", fontweight="bold")

    ax.set_title(title, fontsize=FIGURE_TITLE_FS, fontweight="bold", color=title_color, pad=8)
    if horizontal:
        ax.set_ylabel(xlabel, fontsize=FIGURE_LABEL_FS)
        ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.grid(axis="x", alpha=0.18, linewidth=0.8)
        ax.tick_params(axis="x", labelsize=FIGURE_TICK_FS)
        ax.tick_params(axis="y", labelsize=FIGURE_TICK_FS)
        ax.set_yticks(list(range(1, len(cleaned_labels) + 1)))
        ax.set_yticklabels(cleaned_labels)
        for tick, series_name in zip(ax.get_yticklabels(), cleaned_series):
            if label_color_fn is not None:
                tick.set_color(label_color_fn(series_name))
            else:
                tick.set_color("#1f2937")
            tick.set_fontweight("bold")
    else:
        ax.set_ylabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.set_xlabel(xlabel, fontsize=FIGURE_LABEL_FS)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.tick_params(axis="x", labelsize=FIGURE_TICK_FS)
        ax.tick_params(axis="y", labelsize=FIGURE_TICK_FS)
        ax.set_xticks(list(range(1, len(cleaned_labels) + 1)))
        ax.set_xticklabels(cleaned_labels, rotation=30, ha="right")
        for tick, series_name in zip(ax.get_xticklabels(), cleaned_series):
            if label_color_fn is not None:
                tick.set_color(label_color_fn(series_name))
            else:
                tick.set_color("#1f2937")
            tick.set_fontweight("bold")
    if cleaned_top_labels:
        if horizontal:
            xlim = ax.get_xlim()
            xr = float(xlim[1] - xlim[0]) if np.isfinite(xlim[1] - xlim[0]) and (xlim[1] - xlim[0]) > 0 else 1.0
            text_x = float(np.nanmax(np.concatenate(cleaned_values))) + max(0.015 * xr, 0.02)
            for ypos, top_label in zip(range(1, len(cleaned_top_labels) + 1), cleaned_top_labels):
                if not top_label:
                    continue
                ax.text(
                    text_x,
                    ypos,
                    top_label,
                    ha="left",
                    va="center",
                    fontsize=FIGURE_NOTE_FS,
                    color="#6b7280",
                    fontweight="normal",
                    clip_on=False,
                )
            ax.set_xlim(xlim[0], max(xlim[1], text_x + 0.08 * xr))
        else:
            ax.tick_params(axis="x", pad=18)
            for xpos, top_label in zip(range(1, len(cleaned_top_labels) + 1), cleaned_top_labels):
                if not top_label:
                    continue
                ax.text(
                    xpos,
                    -0.02,
                    top_label,
                    transform=ax.get_xaxis_transform(),
                    rotation=30,
                    ha="right",
                    va="bottom",
                    fontsize=FIGURE_NOTE_FS,
                    color="#6b7280",
                    fontweight="normal",
                )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return True


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
    significance_flags: Sequence[bool] | None = None,
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    top_labels: Sequence[str] | None = None,
    horizontal: bool = False,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4), constrained_layout=True)
    plotted = draw_boxplot_series(
        ax,
        values_by_series,
        labels,
        series_names,
        series_colors,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        title_color=title_color,
        label_color_fn=label_color_fn,
        edge_color=edge_color,
        significance_flags=significance_flags,
        comparison_rows=comparison_rows,
        top_labels=top_labels,
        horizontal=horizontal,
    )
    if not plotted:
        plt.close(fig)
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]
