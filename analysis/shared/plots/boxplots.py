from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np

from analysis.shared.roi_split import split_group_hatch
from analysis.shared.state_utils import state_display_color
from analysis.shared.statistics import is_significant_row
from analysis.shared.plots.figure_io import save_figure


FIGURE_WIDTH_MM = 190.0
FIGURE_HEIGHT_MM = 125.0
FIGURE_TITLE_FS = 12
FIGURE_LABEL_FS = 11
FIGURE_TICK_FS = 9
FIGURE_NOTE_FS = 9


def _hatch_contrast_color(color: Any) -> str:
    try:
        red, green, blue = mcolors.to_rgb(color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "#ffffff" if luminance < 0.58 else "#1f2937"
    except Exception:
        return "#1f2937"



def _short_group_label(group_key: Any, display: Any = None) -> str:
    key = str(group_key or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "high_activity_high_frequency": "HA/HF",
        "high_activity_low_frequency": "HA/LF",
        "low_activity_high_frequency": "LA/HF",
        "low_activity_low_frequency": "LA/LF",
        "more_active": "More active",
        "less_active": "Less active",
        "higher_frequency": "Higher freq.",
        "lower_frequency": "Lower freq.",
    }
    if key in aliases:
        return aliases[key]
    text = str(display or group_key or "").strip()
    return text.replace(" / ", "\n") if " / " in text else text



def _render_normalized_grouped_boxplot(
    rows: Sequence[Mapping[str, Any]],
    *,
    state_order: Sequence[str],
    title: str,
    ylabel: str,
    xlabel: str,
    title_color: str,
    y_limits: tuple[float, float] | None = None,
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    horizontal: bool = False,
) -> plt.Figure | None:
    """Render grouped series with optional basal/apical secondary pairs."""
    if not rows:
        return None
    state_keys = [_canonical_boxplot_key(value) for value in state_order if _canonical_boxplot_key(value)]
    if not state_keys:
        state_keys = list(dict.fromkeys(_canonical_boxplot_key(row.get("state")) for row in rows if _canonical_boxplot_key(row.get("state"))))
    fixed_groups = ["low_activity_low_frequency", "low_activity_high_frequency", "high_activity_low_frequency", "high_activity_high_frequency"]
    observed_groups = { _canonical_boxplot_key(row.get("group")) for row in rows }
    if observed_groups.intersection(fixed_groups):
        groups = list(fixed_groups)
    else:
        groups = [group for group in dict.fromkeys(_canonical_boxplot_key(row.get("group")) for row in rows) if group]
    secondary_order = list(dict.fromkeys(str(row.get("secondary") or "").strip().lower() for row in rows if str(row.get("secondary") or "").strip()))
    if not groups:
        return None
    state_rows: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    state_labels: dict[str, str] = {}
    state_colors: dict[str, str] = {}
    group_labels: dict[str, str] = {}
    group_hatches: dict[str, str] = {}
    for row in rows:
        state = _canonical_boxplot_key(row.get("state"))
        group = _canonical_boxplot_key(row.get("group"))
        secondary = str(row.get("secondary") or "").strip().lower()
        values = np.asarray(row.get("values", []), dtype=float)
        values = values[np.isfinite(values)]
        if not state or not group or not values.size:
            continue
        state_rows.setdefault(state, {})[(group, secondary)] = {**row, "values": values}
        state_labels.setdefault(state, str(row.get("state_label") or state.replace("_", " ").title()))
        state_colors.setdefault(state, str(row.get("state_color") or state_display_color(state)))
        group_labels.setdefault(group, str(row.get("group_label") or _short_group_label(group, group)))
        group_hatches.setdefault(group, str(row.get("hatch") or split_group_hatch(group)))
    state_keys = [state for state in state_keys if state in state_rows]
    if not observed_groups.intersection(fixed_groups):
        groups = [group for group in groups if any(group_key == group for data in state_rows.values() for group_key, _ in data)]
    secondary_order = [secondary for secondary in secondary_order if any(key[1] == secondary for data in state_rows.values() for key in data)] or [""]
    for group in groups:
        group_labels.setdefault(group, _short_group_label(group, group))
        group_hatches.setdefault(group, split_group_hatch(group))
    if not state_keys or not groups:
        return None
    group_offsets = {group: 0.0 if len(groups) == 1 else float(offset) for group, offset in zip(groups, np.linspace(-0.60, 0.60, len(groups)))}
    secondary_offsets = {secondary: 0.0 if len(secondary_order) == 1 else float(offset) for secondary, offset in zip(secondary_order, np.linspace(-0.075, 0.075, len(secondary_order)))}
    box_width = max(0.055, min(0.085, 0.16 / max(len(secondary_order), 1)))
    state_step = 1.85 if not horizontal else 1.0
    figure_width_mm = max(FIGURE_WIDTH_MM, 25.4 * (1.05 * state_step * len(state_keys) + 3.0))
    fig, ax = plt.subplots(figsize=(figure_width_mm / 25.4, FIGURE_HEIGHT_MM / 25.4), constrained_layout=False)
    state_position = {state: 1.0 + state_step * index for index, state in enumerate(state_keys)}
    positions: list[float] = []
    values_list: list[np.ndarray] = []
    styles: list[Mapping[str, Any]] = []
    position_lookup: dict[tuple[str, str, str], float] = {}
    for state in state_keys:
        for group_index, group in enumerate(groups):
            center = state_position[state] + group_offsets[group]
            if not horizontal:
                ax.axvspan(center - 0.20, center + 0.20, facecolor="#64748b" if group_index % 2 == 0 else "#94a3b8", alpha=0.035, linewidth=0, zorder=0)
            for secondary in secondary_order:
                position = center + secondary_offsets[secondary]
                position_lookup[(state, group, secondary)] = position
                row = state_rows.get(state, {}).get((group, secondary))
                if row is None:
                    continue
                positions.append(position)
                values_list.append(np.asarray(row["values"], dtype=float))
                styles.append(row)
    if not values_list:
        plt.close(fig)
        return None
    bp = ax.boxplot(values_list, positions=positions, widths=box_width, patch_artist=True, showfliers=False, vert=not horizontal,
                    medianprops={"color": "#111827", "linewidth": 2.2}, whiskerprops={"color": "#555555", "linewidth": 1.8},
                    capprops={"color": "#555555", "linewidth": 1.8}, boxprops={"linewidth": 2.0})
    for patch, style in zip(bp["boxes"], styles):
        face = style.get("face_color") or state_colors.get(_canonical_boxplot_key(style.get("state")), "#d1d5db")
        edge = style.get("edge_color") or face
        patch.set_facecolor(mcolors.to_rgba(face, 0.42))
        patch.set_edgecolor(edge)
        patch.set_hatch(str(style.get("hatch") or ""))
        if style.get("hatch"):
            patch._hatch_color = mcolors.to_rgba(_hatch_contrast_color(face))
    rng = np.random.default_rng(7)
    jitter = min(0.06, box_width * 0.30)
    for position, values, style in zip(positions, values_list, styles):
        color = style.get("edge_color") or style.get("face_color") or "#334155"
        offsets = rng.uniform(-jitter, jitter, size=values.size)
        if horizontal:
            ax.scatter(values, np.full(values.shape, position) + offsets, s=11, alpha=0.48, color=color, edgecolor="none", zorder=3)
        else:
            ax.scatter(np.full(values.shape, position) + offsets, values, s=11, alpha=0.48, color=color, edgecolor="none", zorder=3)
    if not horizontal:
        ax.margins(y=0.14)
        for index, (position, values, style) in enumerate(zip(positions, values_list, styles)):
            if not values.size:
                continue
            state = _canonical_boxplot_key(style.get("state"))
            label_color = state_colors.get(state, style.get("edge_color") or "#334155")
            ax.annotate(
                f"n={values.size}",
                (position, float(np.nanmax(values))),
                xytext=(0, 4 + 3 * (index % 2)),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=max(7, FIGURE_NOTE_FS - 1),
                color=label_color,
            )
    if horizontal:
        ax.set_yticks([state_position[state] for state in state_keys])
        ax.set_yticklabels([state_labels[state] for state in state_keys])
        ax.set_ylabel(xlabel, fontsize=FIGURE_LABEL_FS)
        ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.set_ylim(0.5, len(state_keys) + 0.5)
    else:
        tick_positions = [state_position[state] + group_offsets[group] for state in state_keys for group in groups]
        tick_labels = [_short_group_label(group, group_labels[group]) for state in state_keys for group in groups]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=max(8, FIGURE_TICK_FS - 1))
        state_axis = ax.secondary_xaxis("top")
        state_axis.set_xticks([state_position[state] for state in state_keys])
        state_axis.set_xticklabels([state_labels[state] for state in state_keys])
        for tick, state in zip(state_axis.get_xticklabels(), state_keys):
            tick.set_color(state_colors[state])
            tick.set_fontweight("bold")
        state_axis.tick_params(axis="x", labelsize=FIGURE_TICK_FS, pad=5, length=0)
        state_axis.spines["top"].set_visible(False)
        ax.set_xlabel("Activity / frequency subset", fontsize=FIGURE_LABEL_FS, labelpad=24)
        ax.set_ylabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.set_xlim(min(state_position.values()) - 0.84, max(state_position.values()) + 0.84)
    if y_limits is not None and not horizontal:
        ax.set_ylim(*y_limits)
    ax.grid(axis="x" if horizontal else "y", alpha=0.18)
    ax.set_title(title, fontsize=FIGURE_TITLE_FS, fontweight="bold", color=title_color, pad=10)
    if comparison_rows and not horizontal:
        y0, y1 = ax.get_ylim()
        y_range = max(float(y1 - y0), 1e-6)
        for index, row in enumerate(comparison_rows):
            if not is_significant_row(dict(row), p_key="shuffle_p"):
                continue
            x1, x2 = row.get("x1"), row.get("x2")
            if x1 is None or x2 is None:
                state_a = _canonical_boxplot_key(row.get("state_a") or row.get("state"))
                state_b = _canonical_boxplot_key(row.get("state_b"))
                if state_a and state_b:
                    x1, x2 = state_position.get(state_a), state_position.get(state_b)
                elif state_a:
                    group = _canonical_boxplot_key(row.get("group") or row.get("split_group"))
                    secondary = str(row.get("compartment") or "").strip().lower()
                    if group and (state_a, group, secondary) in position_lookup:
                        x1 = position_lookup[(state_a, group, secondary)]
                        other = "apical" if secondary == "basal" else "basal"
                        x2 = position_lookup.get((state_a, group, other), x1)
            if x1 is None or x2 is None:
                continue
            star = _boxplot_significance_stars(row.get("shuffle_p"))
            if not star:
                continue
            y = min(y1 - 0.04 * y_range, y0 + (0.88 + 0.035 * (index % 3)) * y_range)
            ax.plot([float(x1), float(x2)], [y, y], color="#8b0000", linewidth=1.0, clip_on=False)
            ax.text((float(x1) + float(x2)) / 2.0, y, star, ha="center", va="bottom", color="#8b0000", fontweight="bold")
    handles = [Patch(facecolor="#ffffff", edgecolor="#334155", hatch=group_hatches[group], label=group_labels[group]) for group in groups]
    if len(secondary_order) > 1:
        handles = [Patch(facecolor="#d1d5db", edgecolor="#555555", label=secondary.title()) for secondary in secondary_order] + handles
    if handles:
        fig.legend(handles=handles, frameon=False, fontsize=max(FIGURE_NOTE_FS - 1, 8), loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(4, len(handles)))
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.88))
    return fig


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
    for index, row in enumerate(annotation_rows):
        significant = bool(row.get("significant", False)) or is_significant_row(dict(row), p_key="adjusted_pvalue") or is_significant_row(dict(row), p_key="shuffle_p")
        if not significant:
            continue
        label = str(row.get("star") or "*")
        if horizontal:
            ax.text(0.98, index, label, transform=ax.get_yaxis_transform(), ha="right", va="center", color="#8b0000", fontweight="bold")
        else:
            ax.text(index + 1, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color="#8b0000", fontweight="bold")


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
    for position, values, color in zip(range(1, len(cleaned_values) + 1), cleaned_values, cleaned_colors):
        jitter = rng.normal(0.0, 0.08, size=values.size)
        if horizontal:
            ax.scatter(
                values,
                np.full(values.shape, position, dtype=float) + jitter,
                s=12,
                alpha=0.55,
                color=color,
                edgecolors='none',
                zorder=3,
            )
        else:
            ax.scatter(
                np.full(values.shape, position, dtype=float) + jitter,
                values,
                s=12,
                alpha=0.55,
                color=color,
                edgecolors='none',
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
    save_figure(fig, png, extra_formats=("svg",), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]



def _canonical_boxplot_key(value: Any) -> str:
    text = str(value or '').strip().lower().replace(' ', '_').replace('-', '_')
    while '__' in text:
        text = text.replace('__', '_')
    return text


def _boxplot_display_label(row: Mapping[str, Any], label_col: str | None, key: str) -> str:
    if label_col:
        label = str(row.get(label_col) or '').strip()
        if label:
            return label
    return key.replace('_', ' ').strip().title()


def _boxplot_color(row: Mapping[str, Any], color_col: str | None, fallback: str) -> str:
    if color_col:
        color = str(row.get(color_col) or '').strip()
        if color:
            return color
    return fallback


def plot_grouped_boxplot_series(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    *,
    state_col: str,
    value_col: str,
    state_order: Sequence[str],
    stem: str,
    title: str,
    ylabel: str,
    xlabel: str = 'State',
    title_color: str = '#334155',
    edge_color: str = '#334155',
    group_col: str = 'split_group',
    state_label_col: str = 'state_display',
    state_color_col: str = 'state_color',
    group_label_col: str = 'split_group_display',
    group_color_col: str = 'split_group_color',
    group_rank_col: str = 'split_group_rank',
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    horizontal: bool = False,
    secondary_col: str | None = None,
    values_col: str | None = None,
    face_color_col: str | None = None,
    edge_color_col: str | None = None,
    hatch_col: str | None = None,
    return_figure: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> list[Path] | plt.Figure | None:
    cleaned_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not cleaned_rows:
        return []
    if secondary_col:
        normalized = []
        for row in cleaned_rows:
            raw_values = row.get(values_col or value_col)
            if isinstance(raw_values, (list, tuple, np.ndarray)):
                values = raw_values
            else:
                try:
                    values = [float(raw_values)]
                except Exception:
                    values = []
            normalized.append({
                "state": row.get(state_col),
                "group": row.get(group_col),
                "secondary": row.get(secondary_col),
                "values": values,
                "state_label": row.get(state_label_col),
                "state_color": row.get(state_color_col) if state_color_col else None,
                "group_label": row.get(group_label_col) if group_label_col else None,
                "face_color": row.get(face_color_col) if face_color_col else None,
                "edge_color": row.get(edge_color_col) if edge_color_col else None,
                "hatch": row.get(hatch_col) if hatch_col else None,
            })
        figure = _render_normalized_grouped_boxplot(
            normalized,
            state_order=state_order,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            title_color=title_color,
            y_limits=y_limits,
            comparison_rows=comparison_rows,
            horizontal=horizontal,
        )
        if return_figure:
            return figure
        if figure is None:
            return []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / f"{stem}.png"
        save_figure(figure, png, extra_formats=("svg",), bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return [png, output_dir / f"{stem}.svg"]

    state_keys = [_canonical_boxplot_key(state) for state in state_order if _canonical_boxplot_key(state)]
    if not state_keys:
        seen_states: list[str] = []
        for row in cleaned_rows:
            state_key = _canonical_boxplot_key(row.get(state_col))
            if state_key and state_key not in seen_states:
                seen_states.append(state_key)
        state_keys = seen_states

    state_labels: dict[str, str] = {}
    state_colors: dict[str, str] = {}
    state_rows: dict[str, list[dict[str, Any]]] = {state: [] for state in state_keys}
    group_rows: dict[str, dict[str, list[float]]] = {state: {} for state in state_keys}
    group_labels: dict[str, str] = {}
    group_hatches: dict[str, str] = {}
    group_ranks: dict[str, float] = {}

    for row in cleaned_rows:
        state_key = _canonical_boxplot_key(row.get(state_col))
        if not state_key:
            continue
        if state_key not in state_rows:
            state_rows[state_key] = []
            state_keys.append(state_key)
        state_rows[state_key].append(row)
        state_labels.setdefault(state_key, _boxplot_display_label(row, state_label_col, state_key))
        state_colors.setdefault(state_key, state_display_color(row.get(state_label_col) or row.get(state_col) or state_key))
        if not group_col:
            continue
        group_key = _canonical_boxplot_key(row.get(group_col))
        if not group_key:
            continue
        try:
            value = float(row.get(value_col))
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        group_rows.setdefault(state_key, {}).setdefault(group_key, []).append(float(value))
        group_labels.setdefault(group_key, _boxplot_display_label(row, group_label_col, group_key))
        group_hatches.setdefault(group_key, split_group_hatch(row.get(group_col) or group_key))
        if group_rank_col:
            try:
                rank_value = float(row.get(group_rank_col))
            except Exception:
                rank_value = float('nan')
            if np.isfinite(rank_value):
                current = group_ranks.get(group_key)
                if current is None or rank_value < current:
                    group_ranks[group_key] = rank_value

    present_state_keys: list[str] = []
    for state_key in state_keys:
        rows_for_state = group_rows.get(state_key, {})
        if not rows_for_state:
            values = []
            for row in state_rows.get(state_key, []):
                try:
                    value = float(row.get(value_col))
                except Exception:
                    continue
                if np.isfinite(value):
                    values.append(value)
            if not values:
                continue
            present_state_keys.append(state_key)
        else:
            has_values = any(len(values) for values in rows_for_state.values())
            if has_values:
                present_state_keys.append(state_key)

    if not present_state_keys:
        return []

    present_group_keys: list[str] = []
    ordered_group_keys = list(group_labels)
    if group_ranks:
        ordered_group_keys = sorted(ordered_group_keys, key=lambda group: (group_ranks.get(group, float('inf')), group))
    for group_key in ordered_group_keys:
        has_values = any(len(group_rows.get(state_key, {}).get(group_key, [])) for state_key in present_state_keys)
        if has_values:
            present_group_keys.append(group_key)

    if len(present_group_keys) < 2 or not group_col:
        values_by_series: list[list[float]] = []
        labels: list[str] = []
        series_names: list[str] = []
        series_colors: list[str] = []
        for state_key in present_state_keys:
            values: list[float] = []
            for row in state_rows.get(state_key, []):
                try:
                    value = float(row.get(value_col))
                except Exception:
                    continue
                if np.isfinite(value):
                    values.append(value)
            if not values:
                continue
            values_by_series.append(values)
            labels.append(state_labels.get(state_key, state_key.replace('_', ' ').title()))
            series_names.append(state_key)
            series_colors.append(state_colors.get(state_key, state_display_color(state_labels.get(state_key, state_key))))
        if not values_by_series:
            return []
        return plot_boxplot_series(
            values_by_series,
            labels,
            series_names,
            series_colors,
            output_dir,
            stem=stem,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            title_color=title_color,
            edge_color=edge_color,
            comparison_rows=comparison_rows,
        )

    if len(present_group_keys) == 1:
        offsets = np.asarray([0.0], dtype=float)
    else:
        offsets = np.linspace(-0.60, 0.60, len(present_group_keys))
    box_width = max(0.07, min(0.12, 0.32 / max(len(present_group_keys), 1)))
    state_step = 1.85 if not horizontal else 1.0

    figure_width_mm = max(FIGURE_WIDTH_MM, 25.4 * (1.05 * state_step * len(present_state_keys) + 3.0))
    fig, ax = plt.subplots(figsize=(figure_width_mm / 25.4, FIGURE_HEIGHT_MM / 25.4), constrained_layout=False)
    series_values: list[np.ndarray] = []
    series_positions: list[float] = []
    series_colors: list[str] = []
    series_hatches: list[str] = []
    state_position_lookup = {
        state_key: 1.0 + state_step * index
        for index, state_key in enumerate(state_keys)
    }
    if not horizontal:
        for state_key in state_keys:
            state_center = state_position_lookup[state_key]
            for group_index, group_key in enumerate(present_group_keys):
                slot_center = state_center + float(offsets[group_index])
                slot_half_width = box_width / 2.0 + 0.06
                ax.axvspan(
                    slot_center - slot_half_width,
                    slot_center + slot_half_width,
                    facecolor="#64748b" if group_index % 2 == 0 else "#94a3b8",
                    alpha=0.035,
                    linewidth=0,
                    zorder=0,
                )

    for state_key in present_state_keys:
        group_map = group_rows.get(state_key, {})
        for group_index, group_key in enumerate(present_group_keys):
            values = np.asarray(group_map.get(group_key, []), dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            series_values.append(values)
            series_positions.append(state_position_lookup[state_key] + float(offsets[group_index]))
            series_colors.append(state_colors.get(state_key, state_display_color(state_labels.get(state_key, state_key))))
            series_hatches.append(group_hatches.get(group_key, split_group_hatch(group_key)))

    if not series_values:
        plt.close(fig)
        return []

    bp = ax.boxplot(
        series_values,
        positions=series_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        vert=not horizontal,
        medianprops={'color': '#111827', 'linewidth': 2.2},
        whiskerprops={'color': '#555555', 'linewidth': 1.8},
        capprops={'color': '#555555', 'linewidth': 1.8},
        boxprops={'linewidth': 2.0},
    )
    for patch, color, hatch in zip(bp.get('boxes', []), series_colors, series_hatches):
        patch.set_facecolor(mcolors.to_rgba(color, 0.26))
        patch.set_edgecolor(color)
        patch.set_hatch(hatch or '')

    rng = np.random.default_rng(0)
    jitter_half_width = min(0.06, box_width * 0.30)
    for position, values, color in zip(series_positions, series_values, series_colors):
        jitter = rng.uniform(-jitter_half_width, jitter_half_width, size=values.size)
        if horizontal:
            ax.scatter(
                values,
                np.full(values.shape, position, dtype=float) + jitter,
                s=12,
                alpha=0.55,
                color=color,
                edgecolors='none',
                zorder=3,
            )
        else:
            ax.scatter(
                np.full(values.shape, position, dtype=float) + jitter,
                values,
                s=12,
                alpha=0.55,
                color=color,
                edgecolors='none',
                zorder=3,
            )

    if comparison_rows:
        annotation_rows: list[dict[str, Any]] = []
        for row in comparison_rows:
            if not isinstance(row, Mapping):
                continue
            if not is_significant_row(dict(row), p_key='shuffle_p'):
                continue
            x1 = row.get('x1')
            x2 = row.get('x2')
            if x1 is None or x2 is None:
                state_a = _canonical_boxplot_key(row.get('state_a') or row.get('state_a_display') or row.get('x1_label') or row.get('x1_state'))
                state_b = _canonical_boxplot_key(row.get('state_b') or row.get('state_b_display') or row.get('x2_label') or row.get('x2_state'))
                if not state_a or not state_b:
                    continue
                x1 = state_position_lookup.get(state_a)
                x2 = state_position_lookup.get(state_b)
            if x1 is None or x2 is None:
                continue
            annotation_rows.append({'x1': float(x1), 'x2': float(x2), 'shuffle_p': row.get('shuffle_p')})
        _draw_boxplot_significance_annotations(ax, annotation_rows, horizontal=horizontal)

    if horizontal:
        ax.set_yticks(list(state_position_lookup.values()))
        ax.set_yticklabels([state_labels.get(state_key, state_key.replace('_', ' ').title()) for state_key in state_keys])
        for tick, state_key in zip(ax.get_yticklabels(), state_keys):
            tick.set_color(state_colors.get(state_key, '#1f2937'))
            tick.set_fontweight('bold')
        ax.set_ylabel(xlabel, fontsize=FIGURE_LABEL_FS)
        ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.grid(axis='x', alpha=0.18, linewidth=0.8)
        ax.tick_params(axis='x', labelsize=FIGURE_TICK_FS)
        ax.tick_params(axis='y', labelsize=FIGURE_TICK_FS)
        if series_values:
            all_values = np.concatenate(series_values)
            finite = all_values[np.isfinite(all_values)]
            if finite.size:
                x_min = float(np.nanmin(finite))
                x_max = float(np.nanmax(finite))
                pad = max(0.06 * (x_max - x_min) if x_max > x_min else 0.1, 0.05)
                ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(0.5, float(len(state_keys)) + 0.5)
    else:
        group_tick_positions = [
            state_position_lookup[state_key] + float(offsets[group_index])
            for state_key in state_keys
            for group_index in range(len(present_group_keys))
        ]
        group_tick_labels = [
            _short_group_label(group_key, group_labels.get(group_key, group_key))
            for state_key in state_keys
            for group_key in present_group_keys
        ]
        ax.set_xticks(group_tick_positions)
        ax.set_xticklabels(
            group_tick_labels,
            rotation=0,
            ha="center",
            va="top",
            fontsize=max(8, FIGURE_TICK_FS - 1),
            linespacing=0.9,
        )
        state_axis = ax.secondary_xaxis("top")
        state_axis.set_xticks([state_position_lookup[state_key] for state_key in state_keys])
        state_axis.set_xticklabels([state_labels.get(state_key, state_key.replace("_", " ").title()) for state_key in state_keys])
        for tick, state_key in zip(state_axis.get_xticklabels(), state_keys):
            tick.set_color(state_colors.get(state_key, "#1f2937"))
            tick.set_fontweight("bold")
        state_axis.tick_params(axis="x", labelsize=FIGURE_TICK_FS, pad=5, length=0)
        state_axis.spines["top"].set_visible(False)
        ax.set_ylabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.set_xlabel("Activity / frequency subset", fontsize=FIGURE_LABEL_FS, labelpad=24)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.tick_params(axis="x", labelsize=max(8, FIGURE_TICK_FS - 1), pad=7, length=0)
        ax.tick_params(axis="y", labelsize=FIGURE_TICK_FS)
        if series_values:
            all_values = np.concatenate(series_values)
            finite = all_values[np.isfinite(all_values)]
            if finite.size:
                y_min = float(np.nanmin(finite))
                y_max = float(np.nanmax(finite))
                pad = max(0.06 * (y_max - y_min) if y_max > y_min else 0.1, 0.05)
                ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(min(state_position_lookup.values()) - 0.82, max(state_position_lookup.values()) + 0.82)

    ax.set_title(title, fontsize=FIGURE_TITLE_FS, fontweight='bold', color=title_color, pad=10)
    if len(present_group_keys) > 1:
        legend_handles = [
            Patch(
                facecolor='#ffffff',
                edgecolor='#334155',
                linewidth=1.2,
                hatch=group_hatches.get(group_key, split_group_hatch(group_key)),
                label=group_labels.get(group_key, group_key.replace('_', ' ').title()),
            )
            for group_key in present_group_keys
        ]
        fig.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=max(FIGURE_NOTE_FS - 1, 8),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2 if len(legend_handles) > 2 else len(legend_handles),
            columnspacing=0.8,
            handletextpad=0.35,
            handlelength=1.0,
            borderaxespad=0.0,
        )
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.88))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f'{stem}.png'
    svg = output_dir / f'{stem}.svg'
    save_figure(fig, png, extra_formats=("svg",), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return [png, svg]
