
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from analysis.shared.plots.boxplots import draw_boxplot_series
from analysis.shared.roi_split import WINDOW_DISPLAY_LABELS, WINDOW_LABELS
from analysis.shared.state_utils import canonical_state_label, ensure_dir, safe_filename_component


DEFAULT_BINARY_GROUPS = (
    ("more_active", "More active", "#4c78a8"),
    ("less_active", "Less active", "#f58518"),
)
DEFAULT_QUADRANT_GROUPS = (
    ("high_activity_high_frequency", "High activity / high frequency", "#4c78a8"),
    ("high_activity_low_frequency", "High activity / low frequency", "#72b7b2"),
    ("low_activity_high_frequency", "Low activity / high frequency", "#f58518"),
    ("low_activity_low_frequency", "Low activity / low frequency", "#e45756"),
)
DEFAULT_GROUP_COLORS = ("#4c78a8", "#f58518", "#72b7b2", "#e45756", "#8b5cf6", "#10b981")
RESPONSE_LABELS = {
    "mean": "Mean activity",
    "corr": "Correlation",
    "mean_corr": "Correlation",
    "event_frequency_per_min": "Event frequency / min",
    "dendrite_event_frequency_per_min": "Dendrite event frequency / min",
    "spine_event_frequency_per_min": "Spine event frequency / min",
    "coincident_event_frequency_per_min": "Coincident event frequency / min",
    "noncoincident_event_frequency_per_min": "Noncoincident event frequency / min",
    "mean_dendrite_activity": "Mean dendrite activity",
    "mean_spine_activity_per_dendrite": "Mean spine activity per dendrite",
}


def _response_label(column: Any) -> str:
    key = str(column or "").strip()
    if not key:
        return "Response"
    if key in RESPONSE_LABELS:
        return RESPONSE_LABELS[key]
    text = key.replace("_per_min", " per min").replace("_", " ")
    text = " ".join(text.split())
    return text[:1].upper() + text[1:] if text else key


def _group_specs_from_bundle(bundle: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    group_series = [str(value).strip().lower() for value in bundle.get("group_series", []) if str(value).strip()]
    group_labels = [str(value).strip() for value in bundle.get("group_labels", []) if str(value).strip()]
    group_colors = [str(value).strip() for value in bundle.get("group_colors", []) if str(value).strip()]
    if not group_series:
        seen: list[str] = []
        for row in rows:
            group = str(row.get("group") or "").strip().lower()
            if group and group not in seen:
                seen.append(group)
        group_series = seen
    if not group_series:
        split_mode = canonical_state_label(bundle.get("split_mode"))
        default_groups = DEFAULT_QUADRANT_GROUPS if split_mode in {"activity_frequency", "quadrant"} else DEFAULT_BINARY_GROUPS
        return [
            {"group": group, "label": label, "color": color}
            for group, label, color in default_groups
        ]
    if not group_labels:
        group_labels = [group.replace("_", " ").title() for group in group_series]
    if len(group_colors) < len(group_series):
        palette = list(DEFAULT_GROUP_COLORS)
        for index in range(len(group_series) - len(group_colors)):
            group_colors.append(palette[index % len(palette)])
    return [
        {
            "group": group,
            "label": group_labels[index] if index < len(group_labels) else group.replace("_", " ").title(),
            "color": group_colors[index] if index < len(group_colors) else DEFAULT_GROUP_COLORS[index % len(DEFAULT_GROUP_COLORS)],
        }
        for index, group in enumerate(group_series)
    ]


def roi_split_figure_output_dir(
    result_root: Path | str,
    *,
    roi_type: Any,
    split_name: Any,
    compartment: Any | None = None,
    branch_name: Any | None = None,
    basis_name: Any | None = None,
) -> Path:
    branch_text = str(branch_name or "").strip().lower()
    basis_text = canonical_state_label(basis_name)
    compartment_text = str(compartment or "").strip().lower()
    root_path = Path(result_root)
    branch_component = safe_filename_component(branch_text)
    basis_component = safe_filename_component(basis_text)
    root_parts = [safe_filename_component(part) for part in root_path.parts[-2:]]
    already_leaf_scoped = branch_text and basis_text and len(root_parts) >= 2 and root_parts[-2] == branch_component and root_parts[-1] == basis_component
    if branch_text and basis_text and not already_leaf_scoped:
        parts = ["figures", branch_component, basis_component, "roi_split", safe_filename_component(roi_type)]
        if compartment_text:
            parts.append(safe_filename_component(compartment_text))
        return ensure_dir(root_path.joinpath(*parts))
    parts = ["figures", "roi_split", safe_filename_component(roi_type)]
    if compartment_text:
        parts.append(safe_filename_component(compartment_text))
    parts.append(safe_filename_component(split_name))
    return ensure_dir(Path(result_root).joinpath(*parts))


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]


def _window_rows(window_rows: Sequence[Mapping[str, Any]], window: str) -> list[Mapping[str, Any]]:
    window_key = canonical_state_label(window)
    return [row for row in window_rows if canonical_state_label(row.get("window")) == window_key]


def _comparison_rows_for_panel(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    window: str,
    response_column: str,
) -> list[Mapping[str, Any]]:
    window_key = canonical_state_label(window)
    response_key = str(response_column or "").strip().lower()
    rows: list[Mapping[str, Any]] = []
    for row in comparison_rows:
        if not isinstance(row, Mapping):
            continue
        if canonical_state_label(row.get("window")) != window_key:
            continue
        if str(row.get("response_column") or "").strip().lower() != response_key:
            continue
        rows.append(row)
    return rows


def plot_roi_split_bundle_figure(
    bundle: Mapping[str, Any],
    result_root: Path | str,
) -> list[Path]:
    if plt is None or not isinstance(bundle, Mapping):
        return []

    roi_type = str(bundle.get("roi_type") or "roi").strip().lower() or "roi"
    split_name = str(bundle.get("split_name") or "split").strip().lower() or "split"
    branch_name = str(bundle.get("branch_name") or "").strip().lower() or None
    basis_name = str(bundle.get("basis_name") or "").strip().lower() or None
    compartment = str(bundle.get("compartment") or "").strip().lower() or None
    response_columns = [str(column).strip() for column in bundle.get("response_columns", []) if str(column).strip()]
    window_rows = [row for row in bundle.get("window_response_rows", []) if isinstance(row, Mapping)]
    comparison_rows = [row for row in bundle.get("window_response_comparison_rows", []) if isinstance(row, Mapping)]
    if not response_columns or not window_rows:
        return []

    group_specs = _group_specs_from_bundle(bundle, window_rows)
    group_series = [str(spec["group"]).strip().lower() for spec in group_specs if str(spec.get("group") or "").strip()]
    group_labels = [str(spec["label"]) for spec in group_specs if str(spec.get("group") or "").strip()]
    group_colors = [str(spec["color"]) for spec in group_specs if str(spec.get("group") or "").strip()]
    if not group_series:
        return []
    group_color_map = {series: color for series, color in zip(group_series, group_colors)}

    output_dir = roi_split_figure_output_dir(
        result_root,
        roi_type=roi_type,
        split_name=split_name,
        compartment=compartment,
        branch_name=branch_name,
        basis_name=basis_name,
    )
    stem_parts = ["roi_split", safe_filename_component(roi_type)]
    if compartment:
        stem_parts.append(safe_filename_component(compartment))
    stem_parts.append(safe_filename_component(split_name))
    if basis_name:
        stem_parts.append(safe_filename_component(basis_name))
    stem = "_".join(stem_parts)

    windows = list(WINDOW_LABELS)
    fig_width = max(8.0, 3.8 * len(response_columns) + 1.0)
    fig_height = max(5.2, 2.2 * len(windows) + 0.5)
    fig, axes = plt.subplots(
        len(windows),
        len(response_columns),
        figsize=(fig_width, fig_height),
        sharey="col" if len(response_columns) > 1 else False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.09, top=0.91, wspace=0.20, hspace=0.32)
    any_panel_plotted = False

    for row_index, window in enumerate(windows):
        panel_rows = _window_rows(window_rows, window)
        row_label = WINDOW_DISPLAY_LABELS.get(window, str(window).replace("_", " ").title())
        for col_index, response_column in enumerate(response_columns):
            ax = axes[row_index, col_index]
            response_rows: list[tuple[Mapping[str, Any], float]] = []
            for row in panel_rows:
                try:
                    value_f = float(row.get(response_column))
                except Exception:
                    continue
                if not np.isfinite(value_f):
                    continue
                response_rows.append((row, value_f))
            values_by_group: list[list[float]] = [[] for _ in group_series]
            for row, value_f in response_rows:
                group = str(row.get("group") or "").strip().lower()
                if group not in group_series:
                    continue
                values_by_group[group_series.index(group)].append(value_f)
            if sum(bool(values) for values in values_by_group) < 2:
                ax.set_axis_off()
                continue

            panel_comparisons = _comparison_rows_for_panel(comparison_rows, window=window, response_column=response_column)
            plotted = draw_boxplot_series(
                ax,
                values_by_group,
                group_labels,
                group_series,
                group_colors,
                title="",
                ylabel="",
                xlabel="",
                comparison_rows=panel_comparisons,
                top_labels=[f"n={len(values)}" for values in values_by_group],
                label_color_fn=lambda series: group_color_map.get(str(series).strip().lower(), "#1f2937"),
            )
            if not plotted:
                ax.set_axis_off()
                continue
            any_panel_plotted = True
            if row_index == 0:
                ax.set_title(_response_label(response_column), fontsize=12, fontweight="bold", pad=8)
            else:
                ax.set_title("")
            if col_index == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=28)
                ax.yaxis.set_label_coords(-0.16, 0.5)
            else:
                ax.tick_params(axis="y", labelleft=False)
                ax.set_ylabel("")
            if row_index < len(windows) - 1:
                ax.tick_params(axis="x", labelbottom=False)

    if not any_panel_plotted:
        plt.close(fig)
        return []

    title_bits = ["ROI split"]
    if branch_name:
        title_bits.append(branch_name.replace("_", " ").title())
    if basis_name:
        title_bits.append(basis_name.replace("_", " ").title())
    if roi_type:
        title_bits.append(roi_type.replace("_", " ").title())
    if compartment:
        title_bits.append(compartment.title())
    fig.suptitle(" - ".join(title_bits), fontsize=14, fontweight="bold", y=0.985)
    return _save_figure(fig, output_dir, stem)


__all__ = [
    "plot_roi_split_bundle_figure",
    "roi_split_figure_output_dir",
]
