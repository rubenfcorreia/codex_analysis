#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
ANALYSIS_DIR = ROOT_DIR / "analysis"
DENDRITES_PIPELINE_DIR = ANALYSIS_DIR / "dendrites_pipeline"
for extra_path in (ROOT_DIR, ANALYSIS_DIR, DENDRITES_PIPELINE_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

if __name__ == "__main__":
    sys.modules.setdefault("analysis.deprecated.visual_response.poster_ready_visual_response", sys.modules[__name__])

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D
except Exception:  # pragma: no cover - matplotlib is required for the real figure generation
    plt = None

from poster_plotting import (
    POSTER_FONT_SIZE,
    POSTER_LABEL_SIZE,
    POSTER_LEGEND_SIZE,
    POSTER_NOTE_SIZE,
    POSTER_SUPTITLE_SIZE,
    POSTER_TITLE_SIZE,
    save_figure,
    configure_poster_matplotlib,
    set_sparse_numeric_ticks,
)

from analysis.dendrites_pipeline.dendrites_pipeline import (
    as_float,
    as_int,
    determine_conversion_mode,
    ensure_dir,
    load_conversion_library,
    locate_conversion_file,
    normalize_conversion_library,
    resolve_repo_root,
    safe_filename_component,
)

import analysis.dendrites_pipeline.figures.sleep_dendrite_spine_day_figures as sday

from analysis.dendrites_pipeline.figures.sleep_dendrite_spine_day_figures import (
    detail_label_from_local_ids,
    extract_roi_coordinates,
    load_mean_image,
    load_stat_path,
    plot_roi_overlays,
    resolve_ops_path,
)

from analysis.deprecated.visual_response.movie_visual_response import (
    DEFAULT_REMOTE_REPO_BASE,
    MOVIE_CATEGORY_COLORS,
    SOMA_TRACE_COLOR,
    add_significance_bracket,
    build_retinotopy_response_map,
    compact_list as movie_compact_list,
    compute_movie_compartment_statistics,
    get_category_trial_values,
    load_experiment_bundle,
    load_json_config_file,
    merge_config,
    normalize_soma_group_map,
    pool_movie_session_summaries,
    pool_soma_session_summaries,
    select_dendrite_entries,
    summarize_movie_session,
    summarize_soma_session,
    validate_group_map,
    load_experiments,
)

if plt is not None:
    configure_poster_matplotlib()

POSTER_OUTPUT_DIR = ROOT_DIR / "results" / "poster_ready"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "movie_visual_response_config.json"
FIGURE_WIDTH_CM = 36.0
FIGURE_HEIGHT_CM = 20.0
PANEL_TITLE_SIZE = max(6, POSTER_TITLE_SIZE - 15)
PANEL_LABEL_SIZE = max(6, POSTER_LABEL_SIZE - 9)
PANEL_FONT_SIZE = max(5, POSTER_FONT_SIZE - 9)
PANEL_NOTE_SIZE = max(5, POSTER_NOTE_SIZE - 8)
TOP_ROW_COLORS = {
    "soma": SOMA_TRACE_COLOR,
    "apical": MOVIE_CATEGORY_COLORS["apical"],
    "basal": MOVIE_CATEGORY_COLORS["basal"],
}


RETINO_PANEL_WIDTH_SCALE = 0.75
SOMA_GRATING_TRACE_YMAX = 0.3



@dataclass
class MeanImageSpec:
    title: str
    mean_img: np.ndarray
    overlays: List[Dict[str, Any]]
    color: str


@dataclass
class TracePanelSpec:
    title: str
    summary: Dict[str, Any]
    color: str
    kind: str
    x_label: str
    y_label: str = "dF/F"


@dataclass
class BoxplotSpec:
    title: str
    summary: Dict[str, Any]
    color: str
    kind: str
    category_name: str
    y_label: str = "dF/F"


@dataclass
class RetinoSpec:
    title: str
    summary: Dict[str, Any]
    color: str


@dataclass
class RowSpec:
    name: str
    color: str
    mean_image: MeanImageSpec
    trace_panels: List[TracePanelSpec]
    box_panels: List[BoxplotSpec]
    retino: Optional[RetinoSpec] = None


@dataclass
class PosterSpec:
    group_name: str
    rows: List[RowSpec]


def cm_to_inches(value_cm: float) -> float:
    return float(value_cm) / 2.54


def tint_palette(base_color: str, n_items: int) -> List[str]:
    if n_items <= 1:
        return [base_color]
    return [lighten_color(base_color, 0.10 + 0.45 * (index / max(n_items - 1, 1))) for index in range(n_items)]


def lighten_color(color: str, mix_with_white: float = 0.65) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - mix_with_white) + np.ones(3, dtype=float) * mix_with_white
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def build_movie_experiment_summaries(
    experiments: Dict[str, Any],
    exp_ids: Sequence[str],
    compartment: str,
    pre_window_s: float,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for exp_id in exp_ids:
        exp = experiments.get(exp_id)
        if exp is None or exp.cut_array is None:
            continue
        summaries.append(summarize_movie_session(exp, compartment, pre_window_s))
    return summaries


def build_soma_experiment_summaries(
    experiments: Dict[str, Any],
    exp_ids: Sequence[str],
    pre_window_s: float,
    post_window_s: float,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for exp_id in exp_ids:
        exp = experiments.get(exp_id)
        if exp is None:
            continue
        if exp.cut_array is None and exp.soma_trace is None:
            continue
        summaries.append(summarize_soma_session(exp, pre_window_s, post_window_s))
    return summaries


def pick_representative_experiment(experiments: Dict[str, Any], exp_ids: Sequence[str]) -> Optional[Any]:
    for exp_id in exp_ids:
        exp = experiments.get(exp_id)
        if exp is not None:
            return exp
    return None


def build_mean_image_spec(exp: Any, base_color: str, title: str, soma_fallback: bool = False) -> MeanImageSpec:
    ops_path = resolve_ops_path(exp.exp_root, 0)
    mean_img, _ = load_mean_image(ops_path)
    stat_path = load_stat_path(ops_path)
    overlays: List[Dict[str, Any]] = []

    if soma_fallback:
        if stat_path is not None:
            stat = np.load(stat_path, allow_pickle=True)
            if stat.shape[0] > 0:
                coords = extract_roi_coordinates({}, stat_entry=stat[0])
                if coords is not None:
                    overlays.append(
                        {
                            "coords": coords,
                            "color": base_color,
                            "linewidth": 2.2,
                            "label": "ROI 1",
                            "kind": "roi",
                        }
                    )
        return MeanImageSpec(title=title, mean_img=mean_img, overlays=overlays, color=base_color)

    dendrite_entries = select_dendrite_entries(exp.conversion_library)
    overlay_colors = tint_palette(base_color, max(len(dendrite_entries), 1))
    for index, entry in enumerate(dendrite_entries):
        coords = None
        try:
            coords = build_contour_coordinates(exp.conversion_library, entry, stat_path)
        except Exception:
            coords = None
        if coords is None:
            continue
        label = detail_label_from_local_ids(entry, str(entry.get("general_roi_id") or f"D{index + 1}"), is_child=False)
        overlays.append(
            {
                "coords": coords,
                "color": overlay_colors[index],
                "linewidth": 2.0,
                "label": label,
                "kind": "dendrite",
            }
        )
    return MeanImageSpec(title=title, mean_img=mean_img, overlays=overlays, color=base_color)


def build_contour_coordinates(raw_library: Dict[Any, Any], entry: Dict[str, Any], stat_path: Optional[Path]) -> Optional[np.ndarray]:
    plane_roi_id = as_int(entry.get("plane_roi_id"))
    general_roi_id = entry.get("general_roi_id")
    if stat_path is None:
        return None
    return load_contour_data(raw_library, general_roi_id, plane_roi_id, stat_path)


def load_contour_data(
    raw_library: Dict[Any, Any],
    general_roi_id: Any,
    plane_roi_id: Optional[int],
    stat_path: Optional[Path],
) -> Optional[np.ndarray]:
    lookup_keys: List[Any] = [general_roi_id]
    general_roi_text = str(general_roi_id)
    if general_roi_text.isdigit():
        lookup_keys.append(general_roi_text)
        lookup_keys.append(int(general_roi_text))
    raw_entry = None
    for lookup_key in lookup_keys:
        if lookup_key in raw_library:
            raw_entry = raw_library[lookup_key]
            break
    if raw_entry is None:
        for entry in raw_library.values():
            entry_general_id = entry.get("general_roi_id") if isinstance(entry, dict) else None
            if str(entry_general_id) == general_roi_text:
                raw_entry = entry.get("raw_entry", entry) if isinstance(entry, dict) else entry
                break
    if raw_entry is None:
        return None
    if isinstance(raw_entry, dict) and "raw_entry" in raw_entry and isinstance(raw_entry["raw_entry"], dict):
        raw_entry = raw_entry["raw_entry"]
    stat_entry = None
    if stat_path is not None and plane_roi_id is not None:
        stat = np.load(stat_path, allow_pickle=True)
        if 0 <= plane_roi_id < stat.shape[0]:
            stat_entry = stat[plane_roi_id]
    return extract_roi_coordinates(raw_entry, stat_entry)


def build_soma_mean_image_spec(exp: Any, title: str, base_color: str) -> MeanImageSpec:
    return build_mean_image_spec(exp, base_color, title, soma_fallback=True)


def build_trace_panel(title: str, summary: Dict[str, Any], color: str, kind: str, x_label: str) -> TracePanelSpec:
    return TracePanelSpec(title=title, summary=summary, color=color, kind=kind, x_label=x_label)


def build_movie_box_panel(title: str, summary: Dict[str, Any], color: str, category_name: str) -> BoxplotSpec:
    return BoxplotSpec(title=title, summary=summary, color=color, kind="movie", category_name=category_name)


def build_soma_box_panel(title: str, summary: Dict[str, Any], color: str) -> BoxplotSpec:
    return BoxplotSpec(title=title, summary=summary, color=color, kind="soma", category_name="grating")


def build_retino_spec(title: str, summary: Dict[str, Any], color: str) -> RetinoSpec:
    return RetinoSpec(title=title, summary=summary, color=color)


def compute_box_y_limits(box_specs: Sequence[BoxplotSpec], percentiles: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    values: List[np.ndarray] = []
    for spec in box_specs:
        if spec.kind == "soma":
            arrays = [
                np.asarray(spec.summary.get("paired_baseline_values", []), dtype=float),
                np.asarray(spec.summary.get("paired_stimulus_values", []), dtype=float),
            ]
        else:
            baseline_values, stimulus_values, blank_values = get_category_trial_values(spec.summary)
            arrays = [np.asarray(baseline_values, dtype=float), np.asarray(stimulus_values, dtype=float)]
            if spec.category_name != "blank":
                arrays.append(np.asarray(blank_values, dtype=float))
        for value_array in arrays:
            if value_array.size:
                values.append(value_array)
    if not values:
        return None
    stacked = np.concatenate(values)
    if percentiles is not None:
        lower_q, upper_q = percentiles
        y_min = float(np.nanpercentile(stacked, lower_q))
        y_max = float(np.nanpercentile(stacked, upper_q))
    else:
        y_min = float(np.nanmin(stacked))
        y_max = float(np.nanmax(stacked))
    pad = max(0.08 * max(y_max - y_min, 1e-6), 0.02)
    return y_min - pad, y_max + pad


def hide_y_axis(ax: Any) -> None:
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
    ax.set_ylabel("")
    ax.yaxis.set_visible(False)
    if "left" in ax.spines:
        ax.spines["left"].set_visible(False)
    if "right" in ax.spines:
        ax.spines["right"].set_visible(False)


def offset_tick_label(ax: Any, label_text: str = "during", dy: float = -0.08) -> None:
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == label_text:
            x_pos, y_pos = tick_label.get_position()
            tick_label.set_y(y_pos + dy)
            break


def compute_trace_y_limits(trace_specs: Sequence[TracePanelSpec], percentiles: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    values: List[np.ndarray] = []
    for spec in trace_specs:
        mean_trace = np.asarray(spec.summary.get("mean_trace", []), dtype=float)
        if mean_trace.size == 0:
            continue
        lower = mean_trace.copy()
        upper = mean_trace.copy()
        std_trace = np.asarray(spec.summary.get("std_trace", []), dtype=float)
        if std_trace.size == mean_trace.size:
            lower = mean_trace - std_trace
            upper = mean_trace + std_trace
        values.extend([lower, upper])
    if not values:
        return None
    stacked = np.concatenate([value[np.isfinite(value)] for value in values if value.size])
    if stacked.size == 0:
        return None
    if percentiles is not None:
        lower_q, upper_q = percentiles
        y_min = float(np.nanpercentile(stacked, lower_q))
        y_max = float(np.nanpercentile(stacked, upper_q))
    else:
        y_min = float(np.nanmin(stacked))
        y_max = float(np.nanmax(stacked))
    pad = max(0.08 * max(y_max - y_min, 1e-6), 0.02)
    return y_min - pad, y_max + pad


def draw_mean_image(ax: Any, spec: MeanImageSpec) -> None:
    plot_roi_overlays(
        ax,
        spec.mean_img,
        [dict(item, coords=item["coords"]) for item in spec.overlays],
        spec.title,
        title_pad=1.6,
        title_fontsize=PANEL_TITLE_SIZE,
    )
    ax.title.set_color(spec.color)


def draw_trace_panel(ax: Any, spec: TracePanelSpec, y_limits: Optional[Tuple[float, float]] = None, show_y_axis: bool = True, show_title: bool = True) -> None:
    mean_trace = np.asarray(spec.summary.get("mean_trace", []), dtype=float)
    std_trace = np.asarray(spec.summary.get("std_trace", []), dtype=float)
    t = np.asarray(spec.summary.get("t", []), dtype=float)
    if mean_trace.size == 0:
        ax.text(0.5, 0.5, f"No data for {spec.title}", transform=ax.transAxes, ha="center", va="center", fontsize=PANEL_NOTE_SIZE)
        ax.set_axis_off()
        return
    if t.size != mean_trace.size:
        if t.size > 1:
            t = np.linspace(float(t[0]), float(t[-1]), int(mean_trace.size))
        else:
            t = np.arange(mean_trace.size, dtype=float)
    ax.plot(t, mean_trace, color=spec.color, linewidth=1.8)
    fill_band = std_trace if std_trace.size == mean_trace.size else np.asarray([], dtype=float)
    if fill_band.size == mean_trace.size:
        ax.fill_between(t, mean_trace - fill_band, mean_trace + fill_band, color="#D0D0D0", alpha=0.55, linewidth=0)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=0.9)
    stimulus_end_s = as_float(spec.summary.get("stimulus_end_s"))
    if stimulus_end_s is not None and np.isfinite(stimulus_end_s):
        ax.axvspan(0.0, float(stimulus_end_s), color="#D0D0D0", alpha=0.25, linewidth=0)
    if t.size:
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    if y_limits is not None:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    if show_title:
        ax.set_title(spec.title, fontsize=PANEL_TITLE_SIZE, color=spec.color, pad=1.2)
    else:
        ax.set_title("")
    ax.set_xlabel(spec.x_label, fontsize=PANEL_LABEL_SIZE, labelpad=2.0)
    if show_y_axis:
        ax.set_ylabel(spec.y_label, fontsize=PANEL_LABEL_SIZE, labelpad=4.0)
        set_sparse_numeric_ticks(ax, axis="both", nbins=4)
        ax.tick_params(axis="y", labelsize=PANEL_FONT_SIZE, pad=1.5)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
        ax.yaxis.set_visible(False)
        set_sparse_numeric_ticks(ax, axis="x", nbins=4)
    ax.tick_params(axis="x", labelsize=PANEL_FONT_SIZE, pad=2.0, length=6, width=0.9)


def draw_movie_boxplot(ax: Any, spec: BoxplotSpec, y_limits: Optional[Tuple[float, float]] = None, show_y_axis: bool = True) -> None:
    baseline_values, stimulus_values, blank_values = get_category_trial_values(spec.summary)
    box_records: List[Tuple[np.ndarray, float, str, str, str]] = []
    comparison_groups: List[Tuple[str, float, float]] = []
    if spec.category_name == "blank":
        series = [
            (baseline_values, 1.0, "pre", lighten_color(spec.color, 0.65), spec.color),
            (stimulus_values, 2.0, "during", lighten_color(spec.color, 0.35), spec.color),
        ]
        comparison_groups.append(("paired", 1.0, 2.0))
    else:
        series = [
            (baseline_values, 1.0, "pre", lighten_color(spec.color, 0.65), spec.color),
            (stimulus_values, 2.0, "during", lighten_color(spec.color, 0.20), spec.color),
            (blank_values, 3.0, "blank", "#D5D5D5", "#777777"),
        ]
        comparison_groups.append(("paired", 1.0, 2.0))
        comparison_groups.append(("blank", 2.0, 3.0))
    for values, xpos, label, facecolor, edgecolor in series:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        box_records.append((values, xpos, label, facecolor, edgecolor))

    if not box_records:
        ax.text(0.5, 0.5, f"No data for {spec.title}", transform=ax.transAxes, ha="center", va="center", fontsize=PANEL_NOTE_SIZE)
        ax.set_axis_off()
        return

    data = [record[0] for record in box_records]
    positions = [record[1] for record in box_records]
    labels = [record[2] for record in box_records]
    box_width = 0.62 if spec.category_name == "blank" else 0.50
    bp = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True, showfliers=False)
    for patch, (_, _, _, facecolor, edgecolor) in zip(bp["boxes"], box_records):
        patch.set_facecolor(facecolor)
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(1.4)
        patch.set_alpha(0.95)
    for whisker in bp["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.5)

    y_values = np.concatenate([record[0] for record in box_records if record[0].size])
    y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
    y_min = float(np.nanmin(y_values)) if y_values.size else 0.0
    y_span = max(y_max - y_min, 1e-6)
    bracket_y = y_max + 0.08 * y_span
    bracket_step = 0.10 * y_span
    upper_limit = bracket_y + max(len(comparison_groups) - 1, 0) * bracket_step + 0.18 * y_span
    lower_limit = y_min - 0.12 * y_span
    ax.set_ylim(lower_limit, upper_limit)
    for comparison_index, (comparison_kind, x1, x2) in enumerate(comparison_groups):
        stats_block = spec.summary.get("stats", {})
        comparison = stats_block.get("paired_pre_vs_stimulus") if comparison_kind == "paired" else stats_block.get("stimulus_vs_blank")
        if not comparison or not comparison.get("significant"):
            continue
        add_significance_bracket(
            ax,
            x1,
            x2,
            bracket_y + comparison_index * bracket_step,
            comparison.get("star", "*"),
            spec.color if comparison_kind == "paired" else "#777777",
        )

    if spec.category_name == "blank":
        ax.set_xlim(0.5, 2.5)
    else:
        ax.set_xlim(0.5, 3.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=PANEL_FONT_SIZE, rotation=0)
    offset_tick_label(ax, "during", dy=-0.16)
    if show_y_axis:
        ax.tick_params(axis="y", labelsize=PANEL_FONT_SIZE, pad=1.5)
        ax.set_ylabel(spec.y_label, fontsize=PANEL_LABEL_SIZE, labelpad=4.0)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
        ax.yaxis.set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.18)


def draw_soma_boxplot(ax: Any, spec: BoxplotSpec, y_limits: Optional[Tuple[float, float]] = None, show_y_axis: bool = True) -> None:
    paired_baseline_values = np.asarray(spec.summary.get("paired_baseline_values", []), dtype=float)
    paired_stimulus_values = np.asarray(spec.summary.get("paired_stimulus_values", []), dtype=float)
    if not paired_baseline_values.size or not paired_stimulus_values.size:
        ax.text(0.5, 0.5, f"No data for {spec.title}", transform=ax.transAxes, ha="center", va="center", fontsize=PANEL_NOTE_SIZE)
        ax.set_axis_off()
        return
    bp = ax.boxplot([paired_baseline_values, paired_stimulus_values], positions=[1.0, 2.0], widths=0.55, patch_artist=True, showfliers=False)
    box_records = [
        (1.0, lighten_color(spec.color, 0.70), spec.color),
        (2.0, lighten_color(spec.color, 0.35), spec.color),
    ]
    for patch, (_, facecolor, edgecolor) in zip(bp["boxes"], box_records):
        patch.set_facecolor(facecolor)
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(1.4)
        patch.set_alpha(0.95)
    for whisker in bp["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.5)
    y_values = np.concatenate([paired_baseline_values, paired_stimulus_values])
    y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
    y_min = float(np.nanmin(y_values)) if y_values.size else 0.0
    y_span = max(y_max - y_min, 1e-6)
    bracket_y = y_max + 0.08 * y_span
    ax.set_ylim(y_min - 0.12 * y_span, bracket_y + 0.18 * y_span)
    stats_block = spec.summary.get("stats", {}).get("paired_pre_vs_stimulus") or {}
    if stats_block.get("significant"):
        add_significance_bracket(ax, 1.0, 2.0, bracket_y, stats_block.get("star", "*"), spec.color)
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(["pre", "during"], fontsize=PANEL_FONT_SIZE, rotation=0)
    offset_tick_label(ax, "during", dy=-0.16)
    if show_y_axis:
        ax.tick_params(axis="y", labelsize=PANEL_FONT_SIZE, pad=1.5)
        ax.set_ylabel(spec.y_label, fontsize=PANEL_LABEL_SIZE, labelpad=4.0)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
        ax.yaxis.set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.18)


def draw_retino_panel(ax: Any, spec: RetinoSpec) -> None:
    retino_map = build_retinotopy_response_map(spec.summary.get("response_points", []))
    if not retino_map.get("available"):
        ax.text(0.5, 0.5, "No retinotopy map available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=PANEL_NOTE_SIZE)
        ax.set_axis_off()
        return

    response_matrix = np.asarray(retino_map.get("response_matrix", []), dtype=float)
    x_values = np.asarray(retino_map.get("x_values", []), dtype=float)
    y_values = np.asarray(retino_map.get("y_values", []), dtype=float)

    if response_matrix.size == 0 or x_values.size == 0 or y_values.size == 0:
        ax.text(0.5, 0.5, "No retinotopy map available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=PANEL_NOTE_SIZE)
        ax.set_axis_off()
        return

    x_step = float(np.nanmin(np.diff(x_values))) if x_values.size > 1 else 1.0
    y_step = float(np.nanmin(np.diff(y_values))) if y_values.size > 1 else 1.0

    extent = [
        float(x_values.min() - x_step / 2.0),
        float(x_values.max() + x_step / 2.0),
        float(y_values.min() - y_step / 2.0),
        float(y_values.max() + y_step / 2.0),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "soma_retino",
        ["#F5FBF7", spec.color]
    )

    im = ax.imshow(
        response_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        extent=extent,
    )

    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    ax.set_xticklabels([f"{value:g}" for value in x_values], fontsize=PANEL_FONT_SIZE)
    ax.set_yticklabels([f"{value:g}" for value in y_values], fontsize=PANEL_FONT_SIZE)
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", pad=1.0)
    ax.set_title(spec.title, fontsize=PANEL_TITLE_SIZE, color=spec.color, pad=1.0)
    ax.set_xlabel(str(r"x"),
                  fontsize=PANEL_LABEL_SIZE, labelpad=1.0)
    ax.set_ylabel(str(r"y"),
                  fontsize=PANEL_LABEL_SIZE, labelpad=0.5)

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        fraction=0.022,
        pad=0.008,
        location="left",
    )
    cbar.set_label("dF/F", fontsize=PANEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=PANEL_FONT_SIZE)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    # IMPORTANT: shrink AFTER colorbar is created
    pos = ax.get_position()
    scale = 0.75
    new_width = pos.width * scale

    ax.set_position([
        pos.x0 + (pos.width - new_width) * 0.20,  # keep slightly left
        pos.y0,
        new_width,
        pos.height,
    ])

def render_row(fig: Any, outer: Any, row_index: int, row: RowSpec) -> None:
    if row.retino is not None:
        axes = [
            fig.add_subplot(outer[row_index, 0:2]),  # soma image
            None,
            fig.add_subplot(outer[row_index, 4]),    # grating onset, aligned below
            fig.add_subplot(outer[row_index, 5]),    # boxplot, aligned below
        ]

        retino_grid = outer[row_index, 2:4].subgridspec(
            1,
            2,
            width_ratios=[0.78, 0.22],  # retino + empty spacer
            wspace=0.0,
        )
        axes[1] = fig.add_subplot(retino_grid[0, 0])
        trace_limits = compute_trace_y_limits(row.trace_panels)
        box_limits = compute_box_y_limits(row.box_panels)
        if trace_limits is not None and box_limits is not None:
            shared_limits = (min(trace_limits[0], box_limits[0]), max(trace_limits[1], box_limits[1]))
        else:
            shared_limits = trace_limits or box_limits
        soma_trace_limits = trace_limits or shared_limits
        if soma_trace_limits is not None:
            soma_trace_limits = (float(soma_trace_limits[0]), SOMA_GRATING_TRACE_YMAX)
        else:
            soma_trace_limits = (0.0, SOMA_GRATING_TRACE_YMAX)

        draw_mean_image(axes[0], row.mean_image)
        draw_retino_panel(axes[1], row.retino)
        draw_trace_panel(axes[2], row.trace_panels[0], y_limits=soma_trace_limits, show_y_axis=True, show_title=True)
        draw_soma_boxplot(axes[3], row.box_panels[0], y_limits=shared_limits, show_y_axis=False)
    else:
        spans = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        axes = [fig.add_subplot(outer[row_index, start:end]) for start, end in spans]
        trace_percentiles = (12.0, 88.0) if row.name.lower() == "apical" else None
        box_percentiles = (15.0, 85.0) if row.name.lower() == "apical" else None
        trace_limits = compute_trace_y_limits(row.trace_panels, percentiles=trace_percentiles)
        box_limits = compute_box_y_limits(row.box_panels, percentiles=box_percentiles)
        if trace_limits is not None and box_limits is not None:
            shared_limits = (min(trace_limits[0], box_limits[0]), max(trace_limits[1], box_limits[1]))
        else:
            shared_limits = trace_limits or box_limits
        draw_mean_image(axes[0], row.mean_image)
        draw_trace_panel(axes[1], row.trace_panels[0], y_limits=shared_limits, show_y_axis=True, show_title=True)
        draw_movie_boxplot(axes[2], row.box_panels[0], y_limits=shared_limits, show_y_axis=False)
        draw_trace_panel(axes[3], row.trace_panels[1], y_limits=shared_limits, show_y_axis=False, show_title=False)
        draw_movie_boxplot(axes[4], row.box_panels[1], y_limits=shared_limits, show_y_axis=False)


def build_poster_spec(group_name: str, experiments: Dict[str, Any], group: Dict[str, Any]) -> PosterSpec:
    basal_expids = list(group.get("basal_expids", []))
    apical_expids = list(group.get("apical_expids", []))
    soma_expids = list(group.get("soma_expids", []))

    basal_movie_summaries = build_movie_experiment_summaries(experiments, basal_expids, "basal", 0.5)
    apical_movie_summaries = build_movie_experiment_summaries(experiments, apical_expids, "apical", 0.5)
    soma_summaries = build_soma_experiment_summaries(experiments, soma_expids, 1.0, 2.0)

    basal_summary = pool_movie_session_summaries(basal_movie_summaries)
    apical_summary = pool_movie_session_summaries(apical_movie_summaries)
    soma_summary = pool_soma_session_summaries(soma_summaries)

    basal_exp = pick_representative_experiment(experiments, basal_expids)
    apical_exp = pick_representative_experiment(experiments, apical_expids)
    soma_exp = pick_representative_experiment(experiments, soma_expids)
    if basal_exp is None or apical_exp is None or soma_exp is None:
        raise RuntimeError("Could not resolve representative experiments for one or more poster rows")

    basal_row = RowSpec(
        name="Basal",
        color=TOP_ROW_COLORS["basal"],
        mean_image=build_mean_image_spec(basal_exp, TOP_ROW_COLORS["basal"], "Basal Dendrites"),
        trace_panels=[
            build_trace_panel("Blank onset", basal_summary.get("blank", {}), TOP_ROW_COLORS["basal"], "movie", "Time (s)"),
            build_trace_panel("Movies onset", basal_summary.get("movies", {}), TOP_ROW_COLORS["basal"], "movie", "Time (s)"),
        ],
        box_panels=[
            build_movie_box_panel("Blank trial distributions", basal_summary.get("blank", {}), TOP_ROW_COLORS["basal"], "blank"),
            build_movie_box_panel("Movies trial distributions", basal_summary.get("movies", {}), TOP_ROW_COLORS["basal"], "movies"),
        ],
    )

    apical_row = RowSpec(
        name="Apical",
        color=TOP_ROW_COLORS["apical"],
        mean_image=build_mean_image_spec(apical_exp, TOP_ROW_COLORS["apical"], "Apical Dendrites"),
        trace_panels=[
            build_trace_panel("Blank onset", apical_summary.get("blank", {}), TOP_ROW_COLORS["apical"], "movie", "Time (s)"),
            build_trace_panel("Movies onset", apical_summary.get("movies", {}), TOP_ROW_COLORS["apical"], "movie", "Time (s)"),
        ],
        box_panels=[
            build_movie_box_panel("Blank trial distributions", apical_summary.get("blank", {}), TOP_ROW_COLORS["apical"], "blank"),
            build_movie_box_panel("Movies trial distributions", apical_summary.get("movies", {}), TOP_ROW_COLORS["apical"], "movies"),
        ],
    )

    soma_row = RowSpec(
        name="Soma",
        color=TOP_ROW_COLORS["soma"],
        mean_image=build_soma_mean_image_spec(soma_exp, "Soma", TOP_ROW_COLORS["soma"]),
        trace_panels=[
            build_trace_panel("Grating onset", soma_summary, TOP_ROW_COLORS["soma"], "soma", "Time (s)"),
        ],
        box_panels=[
            build_soma_box_panel("Grating trial distributions", soma_summary, TOP_ROW_COLORS["soma"]),
        ],
        retino=build_retino_spec("Rapid retinotopy", soma_summary, TOP_ROW_COLORS["soma"]),
    )

    return PosterSpec(group_name=group_name, rows=[soma_row, apical_row, basal_row])


def render_poster_spec(spec: PosterSpec, output_path: Path) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required to render the poster figure")

    old_note_size = getattr(sday, "POSTER_NOTE_SIZE", None)
    old_title_size = getattr(sday, "POSTER_TITLE_SIZE", None)
    try:
        if old_note_size is not None:
            sday.POSTER_NOTE_SIZE = PANEL_NOTE_SIZE
        if old_title_size is not None:
            sday.POSTER_TITLE_SIZE = PANEL_TITLE_SIZE
        fig = plt.figure(figsize=(cm_to_inches(FIGURE_WIDTH_CM), cm_to_inches(FIGURE_HEIGHT_CM)))
        outer = fig.add_gridspec(
            3,
            6,
            width_ratios=[1.35, 0.1, 1.08, 0.86, 1.08, 0.86],
            height_ratios=[0.84, 0.80, 0.80],
            hspace=0.60,
            wspace=0.3,
        )

        for row_index, row in enumerate(spec.rows):
            render_row(fig, outer, row_index, row)

        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.975)
        return save_figure(fig, output_path, dpi=300, extra_formats=())
    finally:
        if old_note_size is not None:
            sday.POSTER_NOTE_SIZE = old_note_size
        if old_title_size is not None:
            sday.POSTER_TITLE_SIZE = old_title_size


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poster-ready visual response figure compositor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the visual-response config JSON.")
    parser.add_argument("--repo-base", type=Path, default=None, help="Override the processed repository base.")
    parser.add_argument("--remote-repo-base", type=Path, default=None, help="Override the raw remote repository base.")
    parser.add_argument("--output-dir", type=Path, default=POSTER_OUTPUT_DIR, help="Directory for SVG output.")
    return parser


def build_group_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    group_map = normalize_soma_group_map(config.get("soma_group_map"))
    if not group_map:
        raise SystemExit("soma_group_map is required")
    return group_map


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    file_config = load_json_config_file(args.config if args.config and args.config.exists() else None)
    cli_config = {
        "repo_base": str(args.repo_base) if args.repo_base is not None else None,
        "remote_repo_base": str(args.remote_repo_base) if args.remote_repo_base is not None else None,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
    }
    config = merge_config(cli_config, file_config)

    repo_base = Path(config.get("repo_base") or f"/home/rubencorreia/data/Repository")
    remote_repo_base = Path(config.get("remote_repo_base") or DEFAULT_REMOTE_REPO_BASE)
    output_dir = Path(config.get("output_dir") or POSTER_OUTPUT_DIR)
    basal_expids = movie_compact_list(config.get("basal_expids"))
    apical_expids = movie_compact_list(config.get("apical_expids"))
    soma_expids = movie_compact_list(config.get("soma_expids"))

    if not basal_expids:
        raise SystemExit("basal_expids is required")
    if not apical_expids:
        raise SystemExit("apical_expids is required")
    if not soma_expids:
        raise SystemExit("soma_expids is required")

    group_map = build_group_specs(config)
    validation_alerts = validate_group_map(group_map, basal_expids, apical_expids, soma_expids)
    if validation_alerts:
        raise SystemExit("\n".join(validation_alerts))

    experiments, load_alerts = load_experiments(repo_base, basal_expids, apical_expids, soma_expids, 0, remote_repo_base=remote_repo_base)
    _ = load_alerts

    for group in group_map:
        group_name = str(group.get("label") or group.get("name") or "group")
        poster_spec = build_poster_spec(group_name, experiments, group)
        group_output_dir = ensure_dir(Path(output_dir))
        out_path = group_output_dir / f"{safe_filename_component(group_name)}_poster.svg"
        render_poster_spec(poster_spec, out_path)


if __name__ == "__main__":
    main()
