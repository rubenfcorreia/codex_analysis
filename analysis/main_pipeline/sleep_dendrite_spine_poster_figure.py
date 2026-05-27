#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception:  # pragma: no cover - matplotlib is required for the real run
    plt = None

from poster_plotting import (
    POSTER_DPI,
    POSTER_FONT_SIZE,
    POSTER_LABEL_SIZE,
    POSTER_LEGEND_SIZE,
    POSTER_NOTE_SIZE,
    POSTER_TITLE_SIZE,
    SVGFigure,
    SVGPanel,
    configure_poster_matplotlib,
    save_figure,
    set_sparse_numeric_ticks,
)

from sleep_dendrite_spine_pipeline import (
    REPORT_SIGNIFICANCE_ALPHA,
    _draw_boxplot_significance_annotations,
    _mixed_model_response_payload,
    _mixed_model_term_component_label,
    _mixed_model_term_interaction_value_label,
    _mixed_model_term_kind,
    _mixed_model_term_value_label,
    _pad_boxplot_ylim,
    _set_boxplot_colors,
    as_float,
    basal_apical_comparison,
    build_state_summary_gallery_results,
    build_mixed_model_table,
    canonical_state_label,
    color_state_tick_labels,
    ensure_dir,
    flatten_state_summary_values,
    format_requested_state_label,
    load_cached_analysis_table,
    load_npz_cache,
    mixed_model_design_row,
    mixed_model_response_display_label,
    padded_value_limits,
    resolve_analysis_state_selections,
    run_mixed_model_analysis,
    run_mixed_model_family,
    summarize_state_values,
    selected_mixed_model_state_labels,
    set_requested_state_ticks,
    state_display_color,
    state_summary_y_limits,
)

if plt is not None:
    configure_poster_matplotlib()


DEFAULT_CACHE_PATH = ROOT_DIR / "results" / "main_pipeline" / "cache" / "sleep_dendrite_spine_cache.npz"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "poster_ready"
DEFAULT_OUTPUT_STEM = "dendrite_state_mixed_model_poster_ready"
DEFAULT_WIDTH_CM = 35.0
DEFAULT_HEIGHT_CM = 29.0
DEFAULT_GAP = 20.0
DEFAULT_MARGIN = 20.0

EVENT_FREQUENCY_METRICS = {
    "dendrite_event_frequency_per_min",
    "spine_event_frequency_per_min",
    "coincident_event_frequency_per_min",
    "noncoincident_event_frequency_per_min",
}
EVENT_FREQUENCY_FOREST_SOURCE = (
    ROOT_DIR
    / "results"
    / "main_pipeline"
    / "figures"
    / "mixed_model"
    / "mixed_model_forest_components"
    / "mixed_model_forest_03_dendrite_event_frequency_per_min.svg"
)


def cm_to_inch(value_cm: float) -> float:
    return float(value_cm) / 2.54


def load_cache(path: Path) -> Dict[str, Any]:
    cache = load_npz_cache(path)
    if not isinstance(cache, dict):
        raise TypeError(f"Expected a cache dictionary from {path}")
    return cache


def _svg_dimension_to_float(value: Any) -> float:
    text = str(value).strip().lower().replace("px", "").replace("pt", "")
    try:
        return float(text)
    except Exception:
        return float("nan")


def set_svg_physical_size(svg_path: Path, width_cm: float) -> None:
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    width = _svg_dimension_to_float(root.attrib.get("width"))
    height = _svg_dimension_to_float(root.attrib.get("height"))
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0:
        return
    aspect = height / width
    root.attrib["width"] = f"{float(width_cm):.4f}cm"
    root.attrib["height"] = f"{float(width_cm) * aspect:.4f}cm"
    tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)


def _pad_boxplot_limits(
    data_arrays: Sequence[np.ndarray],
    value_limit: Optional[Tuple[float, float]] = None,
    pad_fraction: float = 0.14,
) -> Optional[Tuple[float, float]]:
    finite: List[np.ndarray] = []
    for arr in data_arrays:
        values = np.asarray(arr, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if value_limit is not None:
        low, high = map(float, value_limit)
    elif finite:
        merged = np.concatenate(finite)
        low = float(np.nanmin(merged))
        high = float(np.nanmax(merged))
    else:
        return None
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        pad = max(0.1, abs(low) * pad_fraction + 0.05)
    else:
        pad = max((high - low) * pad_fraction, 0.05)
    return low - pad, high + pad


def _compact_term_value_label(term: str) -> str:
    kind = _mixed_model_term_kind(term)
    if kind in {"state", "compartment"}:
        return _mixed_model_term_interaction_value_label(term)
    return _mixed_model_term_component_label(term)


def poster_mixed_model_term_label(term: str) -> str:
    if term == "Intercept":
        return term
    if ":" in term:
        return " x ".join(_compact_term_value_label(part) for part in term.split(":"))
    if _mixed_model_term_kind(term) == "state":
        return _mixed_model_term_component_label(term)
    return _compact_term_value_label(term)


def ensure_mixed_model_forest_response(
    mixed_model_results: Dict[str, Any],
    cache: Dict[str, Any],
    response: str,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
) -> Dict[str, Any]:
    summary_rows = mixed_model_results.get("summary_rows", {})
    if isinstance(summary_rows, dict) and summary_rows.get(response):
        return mixed_model_results

    table_entry = load_cached_analysis_table(
        cache,
        "mixed_model_table",
        expected_meta={"analysis_unit": str(cache.get("analysis_unit", "day"))},
        rebuild=bool(cache.get("config", {}).get("analysis_tables_rebuild")) or bool(cache.get("config", {}).get("rebuild")),
    )
    if table_entry is not None:
        table_rows = list(table_entry.get("table_rows", []))
    else:
        table_rows, _ = build_mixed_model_table(cache)

    selected_state_order = [canonical_state_label(state) for state in state_comparison_states if state is not None and str(state).strip()]
    selected_basal_apical_states = [
        canonical_state_label(state)
        for state in basal_apical_states
        if canonical_state_label(state) in set(selected_state_order)
    ]
    contrast_specs = [
        {"kind": "state_pair", "state_a": state_a, "state_b": state_b}
        for state_a, state_b in combinations(selected_state_order, 2)
    ]
    contrast_specs.extend({"kind": "basal_apical", "state": state} for state in selected_basal_apical_states)

    result = run_mixed_model_family(
        table_rows,
        response,
        "selected_state",
        contrast_specs,
        shuffle_n,
        state_order=selected_state_order,
        state_filter=selected_state_order,
    )
    if not result.get("summary_rows") or result.get("design") is None:
        return mixed_model_results

    merged = dict(mixed_model_results)
    merged_summary_rows = dict(mixed_model_results.get("summary_rows", {})) if isinstance(mixed_model_results.get("summary_rows", {}), dict) else {}
    merged_summary_rows[response] = list(result.get("summary_rows", []))
    merged["summary_rows"] = merged_summary_rows

    merged_designs = dict(mixed_model_results.get("designs", {})) if isinstance(mixed_model_results.get("designs", {}), dict) else {}
    merged_designs[response] = dict(result.get("design", {}))
    merged["designs"] = merged_designs

    merged_equations = dict(mixed_model_results.get("model_equations", {})) if isinstance(mixed_model_results.get("model_equations", {}), dict) else {}
    merged_equations[response] = result.get("equation")
    merged["model_equations"] = merged_equations

    merged_tested_terms = dict(mixed_model_results.get("tested_terms", {})) if isinstance(mixed_model_results.get("tested_terms", {}), dict) else {}
    merged_tested_terms[response] = list(result.get("tested_terms", []))
    merged["tested_terms"] = merged_tested_terms

    merged_tested_contrasts = dict(mixed_model_results.get("tested_contrasts", {})) if isinstance(mixed_model_results.get("tested_contrasts", {}), dict) else {}
    merged_tested_contrasts[response] = list(result.get("tested_contrasts", []))
    merged["tested_contrasts"] = merged_tested_contrasts

    merged["contrast_rows"] = list(mixed_model_results.get("contrast_rows", [])) + list(result.get("contrast_rows", []))

    selection = dict(mixed_model_results.get("selection", {})) if isinstance(mixed_model_results.get("selection", {}), dict) else {}
    selection.setdefault("state_comparison_states", list(selected_state_order))
    selection.setdefault("basal_apical_states", list(selected_basal_apical_states))
    merged["selection"] = selection
    merged["available"] = bool(result.get("design"))
    merged.setdefault("alerts", list(mixed_model_results.get("alerts", [])))
    merged["p_value_source"] = result.get("p_value_source", merged.get("p_value_source"))
    merged["p_value_source_requested"] = result.get("p_value_source_requested", merged.get("p_value_source_requested"))
    return merged


def _svg_text_kind_label(text: str) -> Optional[str]:
    normalized = " ".join(str(text).split())
    if normalized == "Intercept":
        return "intercept"
    lower = normalized.lower()
    if lower.startswith("state:"):
        return "state"
    if lower.startswith("compartment:"):
        return "compartment"
    if " × " in normalized or " x " in normalized:
        return "interaction"
    return None


def _recolor_svg_text_by_kind(svg_root: Any) -> None:
    term_colors = {
        "intercept": "#7f7f7f",
        "state": "#1f77b4",
        "compartment": "#2ca02c",
        "interaction": "#ff7f0e",
    }
    for elem in svg_root.iter():
        if not str(elem.tag).endswith("text") or not elem.text:
            continue
        kind = _svg_text_kind_label(elem.text)
        if kind is None:
            continue
        fill = term_colors[kind]
        style = str(elem.attrib.get("style", ""))
        if "fill:" in style:
            style = re.sub(r"fill:\s*[^;]+", f"fill: {fill}", style)
        elif style:
            style = f"{style.rstrip('; ')}; fill: {fill}"
        else:
            style = f"fill: {fill}"
        style = re.sub(r"font-weight:\s*(?:700|bold)", "font-weight: normal", style)
        style = re.sub(r"font:\s*(?:700|bold)\s+", "font: ", style)
        elem.attrib["style"] = style


def _prefix_svg_ids(svg_root: Any, prefix: str) -> None:
    id_map: Dict[str, str] = {}
    for elem in svg_root.iter():
        old_id = elem.attrib.get("id")
        if not old_id:
            continue
        new_id = f"{prefix}{old_id}"
        id_map[old_id] = new_id
        elem.attrib["id"] = new_id
    if not id_map:
        return
    for elem in svg_root.iter():
        for attr, value in list(elem.attrib.items()):
            if not isinstance(value, str) or not value:
                continue
            updated = value
            for old_id, new_id in id_map.items():
                updated = updated.replace(f"url(#{old_id})", f"url(#{new_id})")
                updated = updated.replace(f"#{old_id}", f"#{new_id}")
            if updated != value:
                elem.attrib[attr] = updated


def _extract_axes_bbox_from_svg_group(axes_group: Any) -> Optional[Tuple[float, float, float, float]]:
    for child in list(axes_group):
        child_id = str(child.attrib.get("id", ""))
        if not child_id.startswith("patch_"):
            continue
        for path_elem in child.iter():
            if not str(path_elem.tag).endswith("path"):
                continue
            d = str(path_elem.attrib.get("d", ""))
            numbers = [float(token) for token in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)]
            if len(numbers) < 8:
                continue
            xs = numbers[0::2]
            ys = numbers[1::2]
            return min(xs), min(ys), max(xs), max(ys)
    return None


def inject_svg_panel(svg_path: Path, panel_path: Path, axes_id: str = "axes_4") -> bool:
    if not svg_path.exists() or not panel_path.exists():
        return False
    outer_tree = ET.parse(str(svg_path))
    outer_root = outer_tree.getroot()
    target_axes = None
    for elem in outer_root.iter():
        if str(elem.tag).endswith("g") and elem.attrib.get("id") == axes_id:
            target_axes = elem
            break
    if target_axes is None:
        return False
    bbox = _extract_axes_bbox_from_svg_group(target_axes)
    if bbox is None:
        return False
    panel_tree = ET.parse(str(panel_path))
    panel_root = panel_tree.getroot()
    _recolor_svg_text_by_kind(panel_root)
    _prefix_svg_ids(panel_root, "eventfreq_")
    x0, y0, x1, y1 = bbox
    panel_root.attrib["x"] = f"{x0:.4f}"
    panel_root.attrib["y"] = f"{y0:.4f}"
    panel_root.attrib["width"] = f"{(x1 - x0):.4f}"
    panel_root.attrib["height"] = f"{(y1 - y0):.4f}"
    panel_root.attrib["preserveAspectRatio"] = "xMidYMid meet"
    for child in list(target_axes):
        target_axes.remove(child)
    target_axes.append(panel_root)
    outer_tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
    return True


def build_basal_apical_comparisons(
    cache: Dict[str, Any],
    metric_key: str,
    state_order: Sequence[str],
    shuffle_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for state in state_order:
        comparison = basal_apical_comparison(cache, metric_key, state, shuffle_n)
        rows.append(comparison)
    return rows


def draw_state_summary_compartment_comparison_panel(
    ax: Any,
    *,
    metric_key: str,
    metric_title: str,
    basal_summary: Dict[str, Dict[str, List[float]]],
    apical_summary: Dict[str, Dict[str, List[float]]],
    state_order: Sequence[str],
    y_limit: Optional[Tuple[float, float]] = None,
    comparison_rows: Optional[Sequence[Dict[str, Any]]] = None,
    horizontal: bool = False,
    show_legend: bool = True,
    show_counts: bool = False,
) -> None:
    rng = np.random.default_rng(7)
    all_data: List[np.ndarray] = []
    compartment_counts: Dict[str, Dict[str, int]] = {state: {"basal": 0, "apical": 0} for state in state_order}
    compartment_specs = [
        ("basal", basal_summary, "#4C72B0", -0.18),
        ("apical", apical_summary, "#DD8452", 0.18),
    ]
    for compartment, summary, color, offset in compartment_specs:
        positions: List[float] = []
        data: List[np.ndarray] = []
        for idx, state in enumerate(state_order, start=1):
            arr = flatten_state_summary_values(summary.get(state, {}))
            if arr.size:
                positions.append(float(idx) + offset)
                data.append(arr)
                compartment_counts[state][compartment] += int(arr.size)
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False, vert=not horizontal)
        _set_boxplot_colors(bp, [color] * len(data))
        for pos, arr in zip(positions, data):
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            if horizontal:
                ax.scatter(arr, np.full(arr.size, pos) + jitter, s=14, alpha=0.48, color=color, edgecolor="none")
            else:
                ax.scatter(np.full(arr.size, pos) + jitter, arr, s=14, alpha=0.48, color=color, edgecolor="none")
        all_data.extend(data)

    if horizontal:
        set_requested_state_ticks(ax, state_order, axis="y")
        if show_counts:
            x0, x1 = ax.get_xlim()
            x_range = x1 - x0
            count_x = x1 + max(0.02 * x_range, 0.05)
            for compartment, summary, color, offset in compartment_specs:
                for idx, state in enumerate(state_order, start=1):
                    arr = flatten_state_summary_values(summary.get(state, {}))
                    if not arr.size:
                        continue
                    ax.text(
                        count_x,
                        float(idx) + offset,
                        f"n={arr.size}",
                        color=color,
                        fontsize=max(12, POSTER_NOTE_SIZE),
                        ha="left",
                        va="center",
                        clip_on=False,
                    )
        ax.set_ylabel("State", fontsize=max(15, POSTER_LABEL_SIZE - 3))
        ax.set_xlabel(
            "Events / min" if metric_key in EVENT_FREQUENCY_METRICS else "dF/F",
            fontsize=max(13, POSTER_LABEL_SIZE - 2),
        )
        ax.set_title(metric_title, fontsize=max(17, POSTER_TITLE_SIZE - 5), pad=1)
        x_limits = _pad_boxplot_limits(all_data, y_limit)
        if x_limits is not None:
            ax.set_xlim(x_limits)
        ax.tick_params(axis="x", labelsize=POSTER_FONT_SIZE)
        ax.grid(axis="x", alpha=0.25)
        comparison_subset = [
            {
                "x1": float(state_order.index(str(row.get("state"))) + 1 - 0.18),
                "x2": float(state_order.index(str(row.get("state"))) + 1 + 0.18),
                "shuffle_p": row.get("shuffle_p"),
            }
            for row in (comparison_rows or [])
            if str(row.get("comparison")) == "basal_vs_apical"
            and str(row.get("metric")) == metric_key
            and str(row.get("state")) in state_order
            and as_float(row.get("shuffle_p")) is not None
            and float(as_float(row.get("shuffle_p")) or 1.0) < REPORT_SIGNIFICANCE_ALPHA
        ]
        _draw_boxplot_significance_annotations(ax, comparison_subset, orientation="horizontal")
    else:
        set_requested_state_ticks(ax, state_order)
        ax.set_ylabel(
            "Events / min" if metric_key in EVENT_FREQUENCY_METRICS else "dF/F",
            fontsize=POSTER_LABEL_SIZE,
        )
        ax.set_title(metric_title, fontsize=max(17, POSTER_TITLE_SIZE - 5), pad=1)
        _pad_boxplot_ylim(ax, all_data, y_limit=y_limit)
        ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
        ax.grid(axis="y", alpha=0.25)
        comparison_subset = [
            {
                "x1": float(state_order.index(str(row.get("state"))) + 1 - 0.18),
                "x2": float(state_order.index(str(row.get("state"))) + 1 + 0.18),
                "shuffle_p": row.get("shuffle_p"),
            }
            for row in (comparison_rows or [])
            if str(row.get("comparison")) == "basal_vs_apical"
            and str(row.get("metric")) == metric_key
            and str(row.get("state")) in state_order
            and as_float(row.get("shuffle_p")) is not None
            and float(as_float(row.get("shuffle_p")) or 1.0) < REPORT_SIGNIFICANCE_ALPHA
        ]
        _draw_boxplot_significance_annotations(ax, comparison_subset)
    if show_legend:
        legend_handles = [
            Line2D([0], [0], color="#4C72B0", marker="s", linestyle="", markersize=8, label="Basal"),
            Line2D([0], [0], color="#DD8452", marker="s", linestyle="", markersize=8, label="Apical"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=POSTER_LEGEND_SIZE)


def draw_mixed_model_forest_panel(

    ax: Any,
    *,
    results: Dict[str, Any],
    response: str,
    model_key: str = "mixed_model",
    show_legend: bool = False,
) -> bool:
    design, rows = _mixed_model_response_payload(results, response, model_key=model_key)
    if design is None or not rows:
        return False

    term_lookup = {str(row.get("term")): dict(row) for row in rows}
    fixed_effect_names = [str(term) for term in design.get("fixed_effect_names", [])]
    selected_state_set = {canonical_state_label(state) for state in selected_mixed_model_state_labels(results)}
    display_terms: List[str] = []
    for term in fixed_effect_names:
        kind = _mixed_model_term_kind(term)
        if kind == "state":
            if canonical_state_label(_mixed_model_term_value_label(term)) not in selected_state_set:
                continue
        elif kind == "interaction":
            state_terms = [part for part in str(term).split(":") if part.startswith("state[")]
            if state_terms and not any(
                canonical_state_label(_mixed_model_term_value_label(part)) in selected_state_set for part in state_terms
            ):
                continue
        display_terms.append(term)

    if not display_terms:
        return False

    all_bounds: List[float] = []
    for term in display_terms:
        row = term_lookup.get(term)
        if row is None:
            continue
        estimate = as_float(row.get("estimate"))
        se = as_float(row.get("se"))
        if estimate is None or not np.isfinite(estimate):
            continue
        if se is None or not np.isfinite(se):
            all_bounds.append(float(estimate))
            continue
        ci = 1.96 * se
        all_bounds.extend([float(estimate - ci), float(estimate + ci)])
    all_bounds.append(0.0)
    x_limits = padded_value_limits(all_bounds)

    y_positions = np.arange(len(display_terms))[::-1]
    significance_map: Dict[str, bool] = {}
    term_colors = {
        "intercept": "#7f7f7f",
        "state": "#1f77b4",
        "compartment": "#2ca02c",
        "interaction": "#ff7f0e",
        "covariate": "#9467bd",
    }
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label)
        for label, color in [
            ("intercept", term_colors["intercept"]),
            ("state", term_colors["state"]),
            ("compartment", term_colors["compartment"]),
            ("interaction", term_colors["interaction"]),
            ("covariate", term_colors["covariate"]),
        ]
    ]
    legend_handles.append(
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=10, label="p < 0.05")
    )

    for y_pos, term in zip(y_positions, display_terms):
        row = term_lookup.get(term, {})
        estimate = as_float(row.get("estimate"))
        se = as_float(row.get("se"))
        p_value = as_float(row.get("p_value"))
        significant = bool(p_value is not None and np.isfinite(p_value) and p_value < REPORT_SIGNIFICANCE_ALPHA)
        significance_map[term] = significant
        if estimate is None or not np.isfinite(estimate):
            continue
        ci = 1.96 * se if se is not None and np.isfinite(se) else float("nan")
        color = term_colors.get(_mixed_model_term_kind(term), term_colors["covariate"])
        if np.isfinite(ci):
            ax.errorbar(estimate, y_pos, xerr=ci, fmt="none", ecolor=color, elinewidth=1.5, capsize=3, zorder=1)
        ax.scatter(
            estimate,
            y_pos,
            s=54,
            color=color,
            edgecolor="#222222" if significant else "white",
            linewidth=0.9,
            zorder=2,
        )
        if significant:
            ax.scatter(estimate, y_pos, s=120, marker="*", color="#111111", zorder=3)

    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlim(x_limits)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([poster_mixed_model_term_label(term) for term in display_terms], fontsize=max(9, POSTER_FONT_SIZE - 4))
    for tick_label, term in zip(ax.get_yticklabels(), display_terms):
        tick_label.set_color(term_colors.get(_mixed_model_term_kind(term), term_colors["covariate"]))
        tick_label.set_fontweight("normal")
    ax.tick_params(axis="y", pad=4)
    ax.set_title(mixed_model_response_display_label(response), fontsize=max(16, POSTER_TITLE_SIZE - 7), pad=2)
    ax.set_xlabel("Estimate (95% CI)", fontsize=max(17, POSTER_LABEL_SIZE - 1))
    ax.set_ylabel("Term", fontsize=max(15, POSTER_LABEL_SIZE - 3))
    ax.tick_params(axis="x", labelsize=POSTER_FONT_SIZE)
    ax.grid(axis="x", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    ax.text(
        0.99,
        0.02,
        f"n terms = {len(display_terms)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=POSTER_NOTE_SIZE,
        color="#444444",
    )
    if show_legend:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=max(13, POSTER_LEGEND_SIZE - 2),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            columnspacing=0.9,
            handletextpad=0.45,
        )
    return True


def build_figure(
    cache: Dict[str, Any],
    *,
    width_cm: float = DEFAULT_WIDTH_CM,
    height_cm: float = DEFAULT_HEIGHT_CM,
) -> Tuple[Any, Dict[str, Any], List[str]]:
    config = cache.get("config", {}) if isinstance(cache.get("config", {}), dict) else {}
    movie_expids = config.get("movie_expids")
    sleep_expids = config.get("sleep_expids")
    state_comparison_states, basal_apical_states, selection_meta = resolve_analysis_state_selections(
        config,
        movie_expids=movie_expids,
        sleep_expids=sleep_expids,
    )
    analysis_state_selection = {
        "state_comparison_states": list(state_comparison_states),
        "basal_apical_states": list(basal_apical_states),
        "compare_states": selection_meta.get("compare_states"),
        "state_mode": selection_meta.get("state_mode"),
        "movie_trial_types": selection_meta.get("movie_trial_types"),
        "state_mode_source": selection_meta.get("state_mode_source"),
        "movie_trial_types_source": selection_meta.get("movie_trial_types_source"),
        "alerts": list(selection_meta.get("alerts", [])),
    }
    plot_results: Dict[str, Any] = {"analysis_state_selection": analysis_state_selection}

    shuffle_n = int(config.get("shuffle_n", 200) or 200)

    analysis_results_path = ROOT_DIR / "results" / "main_pipeline" / "analysis_results.json"
    analysis_results: Dict[str, Any] = {}
    if analysis_results_path.exists():
        try:
            loaded_results = json.loads(analysis_results_path.read_text())
            if isinstance(loaded_results, dict):
                analysis_results = loaded_results
        except Exception:
            analysis_results = {}

    summary_metrics = [
        "dendrite_mean",
        "dendrite_event_frequency_per_min",
    ]
    basal_state_results = {
        metric: summarize_state_values(cache, metric, state_comparison_states, compartment_filter="basal")
        for metric in summary_metrics
    }
    apical_state_results = {
        metric: summarize_state_values(cache, metric, state_comparison_states, compartment_filter="apical")
        for metric in summary_metrics
    }
    basal_apical_rows = {
        metric: [
            row
            for row in analysis_results.get("basal_apical_comparisons", [])
            if str(row.get("comparison")) == "basal_vs_apical"
            and str(row.get("metric")) == metric
            and str(row.get("state")) in state_comparison_states
        ]
        for metric in summary_metrics
    }
    for metric in summary_metrics:
        if basal_apical_rows.get(metric):
            continue
        basal_apical_rows[metric] = build_basal_apical_comparisons(cache, metric, state_comparison_states, shuffle_n)

    mixed_model_results = analysis_results.get("mixed_model_selected_state", {})
    if not isinstance(mixed_model_results, dict) or not mixed_model_results:
        mixed_model_results = analysis_results.get("mixed_model", {})
    if not isinstance(mixed_model_results, dict):
        mixed_model_results = {}
    mixed_model_results = ensure_mixed_model_forest_response(
        mixed_model_results,
        cache,
        "mean_dendrite_activity",
        state_comparison_states,
        basal_apical_states,
        shuffle_n,
    )
    mixed_model_results = ensure_mixed_model_forest_response(
        mixed_model_results,
        cache,
        "dendrite_event_frequency_per_min",
        state_comparison_states,
        basal_apical_states,
        shuffle_n,
    )
    plot_results["mixed_model"] = mixed_model_results

    width_in = cm_to_inch(width_cm)
    height_in = cm_to_inch(height_cm)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width_in, height_in),
        squeeze=False,
        sharey="row",
        gridspec_kw={
            "width_ratios": [0.78, 1.22],
            "height_ratios": [0.95, 1.05],
        },
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.89,
        bottom=0.12,
        wspace=0.18,
        hspace=0.22,
    )

    boxplot_state_order = list(reversed(list(state_comparison_states)))

    draw_state_summary_compartment_comparison_panel(
        axes[0, 0],
        metric_key="dendrite_mean",
        metric_title="Dendrite mean dF/F",
        basal_summary=basal_state_results.get("dendrite_mean", {}),
        apical_summary=apical_state_results.get("dendrite_mean", {}),
        state_order=boxplot_state_order,
        y_limit=None,
        comparison_rows=basal_apical_rows.get("dendrite_mean", []),
        horizontal=True,
        show_legend=True,
        show_counts=True,
    )
    draw_state_summary_compartment_comparison_panel(
        axes[0, 1],
        metric_key="dendrite_event_frequency_per_min",
        metric_title="Dendrite calcium event frequency (per min)",
        basal_summary=basal_state_results.get("dendrite_event_frequency_per_min", {}),
        apical_summary=apical_state_results.get("dendrite_event_frequency_per_min", {}),
        state_order=boxplot_state_order,
        y_limit=None,
        comparison_rows=basal_apical_rows.get("dendrite_event_frequency_per_min", []),
        horizontal=True,
        show_legend=False,
        show_counts=True,
    )
    axes[0, 1].tick_params(axis="y", labelleft=False)
    axes[0, 1].set_ylabel("")

    draw_mixed_model_forest_panel(
        axes[1, 0],
        results=plot_results,
        response="mean_dendrite_activity",
        model_key="mixed_model",
        show_legend=False,
    )
    draw_mixed_model_forest_panel(
        axes[1, 1],
        results=plot_results,
        response="dendrite_event_frequency_per_min",
        model_key="mixed_model",
        show_legend=False,
    )
    axes[1, 1].tick_params(axis="y", labelleft=False)
    axes[1, 1].set_ylabel("")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label)
        for label, color in [
            ("intercept", "#7f7f7f"),
            ("state", "#1f77b4"),
            ("compartment", "#2ca02c"),
            ("interaction", "#ff7f0e"),
            ("covariate", "#9467bd"),
        ]
    ]
    legend_handles.append(
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=10, label="p < 0.05")
    )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=max(13, POSTER_LEGEND_SIZE - 2),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=6,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    return fig, analysis_state_selection, state_comparison_states

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Create the poster-ready 2x2 dendrite summary figure.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the raw sleep_dendrite_spine_cache.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the poster-ready SVG and PNG will be saved.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Base filename stem for the exported figure.",
    )
    parser.add_argument(
        "--width-cm",
        type=float,
        default=DEFAULT_WIDTH_CM,
        help="Target figure width in centimeters.",
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=DEFAULT_HEIGHT_CM,
        help="Target figure height in centimeters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if plt is None:
        raise RuntimeError("matplotlib is unavailable, so the poster figure cannot be rendered.")

    cache = load_cache(args.cache_path)
    output_dir = ensure_dir(args.output_dir)
    output_stem = str(args.output_stem).strip() or DEFAULT_OUTPUT_STEM
    svg_path = output_dir / f"{output_stem}.svg"

    figure, _, _ = build_figure(cache, width_cm=args.width_cm, height_cm=args.height_cm)
    save_figure(figure, svg_path, extra_formats=())
    set_svg_physical_size(svg_path, args.width_cm)

    print(f"Saved SVG: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
