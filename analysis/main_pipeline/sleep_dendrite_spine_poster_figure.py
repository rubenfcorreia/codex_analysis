#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
from collections import defaultdict
import json
import math
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
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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

import sleep_dendrite_spine_poster_common as poster_common

from sleep_dendrite_spine_pipeline import (
    REPORT_SIGNIFICANCE_ALPHA,
    SPINE_COACTIVITY_ANCHOR_STATE,
    _draw_boxplot_significance_annotations,
    _spine_coactivity_basal_apical_distribution_rows,
    _spine_coactivity_pair_state_display_label,
    _spine_coactivity_pair_state_rows,
    as_float,
    basal_apical_comparison,
    build_filtered_spine_coactivity_results,
    build_state_summary_gallery_results,
    build_mixed_model_table,
    canonical_state_label,
    color_state_tick_labels,
    ensure_dir,
    filter_rows_by_spine_coactivity,
    flatten_state_summary_values,
    format_requested_state_label,
    gallery_compartment_suffix,
    load_cached_analysis_table,
    load_npz_cache,
    mixed_model_design_row,
    mixed_model_response_display_label,
    padded_value_limits,
    resolve_analysis_state_selections,
    run_mixed_model_analysis,
    run_mixed_model_family,
    selected_matrix_plot_state_labels,
    selected_mixed_model_state_labels,
    set_requested_state_ticks,
    set_sparse_colorbar_ticks,
    set_sparse_numeric_ticks,
    spine_coactivity_anchor_state_compartments,
    spine_coactivity_basal_apical_distribution_output_name,
    spine_coactivity_output_compartments,
    spine_coactivity_pair_state_output_name,
    state_display_color,
    state_summary_y_limits,
    summarize_state_values,
    summarize_state_values_by_dendrite,
    _mixed_model_response_payload,
    _mixed_model_term_component_label,
    _mixed_model_term_interaction_value_label,
    _mixed_model_term_kind,
    _mixed_model_term_value_label,
    _pad_boxplot_ylim,
    _set_boxplot_colors,
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
    try:
        cache = load_npz_cache(path)
        if not isinstance(cache, dict):
            raise TypeError(f"Expected a cache dictionary from {path}")
        return cache
    except Exception:
        json_path = ROOT_DIR / "results" / "main_pipeline" / "analysis_results.json"
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text())
                if isinstance(payload, dict):
                    return {
                        "config": dict(payload.get("config", {})) if isinstance(payload.get("config"), dict) else {},
                        "analysis_unit": payload.get("analysis_unit"),
                        "alerts": list(payload.get("alerts", [])) if isinstance(payload.get("alerts", []), list) else [],
                        "run_parameters": dict(payload.get("run_parameters", {})) if isinstance(payload.get("run_parameters"), dict) else {},
                    }
            except Exception:
                pass
        return {}


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


DEFAULT_SPINE_COACTIVITY_OUTPUT_STEM = "spine_coactivity_poster_ready"
DEFAULT_SPINE_COACTIVITY_WIDTH_CM = 35.0
DEFAULT_SPINE_COACTIVITY_HEIGHT_CM = 15.0
SPINE_COACTIVITY_COMPOSITE_STEM = "spine_coactivity_poster_ready"


def load_analysis_results_payload() -> Dict[str, Any]:
    json_path = ROOT_DIR / "results" / "main_pipeline" / "analysis_results.json"
    payload: Dict[str, Any] = {}
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text())
            if isinstance(loaded, dict):
                payload = dict(loaded)
        except Exception:
            pass
    spine_coactivity = payload.get("spine_coactivity")
    if isinstance(spine_coactivity, dict) and spine_coactivity.get("table_rows"):
        return payload

    def _load_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
        if not csv_path.exists():
            return []
        try:
            with csv_path.open("r", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            return []

    coactivity_base = ROOT_DIR / "results" / "main_pipeline"
    table_rows = _load_csv_rows(coactivity_base / "spine_coactivity_table.csv")
    if not table_rows:
        cache_path = coactivity_base / "cache" / "sleep_dendrite_spine_cache_analysis_results_cache.npz"
        if cache_path.exists():
            try:
                payload = load_npz_cache(cache_path)
                if isinstance(payload, dict):
                    if isinstance(payload.get("analysis_results"), dict):
                        return dict(payload.get("analysis_results", {}))
                    return payload
            except Exception:
                pass
        return payload if isinstance(payload, dict) else {}

    coactivity_payload = dict(spine_coactivity) if isinstance(spine_coactivity, dict) else {}
    coactivity_payload["table_rows"] = table_rows
    coactivity_payload.setdefault("pair_state_rows", list(table_rows))

    state_summary_rows = _load_csv_rows(coactivity_base / "spine_coactivity_state_summary.csv")
    if state_summary_rows:
        coactivity_payload.setdefault("state_summary_rows", state_summary_rows)
    pair_summary_rows = _load_csv_rows(coactivity_base / "spine_coactivity_pair_summary.csv")
    if pair_summary_rows:
        coactivity_payload.setdefault("pair_summary_rows", pair_summary_rows)
    compartment_summary_rows = _load_csv_rows(coactivity_base / "spine_coactivity_compartment_summary.csv")
    if compartment_summary_rows:
        coactivity_payload.setdefault("compartment_summary_rows", compartment_summary_rows)
    state_agreement_rows = _load_csv_rows(coactivity_base / "spine_coactivity_state_agreement.csv")
    if state_agreement_rows:
        coactivity_payload.setdefault("state_agreement_rows", state_agreement_rows)

    payload["spine_coactivity"] = coactivity_payload
    return payload


def _export_single_axis_figure(fig: Any, path: Path) -> Path:
    save_figure(fig, path, extra_formats=())
    return path



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


# Spine coactivity poster mode.

def _save_svg_figure_exact(fig: Any, path: Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=POSTER_DPI, bbox_inches=None, pad_inches=0)
    if plt is not None:
        plt.close(fig)
    return output_path


def _spine_coactivity_basal_apical_payload(
    results: Dict[str, Any],
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Optional[Dict[str, Any]]:
    compartment_state_rows, state_labels = _spine_coactivity_basal_apical_distribution_rows(
        results,
        anchor_state_filter=anchor_state_filter,
        coactive_only=coactive_only,
    )
    if not state_labels:
        return None
    all_values: List[float] = []
    for compartment in ("basal", "apical"):
        for state in state_labels:
            state_rows = compartment_state_rows.get(compartment, {}).get(state, [])
            values = np.asarray(
                [as_float(row.get("mean_coactivity_r")) for row in state_rows if np.isfinite(as_float(row.get("mean_coactivity_r")))],
                dtype=float,
            )
            if values.size:
                all_values.extend(float(value) for value in values)
    if not all_values:
        return None
    return {
        "compartment_state_rows": compartment_state_rows,
        "state_labels": list(state_labels),
        "x_limits": padded_value_limits(np.asarray(all_values, dtype=float)),
    }

def _spine_coactivity_distribution_payload(
    results: Dict[str, Any],
    compartment_filter: Optional[str] = None,
    state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Optional[Dict[str, Any]]:
    rows, state_labels = _spine_coactivity_pair_state_rows(
        results,
        compartment_filter=compartment_filter,
        anchor_state_filter=state_filter,
        coactive_only=coactive_only,
    )
    if not rows or not state_labels:
        return None

    if state_filter is not None:
        wanted_states = [canonical_state_label(state_filter)]
    else:
        wanted_states = list(state_labels)

    labels: List[str] = []
    series: List[np.ndarray] = []
    significance_masks: List[np.ndarray] = []
    all_values: List[float] = []

    for state in wanted_states:
        state_rows = [
            row for row in rows
            if canonical_state_label(row.get("state")) == state
        ]
        values: List[float] = []
        sig_mask: List[bool] = []

        for row in state_rows:
            value = as_float(row.get("coactivity_r"))
            if value is None or not np.isfinite(value):
                continue
            values.append(float(value))
            sig_mask.append(_coactivity_row_is_significant(row))

        if not values:
            continue

        labels.append(state)
        arr = np.asarray(values, dtype=float)
        series.append(arr)
        significance_masks.append(np.asarray(sig_mask, dtype=bool))
        all_values.extend(values)

    if not labels:
        return None

    return {
        "rows": rows,
        "state_labels": labels,
        "series": series,
        "significance_masks": significance_masks,
        "x_limits": padded_value_limits(np.asarray(all_values, dtype=float)),
    }

def _spine_coactivity_heatmap_payload(
    results: Dict[str, Any],
    compartment_filter: Optional[str] = None,
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Optional[Dict[str, Any]]:
    rows, state_labels = _spine_coactivity_pair_state_rows(
        results,
        compartment_filter=compartment_filter,
        anchor_state_filter=anchor_state_filter,
        coactive_only=coactive_only,
    )
    if not rows or not state_labels:
        return None
    coactivity = results.get("spine_coactivity", {})
    pair_summary_rows = [row for row in coactivity.get("pair_summary_rows", []) if isinstance(row, dict)] if isinstance(coactivity, dict) else []
    if compartment_filter is not None:
        pair_summary_rows = [row for row in pair_summary_rows if str(row.get("compartment")) == compartment_filter]
    pair_summary_lookup = {str(row.get("global_pair_id")): dict(row) for row in pair_summary_rows}
    pair_rows_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_rows_by_id[str(row.get("global_pair_id"))].append(dict(row))
    pair_ids = list(pair_rows_by_id)
    if not pair_ids:
        return None

    def sort_key(pair_id: str) -> Tuple[int, float, int, float, str]:
        summary = pair_summary_lookup.get(pair_id, {})
        range_value = as_float(summary.get("coactivity_r_range"))
        profile_similarity = as_float(summary.get("profile_similarity_r"))
        return (
            0 if range_value is not None and np.isfinite(range_value) else 1,
            -(range_value if range_value is not None and np.isfinite(range_value) else float("-inf")),
            0 if profile_similarity is not None and np.isfinite(profile_similarity) else 1,
            -(profile_similarity if profile_similarity is not None and np.isfinite(profile_similarity) else float("-inf")),
            pair_id,
        )

    pair_ids = sorted(pair_ids, key=sort_key)
    matrix = np.full((len(pair_ids), len(state_labels)), np.nan, dtype=float)
    pair_labels: List[str] = []
    for row_index, pair_id in enumerate(pair_ids):
        pair_rows = pair_rows_by_id.get(pair_id, [])
        reference_row = pair_rows[0] if pair_rows else pair_summary_lookup.get(pair_id, {})
        pair_labels.append(_spine_coactivity_pair_state_display_label(reference_row))
        for col_index, state in enumerate(state_labels):
            state_value = next(
                (
                    as_float(row.get("coactivity_r"))
                    for row in pair_rows
                    if canonical_state_label(row.get("state")) == state and np.isfinite(as_float(row.get("coactivity_r")))
                ),
                None,
            )
            if state_value is not None and np.isfinite(state_value):
                matrix[row_index, col_index] = float(state_value)
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return None
    max_abs = float(np.nanmax(np.abs(finite_values))) if finite_values.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    return {
        "matrix": matrix,
        "pair_labels": pair_labels,
        "state_labels": list(state_labels),
        "max_abs": max_abs,
        "pair_ids": pair_ids,
    }


def _draw_spine_coactivity_basal_apical_distribution_panel(
    ax: Any,
    payload: Dict[str, Any],
    *,
    title: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_y_ticklabels: bool = True,
    show_legend: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
) -> None:
    compartment_state_rows = payload["compartment_state_rows"]
    state_labels = list(payload["state_labels"])
    state_positions = np.arange(1, len(state_labels) + 1, dtype=float)
    compartment_colors = {"basal": "#1f77b4", "apical": "#d95f02"}
    compartment_offsets = {"basal": -0.16, "apical": 0.16}
    compartment_order = [comp for comp in ["basal", "apical"] if any(compartment_state_rows.get(comp, {}).get(state) for state in state_labels)]
    for state_index, state in enumerate(state_labels):
        base_pos = state_positions[state_index]
        for compartment in compartment_order:
            pair_rows = compartment_state_rows.get(compartment, {}).get(state, [])
            values = np.asarray(
                [
                    as_float(row.get("mean_coactivity_r"))
                    for row in pair_rows
                    if np.isfinite(as_float(row.get("mean_coactivity_r")))
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            pos = base_pos + compartment_offsets.get(compartment, 0.0)
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=0.26,
                patch_artist=True,
                showfliers=False,
                vert=False,
            )
            _set_boxplot_colors(bp, [compartment_colors.get(compartment, "#444444")])
            jitter = np.random.default_rng(91 if compartment == "basal" else 92).uniform(-0.08, 0.08, size=values.size)
            ax.scatter(
                values,
                np.full(values.size, pos) + jitter,
                s=7,
                alpha=0.45,
                color=compartment_colors.get(compartment, "#444444"),
                edgecolor="none",
                zorder=3,
            )
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=max(10, POSTER_TITLE_SIZE - 12), pad=2)
    ax.set_xlabel("Mean coactivity coefficient per spine pair" if show_xlabel else "", fontsize=max(15, POSTER_LABEL_SIZE - 1))
    ax.set_ylabel("State" if show_ylabel else "", fontsize=max(15, POSTER_LABEL_SIZE - 3))
    ax.set_yticks(state_positions)
    ax.set_yticklabels([format_requested_state_label(state) for state in state_labels])
    color_state_tick_labels(ax, state_labels, axis="y")
    if not show_y_ticklabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="both", labelsize=max(13, POSTER_FONT_SIZE - 2))
    ax.grid(axis="x", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    else:
        all_values = [
            as_float(row.get("mean_coactivity_r"))
            for comp in compartment_order
            for state in state_labels
            for row in compartment_state_rows.get(comp, {}).get(state, [])
            if np.isfinite(as_float(row.get("mean_coactivity_r")))
        ]
        if all_values:
            limits = padded_value_limits(np.asarray(all_values, dtype=float))
            if limits is not None:
                ax.set_xlim(limits)
    if show_legend and compartment_order:
        legend_handles = [
            Line2D([0], [0], color=compartment_colors.get(compartment, "#444444"), marker="s", linestyle="", markersize=8, label=compartment.capitalize())
            for compartment in compartment_order
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=max(12, POSTER_LEGEND_SIZE - 3))
    anchor_text = format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)
    ax.invert_yaxis()



def _draw_spine_coactivity_distribution_panel(
    ax: Any,
    payload: Dict[str, Any],
    *,
    title: str,
    state_filter: Optional[str] = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_y_ticklabels: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    highlight_color: str = "#1f77b4",
    show_legend: bool = True,
) -> None:
    state_labels = list(payload["state_labels"])
    series = [np.asarray(arr, dtype=float) for arr in payload["series"]]
    significance_masks = [
        np.asarray(mask, dtype=bool)
        for mask in payload.get("significance_masks", [np.ones_like(arr, dtype=bool) for arr in series])
    ]

    positions = np.arange(1, len(state_labels) + 1)

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        vert=False,
    )
    _set_boxplot_colors(bp, ["#bdbdbd"] * len(series))

    rng = np.random.default_rng(11)
    for pos, arr, sig_mask in zip(positions, series, significance_masks):
        finite_mask = np.isfinite(arr)
        finite = arr[finite_mask]
        sig = sig_mask[finite_mask]

        if finite.size == 0:
            continue

        jitter = rng.uniform(-0.12, 0.12, size=finite.size)

        ns_values = finite[~sig]
        ns_y = np.full(ns_values.size, pos) + jitter[~sig]
        if ns_values.size:
            ax.scatter(
                ns_values,
                ns_y,
                s=7,
                alpha=0.38,
                color="#9e9e9e",
                edgecolor="none",
                zorder=2,
                label="ns" if pos == positions[0] else None,
            )

        sig_values = finite[sig]
        sig_y = np.full(sig_values.size, pos) + jitter[sig]
        if sig_values.size:
            ax.scatter(
                sig_values,
                sig_y,
                s=10,
                alpha=0.72,
                color=highlight_color,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
                label="significant" if pos == positions[0] else None,
            )

        ax.text(
            0.98,
            pos,
            f"sig={sig_values.size} | ns={ns_values.size}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=max(7, POSTER_NOTE_SIZE - 5),
        )

    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=max(10, POSTER_TITLE_SIZE - 12), pad=2)
    ax.set_ylabel("State" if show_ylabel else "", fontsize=max(13, POSTER_LABEL_SIZE - 5))
    ax.set_xlabel("Coactivity coefficient" if show_xlabel else "", fontsize=max(13, POSTER_LABEL_SIZE - 4))

    ax.set_yticks(positions)
    ax.set_yticklabels([format_requested_state_label(state) for state in state_labels])
    if not show_y_ticklabels:
        ax.tick_params(axis="y", labelleft=False)

    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=max(11, POSTER_FONT_SIZE - 4))
    ax.grid(axis="x", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)

    if x_limits is not None:
        ax.set_xlim(x_limits)
    else:
        all_values = np.concatenate(series) if series else np.asarray([], dtype=float)
        if all_values.size:
            limits = padded_value_limits(all_values)
            if limits is not None:
                ax.set_xlim(limits)

    note_state = format_requested_state_label(state_filter) if state_filter is not None else "selected states"
    ax.text(
        0.02,
        0.98,
        f"state={note_state}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=max(7, POSTER_NOTE_SIZE - 5),
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=highlight_color, markersize=6, label="significant"),
            Line2D([0], [0], marker="o", linestyle="", color="#9e9e9e", markersize=6, label="ns"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=max(9, POSTER_LEGEND_SIZE - 5), loc="lower right")

def _draw_spine_coactivity_heatmap_panel(
    ax: Any,
    payload: Dict[str, Any],
    *,
    title: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_x_ticklabels: bool = True,
    show_y_ticklabels: bool = False,
    colorbar: bool = False,
    colorbar_label: str = "Coactivity coefficient",
    shared_max_abs: Optional[float] = None,
) -> None:
    from matplotlib.colors import Normalize

    matrix = np.asarray(payload["matrix"], dtype=float)
    state_labels = list(payload["state_labels"])
    pair_labels = list(payload["pair_labels"])
    max_abs = float(shared_max_abs if shared_max_abs is not None else payload["max_abs"])
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=max(16, POSTER_TITLE_SIZE - 6), pad=2)
    ax.set_xlabel("State" if show_xlabel else "", fontsize=max(15, POSTER_LABEL_SIZE - 2))
    ax.set_ylabel("Spine pair" if show_ylabel else "", fontsize=max(15, POSTER_LABEL_SIZE - 2))
    ax.set_xticks(np.arange(len(state_labels)))
    ax.set_xticklabels([format_requested_state_label(state) for state in state_labels], rotation=40, ha="right")
    color_state_tick_labels(ax, state_labels, axis="x")
    y_positions = np.arange(len(pair_labels))
    label_step = max(1, int(math.ceil(len(pair_labels) / 20.0)))
    sparse_y_labels = [label if (idx % label_step == 0 or idx in {0, len(pair_labels) - 1}) else "" for idx, label in enumerate(pair_labels)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sparse_y_labels, fontsize=max(8, POSTER_FONT_SIZE - 5))
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    if not show_x_ticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    if not show_y_ticklabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(-0.5, len(state_labels) - 0.5)
    ax.set_ylim(len(pair_labels) - 0.5, -0.5)
    ax.grid(which="major", color="white", linestyle="-", linewidth=0.6, alpha=0.55)
    anchor_text = format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)
    ax.text(
        0.02,
        0.98,
        f"anchor={anchor_text} | n pairs={len(pair_labels)} | coactive pairs only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    if colorbar:
        mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=-max_abs, vmax=max_abs), cmap="coolwarm")
        mappable.set_array([])
        cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=POSTER_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
        set_sparse_colorbar_ticks(cbar, nbins=5)

def _spine_coactivity_selected_state_comparison_payload(
    results: Dict[str, Any],
    *,
    compartment_filter: str,
    selected_states: Sequence[str],
    anchor_state: str = SPINE_COACTIVITY_ANCHOR_STATE,
) -> Optional[Dict[str, Any]]:
    rows, available_state_labels = _spine_coactivity_pair_state_rows(
        results,
        compartment_filter=compartment_filter,
        anchor_state_filter=anchor_state,
        coactive_only=False,
    )
    if not rows:
        return None

    anchor_state = canonical_state_label(anchor_state)
    selected_state_labels = [
        canonical_state_label(state)
        for state in selected_states
        if canonical_state_label(state) in {canonical_state_label(s) for s in available_state_labels}
    ]

    if not selected_state_labels:
        selected_state_labels = [canonical_state_label(s) for s in available_state_labels]

    anchor_sig_pair_ids = {
        _global_pair_id(row)
        for row in rows
        if canonical_state_label(row.get("state")) == anchor_state
        and _global_pair_id(row)
        and _coactivity_row_is_significant(row)
    }

    if not anchor_sig_pair_ids:
        return None

    series: List[np.ndarray] = []
    labels: List[str] = []
    all_values: List[float] = []

    for state in selected_state_labels:
        values = [
            as_float(row.get("coactivity_r"))
            for row in rows
            if canonical_state_label(row.get("state")) == state
            and _global_pair_id(row) in anchor_sig_pair_ids
            and np.isfinite(as_float(row.get("coactivity_r")))
        ]
        values = [float(v) for v in values if v is not None and np.isfinite(v)]

        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        labels.append(state)
        series.append(arr)
        all_values.extend(values)

    if not series:
        return None

    return {
        "state_labels": labels,
        "series": series,
        "n_anchor_pairs": len(anchor_sig_pair_ids),
        "y_limits": padded_value_limits(np.asarray(all_values, dtype=float)),
    }


def _draw_spine_coactivity_selected_state_comparison_panel(
    ax: Any,
    payload: Dict[str, Any],
    *,
    title: str,
    color: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_x_ticklabels: bool = True,
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    state_labels = list(payload["state_labels"])
    series = [np.asarray(arr, dtype=float) for arr in payload["series"]]
    positions = np.arange(1, len(state_labels) + 1)

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )
    _set_boxplot_colors(bp, [color] * len(series))

    rng = np.random.default_rng(23)
    for pos, arr in zip(positions, series):
        finite = arr[np.isfinite(arr)]
        if not finite.size:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=finite.size)
        ax.scatter(
            np.full(finite.size, pos) + jitter,
            finite,
            s=8,
            alpha=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.text(
            pos,
            0.98,
            f"n={finite.size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=max(7, POSTER_NOTE_SIZE - 5),
        )

    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=max(10, POSTER_TITLE_SIZE - 12), pad=2)
    ax.set_ylabel("Coactivity coefficient" if show_ylabel else "", fontsize=max(13, POSTER_LABEL_SIZE - 5))
    ax.set_xlabel("State" if show_xlabel else "", fontsize=max(13, POSTER_LABEL_SIZE - 5))

    ax.set_xticks(positions)
    ax.set_xticklabels([format_requested_state_label(state) for state in state_labels], rotation=35, ha="right")
    color_state_tick_labels(ax, state_labels, axis="x")

    if not show_x_ticklabels:
        ax.tick_params(axis="x", labelbottom=False)

    ax.tick_params(axis="both", labelsize=max(10, POSTER_FONT_SIZE - 5))
    ax.grid(axis="y", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="y", nbins=5)

    if y_limits is not None:
        ax.set_ylim(y_limits)
    else:
        limits = payload.get("y_limits")
        if limits is not None:
            ax.set_ylim(limits)

    ax.text(
        0.02,
        0.98,
        f"anchor={format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} | significant anchor pairs only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=max(7, POSTER_NOTE_SIZE - 5),
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    
def render_spine_coactivity_component_svgs(results: Dict[str, Any], output_dir: Path) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable, so the spine-coactivity poster cannot be rendered.")
    output_dir = ensure_dir(output_dir)
    written: List[Path] = []

    combined_payload = _spine_coactivity_basal_apical_payload(results, anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE, coactive_only=True)
    if combined_payload is None:
        raise RuntimeError("No spine coactivity rows were available for the basal-vs-apical poster panel.")
    state_labels = combined_payload["state_labels"]
    fig = plt.figure(figsize=(min(max(8.6, 0.62 * len(state_labels) + 4.2), 12.8), min(max(7.0, 0.44 * len(state_labels) + 4.0), 11.5)))
    ax = fig.add_subplot(1, 1, 1)
    _draw_spine_coactivity_basal_apical_distribution_panel(
        ax,
        combined_payload,
        title="Quiet awake movies coactive-pair distribution - Basal vs apical",
    )
    written.append(_save_svg_figure_exact(fig, output_dir / spine_coactivity_basal_apical_distribution_output_name(SPINE_COACTIVITY_ANCHOR_STATE, True)))

    for compartment in ["basal", "apical"]:
        compartment_results = build_filtered_spine_coactivity_results(results, compartment)
        distribution_payload = _spine_coactivity_distribution_payload(
            compartment_results,
            compartment_filter=compartment,
            state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        if distribution_payload is None:
            raise RuntimeError(f"No spine coactivity rows were available for the {compartment} anchor distribution panel.")
        fig = plt.figure(figsize=(6.6, 3.9))
        ax = fig.add_subplot(1, 1, 1)
        _draw_spine_coactivity_distribution_panel(
            ax,
            distribution_payload,
            title=f"Quiet awake movies coactive-pair distribution - {compartment.capitalize()}",
            state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
        )

        written.append(_save_svg_figure_exact(fig, output_dir / spine_coactivity_pair_state_output_name("anchor_distribution", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True)))

    for compartment in ["basal", "apical"]:
        compartment_results = build_filtered_spine_coactivity_results(results, compartment)
        heatmap_payload = _spine_coactivity_heatmap_payload(
            compartment_results,
            compartment_filter=compartment,
            anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        if heatmap_payload is None:
            raise RuntimeError(f"No spine coactivity rows were available for the {compartment} pair-state heatmap panel.")
        fig = plt.figure(figsize=(7.4, min(max(5.2, 0.14 * len(heatmap_payload['pair_labels']) + 3.4), 14.0)))
        ax = fig.add_subplot(1, 1, 1)
        _draw_spine_coactivity_heatmap_panel(
            ax,
            heatmap_payload,
            title=f"Quiet awake movies coactive pairs across states - {compartment.capitalize()}",
            colorbar=True,
        )
        written.append(_save_svg_figure_exact(fig, output_dir / spine_coactivity_pair_state_output_name("pair_state_heatmap", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True)))

    return written


def build_spine_coactivity_poster_figure(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    width_cm: float = DEFAULT_SPINE_COACTIVITY_WIDTH_CM,
    height_cm: float = DEFAULT_SPINE_COACTIVITY_HEIGHT_CM,
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
    poster_results: Dict[str, Any] = dict(results) if isinstance(results, dict) else {}
    poster_results["analysis_state_selection"] = analysis_state_selection

    fig = plt.figure(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
    outer = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.38,
        hspace=0.45,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.18)

    left_ax = fig.add_subplot(outer[:, 0])
    left_payload = _spine_coactivity_basal_apical_payload(
        poster_results,
        anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
        coactive_only=True,
    )
    if left_payload is None:
        raise RuntimeError("Could not build the basal-vs-apical spine-coactivity poster panel.")
    _draw_spine_coactivity_basal_apical_distribution_panel(
        left_ax,
        left_payload,
        title="Quiet awake movies coactive-pair distribution - Basal vs apical",
    )

    mid_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[:, 1], hspace=0.38)
    right_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[:, 2], hspace=0.38)

    mid_top = fig.add_subplot(mid_spec[0, 0])
    mid_bottom = fig.add_subplot(mid_spec[1, 0], sharex=mid_top)
    right_top = fig.add_subplot(right_spec[0, 0])
    right_bottom = fig.add_subplot(right_spec[1, 0], sharex=right_top)

    mid_payloads = []
    for compartment in ["basal", "apical"]:
        compartment_results = build_filtered_spine_coactivity_results(poster_results, compartment)
        payload = _spine_coactivity_distribution_payload(
            compartment_results,
            compartment_filter=compartment,
            state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=False,
        )
        if payload is None:
            raise RuntimeError(f"Could not build the quiet-awake-movies coactivity distribution panel for {compartment}.")
        mid_payloads.append((compartment, payload))

    mid_limits_values = np.concatenate([arr for _, payload in mid_payloads for arr in payload["series"]])
    mid_limits = padded_value_limits(mid_limits_values)

    _draw_spine_coactivity_distribution_panel(
        mid_top,
        mid_payloads[0][1],
        title="Quiet awake movie spine-pair coactivity - Basal",
        state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
        show_y_ticklabels=False,
        x_limits=mid_limits,
        highlight_color="#1f77b4",
        show_legend=True,
    )
    _draw_spine_coactivity_distribution_panel(
        mid_bottom,
        mid_payloads[1][1],
        title="Quiet awake movie spine-pair coactivity - Apical",
        state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
        show_y_ticklabels=False,
        x_limits=mid_limits,
        highlight_color="#d95f02",
        show_legend=False,
    )
    mid_top.set_ylabel("")
    mid_bottom.set_ylabel("")
    mid_top.tick_params(axis="x", labelbottom=False)
    mid_top.set_xlabel("")

    right_payloads = []
    for compartment in ["basal", "apical"]:
        compartment_results = build_filtered_spine_coactivity_results(poster_results, compartment)
        payload = _spine_coactivity_selected_state_comparison_payload(
            compartment_results,
            compartment_filter=compartment,
            selected_states=state_comparison_states,
            anchor_state=SPINE_COACTIVITY_ANCHOR_STATE,
        )
        if payload is None:
            raise RuntimeError(f"Could not build the selected-state coactivity comparison panel for {compartment}.")
        right_payloads.append((compartment, payload))

    right_limits_values = np.concatenate([arr for _, payload in right_payloads for arr in payload["series"]])
    right_y_limits = padded_value_limits(right_limits_values)

    _draw_spine_coactivity_selected_state_comparison_panel(
        right_top,
        right_payloads[0][1],
        title="Basal coactive spine pairs across selected states",
        color="#1f77b4",
        show_x_ticklabels=False,
        y_limits=right_y_limits,
    )
    _draw_spine_coactivity_selected_state_comparison_panel(
        right_bottom,
        right_payloads[1][1],
        title="Apical coactive spine pairs across selected states",
        color="#d95f02",
        y_limits=right_y_limits,
    )
    right_top.set_xlabel("")
    right_top.tick_params(axis="x", labelbottom=False)
    return fig, analysis_state_selection, list(state_comparison_states)


def _coactivity_row_is_significant(row: Dict[str, Any]) -> bool:
    """Return True if a spine-pair row should be treated as significant/coactive."""
    for key in (
        "significant",
        "is_significant",
        "coactivity_significant",
        "is_coactive",
        "coactive",
        "anchor_coactive",
        "quiet_awake_movie_coactive",
    ):
        if key in row:
            value = row.get(key)
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "sig", "significant", "coactive"}:
                return True
            if text in {"0", "false", "no", "n", "ns", "non-significant", "nonsignificant"}:
                return False

    for key in (
        "shuffle_p",
        "p_value",
        "p",
        "coactivity_p",
        "coactivity_p_value",
        "anchor_shuffle_p",
        "anchor_p_value",
    ):
        p_value = as_float(row.get(key))
        if p_value is not None and np.isfinite(p_value):
            return float(p_value) < REPORT_SIGNIFICANCE_ALPHA

    return False


def _global_pair_id(row: Dict[str, Any]) -> str:
    return str(row.get("global_pair_id", row.get("pair_id", "")))

def build_figure(
    cache: Dict[str, Any],
    *,
    results: Optional[Dict[str, Any]] = None,
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

    analysis_results: Dict[str, Any] = dict(results) if isinstance(results, dict) else poster_common.load_analysis_results_payload()

    summary_metrics = [
        "dendrite_mean",
        "dendrite_event_frequency_per_min",
    ]
    basal_state_results = {
        metric: summarize_state_values_by_dendrite(
            cache,
            metric,
            state_comparison_states,
            compartment_filter="basal",
        )
        for metric in summary_metrics
    }

    apical_state_results = {
        metric: summarize_state_values_by_dendrite(
            cache,
            metric,
            state_comparison_states,
            compartment_filter="apical",
        )
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

def write_mixed_model_poster_figure(
    cache: Dict[str, Any],
    output_dir: Path,
    *,
    results: Optional[Dict[str, Any]] = None,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    width_cm: float = DEFAULT_WIDTH_CM,
    height_cm: float = DEFAULT_HEIGHT_CM,
) -> Path:
    figure, _, _ = build_figure(cache, results=results, width_cm=width_cm, height_cm=height_cm)
    output_dir = poster_common.ensure_dir(output_dir)
    stem = str(output_stem).strip() or DEFAULT_OUTPUT_STEM
    svg_path = output_dir / f"{stem}.svg"
    poster_common.save_svg_figure_exact(figure, svg_path)
    poster_common.set_svg_physical_size(svg_path, float(width_cm))
    return svg_path


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
    cache = poster_common.load_cache(args.cache_path)
    results = poster_common.load_analysis_results_payload()
    if not isinstance(results, dict) or not results:
        raise RuntimeError("No analysis_results payload was available for the mixed-model poster.")
    svg_path = write_mixed_model_poster_figure(
        cache,
        args.output_dir,
        results=results,
        output_stem=str(args.output_stem).strip() or DEFAULT_OUTPUT_STEM,
        width_cm=float(args.width_cm),
        height_cm=float(args.height_cm),
    )
    print(f"Saved SVG: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
