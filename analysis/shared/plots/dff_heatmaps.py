"""Representative dF/F heatmaps with calcium-event onset markers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from analysis.shared.analysis_families.common_helpers import build_event_info
from analysis.shared.analysis_families.core import make_global_bouton_id, make_global_soma_id, shared_time_axis
from analysis.shared.plots.figure_io import save_figure
from analysis.shared.state_utils import canonical_state_label, state_display_label


CLASS_ORDER = (
    "high_activity_high_frequency",
    "low_activity_high_frequency",
    "high_activity_low_frequency",
    "low_activity_low_frequency",
)
CLASS_LABELS = {
    "high_activity_high_frequency": "High frequency / high activity",
    "low_activity_high_frequency": "High frequency / low activity",
    "high_activity_low_frequency": "Low frequency / high activity",
    "low_activity_low_frequency": "Low frequency / low activity",
}
CLASS_COLORS = {
    "high_activity_high_frequency": "#4c78a8",
    "low_activity_high_frequency": "#f58518",
    "high_activity_low_frequency": "#72b7b2",
    "low_activity_low_frequency": "#e45756",
}


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    return text or "roi"


def _safe_filename(value: Any) -> str:
    text = _safe_id(value)
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text)


def _trace_and_time(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(record.get("trace", []), dtype=float).reshape(-1)
    time = np.asarray(record.get("time", []), dtype=float).reshape(-1)
    usable = min(trace.size, time.size)
    return trace[:usable], time[:usable]


def _downsample(time: np.ndarray, trace: np.ndarray, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or time.size <= max_points:
        return time, trace
    indices = np.linspace(0, time.size - 1, int(max_points), dtype=int)
    return time[indices], trace[indices]


def _event_onsets(record: Mapping[str, Any], time: np.ndarray, trace: np.ndarray, method: str) -> np.ndarray:
    event_info = record.get("event_info")
    if not isinstance(event_info, Mapping) or str(event_info.get("method") or event_info.get("primary_method") or "") != method:
        event_info = build_event_info(trace, time, method=method, include_all_methods=False)
    runs = event_info.get("event_runs", []) if isinstance(event_info, Mapping) else []
    indices = []
    for run in runs or []:
        try:
            index = int(run[0]) if isinstance(run, (tuple, list)) else int(run)
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= index < time.size and np.isfinite(time[index]):
            indices.append(time[index])
    return np.asarray(indices, dtype=float)


def _record_metrics(record: Mapping[str, Any], method: str) -> dict[str, Any] | None:
    time, trace = _trace_and_time(record)
    finite = np.isfinite(time) & np.isfinite(trace)
    if finite.sum() < 3:
        return None
    compact_time = time[finite]
    compact_trace = trace[finite]
    event_info = build_event_info(compact_trace, compact_time, method=method, include_all_methods=False)
    frequency = _finite_float(event_info.get("event_frequency_per_min"))
    activity = _finite_float(record.get("mean_dff"))
    if not np.isfinite(activity):
        activity = _finite_float(record.get("mean_activity"))
    if not np.isfinite(activity):
        activity = float(np.nanmean(trace[finite]))
    return {
        **dict(record),
        "time": time,
        "trace": trace,
        "event_info": event_info,
        "event_frequency_per_min": frequency,
        "mean_dff": activity,
        "event_onsets": _event_onsets({**record, "event_info": event_info}, compact_time, compact_trace, method),
    }


def _class_lookup(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        group = str(row.get("split_group") or row.get("group") or "").strip().lower()
        if group not in CLASS_ORDER:
            continue
        compartment = str(row.get("compartment") or "").strip().lower()
        subject = _safe_id(row.get("subject_id") or row.get("unit_id") or row.get("roi_key") or row.get("roi_id"))
        state = canonical_state_label(row.get("state") or row.get("state_label") or "")
        if compartment and subject:
            lookup[(compartment, subject, state)] = group
            lookup[(compartment, subject, "")] = group
    return lookup


def _assign_missing_classes(records: list[dict[str, Any]]) -> None:
    """Use existing labels when present; otherwise apply deterministic median splits."""
    pending = [record for record in records if str(record.get("split_group") or "") not in CLASS_ORDER]
    for state in sorted({canonical_state_label(row.get("state") or "") for row in pending}):
        state_rows = [row for row in pending if canonical_state_label(row.get("state") or "") == state]
        activity = np.asarray([_finite_float(row.get("mean_dff")) for row in state_rows], dtype=float)
        frequency = np.asarray([_finite_float(row.get("event_frequency_per_min")) for row in state_rows], dtype=float)
        activity_cut = float(np.nanmedian(activity[np.isfinite(activity)])) if np.isfinite(activity).any() else 0.0
        frequency_cut = float(np.nanmedian(frequency[np.isfinite(frequency)])) if np.isfinite(frequency).any() else 0.0
        for row in state_rows:
            high_activity = _finite_float(row.get("mean_dff")) >= activity_cut
            high_frequency = _finite_float(row.get("event_frequency_per_min")) >= frequency_cut
            row["split_group"] = (
                "high_activity_high_frequency" if high_activity and high_frequency else
                "low_activity_high_frequency" if not high_activity and high_frequency else
                "high_activity_low_frequency" if high_activity else
                "low_activity_low_frequency"
            )


def render_dff_heatmaps(
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    event_detection_method: str = "amplitude",
    max_rois_per_class: int = 10,
    max_trace_points: int | None = None,
    output_formats: Sequence[str] = ("svg", "png"),
) -> list[str]:
    """Render one state-aware heatmap per compartment and write JSON metadata."""
    prepared: list[dict[str, Any]] = []
    for raw in records:
        metrics = _record_metrics(raw, event_detection_method)
        if metrics is None:
            continue
        metrics["compartment"] = str(metrics.get("compartment") or "other").strip().lower()
        metrics["state"] = canonical_state_label(metrics.get("state") or "all") or "all"
        prepared.append(metrics)
    if not prepared:
        return []
    _assign_missing_classes(prepared)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in prepared:
        grouped[(record["compartment"], record["state"])].append(record)
    for (compartment, state), group_records in sorted(grouped.items()):
        selected: list[dict[str, Any]] = []
        for class_name in CLASS_ORDER:
            candidates = [row for row in group_records if row.get("split_group") == class_name]
            candidates.sort(key=lambda row: (-_finite_float(row.get("event_frequency_per_min")) if np.isfinite(_finite_float(row.get("event_frequency_per_min"))) else float("inf"), -_finite_float(row.get("mean_dff")) if np.isfinite(_finite_float(row.get("mean_dff"))) else float("inf"), _safe_id(row.get("roi_id") or row.get("roi_key") or row.get("unit_id"))))
            selected.extend(candidates[: max(0, int(max_rois_per_class))])
        if not selected:
            continue
        traces: list[np.ndarray] = []
        times: list[np.ndarray] = []
        for row in selected:
            time, trace = _downsample(*_trace_and_time(row), max_trace_points)
            traces.append(trace)
            times.append(time)
        common_length = min(trace.size for trace in traces)
        if common_length < 3:
            continue
        time = times[0][:common_length]
        matrix = np.vstack([trace[:common_length] for trace in traces])
        finite = np.isfinite(matrix)
        if not finite.any():
            continue
        low, high = np.nanpercentile(matrix[finite], [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(matrix[finite])), float(np.nanmax(matrix[finite]) + 1e-9)
        fig_height = max(4.5, 1.8 + 0.22 * len(selected))
        fig, ax = plt.subplots(figsize=(13.5, fig_height))
        image = ax.imshow(matrix, aspect="auto", origin="upper", interpolation="nearest", extent=[time[0], time[-1], len(selected) - 0.5, -0.5], cmap="magma", vmin=low, vmax=high)
        boundaries: list[int] = []
        cursor = 0
        for class_name in CLASS_ORDER:
            count = sum(row.get("split_group") == class_name for row in selected)
            if count:
                if cursor:
                    boundaries.append(cursor)
                cursor += count
        for boundary in boundaries:
            ax.axhline(boundary - 0.5, color="#00d7e6", linewidth=2.0)
        for row_index, row in enumerate(selected):
            for onset in np.asarray(row.get("event_onsets", []), dtype=float):
                if time[0] <= onset <= time[-1]:
                    ax.vlines(onset, row_index - 0.32, row_index + 0.32, color="white", linewidth=1.0, alpha=0.95)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Representative ROI", fontsize=11)
        ax.set_title(f"{compartment.capitalize()} dF/F heatmap — {state_display_label(state)}", fontsize=14, pad=12)
        ax.text(0.0, -0.13, "Rows sorted by class, event frequency ↓, mean dF/F ↓, ROI ID", transform=ax.transAxes, fontsize=9, va="top")
        cursor = 0
        for class_name in CLASS_ORDER:
            class_rows = [row for row in selected if row.get("split_group") == class_name]
            if class_rows:
                center = cursor + (len(class_rows) - 1) / 2
                ax.text(1.01, center, f"{CLASS_LABELS[class_name]}\n(n={len(class_rows)})", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=CLASS_COLORS[class_name])
                cursor += len(class_rows)
        cbar = fig.colorbar(image, ax=ax, pad=0.14)
        cbar.set_label("dF/F", fontsize=11)
        fig.text(0.01, 0.01, f"White ticks = detected calcium-event onsets ({event_detection_method}); cyan lines = class boundaries", fontsize=8.5)
        fig.subplots_adjust(left=0.08, right=0.78, bottom=0.13, top=0.91)
        directory = output_root / _safe_filename(compartment)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"dff_heatmap_{_safe_filename(state)}"
        for output_format in output_formats:
            output_path = directory / f"{stem}.{output_format}"
            save_figure(fig, output_path, extra_formats=())
            saved.append(str(output_path))
        metadata = {
            "compartment": compartment,
            "state": state,
            "event_detection_method": event_detection_method,
            "sorting": ["class", "event_frequency_per_min_desc", "mean_dff_desc", "roi_id_asc"],
            "color_scale": {"vmin": float(low), "vmax": float(high)},
            "rois": [
                {
                    "roi_id": _safe_id(row.get("roi_id") or row.get("roi_key") or row.get("unit_id")),
                    "class": row.get("split_group"),
                    "state": row.get("state"),
                    "event_frequency_per_min": row.get("event_frequency_per_min"),
                    "mean_dff": row.get("mean_dff"),
                    "event_onsets_s": np.asarray(row.get("event_onsets", []), dtype=float).tolist(),
                }
                for row in selected
            ],
        }
        metadata_path = directory / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
        saved.append(str(metadata_path))
        plt.close(fig)
    return saved


def records_from_soma_contexts(contexts: Sequence[Any], states: Sequence[str], split_rows: Sequence[Mapping[str, Any]], event_detection_method: str) -> list[dict[str, Any]]:
    lookup = _class_lookup(split_rows)
    records: list[dict[str, Any]] = []
    from analysis.shared.analysis_families.state import state_masks_for_context
    for ctx in contexts:
        time = shared_time_axis(ctx)
        masks = state_masks_for_context(ctx, states)
        for compartment, entity in (("soma", ctx.soma), ("bouton", ctx.bouton)):
            matrix = np.asarray(entity.matrix(), dtype=float)
            ids = list(entity.roi_ids()) if hasattr(entity, "roi_ids") else list(range(matrix.shape[0]))
            for roi_index, trace in enumerate(matrix):
                roi_id = ids[roi_index] if roi_index < len(ids) else roi_index
                unit_id = (make_global_soma_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=ctx.soma_channel, roi_id=roi_id) if compartment == "soma" else make_global_bouton_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=ctx.bouton_channel, roi_id=roi_id))
                for state, mask in masks.items():
                    usable = min(time.size, trace.size, np.asarray(mask).size)
                    masked_trace = np.asarray(trace[:usable], dtype=float).copy()
                    masked_trace[~np.asarray(mask[:usable], dtype=bool)] = np.nan
                    state_key = canonical_state_label(state)
                    group = lookup.get((compartment, unit_id, state_key), lookup.get((compartment, unit_id, ""), ""))
                    records.append({"compartment": compartment, "state": state_key, "roi_id": unit_id, "time": time[:usable], "trace": masked_trace, "split_group": group})
    return records


def records_from_dendrite_cache(cache: Mapping[str, Any], states: Sequence[str], split_rows: Sequence[Mapping[str, Any]], event_detection_method: str) -> list[dict[str, Any]]:
    lookup = _class_lookup(split_rows)
    records: list[dict[str, Any]] = []
    for animal_id, animal in (cache.get("animals", {}) or {}).items():
        for dendrite_id, dendrite in (animal.get("dendrites", {}) or {}).items():
            for exp_id, obs in (dendrite.get("observations", {}) or {}).items():
                exp_meta = (cache.get("experiments", {}) or {}).get(exp_id, {})
                masks = exp_meta.get("state_masks", {}) if isinstance(exp_meta, Mapping) else {}
                for compartment, roi_id, source in [("dendrite", dendrite_id, obs)]:
                    records.extend(_dendrite_observation_records(source, compartment, roi_id, masks, states, lookup, event_detection_method))
                for spine_id, spine in (dendrite.get("spines", {}) or {}).items():
                    spine_obs = (spine.get("observations", {}) or {}).get(exp_id)
                    if spine_obs is not None:
                        records.extend(_dendrite_observation_records(spine_obs, "spine", spine_id, masks, states, lookup, event_detection_method))
    return records


def _dendrite_observation_records(source: Mapping[str, Any], compartment: str, roi_id: str, masks: Mapping[str, Any], states: Sequence[str], lookup: Mapping[tuple[str, str, str], str], event_detection_method: str) -> list[dict[str, Any]]:
    trace = source.get("spine_specific") if compartment == "spine" and source.get("spine_specific") is not None else source.get("trace")
    time = np.asarray(source.get("time", []), dtype=float).reshape(-1)
    trace = np.asarray(trace if trace is not None else [], dtype=float).reshape(-1)
    usable = min(time.size, trace.size)
    rows: list[dict[str, Any]] = []
    for state in states:
        state_key = canonical_state_label(state)
        mask = np.asarray(masks.get(state_key, np.ones(usable, dtype=bool)), dtype=bool)[:usable]
        masked_trace = trace[:usable].copy()
        masked_trace[~mask] = np.nan
        rows.append({"compartment": compartment, "state": state_key, "roi_id": roi_id, "time": time[:usable], "trace": masked_trace, "event_info": source.get("event_info"), "split_group": lookup.get((compartment, _safe_id(roi_id), state_key), lookup.get((compartment, _safe_id(roi_id), ""), ""))})
    return rows


__all__ = ["render_dff_heatmaps", "records_from_soma_contexts", "records_from_dendrite_cache"]
