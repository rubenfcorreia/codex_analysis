"""Preset-specific state-transition metrics for the soma/bouton pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.shared.shared_calcium_response import build_masked_event_summary
from analysis.shared.state_transitions import (
    SLEEP_STATE_LABELS,
    add_window_metadata,
    aligned_trace_segment,
    interval_mask,
    normalize_transition_config,
    paired_transition_summaries,
    plot_transition_summaries,
    transition_events,
)

from .core import ExperimentContext, make_global_bouton_id, make_global_soma_id, shared_time_axis
from .state import state_masks_for_context


def _trace_window_value(trace: np.ndarray, time: np.ndarray, start: float, end: float) -> float:
    usable = min(trace.size, time.size)
    if usable <= 0:
        return float("nan")
    values = trace[:usable][interval_mask(time[:usable], start, end)]
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _trace_window_frequency(trace: np.ndarray, time: np.ndarray, start: float, end: float, method: str) -> float:
    usable = min(trace.size, time.size)
    if usable <= 0:
        return float("nan")
    time_slice = time[:usable]
    mask = interval_mask(time_slice, start, end)
    info = build_masked_event_summary(trace[:usable], time_slice, mask, method=method)
    try:
        value = float(info.get("event_frequency_per_min", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _append_entity_rows(
    rows: list[dict[str, Any]],
    *,
    ctx: ExperimentContext,
    event: Mapping[str, Any],
    trace: np.ndarray,
    time: np.ndarray,
    compartment: str,
    entity_id: str,
    event_method: str,
    metrics: Sequence[str],
    trace_segments: list[dict[str, Any]],
) -> None:
    for metric_group in metrics:
        metric_names = ["activity"] if metric_group == "activity" else ["event_frequency"]
        for metric in metric_names:
            if metric == "activity":
                pre_value = _trace_window_value(trace, time, float(event["pre_start_time"]), float(event["pre_end_time"]))
                post_value = _trace_window_value(trace, time, float(event["post_start_time"]), float(event["post_end_time"]))
                metric_name = "mean_activity"
            else:
                pre_value = _trace_window_frequency(trace, time, float(event["pre_start_time"]), float(event["pre_end_time"]), event_method)
                post_value = _trace_window_frequency(trace, time, float(event["post_start_time"]), float(event["post_end_time"]), event_method)
                metric_name = "event_frequency_per_min"
            row = dict(event)
            row.update(
                {
                    "expid": ctx.expid,
                    "mode": ctx.mode,
                    "animal_id": ctx.animal_id,
                    "date": ctx.date,
                    "day_id": ctx.day_id,
                    "compartment": compartment,
                    "entity_id": entity_id,
                    "metric": metric_name,
                    "pre_value": pre_value,
                    "post_value": post_value,
                    "event_detection_method": event_method,
                }
            )
            rows.append(row)
            if metric == "activity":
                relative, values = aligned_trace_segment(trace, time, event)
                trace_segments.append({"scope": event["scope"], "window_mode": event["window_mode"], "state_before": event["state_before"], "state_after": event["state_after"], "metric": metric_name, "compartment": compartment, "window_s": event["window_s"], "relative_time_s": relative, "values": values})


def run_transition_analysis(
    contexts: Sequence[ExperimentContext],
    selected_states_by_mode: Mapping[str, Sequence[str]],
    config: Mapping[str, Any] | None,
    *,
    event_detection_method: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    transition_config = normalize_transition_config(config)
    result: dict[str, Any] = {
        "schema_version": transition_config["schema_version"],
        "config": transition_config,
        "event_rows": [],
        "summary_rows": [],
        "figure_paths": [],
        "alerts": [],
    }
    if not transition_config["enabled"]:
        return result
    event_rows: list[dict[str, Any]] = []
    trace_segments: list[dict[str, Any]] = []
    for ctx in contexts:
        selected = [str(state) for state in selected_states_by_mode.get(ctx.mode, []) if str(state).strip()]
        if len(selected) < 2:
            continue
        masks = state_masks_for_context(ctx, selected)
        time = np.asarray(shared_time_axis(ctx), dtype=float).reshape(-1)
        if time.size < 2 or not masks:
            continue
        for scope in transition_config["scopes"]:
            scoped_states = selected if scope == "all_states" else [state for state in selected if state.lower() in SLEEP_STATE_LABELS]
            if len(scoped_states) < 2:
                continue
            raw_events = transition_events(time, masks, scoped_states, scope=scope, window_s=transition_config["window_s"])
            for event_index, raw_event in enumerate(raw_events):
                for window_mode in transition_config["window_modes"]:
                    event = add_window_metadata(raw_event, window_mode)
                    if event is None:
                        continue
                    event = dict(event)
                    event["transition_id"] = f"{ctx.expid}:{scope}:{event_index}"
                    event["window_mode"] = window_mode
                    soma_matrix = np.asarray(ctx.soma.matrix(), dtype=float)
                    bouton_matrix = np.asarray(ctx.bouton.matrix(), dtype=float)
                    soma_time = np.asarray(ctx.soma.t, dtype=float).reshape(-1)
                    bouton_time = np.asarray(ctx.bouton.t, dtype=float).reshape(-1)
                    soma_ids = list(ctx.soma.roi_ids()) if hasattr(ctx.soma, "roi_ids") else list(range(soma_matrix.shape[0]))
                    bouton_ids = list(ctx.bouton.roi_ids()) if hasattr(ctx.bouton, "roi_ids") else list(range(bouton_matrix.shape[0]))
                    for index, trace in enumerate(soma_matrix):
                        roi_id = soma_ids[index] if index < len(soma_ids) else index
                        entity_id = make_global_soma_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=ctx.soma_channel, roi_id=roi_id)
                        _append_entity_rows(event_rows, ctx=ctx, event=event, trace=np.asarray(trace, dtype=float), time=soma_time, compartment="soma", entity_id=entity_id, event_method=event_detection_method, metrics=transition_config["metrics"], trace_segments=trace_segments)
                    for index, trace in enumerate(bouton_matrix):
                        roi_id = bouton_ids[index] if index < len(bouton_ids) else index
                        entity_id = make_global_bouton_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=ctx.bouton_channel, roi_id=roi_id)
                        _append_entity_rows(event_rows, ctx=ctx, event=event, trace=np.asarray(trace, dtype=float), time=bouton_time, compartment="bouton", entity_id=entity_id, event_method=event_detection_method, metrics=transition_config["metrics"], trace_segments=trace_segments)
    result["event_rows"] = event_rows
    result["summary_rows"] = paired_transition_summaries(event_rows)
    if output_root is not None:
        result["figure_paths"] = plot_transition_summaries(event_rows, Path(output_root), pipeline_name="soma_bouton", trace_segments=trace_segments)
    if not event_rows:
        result["alerts"].append("No valid within-experiment state transitions were found for this preset.")
    return result
