"""Preset-specific state-transition metrics for dendrite/spine data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.shared.state_transitions import (
    SLEEP_STATE_LABELS,
    add_window_metadata,
    aggregate_transition_rows_expday,
    aligned_trace_segment,
    interval_mask,
    normalize_transition_config,
    paired_transition_summaries,
    plot_transition_summaries,
    transition_events,
)

class TransitionOverflowError(RuntimeError):
    def __init__(self, *, max_event_rows: int, rows: list[dict[str, Any]], trace_segments: list[dict[str, Any]], context: Mapping[str, Any]):
        self.max_event_rows = int(max_event_rows)
        self.rows = rows
        self.trace_segments = trace_segments
        self.context = dict(context)
        super().__init__(
            f"transition analysis exceeded max_event_rows={self.max_event_rows} "
            f"for {self.context.get('exp_id')}/{self.context.get('entity_id')}/{self.context.get('scope')}"
        )


from analysis.shared.plots.dendrite_plot_support import (
    annotate_spine_event_info,
    as_float,
    build_state_masked_event_info,
    observation_compartment,
    values_from_observation,
)


def _window_values(trace: Any, time: Any, event: Mapping[str, Any], side: str) -> np.ndarray:
    if trace is None or time is None:
        return np.asarray([], dtype=float)
    time_array = np.asarray(time, dtype=float).reshape(-1)
    trace_array = np.asarray(trace, dtype=float).reshape(-1)
    usable = min(time_array.size, trace_array.size)
    if usable <= 0:
        return np.asarray([], dtype=float)
    start = float(event[f"{side}_start_time"])
    end = float(event[f"{side}_end_time"])
    return values_from_observation(trace_array[:usable], interval_mask(time_array[:usable], start, end))


def _frequency(trace: Any, time: Any, event_info: Mapping[str, Any], event: Mapping[str, Any], side: str, metric: str) -> float:
    if trace is None or time is None:
        return float("nan")
    time_array = np.asarray(time, dtype=float).reshape(-1)
    trace_array = np.asarray(trace, dtype=float).reshape(-1)
    usable = min(time_array.size, trace_array.size)
    if usable <= 0:
        return float("nan")
    mask = interval_mask(time_array[:usable], float(event[f"{side}_start_time"]), float(event[f"{side}_end_time"]))
    info = build_state_masked_event_info(trace_array[:usable], time_array[:usable], mask, event_info)
    value = as_float(info.get(metric, info.get("event_frequency_per_min")))
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _row(event: Mapping[str, Any], *, exp_id: str, animal_id: str, day_id: str, compartment: str, entity_id: str, metric: str, pre_value: float, post_value: float) -> dict[str, Any]:
    payload = dict(event)
    payload.update(
        {
            "expid": exp_id,
            "animal_id": animal_id,
            "day_id": day_id,
            "expday": f"{exp_id}|{day_id}",
            "compartment": compartment,
            "entity_id": entity_id,
            "metric": metric,
            "pre_value": float(pre_value) if np.isfinite(pre_value) else float("nan"),
            "post_value": float(post_value) if np.isfinite(post_value) else float("nan"),
        }
    )
    return payload


def _run_transition_analysis_impl(
    cache: Mapping[str, Any],
    selected_states: Sequence[str],
    config: Mapping[str, Any] | None,
    *,
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
        "complete": True,
        "status": "complete",
        "bound_reason": None,
        "max_event_rows": transition_config.get("max_event_rows"),
        "rows_emitted": 0,
    }
    if not transition_config["enabled"]:
        return result
    event_rows: list[dict[str, Any]] = []
    trace_segments: list[dict[str, Any]] = []
    max_event_rows = transition_config.get("max_event_rows")

    def append_event_row(row: dict[str, Any]) -> None:
        if max_event_rows is not None and len(event_rows) >= int(max_event_rows):
            raise TransitionOverflowError(
                max_event_rows=int(max_event_rows),
                rows=event_rows,
                trace_segments=trace_segments,
                context={
                    "exp_id": str(row.get("expid", "")),
                    "entity_id": str(row.get("entity_id", "")),
                    "scope": str(row.get("scope", "")),
                },
            )
        event_rows.append(row)

    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    for animal_id, animal_entry in animals.items():
        for dendrite_id, dendrite_record in (animal_entry.get("dendrites", {}) or {}).items():
            observations = dendrite_record.get("observations", {}) or {}
            spines = dendrite_record.get("spines", {}) or {}
            for exp_id, d_obs in observations.items():
                exp_meta = experiments.get(exp_id) or experiments.get(str(exp_id))
                if not isinstance(exp_meta, Mapping):
                    continue
                time = np.asarray(exp_meta.get("time", []), dtype=float).reshape(-1)
                state_masks = exp_meta.get("state_masks", {})
                if time.size < 2 or not isinstance(state_masks, Mapping):
                    continue
                compartment = observation_compartment(cache, exp_id, d_obs) or str(exp_meta.get("compartment") or "")
                for scope in transition_config["scopes"]:
                    scoped_states = [str(state) for state in selected_states if scope == "all_states" or str(state).lower() in SLEEP_STATE_LABELS]
                    if len(scoped_states) < 2:
                        continue
                    raw_events = transition_events(time, state_masks, scoped_states, scope=scope, window_s=transition_config["window_s"])
                    for event_index, raw_event in enumerate(raw_events):
                        for window_mode in transition_config["window_modes"]:
                            event = add_window_metadata(raw_event, window_mode)
                            if event is None:
                                continue
                            event = dict(event)
                            event["transition_id"] = f"{exp_id}:{dendrite_id}:{scope}:{event_index}"
                            d_trace = d_obs.get("trace")
                            d_time = d_obs.get("time")
                            dendrite_id_text = str(dendrite_id)
                            dendrite_pre_event_info: dict[str, Any] = {}
                            dendrite_post_event_info: dict[str, Any] = {}
                            if "event_frequency" in transition_config["metrics"] and d_trace is not None and d_time is not None:
                                dendrite_event_info = d_obs.get("event_info") or {}
                                dendrite_pre_event_info = build_state_masked_event_info(
                                    d_trace, d_time,
                                    interval_mask(d_time, float(event["pre_start_time"]), float(event["pre_end_time"])),
                                    dendrite_event_info, include_all_methods=False,
                                )
                                dendrite_post_event_info = build_state_masked_event_info(
                                    d_trace, d_time,
                                    interval_mask(d_time, float(event["post_start_time"]), float(event["post_end_time"])),
                                    dendrite_event_info, include_all_methods=False,
                                )
                            if d_trace is not None and d_time is not None:
                                if "activity" in transition_config["metrics"]:
                                    pre = _window_values(d_trace, d_time, event, "pre")
                                    post = _window_values(d_trace, d_time, event, "post")
                                    append_event_row(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(d_obs.get("day_id") or exp_id), compartment=str(compartment), entity_id=dendrite_id_text, metric="dendrite_mean", pre_value=float(np.nanmean(pre)) if pre.size else float("nan"), post_value=float(np.nanmean(post)) if post.size else float("nan")))
                                    relative, values = aligned_trace_segment(d_trace, d_time, event)
                                    trace_segments.append({"scope": scope, "window_mode": window_mode, "state_before": event["state_before"], "state_after": event["state_after"], "metric": "dendrite_mean", "compartment": str(compartment), "window_s": event["window_s"], "relative_time_s": relative, "values": values, "expday": f"{exp_id}|{d_obs.get('day_id') or exp_id}", "entity_id": dendrite_id_text, "animal_id": str(animal_id)})
                                if "event_frequency" in transition_config["metrics"]:
                                    dendrite_pre_frequency = as_float(dendrite_pre_event_info.get("event_frequency_per_min"))
                                    dendrite_post_frequency = as_float(dendrite_post_event_info.get("event_frequency_per_min"))
                                    append_event_row(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(d_obs.get("day_id") or exp_id), compartment=str(compartment), entity_id=dendrite_id_text, metric="dendrite_event_frequency_per_min", pre_value=dendrite_pre_frequency if dendrite_pre_frequency is not None else float("nan"), post_value=dendrite_post_frequency if dendrite_post_frequency is not None else float("nan")))
                            for spine_id in d_obs.get("spine_ids", []) or []:
                                s_obs = (spines.get(spine_id, {}) or {}).get("observations", {}).get(exp_id)
                                if not isinstance(s_obs, Mapping):
                                    continue
                                s_trace = s_obs.get("spine_specific")
                                s_time = s_obs.get("time")
                                if s_trace is None or s_time is None:
                                    continue
                                entity_id = str(spine_id)
                                if "activity" in transition_config["metrics"]:
                                    pre = _window_values(s_trace, s_time, event, "pre")
                                    post = _window_values(s_trace, s_time, event, "post")
                                    append_event_row(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(s_obs.get("day_id") or exp_id), compartment=str(observation_compartment(cache, exp_id, s_obs) or compartment), entity_id=entity_id, metric="spine_specific_mean", pre_value=float(np.nanmean(pre)) if pre.size else float("nan"), post_value=float(np.nanmean(post)) if post.size else float("nan")))
                                    relative, values = aligned_trace_segment(s_trace, s_time, event)
                                    trace_segments.append({"scope": scope, "window_mode": window_mode, "state_before": event["state_before"], "state_after": event["state_after"], "metric": "spine_specific_mean", "compartment": str(observation_compartment(cache, exp_id, s_obs) or compartment), "window_s": event["window_s"], "relative_time_s": relative, "values": values, "expday": f"{exp_id}|{s_obs.get('day_id') or exp_id}", "entity_id": entity_id, "animal_id": str(animal_id)})
                                if "event_frequency" in transition_config["metrics"]:
                                    s_event_info = s_obs.get("event_info") or {}
                                    s_event_trace = s_obs.get("trace")
                                    pre_mask = interval_mask(s_time, float(event["pre_start_time"]), float(event["pre_end_time"]))
                                    post_mask = interval_mask(s_time, float(event["post_start_time"]), float(event["post_end_time"]))
                                    pre_s = build_state_masked_event_info(s_event_trace, s_time, pre_mask, s_event_info, include_all_methods=False)
                                    post_s = build_state_masked_event_info(s_event_trace, s_time, post_mask, s_event_info, include_all_methods=False)
                                    pre_d = dendrite_pre_event_info
                                    post_d = dendrite_post_event_info
                                    pre_spine = as_float(pre_s.get("event_frequency_per_min"))
                                    post_spine = as_float(post_s.get("event_frequency_per_min"))
                                    append_event_row(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(s_obs.get("day_id") or exp_id), compartment=str(observation_compartment(cache, exp_id, s_obs) or compartment), entity_id=entity_id, metric="spine_event_frequency_per_min", pre_value=pre_spine if pre_spine is not None else float("nan"), post_value=post_spine if post_spine is not None else float("nan")))
                                    for metric in ("coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"):
                                        pre_value = as_float(annotate_spine_event_info(pre_s, pre_d).get(metric))
                                        post_value = as_float(annotate_spine_event_info(post_s, post_d).get(metric))
                                        append_event_row(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(s_obs.get("day_id") or exp_id), compartment=str(observation_compartment(cache, exp_id, s_obs) or compartment), entity_id=entity_id, metric=metric, pre_value=pre_value if pre_value is not None else float("nan"), post_value=post_value if post_value is not None else float("nan")))
    result["event_rows"] = event_rows
    result["rows_emitted"] = int(len(event_rows))
    result["pooled_summary_rows"] = paired_transition_summaries(event_rows)
    balanced_rows = aggregate_transition_rows_expday(event_rows)
    result["summary_rows"] = paired_transition_summaries(balanced_rows)
    if output_root is not None:
        result["figure_paths"] = plot_transition_summaries(event_rows, Path(output_root), pipeline_name="dendrites", trace_segments=trace_segments)
    if not event_rows:
        result["alerts"].append("No valid within-experiment state transitions were found for this preset.")
    return result



def run_transition_analysis(
    cache: Mapping[str, Any],
    selected_states: Sequence[str],
    config: Mapping[str, Any] | None,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    normalized = normalize_transition_config(config)
    try:
        return _run_transition_analysis_impl(cache, selected_states, normalized, output_root=output_root)
    except TransitionOverflowError as exc:
        if normalized.get("transition_overflow_policy") != "partial":
            raise
        result: dict[str, Any] = {
            "schema_version": normalized["schema_version"],
            "config": normalized,
            "event_rows": list(exc.rows),
            "pooled_summary_rows": paired_transition_summaries(exc.rows),
            "summary_rows": paired_transition_summaries(aggregate_transition_rows_expday(exc.rows)),
            "figure_paths": [],
            "alerts": [],
            "complete": False,
            "status": "bounded",
            "bound_reason": "max_event_rows_exceeded",
            "max_event_rows": int(exc.max_event_rows),
            "rows_emitted": int(len(exc.rows)),
            "overflow_context": dict(exc.context),
        }
        result["alerts"].append(
            f"[BOUNDED] Transition analysis stopped after {len(exc.rows)} rows at max_event_rows={exc.max_event_rows} "
            f"for {exc.context.get('exp_id')}/{exc.context.get('entity_id')}/{exc.context.get('scope')}."
        )
        if output_root is not None and exc.rows:
            result["figure_paths"] = plot_transition_summaries(
                exc.rows, Path(output_root), pipeline_name="dendrites", trace_segments=exc.trace_segments
            )
        return result
