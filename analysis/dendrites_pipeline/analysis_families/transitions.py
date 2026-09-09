"""Preset-specific state-transition metrics for dendrite/spine data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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

from analysis.dendrites_pipeline.dendrites_pipeline import (
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
            "compartment": compartment,
            "entity_id": entity_id,
            "metric": metric,
            "pre_value": float(pre_value) if np.isfinite(pre_value) else float("nan"),
            "post_value": float(post_value) if np.isfinite(post_value) else float("nan"),
        }
    )
    return payload


def run_transition_analysis(
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
    }
    if not transition_config["enabled"]:
        return result
    event_rows: list[dict[str, Any]] = []
    trace_segments: list[dict[str, Any]] = []
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
                            if d_trace is not None and d_time is not None:
                                dendrite_id_text = str(dendrite_id)
                                if "activity" in transition_config["metrics"]:
                                    pre = _window_values(d_trace, d_time, event, "pre")
                                    post = _window_values(d_trace, d_time, event, "post")
                                    event_rows.append(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(d_obs.get("day_id") or exp_id), compartment=str(compartment), entity_id=dendrite_id_text, metric="dendrite_mean", pre_value=float(np.nanmean(pre)) if pre.size else float("nan"), post_value=float(np.nanmean(post)) if post.size else float("nan")))
                                    relative, values = aligned_trace_segment(d_trace, d_time, event)
                                    trace_segments.append({"scope": scope, "window_mode": window_mode, "state_before": event["state_before"], "state_after": event["state_after"], "metric": "dendrite_mean", "compartment": str(compartment), "window_s": event["window_s"], "relative_time_s": relative, "values": values})
                                if "event_frequency" in transition_config["metrics"]:
                                    d_event_info = d_obs.get("event_info") or {}
                                    event_rows.append(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(d_obs.get("day_id") or exp_id), compartment=str(compartment), entity_id=dendrite_id_text, metric="dendrite_event_frequency_per_min", pre_value=_frequency(d_trace, d_time, d_event_info, event, "pre", "event_frequency_per_min"), post_value=_frequency(d_trace, d_time, d_event_info, event, "post", "event_frequency_per_min")))
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
                                    event_rows.append(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(s_obs.get("day_id") or exp_id), compartment=str(observation_compartment(cache, exp_id, s_obs) or compartment), entity_id=entity_id, metric="spine_specific_mean", pre_value=float(np.nanmean(pre)) if pre.size else float("nan"), post_value=float(np.nanmean(post)) if post.size else float("nan")))
                                    relative, values = aligned_trace_segment(s_trace, s_time, event)
                                    trace_segments.append({"scope": scope, "window_mode": window_mode, "state_before": event["state_before"], "state_after": event["state_after"], "metric": "spine_specific_mean", "compartment": str(observation_compartment(cache, exp_id, s_obs) or compartment), "window_s": event["window_s"], "relative_time_s": relative, "values": values})
                                if "event_frequency" in transition_config["metrics"]:
                                    s_event_info = s_obs.get("event_info") or {}
                                    s_event_trace = s_obs.get("trace")
                                    dendrite_event_info = d_obs.get("event_info") or {}
                                    pre_spine = _frequency(s_event_trace, s_time, s_event_info, event, "pre", "spine_event_frequency_per_min")
                                    post_spine = _frequency(s_event_trace, s_time, s_event_info, event, "post", "spine_event_frequency_per_min")
                                    event_rows.append(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(s_obs.get("day_id") or exp_id), compartment=str(observation_compartment(cache, exp_id, s_obs) or compartment), entity_id=entity_id, metric="spine_event_frequency_per_min", pre_value=pre_spine, post_value=post_spine))
                                    for metric in ("coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"):
                                        pre_s = build_state_masked_event_info(s_event_trace, s_time, interval_mask(s_time, float(event["pre_start_time"]), float(event["pre_end_time"])), s_event_info)
                                        post_s = build_state_masked_event_info(s_event_trace, s_time, interval_mask(s_time, float(event["post_start_time"]), float(event["post_end_time"])), s_event_info)
                                        pre_d = build_state_masked_event_info(d_trace, d_time, interval_mask(d_time, float(event["pre_start_time"]), float(event["pre_end_time"])), dendrite_event_info) if d_trace is not None and d_time is not None else {}
                                        post_d = build_state_masked_event_info(d_trace, d_time, interval_mask(d_time, float(event["post_start_time"]), float(event["post_end_time"])), dendrite_event_info) if d_trace is not None and d_time is not None else {}
                                        pre_value = as_float(annotate_spine_event_info(pre_s, pre_d).get(metric))
                                        post_value = as_float(annotate_spine_event_info(post_s, post_d).get(metric))
                                        event_rows.append(_row(event, exp_id=str(exp_id), animal_id=str(animal_id), day_id=str(s_obs.get("day_id") or exp_id), compartment=str(observation_compartment(cache, exp_id, s_obs) or compartment), entity_id=entity_id, metric=metric, pre_value=pre_value if pre_value is not None else float("nan"), post_value=post_value if post_value is not None else float("nan")))
    result["event_rows"] = event_rows
    result["summary_rows"] = paired_transition_summaries(event_rows)
    if output_root is not None:
        result["figure_paths"] = plot_transition_summaries(event_rows, Path(output_root), pipeline_name="dendrites", trace_segments=trace_segments)
    if not event_rows:
        result["alerts"].append("No valid within-experiment state transitions were found for this preset.")
    return result
