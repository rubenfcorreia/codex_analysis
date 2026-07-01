from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from analysis.compartment_common import canonical_state_label, find_first_key, read_pickle
from analysis.main_pipeline.analysis_families.shared_metrics import (
    DEFAULT_EVENT_DETECTION_METHOD,
    DEFAULT_VISUAL_RESPONSE_METRIC,
    EVENT_DETECTION_METHODS,
    VISUAL_RESPONSE_METRICS,
    get_active_event_detection_method,
    get_active_visual_response_metric,
    normalize_event_detection_method,
    normalize_visual_response_metric,
    set_active_event_detection_method,
    set_active_visual_response_metric,
    visual_response_metric_field,
    visual_response_metric_label,
)
from analysis.main_pipeline.sleep_dendrite_spine_pipeline import (
    apply_bonferroni_correction,
    build_event_info,
    build_state_masks_movie,
    choose_locomotion_threshold,
    extract_cut_neural_bundle,
    trial_activity_means,
    welch_ttest_summary,
)


VISUAL_RESPONSE_VISUAL_TRIAL_TYPES = ("movies", "gratings", "zebras")
VISUAL_RESPONSE_BLANK_TRIAL_TYPE = "blank"
VISUAL_RESPONSE_COHORTS = ("all", "responsive", "nonresponsive")

DEFAULT_VISUAL_RESPONSE_COHORT = "all"


def event_frequency_per_minute(event_count: int, duration_seconds: float) -> float:
    if duration_seconds is None or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        return float("nan")
    return float(event_count) * 60.0 / float(duration_seconds)


def estimate_trace_step_seconds(time: np.ndarray) -> float:
    try:
        time = np.asarray(time, dtype=float).ravel()
    except Exception:
        return float("nan")
    finite = time[np.isfinite(time)]
    if finite.size <= 1:
        return float("nan")
    diffs = np.diff(finite)
    valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if valid_diffs.size == 0:
        return float("nan")
    return float(np.nanmedian(valid_diffs))


def estimate_trace_duration_seconds(time: np.ndarray) -> float:
    try:
        time = np.asarray(time, dtype=float).ravel()
    except Exception:
        return float("nan")
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return float(1.0)
    diffs = np.diff(finite)
    valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    step = float(np.nanmedian(valid_diffs)) if valid_diffs.size else float("nan")
    if not np.isfinite(step) or step <= 0:
        return float(finite.size)
    return float(max(finite[-1] - finite[0] + step, step))


def build_event_summary(
    trace: np.ndarray,
    time: Optional[np.ndarray] = None,
    *,
    method: Optional[str] = None,
    include_all_methods: bool = True,
) -> Dict[str, Any]:
    trace = np.asarray(trace, dtype=float)
    method = get_active_event_detection_method(method)
    event_info = build_event_info(trace, time, method=method, include_all_methods=include_all_methods)
    event_info["primary_method"] = method
    event_info["event_detection_methods"] = list(EVENT_DETECTION_METHODS)
    return event_info


def build_masked_event_summary(
    trace: np.ndarray,
    time: Optional[np.ndarray],
    mask: np.ndarray,
    *,
    method: Optional[str] = None,
    threshold: Optional[float] = None,
    min_consecutive_frames: int = 3,
    include_all_methods: bool = True,
) -> Dict[str, Any]:
    trace = np.asarray(trace, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    method = get_active_event_detection_method(method)
    if trace.size == 0 or mask.size != trace.size or not np.any(mask):
        event_info = build_event_info(np.asarray([], dtype=float), None, method=method, include_all_methods=include_all_methods)
        event_info["duration_seconds"] = float("nan")
        event_info["event_frequency_per_min"] = float("nan")
        event_info["min_consecutive_frames"] = int(min_consecutive_frames)
        event_info["sigma_factor"] = 3.0
        event_info["primary_method"] = method
        event_info["event_detection_methods"] = list(EVENT_DETECTION_METHODS)
        return event_info
    masked_trace = trace[mask]
    event_info = build_event_info(masked_trace, time, method=method, include_all_methods=include_all_methods)
    step_seconds = estimate_trace_step_seconds(time if time is not None else np.arange(trace.size, dtype=float))
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        duration_seconds = float(masked_trace.size)
    else:
        duration_seconds = float(np.isfinite(masked_trace).sum() * step_seconds)
    event_info["duration_seconds"] = float(duration_seconds)
    event_info["event_frequency_per_min"] = event_frequency_per_minute(int(event_info.get("event_count", 0) or 0), duration_seconds)
    event_info["min_consecutive_frames"] = int(min_consecutive_frames)
    event_info["sigma_factor"] = 3.0
    event_info["primary_method"] = method
    event_info["event_detection_methods"] = list(EVENT_DETECTION_METHODS)
    return event_info


def visual_response_trial_group(state_label: Any) -> Optional[str]:
    canonical = canonical_state_label(state_label)
    if not canonical:
        return None
    if canonical == VISUAL_RESPONSE_BLANK_TRIAL_TYPE or canonical.endswith("_blank"):
        return "blank"
    if canonical in VISUAL_RESPONSE_VISUAL_TRIAL_TYPES or any(
        canonical.endswith(f"_{trial_type}") for trial_type in VISUAL_RESPONSE_VISUAL_TRIAL_TYPES
    ):
        return "visual"
    return None


def load_visual_response_cut_data(
    exp_root: Path,
    channel: int,
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    locomotion_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    selected_path: Optional[Path] = None
    selected_label: Optional[str] = None
    for candidate_path, candidate_label in (
        (exp_root / "cut_intertrials" / f"s2p_ch{channel}_dF_cut.pickle", "cut_intertrials"),
        (exp_root / "cut_with_intertrials" / f"s2p_ch{channel}_dF_cut.pickle", "cut_with_intertrials"),
    ):
        if candidate_path.exists():
            selected_path = candidate_path
            selected_label = candidate_label
            break
    if selected_path is None:
        return None
    cut_time, cut_neural, _ = extract_cut_neural_bundle(selected_path)
    cut_time = np.asarray(cut_time, dtype=float)
    cut_neural = np.asarray(cut_neural, dtype=float)
    clean_rows = [dict(row) for row in trial_rows if isinstance(row, Mapping)]
    columns = list(clean_rows[0].keys()) if clean_rows else []
    wheel_path_candidates = (
        exp_root / "cut" / "wheel.pickle",
        exp_root / "cut_intertrials" / "wheel.pickle",
        exp_root / "cut_with_intertrials" / "wheel.pickle",
    )
    wheel_bundle = None
    for wheel_path in wheel_path_candidates:
        if wheel_path.exists():
            try:
                wheel_bundle = read_pickle(wheel_path)
            except Exception:
                wheel_bundle = None
            break
    wheel_time = find_first_key(wheel_bundle, ["t", "time", "timestamps"]) if isinstance(wheel_bundle, Mapping) else None
    wheel_speed = find_first_key(wheel_bundle, ["speed", "wheel", "motion", "velocity"]) if isinstance(wheel_bundle, Mapping) else None
    wheel_interp = None
    if wheel_time is not None and wheel_speed is not None:
        wheel_interp = np.asarray(np.interp(cut_time, np.asarray(wheel_time, dtype=float), np.asarray(wheel_speed, dtype=float)), dtype=float)
    threshold = choose_locomotion_threshold(
        locomotion_threshold,
        [],
        wheel_interp,
    )
    _, trial_meta, _ = build_state_masks_movie(
        cut_time,
        clean_rows,
        columns,
        None,
        None,
        None,
        threshold,
    )
    return {
        "cut_time": cut_time,
        "cut_neural": cut_neural,
        "trial_meta": trial_meta,
        "source_label": selected_label,
        "source_path": str(selected_path),
    }


def visual_response_trial_rows(
    trace: np.ndarray,
    cut_time: np.ndarray,
    trial_meta: Sequence[Mapping[str, Any]],
    *,
    response_metric: Optional[str] = None,
    event_detection_method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    metric = get_active_visual_response_metric(response_metric)
    event_method = get_active_event_detection_method(event_detection_method)
    trace = np.asarray(trace, dtype=float)
    cut_time = np.asarray(cut_time, dtype=float)
    rows: List[Dict[str, Any]] = []
    for meta in trial_meta:
        if not isinstance(meta, Mapping):
            continue
        trial_label = canonical_state_label(meta.get("state_label"))
        group = visual_response_trial_group(trial_label)
        if group is None:
            continue
        trial_index = int(meta.get("trial_index")) if meta.get("trial_index") is not None else None
        if trial_index is None or trial_index < 0 or trial_index >= trace.shape[0]:
            continue
        trial_trace = np.asarray(trace[trial_index], dtype=float)
        duration_s = float(meta.get("duration")) if meta.get("duration") is not None else None
        if metric == "mean":
            baseline, stimulus = trial_activity_means(trial_trace, cut_time, duration_s)
        else:
            baseline_mask = np.isfinite(trial_trace) & np.isfinite(cut_time) & (cut_time < 0)
            stimulus_mask = np.isfinite(trial_trace) & np.isfinite(cut_time) & (cut_time >= 0)
            if duration_s is not None and np.isfinite(duration_s):
                stimulus_mask &= cut_time < float(duration_s)
            baseline_info = build_event_summary(trial_trace[baseline_mask], cut_time[baseline_mask], method=event_method, include_all_methods=False)
            stimulus_info = build_event_summary(trial_trace[stimulus_mask], cut_time[stimulus_mask], method=event_method, include_all_methods=False)
            baseline = float(baseline_info.get("event_frequency_per_min", float("nan")))
            stimulus = float(stimulus_info.get("event_frequency_per_min", float("nan")))
        if not np.isfinite(stimulus):
            continue
        rows.append(
            {
                "group": group,
                "trial_label": trial_label,
                "baseline": float(baseline),
                "response": float(stimulus),
                "response_metric": metric,
            }
        )
    return rows


def summarize_visual_response_trials(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    response_metric = get_active_visual_response_metric(rows[0].get("response_metric") if rows else None)
    visual_trial_labels: List[str] = []
    blank_trial_labels: List[str] = []
    visual_values: List[float] = []
    blank_values: List[float] = []
    for row in rows:
        trial_label = canonical_state_label(row.get("trial_label"))
        group = str(row.get("group") or "")
        stimulus_value = row.get("response")
        if stimulus_value is None or not np.isfinite(float(stimulus_value)):
            continue
        if group == "visual":
            if trial_label and trial_label not in visual_trial_labels:
                visual_trial_labels.append(trial_label)
            visual_values.append(float(stimulus_value))
        elif group == "blank":
            if trial_label and trial_label not in blank_trial_labels:
                blank_trial_labels.append(trial_label)
            blank_values.append(float(stimulus_value))
    visual_arr = np.asarray(visual_values, dtype=float)
    blank_arr = np.asarray(blank_values, dtype=float)
    mean_visual = float(np.nanmean(visual_arr)) if visual_arr.size else float("nan")
    mean_blank = float(np.nanmean(blank_arr)) if blank_arr.size else float("nan")
    delta = mean_visual - mean_blank if visual_arr.size and blank_arr.size else float("nan")
    blank = welch_ttest_summary(visual_arr, blank_arr)
    if blank.get("available"):
        apply_bonferroni_correction([blank])
    responsive = bool(blank.get("significant", False) and np.isfinite(delta) and float(delta) > 0)
    return {
        "available": bool(visual_arr.size and blank_arr.size),
        "comparison": "visual_response_movie_vs_blank",
        "response_metric": response_metric,
        "statistic": float(blank.get("statistic", float("nan"))),
        "raw_pvalue": float(blank.get("raw_pvalue", float("nan"))),
        "adjusted_pvalue": float(blank.get("adjusted_pvalue", float("nan"))),
        "n_visual_values": int(visual_arr.size),
        "n_blank_values": int(blank_arr.size),
        "mean_visual": mean_visual,
        "mean_blank": mean_blank,
        "delta": delta,
        "paired_stimulus_values": visual_values,
        "blank_reference_values": blank_values,
        "visual_trial_labels": visual_trial_labels,
        "blank_trial_labels": blank_trial_labels,
        "stimulus_vs_blank": blank,
        "significant": bool(blank.get("significant", False)),
        "star": str(blank.get("star", "")),
        "responsive": responsive,
        "cohort": "responsive" if responsive else "nonresponsive",
    }


def summarize_visual_response_entity_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = summarize_visual_response_trials(rows)
    summary["tested"] = int(summary.get("n_visual_values", 0) > 0 and summary.get("n_blank_values", 0) > 0)
    return summary
