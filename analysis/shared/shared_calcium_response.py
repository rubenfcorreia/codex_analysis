from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from analysis.compartment_common import find_first_key, read_pickle
from analysis.shared.cache_utils import stable_hash
from analysis.shared.state_utils import canonical_state_label
from analysis.shared.analysis_families.shared_metrics import (
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
from analysis.shared.analysis_families.common_helpers import (
    apply_bonferroni_correction,
    build_event_info,
    build_state_masks_movie,
    choose_locomotion_threshold,
    classify_movie_name,
    extract_cut_neural_bundle,
    movie_feature_blocks,
    trial_activity_means,
    welch_ttest_summary,
)

VISUAL_RESPONSE_VISUAL_TRIAL_TYPES = ("movies", "gratings", "zebras")
VISUAL_RESPONSE_BLANK_TRIAL_TYPE = "blank"
VISUAL_RESPONSE_COHORTS = ("all", "responsive", "nonresponsive")

DEFAULT_VISUAL_RESPONSE_COHORT = "all"

_VISUAL_RESPONSE_CUT_DATA_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_VISUAL_RESPONSE_CUT_DATA_SCHEMA_VERSION = 2


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
    canonical = canonical.replace("_grating", "_gratings").replace("_zebra", "_zebras")
    if canonical in {"grating", "zebra"}:
        canonical = {"grating": "gratings", "zebra": "zebras"}[canonical]
    if canonical == VISUAL_RESPONSE_BLANK_TRIAL_TYPE or canonical.endswith("_blank"):
        return "blank"
    if canonical in VISUAL_RESPONSE_VISUAL_TRIAL_TYPES or any(
        canonical.endswith(f"_{trial_type}") for trial_type in VISUAL_RESPONSE_VISUAL_TRIAL_TYPES
    ):
        return "visual"
    return None


def _path_signature(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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
    clean_rows = [row for row in trial_rows if isinstance(row, Mapping)]
    columns = list(clean_rows[0].keys()) if clean_rows else []
    cache_key = stable_hash(
        {
            "schema_version": _VISUAL_RESPONSE_CUT_DATA_SCHEMA_VERSION,
            "exp_root": str(exp_root),
            "channel": int(channel),
            "locomotion_threshold": float(locomotion_threshold) if locomotion_threshold is not None else None,
            "selected_path": _path_signature(selected_path),
            "columns": list(columns),
            "trial_rows": clean_rows,
        }
    )
    if cache_key in _VISUAL_RESPONSE_CUT_DATA_CACHE:
        return _VISUAL_RESPONSE_CUT_DATA_CACHE[cache_key]
    cut_time, cut_neural, _ = extract_cut_neural_bundle(selected_path)
    cut_time = np.asarray(cut_time, dtype=float)
    cut_neural = np.asarray(cut_neural, dtype=float)
    if cut_neural.ndim < 3:
        _VISUAL_RESPONSE_CUT_DATA_CACHE[cache_key] = None
        return None
    n_cut_trials = int(cut_neural.shape[1])
    trial_meta: List[Dict[str, Any]] = []
    for trial_index in range(min(n_cut_trials, len(clean_rows))):
        row = clean_rows[trial_index]
        blocks = movie_feature_blocks(row, columns)
        if len(blocks) != 1:
            continue
        video_id = str(blocks[0]["name"])
        category = classify_movie_name(video_id)
        duration = row.get("duration")
        try:
            duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        trial_meta.append(
            {
                "trial_index": trial_index,
                "state_label": category,
                "category": category,
                "movie_trial_type": category,
                "video_id": video_id,
                "duration": duration,
                "trial_row": dict(row),
                "metadata_source": "cut_trial_index",
            }
        )
    result = {
        "schema_version": _VISUAL_RESPONSE_CUT_DATA_SCHEMA_VERSION,
        "cut_time": cut_time,
        "cut_neural": cut_neural,
        "trial_meta": trial_meta,
        "source_label": selected_label,
        "source_path": str(selected_path),
    }
    _VISUAL_RESPONSE_CUT_DATA_CACHE[cache_key] = result
    return result


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
        blank["correction_method"] = "none_single_comparison"
        blank["adjusted_pvalue"] = float(blank.get("raw_pvalue", float("nan")))
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
