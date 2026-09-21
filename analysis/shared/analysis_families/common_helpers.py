from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from analysis.compartment_common import find_first_key, read_pickle
from analysis.shared.state_utils import canonical_state_label, combined_movie_state_label


def interpolate_series(target_t: np.ndarray, source_t: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    target = np.asarray(target_t, dtype=float)
    source_t = np.asarray(source_t, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    valid = np.isfinite(source_t) & np.isfinite(source_y)
    if target.size == 0 or valid.sum() == 0:
        return np.full(target.shape, np.nan, dtype=float)
    source_t, source_y = source_t[valid], source_y[valid]
    order = np.argsort(source_t)
    source_t, source_y = source_t[order], source_y[order]
    if source_t.size == 1:
        return np.full(target.shape, source_y[0], dtype=float)
    return np.interp(target, source_t, source_y, left=source_y[0], right=source_y[-1])


def extract_series_bundle(path: Path, signal_priority: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    bundle = read_pickle(path)
    if isinstance(bundle, Mapping):
        time = find_first_key(bundle, ["t", "time", "timeline"])
        series = find_first_key(bundle, signal_priority)
        if series is None:
            candidates = [value for value in bundle.values() if isinstance(value, (list, tuple, np.ndarray))]
            if not candidates:
                raise KeyError(f"No numeric series found in {path}")
            series = candidates[0]
        values = np.asarray(series, dtype=float)
        axis = np.arange(values.size, dtype=float) if time is None else np.asarray(time, dtype=float)
        return axis, values, dict(bundle)
    values = np.asarray(bundle, dtype=float)
    if values.ndim == 1:
        return np.arange(values.size, dtype=float), values, {"series": values}
    raise TypeError(f"Unexpected series bundle type in {path}")


def choose_locomotion_threshold(explicit_threshold: Optional[float], thresholds: Sequence[float], wheel: Optional[np.ndarray]) -> float:
    if explicit_threshold is not None and np.isfinite(explicit_threshold):
        return float(explicit_threshold)
    finite = [float(value) for value in thresholds if np.isfinite(value)]
    if finite:
        return float(np.median(finite))
    if wheel is not None and np.isfinite(wheel).any():
        return float(np.nanmedian(np.abs(wheel)))
    return 0.0


def build_state_masks_sleep(exp_time: np.ndarray, sleep_state: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    time = np.asarray(exp_time, dtype=float)
    state_time = np.asarray(sleep_state.get("state_10hz_t", []), dtype=float)
    state_codes = np.asarray(sleep_state.get("state_10hz", []), dtype=float)
    codes = np.full(time.shape, -1, dtype=int)
    if state_time.size and state_codes.size:
        inside = (time >= state_time.min()) & (time <= state_time.max())
        if inside.any():
            codes[inside] = np.rint(np.interp(time[inside], state_time, state_codes)).astype(int)
    labels = {0: "quiet_awake", 1: "nrem", 2: "rem", 3: "active_awake"}
    masks = {label: codes == code for code, label in labels.items()}
    masks["all"] = np.ones(time.shape, dtype=bool)
    return masks, {"state_labels": labels, "state_codes_on_calcium_time": codes}


def build_state_masks_movie(exp_time: np.ndarray, trial_rows: Sequence[Mapping[str, Any]], columns: Sequence[str], wheel_time: Optional[np.ndarray], wheel_speed: Optional[np.ndarray], sleep_state: Optional[Mapping[str, Any]], locomotion_threshold: float) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], Optional[np.ndarray]]:
    del columns, sleep_state
    time = np.asarray(exp_time, dtype=float)
    wheel = interpolate_series(time, wheel_time, wheel_speed) if wheel_time is not None and wheel_speed is not None else None
    masks: Dict[str, np.ndarray] = {"all": np.ones(time.shape, dtype=bool)}
    metadata: List[Dict[str, Any]] = []
    for index, row in enumerate(trial_rows):
        label = canonical_state_label(row.get("state_label") or row.get("state") or row.get("trial_type") or "movies")
        start = row.get("start", row.get("onset", row.get("trial_start")))
        end = row.get("end", row.get("trial_end"))
        try:
            start_value = float(start)
            end_value = float(end) if end is not None else start_value
        except (TypeError, ValueError):
            continue
        mask = (time >= start_value) & (time <= end_value)
        key = label or "movies"
        masks.setdefault(key, np.zeros(time.shape, dtype=bool))
        masks[key] |= mask
        metadata.append({"trial_index": index, "state_label": key, "start": start_value, "end": end_value, "locomotion_threshold": locomotion_threshold})
    return masks, metadata, wheel


def paired_comparison(values_by_state: Mapping[str, Mapping[str, Sequence[float]]], state_a: str, state_b: str, metric_name: str, shuffle_n: int) -> Dict[str, Any]:
    del shuffle_n
    subjects = sorted(set(values_by_state.get(state_a, {})) & set(values_by_state.get(state_b, {})))
    a = np.asarray([np.nanmean(values_by_state[state_a][subject]) for subject in subjects], dtype=float)
    b = np.asarray([np.nanmean(values_by_state[state_b][subject]) for subject in subjects], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    result = stats.ttest_rel(a, b, nan_policy="omit") if a.size >= 2 else None
    return {"metric": metric_name, "state_a": state_a, "state_b": state_b, "paired": True, "n_subjects": int(a.size), "effect_size": float(np.nanmean(a - b)) if a.size else float("nan"), "classical_p": float(result.pvalue) if result is not None else float("nan")}


def independent_comparison(values_by_state: Mapping[str, Mapping[str, Sequence[float]]], state_a: str, state_b: str, metric_name: str, shuffle_n: int) -> Dict[str, Any]:
    del shuffle_n
    a = np.asarray([np.nanmean(value) for value in values_by_state.get(state_a, {}).values()], dtype=float)
    b = np.asarray([np.nanmean(value) for value in values_by_state.get(state_b, {}).values()], dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    result = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit") if a.size >= 2 and b.size >= 2 else None
    return {"metric": metric_name, "state_a": state_a, "state_b": state_b, "paired": False, "n_subjects": int(min(a.size, b.size)), "effect_size": float(np.nanmean(a) - np.nanmean(b)) if a.size and b.size else float("nan"), "classical_p": float(result.pvalue) if result is not None else float("nan")}


def apply_bonferroni_correction(records: List[Dict[str, Any]]) -> int:
    valid = [record for record in records if record.get("available") and np.isfinite(record.get("raw_pvalue", np.nan))]
    for record in records:
        record["adjusted_pvalue"] = float("nan")
        record["significant"] = False
        record["star"] = ""
    for record in valid:
        adjusted = min(float(record["raw_pvalue"]) * len(valid), 1.0)
        record["adjusted_pvalue"] = adjusted
        record["significant"] = adjusted < 0.05
        record["star"] = "*" if record["significant"] else ""
    return len(valid)


def build_event_info(trace: np.ndarray, time: Optional[np.ndarray] = None, *, method: str = "derivative", include_all_methods: bool = True) -> Dict[str, Any]:
    values = np.asarray(trace, dtype=float)
    axis = np.arange(values.size, dtype=float) if time is None else np.asarray(time, dtype=float)
    finite = np.isfinite(values)
    baseline = float(np.nanmedian(values[finite])) if finite.any() else 0.0
    scale = float(np.nanstd(values[finite])) if finite.any() else 0.0
    threshold = baseline + 3.0 * (scale if scale > 0 else 1.0)
    if method == "derivative":
        if values.size < 2:
            signal = np.zeros_like(values, dtype=float)
        else:
            signal = np.abs(np.gradient(np.nan_to_num(values, nan=baseline)))
        threshold = float(np.nanmedian(signal) + 3.0 * (np.nanstd(signal) or 1.0)) if signal.size else float("nan")
    else:
        signal = values
    active = np.isfinite(signal) & (signal > threshold)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    runs = [(int(start), int(start)) for start in starts]
    duration = float(np.nanmax(axis) - np.nanmin(axis)) if axis.size > 1 and np.isfinite(axis).any() else float(values.size)
    frequency = float(len(runs) * 60.0 / duration) if duration > 0 else float("nan")
    result = {
        "event_count": int(len(runs)), "event_runs": runs, "event_frequency_per_min": frequency,
        "duration_seconds": duration, "threshold": threshold, "method": method,
        "primary_method": method, "event_detection_methods": ["amplitude", "derivative"],
    }
    if include_all_methods:
        result["methods"] = {name: build_event_info(values, axis, method=name, include_all_methods=False) for name in ("amplitude", "derivative")}
    return result


def trial_activity_means(trace: np.ndarray, time: np.ndarray, duration_s: Optional[float]) -> Tuple[float, float]:
    values = np.asarray(trace, dtype=float)
    axis = np.asarray(time, dtype=float)
    baseline = values[np.isfinite(axis) & (axis < 0)]
    end = float(duration_s) if duration_s is not None and np.isfinite(duration_s) else np.inf
    stimulus = values[np.isfinite(axis) & (axis >= 0) & (axis < end)]
    return (float(np.nanmean(baseline)) if baseline.size else float("nan"), float(np.nanmean(stimulus)) if stimulus.size else float("nan"))


def welch_ttest_summary(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(values_a, dtype=float); b = np.asarray(values_b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    result = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit") if a.size >= 2 and b.size >= 2 else None
    p_value = float(result.pvalue) if result is not None and np.isfinite(result.pvalue) else float("nan")
    return {"available": result is not None, "comparison": "stimulus_vs_blank", "statistic": float(result.statistic) if result is not None else float("nan"), "raw_pvalue": p_value, "adjusted_pvalue": p_value, "n_a": int(a.size), "n_b": int(b.size), "significant": False, "star": ""}


def extract_cut_neural_bundle(path: Path, preferred_keys: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    bundle = read_pickle(path)
    if isinstance(bundle, Mapping):
        time = find_first_key(bundle, ["t", "time", "timeline"])
        keys = list(preferred_keys or []) + ["dF", "dff", "dF/F", "df", "signal"]
        values = find_first_key(bundle, keys)
        if values is None:
            values = next((value for value in bundle.values() if isinstance(value, (list, tuple, np.ndarray))), None)
        array = np.asarray(values, dtype=float)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D cut-neural array, got {array.shape}")
        axis = np.arange(array.shape[-1], dtype=float) if time is None else np.asarray(time, dtype=float)
        return axis, array, dict(bundle)
    array = np.asarray(bundle, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D cut-neural array, got {array.shape}")
    return np.arange(array.shape[-1], dtype=float), array, {"dF": array}
