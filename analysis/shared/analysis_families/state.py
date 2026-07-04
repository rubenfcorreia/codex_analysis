from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analysis.main_pipeline.sleep_dendrite_spine_pipeline import (
    apply_bonferroni_correction,
    build_state_masks_movie,
    build_state_masks_sleep,
    choose_locomotion_threshold,
    extract_series_bundle,
    independent_comparison,
    interpolate_series,
    paired_comparison,
)
from analysis.compartment_common import canonical_state_label, read_pickle, state_display_color, state_display_label
from analysis.shared.shared_calcium_response import build_masked_event_summary

from .core import ExperimentContext, make_global_bouton_id, make_global_soma_id, make_unit_id, shared_time_axis, summarize_activity


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _movie_masks_for_context(ctx: ExperimentContext) -> Dict[str, np.ndarray]:
    exp_time = shared_time_axis(ctx)
    if exp_time.size == 0:
        return {}

    trial_rows: List[Dict[str, str]] = []
    if isinstance(ctx.state_bundle, Mapping):
        rows = ctx.state_bundle.get("rows", [])
        if isinstance(rows, Sequence):
            trial_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not trial_rows:
        return {"all": np.ones(exp_time.shape, dtype=bool)}

    columns = list(trial_rows[0].keys())

    wheel_time = wheel_speed = None
    wheel_path = ctx.exp_root / "recordings" / "wheel.pickle"
    if wheel_path.exists():
        try:
            wheel_time, wheel_speed, _ = extract_series_bundle(wheel_path, ["speed", "wheel", "motion", "velocity"])
        except Exception:
            wheel_time = wheel_speed = None

    sleep_state = None
    sleep_path = ctx.exp_root / "sleep_score" / "sleep_state.pickle"
    if sleep_path.exists():
        try:
            sleep_state = read_pickle(sleep_path)
        except Exception:
            sleep_state = None

    sleep_thresholds: List[float] = []
    if isinstance(sleep_state, Mapping):
        threshold = _float_or_none(sleep_state.get("locomotion_threshold"))
        if threshold is not None:
            sleep_thresholds.append(threshold)

    wheel_interp = None
    if wheel_time is not None and wheel_speed is not None:
        wheel_interp = interpolate_series(exp_time, wheel_time, wheel_speed)
    locomotion_threshold = choose_locomotion_threshold(None, sleep_thresholds, wheel_interp)

    masks, _, _ = build_state_masks_movie(
        exp_time,
        trial_rows,
        columns,
        wheel_time,
        wheel_speed,
        sleep_state if isinstance(sleep_state, Mapping) else None,
        locomotion_threshold,
    )
    return masks


def _sleep_masks_for_context(ctx: ExperimentContext) -> Dict[str, np.ndarray]:
    exp_time = shared_time_axis(ctx)
    if exp_time.size == 0:
        return {}
    sleep_state = ctx.state_bundle if isinstance(ctx.state_bundle, Mapping) else {}
    masks, _ = build_state_masks_sleep(exp_time, dict(sleep_state))
    return masks


def state_masks_for_context(ctx: ExperimentContext, selected_states: Sequence[str]) -> Dict[str, np.ndarray]:
    if ctx.mode == "movie":
        masks = _movie_masks_for_context(ctx)
    else:
        masks = _sleep_masks_for_context(ctx)

    if not selected_states:
        return masks

    ordered: Dict[str, np.ndarray] = {}
    mask_by_canonical = {
        canonical_state_label(state): (state, mask)
        for state, mask in masks.items()
        if canonical_state_label(state)
    }

    for requested_state in selected_states:
        requested_key = canonical_state_label(requested_state)
        if not requested_key:
            continue
        match = mask_by_canonical.get(requested_key)
        if match is None:
            continue
        state, mask = match
        ordered[state] = mask

    if "all" in masks and any(canonical_state_label(state) == "all" for state in selected_states):
        ordered.setdefault("all", masks["all"])

    return ordered


def _event_summary_for_trace(trace: np.ndarray, time: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    trace = np.asarray(trace, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    usable = min(trace.size, time.size, mask.size)
    if usable <= 0:
        return {"event_count": 0, "event_frequency_per_min": float("nan")}
    summary = build_masked_event_summary(trace[:usable], time[:usable], mask[:usable])
    return {
        "event_count": int(summary.get("event_count", 0) or 0),
        "event_frequency_per_min": float(summary.get("event_frequency_per_min", float("nan"))),
    }


def _summarize_roi_trace(trace: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    trace = np.asarray(trace, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    usable = min(trace.size, mask.size) if mask.size else trace.size
    if usable <= 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    if mask.size:
        masked = trace[:usable][mask[:usable]]
    else:
        masked = trace[:usable]
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(np.nanmean(finite)),
        "median": float(np.nanmedian(finite)),
        "std": float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
    }


def _roi_subject_id(row: Mapping[str, Any]) -> str:
    unit_id = row.get("unit_id")
    if unit_id is not None and str(unit_id).strip():
        return str(unit_id)
    for key in ("global_soma_id", "global_bouton_id", "roi_key"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    roi_id = row.get("roi_id")
    if roi_id is not None and str(roi_id).strip():
        return str(roi_id)
    compartment = str(row.get("compartment") or "").strip().lower()
    compartment_id = row.get(f"{compartment}_id") if compartment in {"soma", "bouton"} else None
    if compartment_id is not None and str(compartment_id).strip():
        return str(compartment_id)
    roi_index = row.get("roi_index")
    if roi_index is None:
        roi_index = row.get("bouton_roi_index")
    expid = str(row.get("expid") or "")
    if expid or roi_index is not None:
        return f"{expid}:{compartment}:{roi_index}"
    return ""


def _bundle_roi_ids(bundle: Any, n_rows: int) -> List[Any]:
    roi_ids: List[Any] = []
    if hasattr(bundle, "roi_ids"):
        try:
            roi_ids = list(bundle.roi_ids())
        except Exception:
            roi_ids = []
    if not roi_ids:
        roi_ids = list(range(n_rows))
    if len(roi_ids) < n_rows:
        roi_ids.extend(range(len(roi_ids), n_rows))
    return roi_ids[:n_rows]


def activity_rows_for_context(ctx: ExperimentContext, selected_states: Sequence[str]) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
    soma_matrix = np.asarray(ctx.soma.matrix(), dtype=float)
    bouton_matrix = np.asarray(ctx.bouton.matrix(), dtype=float)
    soma_roi_ids = _bundle_roi_ids(ctx.soma, soma_matrix.shape[0])
    bouton_roi_ids = _bundle_roi_ids(ctx.bouton, bouton_matrix.shape[0])
    time = shared_time_axis(ctx)
    rows: List[Dict[str, Any]] = []
    for state, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        for compartment, matrix, roi_ids in (("soma", soma_matrix, soma_roi_ids), ("bouton", bouton_matrix, bouton_roi_ids)):
            if matrix.size == 0:
                continue
            for roi_index in range(matrix.shape[0]):
                trace = np.asarray(matrix[roi_index], dtype=float)
                summary = _summarize_roi_trace(trace, mask)
                events = _event_summary_for_trace(trace, time, mask)
                roi_id = roi_ids[roi_index] if roi_index < len(roi_ids) else roi_index
                channel = ctx.soma_channel if compartment == "soma" else ctx.bouton_channel
                unit_id = make_unit_id(
                    animal_id=ctx.animal_id,
                    expid=ctx.expid,
                    day_id=ctx.day_id,
                    compartment=compartment,
                    channel=channel,
                    roi_id=roi_id,
                    roi_index=int(roi_index),
                )
                global_soma_id = make_global_soma_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=channel, roi_id=roi_id)
                global_bouton_id = make_global_bouton_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=channel, roi_id=roi_id)
                row = {
                    "expid": ctx.expid,
                    "mode": ctx.mode,
                    "animal_id": ctx.animal_id,
                    "date": ctx.date,
                    "day_id": ctx.day_id,
                    "channel": int(channel),
                    "state": canonical_state_label(state),
                    "state_display": state_display_label(state),
                    "state_color": state_display_color(state),
                    "compartment": compartment,
                    "roi_index": int(roi_index),
                    "roi_id": roi_id,
                    "unit_id": unit_id,
                    "roi_key": unit_id,
                    **summary,
                    "event_count": events["event_count"],
                    "event_frequency_per_min": events["event_frequency_per_min"],
                }
                if compartment == "soma":
                    row["soma_id"] = roi_id
                    row["bouton_id"] = None
                    row["global_soma_id"] = global_soma_id
                    row["global_bouton_id"] = None
                else:
                    row["soma_id"] = None
                    row["bouton_id"] = roi_id
                    row["global_soma_id"] = None
                    row["global_bouton_id"] = global_bouton_id
                rows.append(row)
    return rows


def state_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[tuple, List[float]] = {}
    meta: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (
            row["day_id"],
            row["mode"],
            row["state"],
            row["compartment"],
        )
        grouped.setdefault(key, []).append(float(row["mean"]))
        meta[key] = {
            "day_id": row["day_id"],
            "mode": row["mode"],
            "state": row["state"],
            "state_display": row["state_display"],
            "state_color": row["state_color"],
            "compartment": row["compartment"],
        }
    summary_rows: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        payload = meta[key].copy()
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        total_rois = int(arr.size)
        if finite.size == 0:
            payload.update({
                "n_experiments": int(arr.size),
                "n_rois": total_rois,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            })
        else:
            payload.update({
                "n_experiments": int(arr.size),
                "n_rois": total_rois,
                "mean": float(np.nanmean(finite)),
                "median": float(np.nanmedian(finite)),
                "std": float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0,
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
            })
        summary_rows.append(payload)
    return summary_rows


def _state_values_by_subject(
    rows: Sequence[Mapping[str, Any]],
    selected_states: Sequence[str],
    compartment: str | None = None,
    metric_col: str = "mean",
) -> Dict[str, Dict[str, List[float]]]:
    selected_lookup = {canonical_state_label(state) for state in selected_states if canonical_state_label(state)}
    values_by_state: Dict[str, Dict[str, List[float]]] = {state: {} for state in selected_lookup}
    for row in rows:
        state = canonical_state_label(row.get("state"))
        if state not in selected_lookup:
            continue
        if compartment is not None and str(row.get("compartment") or "") != compartment:
            continue
        subject_id = str(row.get("unit_id") or row.get("roi_key") or row.get("global_soma_id") or row.get("global_bouton_id") or row.get("roi_id") or row.get("soma_id") or row.get("bouton_id") or _roi_subject_id(row) or "")
        if not subject_id:
            continue
        values_by_state.setdefault(state, {}).setdefault(subject_id, []).append(float(row.get(metric_col, float("nan"))))
    return values_by_state


def state_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_states: Sequence[str],
    shuffle_n: int,
    *,
    metric_col: str = "mean",
) -> List[Dict[str, Any]]:
    selected = [state for state in selected_states if canonical_state_label(state)]
    if len(selected) < 2:
        return []
    comparisons: List[Dict[str, Any]] = []
    for compartment in (None, "soma", "bouton"):
        values_by_state = _state_values_by_subject(rows, selected, compartment=compartment, metric_col=metric_col)
        if not any(values_by_state.values()):
            continue
        for idx, state_a in enumerate(selected):
            for state_b in selected[idx + 1:]:
                subjects_a = values_by_state.get(state_a, {})
                subjects_b = values_by_state.get(state_b, {})
                subjects = sorted(set(subjects_a).intersection(subjects_b))
                if len(subjects) >= 2:
                    result = paired_comparison(values_by_state, state_a, state_b, metric_col, shuffle_n)
                else:
                    result = independent_comparison(values_by_state, state_a, state_b, metric_col, shuffle_n)
                result["comparison"] = "state_pair"
                result["compartment"] = compartment or "all"
                result["state_a_display"] = state_a
                result["state_b_display"] = state_b
                result["metric"] = metric_col
                comparisons.append(result)
    return comparisons
def basal_apical_comparison_rows(rows: Sequence[Mapping[str, Any]], selected_states: Sequence[str], shuffle_n: int) -> List[Dict[str, Any]]:
    selected = [state for state in selected_states if canonical_state_label(state)]
    if not selected:
        return []
    comparisons: List[Dict[str, Any]] = []
    for state in selected:
        values_by_state = _state_values_by_subject(rows, [state])
        if state not in values_by_state:
            continue
        result = {
            "comparison": "state_summary",
            "state": state,
            "metric": "mean",
            "n_subjects": len(values_by_state[state]),
            "mean": float(np.nanmean([float(v) for values in values_by_state[state].values() for v in values])) if values_by_state[state] else float("nan"),
        }
        comparisons.append(result)
    return comparisons


def build_state_family_results(rows: Sequence[Mapping[str, Any]], selected_states: Sequence[str], shuffle_n: int) -> Dict[str, Any]:
    summary_rows = state_summary_rows(rows)
    comparisons = state_comparison_rows(rows, selected_states, shuffle_n)
    basal_apical = basal_apical_comparison_rows(rows, selected_states, shuffle_n)
    return {
        "activity_rows": list(rows),
        "summary_rows": summary_rows,
        "state_comparison_rows": comparisons,
        "basal_apical_comparison_rows": basal_apical,
    }

__all__ = [
    "ExperimentContext",
    "activity_rows_for_context",
    "basal_apical_comparison_rows",
    "build_state_family_results",
    "state_comparison_rows",
    "state_masks_for_context",
    "state_summary_rows",
]
