from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analysis.dendrites_pipeline.dendrites_pipeline import (
    apply_bonferroni_correction,
    build_state_masks_movie,
    build_state_masks_sleep,
    choose_locomotion_threshold,
    extract_series_bundle,
    independent_comparison,
    interpolate_series,
    paired_comparison,
)
from analysis.compartment_common import read_pickle
from analysis.shared.roi_split import summarize_mask_duration
from analysis.shared.state_utils import canonical_state_label, state_display_color, state_display_label
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


def activity_rows_for_context(
    ctx: ExperimentContext,
    selected_states: Sequence[str],
    state_masks: Mapping[str, np.ndarray] | None = None,
) -> List[Dict[str, Any]]:
    masks = state_masks if state_masks is not None else state_masks_for_context(ctx, selected_states)
    soma_matrix = np.asarray(ctx.soma.matrix(), dtype=float)
    bouton_matrix = np.asarray(ctx.bouton.matrix(), dtype=float)
    soma_roi_ids = _bundle_roi_ids(ctx.soma, soma_matrix.shape[0])
    bouton_roi_ids = _bundle_roi_ids(ctx.bouton, bouton_matrix.shape[0])
    time = shared_time_axis(ctx)
    rows: List[Dict[str, Any]] = []
    for state, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        state_n_frames, state_duration_s = summarize_mask_duration(time, mask)
        soma_summary = summarize_activity(soma_matrix, mask)
        bouton_summary = summarize_activity(bouton_matrix, mask)
        for compartment, summary, matrix, roi_ids in (("soma", soma_summary, soma_matrix, soma_roi_ids), ("bouton", bouton_summary, bouton_matrix, bouton_roi_ids)):
            if matrix.size == 0:
                continue
            for roi_index in range(matrix.shape[0]):
                trace = np.asarray(matrix[roi_index], dtype=float)
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
                    "state_n_frames": int(state_n_frames),
                    "state_duration_s": float(state_duration_s),
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
    has_split_groups = any(_split_group_value(row) is not None for row in rows)
    split_meta = _split_group_meta(rows) if has_split_groups else {}
    for row in rows:
        split_group = _split_group_value(row) if has_split_groups else None
        key = (
            row["day_id"],
            row["mode"],
            row["state"],
            row["compartment"],
            split_group,
        )
        grouped.setdefault(key, []).append(float(row["mean"]))
        payload = {
            "day_id": row["day_id"],
            "mode": row["mode"],
            "state": row["state"],
            "state_display": row["state_display"],
            "state_color": row["state_color"],
            "compartment": row["compartment"],
        }
        if has_split_groups:
            payload["split_group"] = split_group
            payload["split_group_display"] = None
            payload["split_group_color"] = None
            payload["split_group_rank"] = None
            if split_group is not None:
                split_group_meta = split_meta.get(split_group, {})
                if split_group_meta.get("split_group_display") is not None:
                    payload["split_group_display"] = split_group_meta.get("split_group_display")
                if split_group_meta.get("split_group_color") is not None:
                    payload["split_group_color"] = split_group_meta.get("split_group_color")
                if split_group_meta.get("split_group_rank") is not None:
                    payload["split_group_rank"] = split_group_meta.get("split_group_rank")
        meta[key] = payload
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


def _split_group_value(row: Mapping[str, Any]) -> str | None:
    split_group = str(row.get("split_group") or "").strip()
    return split_group or None


def _split_group_order(rows: Sequence[Mapping[str, Any]]) -> List[str | None]:
    split_groups: List[str | None] = []
    has_unassigned = False
    for row in rows:
        split_group = _split_group_value(row)
        if split_group is None:
            has_unassigned = True
            continue
        if split_group not in split_groups:
            split_groups.append(split_group)
    if not split_groups:
        return [None]
    if has_unassigned:
        split_groups.append(None)
    return split_groups


def _split_group_meta(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        split_group = _split_group_value(row)
        if split_group is None:
            continue
        payload = meta.setdefault(split_group, {})
        display = str(row.get("split_group_display") or split_group).strip() or split_group
        if display and "split_group_display" not in payload:
            payload["split_group_display"] = display
        color = str(row.get("split_group_color") or "").strip()
        if color and "split_group_color" not in payload:
            payload["split_group_color"] = color
        rank_value = row.get("split_group_rank")
        try:
            rank_float = float(rank_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rank_float):
            continue
        current_rank = payload.get("split_group_rank")
        if current_rank is None or rank_float < float(current_rank):
            payload["split_group_rank"] = rank_float
    return meta

def build_state_comparison_row_groups(
    rows: Sequence[Mapping[str, Any]],
    selected_states: Sequence[str],
) -> Dict[str | None, Dict[str, Dict[str, List[Mapping[str, Any]]]]]:
    selected = [state for state in selected_states if canonical_state_label(state)]
    selected_lookup = set(selected)
    if len(selected_lookup) < 2:
        return {}

    grouped_rows: Dict[str | None, Dict[str, Dict[str, List[Mapping[str, Any]]]]] = {
        None: {state: {} for state in selected},
        "soma": {state: {} for state in selected},
        "bouton": {state: {} for state in selected},
    }
    for row in rows:
        state = canonical_state_label(row.get("state"))
        if state not in selected_lookup:
            continue
        day_id = str(row.get("day_id") or "")
        if not day_id:
            continue
        grouped_rows[None].setdefault(state, {}).setdefault(day_id, []).append(row)
        compartment = str(row.get("compartment") or "")
        if compartment in {"soma", "bouton"}:
            grouped_rows[compartment].setdefault(state, {}).setdefault(day_id, []).append(row)
    return grouped_rows




def state_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_states: Sequence[str],
    shuffle_n: int,
    *,
    metric_col: str = "mean",
    grouped_rows: Mapping[str | None, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]] | None = None,
) -> List[Dict[str, Any]]:
    selected = [state for state in selected_states if canonical_state_label(state)]
    if len(selected) < 2:
        return []
    comparisons: List[Dict[str, Any]] = []
    has_split_groups = any(_split_group_value(row) is not None for row in rows)
    split_groups = _split_group_order(rows) if has_split_groups else [None]
    split_meta = _split_group_meta(rows) if has_split_groups else {}
    if not has_split_groups:
        if grouped_rows is None:
            grouped_rows = build_state_comparison_row_groups(rows, selected)
        for compartment in (None, "soma", "bouton"):
            compartment_rows = grouped_rows.get(compartment, {})
            if not any(compartment_rows.values()):
                continue
            values_by_state: Dict[str, Dict[str, List[float]]] = {
                state: {
                    subject_id: [float(row.get(metric_col, float("nan"))) for row in member_rows]
                    for subject_id, member_rows in subject_rows.items()
                }
                for state, subject_rows in compartment_rows.items()
            }
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
                    comparisons.append(result)
        return comparisons
    for split_group in split_groups:
        split_rows = [row for row in rows if _split_group_value(row) == split_group]
        if not split_rows:
            continue
        split_grouped_rows = build_state_comparison_row_groups(split_rows, selected)
        for compartment in (None, "soma", "bouton"):
            compartment_rows = split_grouped_rows.get(compartment, {})
            if not any(compartment_rows.values()):
                continue
            values_by_state: Dict[str, Dict[str, List[float]]] = {
                state: {
                    subject_id: [float(row.get(metric_col, float("nan"))) for row in member_rows]
                    for subject_id, member_rows in subject_rows.items()
                }
                for state, subject_rows in compartment_rows.items()
            }
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
                    if split_group is not None:
                        result["split_group"] = split_group
                        meta = split_meta.get(split_group, {})
                        if meta.get("split_group_display") is not None:
                            result["split_group_display"] = meta.get("split_group_display")
                        if meta.get("split_group_color") is not None:
                            result["split_group_color"] = meta.get("split_group_color")
                        if meta.get("split_group_rank") is not None:
                            result["split_group_rank"] = meta.get("split_group_rank")
                    comparisons.append(result)
    return comparisons



def basal_apical_comparison_rows(rows: Sequence[Mapping[str, Any]], selected_states: Sequence[str], shuffle_n: int) -> List[Dict[str, Any]]:
    selected = [state for state in selected_states if canonical_state_label(state)]
    if not selected:
        return []
    comparisons: List[Dict[str, Any]] = []
    has_split_groups = any(_split_group_value(row) is not None for row in rows)
    split_groups = _split_group_order(rows) if has_split_groups else [None]
    split_meta = _split_group_meta(rows) if has_split_groups else {}
    for split_group in split_groups:
        split_rows = [row for row in rows if _split_group_value(row) == split_group] if has_split_groups else rows
        if not split_rows:
            continue
        for state in selected:
            values_by_state = _state_values_by_subject(split_rows, [state])
            if state not in values_by_state:
                continue
            result = {
                "comparison": "state_summary",
                "state": state,
                "metric": "mean",
                "n_subjects": len(values_by_state[state]),
                "mean": float(np.nanmean([float(v) for values in values_by_state[state].values() for v in values])) if values_by_state[state] else float("nan"),
            }
            if split_group is not None:
                result["split_group"] = split_group
                meta = split_meta.get(split_group, {})
                if meta.get("split_group_display") is not None:
                    result["split_group_display"] = meta.get("split_group_display")
                if meta.get("split_group_color") is not None:
                    result["split_group_color"] = meta.get("split_group_color")
                if meta.get("split_group_rank") is not None:
                    result["split_group_rank"] = meta.get("split_group_rank")
            elif has_split_groups:
                result["split_group"] = None
            comparisons.append(result)
    return comparisons

def build_state_family_results(rows: Sequence[Mapping[str, Any]], selected_states: Sequence[str], shuffle_n: int) -> Dict[str, Any]:
    summary_rows = state_summary_rows(rows)
    comparison_groups = build_state_comparison_row_groups(rows, selected_states)
    comparisons = state_comparison_rows(rows, selected_states, shuffle_n, grouped_rows=comparison_groups)
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
    "build_state_comparison_row_groups",
    "build_state_family_results",
    "state_comparison_rows",
    "state_masks_for_context",
    "state_summary_rows",
]
