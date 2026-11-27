from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analysis.shared.analysis_families.common_helpers import independent_comparison, paired_comparison
from analysis.shared.state_utils import canonical_state_label


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def activity_probability_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source in rows:
        if not isinstance(source, Mapping):
            continue
        row = dict(source)
        valid_frames = _finite_float(row.get("n"))
        active_frames = _finite_float(row.get("event_active_frame_count"))
        probability = float("nan")
        if np.isfinite(valid_frames) and valid_frames > 0 and np.isfinite(active_frames):
            probability = float(np.clip(active_frames / valid_frames, 0.0, 1.0))
        row.update({
            "metric": "event_probability",
            "valid_frame_count": int(valid_frames) if np.isfinite(valid_frames) and valid_frames >= 0 else 0,
            "event_active_frame_count": int(active_frames) if np.isfinite(active_frames) and active_frames >= 0 else 0,
            "event_probability": probability,
        })
        output.append(row)
    return output


def activity_probability_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], List[float]] = {}
    members: Dict[tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("day_id") or ""),
            str(row.get("mode") or ""),
            str(row.get("state") or ""),
            str(row.get("compartment") or ""),
        )
        grouped.setdefault(key, []).append(_finite_float(row.get("event_probability")))
        members.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for (day_id, mode, state, compartment), values in grouped.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        output.append({
            "day_id": day_id, "mode": mode, "state": state, "compartment": compartment,
            "n_rois": int(len(members[(day_id, mode, state, compartment)])),
            "mean_event_probability": float(np.mean(finite)) if finite.size else float("nan"),
            "median_event_probability": float(np.median(finite)) if finite.size else float("nan"),
        })
    return output


def activity_probability_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_states: Sequence[str],
    shuffle_n: int,
) -> List[Dict[str, Any]]:
    selected = [canonical_state_label(state) for state in selected_states if canonical_state_label(state)]
    comparisons: List[Dict[str, Any]] = []
    for compartment in ("all", "soma", "bouton"):
        values: Dict[str, Dict[str, List[float]]] = {state: {} for state in selected}
        for row in rows:
            state = canonical_state_label(row.get("state"))
            row_compartment = str(row.get("compartment") or "").strip().lower()
            if state not in values or (compartment != "all" and row_compartment != compartment):
                continue
            unit = str(row.get("unit_id") or row.get("roi_key") or row.get("roi_id") or "").strip()
            day = str(row.get("day_id") or row.get("expid") or "").strip()
            if not unit or not day:
                continue
            subject = f"{day}:{unit}"
            values[state].setdefault(subject, []).append(_finite_float(row.get("event_probability")))
        for index, state_a in enumerate(selected):
            for state_b in selected[index + 1:]:
                matched = set(values[state_a]) & set(values[state_b])
                if len(matched) >= 2:
                    result = paired_comparison(values, state_a, state_b, "event_probability", int(shuffle_n))
                else:
                    result = independent_comparison(values, state_a, state_b, "event_probability", int(shuffle_n))
                result.update({"comparison": "state_pair", "compartment": compartment, "state_a_display": state_a, "state_b_display": state_b})
                comparisons.append(result)
    return comparisons


def run_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_states: Sequence[str] = (),
    shuffle_n: int = 0,
) -> Dict[str, Any]:
    probability_rows = activity_probability_rows(rows)
    selected = [str(state) for state in selected_states if str(state).strip()]
    comparisons = activity_probability_comparison_rows(
        probability_rows, selected, int(shuffle_n)
    ) if len(selected) >= 2 else []
    return {
        "available": bool(probability_rows),
        "rows": probability_rows,
        "activity_probability_rows": probability_rows,
        "summary_rows": activity_probability_summary_rows(probability_rows),
        "comparison_rows": comparisons,
        "counts": {"all": len(probability_rows)},
    }


__all__ = ["activity_probability_comparison_rows", "activity_probability_rows", "activity_probability_summary_rows", "run_family"]
