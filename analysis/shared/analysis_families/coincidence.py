from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from analysis.shared.state_utils import canonical_state_label, state_display_color, state_display_label


def _coerce_duration_seconds(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(duration) or duration <= 0:
        return float("nan")
    return float(duration)


def event_frequency_per_minute(event_count: int, duration_seconds: float) -> float:
    if duration_seconds is None or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        return float("nan")
    return float(event_count) * 60.0 / float(duration_seconds)


def event_run_onsets_match(run_a: Sequence[Any], run_b: Sequence[Any]) -> bool:
    try:
        return int(run_a[0]) == int(run_b[0])
    except Exception:
        return False


def _normalize_event_runs(runs: Any) -> List[Tuple[int, int]]:
    normalized: List[Tuple[int, int]] = []
    for run in runs or []:
        try:
            start, end = run
            normalized.append((int(start), int(end)))
        except Exception:
            continue
    normalized.sort(key=lambda item: (item[0], item[1]))
    return normalized


def coincident_event_runs(source_event_info: Mapping[str, Any] | None, reference_event_info: Mapping[str, Any] | None) -> List[Tuple[int, int]]:
    source_runs = _normalize_event_runs((source_event_info or {}).get("event_runs"))
    reference_onsets = {int(run[0]) for run in _normalize_event_runs((reference_event_info or {}).get("event_runs"))}
    return [run for run in source_runs if int(run[0]) in reference_onsets]


def _finite_values(values: Sequence[Any]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _finite_mean(values: Sequence[Any]) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.nanmean(finite))


def _finite_median(values: Sequence[Any]) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.nanmedian(finite))


def annotate_event_coincidence(
    source_event_info: Mapping[str, Any] | None,
    reference_event_info: Mapping[str, Any] | None,
    *,
    source_name: str = "source",
    reference_name: str = "reference",
) -> Dict[str, Any]:
    source_name = str(source_name).strip() or "source"
    reference_name = str(reference_name).strip() or "reference"
    event_info: Dict[str, Any] = dict(source_event_info or {})
    source_runs = _normalize_event_runs(event_info.get("event_runs"))
    reference_event_info = dict(reference_event_info or {})
    reference_runs = _normalize_event_runs(reference_event_info.get("event_runs"))
    reference_onsets = {int(start) for start, _ in reference_runs}
    coincident_runs = [run for run in source_runs if int(run[0]) in reference_onsets]

    source_event_count = int(event_info.get("event_count", len(source_runs)) or 0)
    reference_event_count = int(reference_event_info.get("event_count", len(reference_runs)) or 0)
    coincident_count = int(len(coincident_runs))
    noncoincident_count = int(max(source_event_count - coincident_count, 0))

    duration_seconds = _coerce_duration_seconds(event_info.get("duration_seconds"))
    if not np.isfinite(duration_seconds):
        duration_seconds = _coerce_duration_seconds(reference_event_info.get("duration_seconds"))

    source_event_frequency = event_frequency_per_minute(source_event_count, duration_seconds)
    reference_event_frequency = event_frequency_per_minute(reference_event_count, duration_seconds)
    coincident_event_frequency = event_frequency_per_minute(coincident_count, duration_seconds)
    noncoincident_event_frequency = event_frequency_per_minute(noncoincident_count, duration_seconds)
    coincident_fraction = float(coincident_count / source_event_count) if source_event_count > 0 else float("nan")
    directional_prefix = f"{source_name}_to_{reference_name}"

    event_info[f"{source_name}_event_count"] = source_event_count
    event_info[f"{reference_name}_event_count"] = reference_event_count
    event_info[f"{source_name}_event_frequency_per_min"] = source_event_frequency
    event_info[f"{reference_name}_event_frequency_per_min"] = reference_event_frequency
    event_info["coincident_event_count"] = coincident_count
    event_info["noncoincident_event_count"] = noncoincident_count
    event_info["coincident_event_fraction"] = coincident_fraction
    event_info["coincident_event_frequency_per_min"] = coincident_event_frequency
    event_info["noncoincident_event_frequency_per_min"] = noncoincident_event_frequency
    event_info["coincident"] = bool(coincident_count > 0)
    event_info["coincident_event_runs"] = coincident_runs
    event_info["noncoincident_event_runs"] = [run for run in source_runs if run not in coincident_runs]
    event_info[f"{directional_prefix}_coincident_event_count"] = coincident_count
    event_info[f"{directional_prefix}_noncoincident_event_count"] = noncoincident_count
    event_info[f"{directional_prefix}_coincident_event_fraction"] = coincident_fraction
    event_info[f"{directional_prefix}_coincident_event_frequency_per_min"] = coincident_event_frequency
    event_info[f"{directional_prefix}_noncoincident_event_frequency_per_min"] = noncoincident_event_frequency
    event_info[f"{directional_prefix}_coincident"] = bool(coincident_count > 0)
    event_info[f"{directional_prefix}_coincident_event_runs"] = coincident_runs
    event_info[f"{directional_prefix}_noncoincident_event_runs"] = [run for run in source_runs if run not in coincident_runs]

    methods = event_info.get("methods")
    if isinstance(methods, Mapping):
        reference_methods = reference_event_info.get("methods") if isinstance(reference_event_info, Mapping) else {}
        annotated_methods: Dict[str, Any] = {}
        for method_name, method_source_info in methods.items():
            method_reference_info = None
            if isinstance(reference_methods, Mapping):
                method_reference_info = reference_methods.get(method_name)
            if method_reference_info is None:
                method_reference_info = reference_event_info
            annotated_methods[str(method_name)] = annotate_event_coincidence(
                method_source_info if isinstance(method_source_info, Mapping) else {},
                method_reference_info if isinstance(method_reference_info, Mapping) else {},
                source_name=source_name,
                reference_name=reference_name,
            )
        event_info["methods"] = annotated_methods

    return event_info


def annotate_spine_event_info(spine_event_info: Mapping[str, Any] | None, dendrite_event_info: Mapping[str, Any] | None) -> Dict[str, Any]:
    return annotate_event_coincidence(
        spine_event_info,
        dendrite_event_info,
        source_name="spine",
        reference_name="dendrite",
    )


def build_bidirectional_coincidence_metrics(
    first_event_info: Mapping[str, Any] | None,
    second_event_info: Mapping[str, Any] | None,
    *,
    first_name: str = "soma",
    second_name: str = "bouton",
) -> Dict[str, Any]:
    first_name = str(first_name).strip() or "source"
    second_name = str(second_name).strip() or "reference"
    forward = annotate_event_coincidence(first_event_info, second_event_info, source_name=first_name, reference_name=second_name)
    reverse = annotate_event_coincidence(second_event_info, first_event_info, source_name=second_name, reference_name=first_name)
    row = dict(forward)
    reverse_prefix = f"{second_name}_to_{first_name}_"
    for key, value in reverse.items():
        if key.startswith(reverse_prefix):
            row[key] = value

    forward_fraction = forward.get(f"{first_name}_to_{second_name}_coincident_event_fraction")
    reverse_fraction = reverse.get(f"{second_name}_to_{first_name}_coincident_event_fraction")
    pair_fraction = _finite_mean([forward_fraction, reverse_fraction])
    if not np.isfinite(pair_fraction):
        pair_strength = 0.0
    else:
        pair_strength = float(pair_fraction)
    pair_frequency = _finite_mean(
        [
            forward.get(f"{first_name}_to_{second_name}_coincident_event_frequency_per_min"),
            reverse.get(f"{second_name}_to_{first_name}_coincident_event_frequency_per_min"),
        ]
    )
    pair_count = int(forward.get("coincident_event_count", 0) or 0)

    row["pair_coincident_event_count"] = pair_count
    row["pair_coincident_event_fraction"] = pair_fraction
    row["pair_coincidence_strength"] = pair_strength
    row["pair_coincident_event_frequency_per_min"] = pair_frequency
    row["pair_coincident"] = bool(pair_count > 0)
    return row


def _summary_payload_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "day_id": str(row.get("day_id") or ""),
        "mode": str(row.get("mode") or ""),
        "state": canonical_state_label(row.get("state")),
        "state_display": str(row.get("state_display") or state_display_label(row.get("state"))),
        "state_color": str(row.get("state_color") or state_display_color(row.get("state"))),
        "comparison_name": str(row.get("comparison_name") or "soma_bouton_coincidence"),
        "pair_mode": str(row.get("pair_mode") or "cross_compartment"),
        "compartment": str(row.get("compartment") or "soma_vs_bouton"),
    }
    if "cohort" in row:
        payload["cohort"] = str(row.get("cohort") or "")
    return payload


def coincidence_state_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_cols: Sequence[str] = ("day_id", "mode", "state"),
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    group_keys = [str(col) for col in group_cols if str(col).strip()]
    grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    meta: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for row in rows:
        key_parts: List[str] = []
        for column in group_keys:
            value = str(row.get(column) or "").strip()
            if not value:
                break
            key_parts.append(value)
        if len(key_parts) != len(group_keys):
            continue
        key = tuple(key_parts)
        grouped.setdefault(key, []).append(row)
        payload = _summary_payload_from_row(row)
        for column in group_keys:
            payload[column] = str(row.get(column) or "")
        meta[key] = payload

    summary_rows: List[Dict[str, Any]] = []
    for key, members in grouped.items():
        payload = dict(meta[key])
        pair_counts = _finite_values([row.get("pair_coincident_event_count", row.get("coincident_event_count", float("nan"))) for row in members])
        pair_strengths = _finite_values([row.get("pair_coincidence_strength", row.get("pair_coincident_event_fraction", float("nan"))) for row in members])
        pair_fractions = _finite_values([row.get("pair_coincident_event_fraction", float("nan")) for row in members])
        pair_frequencies = _finite_values([row.get("pair_coincident_event_frequency_per_min", float("nan")) for row in members])
        soma_event_counts = _finite_values([row.get("soma_event_count", float("nan")) for row in members])
        bouton_event_counts = _finite_values([row.get("bouton_event_count", float("nan")) for row in members])
        soma_to_bouton_fractions = _finite_values([row.get("soma_to_bouton_coincident_event_fraction", float("nan")) for row in members])
        bouton_to_soma_fractions = _finite_values([row.get("bouton_to_soma_coincident_event_fraction", float("nan")) for row in members])
        soma_to_bouton_freq = _finite_values([row.get("soma_to_bouton_coincident_event_frequency_per_min", float("nan")) for row in members])
        bouton_to_soma_freq = _finite_values([row.get("bouton_to_soma_coincident_event_frequency_per_min", float("nan")) for row in members])
        coincident_pairs = int(sum(float(row.get("pair_coincident_event_count", 0) or 0) > 0 for row in members))

        payload.update(
            {
                "n_pairs": int(len(members)),
                "n_coincident_pairs": coincident_pairs,
                "n_noncoincident_pairs": int(len(members) - coincident_pairs),
                "pair_coincident_event_count_sum": float(np.nansum(pair_counts)) if pair_counts.size else float("nan"),
                "pair_coincident_event_count_mean": float(np.nanmean(pair_counts)) if pair_counts.size else float("nan"),
                "pair_coincident_event_count_median": float(np.nanmedian(pair_counts)) if pair_counts.size else float("nan"),
                "pair_coincident_event_fraction_mean": float(np.nanmean(pair_fractions)) if pair_fractions.size else float("nan"),
                "pair_coincidence_strength_mean": float(np.nanmean(pair_strengths)) if pair_strengths.size else float("nan"),
                "pair_coincidence_strength_median": float(np.nanmedian(pair_strengths)) if pair_strengths.size else float("nan"),
                "pair_coincident_event_frequency_per_min_mean": float(np.nanmean(pair_frequencies)) if pair_frequencies.size else float("nan"),
                "soma_event_count_mean": float(np.nanmean(soma_event_counts)) if soma_event_counts.size else float("nan"),
                "bouton_event_count_mean": float(np.nanmean(bouton_event_counts)) if bouton_event_counts.size else float("nan"),
                "soma_to_bouton_coincident_event_fraction_mean": float(np.nanmean(soma_to_bouton_fractions)) if soma_to_bouton_fractions.size else float("nan"),
                "bouton_to_soma_coincident_event_fraction_mean": float(np.nanmean(bouton_to_soma_fractions)) if bouton_to_soma_fractions.size else float("nan"),
                "soma_to_bouton_coincident_event_frequency_per_min_mean": float(np.nanmean(soma_to_bouton_freq)) if soma_to_bouton_freq.size else float("nan"),
                "bouton_to_soma_coincident_event_frequency_per_min_mean": float(np.nanmean(bouton_to_soma_freq)) if bouton_to_soma_freq.size else float("nan"),
            }
        )
        summary_rows.append(payload)

    return summary_rows


def build_coincidence_family_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_cols: Sequence[str] = ("day_id", "mode", "state"),
) -> Dict[str, Any]:
    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    summary_rows = coincidence_state_summary_rows(row_list, group_cols=group_cols)
    counts = {
        "pair_rows": int(len(row_list)),
        "summary_rows": int(len(summary_rows)),
        "day_rows": int(len(summary_rows)),
        "days": int(len({str(row.get("day_id") or "") for row in row_list if str(row.get("day_id") or "").strip()})),
        "states": int(len({canonical_state_label(row.get("state")) for row in row_list if canonical_state_label(row.get("state"))})),
        "coincident_pairs": int(sum(float(row.get("pair_coincident_event_count", 0) or 0) > 0 for row in row_list)),
    }
    return {
        "available": bool(row_list),
        "rows": row_list,
        "pair_rows": row_list,
        "summary_rows": summary_rows,
        "day_rows": summary_rows,
        "counts": counts,
    }


def run_family(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return build_coincidence_family_results(rows)


__all__ = [
    "annotate_event_coincidence",
    "annotate_spine_event_info",
    "build_bidirectional_coincidence_metrics",
    "build_coincidence_family_results",
    "coincident_event_runs",
    "coincidence_state_summary_rows",
    "event_frequency_per_minute",
    "event_run_onsets_match",
    "run_family",
]
