from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from analysis.compartment_common import pairwise_correlation
from analysis.shared.cache_utils import build_pairwise_correlation_cache_key
from analysis.shared.state_utils import canonical_state_label, state_display_color, state_display_label

from .core import ExperimentContext, make_unit_id


@dataclass(frozen=True)
class PairwiseMember:
    compartment: str
    channel: int
    roi_index: int
    roi_id: Any
    unit_id: str
    trace: np.ndarray


def pairwise_member_from_trace(
    *,
    ctx: ExperimentContext,
    compartment: str,
    channel: int,
    roi_index: int,
    roi_id: Any,
    trace: Sequence[float] | np.ndarray,
    unit_id: str | None = None,
) -> PairwiseMember:
    trace_array = np.asarray(trace, dtype=float).reshape(-1)
    resolved_unit_id = unit_id or make_unit_id(
        animal_id=ctx.animal_id,
        expid=ctx.expid,
        day_id=ctx.day_id,
        compartment=compartment,
        channel=channel,
        roi_id=roi_id,
        roi_index=roi_index,
    )
    return PairwiseMember(
        compartment=str(compartment),
        channel=int(channel),
        roi_index=int(roi_index),
        roi_id=roi_id,
        unit_id=str(resolved_unit_id),
        trace=trace_array,
    )


def pairwise_members_from_matrix(
    *,
    ctx: ExperimentContext,
    compartment: str,
    channel: int,
    matrix: Sequence[Sequence[float]] | np.ndarray,
    roi_ids: Sequence[Any],
) -> List[PairwiseMember]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return []
    resolved_roi_ids = list(roi_ids)
    if len(resolved_roi_ids) < arr.shape[0]:
        resolved_roi_ids.extend(range(len(resolved_roi_ids), arr.shape[0]))
    members: List[PairwiseMember] = []
    for roi_index in range(arr.shape[0]):
        roi_id = resolved_roi_ids[roi_index] if roi_index < len(resolved_roi_ids) else roi_index
        members.append(
            pairwise_member_from_trace(
                ctx=ctx,
                compartment=compartment,
                channel=channel,
                roi_index=roi_index,
                roi_id=roi_id,
                trace=arr[roi_index],
            )
        )
    return members


def pairwise_mean_member(
    *,
    ctx: ExperimentContext,
    compartment: str,
    channel: int,
    trace: Sequence[float] | np.ndarray,
    label: str = "mean",
) -> PairwiseMember:
    return pairwise_member_from_trace(
        ctx=ctx,
        compartment=compartment,
        channel=channel,
        roi_index=-1,
        roi_id=label,
        trace=trace,
        unit_id=f"{ctx.day_id}|{compartment}|{label}",
    )


def _pair_iter(left_members: Sequence[PairwiseMember], right_members: Sequence[PairwiseMember] | None) -> Iterable[tuple[PairwiseMember, PairwiseMember]]:
    if right_members is None or right_members is left_members:
        return combinations(left_members, 2)
    return product(left_members, right_members)


def _pair_unit_id(ctx: ExperimentContext, comparison_name: str, left: PairwiseMember, right: PairwiseMember) -> str:
    ordered_units = sorted([str(left.unit_id), str(right.unit_id)])
    return "|".join([str(ctx.day_id), str(comparison_name), *ordered_units])


def _paired_trace_values(
    left: PairwiseMember,
    right: PairwiseMember,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    left_trace = np.asarray(left.trace, dtype=float).reshape(-1)
    right_trace = np.asarray(right.trace, dtype=float).reshape(-1)
    usable = min(left_trace.size, right_trace.size, mask.size)
    if usable <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0
    state_mask = np.asarray(mask[:usable], dtype=bool)
    if not np.any(state_mask):
        return np.array([], dtype=float), np.array([], dtype=float), 0
    left_values = left_trace[:usable][state_mask]
    right_values = right_trace[:usable][state_mask]
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float), 0
    return left_values[valid], right_values[valid], int(valid.sum())


def _masked_trace_values(member: PairwiseMember, mask: np.ndarray, usable: int) -> np.ndarray:
    trace = np.asarray(member.trace, dtype=float).reshape(-1)
    if usable <= 0:
        return np.array([], dtype=float)
    state_mask = np.asarray(mask[:usable], dtype=bool)
    if not np.any(state_mask):
        return np.array([], dtype=float)
    return trace[:usable][state_mask]


def build_pairwise_correlation_rows(
    ctx: ExperimentContext,
    state_masks: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    comparison_name: str,
    left_members: Sequence[PairwiseMember],
    right_members: Sequence[PairwiseMember] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pair_members = list(left_members if right_members is None or right_members is left_members else list(left_members) + list(right_members))
    trace_lengths = {int(np.asarray(member.trace, dtype=float).reshape(-1).size) for member in pair_members}
    can_pre_mask = len(trace_lengths) == 1 and bool(pair_members)
    common_trace_length = next(iter(trace_lengths)) if can_pre_mask else 0
    for state, mask in state_masks.items():
        state_key = canonical_state_label(state)
        if not state_key:
            continue
        mask_array = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_array.size == 0:
            continue
        if can_pre_mask:
            usable = min(mask_array.size, common_trace_length)
            state_member_values = {
                member.unit_id: _masked_trace_values(member, mask_array, usable)
                for member in pair_members
            }
            for left, right in _pair_iter(left_members, right_members):
                left_values = state_member_values[left.unit_id]
                right_values = state_member_values[right.unit_id]
                usable_pair = min(left_values.size, right_values.size)
                if usable_pair <= 0:
                    continue
                left_values = left_values[:usable_pair]
                right_values = right_values[:usable_pair]
                valid = np.isfinite(left_values) & np.isfinite(right_values)
                if not np.any(valid):
                    continue
                corr = pairwise_correlation(left_values[valid], right_values[valid])
                n_timepoints = int(valid.sum())
                compartment_label = left.compartment if left.compartment == right.compartment else f"{left.compartment}_vs_{right.compartment}"
                rows.append(
                    {
                        "expid": ctx.expid,
                        "mode": ctx.mode,
                        "animal_id": ctx.animal_id,
                        "date": ctx.date,
                        "day_id": ctx.day_id,
                        "analysis_unit": "day",
                        "comparison_name": comparison_name,
                        "pair_mode": "within_compartment" if left.compartment == right.compartment else "cross_compartment",
                        "compartment": compartment_label,
                        "left_compartment": left.compartment,
                        "left_channel": int(left.channel),
                        "left_roi_index": int(left.roi_index),
                        "left_roi_id": left.roi_id,
                        "left_unit_id": left.unit_id,
                        "right_compartment": right.compartment,
                        "right_channel": int(right.channel),
                        "right_roi_index": int(right.roi_index),
                        "right_roi_id": right.roi_id,
                        "right_unit_id": right.unit_id,
                        "pair_unit_id": _pair_unit_id(ctx, comparison_name, left, right),
                        "state": state_key,
                        "state_display": state_display_label(state),
                        "state_color": state_display_color(state),
                        "corr": corr,
                        "n_timepoints": int(n_timepoints),
                    }
                )
            continue
        for left, right in _pair_iter(left_members, right_members):
            left_values, right_values, n_timepoints = _paired_trace_values(left, right, mask_array)
            if n_timepoints <= 0:
                continue
            corr = pairwise_correlation(left_values, right_values)
            compartment_label = left.compartment if left.compartment == right.compartment else f"{left.compartment}_vs_{right.compartment}"
            rows.append(
                {
                    "expid": ctx.expid,
                    "mode": ctx.mode,
                    "animal_id": ctx.animal_id,
                    "date": ctx.date,
                    "day_id": ctx.day_id,
                    "analysis_unit": "day",
                    "comparison_name": comparison_name,
                    "pair_mode": "within_compartment" if left.compartment == right.compartment else "cross_compartment",
                    "compartment": compartment_label,
                    "left_compartment": left.compartment,
                    "left_channel": int(left.channel),
                    "left_roi_index": int(left.roi_index),
                    "left_roi_id": left.roi_id,
                    "left_unit_id": left.unit_id,
                    "right_compartment": right.compartment,
                    "right_channel": int(right.channel),
                    "right_roi_index": int(right.roi_index),
                    "right_roi_id": right.roi_id,
                    "right_unit_id": right.unit_id,
                    "pair_unit_id": _pair_unit_id(ctx, comparison_name, left, right),
                    "state": state_key,
                    "state_display": state_display_label(state),
                    "state_color": state_display_color(state),
                    "corr": corr,
                    "n_timepoints": int(n_timepoints),
                }
            )
    return rows


def pairwise_correlation_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[tuple, List[float]] = {}
    meta: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        comparison_name = str(row.get("comparison_name") or "pairwise_correlation")
        key = (
            str(row.get("day_id") or ""),
            str(row.get("mode") or ""),
            comparison_name,
            str(row.get("pair_mode") or "within_compartment"),
            str(row.get("state") or ""),
        )
        grouped.setdefault(key, []).append(float(row.get("corr", float("nan"))))
        meta[key] = {
            "day_id": row.get("day_id"),
            "mode": row.get("mode"),
            "comparison_name": comparison_name,
            "pair_mode": str(row.get("pair_mode") or "within_compartment"),
            "compartment": row.get("compartment"),
            "state": row.get("state"),
            "state_display": row.get("state_display"),
            "state_color": row.get("state_color"),
        }
    summary_rows: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        payload = dict(meta[key])
        payload.update(
            {
                "n_pairs": int(arr.size),
                "mean_corr": float(np.nanmean(arr)),
                "median_corr": float(np.nanmedian(arr)),
                "std_corr": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
                "min_corr": float(np.nanmin(arr)),
                "max_corr": float(np.nanmax(arr)),
            }
        )
        summary_rows.append(payload)
    return summary_rows


def build_pairwise_correlation_family_results(
    ctx: ExperimentContext,
    state_masks: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    comparison_name: str,
    left_members: Sequence[PairwiseMember],
    right_members: Sequence[PairwiseMember] | None = None,
    cache_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    rows = build_pairwise_correlation_rows(
        ctx,
        state_masks,
        comparison_name=comparison_name,
        left_members=left_members,
        right_members=right_members,
    )
    summary_rows = pairwise_correlation_summary_rows(rows)
    results: Dict[str, Any] = {
        "available": bool(rows),
        "comparison_name": comparison_name,
        "rows": rows,
        "summary_rows": summary_rows,
        "counts": {
            "rows": len(rows),
            "summary_rows": len(summary_rows),
        },
    }
    if cache_metadata is not None:
        cache_meta = dict(cache_metadata)
        results["cache_metadata"] = cache_meta
        results["cache_key"] = build_pairwise_correlation_cache_key(**cache_meta)
    return results


__all__ = [
    "PairwiseMember",
    "build_pairwise_correlation_family_results",
    "build_pairwise_correlation_rows",
    "pairwise_correlation_summary_rows",
    "pairwise_mean_member",
    "pairwise_member_from_trace",
    "pairwise_members_from_matrix",
]
