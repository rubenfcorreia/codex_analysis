from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from analysis.compartment_common import pairwise_correlation
from analysis.shared.cache_utils import build_pairwise_correlation_cache_key, build_shared_shift_schedule_key
from analysis.shared.analysis_families.correlation import correlation_analysis_for_batch
from analysis.shared.state_utils import canonical_state_label, state_display_color, state_display_label

from .core import ExperimentContext, make_global_bouton_id, make_global_soma_id, make_unit_id


@dataclass(frozen=True)
class PairwiseMember:
    compartment: str
    channel: int
    roi_index: int
    roi_id: Any
    unit_id: str
    trace: np.ndarray
    shuffle_keys: tuple[str, ...] | None = None


def pairwise_member_from_trace(
    *,
    ctx: ExperimentContext,
    compartment: str,
    channel: int,
    roi_index: int,
    roi_id: Any,
    trace: Sequence[float] | np.ndarray,
    unit_id: str | None = None,
    shuffle_keys: Sequence[str] | None = None,
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
        shuffle_keys=tuple(str(value) for value in (shuffle_keys or (resolved_unit_id,))),
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
        entity = ctx.soma if compartment == "soma" else ctx.bouton
        planes = list(entity.roi_plane_ids()) if hasattr(entity, "roi_plane_ids") else []
        plane = planes[roi_index] if roi_index < len(planes) else None
        members.append(
            pairwise_member_from_trace(
                ctx=ctx,
                compartment=compartment,
                channel=channel,
                roi_index=roi_index,
                roi_id=roi_id,
                trace=arr[roi_index],
                shuffle_keys=((make_global_soma_id if compartment == "soma" else make_global_bouton_id)(animal_id=ctx.animal_id, expid=ctx.expid, day_id=ctx.day_id, channel=channel, roi_id=roi_id, roi_index=None if plane is not None else roi_index, plane=plane),),
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
    shuffle_keys: Sequence[str] | None = None,
) -> PairwiseMember:
    return pairwise_member_from_trace(
        ctx=ctx,
        compartment=compartment,
        channel=channel,
        roi_index=-1,
        roi_id=label,
        trace=trace,
        unit_id=f"{ctx.day_id}|{compartment}|{label}",
        shuffle_keys=shuffle_keys,
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
    # Preserve the state-aligned vector and its NaNs.  Circular-shift
    # inference must shift the original aligned samples; removing each
    # pair's finite samples here would make vector lengths pair-specific.
    return left_values, right_values, int(left_values.size)


def _masked_trace_values(member: PairwiseMember, mask: np.ndarray, usable: int) -> np.ndarray:
    trace = np.asarray(member.trace, dtype=float).reshape(-1)
    if usable <= 0:
        return np.array([], dtype=float)
    state_mask = np.asarray(mask[:usable], dtype=bool)
    if not np.any(state_mask):
        return np.array([], dtype=float)
    return trace[:usable][state_mask]


def _ensure_correlation_shuffle_entry(
    shared_shuffle_cache: Dict[str, Any] | None,
    key: str,
    vector_length: int,
    shuffle_n: int,
    shuffle_seed: int,
    min_shift_frames: int,
) -> None:
    if shared_shuffle_cache is None or shuffle_n <= 0 or vector_length <= 1:
        return
    entries = shared_shuffle_cache.setdefault("entries", {})
    entry = entries.get(key)
    if isinstance(entry, Mapping) and np.asarray(entry.get("shifts", []), dtype=int).size >= shuffle_n:
        return
    minimum = max(1, min(int(min_shift_frames), vector_length - 1))
    shifts = np.random.default_rng(int(shuffle_seed)).integers(minimum, vector_length, size=int(shuffle_n))
    entries[key] = {
        "key": key,
        "vector_length": int(vector_length),
        "shuffle_n": int(shuffle_n),
        "shuffle_seed": int(shuffle_seed),
        "min_shift_frames": int(min_shift_frames),
        "shifts": shifts,
    }


def build_pairwise_correlation_rows(
    ctx: ExperimentContext,
    state_masks: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    comparison_name: str,
    left_members: Sequence[PairwiseMember],
    right_members: Sequence[PairwiseMember] | None = None,
    correlation_shuffle_n: int = 0,
    correlation_method: str = "pearson",
    correlation_inference: str = "none",
    correlation_shuffle_seed: int = 12345,
    correlation_min_shift_frames: int = 1,
    correlation_batch_size: int = 64,
    correlation_shift_block_size: int = 8,
    shared_shuffle_cache: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diagnostics = None
    if isinstance(shared_shuffle_cache, dict):
        diagnostics = shared_shuffle_cache.setdefault(
            "pairwise_diagnostics",
            {"null_model": str((shared_shuffle_cache or {}).get("null_model", "mask_aware")), "states": 0, "pairs_prepared": 0, "rows_generated": 0, "observed_and_pair_prep_elapsed_s": 0.0, "null_elapsed_s": 0.0, "row_build_elapsed_s": 0.0, "by_comparison": {}},
        )
    comparison_diagnostics = diagnostics.setdefault("by_comparison", {}).setdefault(
        str(comparison_name),
        {"states": 0, "pairs_prepared": 0, "rows_generated": 0, "observed_and_pair_prep_elapsed_s": 0.0, "null_elapsed_s": 0.0, "row_build_elapsed_s": 0.0},
    ) if diagnostics is not None else None
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
        state_member_values = None
        if can_pre_mask:
            usable = min(mask_array.size, common_trace_length)
            state_member_values = {
                member.unit_id: _masked_trace_values(member, mask_array, usable)
                for member in pair_members
            }
        state_started = time.perf_counter()
        pending: List[Dict[str, Any]] = []
        for left, right in _pair_iter(left_members, right_members):
            if state_member_values is not None:
                left_values = state_member_values[left.unit_id]
                right_values = state_member_values[right.unit_id]
                usable_pair = min(left_values.size, right_values.size)
                if usable_pair <= 0:
                    continue
                left_values = left_values[:usable_pair]
                right_values = right_values[:usable_pair]
                if not np.any(np.isfinite(left_values) & np.isfinite(right_values)):
                    continue
            else:
                left_values, right_values, n_timepoints = _paired_trace_values(left, right, mask_array)
                if n_timepoints <= 0:
                    continue
            corr = pairwise_correlation(left_values, right_values)
            n_timepoints = int(left_values.size)
            pair_key = _pair_unit_id(ctx, comparison_name, left, right)
            shuffle_key = build_shared_shift_schedule_key(
                family="soma_pairwise_correlation",
                signal=str(comparison_name),
                analysis_unit="day",
                animal_id=str(ctx.animal_id),
                day_id=str(ctx.day_id),
                vector_length=n_timepoints,
                correlation_method=correlation_method,
                correlation_inference=correlation_inference,
                shuffle_n=correlation_shuffle_n,
                shuffle_seed=correlation_shuffle_seed,
                min_shift_frames=correlation_min_shift_frames,
                null_model=str((shared_shuffle_cache or {}).get("null_model", "mask_aware")),
            )
            _ensure_correlation_shuffle_entry(
                shared_shuffle_cache,
                shuffle_key,
                n_timepoints,
                correlation_shuffle_n,
                correlation_shuffle_seed,
                correlation_min_shift_frames,
            )
            pending.append({
                "left": left,
                "right": right,
                "left_values": left_values,
                "right_values": right_values,
                "corr": corr,
                "n_timepoints": n_timepoints,
                "pair_key": pair_key,
                "shuffle_key": shuffle_key,
                "signal_a_key": tuple(left.shuffle_keys or (left.unit_id,)),
                "signal_b_key": tuple(right.shuffle_keys or (right.unit_id,)),
                "signal_masks": (np.asarray(mask_array, dtype=bool), np.asarray(mask_array, dtype=bool)),
            })
        if diagnostics is not None:
            elapsed = time.perf_counter() - state_started
            diagnostics["states"] = int(diagnostics.get("states", 0)) + 1
            diagnostics["pairs_prepared"] = int(diagnostics.get("pairs_prepared", 0)) + len(pending)
            diagnostics["observed_and_pair_prep_elapsed_s"] = float(diagnostics.get("observed_and_pair_prep_elapsed_s", 0.0)) + elapsed
            if comparison_diagnostics is not None:
                comparison_diagnostics["states"] += 1
                comparison_diagnostics["pairs_prepared"] += len(pending)
                comparison_diagnostics["observed_and_pair_prep_elapsed_s"] += elapsed
        if not pending:
            continue
        null_started = time.perf_counter()
        inferential = correlation_analysis_for_batch(
            [(item["left_values"], item["right_values"]) for item in pending],
            correlation_shuffle_n,
            shared_shuffle_cache=shared_shuffle_cache,
            shared_shuffle_keys=[item["shuffle_key"] for item in pending],
            signal_keys=[(item["signal_a_key"], item["signal_b_key"]) for item in pending],
            signal_masks=[item["signal_masks"] for item in pending],
            result_keys=[f"{item['pair_key']}|{state_key}" for item in pending],
            correlation_method=correlation_method,
            correlation_inference=correlation_inference,
            shuffle_seed=correlation_shuffle_seed,
            min_shift_frames=correlation_min_shift_frames,
            max_batch_size=correlation_batch_size,
            shift_block_size=correlation_shift_block_size,
            null_model=str((shared_shuffle_cache or {}).get("null_model", "mask_aware")),
        )
        if diagnostics is not None:
            elapsed = time.perf_counter() - null_started
            diagnostics["null_elapsed_s"] = float(diagnostics.get("null_elapsed_s", 0.0)) + elapsed
            if comparison_diagnostics is not None:
                comparison_diagnostics["null_elapsed_s"] += elapsed
        row_started = time.perf_counter()
        for item, inferential_row in zip(pending, inferential):
            left = item["left"]
            right = item["right"]
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
                    "pair_unit_id": item["pair_key"],
                    "state": state_key,
                    "state_display": state_display_label(state),
                    "state_color": state_display_color(state),
                    "corr": item["corr"],
                    "effect_size": item["corr"],
                    "lower_ci": None,
                    "upper_ci": None,
                    "p_value": inferential_row.get("p_value") if correlation_inference != "none" else None,
                    "shuffle_p": inferential_row.get("shuffle_p") if correlation_inference != "none" else None,
                    "shuffle_n_requested": inferential_row.get("shuffle_n_requested", 0),
                    "shuffle_n_success": inferential_row.get("shuffle_n_success", 0),
                    "p_value_source": inferential_row.get("p_value_source", "none_descriptive"),
                    "inferential_status": inferential_row.get("status", "descriptive"),
                    "inferential_unit": "day_pair",
                    "correction_family": None,
                    "n_timepoints": item["n_timepoints"],
                }
            )
        if diagnostics is not None:
            elapsed = time.perf_counter() - row_started
            diagnostics["rows_generated"] = int(diagnostics.get("rows_generated", 0)) + len(inferential)
            diagnostics["row_build_elapsed_s"] = float(diagnostics.get("row_build_elapsed_s", 0.0)) + elapsed
            if comparison_diagnostics is not None:
                comparison_diagnostics["rows_generated"] += len(inferential)
                comparison_diagnostics["row_build_elapsed_s"] += elapsed
    return rows


def pairwise_correlation_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[tuple, List[float]] = {}
    meta: Dict[tuple, Dict[str, Any]] = {}
    has_split_group = any(str(row.get("split_group") or "").strip() for row in rows)
    for row in rows:
        comparison_name = str(row.get("comparison_name") or "pairwise_correlation")
        split_group = str(row.get("split_group") or "").strip() if has_split_group else ""
        key = (
            str(row.get("day_id") or ""),
            str(row.get("mode") or ""),
            comparison_name,
            str(row.get("pair_mode") or "within_compartment"),
            str(row.get("state") or ""),
            split_group,
        )
        grouped.setdefault(key, []).append(float(row.get("corr", float("nan"))))
        payload: Dict[str, Any] = {
            "day_id": row.get("day_id"),
            "mode": row.get("mode"),
            "comparison_name": comparison_name,
            "pair_mode": str(row.get("pair_mode") or "within_compartment"),
            "compartment": row.get("compartment"),
            "state": row.get("state"),
            "state_display": row.get("state_display"),
            "state_color": row.get("state_color"),
        }
        if has_split_group:
            payload.update(
                {
                    "split_group": split_group or None,
                    "split_group_display": str(row.get("split_group_display") or split_group).strip() or None,
                    "split_group_color": str(row.get("split_group_color") or "").strip() or None,
                    "split_group_rank": row.get("split_group_rank"),
                }
            )
        meta[key] = payload
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
                "effect_size": float(np.nanmean(arr)),
                "lower_ci": None,
                "upper_ci": None,
                "p_value_source": "none_descriptive",
                "inferential_status": "descriptive",
                "inferential_unit": "day_pair",
                "correction_family": None,
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
