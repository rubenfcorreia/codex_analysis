from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analysis.shared.analysis_families.core import ExperimentContext, make_global_soma_id
from analysis.shared.analysis_families.pairwise import (
    build_pairwise_correlation_rows,
    pairwise_correlation_summary_rows,
    pairwise_mean_member,
    pairwise_members_from_matrix,
)

from .soma_state import state_masks_for_context


def _bouton_roi_ids(ctx: ExperimentContext, matrix: np.ndarray) -> List[Any]:
    roi_ids = list(ctx.bouton.roi_ids()) if hasattr(ctx.bouton, "roi_ids") else []
    if len(roi_ids) < matrix.shape[0]:
        roi_ids.extend(range(len(roi_ids), matrix.shape[0]))
    return roi_ids[: matrix.shape[0]]


def _soma_roi_ids(ctx: ExperimentContext, matrix: np.ndarray) -> List[Any]:
    roi_ids = list(ctx.soma.roi_ids()) if hasattr(ctx.soma, "roi_ids") else []
    if len(roi_ids) < matrix.shape[0]:
        roi_ids.extend(range(len(roi_ids), matrix.shape[0]))
    return roi_ids[: matrix.shape[0]]


def bouton_soma_correlation_rows(
    ctx: ExperimentContext,
    selected_states: Sequence[str],
    state_masks: Mapping[str, np.ndarray] | None = None,
    correlation_shuffle_n: int = 0,
    correlation_method: str = "pearson",
    correlation_inference: str = "none",
    correlation_shuffle_seed: int = 12345,
    correlation_min_shift_frames: int = 1,
    correlation_batch_size: int = 64,
    correlation_shift_block_size: int = 8,
    shared_shuffle_cache: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    masks = state_masks if state_masks is not None else state_masks_for_context(ctx, selected_states)
    soma_matrix = np.asarray(ctx.soma.matrix(), dtype=float)
    bouton_matrix = np.asarray(ctx.bouton.matrix(), dtype=float)
    if soma_matrix.size == 0 or bouton_matrix.size == 0:
        return []
    soma_mean = np.nanmean(soma_matrix, axis=0)
    soma_member = pairwise_mean_member(
        ctx=ctx,
        compartment="soma",
        channel=ctx.soma_channel,
        trace=soma_mean,
        label="mean",
        shuffle_keys=tuple(make_global_soma_id(animal_id=ctx.animal_id, expid=ctx.expid, day_id=ctx.day_id, channel=ctx.soma_channel, roi_id=roi_id, roi_index=None if (hasattr(ctx.soma, 'roi_plane_ids') and index < len(ctx.soma.roi_plane_ids()) and ctx.soma.roi_plane_ids()[index] is not None) else index, plane=(ctx.soma.roi_plane_ids()[index] if hasattr(ctx.soma, 'roi_plane_ids') and index < len(ctx.soma.roi_plane_ids()) else None)) for index, roi_id in enumerate(_soma_roi_ids(ctx, soma_matrix))),
    )
    bouton_members = pairwise_members_from_matrix(
        ctx=ctx,
        compartment="bouton",
        channel=ctx.bouton_channel,
        matrix=bouton_matrix,
        roi_ids=_bouton_roi_ids(ctx, bouton_matrix),
    )
    return build_pairwise_correlation_rows(
        ctx,
        masks,
        comparison_name="bouton_soma",
        left_members=[soma_member],
        right_members=bouton_members,
        correlation_shuffle_n=correlation_shuffle_n,
        correlation_method=correlation_method,
        correlation_inference=correlation_inference,
        correlation_shuffle_seed=correlation_shuffle_seed,
        correlation_min_shift_frames=correlation_min_shift_frames,
        correlation_batch_size=correlation_batch_size,
        correlation_shift_block_size=correlation_shift_block_size,
        shared_shuffle_cache=shared_shuffle_cache,
    )


def soma_pairwise_correlation_rows(
    ctx: ExperimentContext,
    selected_states: Sequence[str],
    state_masks: Mapping[str, np.ndarray] | None = None,
    correlation_shuffle_n: int = 0,
    correlation_method: str = "pearson",
    correlation_inference: str = "none",
    correlation_shuffle_seed: int = 12345,
    correlation_min_shift_frames: int = 1,
    correlation_batch_size: int = 64,
    correlation_shift_block_size: int = 8,
    shared_shuffle_cache: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    masks = state_masks if state_masks is not None else state_masks_for_context(ctx, selected_states)
    soma_matrix = np.asarray(ctx.soma.matrix(), dtype=float)
    if soma_matrix.size == 0:
        return []
    soma_members = pairwise_members_from_matrix(
        ctx=ctx,
        compartment="soma",
        channel=ctx.soma_channel,
        matrix=soma_matrix,
        roi_ids=_soma_roi_ids(ctx, soma_matrix),
    )
    return build_pairwise_correlation_rows(
        ctx,
        masks,
        comparison_name="soma_pairwise",
        left_members=soma_members,
        correlation_shuffle_n=correlation_shuffle_n,
        correlation_method=correlation_method,
        correlation_inference=correlation_inference,
        correlation_shuffle_seed=correlation_shuffle_seed,
        correlation_min_shift_frames=correlation_min_shift_frames,
        correlation_batch_size=correlation_batch_size,
        correlation_shift_block_size=correlation_shift_block_size,
        shared_shuffle_cache=shared_shuffle_cache,
    )


def bouton_pairwise_correlation_rows(
    ctx: ExperimentContext,
    selected_states: Sequence[str],
    state_masks: Mapping[str, np.ndarray] | None = None,
    correlation_shuffle_n: int = 0,
    correlation_method: str = "pearson",
    correlation_inference: str = "none",
    correlation_shuffle_seed: int = 12345,
    correlation_min_shift_frames: int = 1,
    correlation_batch_size: int = 64,
    correlation_shift_block_size: int = 8,
    shared_shuffle_cache: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    masks = state_masks if state_masks is not None else state_masks_for_context(ctx, selected_states)
    bouton_matrix = np.asarray(ctx.bouton.matrix(), dtype=float)
    if bouton_matrix.size == 0:
        return []
    bouton_members = pairwise_members_from_matrix(
        ctx=ctx,
        compartment="bouton",
        channel=ctx.bouton_channel,
        matrix=bouton_matrix,
        roi_ids=_bouton_roi_ids(ctx, bouton_matrix),
    )
    return build_pairwise_correlation_rows(
        ctx,
        masks,
        comparison_name="bouton_pairwise",
        left_members=bouton_members,
        correlation_shuffle_n=correlation_shuffle_n,
        correlation_method=correlation_method,
        correlation_inference=correlation_inference,
        correlation_shuffle_seed=correlation_shuffle_seed,
        correlation_min_shift_frames=correlation_min_shift_frames,
        correlation_batch_size=correlation_batch_size,
        correlation_shift_block_size=correlation_shift_block_size,
        shared_shuffle_cache=shared_shuffle_cache,
    )


def correlation_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return pairwise_correlation_summary_rows(rows)


__all__ = [
    "bouton_pairwise_correlation_rows",
    "bouton_soma_correlation_rows",
    "correlation_summary_rows",
    "soma_pairwise_correlation_rows",
]
