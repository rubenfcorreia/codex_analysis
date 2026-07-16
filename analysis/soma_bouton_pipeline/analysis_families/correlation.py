from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analysis.shared.analysis_families.core import ExperimentContext
from analysis.shared.analysis_families.pairwise import (
    build_pairwise_correlation_rows,
    pairwise_correlation_summary_rows,
    pairwise_mean_member,
    pairwise_members_from_matrix,
)

from .state import state_masks_for_context


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


def bouton_soma_correlation_rows(ctx: ExperimentContext, selected_states: Sequence[str]) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
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
    )


def soma_pairwise_correlation_rows(ctx: ExperimentContext, selected_states: Sequence[str]) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
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
    )


def bouton_pairwise_correlation_rows(ctx: ExperimentContext, selected_states: Sequence[str]) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
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
    )


def correlation_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return pairwise_correlation_summary_rows(rows)


__all__ = [
    "bouton_pairwise_correlation_rows",
    "bouton_soma_correlation_rows",
    "correlation_summary_rows",
    "soma_pairwise_correlation_rows",
]
