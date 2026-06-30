from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ...compartment_common import canonical_state_label, pairwise_correlation, state_display_color, state_display_label
from .core import ExperimentContext
from .state import state_masks_for_context


def bouton_soma_correlation_rows(ctx: ExperimentContext, selected_states: Sequence[str]) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
    soma_matrix = ctx.soma.matrix()
    bouton_matrix = ctx.bouton.matrix()
    soma_mean = np.nanmean(soma_matrix, axis=0) if soma_matrix.size else np.array([], dtype=float)
    rows: List[Dict[str, Any]] = []
    for state, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if soma_mean.size == 0 or bouton_matrix.size == 0:
            continue
        t_len = min(soma_mean.size, bouton_matrix.shape[1], mask.size)
        if t_len == 0:
            continue
        state_mask = mask[:t_len]
        if not np.any(state_mask):
            continue
        soma_state = soma_mean[:t_len][state_mask]
        for roi_index in range(bouton_matrix.shape[0]):
            bouton_state = bouton_matrix[roi_index, :t_len][state_mask]
            corr = pairwise_correlation(soma_state, bouton_state)
            rows.append(
                {
                    "expid": ctx.expid,
                    "mode": ctx.mode,
                    "animal_id": ctx.animal_id,
                    "date": ctx.date,
                    "day_id": ctx.day_id,
                    "state": canonical_state_label(state),
                    "state_display": state_display_label(state),
                    "state_color": state_display_color(state),
                    "bouton_roi_index": roi_index,
                    "corr": corr,
                    "n_timepoints": int(np.isfinite(soma_state).sum()),
                }
            )
    return rows


def correlation_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[tuple, List[float]] = {}
    meta: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row["day_id"], row["mode"], row["state"])
        grouped.setdefault(key, []).append(float(row["corr"]))
        meta[key] = {
            "day_id": row["day_id"],
            "mode": row["mode"],
            "state": row["state"],
            "state_display": row["state_display"],
            "state_color": row["state_color"],
        }
    summary_rows: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        payload = meta[key].copy()
        payload.update(
            {
                "n_boutons": int(arr.size),
                "mean_corr": float(np.nanmean(arr)),
                "median_corr": float(np.nanmedian(arr)),
                "std_corr": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
                "min_corr": float(np.nanmin(arr)),
                "max_corr": float(np.nanmax(arr)),
            }
        )
        summary_rows.append(payload)
    return summary_rows

