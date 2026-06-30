from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ...compartment_common import canonical_state_label, lagged_correlation, state_display_color, state_display_label, summarize_lag_scan
from .core import ExperimentContext
from .state import state_masks_for_context


def lag_scan_rows(ctx: ExperimentContext, selected_states: Sequence[str], lag_window_s: float = 2.0, lag_step_s: float = 0.1) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
    soma_matrix = ctx.soma.matrix()
    bouton_matrix = ctx.bouton.matrix()
    soma_mean = np.nanmean(soma_matrix, axis=0) if soma_matrix.size else np.array([], dtype=float)
    if soma_mean.size == 0 or bouton_matrix.size == 0:
        return []
    t = ctx.soma.t if ctx.soma.t.size else ctx.bouton.t
    if t.size == 0:
        return []
    t_len = min(t.size, soma_mean.size, bouton_matrix.shape[1])
    t = t[:t_len]
    soma_mean = soma_mean[:t_len]
    bouton_matrix = bouton_matrix[:, :t_len]
    lags_s = np.arange(-lag_window_s, lag_window_s + 0.5 * lag_step_s, lag_step_s)
    rows: List[Dict[str, Any]] = []
    for state, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)[:t_len]
        if not np.any(mask):
            continue
        state_t = t[mask]
        state_soma = soma_mean[mask]
        for roi_index in range(bouton_matrix.shape[0]):
            bouton_state = bouton_matrix[roi_index, :][mask]
            lag_values_t, corrs = lagged_correlation(state_t, state_soma, state_t, bouton_state, lags_s)
            summary = summarize_lag_scan(lag_values_t, corrs)
            for lag, corr in zip(lag_values_t, corrs):
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
                        "lag_s": float(lag),
                        "corr": float(corr),
                        "zero_lag_corr": summary["zero_lag_corr"],
                        "best_corr": summary["best_corr"],
                        "best_lag_s": summary["best_lag_s"],
                        "mean_corr": summary["mean_corr"],
                        "area_abs_corr": summary["area_abs_corr"],
                    }
                )
    return rows


def lag_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["day_id"], row["mode"], row["state"])
        grouped.setdefault(key, []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for key, members in grouped.items():
        corr_values = np.asarray([float(row["corr"]) for row in members], dtype=float)
        zero_values = np.asarray([float(row["zero_lag_corr"]) for row in members], dtype=float)
        best_values = np.asarray([float(row["best_corr"]) for row in members], dtype=float)
        lag_values = np.asarray([float(row["best_lag_s"]) for row in members], dtype=float)
        first = members[0]
        summary_rows.append(
            {
                "day_id": first["day_id"],
                "mode": first["mode"],
                "state": first["state"],
                "state_display": first["state_display"],
                "state_color": first["state_color"],
                "n_boutons": int(len(members)),
                "mean_corr": float(np.nanmean(corr_values)),
                "zero_lag_corr": float(np.nanmean(zero_values)),
                "best_corr": float(np.nanmean(best_values)),
                "best_lag_s": float(np.nanmean(lag_values)),
                "corr_std": float(np.nanstd(corr_values, ddof=1)) if corr_values.size > 1 else 0.0,
            }
        )
    return summary_rows

