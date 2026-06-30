from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analysis.main_pipeline.sleep_dendrite_spine_pipeline import (
    build_state_masks_movie,
    build_state_masks_sleep,
    choose_locomotion_threshold,
    extract_series_bundle,
    interpolate_series,
)

from ...compartment_common import canonical_state_label, read_pickle, state_display_color, state_display_label
from .core import ExperimentContext, summarize_activity


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _shared_time_axis(ctx: ExperimentContext) -> np.ndarray:
    if ctx.soma.t.size:
        return np.asarray(ctx.soma.t, dtype=float)
    if ctx.bouton.t.size:
        return np.asarray(ctx.bouton.t, dtype=float)
    return np.array([], dtype=float)


def _movie_masks_for_context(ctx: ExperimentContext) -> Dict[str, np.ndarray]:
    exp_time = _shared_time_axis(ctx)
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
    exp_time = _shared_time_axis(ctx)
    if exp_time.size == 0:
        return {}
    sleep_state = ctx.state_bundle if isinstance(ctx.state_bundle, Mapping) else {}
    masks, _ = build_state_masks_sleep(exp_time, dict(sleep_state))
    return masks


def state_masks_for_context(ctx: ExperimentContext, selected_states: Sequence[str]) -> Dict[str, np.ndarray]:
    del selected_states
    if ctx.mode == "movie":
        return _movie_masks_for_context(ctx)
    return _sleep_masks_for_context(ctx)


def activity_rows_for_context(ctx: ExperimentContext, selected_states: Sequence[str]) -> List[Dict[str, Any]]:
    masks = state_masks_for_context(ctx, selected_states)
    soma_matrix = ctx.soma.matrix()
    bouton_matrix = ctx.bouton.matrix()
    rows: List[Dict[str, Any]] = []
    for state, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        soma_summary = summarize_activity(soma_matrix, mask)
        bouton_summary = summarize_activity(bouton_matrix, mask)
        for compartment, summary in (("soma", soma_summary), ("bouton", bouton_summary)):
            row = {
                "expid": ctx.expid,
                "mode": ctx.mode,
                "animal_id": ctx.animal_id,
                "date": ctx.date,
                "day_id": ctx.day_id,
                "state": canonical_state_label(state),
                "state_display": state_display_label(state),
                "state_color": state_display_color(state),
                "compartment": compartment,
                "n": summary["n"],
                "mean": summary["mean"],
                "median": summary["median"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
            }
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
        if finite.size == 0:
            payload.update(
                {
                    "n_experiments": int(arr.size),
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )
        else:
            payload.update(
                {
                    "n_experiments": int(arr.size),
                    "mean": float(np.nanmean(finite)),
                    "median": float(np.nanmedian(finite)),
                    "std": float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "min": float(np.nanmin(finite)),
                    "max": float(np.nanmax(finite)),
                }
            )
        summary_rows.append(payload)
    return summary_rows
