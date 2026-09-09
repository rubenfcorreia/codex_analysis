"""Utilities for within-experiment state-transition analyses."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TRANSITION_SCHEMA_VERSION = 1
SLEEP_STATE_LABELS = frozenset({"active_awake", "quiet_awake", "wake", "nrem", "rem"})


def normalize_transition_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    enabled = bool(raw.get("enabled", False))
    try:
        window_s = float(raw.get("window_s", 60.0))
    except (TypeError, ValueError):
        window_s = 60.0
    if not np.isfinite(window_s) or window_s <= 0:
        window_s = 60.0
    modes = raw.get("window_modes", ["strict", "max_available"])
    if isinstance(modes, str):
        modes = [modes]
    modes = [str(value).strip().lower() for value in (modes or [])]
    modes = [mode for mode in ("strict", "max_available") if mode in modes]
    scopes = raw.get("scopes", ["all_states", "sleep_states"])
    if isinstance(scopes, str):
        scopes = [scopes]
    scopes = [str(value).strip().lower() for value in (scopes or [])]
    scopes = [scope for scope in ("all_states", "sleep_states") if scope in scopes]
    metrics = raw.get("metrics", ["activity", "event_frequency"])
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [str(value).strip().lower() for value in (metrics or [])]
    metrics = [metric for metric in ("activity", "event_frequency") if metric in metrics]
    return {
        "enabled": enabled,
        "window_s": float(window_s),
        "window_modes": list(dict.fromkeys(modes)),
        "scopes": list(dict.fromkeys(scopes)),
        "metrics": list(dict.fromkeys(metrics)),
        "schema_version": TRANSITION_SCHEMA_VERSION,
    }


def _as_bool_mask(mask: Any, size: int) -> np.ndarray:
    array = np.asarray(mask, dtype=bool).reshape(-1)
    if array.size < size:
        padded = np.zeros(size, dtype=bool)
        padded[: array.size] = array
        return padded
    return array[:size]


def state_sequence(
    time: Sequence[float] | np.ndarray,
    state_masks: Mapping[str, Any],
    selected_states: Sequence[str],
) -> tuple[np.ndarray, list[str | None]]:
    time_array = np.asarray(time, dtype=float).reshape(-1)
    ordered_states = [str(state) for state in selected_states if str(state).strip()]
    masks = {state: _as_bool_mask(state_masks.get(state), time_array.size) for state in ordered_states}
    labels: list[str | None] = [None] * time_array.size
    for index in range(time_array.size):
        for state in ordered_states:
            if masks[state][index]:
                labels[index] = state
                break
    return time_array, labels


def transition_events(
    time: Sequence[float] | np.ndarray,
    state_masks: Mapping[str, Any],
    selected_states: Sequence[str],
    *,
    scope: str,
    window_s: float = 60.0,
) -> list[dict[str, Any]]:
    time_array, labels = state_sequence(time, state_masks, selected_states)
    if time_array.size < 2:
        return []
    finite = np.isfinite(time_array)
    events: list[dict[str, Any]] = []
    finite_diffs = np.diff(time_array)[np.isfinite(np.diff(time_array)) & (np.diff(time_array) > 0)]
    nominal_dt = float(np.median(finite_diffs)) if finite_diffs.size else float("nan")
    max_step = nominal_dt * 1.5 if np.isfinite(nominal_dt) else float("inf")
    run_start = 0
    while run_start < time_array.size:
        label = labels[run_start]
        run_end = run_start + 1
        while run_end < time_array.size and labels[run_end] == label and (time_array[run_end] - time_array[run_end - 1]) <= max_step:
            run_end += 1
        if label is not None and run_end < time_array.size:
            next_label = labels[run_end]
            next_end = run_end + 1
            while next_end < time_array.size and labels[next_end] == next_label and (time_array[next_end] - time_array[next_end - 1]) <= max_step:
                next_end += 1
            if next_label is not None and next_label != label:
                boundary = float(time_array[run_end])
                source_start = float(time_array[run_start])
                destination_end = float(time_array[next_end - 1])
                pre_available = boundary - source_start
                post_available = destination_end - boundary
                if (
                    finite[run_start]
                    and finite[run_end]
                    and finite[next_end - 1]
                    and np.isfinite(pre_available)
                    and np.isfinite(post_available)
                    and pre_available > 0
                    and post_available > 0
                ):
                    events.append(
                        {
                            "transition_index": int(run_end),
                            "transition_time": boundary,
                            "state_before": str(label),
                            "state_after": str(next_label),
                            "source_start_time": source_start,
                            "destination_end_time": destination_end,
                            "pre_available_s": float(pre_available),
                            "post_available_s": float(post_available),
                            "scope": str(scope),
                            "window_s": float(window_s),
                        }
                    )
            run_start = next_end if next_label is not None else run_end
        else:
            run_start = run_end
    return events


def event_window(event: Mapping[str, Any], side: str, mode: str) -> tuple[float, float] | None:
    transition = float(event["transition_time"])
    window_s = float(event["window_s"])
    if side == "pre":
        available = float(event["pre_available_s"])
        start = max(float(event["source_start_time"]), transition - window_s)
        end = transition
    elif side == "post":
        available = float(event["post_available_s"])
        start = transition
        end = min(float(event["destination_end_time"]), transition + window_s)
    else:
        raise ValueError(f"Unknown transition side: {side}")
    if mode == "strict" and available + 1e-9 < window_s:
        return None
    if end <= start:
        return None
    return float(start), float(end)


def add_window_metadata(event: Mapping[str, Any], mode: str) -> dict[str, Any] | None:
    pre = event_window(event, "pre", mode)
    post = event_window(event, "post", mode)
    if pre is None or post is None:
        return None
    result = dict(event)
    result.update(
        {
            "window_mode": mode,
            "pre_start_time": pre[0],
            "pre_end_time": pre[1],
            "post_start_time": post[0],
            "post_end_time": post[1],
            "pre_duration_s": float(pre[1] - pre[0]),
            "post_duration_s": float(post[1] - post[0]),
        }
    )
    return result


def interval_mask(time: Sequence[float] | np.ndarray, start: float, end: float) -> np.ndarray:
    values = np.asarray(time, dtype=float).reshape(-1)
    return np.isfinite(values) & (values >= float(start)) & (values < float(end))


def aligned_trace_segment(
    trace: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray,
    event: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid trace samples aligned to an event's transition time."""
    trace_array = np.asarray(trace, dtype=float).reshape(-1)
    time_array = np.asarray(time, dtype=float).reshape(-1)
    usable = min(trace_array.size, time_array.size)
    if usable <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    relative_time = time_array[:usable] - float(event["transition_time"])
    start = float(event["pre_start_time"])
    end = float(event["post_end_time"])
    keep = np.isfinite(relative_time) & np.isfinite(trace_array[:usable]) & (time_array[:usable] >= start) & (time_array[:usable] < end)
    return relative_time[keep], trace_array[:usable][keep]


def paired_transition_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    group_fields = ("scope", "window_mode", "state_before", "state_after", "metric", "compartment")
    for row in rows:
        if row.get("pre_value") is None or row.get("post_value") is None:
            continue
        groups[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        pre = np.asarray([float(row["pre_value"]) for row in group_rows], dtype=float)
        post = np.asarray([float(row["post_value"]) for row in group_rows], dtype=float)
        keep = np.isfinite(pre) & np.isfinite(post)
        pre, post = pre[keep], post[keep]
        if pre.size == 0:
            continue
        delta = post - pre
        payload = dict(zip(group_fields, key))
        payload.update(
            {
                "comparison": "state_transition_pre_post",
                "n_transition_events": int(pre.size),
                "n_entities": int(len({str(row.get("entity_id", "")) for row, keep_row in zip(group_rows, keep) if keep_row})),
                "mean_pre": float(np.mean(pre)),
                "mean_post": float(np.mean(post)),
                "mean_change": float(np.mean(delta)),
                "median_change": float(np.median(delta)),
                "mean_relative_change": float(np.mean(delta / np.where(np.abs(pre) > 1e-12, np.abs(pre), np.nan))),
            }
        )
        if pre.size >= 2:
            try:
                from scipy import stats

                test = stats.ttest_rel(post, pre, nan_policy="omit")
                payload["paired_statistic"] = float(test.statistic)
                payload["paired_pvalue"] = float(test.pvalue)
            except Exception:
                payload["paired_statistic"] = float("nan")
                payload["paired_pvalue"] = float("nan")
        else:
            payload["paired_statistic"] = float("nan")
            payload["paired_pvalue"] = float("nan")
        output.append(payload)
    return output


def plot_transition_summaries(
    event_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    pipeline_name: str,
    trace_segments: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if not event_rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = ("scope", "window_mode", "state_before", "state_after", "metric", "compartment")
    for row in event_rows:
        if row.get("pre_value") is not None and row.get("post_value") is not None:
            groups[tuple(str(row.get(field, "")) for field in fields)].append(row)
    summary_lookup = {
        tuple(str(row.get(field, "")) for field in fields): row
        for row in paired_transition_summaries(event_rows)
    }
    saved: list[str] = []
    for key, rows in sorted(groups.items()):
        scope, mode, state_before, state_after, metric, compartment = key
        pre = np.asarray([float(row["pre_value"]) for row in rows], dtype=float)
        post = np.asarray([float(row["post_value"]) for row in rows], dtype=float)
        keep = np.isfinite(pre) & np.isfinite(post)
        pre, post = pre[keep], post[keep]
        if pre.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        positions = np.asarray([1.0, 2.0])
        box = ax.boxplot([pre, post], positions=positions, widths=0.55, showfliers=False, patch_artist=True)
        for patch, color in zip(box["boxes"], ("#4c78a8", "#e45756")):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_xticks([1, 2], ["Before", "After"])
        ax.set_ylabel(metric)
        ax.set_title(f"{state_before} → {state_after} | {compartment}\n{scope}, {mode}")
        summary = summary_lookup.get(key, {})
        n_entities = int(summary.get("n_entities", 0))
        try:
            p_value = float(summary.get("paired_pvalue", np.nan))
        except (TypeError, ValueError):
            p_value = float("nan")
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        if stars:
            y_max = float(np.nanmax(np.concatenate((pre, post))))
            y_min = float(np.nanmin(np.concatenate((pre, post))))
            y_span = max(y_max - y_min, 1e-9)
            star_y = y_max + 0.12 * y_span
            ax.text(1.5, star_y, stars, ha="center", va="bottom", fontsize=12)
        ax.text(0.02, 0.97, f"transition events n={pre.size}; entities n={n_entities}", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        filename = "_".join(str(part).replace("/", "-") or "all" for part in (pipeline_name,) + key) + ".svg"
        path = Path(output_root) / "figures" / "state_transitions" / scope / mode / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    trace_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for segment in trace_segments or ():
        if segment.get("relative_time_s") is None or segment.get("values") is None:
            continue
        trace_groups[tuple(str(segment.get(field, "")) for field in ("scope", "window_mode", "metric", "compartment"))].append(segment)
    for key, segments in sorted(trace_groups.items()):
        scope, mode, metric, compartment = key
        window_s = max(float(segment.get("window_s", 60.0)) for segment in segments)
        grid = np.linspace(-window_s, window_s, 241)
        direction_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for segment in segments:
            relative = np.asarray(segment["relative_time_s"], dtype=float)
            values = np.asarray(segment["values"], dtype=float)
            keep = np.isfinite(relative) & np.isfinite(values)
            if keep.sum() < 2:
                continue
            relative, values = relative[keep], values[keep]
            order = np.argsort(relative)
            relative, values = relative[order], values[order]
            unique, unique_indices = np.unique(relative, return_index=True)
            values = values[unique_indices]
            interpolated = np.full(grid.shape, np.nan, dtype=float)
            valid_grid = (grid >= unique[0]) & (grid <= unique[-1])
            interpolated[valid_grid] = np.interp(grid[valid_grid], unique, values)
            direction_groups[(str(segment.get("state_before", "")), str(segment.get("state_after", "")))].append(interpolated)
        if not direction_groups:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        for direction_index, (direction, traces) in enumerate(sorted(direction_groups.items())):
            matrix = np.asarray(traces, dtype=float)
            count = np.sum(np.isfinite(matrix), axis=0)
            mean = np.full(matrix.shape[1], np.nan, dtype=float)
            sem = np.full(matrix.shape[1], np.nan, dtype=float)
            valid_columns = count > 0
            mean[valid_columns] = np.nansum(matrix[:, valid_columns], axis=0) / count[valid_columns]
            sem_columns = count > 1
            if np.any(sem_columns):
                centered = matrix[:, sem_columns] - mean[sem_columns]
                centered[~np.isfinite(centered)] = np.nan
                sem[sem_columns] = np.sqrt(np.nansum(centered ** 2, axis=0) / (count[sem_columns] - 1)) / np.sqrt(count[sem_columns])
            color = ("#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214")[direction_index % 5]
            label = f"{direction[0]} → {direction[1]} (n={len(traces)})"
            ax.plot(grid, mean, color=color, linewidth=2.0, label=label)
            ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlim(-window_s, window_s)
        ax.set_xlabel("Time from transition (s)")
        ax.set_ylabel("dF/F")
        ax.set_title(f"Aligned mean dF/F | {compartment}\n{scope}, {mode}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        filename = "_".join(str(part).replace("/", "-") or "all" for part in (pipeline_name,) + key) + "_trace.svg"
        path = Path(output_root) / "figures" / "state_transitions" / scope / mode / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved
