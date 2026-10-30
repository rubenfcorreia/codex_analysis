"""Utilities for within-experiment state-transition analyses."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.shared.plots.figure_io import save_figure

TRANSITION_SCHEMA_VERSION = 2
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
    max_event_rows = raw.get("max_event_rows")
    overflow_policy = str(raw.get("transition_overflow_policy", raw.get("overflow_policy", "fail")) or "fail").strip().lower()
    if overflow_policy not in {"fail", "partial"}:
        overflow_policy = "fail"
    try:
        max_event_rows = int(max_event_rows) if max_event_rows is not None else None
    except (TypeError, ValueError):
        max_event_rows = None
    if max_event_rows is not None and max_event_rows <= 0:
        max_event_rows = None
    return {
        "enabled": enabled,
        "window_s": float(window_s),
        "window_modes": list(dict.fromkeys(modes)),
        "scopes": list(dict.fromkeys(scopes)),
        "metrics": list(dict.fromkeys(metrics)),
        "max_event_rows": max_event_rows,
        "transition_overflow_policy": overflow_policy,
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
        aggregation_level = str(group_rows[0].get("aggregation_level", "event"))
        expdays = {str(row.get("expday", "")) for row, keep_row in zip(group_rows, keep) if keep_row}
        entities = {str(row.get("entity_id", "")) for row, keep_row in zip(group_rows, keep) if keep_row}
        animals = {str(row.get("animal_id", "")) for row, keep_row in zip(group_rows, keep) if keep_row}
        if aggregation_level == "expday":
            entity_count = sum(int(row.get("n_entities", 0) or 0) for row, keep_row in zip(group_rows, keep) if keep_row)
        else:
            entity_count = len(entities - {""})
        raw_events = sum(int(row.get("n_transition_events", 1) or 0) for row, keep_row in zip(group_rows, keep) if keep_row)
        payload.update(
            {
                "comparison": "state_transition_pre_post",
                "aggregation_level": aggregation_level,
                "n_expdays": int(len(expdays)),
                "n_animals": int(len(animals - {""})),
                "n_transition_events": int(raw_events),
                "n_entities": int(entity_count),
                "n_replicates": int(pre.size),
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

                if np.allclose(post, pre, equal_nan=True):
                    payload["paired_statistic"] = 0.0
                    payload["paired_pvalue"] = 1.0
                    payload["paired_test_status"] = "identical_values"
                else:
                    test = stats.ttest_rel(post, pre, nan_policy="omit")
                    payload["paired_statistic"] = float(test.statistic)
                    payload["paired_pvalue"] = float(test.pvalue)
                    payload["paired_test_status"] = "ok"
            except Exception as exc:
                payload["paired_statistic"] = float("nan")
                payload["paired_pvalue"] = float("nan")
                payload["paired_test_status"] = "failed"
                payload["paired_test_warning"] = str(exc)
        else:
            payload["paired_statistic"] = float("nan")
            payload["paired_pvalue"] = float("nan")
        output.append(payload)
    return output


def _finite_mean(values: Sequence[Any]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def aggregate_transition_rows_expday(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Balance transition values so each experiment-day contributes once."""
    if not rows:
        return []
    fields = ("scope", "window_mode", "state_before", "state_after", "metric", "compartment")
    entity_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pre_value") is None or row.get("post_value") is None:
            continue
        expday = str(row.get("expday") or row.get("day_id") or row.get("expid") or "")
        key = tuple(str(row.get(field, "")) for field in fields) + (expday, str(row.get("entity_id", "")))
        entity_groups[key].append(row)

    entity_rows: list[dict[str, Any]] = []
    for key, group in entity_groups.items():
        *field_values, expday, entity_id = key
        first = dict(group[0])
        first.update(dict(zip(fields, field_values)))
        first.update({"aggregation_level": "entity_expday", "expday": expday, "entity_id": entity_id, "pre_value": _finite_mean([row.get("pre_value") for row in group]), "post_value": _finite_mean([row.get("post_value") for row in group]), "n_transition_events": len(group), "n_entities": 1})
        entity_rows.append(first)

    expday_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in entity_rows:
        key = tuple(str(row.get(field, "")) for field in fields) + (str(row.get("expday", "")),)
        expday_groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(expday_groups.items()):
        *field_values, expday = key
        first = dict(group[0])
        first.update(dict(zip(fields, field_values)))
        first.update({"aggregation_level": "expday", "expday": expday, "pre_value": _finite_mean([row.get("pre_value") for row in group]), "post_value": _finite_mean([row.get("post_value") for row in group]), "n_transition_events": int(sum(int(row.get("n_transition_events", 0) or 0) for row in group)), "n_entities": int(len({str(row.get("entity_id", "")) for row in group} - {""}))})
        output.append(first)
    return output


def plot_transition_summaries(
    event_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    pipeline_name: str,
    trace_segments: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Write one three-panel figure for each transition/scope/window/compartment."""
    if not event_rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    fields = ("scope", "window_mode", "state_before", "state_after", "compartment")
    metric_rows: dict[tuple[str, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in aggregate_transition_rows_expday(event_rows):
        if row.get("pre_value") is None or row.get("post_value") is None:
            continue
        key = tuple(str(row.get(field, "")) for field in fields)
        metric_rows[key][str(row.get("metric", ""))].append(row)

    trace_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for index, raw_segment in enumerate(trace_segments or ()):
        segment = dict(raw_segment)
        if segment.get("relative_time_s") is None or segment.get("values") is None:
            continue
        segment.setdefault("expday", segment.get("day_id") or segment.get("expid") or f"segment-{index}")
        segment.setdefault("entity_id", f"segment-{index}")
        key = tuple(str(segment.get(field, "")) for field in fields)
        trace_groups[key].append(segment)

    saved: list[str] = []
    colors = ("#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214")
    for key in sorted(set(metric_rows) | set(trace_groups)):
        scope, mode, state_before, state_after, compartment = key
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": (1.55, 1.0, 1.0)})
        ax_trace, ax_activity, ax_frequency = axes

        segments = trace_groups.get(key, [])
        directions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for segment in segments:
            directions[(str(segment.get("state_before", state_before)), str(segment.get("state_after", state_after)))].append(segment)
        if directions:
            window_s = max(float(segment.get("window_s", 60.0)) for segment in segments)
            grid = np.linspace(-window_s, window_s, 241)
            for direction_index, (direction, direction_segments) in enumerate(sorted(directions.items())):
                matrices = []
                for segment in direction_segments:
                    relative = np.asarray(segment["relative_time_s"], dtype=float)
                    values = np.asarray(segment["values"], dtype=float)
                    keep = np.isfinite(relative) & np.isfinite(values)
                    if keep.sum() < 2:
                        continue
                    relative, values = relative[keep], values[keep]
                    order = np.argsort(relative)
                    relative, values = relative[order], values[order]
                    unique, indices = np.unique(relative, return_index=True)
                    if unique.size < 2:
                        continue
                    matrix = np.full(grid.shape, np.nan)
                    valid = (grid >= unique[0]) & (grid <= unique[-1])
                    matrix[valid] = np.interp(grid[valid], unique, values[indices])
                    matrices.append(matrix)
                if not matrices:
                    continue
                matrix = np.asarray(matrices)
                mean = np.nanmean(matrix, axis=0)
                count = np.sum(np.isfinite(matrix), axis=0)
                sem = np.full(mean.shape, np.nan)
                valid_sem = count > 1
                if np.any(valid_sem):
                    sem[valid_sem] = np.nanstd(matrix[:, valid_sem], axis=0, ddof=1) / np.sqrt(count[valid_sem])
                color = colors[direction_index % len(colors)]
                label = f"{direction[0]} -> {direction[1]} (n={len(matrices)})"
                ax_trace.plot(grid, mean, color=color, lw=2.0, label=label)
                ax_trace.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
            ax_trace.axvline(0, color="#333333", ls="--", lw=1)
            ax_trace.set_xlim(-window_s, window_s)
        else:
            ax_trace.text(0.5, 0.5, "No aligned trace data", transform=ax_trace.transAxes, ha="center", va="center", color="#666666")
        ax_trace.set_title("Aligned dF/F")
        ax_trace.set_xlabel("Time from transition (s)")
        ax_trace.set_ylabel("dF/F")
        if ax_trace.has_data():
            ax_trace.legend(frameon=False, fontsize=8)

        for ax, metric, title, ylabel in (
            (ax_activity, "mean_activity", "Mean activity difference", "After - before"),
            (ax_frequency, "event_frequency_per_min", "Event-frequency difference", "After - before"),
        ):
            grouped = metric_rows.get(key, {}).get(metric, [])
            direction_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
            for row in grouped:
                direction_groups[(str(row.get("state_before", "")), str(row.get("state_after", "")))].append(row)
            plotted = False
            positions = []
            labels = []
            for direction_index, (direction, rows) in enumerate(sorted(direction_groups.items())):
                pre = np.asarray([row.get("pre_value", np.nan) for row in rows], dtype=float)
                post = np.asarray([row.get("post_value", np.nan) for row in rows], dtype=float)
                keep = np.isfinite(pre) & np.isfinite(post)
                if not np.any(keep):
                    continue
                differences = post[keep] - pre[keep]
                xpos = direction_index + 1
                color = colors[direction_index % len(colors)]
                jitter = np.linspace(-0.08, 0.08, differences.size) if differences.size > 1 else np.asarray([0.0])
                ax.scatter(np.full(differences.size, xpos) + jitter, differences, s=14, alpha=0.55, color=color)
                ax.plot([xpos - 0.16, xpos + 0.16], [np.mean(differences)] * 2, color=color, lw=2.4)
                ax.text(xpos, np.mean(differences), f"delta={np.mean(differences):.3g}", ha="center", va="bottom", fontsize=8)
                positions.append(xpos)
                labels.append(f"{direction[0]} ->\n{direction[1]}")
                plotted = True
            ax.axhline(0, color="#333333", ls="--", lw=1)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if positions:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)
            if not plotted:
                ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center", color="#666666")
            ax.grid(axis="y", alpha=0.2)

        fig.suptitle(f"{state_before} -> {state_after} | {compartment}\n{scope}, {mode}", fontsize=14)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.82, wspace=0.32)
        filename = "state_transition_" + "_".join(str(part).replace("/", "-") or "all" for part in (pipeline_name,) + key) + ".svg"
        path = Path(output_root) / scope / mode / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, path, extra_formats=(), format="svg", bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved
