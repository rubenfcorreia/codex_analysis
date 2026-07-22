from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from analysis.shared.analysis_families.coincidence import coincident_event_runs, event_run_onsets_match
from analysis.shared.state_utils import ensure_dir, safe_filename_component


DEFAULT_COINCIDENCE_EXAMPLE_FIGURES_DIRNAME = "coincidence_event_examples"


def _figure_dir(root: Path | str, *parts: str) -> Path:
    return ensure_dir(Path(root).joinpath(*parts))


def coincidence_example_figure_dir(
    root: Path | str,
    *,
    mode: Any,
    day_id: Any,
    state: Any,
    event_detection_method: Any = None,
) -> Path:
    parts = [DEFAULT_COINCIDENCE_EXAMPLE_FIGURES_DIRNAME]
    if event_detection_method is not None and str(event_detection_method).strip():
        parts.append(safe_filename_component(event_detection_method))
    parts.extend(
        [
            safe_filename_component(mode),
            safe_filename_component(day_id),
            safe_filename_component(state),
        ]
    )
    return _figure_dir(root, *parts)


def build_coincidence_example_figure_path(
    output_dir: Path | str,
    *,
    expid: Any,
    mode: Any,
    day_id: Any,
    state: Any,
    pair_unit_id: Any,
    rank: int,
    event_detection_method: Any = None,
) -> Path:
    figure_dir = coincidence_example_figure_dir(
        output_dir,
        mode=mode,
        day_id=day_id,
        state=state,
        event_detection_method=event_detection_method,
    )
    figure_name = f"{safe_filename_component(expid)}_{int(rank):02d}_{safe_filename_component(pair_unit_id)}_coincidence_example.svg"
    return figure_dir / figure_name


def _event_run_center(run: Tuple[int, int]) -> float:
    return 0.5 * (float(int(run[0])) + float(int(run[1])))


def _event_run_overlaps_window(window_start: int, window_end: int, run: Tuple[int, int]) -> bool:
    return bool(max(int(window_start), int(run[0])) < min(int(window_end), int(run[1])))


def _window_overlaps_any(window_start: int, window_end: int, runs: Sequence[Tuple[int, int]]) -> bool:
    return any(_event_run_overlaps_window(window_start, window_end, run) for run in runs)


def _select_event_example_windows(
    trace_size: int,
    focus_event_runs: Sequence[Tuple[int, int]],
    coincident_focus_runs: Sequence[Tuple[int, int]],
    *,
    max_examples: int = 10,
    pad_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    trace_size = int(trace_size)
    if trace_size <= 0:
        return []
    if pad_frames is None:
        pad_frames = max(4, min(25, max(6, trace_size // 12)))
    pad_frames = int(max(1, pad_frames))

    selected: List[Dict[str, Any]] = []
    ordered_focus = sorted([(int(start), int(end)) for start, end in focus_event_runs], key=_event_run_center)
    ordered_coincident = sorted([(int(start), int(end)) for start, end in coincident_focus_runs], key=_event_run_center)
    coincident_onsets = {int(run[0]) for run in ordered_coincident}
    focus_only = [run for run in ordered_focus if int(run[0]) not in coincident_onsets]

    def _append_run(run: Tuple[int, int], kind: str) -> None:
        window_start = max(0, int(run[0]) - pad_frames)
        window_end = min(trace_size, int(run[1]) + pad_frames)
        if window_end <= window_start:
            return
        selected.append(
            {
                "kind": kind,
                "label": kind.replace("_", " "),
                "run": run,
                "window_start": window_start,
                "window_end": window_end,
            }
        )

    for run in ordered_coincident[:max_examples]:
        _append_run(run, "coincident")

    if len(selected) < max_examples:
        remaining = max_examples - len(selected)
        if len(focus_only) > remaining:
            indices = np.unique(np.round(np.linspace(0, len(focus_only) - 1, remaining)).astype(int))
            focus_only = [focus_only[index] for index in indices[:remaining]]
        for run in focus_only:
            if len(selected) >= max_examples:
                break
            _append_run(run, "focus_event")

    if len(selected) < max_examples:
        context_centers = np.linspace(0, max(trace_size - 1, 0), max_examples * 4 if trace_size > 1 else 1)
        for center_value in context_centers:
            if len(selected) >= max_examples:
                break
            center = int(round(float(center_value)))
            window_start = max(0, center - pad_frames)
            window_end = min(trace_size, center + pad_frames + 1)
            if window_end <= window_start:
                continue
            if _window_overlaps_any(window_start, window_end, ordered_focus):
                continue
            if any(_window_overlaps_any(window_start, window_end, [item["run"]]) for item in selected if item.get("run") is not None):
                continue
            selected.append(
                {
                    "kind": "context",
                    "label": "context",
                    "run": None,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )

    if not selected:
        selected.append(
            {
                "kind": "context",
                "label": "context",
                "run": None,
                "window_start": 0,
                "window_end": trace_size,
            }
        )

    selected = sorted(selected, key=lambda item: (int(item["window_start"]), int(item["window_end"]), str(item["kind"])))
    if len(selected) > max_examples:
        selected = selected[:max_examples]
    while len(selected) < max_examples:
        selected.append(dict(selected[-1]))
    return selected


def _display_trace(trace: np.ndarray, event_info: Mapping[str, Any]) -> np.ndarray:
    method = str(event_info.get("method") or event_info.get("primary_method") or "amplitude").strip().lower()
    trace = np.asarray(trace, dtype=float).reshape(-1)
    if method == "derivative":
        return np.diff(trace, prepend=np.nan)
    return trace


def _display_label(label: str, event_info: Mapping[str, Any]) -> str:
    method = str(event_info.get("method") or event_info.get("primary_method") or "amplitude").strip().lower()
    if method == "derivative":
        return f"First derivative of {label}"
    return label


def plot_coincidence_event_example_figure(
    *,
    output_path: Path | str,
    time: Sequence[float] | np.ndarray,
    focus_trace: Sequence[float] | np.ndarray,
    reference_trace: Sequence[float] | np.ndarray,
    focus_event_info: Mapping[str, Any],
    reference_event_info: Mapping[str, Any],
    title: str,
    focus_label: str,
    reference_label: str,
    max_examples: int = 10,
) -> Optional[str]:
    if plt is None:
        return None

    time = np.asarray(time, dtype=float).reshape(-1)
    focus_trace = np.asarray(focus_trace, dtype=float).reshape(-1)
    reference_trace = np.asarray(reference_trace, dtype=float).reshape(-1)
    usable = min(time.size, focus_trace.size, reference_trace.size)
    if usable <= 0:
        return None
    time = time[:usable]
    focus_trace = focus_trace[:usable]
    reference_trace = reference_trace[:usable]

    valid = np.isfinite(time) & (np.isfinite(focus_trace) | np.isfinite(reference_trace))
    if not np.any(valid):
        return None

    focus_event_info = dict(focus_event_info or {})
    reference_event_info = dict(reference_event_info or {})
    focus_event_runs = [(int(start), int(end)) for start, end in (focus_event_info.get("event_runs") or [])]
    reference_event_runs = [(int(start), int(end)) for start, end in (reference_event_info.get("event_runs") or [])]
    coincident_focus_runs = coincident_event_runs(focus_event_info, reference_event_info)
    selected_windows = _select_event_example_windows(
        usable,
        focus_event_runs,
        coincident_focus_runs,
        max_examples=max_examples,
    )
    if not selected_windows:
        return None

    display_focus_trace = _display_trace(focus_trace, focus_event_info)
    display_reference_trace = _display_trace(reference_trace, reference_event_info)
    focus_display_label = _display_label(focus_label, focus_event_info)
    reference_display_label = _display_label(reference_label, reference_event_info)
    focus_threshold = focus_event_info.get("threshold")
    try:
        focus_threshold = float(focus_threshold)
    except Exception:
        focus_threshold = float("nan")

    focus_color = "#7a5195"
    reference_color = "#4477aa"
    coincident_color = "#2ca02c"
    focus_event_color = "#f58518"
    reference_event_color = "#1f77b4"

    compact_rc = {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 16,
    }

    n_panels = len(selected_windows)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    with plt.rc_context(compact_rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, max(4.0, 2.35 * nrows)), squeeze=False)
        axes_flat = axes.ravel()
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        for idx, window in enumerate(selected_windows):
            ax = axes_flat[idx]
            window_start = int(window["window_start"])
            window_end = int(window["window_end"])
            window_time = time[window_start:window_end]
            window_focus = display_focus_trace[window_start:window_end]
            window_reference = display_reference_trace[window_start:window_end]
            window_valid = np.isfinite(window_time) & (np.isfinite(window_focus) | np.isfinite(window_reference))
            if not np.any(window_valid):
                ax.text(0.5, 0.5, "No valid signal", transform=ax.transAxes, ha="center", va="center", fontsize=9)
                ax.set_axis_off()
                continue

            window_focus_runs = [run for run in focus_event_runs if _window_overlaps_any(window_start, window_end, [run])]
            window_reference_runs = [run for run in reference_event_runs if _window_overlaps_any(window_start, window_end, [run])]
            window_coincident_runs = [run for run in coincident_focus_runs if _window_overlaps_any(window_start, window_end, [run])]

            if window_coincident_runs:
                reference_idx = int(window_coincident_runs[0][0])
                reference_time = float(time[reference_idx]) if 0 <= reference_idx < time.size else float(window_time[0])
            elif window_focus_runs:
                reference_idx = int(window_focus_runs[0][0])
                reference_time = float(time[reference_idx]) if 0 <= reference_idx < time.size else float(window_time[0])
            else:
                reference_time = float(np.nanmedian(window_time[window_valid]))

            window_time_rel = window_time - reference_time

            ax_focus = ax
            ax_reference = ax_focus.twinx()

            ax_focus.plot(
                window_time_rel[window_valid],
                window_focus[window_valid],
                color=focus_color,
                linewidth=1.1,
                label=focus_display_label,
            )
            ax_reference.plot(
                window_time_rel[window_valid],
                window_reference[window_valid],
                color=reference_color,
                linewidth=1.0,
                alpha=0.9,
                label=reference_display_label,
            )

            if np.isfinite(focus_threshold):
                ax_focus.axhline(focus_threshold, color="#8b0000", linestyle="--", linewidth=0.8, alpha=0.9)

            for run in window_reference_runs:
                start_i, end_i = run
                if 0 <= start_i < time.size and 0 <= end_i - 1 < time.size:
                    color = coincident_color if any(event_run_onsets_match(run, focus_run) for focus_run in window_focus_runs) else reference_event_color
                    ax_reference.axvspan(
                        time[start_i] - reference_time,
                        time[end_i - 1] - reference_time,
                        color=color,
                        alpha=0.08 if color == reference_event_color else 0.10,
                        lw=0,
                        zorder=1,
                    )

            for run in window_focus_runs:
                start_i, end_i = run
                if 0 <= start_i < time.size and 0 <= end_i - 1 < time.size:
                    coincident = any(event_run_onsets_match(run, ref_run) for ref_run in reference_event_runs)
                    color = coincident_color if coincident else focus_event_color
                    ax_focus.axvspan(
                        time[start_i] - reference_time,
                        time[end_i - 1] - reference_time,
                        color=color,
                        alpha=0.20 if coincident else 0.14,
                        lw=0,
                        zorder=4,
                    )

            ax_focus.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
            ax_focus.set_title(str(window.get("label") or "event"), fontsize=10)
            ax_focus.set_xlabel("Time (s)")
            ax_focus.set_ylabel(focus_display_label)
            ax_reference.set_ylabel(reference_display_label)
            ax_focus.grid(axis="y", alpha=0.2)
            y0, y1 = ax_focus.get_ylim()
            y_range = y1 - y0 if np.isfinite(y0) and np.isfinite(y1) else float("nan")
            if np.isfinite(y_range) and y_range > 0:
                info = (
                    f"focus={len(window_focus_runs)} | reference={len(window_reference_runs)} | "
                    f"coincident={len(window_coincident_runs)}"
                )
                ax_focus.text(
                    0.02,
                    0.98,
                    info,
                    transform=ax_focus.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#444444",
                )

        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()

        fig.tight_layout(rect=(0, 0, 1, 0.985))
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(output_path)


__all__ = [
    "DEFAULT_COINCIDENCE_EXAMPLE_FIGURES_DIRNAME",
    "build_coincidence_example_figure_path",
    "coincidence_example_figure_dir",
    "plot_coincidence_event_example_figure",
]
