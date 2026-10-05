from __future__ import annotations

"""Pipeline-local mean dF/F progression summaries and figures."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis.shared.analysis_families.common_helpers import classify_movie_name, movie_feature_blocks


DEFAULT_PROGRESSION_CONFIG = {
    "enabled": False,
    "blank_bin_s": 1.0,
    "sleep_time_step_s": 1.0,
    "max_blank_duration_s": None,
    "max_sleep_duration_s": None,
    "spine_signals": ["raw", "spine_specific"],
}


def _finite_mean(values: Sequence[Any]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_sem(values: Sequence[Any]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan")


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _trial_label(row: Mapping[str, Any]) -> str:
    # Use the same movie-feature classification used by the shared/dendrite
    # state detector, while also accepting already-normalized state labels.
    values = [row.get("state_label"), row.get("state"), row.get("trial_type"), row.get("category")]
    if any("blank" in str(value or "").strip().lower() for value in values):
        return "blank"
    blocks = movie_feature_blocks(row, list(row.keys()))
    if len(blocks) == 1 and classify_movie_name(blocks[0].get("name")) == "blank":
        return "blank"
    return str(next((value for value in values if value not in (None, "")), "")).strip().lower()


def _trial_interval(row: Mapping[str, Any]) -> tuple[float, float] | None:
    start = row.get("start", row.get("onset", row.get("trial_start", row.get("time"))))
    end = row.get("end", row.get("trial_end"))
    start_value = _as_float(start)
    if start_value is None:
        return None
    end_value = _as_float(end)
    if end_value is None:
        duration = _as_float(row.get("duration"))
        end_value = start_value + duration if duration is not None else start_value
    return (start_value, end_value) if end_value > start_value else None


def _normalise_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_PROGRESSION_CONFIG)
    if isinstance(config, Mapping):
        merged.update(dict(config))
    try:
        merged["blank_bin_s"] = max(float(merged.get("blank_bin_s", 1.0)), 1e-6)
    except (TypeError, ValueError):
        merged["blank_bin_s"] = 1.0
    max_duration = _as_float(merged.get("max_blank_duration_s"))
    merged["max_blank_duration_s"] = max_duration if max_duration and max_duration > 0 else None
    try:
        merged["sleep_time_step_s"] = max(float(merged.get("sleep_time_step_s", 1.0)), 1e-6)
    except (TypeError, ValueError):
        merged["sleep_time_step_s"] = 1.0
    max_sleep_duration = _as_float(merged.get("max_sleep_duration_s"))
    merged["max_sleep_duration_s"] = max_sleep_duration if max_sleep_duration and max_sleep_duration > 0 else None
    signals = merged.get("spine_signals", ["raw", "spine_specific"])
    if isinstance(signals, str):
        signals = [part.strip() for part in signals.split(",") if part.strip()]
    merged["spine_signals"] = [str(signal).strip().lower() for signal in signals] if isinstance(signals, Sequence) else ["raw", "spine_specific"]
    return merged


def _base_row(*, pipeline: str, compartment: str, signal_type: str, expid: str, animal_id: Any, date: Any) -> dict[str, Any]:
    return {
        "pipeline": str(pipeline),
        "compartment": str(compartment),
        "signal_type": str(signal_type),
        "expid": str(expid),
        "animal_id": str(animal_id or ""),
        "date": str(date or ""),
    }


def _trace_matrix_mean(matrix: Any, time: Any, start: float, end: float) -> tuple[float, int, int]:
    values = np.asarray(matrix, dtype=float)
    axis = np.asarray(time, dtype=float).reshape(-1)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or axis.size == 0:
        return float("nan"), 0, 0
    usable = min(values.shape[1], axis.size)
    mask = np.isfinite(axis[:usable]) & (axis[:usable] >= start) & (axis[:usable] < end)
    selected = values[:, :usable][:, mask]
    finite = selected[np.isfinite(selected)]
    return (_finite_mean(finite), int(values.shape[0]), int(finite.size)) if finite.size else (float("nan"), int(values.shape[0]), 0)


def _expid_sort_key(expid: Any) -> tuple[str, int, str]:
    text = str(expid)
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d+)", text)
    return (match.group(1), int(match.group(2)), text) if match else (text, 0, text)


def _sleep_rows_from_segments(*, pipeline: str, segments: Mapping[tuple[str, str, str, str], Sequence[Mapping[str, Any]]], time_step_s: float, max_duration_s: float | None) -> list[dict[str, Any]]:
    """Concatenate same-day segments and interpolate animal-days to a shared grid."""
    day_traces: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for (compartment, signal_type, animal_id, date), source_segments in segments.items():
        elapsed = 0.0
        trace_parts, time_parts, boundaries, source_expids = [], [], [], []
        for segment in sorted(source_segments, key=lambda item: _expid_sort_key(item.get("expid", ""))):
            axis = np.asarray(segment.get("time", []), dtype=float).reshape(-1)
            matrix = np.asarray(segment.get("matrix", []), dtype=float)
            if matrix.ndim == 1:
                matrix = matrix[None, :]
            usable = min(axis.size, matrix.shape[1] if matrix.ndim == 2 else 0)
            if usable == 0:
                continue
            axis, matrix = axis[:usable], matrix[:, :usable]
            valid_axis = np.isfinite(axis)
            if not np.any(valid_axis):
                continue
            axis, matrix = axis[valid_axis], matrix[:, valid_axis]
            order = np.argsort(axis)
            axis, matrix = axis[order], matrix[:, order]
            finite = np.isfinite(matrix)
            counts = finite.sum(axis=0)
            trace = np.full(axis.size, np.nan)
            good = counts > 0
            if np.any(good):
                trace[good] = np.nansum(matrix[:, good], axis=0) / counts[good]
            if not np.any(np.isfinite(trace)):
                continue
            local = axis - float(axis[0])
            duration = float(local[-1]) if local.size > 1 else 0.0
            boundaries.append({"expid": str(segment.get("expid", "")), "start_s": elapsed, "end_s": elapsed + duration, "n_rois": int(matrix.shape[0]), "n_frames": int(np.sum(np.isfinite(trace)))})
            trace_parts.append(trace)
            time_parts.append(local + elapsed)
            source_expids.append(str(segment.get("expid", "")))
            elapsed += duration
        if trace_parts:
            day_traces[(compartment, signal_type, animal_id, date)] = {"time": np.concatenate(time_parts), "trace": np.concatenate(trace_parts), "source_expids": source_expids, "boundaries": boundaries, "duration": elapsed}
    by_series: dict[tuple[str, str], list[tuple[tuple[str, str, str, str], dict[str, Any]]]] = defaultdict(list)
    for key, value in day_traces.items():
        by_series[(key[0], key[1])].append((key, value))
    rows: list[dict[str, Any]] = []
    for (compartment, signal_type), days in by_series.items():
        duration = min(value["duration"] for _, value in days)
        if max_duration_s is not None:
            duration = min(duration, max_duration_s)
        grid = np.arange(0.0, duration + time_step_s * 0.5, time_step_s) if duration > 0 else np.asarray([0.0])
        for (compartment, signal_type, animal_id, date), value in days:
            axis, trace = np.asarray(value["time"]), np.asarray(value["trace"])
            valid = np.isfinite(axis) & np.isfinite(trace)
            if not np.any(valid):
                continue
            x, y = axis[valid], trace[valid]
            unique, indices = np.unique(x, return_index=True)
            y = y[indices]
            interpolated = np.interp(grid, unique, y, left=np.nan, right=np.nan)
            for elapsed_time, mean in zip(grid, interpolated):
                if not np.isfinite(mean):
                    continue
                first_expid = value["source_expids"][0] if value["source_expids"] else ""
                row = _base_row(pipeline=pipeline, compartment=compartment, signal_type=signal_type, expid=first_expid, animal_id=animal_id, date=date)
                row.update({"day_id": f"{animal_id}_{date}", "source_expids": json.dumps(value["source_expids"]), "segment_boundaries": json.dumps(value["boundaries"]), "elapsed_time_s": float(elapsed_time), "mean_dff": float(mean), "n_rois": int(max((item["n_rois"] for item in value["boundaries"]), default=0)), "n_trials": 0, "n_frames": int(np.sum(np.abs(axis - elapsed_time) <= time_step_s / 2)), "status": "ok"})
                rows.append(row)
    return rows


def _blank_rows_for_traces(*, pipeline: str, expid: str, animal_id: Any, date: Any, time: Any, trial_rows: Sequence[Mapping[str, Any]], traces: Mapping[tuple[str, str], Sequence[Any]], bin_s: float, max_duration_s: float | None) -> list[dict[str, Any]]:
    intervals = [interval for trial in trial_rows if _trial_label(trial) == "blank" for interval in [_trial_interval(trial)] if interval is not None]
    rows: list[dict[str, Any]] = []
    if not intervals:
        return rows
    max_duration = max((end - start for start, end in intervals), default=0.0)
    if max_duration_s is not None:
        max_duration = min(max_duration, max_duration_s)
    n_bins = int(math.ceil(max_duration / bin_s)) if max_duration > 0 else 0
    for (compartment, signal_type), matrix in sorted(traces.items()):
        for bin_index in range(n_bins):
            bin_start = bin_index * bin_s
            bin_end = min((bin_index + 1) * bin_s, max_duration)
            values: list[float] = []
            total_frames = 0
            n_rois = 0
            for trial_start, trial_end in intervals:
                local_end = min(trial_end, trial_start + bin_end)
                local_start = trial_start + bin_start
                if local_end <= local_start:
                    continue
                mean, roi_count, frame_count = _trace_matrix_mean(matrix, time, local_start, local_end)
                if np.isfinite(mean):
                    values.append(mean)
                n_rois = max(n_rois, roi_count)
                total_frames += frame_count
            if not values:
                continue
            row = _base_row(pipeline=pipeline, compartment=compartment, signal_type=signal_type, expid=expid, animal_id=animal_id, date=date)
            row.update({"time_bin_s": float(bin_start), "time_bin_end_s": float(bin_end), "mean_dff": _finite_mean(values), "n_rois": int(n_rois), "n_trials": int(len(values)), "n_frames": int(total_frames), "status": "ok"})
            rows.append(row)
    return rows


def _add_summary_rows(blank_rows: Sequence[Mapping[str, Any]], sleep_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, Any], list[float]] = defaultdict(list)
    for row in blank_rows:
        grouped[("blank_trial", str(row.get("compartment")), str(row.get("signal_type")), str(row.get("time_bin_s")), None)].append(float(row.get("mean_dff", np.nan)))
    for row in sleep_rows:
        grouped[("sleep_progression", str(row.get("compartment")), str(row.get("signal_type")), str(row.get("elapsed_time_s")), None)].append(float(row.get("mean_dff", np.nan)))
    result: list[dict[str, Any]] = []
    for (analysis, compartment, signal_type, time_bin, session_order), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        finite = [value for value in values if np.isfinite(value)]
        result.append({"analysis": analysis, "compartment": compartment, "signal_type": signal_type, "time_bin_s": time_bin, "session_order": session_order, "mean_dff": _finite_mean(finite), "sem_dff": _finite_sem(finite), "n_sessions": len(finite)})
    return result


def _session_order(rows: list[dict[str, Any]]) -> None:
    keys = {}
    for row in rows:
        expid = str(row.get("expid", ""))
        match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d+)", expid)
        keys[expid] = (match.group(1), int(match.group(2))) if match else (expid, 0)
    for row in rows:
        row["session_order"] = int(sorted(keys, key=lambda expid: keys[expid]).index(str(row.get("expid", ""))) + 1)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(root: Path, blank_rows: Sequence[Mapping[str, Any]], sleep_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    compartments = sorted({str(row.get("compartment")) for row in [*blank_rows, *sleep_rows]})
    for compartment in compartments:
        comp_root = root / compartment
        figure_root = comp_root / "figures"
        figure_root.mkdir(parents=True, exist_ok=True)
        signals = sorted({str(row.get("signal_type")) for row in [*blank_rows, *sleep_rows] if str(row.get("compartment")) == compartment})
        figure_specs = [(signals, f"{compartment}_progression.svg")]
        if compartment == "spine":
            figure_specs.extend(([signal], f"spine_{signal}_progression.svg") for signal in signals)
        for figure_signals, filename in figure_specs:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
            for signal in figure_signals:
                b = [row for row in blank_rows if row.get("compartment") == compartment and row.get("signal_type") == signal]
                s = [row for row in sleep_rows if row.get("compartment") == compartment and row.get("signal_type") == signal]
                if b:
                    grouped: dict[float, list[float]] = defaultdict(list)
                    for row in b:
                        grouped[float(row["time_bin_s"])].append(float(row["mean_dff"]))
                    x = sorted(grouped)
                    y = [_finite_mean(grouped[value]) for value in x]
                    e = [_finite_sem(grouped[value]) for value in x]
                    axes[0].plot(x, y, marker="o", label=signal)
                    axes[0].fill_between(x, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e), alpha=0.15)
                if s:
                    grouped_sleep: dict[float, list[float]] = defaultdict(list)
                    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                    for row in s:
                        grouped_sleep[float(row.get("elapsed_time_s", 0.0))].append(float(row["mean_dff"]))
                        by_day[str(row.get("day_id", row.get("animal_id", "")))].append(row)
                    for day_rows in by_day.values():
                        day_rows = sorted(day_rows, key=lambda row: float(row.get("elapsed_time_s", 0.0)))
                        axes[1].plot([float(row.get("elapsed_time_s", 0.0)) / 60.0 for row in day_rows], [float(row["mean_dff"]) for row in day_rows], color="0.5", alpha=0.22, linewidth=0.8)
                    x = sorted(grouped_sleep)
                    y = [_finite_mean(grouped_sleep[value]) for value in x]
                    e = [_finite_sem(grouped_sleep[value]) for value in x]
                    axes[1].plot(np.asarray(x) / 60.0, y, linewidth=2.2, label=signal)
                    axes[1].fill_between(np.asarray(x) / 60.0, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e), alpha=0.15)
            axes[0].set(title="Blank-trial progression", xlabel="Time from blank onset (s)", ylabel="Mean dF/F")
            axes[1].set(title="Sleep progression", xlabel="Elapsed sleep time (min)", ylabel="Mean dF/F")
            for axis in axes:
                axis.grid(alpha=0.2)
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(handles=handles, labels=labels, frameon=False)
            path = figure_root / filename
            fig.savefig(path, format="svg")
            plt.close(fig)
            paths.append(str(path))
    return paths


def _finalize(root: Path, blank_rows: list[dict[str, Any]], sleep_rows: list[dict[str, Any]], config: Mapping[str, Any], pipeline: str) -> dict[str, Any]:
    for compartment in sorted({str(row.get("compartment")) for row in [*blank_rows, *sleep_rows]}):
        comp_root = root / compartment
        _write_rows(comp_root / "blank_trial_progression.csv", [row for row in blank_rows if row.get("compartment") == compartment])
        _write_rows(comp_root / "sleep_expid_progression.csv", [row for row in sleep_rows if row.get("compartment") == compartment])
        _write_rows(comp_root / "progression_summary.csv", [row for row in _add_summary_rows(blank_rows, sleep_rows) if row.get("compartment") == compartment])
    figures = _plot_outputs(root, blank_rows, sleep_rows)
    manifest = {"pipeline": pipeline, "config": dict(config), "blank_rows": len(blank_rows), "sleep_rows": len(sleep_rows), "figures": figures}
    root.mkdir(parents=True, exist_ok=True)
    (root / "progression_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest


def run_soma_bouton_progression(contexts: Iterable[Any], output_root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _normalise_config(config)
    if not cfg.get("enabled"):
        return {"pipeline": "soma_bouton_pipeline", "disabled": True}
    blank_rows: list[dict[str, Any]] = []
    sleep_segments: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for ctx in contexts:
        traces = {
            ("soma", "raw"): (np.asarray(ctx.soma.matrix(), dtype=float), np.asarray(ctx.soma.t, dtype=float)),
            ("bouton", "raw"): (np.asarray(ctx.bouton.matrix(), dtype=float), np.asarray(ctx.bouton.t, dtype=float)),
        }
        if ctx.mode == "movie":
            trial_rows = list(ctx.state_bundle.get("rows", [])) if isinstance(ctx.state_bundle, Mapping) else []
            for (compartment, signal_type), (matrix, time) in traces.items():
                blank_rows.extend(_blank_rows_for_traces(pipeline="soma_bouton_pipeline", expid=ctx.expid, animal_id=ctx.animal_id, date=ctx.date, time=time, trial_rows=trial_rows, traces={(compartment, signal_type): matrix}, bin_s=float(cfg["blank_bin_s"]), max_duration_s=cfg["max_blank_duration_s"]))
        elif ctx.mode == "sleep":
            for (compartment, signal_type), (matrix, time) in traces.items():
                key = (compartment, signal_type, str(ctx.animal_id or ""), str(ctx.date or ""))
                sleep_segments[key].append({"expid": str(ctx.expid), "time": time, "matrix": matrix})
    sleep_rows = _sleep_rows_from_segments(pipeline="soma_bouton_pipeline", segments=sleep_segments, time_step_s=float(cfg["sleep_time_step_s"]), max_duration_s=cfg["max_sleep_duration_s"])
    return _finalize(Path(output_root), blank_rows, sleep_rows, cfg, "soma_bouton_pipeline")


def run_dendrite_spine_progression(source_cache: Mapping[str, Any], output_root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _normalise_config(config)
    if not cfg.get("enabled"):
        return {"pipeline": "dendrites_pipeline", "disabled": True}
    raw = source_cache.get("animals", {}) if isinstance(source_cache, Mapping) else {}
    experiments = source_cache.get("experiments", {}) if isinstance(source_cache, Mapping) else {}
    config_data = source_cache.get("config", {}) if isinstance(source_cache, Mapping) else {}
    movie_expids = {str(value) for value in (config_data.get("movie_expids", []) if isinstance(config_data, Mapping) else [])}
    sleep_expids = {str(value) for value in (config_data.get("sleep_expids", []) if isinstance(config_data, Mapping) else [])}
    blank_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    times: dict[str, np.ndarray] = {}
    animals_by_expid: dict[str, str] = {}
    dates_by_expid: dict[str, str] = {}
    for animal_id, animal in raw.items() if isinstance(raw, Mapping) else []:
        for dendrite in animal.get("dendrites", {}).values() if isinstance(animal, Mapping) else []:
            for expid, obs in dendrite.get("observations", {}).items():
                expid = str(expid)
                if expid not in movie_expids and expid not in sleep_expids:
                    continue
                exp_meta = experiments.get(expid, {}) if isinstance(experiments, Mapping) else {}
                time = obs.get("time", exp_meta.get("time", []))
                animals_by_expid.setdefault(expid, str(exp_meta.get("animal_id") or animal_id or ""))
                dates_by_expid.setdefault(expid, str(exp_meta.get("date") or dendrite.get("date") or ""))
                times.setdefault(expid, np.asarray(time, dtype=float))
                grouped[(expid, "dendrite")]["raw"].append(np.asarray(obs.get("trace", []), dtype=float))
                for spine in dendrite.get("spines", {}).values():
                    s_obs = spine.get("observations", {}).get(expid)
                    if not isinstance(s_obs, Mapping):
                        continue
                    for signal, key in (("raw", "trace"), ("spine_specific", "spine_specific")):
                        if signal in cfg["spine_signals"]:
                            grouped[(expid, "spine")][signal].append(np.asarray(s_obs.get(key, []), dtype=float))
    sleep_segments: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for expid in sorted(movie_expids):
        exp_meta = experiments.get(expid, {}) if isinstance(experiments, Mapping) else {}
        trial_meta = exp_meta.get("trial_meta", []) if isinstance(exp_meta, Mapping) else []
        # Keep the original movie metadata: blank trials may be identified by
        # F1_name (the canonical ...\00000 clip) rather than state_label.
        trial_rows = [dict(meta) for meta in trial_meta if isinstance(meta, Mapping)]
        for (group_expid, compartment), signals in grouped.items():
            if group_expid != expid or expid not in times:
                continue
            for signal, matrices in signals.items():
                valid = [matrix for matrix in matrices if matrix.size]
                if valid:
                    blank_rows.extend(_blank_rows_for_traces(pipeline="dendrites_pipeline", expid=expid, animal_id=animals_by_expid.get(expid, ""), date=dates_by_expid.get(expid, ""), time=times[expid], trial_rows=trial_rows, traces={(compartment, signal): np.vstack(valid)}, bin_s=float(cfg["blank_bin_s"]), max_duration_s=cfg["max_blank_duration_s"]))
    for (expid, compartment), signals in grouped.items():
        if expid not in sleep_expids or expid not in times:
            continue
        animal_id, date = animals_by_expid.get(expid, ""), dates_by_expid.get(expid, "")
        for signal, matrices in signals.items():
            valid = [matrix for matrix in matrices if matrix.size]
            if valid:
                sleep_segments[(compartment, signal, animal_id, date)].append({"expid": expid, "time": times[expid], "matrix": np.vstack(valid)})
    sleep_rows = _sleep_rows_from_segments(pipeline="dendrites_pipeline", segments=sleep_segments, time_step_s=float(cfg["sleep_time_step_s"]), max_duration_s=cfg["max_sleep_duration_s"])
    return _finalize(Path(output_root), blank_rows, sleep_rows, cfg, "dendrites_pipeline")


__all__ = ["DEFAULT_PROGRESSION_CONFIG", "run_dendrite_spine_progression", "run_soma_bouton_progression"]
