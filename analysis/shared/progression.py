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
from analysis.shared.shared_calcium_response import load_visual_response_cut_data


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



def _blank_rows_for_cut_traces(*, pipeline: str, expid: str, animal_id: Any, date: Any, cut_time: Any, trial_meta: Sequence[Mapping[str, Any]], traces: Mapping[tuple[str, str], Any], bin_s: float, max_duration_s: float | None) -> list[dict[str, Any]]:
    blank_trials = [meta for meta in trial_meta if isinstance(meta, Mapping) and _trial_label(meta) == "blank"]
    axis = np.asarray(cut_time, dtype=float).reshape(-1)
    if not blank_trials or axis.size == 0:
        return []
    durations = [_as_float(meta.get("duration")) for meta in blank_trials]
    durations = [value for value in durations if value is not None and value > 0]
    if not durations:
        positive_time = axis[np.isfinite(axis) & (axis >= 0)]
        durations = [float(np.max(positive_time)) if positive_time.size else 0.0]
    max_duration = max(durations, default=0.0)
    if max_duration_s is not None:
        max_duration = min(max_duration, max_duration_s)
    n_bins = int(math.ceil(max_duration / bin_s)) if max_duration > 0 else 0
    rows: list[dict[str, Any]] = []
    for (compartment, signal_type), matrix in sorted(traces.items()):
        values_matrix = np.asarray(matrix, dtype=float)
        if values_matrix.ndim != 3:
            continue
        for bin_index in range(n_bins):
            bin_start = bin_index * bin_s
            bin_end = min((bin_index + 1) * bin_s, max_duration)
            values: list[float] = []
            total_frames = 0
            for meta in blank_trials:
                try:
                    trial_index = int(meta.get("trial_index"))
                except (TypeError, ValueError):
                    continue
                if trial_index < 0 or trial_index >= values_matrix.shape[1]:
                    continue
                trial_duration = _as_float(meta.get("duration")) or max_duration
                local_end = min(trial_duration, bin_end)
                if local_end <= bin_start:
                    continue
                mask = np.isfinite(axis) & (axis >= bin_start) & (axis < local_end)
                selected = values_matrix[:, trial_index, :][:, :axis.size][:, mask]
                finite = selected[np.isfinite(selected)]
                if finite.size:
                    values.append(_finite_mean(finite))
                    total_frames += int(finite.size)
            if not values:
                continue
            row = _base_row(pipeline=pipeline, compartment=compartment, signal_type=signal_type, expid=expid, animal_id=animal_id, date=date)
            row.update({"day_id": f"{animal_id}_{date}", "source_expids": json.dumps([str(expid)]), "duration_policy": "common_overlap", "time_bin_s": float(bin_start), "time_bin_end_s": float(bin_end), "mean_dff": _finite_mean(values), "n_rois": int(values_matrix.shape[0]), "n_trials": int(len(values)), "n_frames": int(total_frames), "status": "ok"})
            rows.append(row)
    return rows

def _expid_sort_key(expid: Any) -> tuple[str, int, str]:
    text = str(expid)
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d+)", text)
    return (match.group(1), int(match.group(2)), text) if match else (text, 0, text)


def _sleep_rows_from_segments(*, pipeline: str, segments: Mapping[tuple[str, str, str, str], Sequence[Mapping[str, Any]]], time_step_s: float, max_duration_s: float | None, duration_policy: str = "common_overlap") -> list[dict[str, Any]]:
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
        durations = [value["duration"] for _, value in days]
        duration = min(durations) if duration_policy == "common_overlap" else max(durations)
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
            # Preserve each source experiment instead of assigning the whole
            # concatenated day to its first expID.
            for boundary_index, boundary in enumerate(value["boundaries"]):
                start, end = float(boundary["start_s"]), float(boundary["end_s"])
                # Segment boundaries are shared samples; assign each boundary
                # to the later segment only, except for the final endpoint.
                keep_segment = (grid >= start) & ((grid <= end) if boundary_index == len(value["boundaries"]) - 1 else (grid < end))
                if not np.any(keep_segment):
                    continue
                segment_values = np.interp(grid[keep_segment], unique, y, left=np.nan, right=np.nan)
                for elapsed_time, mean in zip(grid[keep_segment], segment_values):
                    if not np.isfinite(mean):
                        continue
                    expid = str(boundary.get("expid", ""))
                    row = _base_row(pipeline=pipeline, compartment=compartment, signal_type=signal_type, expid=expid, animal_id=animal_id, date=date)
                    row.update({"day_id": f"{animal_id}_{date}", "source_expids": json.dumps(value["source_expids"]), "segment_boundaries": json.dumps(value["boundaries"]), "duration_policy": duration_policy, "elapsed_time_s": float(elapsed_time), "mean_dff": float(mean), "n_rois": int(boundary.get("n_rois", 0)), "n_trials": 0, "n_frames": int(np.sum(np.abs(axis - elapsed_time) <= time_step_s / 2)), "status": "ok"})
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
            row.update({"day_id": f"{animal_id}_{date}", "source_expids": json.dumps([str(expid)]), "duration_policy": "common_overlap", "time_bin_s": float(bin_start), "time_bin_end_s": float(bin_end), "mean_dff": _finite_mean(values), "n_rois": int(n_rois), "n_trials": int(len(values)), "n_frames": int(total_frames), "status": "ok"})
            rows.append(row)
    return rows


def _weighted_stats(values, weights=None):
    values=np.asarray(values,dtype=float)
    if weights is None:
        values=values[np.isfinite(values)]
        return _finite_mean(values), _finite_sem(values), int(values.size)
    weights=np.asarray(weights,dtype=float)
    good=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    values,weights=values[good],weights[good]
    if not values.size: return float("nan"),float("nan"),0
    mean=float(np.average(values,weights=weights))
    sem=float(np.sqrt(np.average((values-mean)**2,weights=weights)/values.size)) if values.size>1 else 0.0
    return mean,sem,int(values.size)


def _summary_points(rows, time_key, weighting):
    by_day=defaultdict(list)
    for row in rows:
        x=_as_float(row.get(time_key)); y=_as_float(row.get("mean_dff"))
        if x is None or y is None: continue
        day=str(row.get("day_id") or f"{row.get('animal_id','')}_{row.get('date','')}")
        by_day[(day,x)].append((y,float(row.get("n_rois",1) or 1)))
    by_x=defaultdict(list)
    for (_day,x), vals in by_day.items():
        y,_,_=_weighted_stats([v for v,_ in vals],[w for _,w in vals] if weighting=="roi_weighted" else None)
        by_x[x].append((y,sum(w for _,w in vals) if weighting=="roi_weighted" else 1.0))
    xs=sorted(by_x); ys=[]; es=[]
    for x in xs:
        y,e,_=_weighted_stats([v for v,_ in by_x[x]],[w for _,w in by_x[x]] if weighting=="roi_weighted" else None)
        ys.append(y); es.append(e)
    return xs,ys,es,len({day for day,_ in by_day})


def _add_summary_rows(blank_rows, sleep_rows, weighting="equal_day"):
    result=[]
    for analysis,rows,key in (("blank_trial",blank_rows,"time_bin_s"),("sleep_progression",sleep_rows,"elapsed_time_s")):
        if not rows: continue
        xs,ys,es,n_days=_summary_points(rows,key,weighting)
        for x,y,e in zip(xs,ys,es):
            result.append({"analysis":analysis,"compartment":str(rows[0].get("compartment","")),"signal_type":str(rows[0].get("signal_type","")),"time_bin_s":x,"duration_policy":str(rows[0].get("duration_policy","common_overlap")),"weighting":weighting,"mean_dff":y,"sem_dff":e,"n_days":n_days})
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


def _plot_outputs(root, blank_rows, sleep_rows_by_policy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths=[]
    all_sleep=[r for rows in sleep_rows_by_policy.values() for r in rows]
    compartments=sorted({str(r.get("compartment")) for r in [*blank_rows,*all_sleep]})
    for compartment in compartments:
        figroot=root/compartment/"figures"; figroot.mkdir(parents=True,exist_ok=True)
        signals=sorted({str(r.get("signal_type")) for r in [*blank_rows,*all_sleep] if str(r.get("compartment"))==compartment})
        specs=[(signals,f"{compartment}_progression")]
        if compartment=="spine": specs += [([s],f"spine_{s}_progression") for s in signals]
        for fs,base in specs:
            for policy in ("common_overlap","all_available"):
                sleep=sleep_rows_by_policy.get(policy,[])
                for weighting in ("equal_day","roi_weighted"):
                    suffix="" if (policy,weighting)==("common_overlap","equal_day") else f"_{policy}" if weighting=="equal_day" else f"_{weighting}" if policy=="common_overlap" else f"_{policy}_{weighting}"
                    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
                    for signal in fs:
                        b=[r for r in blank_rows if r.get("compartment")==compartment and r.get("signal_type")==signal]
                        s=[r for r in sleep if r.get("compartment")==compartment and r.get("signal_type")==signal]
                        for rows,key,ax,scale in ((b,"time_bin_s",axes[0],1.0),(s,"elapsed_time_s",axes[1],1/60.0)):
                            byday=defaultdict(list)
                            for r in rows: byday[str(r.get("day_id") or f"{r.get('animal_id','')}_{r.get('date','')}")].append(r)
                            for dayrows in byday.values():
                                dayrows=sorted(dayrows,key=lambda r:float(r.get(key,0)))
                                ax.plot([float(r[key])*scale for r in dayrows],[float(r["mean_dff"]) for r in dayrows],color="0.5",alpha=.28,lw=.8)
                        if b:
                            x,y,e,n=_summary_points(b,"time_bin_s",weighting)
                            axes[0].plot(x,y,marker="o",label=f"{signal} mean (n={n} days)")
                            axes[0].fill_between(x,np.asarray(y)-np.asarray(e),np.asarray(y)+np.asarray(e),alpha=.15)
                        if s:
                            x,y,e,n=_summary_points(s,"elapsed_time_s",weighting); x=np.asarray(x)/60
                            axes[1].plot(x,y,color="#111111",lw=2.8,label=f"{signal} mean (n={n} days)")
                            axes[1].fill_between(x,np.asarray(y)-np.asarray(e),np.asarray(y)+np.asarray(e),alpha=.15)
                    axes[0].set(title=f"Blank-trial progression ({weighting})",xlabel="Time from blank onset (s)",ylabel="Mean dF/F")
                    axes[1].set(title=f"Sleep progression ({policy}, {weighting})",xlabel="Elapsed sleep time (min)",ylabel="Mean dF/F")
                    for ax in axes:
                        ax.grid(alpha=.2); h,l=ax.get_legend_handles_labels()
                        if h: ax.legend(h,l,frameon=False)
                    out=figroot/f"{base}{suffix}.svg"
                    try:
                        fig.savefig(out,format="svg")
                    finally:
                        plt.close(fig)
                    paths.append(str(out))
    return paths


def _finalize(root, blank_rows, sleep_rows_by_policy, config, pipeline, figure_root=None):
    root.mkdir(parents=True,exist_ok=True)
    figure_root = Path(figure_root) if figure_root is not None else root
    figure_root.mkdir(parents=True, exist_ok=True)
    common=sleep_rows_by_policy.get("common_overlap",[])
    compartments=sorted({str(r.get("compartment")) for r in [*blank_rows,*common]})
    for compartment in compartments:
        cr=root/compartment
        _write_rows(cr/"blank_trial_progression.csv",[r for r in blank_rows if r.get("compartment")==compartment])
        _write_rows(cr/"sleep_expid_progression.csv",[r for r in common if r.get("compartment")==compartment])
        for policy,rows in sleep_rows_by_policy.items():
            _write_rows(cr/f"sleep_progression_{policy}.csv",[r for r in rows if r.get("compartment")==compartment])
        for weighting in ("equal_day","roi_weighted"):
            summaries=[]
            for policy,rows in sleep_rows_by_policy.items():
                summaries += _add_summary_rows(blank_rows,rows,weighting)
            _write_rows(cr/f"progression_summary_{weighting}.csv",[r for r in summaries if r.get("compartment")==compartment])
    figures=_plot_outputs(figure_root,blank_rows,sleep_rows_by_policy)
    sleep_rows_by_policy_counts = {k: len(v) for k, v in sleep_rows_by_policy.items()}
    manifest={"pipeline":pipeline,"config":dict(config),"blank_rows":len(blank_rows),"sleep_rows":len(common),"sleep_rows_by_policy":sleep_rows_by_policy_counts,"figures":figures,"duration_policies":sorted(sleep_rows_by_policy),"summary_weightings":["equal_day","roi_weighted"]}
    (root/"progression_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str))
    return manifest


def run_soma_bouton_progression(contexts: Iterable[Any], output_root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _normalise_config(config)
    if not cfg.get("enabled"):
        return {"pipeline": "soma_bouton_pipeline", "disabled": True}
    blank_rows: list[dict[str, Any]] = []
    sleep_segments: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for ctx in contexts:
        if ctx.mode == "movie":
            trial_rows = list(ctx.state_bundle.get("rows", [])) if isinstance(ctx.state_bundle, Mapping) else []
            for compartment, channel in (("soma", ctx.soma_channel), ("bouton", ctx.bouton_channel)):
                cut_data = load_visual_response_cut_data(ctx.exp_root, int(channel), trial_rows)
                if cut_data:
                    blank_rows.extend(_blank_rows_for_cut_traces(
                        pipeline="soma_bouton_pipeline",
                        expid=ctx.expid,
                        animal_id=ctx.animal_id,
                        date=ctx.date,
                        cut_time=cut_data["cut_time"],
                        trial_meta=cut_data.get("trial_meta", []),
                        traces={(compartment, "raw"): cut_data["cut_neural"]},
                        bin_s=float(cfg["blank_bin_s"]),
                        max_duration_s=cfg["max_blank_duration_s"],
                    ))
        elif ctx.mode == "sleep":
            traces = {
                ("soma", "raw"): (np.asarray(ctx.soma.matrix(), dtype=float), np.asarray(ctx.soma.t, dtype=float)),
                ("bouton", "raw"): (np.asarray(ctx.bouton.matrix(), dtype=float), np.asarray(ctx.bouton.t, dtype=float)),
            }
            for (compartment, signal_type), (matrix, time) in traces.items():
                key = (compartment, signal_type, str(ctx.animal_id or ""), str(ctx.date or ""))
                sleep_segments[key].append({"expid": str(ctx.expid), "time": time, "matrix": matrix})
    sleep_rows_by_policy = {policy: _sleep_rows_from_segments(pipeline="soma_bouton_pipeline", segments=sleep_segments, time_step_s=float(cfg["sleep_time_step_s"]), max_duration_s=cfg["max_sleep_duration_s"], duration_policy=policy) for policy in ("common_overlap", "all_available")}
    return _finalize(Path(output_root), blank_rows, sleep_rows_by_policy, cfg, "soma_bouton_pipeline")


def run_dendrite_spine_progression(source_cache: Mapping[str, Any], output_root: Path, config: Mapping[str, Any] | None = None, *, figure_root: Path | None = None) -> dict[str, Any]:
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
    cut_grouped: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    times: dict[str, np.ndarray] = {}
    animals_by_expid: dict[str, str] = {}
    dates_by_expid: dict[str, str] = {}
    cut_data_by_expid: dict[str, Mapping[str, Any] | None] = {}

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

                if expid in movie_expids and expid not in cut_data_by_expid:
                    source_paths = exp_meta.get("source_paths", {}) if isinstance(exp_meta, Mapping) else {}
                    exp_root = Path(str(source_paths.get("exp_root") or ""))
                    trial_rows = exp_meta.get("trial_rows", []) if isinstance(exp_meta, Mapping) else []
                    cut_data_by_expid[expid] = load_visual_response_cut_data(exp_root, int(config_data.get("channel", 0)), trial_rows) if exp_root.exists() else None
                cut_data = cut_data_by_expid.get(expid)
                try:
                    dendrite_index = int((obs.get("local_ids", {}) or {}).get("conversion_index"))
                except (TypeError, ValueError):
                    dendrite_index = None
                if cut_data and dendrite_index is not None and 0 <= dendrite_index < cut_data["cut_neural"].shape[0]:
                    cut_grouped[(expid, "dendrite")]["raw"].append(np.asarray(cut_data["cut_neural"][dendrite_index], dtype=float))

                for spine in dendrite.get("spines", {}).values():
                    s_obs = spine.get("observations", {}).get(expid)
                    if not isinstance(s_obs, Mapping):
                        continue
                    for signal, key in (("raw", "trace"), ("spine_specific", "spine_specific")):
                        if signal in cfg["spine_signals"]:
                            grouped[(expid, "spine")][signal].append(np.asarray(s_obs.get(key, []), dtype=float))
                    if not cut_data:
                        continue
                    try:
                        spine_index = int((s_obs.get("local_ids", {}) or {}).get("conversion_index"))
                    except (TypeError, ValueError):
                        spine_index = None
                    if spine_index is None or not (0 <= spine_index < cut_data["cut_neural"].shape[0]):
                        continue
                    spine_cut = np.asarray(cut_data["cut_neural"][spine_index], dtype=float)
                    cut_grouped[(expid, "spine")]["raw"].append(spine_cut)
                    if "spine_specific" in cfg["spine_signals"] and dendrite_index is not None and 0 <= dendrite_index < cut_data["cut_neural"].shape[0]:
                        alpha = _as_float(s_obs.get("alpha"))
                        if alpha is not None:
                            cut_grouped[(expid, "spine")]["spine_specific"].append(spine_cut - alpha * np.asarray(cut_data["cut_neural"][dendrite_index], dtype=float))

    for expid in sorted(movie_expids):
        cut_data = cut_data_by_expid.get(expid)
        if not cut_data:
            continue
        for (group_expid, compartment), signals in cut_grouped.items():
            if group_expid != expid:
                continue
            for signal, matrices in signals.items():
                valid = [matrix for matrix in matrices if matrix.size]
                if valid:
                    blank_rows.extend(_blank_rows_for_cut_traces(
                        pipeline="dendrites_pipeline",
                        expid=expid,
                        animal_id=animals_by_expid.get(expid, ""),
                        date=dates_by_expid.get(expid, ""),
                        cut_time=cut_data["cut_time"],
                        trial_meta=cut_data.get("trial_meta", []),
                        traces={(compartment, signal): np.stack(valid, axis=0)},
                        bin_s=float(cfg["blank_bin_s"]),
                        max_duration_s=cfg["max_blank_duration_s"],
                    ))

    sleep_segments: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (expid, compartment), signals in grouped.items():
        if expid not in sleep_expids or expid not in times:
            continue
        animal_id, date = animals_by_expid.get(expid, ""), dates_by_expid.get(expid, "")
        for signal, matrices in signals.items():
            valid = [matrix for matrix in matrices if matrix.size]
            if valid:
                sleep_segments[(compartment, signal, animal_id, date)].append({"expid": expid, "time": times[expid], "matrix": np.vstack(valid)})
    sleep_rows_by_policy = {policy: _sleep_rows_from_segments(pipeline="dendrites_pipeline", segments=sleep_segments, time_step_s=float(cfg["sleep_time_step_s"]), max_duration_s=cfg["max_sleep_duration_s"], duration_policy=policy) for policy in ("common_overlap", "all_available")}
    return _finalize(Path(output_root), blank_rows, sleep_rows_by_policy, cfg, "dendrites_pipeline", figure_root=figure_root)


__all__ = ["DEFAULT_PROGRESSION_CONFIG", "run_dendrite_spine_progression", "run_soma_bouton_progression"]
