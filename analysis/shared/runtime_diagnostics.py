"""Low-overhead runtime diagnostics shared by analysis entrypoints."""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any, Dict


def _file_stats(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                count += 1
                try:
                    size += path.stat().st_size
                except OSError:
                    pass
    return count, size


def snapshot(root: Path) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        open_figures = len(plt.get_fignums())
    except Exception:
        open_figures = None
    files, bytes_written = _file_stats(Path(root))
    return {
        "timestamp": time.time(),
        "open_figures": open_figures,
        "file_count": files,
        "output_bytes": bytes_written,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def finish(start: Dict[str, Any], root: Path) -> Dict[str, Any]:
    end = snapshot(Path(root))
    return {
        "elapsed_s": float(end["timestamp"] - start["timestamp"]),
        "start": start,
        "end": end,
        "figure_leak_count": (end["open_figures"] - start["open_figures"])
        if start.get("open_figures") is not None and end.get("open_figures") is not None else None,
        "output_file_count_delta": int(end["file_count"] - start["file_count"]),
        "output_bytes_delta": int(end["output_bytes"] - start["output_bytes"]),
    }



def _cache_payload_row_counts(path: Path) -> Dict[str, int]:
    """Extract row-like list lengths without materializing CSV exports."""
    actual = path if path.exists() else path.with_suffix(".chunks")
    if not actual.exists():
        return {}
    if actual.is_dir():
        manifest_path = actual / "manifest.json"
        try:
            payload = __import__("json").loads(manifest_path.read_text())
        except (OSError, TypeError, ValueError):
            return {}
        return {"experiments": len(payload.get("experiment_files", []))} if isinstance(payload, dict) else {}
    try:
        import numpy as np
        loaded = np.load(actual, allow_pickle=True)
        value = loaded["cache"]
        payload = value.item() if hasattr(value, "item") else value
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                label = str(key)
                lowered = label.lower()
                if isinstance(child, list) and ("row" in lowered or lowered.endswith("events") or lowered.endswith("comparisons")):
                    counts[f"{prefix}{label}"] = len(child)
                visit(child, f"{prefix}{label}.")
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            counts[prefix.rstrip(".")] = len(value)
    visit(payload)
    return counts


def summarize_stage_groups(stage_timings: Any) -> Dict[str, Dict[str, Any]]:
    """Aggregate nested stage records into stable diagnostic families.

    This is intentionally diagnostic-only: the original stage records remain
    unchanged and no analysis output is derived from these labels.
    """
    if not isinstance(stage_timings, list):
        return {}
    groups = {
        "state_summaries_comparisons": ("state", "summary", "comparison", "basal/apical"),
        "correlations": ("correlation", "pairwise"),
        "matrix_similarity": ("matrix similarity", "matrix_similarity"),
        "mixed_models": ("mixed model", "mixed_model"),
        "spine_coactivity": ("spine coactivity", "coactivity"),
        "direct_trial_type": ("trial-type", "trial type", "trial_type"),
        "plot_preparation": ("figure", "plot", "poster", "heatmap", "plotting"),
        "cache_serialization": ("cache", "serialization", "serialize"),
    }
    result: Dict[str, Dict[str, Any]] = {}
    for stage in stage_timings:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name") or "").lower()
        task = str(stage.get("task") or "").lower()
        text = f"{name} {task}"
        elapsed = float(stage.get("elapsed_s", 0.0) or 0.0)
        matched = [group for group, tokens in groups.items() if any(token in text for token in tokens)]
        for group in matched:
            summary = result.setdefault(group, {"elapsed_s": 0.0, "stage_count": 0, "failed_count": 0, "stages": []})
            summary["elapsed_s"] += elapsed
            summary["stage_count"] += 1
            if str(stage.get("status") or "") == "failed":
                summary["failed_count"] += 1
            summary["stages"].append(str(stage.get("name") or ""))
    for summary in result.values():
        summary["stages"] = list(dict.fromkeys(summary["stages"]))
    return result

def annotate_cache_records(
    cache_summary: Dict[str, Any],
    *,
    stage_timings: Any = None,
    row_counts: Any = None,
    timings: Any = None,
) -> list[Dict[str, Any]]:
    """Attach measured timing, size, and row metadata to cache records.

    ``cache_summary`` has historically had two shapes, so this returns a
    stable list while leaving the legacy summary untouched.
    """
    stages = stage_timings if isinstance(stage_timings, list) else []
    stage_by_name = {
        str(stage.get("name", "")).lower(): float(stage.get("elapsed_s", 0.0) or 0.0)
        for stage in stages if isinstance(stage, dict)
    }
    rows = row_counts if isinstance(row_counts, dict) else {}
    measured = timings if isinstance(timings, dict) else {}
    records: list[Dict[str, Any]] = []
    entries = cache_summary.get("cache_records") if isinstance(cache_summary, dict) else None
    if not isinstance(entries, list):
        entries = []
        if isinstance(cache_summary, dict):
            for name, value in cache_summary.items():
                if not isinstance(value, dict):
                    continue
                path = value.get("path") or value.get("cache_path")
                entries.append({"name": name, "path": path, **value})
            flat_paths = {
                str(key)[:-len("_cache_path")]: value
                for key, value in (cache_summary.items() if isinstance(cache_summary, dict) else [])
                if str(key).endswith("_cache_path")
            }
            for name, path in flat_paths.items():
                entries.append({
                    "name": name,
                    "path": path,
                    "status": cache_summary.get(name + "_cache_status"),
                    "key": cache_summary.get(name + "_cache_key"),
                })
    seen: set[str] = set()
    stage_map = {
        "source": "cache load or rebuild", "analysis_day": "day-level cache construction",
        "analysis_tables": "analysis-table cache load", "shared_shuffle": "shared circular-shift cache",
        "analysis_results": "analysis families", "analysis_run": "pipeline run",
        "pairwise_family": "analysis families", "coincidence_family": "analysis families",
    }
    for value in entries:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or "cache")
        if name in seen:
            continue
        seen.add(name)
        stage_name = stage_map.get(name)
        elapsed = float(measured.get(name, stage_by_name.get(stage_name, 0.0) if stage_name else 0.0) or 0.0)
        status = str(value.get("status") or "unknown")
        record = dict(value)
        record["name"] = name
        record["timing_stage"] = stage_name
        record["load_elapsed_s"] = elapsed if status in {"ok", "reused"} else 0.0
        record["build_elapsed_s"] = elapsed if status in {"built", "rebuilt", "rebuild_requested", "missing", "meta_mismatch", "schema_mismatch", "unreadable"} else 0.0
        if name in rows:
            record["row_counts"] = rows[name]
        elif rows:
            record["row_counts"] = rows
        path_value = record.get("path")
        if path_value:
            path = Path(str(path_value))
            chunk_path = path.with_suffix(".chunks")
            actual = path if path.exists() else (chunk_path if chunk_path.exists() else None)
            record["exists"] = actual is not None
            if actual is not None:
                record["file_size_bytes"] = sum(
                    int(child.stat().st_size) for child in actual.rglob("*") if child.is_file()
                ) if actual.is_dir() else int(actual.stat().st_size)
                payload_counts = _cache_payload_row_counts(actual)
                if payload_counts:
                    record["row_counts"] = payload_counts
        if status in {"rebuild_requested", "meta_mismatch", "schema_mismatch", "unreadable", "missing"}:
            record["invalidation_reason"] = status
        records.append(record)
    return records
