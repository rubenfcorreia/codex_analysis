"""Reusable ROI/day/mode aggregation for initial analysis."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Mapping
from analysis.cache_builders.schema import CACHE_SCHEMA_VERSION
def numeric(value: Any) -> float | None:
    try: result = float(value)
    except (TypeError, ValueError): return None
    return result if result == result else None
def _group_key(record: Mapping[str, Any], mode: str | None = None) -> tuple[str, str, str]:
    return (str(record.get("roi_key") or record.get("unit_id") or record.get("global_roi_id")), str(record.get("day_id") or record.get("expid") or "unknown"), str(mode if mode is not None else record.get("experiment_mode") or "unknown"))
def _aggregate_rows(rows, mode, scope):
    state_values: dict[str, list[float]] = defaultdict(list); freq_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for state, value in (row.get("state_activity") or {}).items():
            value = numeric(value)
            if value is not None: state_values[str(state)].append(value)
        for state, value in (row.get("state_frequency") or {}).items():
            value = numeric(value)
            if value is not None: freq_values[str(state)].append(value)
    first = rows[0]
    return {"roi_key": str(first.get("roi_key")), "day_id": str(first.get("day_id") or first.get("expid") or "unknown"), "experiment_mode": mode, "scope": scope, "expid": first.get("expid"), "animal_id": first.get("animal_id"), "roi_type": first.get("roi_type"), "compartment": first.get("compartment"), "state_activity": {k: sum(v)/len(v) for k,v in sorted(state_values.items()) if v}, "state_frequency": {k: sum(v)/len(v) for k,v in sorted(freq_values.items()) if v}, "visual_responsive": first.get("visual_responsive"), "visual_class": first.get("visual_class"), "source_expids": sorted({str(row.get("expid")) for row in rows if row.get("expid")}), "observation_count": len(rows), "modes_present": [mode], "schema_version": CACHE_SCHEMA_VERSION}
def aggregate_roi_day_mode(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str,str,str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records: groups[_group_key(record)].append(record)
    return [_aggregate_rows(rows, mode, "roi_day_mode") for (_, _, mode), rows in sorted(groups.items())]
def aggregate_roi_day_combined(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str,str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records: groups[(str(record.get("roi_key")), str(record.get("day_id") or record.get("expid") or "unknown"))].append(record)
    output = []
    for (_, _), rows in sorted(groups.items()):
        row = _aggregate_rows(rows, "combined", "roi_day_combined")
        row["source_expids"] = sorted({str(x) for item in rows for x in (item.get("source_expids") or [item.get("expid")]) if x})
        row["observation_count"] = sum(int(item.get("observation_count") or 1) for item in rows)
        row["modes_present"] = sorted({str(item.get("experiment_mode") or "unknown") for item in rows})
        output.append(row)
    return output
def build_preferred_state_views(records: list[Mapping[str, Any]], summarize) -> dict[str, list[dict[str, Any]]]:
    mode_rows = aggregate_roi_day_mode(records); combined_rows = aggregate_roi_day_combined(mode_rows)
    views = {"combined": combined_rows, "movie": [r for r in mode_rows if r["experiment_mode"] == "movie"], "sleep": [r for r in mode_rows if r["experiment_mode"] == "sleep"]}
    return {name: [summarize(row) for row in rows] for name, rows in views.items()}
