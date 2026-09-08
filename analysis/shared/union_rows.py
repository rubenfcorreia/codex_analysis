from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from analysis.shared.cache_utils import (
    analysis_cache_meta_hash,
    load_npz_cache,
    save_npz_cache,
)


UNION_ROWS_CACHE_SCHEMA_VERSION = 1
ROW_FAMILIES = (
    "activity_rows",
    "correlation_rows",
    "soma_pairwise_rows",
    "bouton_pairwise_rows",
    "lag_rows",
    "coincidence_rows",
    "visual_response_rows",
    "dendrites_analysis_table_rows",
)


def union_state_labels(preset_states: Sequence[Mapping[str, Sequence[str]]]) -> Dict[str, list[str]]:
    result: Dict[str, set[str]] = {}
    for states_by_mode in preset_states:
        for mode, states in states_by_mode.items():
            result.setdefault(str(mode), set()).update(str(state) for state in states if str(state))
    return {mode: sorted(states) for mode, states in sorted(result.items())}


def filter_rows_by_states(
    rows: Sequence[Mapping[str, Any]],
    selected_states_by_mode: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    selected = {str(mode): {str(state) for state in states} for mode, states in selected_states_by_mode.items()}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        state = row.get("state")
        if state is None or str(state) == "":
            filtered.append(dict(row))
            continue
        mode = str(row.get("mode") or "")
        allowed = selected.get(mode)
        if allowed is None:
            filtered.append(dict(row))
        elif str(state) in allowed:
            filtered.append(dict(row))
    return filtered


def deduplicate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate rows deterministically without changing their contents."""
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = analysis_cache_meta_hash(dict(row))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def filter_table_rows_by_states(
    tables: Mapping[str, Any],
    selected_states_by_mode: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Return a lossless filtered copy of dendrites analysis-table entries."""
    selected = {str(s) for values in selected_states_by_mode.values() for s in values}
    filtered_tables: Dict[str, Any] = {}
    for table_name, entry in tables.items():
        if not isinstance(entry, Mapping):
            filtered_tables[str(table_name)] = entry
            continue
        copied = dict(entry)
        rows = entry.get("table_rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            kept = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                state = row.get("state")
                if state is None or str(state) == "":
                    kept.append(dict(row))
                    continue
                mode = str(row.get("mode") or "")
                allowed = set(str(s) for s in selected_states_by_mode.get(mode, ())) if mode else selected
                if str(state) in allowed:
                    kept.append(dict(row))
            copied["table_rows"] = deduplicate_rows(kept)
        filtered_tables[str(table_name)] = copied
    return filtered_tables


def union_rows_meta(
    *,
    source_signature: str,
    union_states_by_mode: Mapping[str, Sequence[str]],
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "cache_scope": "source_rows_union",
        "schema_version": UNION_ROWS_CACHE_SCHEMA_VERSION,
        "source_signature": str(source_signature),
        "union_states_by_mode": {str(mode): sorted(str(state) for state in states) for mode, states in union_states_by_mode.items()},
        "parameters": dict(parameters),
    }


def load_union_rows_cache(
    path: Path,
    *,
    expected_meta: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str, float]:
    started = time.perf_counter()
    if not path.exists():
        return None, "missing", time.perf_counter() - started
    try:
        payload = load_npz_cache(path)
        if payload.get("schema_version") != UNION_ROWS_CACHE_SCHEMA_VERSION:
            return None, "schema_mismatch", time.perf_counter() - started
        saved_meta = payload.get("meta")
        if not isinstance(saved_meta, dict) or analysis_cache_meta_hash(saved_meta) != analysis_cache_meta_hash(dict(expected_meta)):
            return None, "invalid", time.perf_counter() - started
        rows = payload.get("rows")
        if not isinstance(rows, dict):
            return None, "invalid", time.perf_counter() - started
        return rows, "reused", time.perf_counter() - started
    except Exception:
        return None, "invalid", time.perf_counter() - started


def save_union_rows_cache(path: Path, rows: Mapping[str, Any], *, meta: Mapping[str, Any]) -> float:
    started = time.perf_counter()
    save_npz_cache(path, {
        "schema_version": UNION_ROWS_CACHE_SCHEMA_VERSION,
        "cache_scope": "source_rows_union",
        "meta": dict(meta),
        "meta_hash": analysis_cache_meta_hash(dict(meta)),
        "rows": {
            family: (
                {str(name): list(values) for name, values in rows.get(family, {}).items()}
                if family == "dendrites_analysis_table_rows" and isinstance(rows.get(family, {}), Mapping)
                else list(rows.get(family, []))
            )
            for family in ROW_FAMILIES
        },
    })
    return time.perf_counter() - started
