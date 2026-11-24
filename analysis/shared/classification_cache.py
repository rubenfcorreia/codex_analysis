"""Helpers for applying staged ROI classifications to family-analysis rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.shared.cache_utils import load_npz_cache


def load_classification_records(path: Any) -> list[dict[str, Any]]:
    if not path:
        return []
    cache_path = Path(str(path))
    if not cache_path.exists():
        return []
    payload = load_npz_cache(cache_path)
    records = payload.get("records", [])
    return [dict(row) for row in records if isinstance(row, Mapping)] if isinstance(records, list) else []


def _keys(row: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get(name)) for name in (
            "roi_key", "unit_id", "global_roi_id", "global_dendrite_id",
            "global_spine_id", "global_soma_id", "global_bouton_id",
        ) if row.get(name) not in (None, "")
    }


def classification_lookup(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        for key in _keys(record):
            lookup[key] = dict(record)
    return lookup


def apply_cached_visual_classification(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = classification_lookup(records)
    output = []
    for row in rows:
        updated = dict(row)
        match = next((lookup[key] for key in _keys(row) if key in lookup), None)
        if match is not None:
            visual_class = str(match.get("visual_class", "")).lower()
            cohort = "responsive" if visual_class == "visual" else "nonresponsive" if visual_class == "non_visual" else "unknown"
            updated["responsive"] = cohort == "responsive"
            updated["cohort"] = cohort
            updated["visual_response_cohort"] = cohort
            updated["classification_source"] = "classification_cache"
        output.append(updated)
    return output
