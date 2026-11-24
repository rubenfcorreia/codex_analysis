"""Apply staged classifications to summary payloads used by analysis families."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from analysis.shared.classification_cache import apply_cached_visual_classification


def apply_summary(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = dict(summary)
    rows = apply_cached_visual_classification(summary.get("rows", []), records)
    output["rows"] = rows
    cohort_ids: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        compartment = str(row.get("compartment") or "unknown")
        cohort = str(row.get("cohort") or "nonresponsive")
        identifier = row.get("global_dendrite_id") or row.get("global_spine_id") or row.get("roi_key")
        cohort_ids.setdefault(compartment, {}).setdefault(cohort, []).append(str(identifier))
        counts.setdefault(compartment, {}).setdefault(cohort, 0)
        counts[compartment][cohort] += 1
    output["cohort_ids"] = cohort_ids
    output["counts"] = counts
    output["classification_source"] = "classification_cache"
    return output
