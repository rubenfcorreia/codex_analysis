"""Build ROI classes from the dedicated initial-analysis cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.cache_builders.common import cache_manifest_path, dependency_signature, load_index_manifest, load_indexed_records, record_scope_key, save_indexed_records, save_json_manifest, stage_is_reusable, stage_manifest
from analysis.shared.cache_config import normalize_staged_config
from analysis.shared.cache_utils import load_npz_cache, save_npz_cache


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def classification_thresholds(records: list[Mapping[str, Any]]) -> dict[str, float | None]:
    return {"activity": _median([float(row["activity_mean"]) for row in records if row.get("activity_mean") is not None]), "frequency": _median([float(row["frequency_mean"]) for row in records if row.get("frequency_mean") is not None])}


def classify_records(records: list[Mapping[str, Any]], thresholds: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    thresholds = dict(thresholds or classification_thresholds(records))
    activity_cut = thresholds.get("activity")
    frequency_cut = thresholds.get("frequency")
    classified = []
    for row in records:
        activity = row.get("activity_mean")
        frequency = row.get("frequency_mean")
        activity_class = "high_activity" if activity_cut is not None and activity is not None and float(activity) >= activity_cut else "low_activity"
        frequency_class = "high_frequency" if frequency_cut is not None and frequency is not None and float(frequency) >= frequency_cut else "low_frequency"
        visual_value = row.get("visual_responsive")
        visual_label = str(row.get("visual_class") or "").strip().lower()
        if visual_value is None and visual_label not in {"visual", "non_visual", "nonresponsive", "responsive"}:
            visual_class = "unknown"
        else:
            visual = bool(visual_value) if visual_value is not None else visual_label in {"visual", "responsive"}
            visual_class = "visual" if visual else "non_visual"
        classified.append({**dict(row), "visual_class": visual_class, "activity_class": activity_class, "frequency_class": frequency_class, "activity_frequency_class": f"{activity_class}|{frequency_class}"})
    return classified


def _group_records(records: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records or []:
        grouped.setdefault(record_scope_key(record), []).append(dict(record))
    return grouped


def build_classification_cache(config_path: Path, pipeline: str, initial_analysis_path: Path, output_path: Path) -> dict[str, Any]:
    with config_path.open() as handle:
        config = normalize_staged_config(json.load(handle), pipeline=pipeline, base_dir=config_path.parent)
    initial = load_npz_cache(initial_analysis_path)
    initial_signature = dependency_signature(initial)
    config_signature = dependency_signature(config.get("classification", {}))
    manifest_path = cache_manifest_path({**config, "cache_paths": {"classification": output_path.parent}}, "classification")
    if not bool(config.get("rebuild") or config.get("classification_cache_rebuild")):
        reusable = stage_is_reusable(manifest_path, output_path, pipeline=pipeline, stage="classification", config_signature=config_signature, upstream_signature=initial_signature)
        if reusable is not None:
            return reusable
    records = initial.get("records")
    if not isinstance(records, list):
        raise ValueError("Initial-analysis cache must contain 'records'")
    clean_records = [row for row in records if isinstance(row, Mapping)]
    existing = load_npz_cache(output_path) if output_path.exists() else {}
    thresholds = existing.get("thresholds") if isinstance(existing, dict) and isinstance(existing.get("thresholds"), dict) else classification_thresholds(clean_records)
    grouped = _group_records(clean_records)
    previous_index = load_index_manifest(output_path.parent, "classification")
    previous_artifacts = dict((previous_index.get("records") or {}).get("artifacts") or {})
    rebuild_ids = {str(value) for value in (config.get("rebuild_expids") or [])}
    reusable_records = []
    changed_records = []
    for scope_key, scope_rows in grouped.items():
        upstream = dependency_signature(scope_rows)
        artifact = previous_artifacts.get(scope_key) or {}
        if str(artifact.get("config_signature")) == str(config_signature) and str(artifact.get("upstream_signature")) == str(upstream) and scope_key.split("|", 1)[0] not in rebuild_ids:
            cached = load_indexed_records(output_path.parent, scope_key)
            if cached is not None:
                reusable_records.extend(cached)
                continue
        changed_records.extend(scope_rows)
    classified = reusable_records + classify_records(changed_records, thresholds)
    payload = {"schema_version": 1, "pipeline": pipeline, "records": classified, "thresholds": thresholds, "initial_analysis_signature": dependency_signature(initial), "config_signature": dependency_signature(config.get("classification", {}))}
    save_npz_cache(output_path, payload)
    indexed = save_indexed_records(
        output_path.parent,
        pipeline=pipeline,
        stage="classification",
        records_by_scope=_group_records(payload.get("roi_records", payload.get("records", []))),
        config_signature=payload["config_signature"],
        upstream_signatures={key: dependency_signature(rows) for key, rows in grouped.items()},
        rebuild_expids=config.get("rebuild_expids", []),
    )
    manifest = stage_manifest(pipeline=pipeline, stage="classification", config_signature=payload["config_signature"], upstream_signature=payload["initial_analysis_signature"], active_expids=sorted({str(row.get("expid")) for row in classified if row.get("expid")}), records={"classification_cache_path": str(output_path), "roi_count": len(classified), "indexed_artifacts": indexed.get("records", {})})
    save_json_manifest(cache_manifest_path({**config, "cache_paths": {"classification": output_path.parent}}, "classification"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_classification_cache(args.config, args.pipeline, args.initial_analysis, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
