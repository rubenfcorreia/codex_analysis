"""Build the initial, family-independent ROI analysis cache.

This stage is intentionally upstream of ROI classification and all analysis
families.  It consumes a processed-cache payload containing ``roi_records``.
Each record must identify an ROI and may contain state-wise activity and event
frequency values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.cache_builders.common import cache_manifest_path, dependency_signature, load_index_manifest, load_indexed_records, record_scope_key, save_indexed_records, save_json_manifest, stage_is_reusable, stage_manifest
from analysis.shared.cache_config import normalize_staged_config
from analysis.shared.cache_utils import load_npz_cache, save_npz_cache
from analysis.cache_builders.initial_analysis_plots import write_initial_analysis_figures
from analysis.cache_builders.schema import CACHE_SCHEMA_VERSION
from analysis.cache_builders.aggregation import build_preferred_state_views


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _preferred_state_label(state: Any) -> str:
    """Use one overall NREM label for pupil-defined NREM substates."""
    label = str(state)
    return "nrem" if label.startswith("nrem_pupil_") else label


def _collapse_state_values(values: Mapping[str, float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    explicit: dict[str, list[float]] = {}
    for state, value in values.items():
        label = _preferred_state_label(state)
        grouped.setdefault(label, []).append(float(value))
        if str(state) == label:
            explicit.setdefault(label, []).append(float(value))
    return {label: sum(explicit.get(label, candidates)) / len(explicit.get(label, candidates)) for label, candidates in grouped.items()}


def _summarize_record(record: Mapping[str, Any], abs_tolerance: float = 0.0, rel_tolerance: float = 0.01) -> dict[str, Any]:
    state_activity = record.get("state_activity") or record.get("activity_by_state") or {}
    state_frequency = record.get("state_frequency") or record.get("frequency_by_state") or {}
    if not isinstance(state_activity, Mapping):
        state_activity = {}
    if not isinstance(state_frequency, Mapping):
        state_frequency = {}
    activity = _collapse_state_values({str(k): value for k, v in state_activity.items() if (value := _numeric(v)) is not None})
    frequency = _collapse_state_values({str(k): value for k, v in state_frequency.items() if (value := _numeric(v)) is not None})
    maximum = max(activity.values()) if activity else None
    tolerance = max(float(abs_tolerance), float(rel_tolerance) * abs(float(maximum))) if maximum is not None else 0.0
    preferred_states = sorted(state for state, value in activity.items() if maximum is not None and float(maximum) - float(value) <= tolerance)
    preferred_state = preferred_states[0] if preferred_states else None
    preferred_state_class = "unknown" if not preferred_states else ("tie" if len(preferred_states) > 1 else preferred_states[0]) if activity else None
    activity_values = list(activity.values())
    frequency_values = list(frequency.values())
    return {
        "roi_key": str(record.get("roi_key") or record.get("unit_id") or record.get("global_roi_id")),
        "roi_type": record.get("roi_type"),
        "experiment_mode": record.get("experiment_mode") or "unknown",
        "expid": record.get("expid") or record.get("exp_id"),
        "animal_id": record.get("animal_id"),
        "day_id": record.get("day_id"),
        "scope": str(record.get("scope") or "observation"),
        "compartment": record.get("compartment"),
        "state_activity": activity,
        "state_frequency": frequency,
        "preferred_state": preferred_state,
        "preferred_states": preferred_states,
        "preferred_state_class": preferred_state_class,
        "activity_mean": sum(activity_values) / len(activity_values) if activity_values else None,
        "frequency_mean": sum(frequency_values) / len(frequency_values) if frequency_values else None,
        "visual_responsive": record.get("visual_responsive"),
        "visual_class": record.get("visual_class"),
        "source_signature": record.get("source_signature"),
        "source_expids": list(record.get("source_expids") or ([record.get("expid")] if record.get("expid") else [])),
        "observation_count": int(record.get("observation_count") or 1),
        "modes_present": list(record.get("modes_present") or [record.get("experiment_mode") or "unknown"]),
    }


def _group_records(records: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records or []:
        grouped.setdefault(record_scope_key(record), []).append(dict(record))
    return grouped


def build_initial_analysis(config_path: Path, pipeline: str, processed_cache_path: Path, output_path: Path) -> dict[str, Any]:
    with config_path.open() as handle:
        config = normalize_staged_config(json.load(handle), pipeline=pipeline, base_dir=config_path.parent)
    processed = load_npz_cache(processed_cache_path)
    processed_signature = dependency_signature(processed)
    config_signature = dependency_signature(config.get("processing", {}), config.get("source", {}), config.get("initial_analysis", {}))
    manifest_path = cache_manifest_path({**config, "cache_paths": {"initial_analysis": output_path.parent}}, "initial_analysis")
    if not bool(config.get("rebuild") or config.get("initial_analysis_cache_rebuild")):
        reusable = stage_is_reusable(manifest_path, output_path, pipeline=pipeline, stage="initial_analysis", config_signature=dependency_signature(config_signature), upstream_signature=processed_signature)
        if reusable is not None:
            return reusable
    rows = processed.get("roi_records")
    if not isinstance(rows, list):
        raise ValueError("Processed cache must contain a list named 'roi_records'")
    initial_settings = config.get("initial_analysis", {}) if isinstance(config.get("initial_analysis", {}), Mapping) else {}
    abs_tolerance = float(initial_settings.get("state_tie_absolute_tolerance", 0.0))
    rel_tolerance = float(initial_settings.get("state_tie_relative_tolerance", 0.01))
    grouped = _group_records(rows)
    previous_index = load_index_manifest(output_path.parent, "initial_analysis")
    previous_artifacts = dict((previous_index.get("records") or {}).get("artifacts") or {})
    rebuild_ids = {str(value) for value in (config.get("rebuild_expids") or [])}
    reusable_records = []
    changed_rows = []
    for scope_key, scope_rows in grouped.items():
        upstream = dependency_signature(scope_rows)
        artifact = previous_artifacts.get(scope_key) or {}
        if str(artifact.get("config_signature")) == str(config_signature) and str(artifact.get("upstream_signature")) == str(upstream) and scope_key.split("|", 1)[0] not in rebuild_ids:
            cached = load_indexed_records(output_path.parent, scope_key)
            if cached is not None:
                reusable_records.extend(cached)
                continue
        changed_rows.extend(scope_rows)
    records = reusable_records + [_summarize_record(row, abs_tolerance, rel_tolerance) for row in changed_rows if isinstance(row, Mapping)]
    preferred_state_views = build_preferred_state_views(rows, lambda row: _summarize_record(row, abs_tolerance, rel_tolerance))
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "record_schema": "roi_observation_summary_v2",
        "views_schema": "roi_day_mode_v2",
        "pipeline": pipeline,
        "records": records,
        "preferred_state_views": preferred_state_views,
        "processed_signature": dependency_signature(processed),
        "config_signature": config_signature,
    }
    save_npz_cache(output_path, payload)
    indexed = save_indexed_records(
        output_path.parent,
        pipeline=pipeline,
        stage="initial_analysis",
        records_by_scope=_group_records(payload.get("roi_records", payload.get("records", []))),
        config_signature=payload["config_signature"],
        upstream_signatures={key: dependency_signature(rows) for key, rows in grouped.items()},
        rebuild_expids=config.get("rebuild_expids", []),
    )
    if bool(initial_settings.get("generate_general_figures", True)):
        figure_root = Path(str(initial_settings["figure_output_dir"])) if initial_settings.get("figure_output_dir") else Path(str(config.get("result_root") or output_path.parent.parent)) / "figures" / "general" / "initial_analysis"
        payload["figure_paths"] = write_initial_analysis_figures(records, figure_root, preferred_state_views=preferred_state_views, pipeline=pipeline)
        save_npz_cache(output_path, payload)
    manifest = stage_manifest(
        pipeline=pipeline,
        stage="initial_analysis",
        config_signature=dependency_signature(payload["config_signature"]),
        upstream_signature=payload["processed_signature"],
        active_expids=sorted({str(row.get("expid")) for row in records if row.get("expid")}),
        records={"initial_analysis_cache_path": str(output_path), "roi_count": len(records), "preferred_state_view_counts": {key: len(value) for key, value in preferred_state_views.items()}, "indexed_artifacts": indexed.get("records", {})},
    )
    save_json_manifest(cache_manifest_path({**config, "cache_paths": {"initial_analysis": output_path.parent}}, "initial_analysis"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_initial_analysis(args.config, args.pipeline, args.processed_cache, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
