"""Build the normalized processed experiment cache used by initial analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy
from typing import Any, Mapping

from analysis.cache_builders.common import cache_manifest_path, dependency_signature, load_index_manifest, load_indexed_records, record_scope_key, save_indexed_records, save_json_manifest, stage_is_reusable, stage_manifest
from analysis.shared.cache_config import normalize_staged_config
from analysis.shared.cache_utils import load_npz_cache, save_npz_cache
from analysis.shared.plots.dendrite_plot_support import classify_visual_responsive_dendrites, classify_visual_responsive_spines


def _dendrite_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    visual_rows = []
    for classifier in (classify_visual_responsive_dendrites, classify_visual_responsive_spines):
        try:
            visual_rows.extend(classifier(source, source_cache=source).get("rows", []))
        except Exception:
            continue
    visual_by_id = {}
    for row in visual_rows:
        for key in ("global_dendrite_id", "global_spine_id", "roi_key"):
            if row.get(key):
                visual_by_id[str(row[key])] = row
    for animal_id, animal in source.get("animals", {}).items():
        for dendrite_id, dendrite in animal.get("dendrites", {}).items():
            for expid, observation in dendrite.get("observations", {}).items():
                records.append({
                    "roi_key": str(dendrite_id), "expid": str(expid), "animal_id": animal_id,
                    "day_id": source.get("experiments", {}).get(expid, {}).get("day_id") or source.get("experiments", {}).get(expid, {}).get("date"),
                    "scope": "observation",
                    "compartment": dendrite.get("compartment"),
                    "state_activity": observation.get("cut_state_means", {}),
                    "state_frequency": observation.get("state_event_frequencies", observation.get("event_frequency", {})),
                    "source_signature": source.get("experiments", {}).get(expid, {}).get("source_signature"),
                    "visual_responsive": visual_by_id.get(str(dendrite_id), {}).get("responsive"),
                    "visual_class": ("visual" if visual_by_id.get(str(dendrite_id), {}).get("responsive") is True else "non_visual" if visual_by_id.get(str(dendrite_id), {}).get("responsive") is False else None),
                })
            for spine_id, spine in dendrite.get("spines", {}).items():
                for expid, observation in spine.get("observations", {}).items():
                    records.append({
                        "roi_key": str(spine_id), "expid": str(expid), "animal_id": animal_id,
                        "day_id": source.get("experiments", {}).get(expid, {}).get("day_id") or source.get("experiments", {}).get(expid, {}).get("date"),
                        "scope": "observation",
                        "compartment": dendrite.get("compartment"),
                        "state_activity": observation.get("cut_state_means", {}),
                        "state_frequency": observation.get("state_event_frequencies", observation.get("event_frequency", {})),
                        "source_signature": source.get("experiments", {}).get(expid, {}).get("source_signature"),
                        "visual_responsive": visual_by_id.get(str(spine_id), {}).get("responsive"),
                        "visual_class": ("visual" if visual_by_id.get(str(spine_id), {}).get("responsive") is True else "non_visual" if visual_by_id.get(str(spine_id), {}).get("responsive") is False else None),
                    })
    return records


def _soma_records(source: dict[str, Any], config: dict[str, Any], impl: Any, repo_root: Path) -> list[dict[str, Any]]:
    records = []
    movie_ids = set(str(value) for value in config.get("movie_expids", []))
    sleep_ids = set(str(value) for value in config.get("sleep_expids", []))
    for row in source.get("experiments", []):
        expid = str(row.get("exp_id") or row.get("expid") or "")
        mode = str(row.get("mode") or ("movie" if expid in movie_ids else "sleep"))
        if not expid:
            continue
        ctx = impl.build_experiment_context(expid=expid, mode=mode, soma_channel=int(config.get("soma_channel", 1)), bouton_channel=int(config.get("bouton_channel", 0)), repo_root=repo_root)
        state_rows = []
        try:
            state_rows = impl.activity_rows_for_context(ctx, [], state_masks=impl.state_masks_for_context(ctx, []))
        except Exception:
            state_rows = []
        state_by_roi = {}
        for state_row in state_rows:
            key = (str(state_row.get("compartment") or ""), int(state_row.get("roi_index", -1)))
            state_by_roi.setdefault(key, {})[str(state_row.get("state") or "all")] = state_row
        visual_by_index = {}
        if mode == "movie":
            for compartment, channel in (("soma", int(config.get("soma_channel", 1))), ("bouton", int(config.get("bouton_channel", 0)))):
                try:
                    visual_rows = impl._visual_response_entity_rows(ctx, compartment=compartment, channel=channel, response_metric=str(config.get("visual_response_metric", "mean")), event_detection_method=str(config.get("event_detection_method", "derivative")), locomotion_threshold=config.get("locomotion_threshold"))
                except Exception:
                    visual_rows = []
                for visual_row in visual_rows:
                    if visual_row.get("roi_index") is not None:
                        visual_by_index[(compartment, int(visual_row["roi_index"]))] = visual_row
        for compartment, entity in (("soma", ctx.soma), ("bouton", ctx.bouton)):
            matrix = entity.matrix()
            roi_ids = list(entity.roi_ids()) if hasattr(entity, "roi_ids") else list(range(len(matrix)))
            for index, trace in enumerate(matrix):
                values = [float(value) for value in trace if value == value]
                fallback_activity = sum(values) / len(values) if values else None
                fallback_frequency = sum(1 for value in values if value > 0) / len(values) if values else None
                roi_states = state_by_roi.get((compartment, index), {})
                state_activity = {state: row_state.get("mean") for state, row_state in roi_states.items()}
                state_frequency = {state: row_state.get("event_frequency_per_min") for state, row_state in roi_states.items()}
                state_activity.setdefault("all", fallback_activity)
                state_frequency.setdefault("all", fallback_frequency)
                records.append({
                    "roi_key": f"{expid}|{compartment}|{roi_ids[index] if index < len(roi_ids) else index}",
                    "expid": expid, "animal_id": getattr(ctx, "animal_id", None), "day_id": getattr(ctx, "day_id", None),
                    "scope": "observation",
                    "compartment": compartment, "state_activity": state_activity,
                    "state_frequency": state_frequency,
                    "source_signature": row.get("source_signature"),
                    "visual_responsive": visual_by_index.get((compartment, index), {}).get("responsive"),
                    "visual_class": ("visual" if visual_by_index.get((compartment, index), {}).get("responsive") is True else "non_visual" if visual_by_index.get((compartment, index), {}).get("responsive") is False else None),
                })
    return records


def _source_scope_signatures(source: Mapping[str, Any]) -> dict[str, str]:
    experiments = source.get("experiments", {})
    rows = experiments.items() if isinstance(experiments, Mapping) else ((str(row.get("exp_id") or row.get("expid")), row) for row in experiments if isinstance(row, Mapping))
    result = {}
    for expid, metadata in rows:
        if not expid or not isinstance(metadata, Mapping):
            continue
        day_id = str(metadata.get("day_id") or metadata.get("date") or expid)
        result[f"{expid}|{day_id}"] = str(metadata.get("source_signature") or dependency_signature(metadata))
    return result


def _filter_source(source: Mapping[str, Any], expids: set[str]) -> dict[str, Any]:
    filtered = deepcopy(dict(source))
    experiments = filtered.get("experiments")
    if isinstance(experiments, Mapping):
        filtered["experiments"] = {str(key): value for key, value in experiments.items() if str(key) in expids}
    elif isinstance(experiments, list):
        filtered["experiments"] = [row for row in experiments if str(row.get("exp_id") or row.get("expid")) in expids]
    animals = filtered.get("animals")
    if isinstance(animals, Mapping):
        for animal in animals.values():
            for dendrite in (animal.get("dendrites", {}) or {}).values():
                dendrite["observations"] = {str(key): value for key, value in (dendrite.get("observations", {}) or {}).items() if str(key) in expids}
                for spine in (dendrite.get("spines", {}) or {}).values():
                    spine["observations"] = {str(key): value for key, value in (spine.get("observations", {}) or {}).items() if str(key) in expids}
    return filtered



def _group_records(records: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records or []:
        grouped.setdefault(record_scope_key(record), []).append(dict(record))
    return grouped


def build_processed_cache(config_path: Path, pipeline: str, source_cache_path: Path, output_path: Path) -> dict[str, Any]:
    with config_path.open() as handle:
        config = normalize_staged_config(json.load(handle), pipeline=pipeline, base_dir=config_path.parent)
    source = load_npz_cache(source_cache_path)
    source_signature = dependency_signature(source)
    config_signature = dependency_signature(config.get("source", {}), config.get("processing", {}), config.get("visual_response_metric", "mean"))
    manifest_path = cache_manifest_path({**config, "cache_paths": {"processed": output_path.parent}}, "processed")
    if not bool(config.get("rebuild") or config.get("processed_cache_rebuild")):
        reusable = stage_is_reusable(manifest_path, output_path, pipeline=pipeline, stage="processed", config_signature=config_signature, upstream_signature=source_signature)
        if reusable is not None:
            return reusable
    scope_signatures = _source_scope_signatures(source)
    previous_index = load_index_manifest(output_path.parent, "processed")
    previous_artifacts = dict((previous_index.get("records") or {}).get("artifacts") or {})
    rebuild_ids = {str(value) for value in (config.get("rebuild_expids") or [])}
    reusable_records = {}
    changed_expids = set()
    for scope_key, upstream_signature in scope_signatures.items():
        artifact = previous_artifacts.get(scope_key) or {}
        if str(artifact.get("config_signature")) == str(config_signature) and str(artifact.get("upstream_signature")) == str(upstream_signature) and scope_key not in rebuild_ids and scope_key.split("|", 1)[0] not in rebuild_ids:
            loaded = load_indexed_records(output_path.parent, scope_key)
            if loaded is not None:
                reusable_records[scope_key] = loaded
                continue
        changed_expids.add(scope_key.split("|", 1)[0])
    if pipeline == "dendrites":
        records = _dendrite_records(_filter_source(source, changed_expids)) if changed_expids else []
    elif pipeline == "soma_bouton":
        from analysis.soma_bouton_pipeline import soma_bouton_pipeline as impl
        repo_root = impl.resolve_repo_path(config.get("repo_base") or Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[2])
        records = _soma_records(_filter_source(source, changed_expids), config, impl, repo_root) if changed_expids else []
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    records = [row for rows in reusable_records.values() for row in rows] + records
    payload = {"schema_version": 1, "pipeline": pipeline, "roi_records": records, "source_signature": dependency_signature(source), "config_signature": config_signature}
    save_npz_cache(output_path, payload)
    indexed = save_indexed_records(
        output_path.parent,
        pipeline=pipeline,
        stage="processed",
        records_by_scope=_group_records(payload.get("roi_records", payload.get("records", []))),
        config_signature=payload["config_signature"],
        upstream_signatures={key: scope_signatures.get(key, dependency_signature(rows)) for key, rows in _group_records(payload.get("roi_records", payload.get("records", []))).items()},
        rebuild_expids=config.get("rebuild_expids", []),
    )
    manifest = stage_manifest(pipeline=pipeline, stage="processed", config_signature=payload["config_signature"], upstream_signature=payload["source_signature"], active_expids=sorted({str(row.get("expid")) for row in records if row.get("expid")}), records={"processed_cache_path": str(output_path), "roi_count": len(records), "indexed_artifacts": indexed.get("records", {})})
    save_json_manifest(cache_manifest_path({**config, "cache_paths": {"processed": output_path.parent}}, "processed"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_processed_cache(args.config, args.pipeline, args.source_cache, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
