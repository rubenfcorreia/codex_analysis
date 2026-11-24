"""Shared cache-stage mechanics.

This module deliberately contains no pipeline-specific scientific logic.  It
provides stable stage names, manifest handling, and dependency signatures for
the standalone builders and for future family plugins.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from analysis.shared.cache_utils import stable_hash

CACHE_STAGES = ("source", "processed", "initial_analysis", "classification", "for_analysis", "shuffle", "results")
CACHE_MANIFEST_SCHEMA_VERSION = 1


def cache_root(config: Mapping[str, Any], stage: str) -> Path:
    if stage not in CACHE_STAGES:
        raise ValueError(f"Unknown cache stage: {stage}")
    paths = config.get("cache_paths") or {}
    configured = paths.get(stage)
    if configured:
        return Path(str(configured))
    root = config.get("cache_root") or (config.get("paths") or {}).get("cache_root")
    if not root:
        root = Path(config.get("result_root", "results")) / "cache"
    return Path(root) / stage


def cache_manifest_path(config: Mapping[str, Any], stage: str) -> Path:
    return cache_root(config, stage) / "manifest.json"


def dependency_signature(*parts: Any) -> str:
    return stable_hash(parts)


def load_json_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Cache manifest is not an object: {path}")
    if int(payload.get("schema_version", -1)) != CACHE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache manifest schema in {path}")
    return payload


def save_json_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("schema_version", CACHE_MANIFEST_SCHEMA_VERSION)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def stage_is_reusable(manifest_path: Path, output_path: Path, *, pipeline: str, stage: str, config_signature: str, upstream_signature: str | None = None, active_expids: list[str] | None = None) -> dict[str, Any] | None:
    """Return a manifest when a cache artifact exactly matches its dependencies."""
    if not output_path.exists():
        return None
    manifest = load_json_manifest(manifest_path)
    if not manifest or manifest.get("pipeline") != pipeline or manifest.get("stage") != stage:
        return None
    if str(manifest.get("config_signature")) != str(config_signature):
        return None
    if upstream_signature is not None and str(manifest.get("upstream_signature")) != str(upstream_signature):
        return None
    if active_expids is not None and sorted(set(map(str, active_expids))) != list(manifest.get("active_expids", [])):
        return None
    result = dict(manifest)
    result.setdefault("records", {})["status"] = "reused"
    return result


def stage_manifest(
    *,
    pipeline: str,
    stage: str,
    config_signature: str,
    upstream_signature: str | None = None,
    active_expids: list[str] | None = None,
    records: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in CACHE_STAGES:
        raise ValueError(f"Unknown cache stage: {stage}")
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "pipeline": str(pipeline),
        "stage": stage,
        "config_signature": str(config_signature),
        "upstream_signature": upstream_signature,
        "active_expids": sorted(set(str(value) for value in (active_expids or []))),
        "records": dict(records or {}),
    }



def record_scope_key(record: Mapping[str, Any]) -> str:
    expid = str(record.get("expid") or record.get("exp_id") or "unknown")
    day_id = str(record.get("day_id") or expid)
    return f"{expid}|{day_id}"


def indexed_artifact_path(root: Path, scope_key: str) -> Path:
    return root / "records" / f"{dependency_signature(scope_key)}.npz"


def load_index_manifest(root: Path, stage: str) -> dict[str, Any]:
    path = root / "index.json"
    if not path.exists():
        return stage_manifest(pipeline="", stage=stage, config_signature="", records={"artifacts": {}})
    loaded = load_json_manifest(path)
    return loaded or stage_manifest(pipeline="", stage=stage, config_signature="", records={"artifacts": {}})



def load_indexed_records(root: Path, scope_key: str) -> list[dict[str, Any]] | None:
    manifest = load_index_manifest(root, "")
    artifact = ((manifest.get("records") or {}).get("artifacts") or {}).get(scope_key)
    if not isinstance(artifact, Mapping):
        return None
    path = Path(str(artifact.get("path", "")))
    if not path.exists():
        return None
    from analysis.shared.cache_utils import load_npz_cache
    try:
        payload = load_npz_cache(path)
    except Exception:
        return None
    records = payload.get("records") if isinstance(payload, Mapping) else None
    return [dict(row) for row in records] if isinstance(records, list) and all(isinstance(row, Mapping) for row in records) else None

def save_indexed_records(root: Path, *, pipeline: str, stage: str, records_by_scope: Mapping[str, list[Mapping[str, Any]]], config_signature: str, upstream_signatures: Mapping[str, str], rebuild_expids: list[str] | None = None) -> dict[str, Any]:
    from analysis.shared.cache_utils import save_npz_cache
    root.mkdir(parents=True, exist_ok=True)
    rebuild = {str(value) for value in (rebuild_expids or [])}
    previous = load_index_manifest(root, stage)
    previous_artifacts = dict((previous.get("records") or {}).get("artifacts") or {})
    artifacts = {}
    reused = 0
    built = 0
    for scope_key, rows in sorted(records_by_scope.items()):
        expid = scope_key.split("|", 1)[0]
        signature = dependency_signature(pipeline, stage, scope_key, config_signature, upstream_signatures.get(scope_key, ""), rows)
        old = previous_artifacts.get(scope_key) or {}
        path = indexed_artifact_path(root, scope_key)
        can_reuse = expid not in rebuild and path.exists() and old.get("signature") == signature
        if can_reuse:
            reused += 1
        else:
            save_npz_cache(path, {"schema_version": 1, "pipeline": pipeline, "stage": stage, "scope_key": scope_key, "records": [dict(row) for row in rows], "signature": signature})
            built += 1
        artifacts[scope_key] = {"path": str(path), "signature": signature, "config_signature": config_signature, "upstream_signature": upstream_signatures.get(scope_key, ""), "expid": expid, "day_id": scope_key.split("|", 1)[-1], "row_count": len(rows), "status": "reused" if can_reuse else "built"}
    manifest = stage_manifest(pipeline=pipeline, stage=stage, config_signature=config_signature, active_expids=sorted({key.split("|", 1)[0] for key in records_by_scope}), records={"artifacts": artifacts, "reused_count": reused, "built_count": built})
    save_json_manifest(root / "index.json", manifest)
    return manifest


__all__ = [
    "CACHE_MANIFEST_SCHEMA_VERSION",
    "CACHE_STAGES",
    "cache_manifest_path",
    "cache_root",
    "dependency_signature",
    "load_json_manifest",
    "save_json_manifest",
    "stage_manifest",
    "record_scope_key",
    "indexed_artifact_path",
    "load_index_manifest",
    "load_indexed_records",
    "save_indexed_records",
]
