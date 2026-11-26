"""Versioned schemas shared by modular analysis cache stages."""
from __future__ import annotations
from typing import Any
CACHE_SCHEMA_VERSION = 2
STAGE_SCHEMA_VERSIONS = {"source": 2, "processed": 2, "initial_analysis": 2, "classification": 2, "for_analysis": 2, "shuffle": 2, "results": 2}
def schema_metadata(stage: str, *, pipeline: str, config_signature: str | None = None, upstream_signature: str | None = None) -> dict[str, Any]:
    return {"schema_version": CACHE_SCHEMA_VERSION, "stage_schema_version": int(STAGE_SCHEMA_VERSIONS.get(stage, CACHE_SCHEMA_VERSION)), "pipeline": str(pipeline), "stage": str(stage), "config_signature": config_signature, "upstream_signature": upstream_signature}
def stage_schema_version(stage: str) -> int:
    return int(STAGE_SCHEMA_VERSIONS.get(stage, CACHE_SCHEMA_VERSION))
