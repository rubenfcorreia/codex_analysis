from __future__ import annotations

import hashlib
import os
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.shared.state_utils import ensure_dir
from analysis.shared.serialization_diagnostics import record_serialization, rss_bytes

ANALYSIS_CACHE_SCHEMA_VERSION = 1
ANALYSIS_TABLE_CACHE_SCHEMA_VERSION = 1
ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION = 2
SHARED_SHUFFLE_CACHE_SCHEMA_VERSION = 1
METHODOLOGY_VERSION = "2026-10-06-statistics-v3"

FAMILY_RESULT_CACHE_STAGES = (
    "visual_response",
    "state",
    "state_transitions",
    "direct_trial_type_comparison",
    "mixed_model",
    "spine_coactivity",
    "correlation",
    "pairwise_correlation",
    "lag",
    "coincidence",
    "matrix_similarity",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def cacheable(*parts: Any) -> str:
    return stable_hash(parts)


def ensure_numpy_pickle_compatibility() -> None:
    try:
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for submodule in ("multiarray", "numeric", "umath", "_multiarray_umath", "overrides", "fromnumeric"):
        try:
            module = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)


def _load_npz_cache_uninstrumented(path: Path) -> Dict[str, Any]:
    """Load a trusted-input object cache.

    NPZ caches may contain pickled Python objects because the active pipeline
    stores nested dictionaries. Never load these files from untrusted sources.
    """
    ensure_numpy_pickle_compatibility()
    try:
        loaded = np.load(path, allow_pickle=True)
        if "cache" not in loaded:
            raise KeyError(f"{path} does not contain a 'cache' entry")
        cache_obj = loaded["cache"]
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        ensure_numpy_pickle_compatibility()
        loaded = np.load(path, allow_pickle=True)
        if "cache" not in loaded:
            raise KeyError(f"{path} does not contain a 'cache' entry")
        cache_obj = loaded["cache"]
    if isinstance(cache_obj, np.ndarray) and cache_obj.dtype == object:
        item = cache_obj.item()
        if isinstance(item, dict):
            return item
    if isinstance(cache_obj, np.ndarray) and cache_obj.size == 1:
        item = cache_obj.reshape(()).item()
        if isinstance(item, dict):
            return item
    raise TypeError(f"Could not recover cache dictionary from {path}")


def load_npz_cache(path: Path) -> Dict[str, Any]:
    import time
    started = time.perf_counter()
    rss_before = rss_bytes()
    try:
        payload = _load_npz_cache_uninstrumented(path)
        record_serialization(
            path=path,
            artifact=path.name,
            format_name="npz",
            elapsed_s=time.perf_counter() - started,
            cache_status="reused",
            schema_version=payload.get("schema_version"),
            status="completed",
            rss_before=rss_before,
            rss_after=rss_bytes(),
        )
        return payload
    except Exception as exc:
        record_serialization(
            path=path,
            artifact=path.name,
            format_name="npz",
            elapsed_s=time.perf_counter() - started,
            cache_status="unreadable",
            status="failed",
            error=str(exc),
            rss_before=rss_before,
            rss_after=rss_bytes(),
        )
        raise


def save_npz_cache(path: Path, payload: Dict[str, Any]) -> None:
    """Persist an object cache atomically so interrupted runs leave a valid file."""
    import time
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    started = time.perf_counter()
    rss_before = rss_bytes()
    try:
        np.savez_compressed(temporary, cache=np.array([payload], dtype=object))
        os.replace(temporary, path)
        record_serialization(
            path=path,
            artifact=path.name,
            format_name="npz",
            elapsed_s=time.perf_counter() - started,
            cache_status="built",
            schema_version=payload.get("schema_version"),
            status="completed",
            rss_before=rss_before,
            rss_after=rss_bytes(),
        )
    except Exception as exc:
        record_serialization(
            path=path,
            artifact=path.name,
            format_name="npz",
            elapsed_s=time.perf_counter() - started,
            cache_status="failed",
            schema_version=payload.get("schema_version"),
            status="failed",
            error=str(exc),
            rss_before=rss_before,
            rss_after=rss_bytes(),
        )
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def analysis_cache_meta_hash(meta: Any) -> str:
    return stable_hash(meta)


def analysis_table_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_analysis_tables_cache.npz")


def analysis_results_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_analysis_results_cache.npz")


def analysis_day_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_analysis_day_cache.npz")


def family_results_cache_dir(cache_path: Path) -> Path:
    return cache_path.parent / "results"


def family_results_cache_path(cache_path: Path, stage: str) -> Path:
    return family_results_cache_dir(cache_path) / f"{cache_path.stem}_{stage}_results_cache.npz"


def family_results_cache_index(cache_path: Path) -> Dict[str, str]:
    return {stage: str(family_results_cache_path(cache_path, stage)) for stage in FAMILY_RESULT_CACHE_STAGES}


def family_results_cache_stage_for_selection(selected_families: Optional[Sequence[str]]) -> str:
    if isinstance(selected_families, str):
        raw_values = [part.strip() for part in selected_families.split(",") if part.strip()]
    else:
        raw_values = [str(value).strip() for value in (selected_families or []) if str(value).strip()]
    selected = set(raw_values)
    if not selected:
        selected = set(FAMILY_RESULT_CACHE_STAGES[1:])
    if "basal_apical" in selected:
        selected.add("state")
    for stage in reversed(FAMILY_RESULT_CACHE_STAGES[1:]):
        if stage in selected:
            return stage
    return FAMILY_RESULT_CACHE_STAGES[-1]


def _family_config_signature(meta: Mapping[str, Any]) -> str:
    return analysis_cache_meta_hash({
        key: value
        for key, value in dict(meta).items()
        if key not in {"source_signature", "scope_signatures"}
    })


def save_family_results_cache(
    cache_path: Path,
    stage: str,
    results: Dict[str, Any],
    *,
    base_meta: Dict[str, Any],
) -> Path:
    family_meta = dict(base_meta)
    family_meta["family_result_stage"] = stage
    payload = {
        "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
        "meta": family_meta,
        "meta_hash": analysis_cache_meta_hash(family_meta),
        "analysis_results": dict(results),
    }
    path = family_results_cache_path(cache_path, stage)
    save_npz_cache(path, payload)
    index_path = path.parent / "family_index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
    except Exception:
        index = {}
    family_entry = {
        "path": str(path),
        "family": str(stage),
        "signature": analysis_cache_meta_hash(family_meta),
        "source_signature": str(family_meta.get("source_signature", "")),
        "analysis_families": list(family_meta.get("analysis_families") or [stage]),
        "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
        "config_signature": _family_config_signature(family_meta),
    }
    unit_root = path.parent / str(stage) / "records"
    unit_artifacts = {}
    scope_signatures = dict(family_meta.get("scope_signatures") or {})
    def iter_list_tables(value: Any, prefix: str = ""):
        if isinstance(value, list):
            yield prefix or "rows", value
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from iter_list_tables(child, child_prefix)

    for table_name, table_rows in iter_list_tables(results):
        grouped = {}
        for row in table_rows:
            if not isinstance(row, dict):
                continue
            expid = row.get("expid") or row.get("exp_id")
            day_id = row.get("day_id") or row.get("date") or expid
            if expid is None:
                continue
            scope_key = f"{expid}|{day_id}|{table_name}"
            grouped.setdefault(scope_key, []).append(dict(row))
        for scope_key, rows in grouped.items():
            base_scope_key = "|".join(scope_key.split("|")[:2])
            unit_signature = analysis_cache_meta_hash({"family": stage, "scope_key": scope_key, "scope_signature": scope_signatures.get(base_scope_key, ""), "family_meta": {key: value for key, value in family_meta.items() if key not in {"source_signature", "scope_signatures"}}, "rows": rows})
            unit_path = unit_root / f"{analysis_cache_meta_hash(scope_key)}.npz"
            save_npz_cache(unit_path, {"schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION, "family": stage, "scope_key": scope_key, "scope_signature": scope_signatures.get(base_scope_key, ""), "table_name": table_name, "rows": rows, "signature": unit_signature})
            unit_artifacts[scope_key] = {"path": str(unit_path), "signature": unit_signature, "scope_signature": scope_signatures.get(base_scope_key, ""), "config_signature": _family_config_signature(family_meta), "scope_key": scope_key, "table_name": table_name, "row_count": len(rows)}
    family_entry["unit_artifacts"] = unit_artifacts
    index[str(stage)] = family_entry
    ensure_dir(index_path.parent)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return path


def load_reusable_family_scope_rows(
    path: Path,
    stage: str,
    scope_signatures: Mapping[str, str],
    *,
    table_name: Optional[str] = None,
    rebuild: bool = False,
    expected_family_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, list[Dict[str, Any]]]:
    """Load list-valued family tables whose expID/day inputs are unchanged."""
    if rebuild:
        return {}
    index_path = family_results_cache_path(path, stage).parent / "family_index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
    except Exception:
        return {}
    entry = index.get(str(stage))
    if not isinstance(entry, dict):
        return {}
    expected_config_signature = _family_config_signature(expected_family_meta) if expected_family_meta is not None else None
    if expected_config_signature is not None and str(entry.get("config_signature", "")) != expected_config_signature:
        return {}
    reusable: Dict[str, list[Dict[str, Any]]] = {}
    for artifact in dict(entry.get("unit_artifacts") or {}).values():
        if not isinstance(artifact, dict):
            continue
        artifact_scope = str(artifact.get("scope_key") or "")
        if table_name is not None and str(artifact.get("table_name") or "") != str(table_name):
            continue
        if not artifact_scope:
            continue
        base_scope = "|".join(artifact_scope.split("|")[:2])
        if str(artifact.get("scope_signature", "")) != str(scope_signatures.get(base_scope, "")):
            continue
        if expected_config_signature is not None and str(artifact.get("config_signature", "")) != expected_config_signature:
            continue
        artifact_path = Path(str(artifact.get("path", "")))
        if not artifact_path.exists():
            continue
        try:
            payload = load_npz_cache(artifact_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION:
            continue
        if str(payload.get("family", "")) != str(stage):
            continue
        if str(payload.get("scope_key", "")) != artifact_scope:
            continue
        if str(payload.get("table_name", "")) != str(artifact.get("table_name", "")):
            continue
        if str(payload.get("signature", "")) != str(artifact.get("signature", "")):
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        reusable.setdefault(base_scope, []).extend(dict(row) for row in rows if isinstance(row, dict))
    return reusable


def family_scope_reuse_report(
    family: str,
    requested_scopes: Mapping[str, Any],
    reused_scopes: Any,
) -> Dict[str, Any]:
    requested = {str(scope) for scope in requested_scopes}
    if isinstance(reused_scopes, Mapping):
        reused = {str(scope) for scope in reused_scopes}
    else:
        reused = {str(scope) for scope in (reused_scopes or [])}
    recomputed = requested.difference(reused)
    return {
        "family": str(family),
        "requested_scope_count": len(requested),
        "reused_scope_count": len(requested.intersection(reused)),
        "recomputed_scope_count": len(recomputed),
        "reused_scopes": sorted(requested.intersection(reused)),
        "recomputed_scopes": sorted(recomputed),
    }


def load_family_results_cache(
    path: Path,
    *,
    expected_meta: Optional[Dict[str, Any]] = None,
    rebuild: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if rebuild:
        return None, "rebuild_requested"
    if not path.exists():
        return None, "missing"
    try:
        cache = load_npz_cache(path)
    except Exception:
        return None, "unreadable"
    if not isinstance(cache, dict):
        return None, "invalid_payload"
    if cache.get("schema_version") != ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION:
        return None, "schema_mismatch"
    if expected_meta is not None:
        expected_hash = analysis_cache_meta_hash(expected_meta)
        saved_meta = cache.get("meta", {})
        if not isinstance(saved_meta, dict):
            saved_meta = {}
        saved_hash = analysis_cache_meta_hash(saved_meta)
        if saved_hash != expected_hash:
            return None, "meta_mismatch"
    results = cache.get("analysis_results")
    if not isinstance(results, dict):
        return None, "invalid_results"
    return cache, "ok"


def source_cache_signature(source_cache: Dict[str, Any]) -> str:
    """Return a stable signature for mapping- or list-shaped source caches."""
    experiments = source_cache.get("experiments", {}) if isinstance(source_cache, dict) else {}
    if isinstance(experiments, Mapping):
        values = {
            str(exp_id): str(exp_meta.get("source_signature", ""))
            for exp_id, exp_meta in sorted(experiments.items())
            if isinstance(exp_meta, Mapping)
        }
    elif isinstance(experiments, list):
        values = {
            str(row.get("exp_id") or row.get("expid")): str(
                row.get("source_signature") or stable_hash(row)
            )
            for row in experiments
            if isinstance(row, Mapping) and (row.get("exp_id") or row.get("expid"))
        }
    else:
        values = {}
    return stable_hash(values)


def analysis_day_cache_meta(
    source_cache: Dict[str, Any],
    analysis_tables: Optional[Dict[str, Any]] = None,
    *,
    analysis_unit: str = "day",
) -> Dict[str, Any]:
    source_config = dict(source_cache.get("config", {}))
    return {
        "analysis_cache_schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "analysis_unit": str(analysis_unit),
        "methodology_version": METHODOLOGY_VERSION,
        "source_config_hash": str(source_cache.get("config_hash", "")),
        "source_signature": source_cache_signature(source_cache),
        "analysis_config_hash": stable_hash({**source_config, "analysis_unit": str(analysis_unit)}),
    }


def save_analysis_day_cache(path: Path, analysis_cache: Dict[str, Any], *, meta: Dict[str, Any]) -> Path:
    payload = {
        "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "meta": meta,
        "meta_hash": analysis_cache_meta_hash(meta),
        "analysis_cache": analysis_cache,
    }
    save_npz_cache(path, payload)
    return path


def shared_shuffle_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_shuffle_cache.npz")


def array_signature(array: Any) -> Optional[str]:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.size == 0:
        return stable_hash({"shape": list(arr.shape), "dtype": str(arr.dtype), "empty": True})
    try:
        contiguous = np.ascontiguousarray(arr)
        digest = hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()
    except Exception:
        digest = stable_hash({"shape": list(arr.shape), "dtype": str(arr.dtype), "values": arr.tolist()})
    return stable_hash({"shape": list(arr.shape), "dtype": str(arr.dtype), "digest": digest})


def shared_shuffle_key(metadata: Dict[str, Any]) -> str:
    return stable_hash(metadata)


def load_shared_shuffle_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        cache = load_npz_cache(path)
    except Exception:
        return None
    if not isinstance(cache, dict):
        return None
    if cache.get("schema_version") != SHARED_SHUFFLE_CACHE_SCHEMA_VERSION:
        return None
    return cache


def save_shared_shuffle_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_npz_cache(path, payload)


def build_shared_shuffle_cache_key(
    *,
    family: str,
    signal: str,
    analysis_unit: str,
    animal_id: str,
    day_id: str,
    source_id: str,
    vector_length: int,
    state_label: Optional[str] = None,
    mask_signature: Optional[str] = None,
    correlation_method: str = "pearson",
    correlation_inference: str = "circular_shift",
    shuffle_n: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    min_shift_frames: Optional[int] = None,
    null_model: Optional[str] = None,
) -> str:
    return shared_shuffle_key(
        {
            "family": family,
            "signal": signal,
            "analysis_unit": analysis_unit,
            "animal_id": animal_id,
            "day_id": day_id,
            "source_id": source_id,
            "vector_length": int(vector_length),
            "state_label": state_label,
            "mask_signature": mask_signature,
            "correlation_method": str(correlation_method),
            "correlation_inference": str(correlation_inference),
            "shuffle_n": None if shuffle_n is None else int(shuffle_n),
            "shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
            "min_shift_frames": None if min_shift_frames is None else int(min_shift_frames),
            "null_model": str(null_model or "mask_aware"),
        }
    )


def build_shared_shuffle_entry(key: str, vector_length: int, shuffle_n: int) -> Dict[str, Any]:
    return {"key": key, "vector_length": int(vector_length), "shuffle_n": int(shuffle_n)}


def build_shared_shift_schedule_key(
    *,
    family: str,
    signal: str,
    analysis_unit: str,
    animal_id: str,
    day_id: str,
    vector_length: int,
    correlation_method: str,
    correlation_inference: str,
    shuffle_n: int,
    shuffle_seed: int,
    min_shift_frames: int,
    null_model: str = "mask_aware",
) -> str:
    """Build a schedule-only key; exact null results remain pair-specific."""
    return shared_shuffle_key({
        "schema": 2,
        "methodology_version": METHODOLOGY_VERSION,
        # Family/signal are retained as API inputs for callers and
        # diagnostics, but are intentionally not part of the schedule key:
        # compatible ROI families in one experiment/day may reuse the same
        # seeded schedule. Pair identity belongs only to exact-result keys.
        "analysis_unit": str(analysis_unit),
        "animal_id": str(animal_id),
        "day_id": str(day_id),
        "vector_length": int(vector_length),
        "correlation_method": str(correlation_method),
        "correlation_inference": str(correlation_inference),
        "shuffle_n": int(shuffle_n),
        "shuffle_seed": int(shuffle_seed),
        "min_shift_frames": int(min_shift_frames),
        "null_model": str(null_model),
    })


def build_pairwise_correlation_cache_key(
    *,
    family: str,
    comparison_name: str,
    analysis_unit: str,
    animal_id: str,
    day_id: str,
    mode: str,
    left_compartment: str,
    left_channel: Optional[int],
    right_compartment: Optional[str] = None,
    right_channel: Optional[int] = None,
    pair_mode: str = "within_compartment",
    selected_states: Optional[Sequence[str]] = None,
    source_signature: Optional[Mapping[str, Any]] = None,
    scope_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    return stable_hash(
        {
            "family": family,
            "comparison_name": comparison_name,
            "analysis_unit": analysis_unit,
            "scope_id": scope_id or day_id,
            "animal_id": animal_id,
            "day_id": day_id,
            "mode": mode,
            "left_compartment": left_compartment,
            "left_channel": left_channel,
            "right_compartment": right_compartment or left_compartment,
            "right_channel": right_channel if right_channel is not None else left_channel,
            "pair_mode": pair_mode,
            "selected_states": list(selected_states or []),
            "source_signature": source_signature,
            "extra_metadata": dict(extra_metadata or {}),
        }
    )


__all__ = [
    "ANALYSIS_CACHE_SCHEMA_VERSION",
    "ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION",
    "ANALYSIS_TABLE_CACHE_SCHEMA_VERSION",
    "FAMILY_RESULT_CACHE_STAGES",
    "SHARED_SHUFFLE_CACHE_SCHEMA_VERSION",
    "METHODOLOGY_VERSION",
    "analysis_cache_meta_hash",
    "analysis_day_cache_meta",
    "analysis_day_cache_path",
    "analysis_results_cache_path",
    "analysis_table_cache_path",
    "array_signature",
    "build_pairwise_correlation_cache_key",
    "cacheable",
    "ensure_numpy_pickle_compatibility",
    "family_results_cache_dir",
    "family_results_cache_index",
    "family_results_cache_path",
    "family_results_cache_stage_for_selection",
    "load_family_results_cache",
    "load_npz_cache",
    "save_analysis_day_cache",
    "save_family_results_cache",
    "save_npz_cache",
    "save_shared_shuffle_cache",
    "shared_shuffle_cache_path",
    "shared_shuffle_key",
    "source_cache_signature",
    "stable_hash",
    "build_shared_shuffle_cache_key",
    "build_shared_shift_schedule_key",
    "build_shared_shuffle_entry",
    "load_shared_shuffle_cache",
]
