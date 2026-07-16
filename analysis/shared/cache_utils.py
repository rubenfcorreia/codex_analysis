from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.shared.state_utils import ensure_dir

ANALYSIS_CACHE_SCHEMA_VERSION = 1
ANALYSIS_TABLE_CACHE_SCHEMA_VERSION = 1
ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION = 1
SHARED_SHUFFLE_CACHE_SCHEMA_VERSION = 1

FAMILY_RESULT_CACHE_STAGES = (
    "visual_response",
    "state",
    "direct_trial_type_comparison",
    "mixed_model",
    "spine_coactivity",
    "correlation",
    "pairwise_correlation",
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


def load_npz_cache(path: Path) -> Dict[str, Any]:
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


def save_npz_cache(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, cache=np.array([payload], dtype=object))


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
    return path


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
    return stable_hash(
        {
            str(exp_id): str(exp_meta.get("source_signature", ""))
            for exp_id, exp_meta in sorted(dict(source_cache.get("experiments", {})).items())
        }
    )


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
        "source_config_hash": str(source_cache.get("config_hash", "")),
        "source_signature": source_cache_signature(source_cache),
        "analysis_config_hash": stable_hash({**source_config, "analysis_unit": str(analysis_unit)}),
        "analysis_tables_signature": analysis_cache_meta_hash(analysis_tables or {}),
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
        }
    )


def build_shared_shuffle_entry(key: str, vector_length: int, shuffle_n: int) -> Dict[str, Any]:
    return {"key": key, "vector_length": int(vector_length), "shuffle_n": int(shuffle_n)}


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
    "build_shared_shuffle_entry",
    "load_shared_shuffle_cache",
]
