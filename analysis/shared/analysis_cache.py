from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from analysis.shared.cache_utils import (
    ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
    ANALYSIS_TABLE_CACHE_SCHEMA_VERSION,
    analysis_cache_meta_hash,
    load_npz_cache,
    save_npz_cache,
)

def save_analysis_tables_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_npz_cache(path, payload)

def load_analysis_results_cache(path: Path, *, expected_meta: Optional[Dict[str, Any]] = None, ignore_meta_keys: Optional[Sequence[str]] = None, rebuild: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
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
        ignored = {str(key) for key in (ignore_meta_keys or ())}
        saved = cache.get("meta", {})
        saved = saved if isinstance(saved, dict) else {}
        saved = {key: value for key, value in saved.items() if key not in ignored}
        expected = {key: value for key, value in expected_meta.items() if key not in ignored}
        if analysis_cache_meta_hash(saved) != analysis_cache_meta_hash(expected):
            return None, "meta_mismatch"
    results = cache.get("analysis_results")
    if not isinstance(results, dict):
        return None, "invalid_results"
    return cache, "ok"

def save_analysis_results_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_npz_cache(path, payload)

def analysis_results_cache_payload(results: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(results)
    for key in ("analysis_cache_summary", "source_cache_summary", "cache_summary", "analysis_report_path", "output_artifacts", "figure_files", "review_figure_files", "checkpoint_gallery", "shared_shuffle_cache", "state_coverage", "stage_timings"):
        payload.pop(key, None)
    return payload

__all__ = ["ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION", "ANALYSIS_TABLE_CACHE_SCHEMA_VERSION", "analysis_cache_meta_hash", "analysis_results_cache_payload", "load_analysis_results_cache", "save_analysis_results_cache", "save_analysis_tables_cache"]
