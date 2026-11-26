"""Incremental source-cache builder for the soma/bouton pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.shared.cache_utils import load_npz_cache, save_npz_cache, stable_hash


def _request_key(mode: str, exp_id: str) -> str:
    return f"{mode}:{exp_id}"


def load_or_build_incremental(
    impl: Any,
    config: Mapping[str, Any],
    repo_root: Path,
    source_cache_path: Path,
    expids_by_mode: Mapping[str, Sequence[str]],
    *,
    rebuild: bool = False,
    force_rebuild_expids: Sequence[str] = (),
) -> dict[str, Any]:
    requests = [
        (mode, str(exp_id))
        for mode in ("movie", "sleep")
        for exp_id in dict.fromkeys(str(value) for value in expids_by_mode.get(mode, []) if str(value))
    ]
    requested = {_request_key(mode, exp_id) for mode, exp_id in requests}
    metadata = {
        "schema_version": 1,
        "pipeline": "soma_bouton_pipeline",
        "soma_channel": int(config.get("soma_channel", 1)),
        "bouton_channel": int(config.get("bouton_channel", 0)),
        "expids_by_mode": {mode: sorted(set(str(value) for value in values)) for mode, values in expids_by_mode.items()},
        "complete": True,
    }
    old = None if rebuild or not source_cache_path.exists() else load_npz_cache(source_cache_path)
    old_metadata = old.get("metadata", {}) if isinstance(old, dict) else {}
    stable = isinstance(old, dict) and old_metadata.get("pipeline") == metadata["pipeline"] and all(
        old_metadata.get(key) == metadata.get(key) for key in ("soma_channel", "bouton_channel", "expids_by_mode")
    )
    old_records = old.get("experiments", []) if stable and isinstance(old.get("experiments", []), list) else []
    records_by_key = {
        _request_key(str(row.get("mode")), str(row.get("exp_id"))): dict(row)
        for row in old_records
        if isinstance(row, Mapping) and row.get("mode") and row.get("exp_id")
    }
    records_by_key = {key: row for key, row in records_by_key.items() if key in requested}

    for mode, exp_id in requests:
        key = _request_key(mode, exp_id)
        record = records_by_key.get(key)
        if exp_id in {str(value) for value in force_rebuild_expids}:
            record = None
        try:
            ctx = impl.build_experiment_context(
                expid=exp_id,
                mode=mode,
                soma_channel=metadata["soma_channel"],
                bouton_channel=metadata["bouton_channel"],
                repo_root=repo_root,
            )
            soma_path = Path(ctx.soma.path)
            bouton_path = Path(ctx.bouton.path)
            source_state = {
                "soma_source_path": str(soma_path),
                "bouton_source_path": str(bouton_path),
                "soma_source_mtime_ns": int(soma_path.stat().st_mtime_ns),
                "bouton_source_mtime_ns": int(bouton_path.stat().st_mtime_ns),
            }
            unchanged = record is not None and all(record.get(k) == value for k, value in source_state.items())
            if unchanged:
                continue
            row = impl.experiment_summary_row(ctx)
            row.update({"mode": mode, **source_state})
            records_by_key[key] = row
        except Exception:
            # Preserve the legacy behavior: missing source data is reported by
            # the caller's normal pipeline path rather than silently inventing a row.
            if record is None:
                raise

    records = [records_by_key[_request_key(mode, exp_id)] for mode, exp_id in requests if _request_key(mode, exp_id) in records_by_key]
    payload = {
        "metadata": metadata,
        "experiments": records,
        "source_signature": stable_hash({"metadata": metadata, "experiments": records}),
        "complete": True,
    }
    save_npz_cache(source_cache_path, payload)
    return payload
